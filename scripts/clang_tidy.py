#!/usr/bin/env python3
"""Run the repository clang-tidy configuration on changed code or all TUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from clang_format_diff import changed_lines, diff_text
from lint_common import (
    ROOT,
    changed_paths,
    header_filter_pattern,
    host_entries,
    needs_full_scan,
    relevant_change,
    root_path,
    select_sources,
    write_host_database,
)

RELEVANT_FILES = {
    ".gitmodules",
    ".clang-tidy",
    "CMakePresets.json",
    ".github/workflows/clang-tidy.yaml",
    "scripts/clang_tidy.py",
    "scripts/clang_format_diff.py",
    "scripts/lint_common.py",
    "tools/clang_tidy/",
}
DIAGNOSTIC_RE = re.compile(
    r"^(.*):(\d+):(\d+):\s+(warning|error|fatal error):\s+"
    r"(.*?)(?:\s+\[([^\]]+)\])?\s*$"
)


def execute_captured(command: list[str]) -> tuple[int, str]:
    print("+", " ".join(str(part) for part in command), flush=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def find_tool(explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        return explicit
    return next((path for name in candidates if (path := shutil.which(name))), None)


def validate_version(binary: str) -> None:
    process = subprocess.run(
        [binary, "--version"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    version = process.stdout + process.stderr
    if (
        process.returncode != 0
        or re.search(r"\b(?:LLVM|clang) version 18\.", version, re.IGNORECASE) is None
    ):
        raise ValueError(
            "clang-tidy 18 is required by the InfiniTrainTidy plugin; "
            f"got: {version.strip() or 'unknown'}"
        )


def has_relevant_changes(ref: str | None) -> bool:
    relevant = [
        path for path in changed_paths(ref) if relevant_change(path, RELEVANT_FILES)
    ]
    if not relevant:
        print("No clang-tidy-relevant changes found.")
        return False
    print("clang-tidy-relevant changes:")
    for path in relevant:
        print(f"  {path.as_posix()}")
    return True


def plugin_arguments(plugin: Path) -> list[str]:
    return ["-load", str(plugin)]


def display_path(path_text: str) -> str:
    path = Path(path_text)
    try:
        if path.is_absolute():
            path = path.resolve().relative_to(ROOT)
    except (OSError, ValueError):
        pass
    return path.as_posix()


def unique_diagnostics(output: str) -> tuple[list[str], int]:
    unique: dict[tuple[str, str, str, str, str, str], int] = {}
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if match is None:
            continue
        path, line_number, column, severity, message, check_name = match.groups()
        key = (
            display_path(path),
            line_number,
            column,
            severity,
            message,
            check_name or "",
        )
        unique[key] = unique.get(key, 0) + 1

    lines = []
    ordered = sorted(
        unique.items(),
        key=lambda item: (
            item[0][0],
            int(item[0][1]),
            int(item[0][2]),
            item[0][3:],
        ),
    )
    for key, count in ordered:
        path, line_number, column, severity, message, check_name = key
        suffix = f" [{check_name}]" if check_name else ""
        repeated = f" (reported {count} times)" if count > 1 else ""
        lines.append(
            f"{path}:{line_number}:{column}: {severity}: {message}{suffix}{repeated}"
        )
    return lines, sum(unique.values())


def write_full_scan_reports(output: str, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_path = report_dir / "clang-tidy-raw.txt"
    unique_path = report_dir / "clang-tidy-unique.txt"
    raw_path.write_text(output, encoding="utf-8")

    diagnostics, total = unique_diagnostics(output)
    duplicate_count = total - len(diagnostics)
    summary = (
        f"Unique diagnostics: {len(diagnostics)}; raw diagnostics: {total}; "
        f"duplicates removed: {duplicate_count}."
    )
    unique_path.write_text("\n".join([summary, *diagnostics, ""]), encoding="utf-8")
    print(summary)
    for diagnostic in diagnostics:
        print(diagnostic)
    print(f"Raw log: {raw_path}")
    print(f"Deduplicated report: {unique_path}")
    return raw_path, unique_path


def run_changed(
    args: argparse.Namespace, clang_tidy: str, entries: list[dict[str, object]]
) -> int:
    changes = changed_paths(args.ref)
    if needs_full_scan(changes, RELEVANT_FILES):
        print("Header, build, dependency or rule changes: scanning all host TUs.")
        return run_all(args, clang_tidy, entries)
    sources = select_sources(entries, changes, RELEVANT_FILES)
    if not sources:
        print("No host translation units require clang-tidy.")
        return 0
    database = write_host_database(args.build_dir, "clang-tidy", entries)
    ranges = changed_lines(diff_text(args.ref))

    def check_source(source: Path) -> tuple[int, str]:
        command = [
            clang_tidy,
            "-p",
            str(database),
            "--config-file=" + str(ROOT / ".clang-tidy"),
            "--header-filter=" + header_filter_pattern(),
            "--quiet",
            *plugin_arguments(args.load_plugin),
        ]
        lines = ranges.get(source.relative_to(ROOT))
        if lines:
            command.append(
                "--line-filter=" + json.dumps([{"name": str(source), "lines": lines}])
            )
        # Pure deletions have no new line range; check their complete host TU.
        command.append(str(source))
        return execute_captured(command)

    failed = False
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(check_source, source) for source in sources]
        for future in as_completed(futures):
            return_code, output = future.result()
            print(output, end="" if output.endswith("\n") else "\n")
            failed = failed or return_code != 0
    return 1 if failed else 0


def run_all(
    args: argparse.Namespace, clang_tidy: str, entries: list[dict[str, object]]
) -> int:
    runner = find_tool(
        args.run_clang_tidy,
        ["run-clang-tidy-18", "run-clang-tidy.py", "run-clang-tidy"],
    )
    if runner is None:
        print("error: run-clang-tidy is required for full host scans", file=sys.stderr)
        return 2
    command = []
    if Path(runner).suffix.lower() == ".py":
        command = [sys.executable, "-W", "ignore::SyntaxWarning"]
    command += [
        runner,
        "-p",
        str(write_host_database(args.build_dir, "clang-tidy", entries)),
        "-clang-tidy-binary",
        clang_tidy,
        "-config-file",
        str(ROOT / ".clang-tidy"),
        "-header-filter",
        header_filter_pattern(),
        "-j",
        str(args.jobs),
        "-quiet",
    ]
    command += plugin_arguments(args.load_plugin)
    return_code, output = execute_captured(command)
    diagnostics, _ = unique_diagnostics(output)
    write_full_scan_reports(output, args.report_dir)
    if return_code != 0 and not diagnostics:
        print(output, file=sys.stderr, end="" if output.endswith("\n") else "\n")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan every project TU in the filtered compile database",
    )
    parser.add_argument(
        "--has-relevant-changes",
        action="store_true",
        help="report whether the diff contains changes that require clang-tidy",
    )
    parser.add_argument(
        "--ref", help="compare with the merge-base of REF (otherwise staged changes)"
    )
    parser.add_argument("--build-dir", type=Path, default=Path("build/lint"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="full-scan report directory (default: BUILD_DIR/clang-tidy-report)",
    )
    parser.add_argument("--jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--clang-tidy")
    parser.add_argument("--run-clang-tidy")
    parser.add_argument(
        "--load-plugin",
        type=Path,
        help="path to the InfiniTrain clang-tidy plugin",
    )
    args = parser.parse_args()

    if args.all and args.ref:
        parser.error("--ref cannot be used with --all; --all scans every project TU")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    try:
        if args.has_relevant_changes:
            return 0 if has_relevant_changes(args.ref) else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.load_plugin is None:
        print(
            "error: --load-plugin is required when running clang-tidy", file=sys.stderr
        )
        return 2

    args.build_dir = root_path(args.build_dir)
    args.load_plugin = root_path(args.load_plugin)
    args.report_dir = (
        root_path(args.report_dir)
        if args.report_dir is not None
        else args.build_dir / "clang-tidy-report"
    )

    if not (args.build_dir / "compile_commands.json").exists():
        print(
            f"error: {args.build_dir / 'compile_commands.json'} does not exist",
            file=sys.stderr,
        )
        return 2
    if not args.load_plugin.is_file():
        print(f"error: {args.load_plugin} does not exist", file=sys.stderr)
        return 2
    clang_tidy = find_tool(args.clang_tidy, ["clang-tidy-18", "clang-tidy"])
    if clang_tidy is None:
        print("error: clang-tidy is required", file=sys.stderr)
        return 2
    try:
        validate_version(clang_tidy)
        entries = host_entries(args.build_dir)
        return (
            run_all(args, clang_tidy, entries)
            if args.all
            else run_changed(args, clang_tidy, entries)
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
