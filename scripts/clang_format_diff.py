#!/usr/bin/env python3
"""Check or format tracked C/C++ files or lines changed in a Git diff."""

from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
from pathlib import Path

from lint_common import ROOT, capture, diff_arguments, is_owned_path

CLANG_FORMAT_VERSION = "16.0.6"
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".maca",
    ".mlu",
    ".cl",
}
HUNK_RE = re.compile(r"^@@ .* \+(\d+)(?:,(\d+))? @@")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )


def diff_text(ref: str | None) -> str:
    return capture(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            *diff_arguments(ref),
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--unified=0",
            "--",
            ".",
            ":(exclude)**/third_party/**",
            ":(exclude)third_party/**",
            ":(exclude)build/**",
        ]
    )


def changed_lines(diff: str) -> dict[Path, list[tuple[int, int]]]:
    result: dict[Path, list[tuple[int, int]]] = {}
    current: Path | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            name = line[4:]
            if name.startswith('"'):
                name = codecs.escape_decode(name[1:-1].encode("utf-8"))[0].decode(
                    "utf-8", errors="surrogateescape"
                )
            current = None if name == "/dev/null" else Path(name.removeprefix("b/"))
            continue
        if current is None:
            continue
        match = HUNK_RE.match(line)
        if match is None:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        if count:
            result.setdefault(current, []).append((start, start + count - 1))
    return result


def validate_version(binary: str) -> bool:
    try:
        process = run([binary, "--version"])
    except OSError as error:
        print(f"error: cannot run formatter {binary}: {error}", file=sys.stderr)
        return False
    version = f"{process.stdout}\n{process.stderr}"
    match = re.search(r"\bversion\s+(\d+\.\d+\.\d+)\b", version)
    if (
        process.returncode != 0
        or match is None
        or match.group(1) != CLANG_FORMAT_VERSION
    ):
        print(
            f"error: clang-format {CLANG_FORMAT_VERSION} is required; "
            f"got: {version.strip() or 'unknown'}",
            file=sys.stderr,
        )
        return False
    return True


def validate_config(binary: str) -> bool:
    process = run(
        [
            binary,
            "-style=file",
            f"--assume-filename={ROOT / 'format_check.cc'}",
            "-dump-config",
        ]
    )
    if process.returncode == 0:
        return True
    print("error: failed to load .clang-format", file=sys.stderr)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    return False


def format_file(
    binary: str, path: Path, ranges: list[tuple[int, int]], check: bool
) -> bool:
    command = [binary, "-style=file"]
    command += ["--dry-run", "--Werror"] if check else ["-i"]
    command += [f"--lines={start}:{end}" for start, end in ranges]
    command.append(str(path))
    process = run(command)
    if process.returncode == 0:
        return True
    print(f"error: clang-format failed for {path}", file=sys.stderr)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="do not modify files")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--ref", help="compare with the merge-base of REF")
    scope.add_argument(
        "--all", action="store_true", help="format all tracked project files"
    )
    parser.add_argument("--clang-format", default="clang-format-16")
    args = parser.parse_args()

    try:
        if not validate_version(args.clang_format) or not validate_config(
            args.clang_format
        ):
            return 2
        if args.all:
            ranges = {
                Path(name): []
                for name in capture(["git", "ls-files", "-z"]).split("\0")
                if name and is_owned_path(Path(name))
            }
        else:
            ranges = changed_lines(diff_text(args.ref))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    files = sorted(
        path
        for path in ranges
        if is_owned_path(path)
        and path.suffix.lower() in SOURCE_SUFFIXES
        and (ROOT / path).is_file()
    )
    if not files:
        print("No supported C/C++ files require clang-format.")
        return 0
    try:
        results = [
            format_file(args.clang_format, path, ranges[path], args.check)
            for path in files
        ]
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
