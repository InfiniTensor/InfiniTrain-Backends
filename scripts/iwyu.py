#!/usr/bin/env python3
"""Run IWYU on changed translation units or the complete project database."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from lint_common import (
    ROOT,
    changed_paths,
    host_entries,
    project_file_pattern,
    relevant_change,
    root_path,
    select_sources,
    write_host_database,
)

RELEVANT_FILES = {
    ".github/workflows/iwyu.yaml",
    ".gitmodules",
    "CMakePresets.json",
    "scripts/iwyu.imp",
    "scripts/iwyu.py",
    "scripts/lint_common.py",
}
CORRECT_RESULT_RE = re.compile(r"^\(.+ has correct #includes/fwd-decls\)$")


def run_check(command: list[str]) -> int:
    """Run IWYU while suppressing non-actionable success messages."""
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    pending_blank = False
    emitted_output = False
    for line in process.stdout:
        if CORRECT_RESULT_RE.fullmatch(line.strip()):
            continue
        if not line.strip():
            pending_blank = True
            continue
        if pending_blank and emitted_output:
            print()
        print(line, end="", flush=True)
        pending_blank = False
        emitted_output = True
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="scan every project TU")
    mode.add_argument(
        "--has-relevant-changes",
        action="store_true",
        help="return 0 when the selected diff should trigger a full IWYU scan",
    )
    parser.add_argument("--ref", help="compare with the merge-base of REF")
    parser.add_argument("--build-dir", type=Path, default=Path("build/lint"))
    parser.add_argument("--jobs", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--iwyu-tool")
    fix_mode = parser.add_mutually_exclusive_group()
    fix_mode.add_argument(
        "--fix", action="store_true", help="apply IWYU output with fix_includes.py"
    )
    fix_mode.add_argument(
        "--fix-dry-run",
        action="store_true",
        help="print the fix_includes.py diff without modifying files",
    )
    parser.add_argument(
        "--fix-header-removals",
        action="store_true",
        help="allow fix_includes.py to remove includes from header files",
    )
    parser.add_argument("--fix-tool")
    args = parser.parse_args()
    args.build_dir = root_path(args.build_dir)

    if args.all and args.ref:
        parser.error("--ref cannot be used with --all; --all scans every project TU")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.has_relevant_changes and (args.fix or args.fix_dry_run):
        parser.error("--has-relevant-changes cannot be combined with a fix mode")
    if args.fix_header_removals and not (args.fix or args.fix_dry_run):
        parser.error("--fix-header-removals requires --fix or --fix-dry-run")
    if args.fix_tool and not (args.fix or args.fix_dry_run):
        parser.error("--fix-tool requires --fix or --fix-dry-run")

    if args.has_relevant_changes:
        try:
            relevant = [
                path
                for path in changed_paths(args.ref)
                if relevant_change(path, RELEVANT_FILES)
            ]
        except (OSError, RuntimeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if relevant:
            print("Relevant IWYU changes:")
            for path in relevant:
                print(f"  {path}")
            return 0
        print("No relevant changes require IWYU.")
        return 1

    tool = args.iwyu_tool or shutil.which("iwyu_tool.py") or shutil.which("iwyu_tool")
    if tool is None:
        print("error: iwyu_tool is required", file=sys.stderr)
        return 2
    fixer = None
    if args.fix or args.fix_dry_run:
        fixer = (
            args.fix_tool
            or shutil.which("fix_includes.py")
            or shutil.which("fix_include")
        )
        if fixer is None:
            print(
                "error: fix_includes.py or fix_include is required for fix mode",
                file=sys.stderr,
            )
            return 2
    try:
        entries = host_entries(args.build_dir)
        sources = (
            sorted({Path(str(entry["file"])) for entry in entries})
            if args.all
            else select_sources(entries, changed_paths(args.ref), RELEVANT_FILES)
        )
        database = (
            write_host_database(args.build_dir, "iwyu", entries) if sources else None
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not sources:
        print("No host translation units require IWYU.")
        return 0

    command = [
        tool,
        "-j",
        str(args.jobs),
        "-p",
        str(database),
        *[str(source) for source in sources],
        "--",
        "-Xiwyu",
        "--cxx17ns",
        "-Xiwyu",
        "--mapping_file=" + str(ROOT / "scripts/iwyu.imp"),
    ]
    if not (args.fix or args.fix_dry_run):
        command.extend(["-Xiwyu", "--error=1"])
    display_command = [
        tool,
        "-j",
        str(args.jobs),
        "-p",
        str(database),
        f"<{len(sources)} translation units>",
        "--",
        *command[command.index("--") + 1 :],
    ]
    print("+", " ".join(display_command), flush=True)
    if not (args.fix or args.fix_dry_run):
        return run_check(command)

    analysis = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(analysis.stdout, end="")
    if analysis.returncode != 0:
        print(
            "error: IWYU analysis failed; no automatic fixes were applied",
            file=sys.stderr,
        )
        return analysis.returncode

    fix_command = [
        fixer,
        "--basedir",
        str(ROOT),
        "--only_re",
        project_file_pattern(),
        "--nocomments",
        "--blank_lines",
        "--noreorder",
        "--nosafe_headers" if args.fix_header_removals else "--safe_headers",
    ]
    if args.fix_dry_run:
        fix_command.append("--dry_run")
    print("+", " ".join(fix_command), flush=True)
    fixed = subprocess.run(
        fix_command,
        cwd=ROOT,
        input=analysis.stdout,
        text=True,
        check=False,
    )
    if args.fix_dry_run:
        print("IWYU fix preview completed; no files were modified.")
    elif fixed.returncode == 0:
        print("IWYU fixes applied. Run formatting, build, and tests before committing.")
    return fixed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
