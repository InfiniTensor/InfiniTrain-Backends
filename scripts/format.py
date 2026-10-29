#!/usr/bin/env python3
"""Format complete files selected by path, staged changes or a Git reference."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from lint_common import ROOT, changed_paths, is_owned_path, root_path

# Supported file types.
SUPPORTED_FILES = {
    ".h": "c",
    ".hh": "c",
    ".hpp": "c",
    ".hxx": "c",
    ".c": "c",
    ".cc": "c",
    ".cpp": "c",
    ".cxx": "c",
    ".cu": "c",
    ".cuh": "c",
    ".maca": "c",
    ".mlu": "c",
    ".cl": "c",
    ".py": "py",
}


def format_file(file: Path, check: bool, formatter) -> bool:
    kind = SUPPORTED_FILES.get(file.suffix.lower())
    if kind is None:
        return True
    binary = formatter[kind]
    command = [binary]
    if kind == "c":
        command += ["-style=file"]
        command += ["--dry-run", "--Werror"] if check else ["-i"]
    elif check:
        command += ["--check"]
    command.append(str(file))
    try:
        process = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
    except OSError as error:
        print(f"error: cannot run {binary}: {error}", file=sys.stderr)
        return False
    if process.returncode != 0:
        print(f"error: {binary} failed for {file}", file=sys.stderr)
        print(process.stdout + process.stderr, file=sys.stderr, end="")
        return False
    if not check:
        print(f"Formatted: {file.relative_to(ROOT)}")
    return True


def git_added_files():
    """Get files selected by staged changes, using repository-relative Git paths."""
    for file in changed_paths(None):
        yield root_path(file)


def git_modified_since_ref(ref):
    """Get files changed from the merge-base of ref to HEAD."""
    for file in changed_paths(ref):
        yield root_path(file)


def list_files(paths):
    """Recursively get repository files under the specified paths."""
    for path in paths:
        if not is_owned_path(path):
            continue
        if path.is_file():
            yield path
        elif path.is_dir():
            for dirpath, subdirectories, filenames in os.walk(path):
                subdirectories[:] = [
                    name
                    for name in subdirectories
                    if is_owned_path(Path(dirpath) / name)
                ]
                for name in filenames:
                    file = Path(dirpath) / name
                    if is_owned_path(file):
                        yield file
        else:
            raise ValueError(f"not a file or directory: {path}")


def filter_in_path(file: Path, paths) -> bool:
    """Determine whether the file is in one of the selected paths."""
    return any(file.is_relative_to(path) for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="compare with the merge-base of REF")
    parser.add_argument(
        "--path", nargs="*", type=Path, help="repository-relative paths"
    )
    parser.add_argument(
        "--check", action="store_true", help="check without modifying files"
    )
    parser.add_argument("--c", default="clang-format-16", help="C/C++ formatter binary")
    parser.add_argument("--py", default="black", help="Python formatter binary")
    args = parser.parse_args()

    try:
        if args.path is not None:
            args.path = [root_path(path) for path in args.path]
        if args.ref is None and args.path is None:
            files = git_added_files()
        elif args.ref is None:
            files = list_files(args.path)
        elif args.path is None:
            files = git_modified_since_ref(args.ref)
        else:
            files = (
                file
                for file in git_modified_since_ref(args.ref)
                if filter_in_path(file, args.path)
            )

        formatted = True
        for file in files:
            if not is_owned_path(file) or not file.is_file():
                continue
            if not format_file(file, args.check, {"c": args.c, "py": args.py}):
                formatted = False
        return 0 if formatted else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
