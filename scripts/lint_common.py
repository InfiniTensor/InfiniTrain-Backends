"""Repository scope and host compile-database helpers for the lint drivers."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOTS = {"backends", "tools"}
HOST_SOURCES = {".c", ".cc", ".cpp", ".cxx"}
DEVICE_SOURCES = {".cu", ".maca", ".mlu", ".cl"}
HEADERS = {".h", ".hh", ".hpp", ".hxx", ".cuh"}
EXCLUDED_DIRS = {".git", "third_party", "build", ".cache", "__pycache__", ".venv"}
BUILD_PREFIXES = ("build-", "build_", "cmake-build-")


def root_path(path: Path) -> Path:
    return (ROOT / path).resolve()


def is_owned_path(path: Path) -> bool:
    try:
        relative = root_path(path).relative_to(ROOT)
    except ValueError:
        return False
    return not any(
        part in EXCLUDED_DIRS or part.startswith(BUILD_PREFIXES)
        for part in relative.parts
    )


def is_project_path(path: Path) -> bool:
    if not is_owned_path(path):
        return False
    relative = root_path(path).relative_to(ROOT)
    return bool(relative.parts) and relative.parts[0] in PROJECT_ROOTS


def capture(command: list[str]) -> str:
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "command failed")
    return process.stdout


def diff_arguments(ref: str | None) -> list[str]:
    if ref:
        base = capture(["git", "merge-base", ref, "HEAD"]).strip()
        return [f"{base}..HEAD"]
    return ["--cached"]


def changed_paths(ref: str | None) -> list[Path]:
    output = capture(
        [
            "git",
            "diff",
            *diff_arguments(ref),
            "--no-renames",
            "--name-only",
            "--diff-filter=ACDM",
            "-z",
            "--",
        ]
    )
    return [Path(name) for name in output.split("\0") if name]


def needs_full_scan(changes: list[Path], relevant_files: set[str]) -> bool:
    for path in changes:
        normalized = path.as_posix()
        if normalized in relevant_files or any(
            normalized.startswith(prefix)
            for prefix in relevant_files
            if prefix.endswith("/")
        ):
            return True
        if path.parts and path.parts[0] == "third_party":
            return True
        if not is_owned_path(path):
            continue
        if path.name == "CMakeLists.txt" or path.suffix.lower() == ".cmake":
            return True
        if is_project_path(path) and path.suffix.lower() in HEADERS:
            return True
    return False


def relevant_change(path: Path, relevant_files: set[str]) -> bool:
    return needs_full_scan([path], relevant_files) or (
        is_project_path(path) and path.suffix.lower() in HOST_SOURCES | DEVICE_SOURCES
    )


def host_entries(build_dir: Path) -> list[dict[str, object]]:
    database = json.loads(
        (build_dir / "compile_commands.json").read_text(encoding="utf-8")
    )
    if not isinstance(database, list):
        raise ValueError("compile_commands.json must contain a list")
    filtered = []
    devices = set()
    for entry in database:
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError("invalid compile database entry: missing file")
        directory = (build_dir / str(entry.get("directory", build_dir))).resolve()
        source = (directory / str(entry["file"])).resolve()
        if not is_project_path(source):
            continue
        if source.suffix.lower() in DEVICE_SOURCES:
            devices.add(source)
        if source.suffix.lower() not in HOST_SOURCES or source.name.endswith(
            "_compile_fail.cc"
        ):
            continue
        filtered.append({**entry, "directory": str(directory), "file": str(source)})
    print(
        f"Host compile commands: {len(filtered)}; "
        f"device translation units excluded: {len(devices)} (.cu/.maca/.mlu/.cl).",
        flush=True,
    )
    if not filtered:
        raise ValueError(
            "No backends/tools host translation units in compile_commands.json. "
            "Configure the repository root with -DINFINITRAIN_BACKEND=<provider> "
            "and -DCMAKE_EXPORT_COMPILE_COMMANDS=ON (see scripts/README.md)."
        )
    return filtered


def write_host_database(
    build_dir: Path, tool: str, entries: list[dict[str, object]]
) -> Path:
    output_dir = build_dir / f"{tool}-host"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compile_commands.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def select_sources(
    entries: list[dict[str, object]], changes: list[Path], relevant_files: set[str]
) -> list[Path]:
    available = {Path(str(entry["file"])) for entry in entries}
    changed_sources = {
        root_path(path)
        for path in changes
        if is_project_path(path)
        and path.suffix.lower() in HOST_SOURCES
        and not path.name.endswith("_compile_fail.cc")
        and root_path(path).is_file()
    }
    missing = changed_sources - available
    if missing:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in sorted(missing))
        raise ValueError(
            f"Changed host sources missing from compile_commands.json: {paths}. "
            "Regenerate the database with the relevant backend/targets enabled."
        )
    if needs_full_scan(changes, relevant_files):
        return sorted(available)
    return sorted(available.intersection(root_path(path) for path in changes))


def project_file_pattern() -> str:
    roots = "|".join(re.escape(root) for root in sorted(PROJECT_ROOTS))
    excluded = "|".join(re.escape(name) for name in sorted(EXCLUDED_DIRS))
    excluded += "|" + "|".join(re.escape(prefix) + "[^/]*" for prefix in BUILD_PREFIXES)
    repository = re.escape(ROOT.as_posix())
    return rf"^(?:{repository}/|\./)?(?!(?:[^/]+/)*(?:{excluded})(?:/|$))(?:{roots})/"


def header_filter_pattern() -> str:
    # clang-tidy uses LLVM's POSIX regex engine, not Python's re engine.
    roots = "|".join(re.escape(root) for root in sorted(PROJECT_ROOTS))
    repository = re.escape(ROOT.as_posix())
    return rf"^({repository}/|[.]/)?({roots})/"
