#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MACA_CONFIG="${SCRIPT_DIR}/test_config_maca.json"
BASE_CONFIG="${REPOSITORY_ROOT}/third_party/InfiniTrain/scripts/test_config.json"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required by the model test runner." >&2
    exit 1
fi

# Reuse the Python standard library already used by upstream's log tools.
# Keep configuration handling here so no helper file or Python package is needed.
json_config() {
    python3 - "$@" <<'PY'
import json
import re
import sys


def load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def merge(base, overrides):
    result = base.copy()
    for key, value in overrides.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def namespaced_path(path, model, mode):
    match = re.search(r"^(.*)/(checkpoint_step_[0-9]+(?:/.*)?)$", path)
    if match:
        return f"{match[1]}/{model}/{mode}/{match[2]}"
    return f"{path}/{model}/{mode}"


def query(arguments):
    # This adapter supports the pinned runner's queries, not arbitrary jq code.
    values = {}
    while arguments and arguments[0].startswith("-"):
        option = arguments.pop(0)
        if option == "-r":
            continue
        if option not in ("--arg", "--argjson"):
            raise ValueError(f"Unsupported upstream jq option: {option}")
        key, value = arguments[:2]
        del arguments[:2]
        values[key] = json.loads(value) if option == "--argjson" else value
    expression, path = arguments
    config = load(path)
    expression = " ".join(expression.split())

    if expression == '.variables[$k] // empty':
        value = config["variables"].get(values["k"])
        if value is not None and value is not False:
            print(text(value))
        return
    if expression == r'.variables | to_entries[] | "\(.key)=\(.value)"':
        for key, value in config["variables"].items():
            print(f"{key}={text(value)}")
        return
    if expression.startswith("def namespaced_path("):
        args = config["test_groups"][values["g"]]["tests"][values["t"]]["args"].copy()
        model = values["model"]
        mode = "resume" if "load" in args else "no_resume"
        source_mode = "no_resume" if "no_resume" in (args.get("load") or "") else "resume"
        if "save" in args:
            args["save"] = namespaced_path(args["save"], model, mode)
        if "load" in args:
            args["load"] = namespaced_path(args["load"], model, source_mode)
        for key, value in args.items():
            print(f"--{key}={text(value)}")
        return

    match = re.fullmatch(r"((?:\.[A-Za-z_]\w*|\[\d+\])+)( \| length| // empty)?", expression)
    if not match:
        raise ValueError(f"Unsupported upstream jq query: {expression}")
    value = config
    for key, index in re.findall(r"\.([A-Za-z_]\w*)|\[(\d+)\]", match[1]):
        if value is None:
            break
        value = value.get(key) if key else value[int(index)]
    if match[2] == " // empty" and (value is None or value is False):
        return
    print(text(len(value) if match[2] == " | length" else value))


def main():
    action, *arguments = sys.argv[1:]
    if action == "source-dir":
        for command in load(arguments[0])["basic_compile_commands"]:
            match = re.search(r"-DINFINITRAIN_SOURCE_DIR=([^ ]+)", command["cmd"])
            if match:
                print(match[1])
                break
    elif action == "merge":
        base, overrides = map(load, arguments)
        config = merge(base, {k: v for k, v in overrides.items() if k != "test_groups"})
        group_overrides = {group["tag"]: group for group in overrides.get("test_groups", [])}
        upstream_tags = {group["tag"] for group in base["test_groups"]}
        for tag in group_overrides:
            if tag not in upstream_tags:
                raise ValueError(f"Unknown upstream test group: {tag}")

        # Inherit every upstream group; MACA entries only override its settings.
        config["test_groups"] = []
        for group in base["test_groups"]:
            tag = group["tag"]
            override = group_overrides.get(tag, {})
            test_ids = {test["id"] for test in group["tests"]}
            unknown_ids = [test["id"] for test in override.get("tests", []) if test["id"] not in test_ids]
            if unknown_ids:
                raise ValueError(f"Unknown upstream tests in {tag}: {', '.join(unknown_ids)}")
            merged_group = merge(group, {k: v for k, v in override.items() if k != "tests"})
            merged_group["tests"] = [
                merge(test, next((item for item in override.get("tests", []) if item["id"] == test["id"]), {}))
                for test in group["tests"]
            ]
            config["test_groups"].append(merged_group)
        json.dump(config, sys.stdout, ensure_ascii=False, indent=2)
        print()
    elif action == "query":
        query(arguments)
    else:
        raise ValueError(f"Unknown configuration action: {action}")


try:
    main()
except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
    sys.exit(f"Error: {error}")
PY
}

CONFIGURED_INFINITRAIN_SOURCE_DIR="$(json_config source-dir "${MACA_CONFIG}")"
INFINITRAIN_SOURCE_DIR="${INFINITRAIN_SOURCE_DIR:-${CONFIGURED_INFINITRAIN_SOURCE_DIR:-${REPOSITORY_ROOT}/third_party/InfiniTrain}}"
UPSTREAM_RUNNER="${INFINITRAIN_SOURCE_DIR}/scripts/run_models_and_profile.bash"

if [[ ! -x "${UPSTREAM_RUNNER}" ]]; then
    echo "Error: InfiniTrain test runner was not found at ${UPSTREAM_RUNNER}." >&2
    echo "Initialize the submodule or set INFINITRAIN_SOURCE_DIR." >&2
    exit 1
fi

# InfiniTrain's runner resolves build and log paths from the working directory.
# Anchor those paths at this repository and merge MACA overrides into the
# pinned upstream configuration before passing it to the runner.
cd "${REPOSITORY_ROOT}"
MERGED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/infinitrain-maca-config.XXXXXX.json")"
trap 'rm -f -- "${MERGED_CONFIG}"' EXIT
json_config merge "${BASE_CONFIG}" "${MACA_CONFIG}" > "${MERGED_CONFIG}"

# Keep the adapter local to the runner; build commands inherit no jq replacement.
# Upstream uses set -e/pipefail but does not enable nounset.
(
    set +u
    jq() { json_config query "$@"; }
    source "${UPSTREAM_RUNNER}" --test-config "${MERGED_CONFIG}" "$@"
)
