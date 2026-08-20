#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
INFINITRAIN_SOURCE_DIR="${INFINITRAIN_SOURCE_DIR:-${REPOSITORY_ROOT}/third_party/InfiniTrain}"
UPSTREAM_RUNNER="${INFINITRAIN_SOURCE_DIR}/scripts/run_models_and_profile.bash"

if [[ ! -x "${UPSTREAM_RUNNER}" ]]; then
    echo "Error: InfiniTrain test runner was not found at ${UPSTREAM_RUNNER}." >&2
    echo "Initialize the submodule or set INFINITRAIN_SOURCE_DIR." >&2
    exit 1
fi

# InfiniTrain's runner resolves build and log paths from the working directory.
# Anchor those paths at this repository and supply MACA's provider configuration.
cd "${REPOSITORY_ROOT}"
exec "${UPSTREAM_RUNNER}" \
    --test-config "${SCRIPT_DIR}/test_config_maca.json" \
    "$@"
