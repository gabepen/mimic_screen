#!/usr/bin/env bash
# Submit stage1 (mapping + EPMC search) from a YAML config.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
CONFIG="${1:?usage: $0 <stage1-config.yaml> [--mapping-only|--search-only]}"
shift
exec python3 "${LAUNCHER_DIR}/run_lit_stage1.py" --config "$CONFIG" "$@"
