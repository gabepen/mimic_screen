#!/usr/bin/env bash
# Re-run synthesis from existing *_graded.json using a stage2 YAML config.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
CONFIG="${1:?usage: $0 <stage2-config.yaml> [--failed-only]}"
shift
exec python3 "${LAUNCHER_DIR}/run_lit_resynth.py" --config "$CONFIG" "$@"
