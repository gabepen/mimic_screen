#!/usr/bin/env bash
# Submit stage2 pipeline (GPU services + CPU download/analysis) from a YAML config.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
CONFIG="${1:?usage: $0 <stage2-config.yaml> [extra run_pipeline_slurm.py flags]}"
shift
exec python3 "${PIPELINE_ROOT}/run_pipeline_slurm.py" --config "$CONFIG" "$@"
