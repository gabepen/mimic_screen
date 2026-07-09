#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
CONFIG="${1:?usage: $0 <stage2-config.yaml> [job ids or --gpu-urls ...]}"
shift
exec python3 "${LAUNCHER_DIR}/run_lit_cpu_manual.py" --config "$CONFIG" "$@"
