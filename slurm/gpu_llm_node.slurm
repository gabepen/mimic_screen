#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gapenunu@ucsc.edu
#SBATCH --job-name=auto-lit-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=auto_lit_gpu_%j.log

set -euo pipefail
set -x

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${GPU_API_PORT:=9000}"
: "${GPU_IMAGE:?GPU_IMAGE must be set (e.g. docker://user/auto-lit-gpu:latest or path to .sif)}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VLLM_PORT="${VLLM_PORT:-8000}"

singularity exec --nv \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${MODEL_DIR}:${MODEL_DIR}" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${GPU_IMAGE}" \
    bash -lc "
set -euo pipefail

export PYTHONPATH=\"${REPO_ROOT}/src:\${PYTHONPATH:-}\"
export VLLM_MODEL_NAME=\"\${VLLM_MODEL_NAME:-qwen3}\"

vllm serve \"${MODEL_DIR}\" \
    --port \"${VLLM_PORT}\" \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --trust-remote-code &

VLLM_PID=\$!

for i in \$(seq 1 60); do
    if /usr/bin/python3 - << 'PY' 2>/dev/null; then break; fi
import os
import socket
import sys

host = '127.0.0.1'
port = int(os.environ.get('VLLM_PORT', '8000'))
s = socket.socket()
s.settimeout(1)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    if [ \$? -eq 0 ]; then
        break
    fi
    sleep 2
done

export VLLM_BASE_URL=\"http://127.0.0.1:${VLLM_PORT}\"
export GPU_API_PORT=\"${GPU_API_PORT}\"

/usr/bin/python3 -m uvicorn auto_lit_search.gpu_server:app --host 0.0.0.0 --port \"${GPU_API_PORT}\"

kill \"\${VLLM_PID}\" || true
wait \"\${VLLM_PID}\" 2>/dev/null || true
"

