#!/bin/bash
#SBATCH --job-name=auto-lit-grader-a5500
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=86:00:00
#SBATCH --output=auto_lit_grader_%j.log

set -euo pipefail
set -x

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${MODEL_DIR:?MODEL_DIR must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${GRADER_API_PORT:=9200}"
: "${GRADER_MAX_TOKENS:=2048}"
: "${GRADER_PAPER_WORKERS:=1}"
: "${VLLM_MAX_NUM_SEQS:=1}"
: "${VLLM_MAX_MODEL_LEN:=40960}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.92}"
: "${GRADER_IMAGE:?GRADER_IMAGE must be set (e.g. docker://user/auto-lit-gpu:latest or path to .sif)}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VLLM_PORT="${VLLM_PORT:-8000}"

singularity exec --nv \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${MODEL_DIR}:${MODEL_DIR}" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${GRADER_IMAGE}" \
    bash -c "
set -euo pipefail

export PYTHONPATH=\"${REPO_ROOT}/src:\${PYTHONPATH:-}\"
export VLLM_MODEL_NAME=\"\${VLLM_MODEL_NAME:-qwen3}\"

PY=\"\${AUTO_LIT_PYTHON:-python3.11}\"
\"\$PY\" -c \"import sys; print('PYTHON', sys.executable, sys.version)\"

vllm serve \"${MODEL_DIR}\" \
    --served-model-name qwen3 \
    --port \"${VLLM_PORT}\" \
    --max-model-len \"${VLLM_MAX_MODEL_LEN}\" \
    --max-num-seqs \"${VLLM_MAX_NUM_SEQS}\" \
    --gpu-memory-utilization \"${VLLM_GPU_MEMORY_UTILIZATION}\" \
    --enable-prefix-caching \
    --trust-remote-code &

VLLM_PID=\$!
export VLLM_PID

export VLLM_PORT=\"${VLLM_PORT}\"
export VLLM_BASE_URL=\"http://127.0.0.1:${VLLM_PORT}\"

\"\$PY\" - << 'PY'
import json
import os
import sys
import time
import urllib.request

base = os.environ.get('VLLM_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
vllm_pid = int(os.environ.get('VLLM_PID', '0') or '0')
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f'{base}/v1/models', timeout=5) as resp:
            payload = json.loads(resp.read().decode())
            if payload.get('data'):
                sys.exit(0)
    except Exception:
        pass
    if vllm_pid > 0:
        try:
            os.kill(vllm_pid, 0)
        except OSError:
            print('vLLM process exited before becoming ready', file=sys.stderr)
            sys.exit(1)
    time.sleep(2)
print('vLLM did not become ready within 300s', file=sys.stderr)
sys.exit(1)
PY

export VLLM_BASE_URL=\"http://127.0.0.1:${VLLM_PORT}\"
export GRADER_API_PORT=\"${GRADER_API_PORT}\"
export GRADER_MAX_TOKENS=\"${GRADER_MAX_TOKENS}\"
export GRADER_PAPER_WORKERS=\"${GRADER_PAPER_WORKERS}\"
export GRADER_SKIP_SYNTHESIS=\"${GRADER_SKIP_SYNTHESIS:-1}\"

\"\$PY\" -m uvicorn auto_lit_search.grader_server:app --host 0.0.0.0 --port \"${GRADER_API_PORT}\"

kill \"\${VLLM_PID}\" || true
wait \"\${VLLM_PID}\" 2>/dev/null || true
"
