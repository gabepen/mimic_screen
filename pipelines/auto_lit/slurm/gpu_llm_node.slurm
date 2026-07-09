#!/bin/bash
#SBATCH --job-name=auto-lit-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=86:00:00
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
    bash -c "
set -euo pipefail

# Optional AUTO_LIT_PYTHON if probe_container_python.slurm shows a non-default binary.

export PYTHONPATH=\"${REPO_ROOT}/src:\${PYTHONPATH:-}\"
export VLLM_MODEL_NAME=\"\${VLLM_MODEL_NAME:-qwen3}\"

PY=\"\${AUTO_LIT_PYTHON:-python3.11}\"
\"\$PY\" -c \"import sys; print('PYTHON', sys.executable, sys.version)\"

vllm serve \"${MODEL_DIR}\" \
    --served-model-name qwen3 \
    --port \"${VLLM_PORT}\" \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --trust-remote-code &

VLLM_PID=\$!

echo \"Waiting for vLLM /v1/models on 127.0.0.1:${VLLM_PORT} ...\"
VLLM_READY=0
for i in \$(seq 1 120); do
    if \"\$PY\" -c \"import json,urllib.request; url='http://127.0.0.1:${VLLM_PORT}/v1/models'; data=json.loads(urllib.request.urlopen(url,timeout=10).read()); raise SystemExit(0 if isinstance(data,dict) and data.get('data') else 1)\" 2>/dev/null; then
        VLLM_READY=1
        echo \"vLLM ready after \${i} attempt(s)\"
        break
    fi
    sleep 5
done
if [[ \"\${VLLM_READY}\" -ne 1 ]]; then
    echo \"ERROR: vLLM did not become ready on port ${VLLM_PORT}\" >&2
    exit 1
fi

export VLLM_BASE_URL=\"http://127.0.0.1:${VLLM_PORT}\"
export GPU_API_PORT=\"${GPU_API_PORT}\"
export HOST_RUBRIC_PATH=\"${HOST_RUBRIC_PATH:-}\"
export MICROBE_RUBRIC_PATH=\"${MICROBE_RUBRIC_PATH:-}\"
export SYNTHESIS_EXCERPT_TOP_K_HOST=\"\${SYNTHESIS_EXCERPT_TOP_K_HOST:-20}\"
export SYNTHESIS_EXCERPT_TOP_K_QUERY=\"\${SYNTHESIS_EXCERPT_TOP_K_QUERY:-10}\"
export SYNTHESIS_MENTION_EXCERPT_MAX_CHARS=\"\${SYNTHESIS_MENTION_EXCERPT_MAX_CHARS:-3000}\"
export SYNTHESIS_MENTION_TOTAL_CHARS=\"\${SYNTHESIS_MENTION_TOTAL_CHARS:-70000}\"
export SYNTHESIS_MENTION_MAX_SITES=\"\${SYNTHESIS_MENTION_MAX_SITES:-4}\"
export SYNTHESIS_MENTION_NO_HIT_FALLBACK_CHARS=\"\${SYNTHESIS_MENTION_NO_HIT_FALLBACK_CHARS:-1500}\"
export SYNTHESIS_FINAL_MAX_TOKENS=\"\${SYNTHESIS_FINAL_MAX_TOKENS:-8192}\"
export SYNTHESIS_CONTEXT_MARGIN=\"\${SYNTHESIS_CONTEXT_MARGIN:-4096}\"
export SYNTHESIS_ENABLE_REPAIR_PASS=\"\${SYNTHESIS_ENABLE_REPAIR_PASS:-1}\"
export SYNTHESIS_MAX_ATTEMPTS=\"\${SYNTHESIS_MAX_ATTEMPTS:-3}\"

\"\$PY\" -m uvicorn auto_lit_search.gpu_server:app --host 0.0.0.0 --port \"${GPU_API_PORT}\"

kill \"\${VLLM_PID}\" || true
wait \"\${VLLM_PID}\" 2>/dev/null || true
"

