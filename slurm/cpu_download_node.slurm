#!/bin/bash
#SBATCH --job-name=auto-lit-cpu
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=auto_lit_cpu_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${PAPER_IDS_PATH:?PAPER_IDS_PATH must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
GPU_URLS="${GPU_URLS:-}"
GPU_JOB_IDS="${GPU_JOB_IDS:-}"
if [[ -z "${GPU_URLS}" && -z "${GPU_JOB_IDS}" ]]; then
    : "${GPU_HOST:?GPU_HOST must be set when GPU_URLS and GPU_JOB_IDS are empty}"
fi
: "${DOCLING_HOST:?DOCLING_HOST must be set (Docling node hostname)}"
GRADER_URLS="${GRADER_URLS:-}"
GRADER_JOB_IDS="${GRADER_JOB_IDS:-}"
NUM_GRADER_NODES="${NUM_GRADER_NODES:-}"
if [[ -z "${GRADER_URLS}" && -z "${GRADER_JOB_IDS}" ]]; then
    : "${GRADER_HOST:?GRADER_HOST must be set when GRADER_URLS and GRADER_JOB_IDS are empty}"
fi
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"
: "${HOST_RUBRIC_PATH:?HOST_RUBRIC_PATH must be set}"
: "${MICROBE_RUBRIC_PATH:?MICROBE_RUBRIC_PATH must be set}"

GPU_API_PORT="${GPU_API_PORT:-9000}"
DOCLING_API_PORT="${DOCLING_API_PORT:-9100}"
GRADER_API_PORT="${GRADER_API_PORT:-9200}"
NUM_SYNTHESIS_NODES="${NUM_SYNTHESIS_NODES:-}"
GRADER_MAX_TOKENS="${GRADER_MAX_TOKENS:-4096}"
STAGE_WATCHDOG_SECONDS="${STAGE_WATCHDOG_SECONDS:-14400}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
COLLECTION_ORG="${COLLECTION_ORG:-ucsc}"
COLLECTION_AUTH_SCOPE="${COLLECTION_AUTH_SCOPE:-email_only}"
COLLECTOR_EMAIL="${COLLECTOR_EMAIL:-}"
COLLECT_MAX_WORKERS="${COLLECT_MAX_WORKERS:-4}"
COLLECT_DISABLE_SEMANTIC_SCHOLAR="${COLLECT_DISABLE_SEMANTIC_SCHOLAR:-}"

if [[ "${COLLECTION_ORG,,}" == "ucsc" && "${COLLECTION_AUTH_SCOPE,,}" == "email_only" && -z "${COLLECTOR_EMAIL}" ]]; then
    echo "COLLECTOR_EMAIL must be set for UCSC email_only collection mode." >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "cpu_download_node: DATA_ROOT=${DATA_ROOT}"
echo "cpu_download_node: PAPER_IDS_PATH=${PAPER_IDS_PATH}"
echo "cpu_download_node: OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "cpu_download_node: GPU_HOST=${GPU_HOST:-} GPU_API_PORT=${GPU_API_PORT}"
if [[ -n "${GPU_URLS}" ]]; then
    echo "cpu_download_node: GPU_URLS=${GPU_URLS}"
fi
if [[ -n "${GPU_JOB_IDS}" ]]; then
    echo "cpu_download_node: GPU_JOB_IDS=${GPU_JOB_IDS} NUM_SYNTHESIS_NODES=${NUM_SYNTHESIS_NODES:-}"
fi
echo "cpu_download_node: DOCLING_HOST=${DOCLING_HOST} DOCLING_API_PORT=${DOCLING_API_PORT}"
echo "cpu_download_node: STAGE_WATCHDOG_SECONDS=${STAGE_WATCHDOG_SECONDS}"
if [[ -n "${GRADER_URLS}" ]]; then
    echo "cpu_download_node: GRADER_URLS=${GRADER_URLS}"
else
    echo "cpu_download_node: GRADER_HOST=${GRADER_HOST:-} GRADER_API_PORT=${GRADER_API_PORT}"
fi
if [[ -n "${GRADER_JOB_IDS}" ]]; then
    echo "cpu_download_node: GRADER_JOB_IDS=${GRADER_JOB_IDS} NUM_GRADER_NODES=${NUM_GRADER_NODES:-}"
fi
GRADER_ENDPOINTS_FILE="${DATA_ROOT}/logs/grader_endpoints_discovered.txt"
GPU_ENDPOINTS_FILE="${DATA_ROOT}/logs/gpu_endpoints_discovered.txt"
GRADER_DISCOVER_INTERVAL="${GRADER_DISCOVER_INTERVAL:-15}"
GPU_DISCOVER_INTERVAL="${GPU_DISCOVER_INTERVAL:-15}"
_DISCOVER_PID=""
_GPU_DISCOVER_PID=""

if [[ -n "${GPU_JOB_IDS}" ]]; then
    export GPU_JOB_IDS GPU_API_PORT GPU_ENDPOINTS_FILE
    bash "${REPO_ROOT}/slurm/gpu_endpoints_discover.sh" || true
    echo "cpu_download_node: host synthesis GPU discovery -> ${GPU_ENDPOINTS_FILE} ($(wc -l < "${GPU_ENDPOINTS_FILE}" 2>/dev/null || echo 0) URL(s))"
    (
        while true; do
            bash "${REPO_ROOT}/slurm/gpu_endpoints_discover.sh" 2>/dev/null || true
            sleep "${GPU_DISCOVER_INTERVAL}"
        done
    ) &
    _GPU_DISCOVER_PID=$!
fi

if [[ -n "${GRADER_JOB_IDS}" ]]; then
    export GRADER_JOB_IDS GRADER_API_PORT GRADER_ENDPOINTS_FILE
    bash "${REPO_ROOT}/slurm/grader_endpoints_discover.sh" || true
    echo "cpu_download_node: host grader discovery -> ${GRADER_ENDPOINTS_FILE} ($(wc -l < "${GRADER_ENDPOINTS_FILE}" 2>/dev/null || echo 0) URL(s))"
    (
        while true; do
            bash "${REPO_ROOT}/slurm/grader_endpoints_discover.sh" 2>/dev/null || true
            sleep "${GRADER_DISCOVER_INTERVAL}"
        done
    ) &
    _DISCOVER_PID=$!
fi

trap '[[ -n "${_DISCOVER_PID}" ]] && kill "${_DISCOVER_PID}" 2>/dev/null || true; [[ -n "${_GPU_DISCOVER_PID}" ]] && kill "${_GPU_DISCOVER_PID}" 2>/dev/null || true' EXIT

echo "cpu_download_node: starting container..."
EXTRA_BINDS=()
for maybe_path in "${PAPER_IDS_PATH}" "${HOST_RUBRIC_PATH}" "${MICROBE_RUBRIC_PATH}" "${IDMAP_CSV:-}" "${INSTRUCTIONS_FILE:-}"; do
    if [[ -n "${maybe_path}" && -e "${maybe_path}" ]]; then
        bind_dir="$(dirname "${maybe_path}")"
        EXTRA_BINDS+=(-B "${bind_dir}:${bind_dir}")
    fi
done
singularity exec \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${EXTRA_BINDS[@]}" \
    --env "PYTHONPATH=${REPO_ROOT}/src,COLLECTION_ORG=${COLLECTION_ORG},COLLECTION_AUTH_SCOPE=${COLLECTION_AUTH_SCOPE},COLLECTOR_EMAIL=${COLLECTOR_EMAIL},COLLECT_MAX_WORKERS=${COLLECT_MAX_WORKERS},COLLECT_DISABLE_SEMANTIC_SCHOLAR=${COLLECT_DISABLE_SEMANTIC_SCHOLAR},GPU_URLS=${GPU_URLS},GPU_JOB_IDS=${GPU_JOB_IDS},GPU_HOST=${GPU_HOST:-},GPU_API_PORT=${GPU_API_PORT},NUM_SYNTHESIS_NODES=${NUM_SYNTHESIS_NODES},GPU_ENDPOINTS_FILE=${GPU_ENDPOINTS_FILE},GRADER_URLS=${GRADER_URLS},GRADER_JOB_IDS=${GRADER_JOB_IDS},GRADER_HOST=${GRADER_HOST:-},GRADER_API_PORT=${GRADER_API_PORT},NUM_GRADER_NODES=${NUM_GRADER_NODES},GRADER_MAX_TOKENS=${GRADER_MAX_TOKENS},STAGE_WATCHDOG_SECONDS=${STAGE_WATCHDOG_SECONDS},GRADER_ENDPOINTS_FILE=${GRADER_ENDPOINTS_FILE},HOST_RUBRIC_PATH=${HOST_RUBRIC_PATH},MICROBE_RUBRIC_PATH=${MICROBE_RUBRIC_PATH}" \
    "${CPU_IMAGE}" \
    python -u -m auto_lit_search.download_node \
        --paper-ids "${PAPER_IDS_PATH}" \
        --data-root "${DATA_ROOT}" \
        --gpu-host "${GPU_HOST:-}" \
        --gpu-port "${GPU_API_PORT}" \
        --docling-host "${DOCLING_HOST}" \
        --docling-port "${DOCLING_API_PORT}" \
        --grader-host "${GRADER_HOST:-}" \
        --grader-port "${GRADER_API_PORT}" \
        --host-rubric-path "${HOST_RUBRIC_PATH}" \
        --microbe-rubric-path "${MICROBE_RUBRIC_PATH}" \
        --output-root "${OUTPUT_ROOT}" \
        $([ -n "${INSTRUCTIONS_FILE:-}" ] && echo --instructions-file "${INSTRUCTIONS_FILE}")
