#!/bin/bash
#SBATCH --job-name=auto-lit-cpu
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=auto_lit_cpu_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${PAPER_IDS_PATH:?PAPER_IDS_PATH must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${GPU_HOST:?GPU_HOST must be set (GPU node hostname)}"
: "${DOCLING_HOST:?DOCLING_HOST must be set (Docling node hostname)}"
: "${GRADER_HOST:?GRADER_HOST must be set (Grader node hostname)}"
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"
: "${HOST_RUBRIC_PATH:?HOST_RUBRIC_PATH must be set}"
: "${MICROBE_RUBRIC_PATH:?MICROBE_RUBRIC_PATH must be set}"

GPU_API_PORT="${GPU_API_PORT:-9000}"
DOCLING_API_PORT="${DOCLING_API_PORT:-9100}"
GRADER_API_PORT="${GRADER_API_PORT:-9200}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
COLLECTION_ORG="${COLLECTION_ORG:-ucsc}"
COLLECTION_AUTH_SCOPE="${COLLECTION_AUTH_SCOPE:-email_only}"
COLLECTOR_EMAIL="${COLLECTOR_EMAIL:-}"
COLLECT_MAX_WORKERS="${COLLECT_MAX_WORKERS:-2}"
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
echo "cpu_download_node: GPU_HOST=${GPU_HOST} GPU_API_PORT=${GPU_API_PORT}"
echo "cpu_download_node: DOCLING_HOST=${DOCLING_HOST} DOCLING_API_PORT=${DOCLING_API_PORT}"
echo "cpu_download_node: GRADER_HOST=${GRADER_HOST} GRADER_API_PORT=${GRADER_API_PORT}"
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
    --env "PYTHONPATH=${REPO_ROOT}/src,COLLECTION_ORG=${COLLECTION_ORG},COLLECTION_AUTH_SCOPE=${COLLECTION_AUTH_SCOPE},COLLECTOR_EMAIL=${COLLECTOR_EMAIL},COLLECT_MAX_WORKERS=${COLLECT_MAX_WORKERS},COLLECT_DISABLE_SEMANTIC_SCHOLAR=${COLLECT_DISABLE_SEMANTIC_SCHOLAR},GRADER_HOST=${GRADER_HOST},GRADER_API_PORT=${GRADER_API_PORT},HOST_RUBRIC_PATH=${HOST_RUBRIC_PATH},MICROBE_RUBRIC_PATH=${MICROBE_RUBRIC_PATH}" \
    "${CPU_IMAGE}" \
    python -u -m auto_lit_search.download_node \
        --paper-ids "${PAPER_IDS_PATH}" \
        --data-root "${DATA_ROOT}" \
        --gpu-host "${GPU_HOST}" \
        --gpu-port "${GPU_API_PORT}" \
        --docling-host "${DOCLING_HOST}" \
        --docling-port "${DOCLING_API_PORT}" \
        --grader-host "${GRADER_HOST}" \
        --grader-port "${GRADER_API_PORT}" \
        --host-rubric-path "${HOST_RUBRIC_PATH}" \
        --microbe-rubric-path "${MICROBE_RUBRIC_PATH}" \
        --output-root "${OUTPUT_ROOT}" \
        $([ -n "${INSTRUCTIONS_FILE:-}" ] && echo --instructions-file "${INSTRUCTIONS_FILE}")
