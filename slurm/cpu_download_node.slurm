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
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"

GPU_API_PORT="${GPU_API_PORT:-9000}"
DOCLING_API_PORT="${DOCLING_API_PORT:-9100}"
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
singularity exec \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    --env "PYTHONPATH=${REPO_ROOT}/src,COLLECTION_ORG=${COLLECTION_ORG},COLLECTION_AUTH_SCOPE=${COLLECTION_AUTH_SCOPE},COLLECTOR_EMAIL=${COLLECTOR_EMAIL},COLLECT_MAX_WORKERS=${COLLECT_MAX_WORKERS},COLLECT_DISABLE_SEMANTIC_SCHOLAR=${COLLECT_DISABLE_SEMANTIC_SCHOLAR}" \
    "${CPU_IMAGE}" \
    python -m auto_lit_search.download_node \
        --paper-ids "${PAPER_IDS_PATH}" \
        --data-root "${DATA_ROOT}" \
        --gpu-host "${GPU_HOST}" \
        --gpu-port "${GPU_API_PORT}" \
        --docling-host "${DOCLING_HOST}" \
        --docling-port "${DOCLING_API_PORT}" \
        --output-root "${OUTPUT_ROOT}" \
        $([ -n "${INSTRUCTIONS_FILE:-}" ] && echo --instructions-file "${INSTRUCTIONS_FILE}")
