#!/bin/bash
#SBATCH --job-name=lit-download
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=download_only_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${PAPER_IDS_PATH:?PAPER_IDS_PATH must be set}"
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

COLLECTION_ORG="${COLLECTION_ORG:-ucsc}"
COLLECTION_AUTH_SCOPE="${COLLECTION_AUTH_SCOPE:-email_only}"
COLLECTOR_EMAIL="${COLLECTOR_EMAIL:-}"
COLLECT_MAX_WORKERS="${COLLECT_MAX_WORKERS:-4}"
COLLECT_DISABLE_SEMANTIC_SCHOLAR="${COLLECT_DISABLE_SEMANTIC_SCHOLAR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/llm_results}"
IDMAP_CSV="${IDMAP_CSV:-}"
DOWNLOAD_ALIGNMENT_ID="${DOWNLOAD_ALIGNMENT_ID:-}"
DOWNLOAD_MAX_ALIGNMENTS="${DOWNLOAD_MAX_ALIGNMENTS:-}"
DOWNLOAD_NO_CACHE="${DOWNLOAD_NO_CACHE:-}"
DOWNLOAD_SKIP_IF_RESULTS="${DOWNLOAD_SKIP_IF_RESULTS:-}"
DOWNLOAD_RETRY_FAILED="${DOWNLOAD_RETRY_FAILED:-}"

if [[ "${COLLECTION_ORG,,}" == "ucsc" && "${COLLECTION_AUTH_SCOPE,,}" == "email_only" && -z "${COLLECTOR_EMAIL}" ]]; then
    echo "COLLECTOR_EMAIL must be set for UCSC email_only collection mode." >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "download_only: DATA_ROOT=${DATA_ROOT}"
echo "download_only: PAPER_IDS_PATH=${PAPER_IDS_PATH}"
echo "download_only: COLLECT_MAX_WORKERS=${COLLECT_MAX_WORKERS}"

EXTRA_BINDS=()
for maybe_path in "${PAPER_IDS_PATH}" "${IDMAP_CSV}"; do
    if [[ -n "${maybe_path}" && -e "${maybe_path}" ]]; then
        bind_dir="$(dirname "${maybe_path}")"
        EXTRA_BINDS+=(-B "${bind_dir}:${bind_dir}")
    fi
done

PY_ARGS=(
    --paper-ids "${PAPER_IDS_PATH}"
    --data-root "${DATA_ROOT}"
    --output-root "${OUTPUT_ROOT}"
)
if [[ -n "${IDMAP_CSV}" ]]; then
    PY_ARGS+=(--idmap-csv "${IDMAP_CSV}")
fi
if [[ -n "${DOWNLOAD_ALIGNMENT_ID}" ]]; then
    PY_ARGS+=(--alignment-id "${DOWNLOAD_ALIGNMENT_ID}")
fi
if [[ -n "${DOWNLOAD_MAX_ALIGNMENTS}" ]]; then
    PY_ARGS+=(--max-alignments "${DOWNLOAD_MAX_ALIGNMENTS}")
fi
if [[ "${DOWNLOAD_NO_CACHE}" == "1" ]]; then
    PY_ARGS+=(--no-cache)
fi
if [[ "${DOWNLOAD_SKIP_IF_RESULTS}" == "1" ]]; then
    PY_ARGS+=(--skip-if-results)
fi
if [[ "${DOWNLOAD_RETRY_FAILED}" == "1" ]]; then
    PY_ARGS+=(--retry-failed)
fi

singularity exec \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${EXTRA_BINDS[@]}" \
    --env "PYTHONPATH=${REPO_ROOT}/src,COLLECTION_ORG=${COLLECTION_ORG},COLLECTION_AUTH_SCOPE=${COLLECTION_AUTH_SCOPE},COLLECTOR_EMAIL=${COLLECTOR_EMAIL},COLLECT_MAX_WORKERS=${COLLECT_MAX_WORKERS},COLLECT_DISABLE_SEMANTIC_SCHOLAR=${COLLECT_DISABLE_SEMANTIC_SCHOLAR},DOWNLOAD_RETRY_FAILED=${DOWNLOAD_RETRY_FAILED}" \
    "${CPU_IMAGE}" \
    python -u -m auto_lit_search.download_packages "${PY_ARGS[@]}"
