#!/bin/bash
#SBATCH --job-name=auto-lit-search
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=auto_lit_search_%j.log

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${INPUT_IDMAP_CSV:?INPUT_IDMAP_CSV must be set (mapping output CSV)}"
: "${OUTPUT_SEARCH_JSON:?OUTPUT_SEARCH_JSON must be set}"
: "${QUERY_TAXID:?QUERY_TAXID must be set (NCBI taxonomy ID for query organism)}"
: "${TARGET_TAXID:?TARGET_TAXID must be set (NCBI taxonomy ID for target organism)}"

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${OUTPUT_SEARCH_JSON}")}"
QUERY_COL="${QUERY_COL:-query}"
TARGET_COL="${TARGET_COL:-target}"
CONDA_ENV="${CONDA_ENV:-evorate}"
: "${MAMBA_BIN:?MAMBA_BIN must be set (export or set in cluster config)}"
NO_CACHE="${NO_CACHE:-}"
ACCESSION_TEXT_OVERLAP="${ACCESSION_TEXT_OVERLAP:-}"
QUERY_TAXIDS="${QUERY_TAXIDS:-${QUERY_TAXID}}"
TARGET_TAXIDS="${TARGET_TAXIDS:-${TARGET_TAXID}}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "search_node: REPO_ROOT=${REPO_ROOT}"
echo "search_node: INPUT_IDMAP_CSV=${INPUT_IDMAP_CSV}"
echo "search_node: OUTPUT_SEARCH_JSON=${OUTPUT_SEARCH_JSON}"
echo "search_node: OUTPUT_DIR=${OUTPUT_DIR}"
echo "search_node: QUERY_TAXIDS=${QUERY_TAXIDS} TARGET_TAXIDS=${TARGET_TAXIDS}"
echo "search_node: QUERY_COL=${QUERY_COL} TARGET_COL=${TARGET_COL}"
echo "search_node: CONDA_ENV=${CONDA_ENV}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    echo "mamba not found at ${MAMBA_BIN}; set MAMBA_BIN or fix miniforge path." >&2
    exit 1
fi

run_python() {
    "${MAMBA_BIN}" run -n "${CONDA_ENV}" python "$@"
}

run_python -u -c "import mygene, lxml; print('search_node: python deps ok')"

PY_ARGS=(
    -m auto_lit_search.search
    -i "${INPUT_IDMAP_CSV}"
    -o "${OUTPUT_SEARCH_JSON}"
    --output-format json
    --output-dir "${OUTPUT_DIR}"
    --query-id-col "${QUERY_COL}"
    --target-id-col "${TARGET_COL}"
    --query-taxid "${QUERY_TAXID}"
    --target-taxid "${TARGET_TAXID}"
    --query-taxids "${QUERY_TAXIDS}"
    --target-taxids "${TARGET_TAXIDS}"
)
if [[ "${NO_CACHE}" == "1" ]]; then
    PY_ARGS+=(--no-cache)
fi
if [[ -n "${ACCESSION_TEXT_OVERLAP}" ]]; then
    PY_ARGS+=(--accession-text-overlap "${ACCESSION_TEXT_OVERLAP}")
fi

run_python -u "${PY_ARGS[@]}"

echo "search_node: finished -> ${OUTPUT_SEARCH_JSON}"
