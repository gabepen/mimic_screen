#!/bin/bash
#SBATCH --job-name=auto-lit-map
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=auto_lit_mapping_%j.log

set -euo pipefail

: "${REPO_ROOT:?REPO_ROOT must be set}"
: "${INPUT_CSV:?INPUT_CSV must be set (alignment table with query/target UniProt IDs)}"
: "${OUTPUT_IDMAP_CSV:?OUTPUT_IDMAP_CSV must be set}"
: "${QUERY_TAXID:?QUERY_TAXID must be set (NCBI taxonomy ID for query organism)}"
: "${TARGET_TAXID:?TARGET_TAXID must be set (NCBI taxonomy ID for target organism)}"

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${OUTPUT_IDMAP_CSV}")}"
QUERY_COL="${QUERY_COL:-query}"
TARGET_COL="${TARGET_COL:-target}"
CONDA_ENV="${CONDA_ENV:-evorate}"
: "${MAMBA_BIN:?MAMBA_BIN must be set (export or set in cluster config)}"
NO_CACHE="${NO_CACHE:-}"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "mapping_node: REPO_ROOT=${REPO_ROOT}"
echo "mapping_node: INPUT_CSV=${INPUT_CSV}"
echo "mapping_node: OUTPUT_IDMAP_CSV=${OUTPUT_IDMAP_CSV}"
echo "mapping_node: OUTPUT_DIR=${OUTPUT_DIR}"
echo "mapping_node: QUERY_TAXID=${QUERY_TAXID} TARGET_TAXID=${TARGET_TAXID}"
echo "mapping_node: QUERY_COL=${QUERY_COL} TARGET_COL=${TARGET_COL}"
echo "mapping_node: CONDA_ENV=${CONDA_ENV}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
    echo "mamba not found at ${MAMBA_BIN}; set MAMBA_BIN or fix miniforge path." >&2
    exit 1
fi

run_python() {
    "${MAMBA_BIN}" run -n "${CONDA_ENV}" python "$@"
}

run_python -u -c "import mygene, lxml; print('mapping_node: python deps ok')"

PY_ARGS=(
    -m auto_lit_search.mapping
    -i "${INPUT_CSV}"
    -o "${OUTPUT_IDMAP_CSV}"
    -d "${OUTPUT_DIR}"
    --query-col "${QUERY_COL}"
    --target-col "${TARGET_COL}"
    --query-taxid "${QUERY_TAXID}"
    --target-taxid "${TARGET_TAXID}"
)
if [[ "${NO_CACHE}" == "1" ]]; then
    PY_ARGS+=(--no-cache)
fi

run_python -u "${PY_ARGS[@]}"

echo "mapping_node: finished -> ${OUTPUT_IDMAP_CSV}"
