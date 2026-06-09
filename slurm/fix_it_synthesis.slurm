#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gapenunu@ucsc.edu
#SBATCH --job-name=auto-lit-resynth
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=86:00:00
#SBATCH --output=auto_lit_resynth_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${GPU_HOST:?GPU_HOST must be set (synthesis GPU node hostname)}"
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"

GPU_API_PORT="${GPU_API_PORT:-9000}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
PAPERS_ROOT="${PAPERS_ROOT:-${DATA_ROOT}/papers}"
INSTRUCTIONS_FILE="${INSTRUCTIONS_FILE:-${REPO_ROOT}/prompts/default_instructions.txt}"
SYNTHESIS_URL="http://${GPU_HOST}:${GPU_API_PORT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "fix_it_synthesis: OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "fix_it_synthesis: SYNTHESIS_URL=${SYNTHESIS_URL}"
echo "fix_it_synthesis: PAPERS_ROOT=${PAPERS_ROOT}"

EXTRA_BINDS=()
for maybe_path in "${INSTRUCTIONS_FILE}"; do
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
    "${CPU_IMAGE}" \
    python -u "${REPO_ROOT}/scripts/fix_it_synthesis.py" \
        --output-root "${OUTPUT_ROOT}" \
        --synthesis-url "${SYNTHESIS_URL}" \
        --papers-root "${PAPERS_ROOT}" \
        --instructions-file "${INSTRUCTIONS_FILE}" \
        --only-needs-fix \
        --skip-missing-results \
        --backup
