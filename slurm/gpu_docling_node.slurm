#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gapenunu@ucsc.edu
#SBATCH --job-name=auto-lit-docling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=auto_lit_docling_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${DOCLING_API_PORT:=9100}"
: "${DOCLING_IMAGE:?DOCLING_IMAGE must be set (e.g. docker://user/docling:latest or path to .sif)}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
singularity exec --nv \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -B "${REPO_ROOT}:${REPO_ROOT}" \
    "${DOCLING_IMAGE}" \
    bash -lc "
set -euo pipefail

export PYTHONPATH=\"${REPO_ROOT}/src:\${PYTHONPATH:-}\"
export DOCLING_API_PORT=\"${DOCLING_API_PORT}\"

/usr/bin/python3 -m uvicorn auto_lit_search.docling_server:app --host 0.0.0.0 --port \"${DOCLING_API_PORT}\"
"

