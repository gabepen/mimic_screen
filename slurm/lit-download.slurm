#!/bin/bash
#SBATCH --job-name=lit-download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=lit_download_%j.log

set -euo pipefail

DATA_ROOT=/private/groups/corbettlab/gabe/auto_lit_eval_data
SEARCH_INPUT="$1"
OUTPUT_CSV="${2:-lit_download_summary.csv}"

singularity exec \
    -B "${DATA_ROOT}:${DATA_ROOT}" \
    lit-download.sif \
    python -m auto_lit_search.collect \
        --input "${SEARCH_INPUT}" \
        --output "${OUTPUT_CSV}" \
        --data-root "${DATA_ROOT}" \
        --batch-size 500

