#!/bin/bash

set -euo pipefail

# Simple sequential pipeline:
#   1) Run lit-download (CPU) to fetch PDFs/text for all papers.
#   2) Run lit-llm (GPU) to serve Qwen3 on those texts.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <search_output_csv_or_json> <download_summary_csv>"
    exit 1
fi

SEARCH_OUTPUT="$1"
DOWNLOAD_SUMMARY="$2"

DOWNLOAD_JOB_ID=$(sbatch --parsable mimic_screen/slurm/lit-download.slurm "${SEARCH_OUTPUT}" "${DOWNLOAD_SUMMARY}")
echo "Submitted lit-download job: ${DOWNLOAD_JOB_ID}"

LLM_JOB_ID=$(sbatch --parsable --dependency=afterok:${DOWNLOAD_JOB_ID} /private/groups/corbettlab/gabe/qwen3-research/lit-llm.slurm)
echo "Submitted lit-llm job (afterok on download): ${LLM_JOB_ID}"

