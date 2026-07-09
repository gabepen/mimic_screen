#!/bin/bash
#SBATCH --job-name=auto-lit-resynth
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=86:00:00
#SBATCH --output=auto_lit_resynth_driver_%j.log

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
: "${SYNTHESIS_URLS:?SYNTHESIS_URLS must be set (semicolon-separated GPU base URLs)}"
: "${CPU_IMAGE:?CPU_IMAGE must be set (path to .sif or docker://...)}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${REPO_ROOT}}"
PAPERS_ROOT="${PAPERS_ROOT:-${DATA_ROOT}/papers}"
INSTRUCTIONS_FILE="${INSTRUCTIONS_FILE:-${PIPELINE_ROOT}/prompts/default_instructions.txt}"
WORKERS="${WORKERS:-1}"
RESYNTH_ALL="${RESYNTH_ALL:-0}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "fix_it_synthesis driver job ${SLURM_JOB_ID} on $(hostname)"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SYNTHESIS_URLS=${SYNTHESIS_URLS}"
echo "WORKERS=${WORKERS}"
echo "INSTRUCTIONS_FILE=${INSTRUCTIONS_FILE}"
echo "RESYNTH_ALL=${RESYNTH_ALL}"

for url in $(echo "${SYNTHESIS_URLS}" | tr ';' ' '); do
  health_url="${url%/}/healthz"
  echo "Waiting for ${health_url} (gpu_server + vLLM) ..."
  ready=0
  for _ in $(seq 1 720); do
    if curl -sf "${health_url}" >/dev/null 2>&1; then
      echo "Synthesis ready: ${url}"
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "ERROR: synthesis GPU not healthy: ${health_url}" >&2
    exit 1
  fi
done

EXTRA_BINDS=()
for maybe_path in "${INSTRUCTIONS_FILE}" "${REPO_ROOT}" "${PIPELINE_ROOT}"; do
  if [[ -n "${maybe_path}" && -e "${maybe_path}" ]]; then
    bind_dir="$(dirname "${maybe_path}")"
    EXTRA_BINDS+=(-B "${bind_dir}:${bind_dir}")
  fi
done

FIX_ARGS=(
  --output-root "${OUTPUT_ROOT}"
  --synthesis-urls "${SYNTHESIS_URLS}"
  --workers "${WORKERS}"
  --papers-root "${PAPERS_ROOT}"
  --instructions-file "${INSTRUCTIONS_FILE}"
  --backup
)
if [[ "${RESYNTH_ALL}" == "1" ]]; then
  echo "Resynth mode: ALL alignments with graded.json"
else
  echo "Resynth mode: --only-needs-fix"
  FIX_ARGS+=(--only-needs-fix)
fi

singularity exec \
  -B "${DATA_ROOT}:${DATA_ROOT}" \
  -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -B "${REPO_ROOT}:${REPO_ROOT}" \
  -B "${PIPELINE_ROOT}:${PIPELINE_ROOT}" \
  "${EXTRA_BINDS[@]}" \
  "${CPU_IMAGE}" \
  python -u "${PIPELINE_ROOT}/scripts/fix_it_synthesis.py" \
    "${FIX_ARGS[@]}"

echo "fix_it_synthesis driver complete."
