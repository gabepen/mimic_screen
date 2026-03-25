#!/bin/bash
#SBATCH --job-name=probe-cntr-py
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=probe_container_python_%j.log

# Print which Python binaries exist in the image and which can import uvicorn.
# Usage (set image, then sbatch this file):
#   export CONTAINER_IMAGE=/path/to/image.sif
#   # or:  export CONTAINER_IMAGE=docker://user/repo:tag
#   sbatch slurm/probe_container_python.slurm
#
# If a path prints uvicorn but default `python3` does not, set before real jobs:
#   export AUTO_LIT_PYTHON=/that/abs/path

set -euo pipefail

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to .sif path or docker://...}"

singularity exec --nv "${CONTAINER_IMAGE}" bash -lc '
set -e
echo "PATH=${PATH}"
echo "== which -a python3 =="
which -a python3 2>/dev/null || true
echo "== try common paths + PATH python3 =="
for p in $(command -v python3 2>/dev/null) /usr/bin/python3 /usr/local/bin/python3; do
  [ -n "${p}" ] && [ -x "${p}" ] || continue
  echo "---- ${p}"
  "${p}" -c "import sys; print(\"  executable:\", sys.executable)" || echo "  (run failed)"
  "${p}" -c "import uvicorn; print(\"  uvicorn OK:\", uvicorn.__file__)" 2>&1 || echo "  (no uvicorn here)"
done
'
