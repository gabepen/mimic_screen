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

# Python comes from the *container image*, not from Slurm "module load" on the host.
# Optional: after `singularity shell $DOCLING_IMAGE`, run `which python3` and set
#   export AUTO_LIT_PYTHON=/that/abs/path
# before sbatch (or --export) so this job uses one known interpreter.

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

PY_UVI=\"\"
if [ -n \"\${AUTO_LIT_PYTHON:-}\" ] && [ -x \"\${AUTO_LIT_PYTHON}\" ]; then
  PY_UVI=\"\${AUTO_LIT_PYTHON}\"
  \"\$PY_UVI\" -c \"import uvicorn\" 2>/dev/null || { echo \"AUTO_LIT_PYTHON cannot import uvicorn: \$PY_UVI\" >&2; exit 1; }
else
  for _c in \"\$(command -v python3 2>/dev/null)\" /usr/local/bin/python3 /usr/bin/python3; do
    [ -n \"\$_c\" ] && [ -x \"\$_c\" ] || continue
    if \"\$_c\" -c \"import uvicorn\" 2>/dev/null; then PY_UVI=\"\$_c\"; break; fi
  done
  [ -n \"\$PY_UVI\" ] || { echo \"No python with uvicorn (set AUTO_LIT_PYTHON or fix image)\" >&2; exit 1; }
fi
\"\$PY_UVI\" -m uvicorn auto_lit_search.docling_server:app --host 0.0.0.0 --port \"${DOCLING_API_PORT}\"
"

