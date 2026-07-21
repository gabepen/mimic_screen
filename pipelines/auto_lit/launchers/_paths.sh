# shellcheck shell=bash
# Shared path resolution for auto-lit launchers.
LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${LAUNCHER_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PIPELINE_ROOT}/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f "${REPO_ROOT}/.env.publishers" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env.publishers"
  set +a
fi
