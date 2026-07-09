#!/usr/bin/env bash
# Resolve GRADER_JOB_IDS -> healthy http://host:port lines (one per line).
# Run on the Slurm compute node HOST (not inside Singularity); squeue works there.
set -euo pipefail

: "${GRADER_JOB_IDS:?GRADER_JOB_IDS required}"
: "${GRADER_ENDPOINTS_FILE:?GRADER_ENDPOINTS_FILE required}"
GRADER_API_PORT="${GRADER_API_PORT:-9200}"

_normalize_squeue_node() {
    local n="${1//[[:space:]]/}"
    [[ -n "$n" ]] || return 1
    if [[ "$n" =~ ^([^][]+)(-)?\[([0-9]+) ]]; then
        local prefix="${BASH_REMATCH[1]}"
        local dash="${BASH_REMATCH[2]}"
        local idx="${BASH_REMATCH[3]}"
        printf '%s%s%s\n' "$prefix" "$dash" "$idx"
        return 0
    fi
    printf '%s\n' "$n"
}

_tmp="$(mktemp)"
trap 'rm -f "$_tmp"' EXIT
_i=0
IFS=':' read -ra _jids <<< "${GRADER_JOB_IDS}"
for _jid in "${_jids[@]}"; do
    [[ -n "${_jid}" ]] || continue
    _port=$((GRADER_API_PORT + _i))
    _raw="$(squeue -j "${_jid}" -h -o "%N" 2>/dev/null | head -1 || true)"
    _node="$(_normalize_squeue_node "${_raw}" 2>/dev/null || true)"
    if [[ -n "${_node}" ]]; then
        _url="http://${_node}:${_port}"
        if curl -sf --max-time 5 "${_url}/healthz" >/dev/null 2>&1; then
            echo "${_url}" >> "${_tmp}"
        fi
    fi
    _i=$((_i + 1))
done

mkdir -p "$(dirname "${GRADER_ENDPOINTS_FILE}")"
if [[ -s "${_tmp}" ]]; then
    sort -u "${_tmp}" > "${GRADER_ENDPOINTS_FILE}.tmp"
else
    : > "${GRADER_ENDPOINTS_FILE}.tmp"
fi
mv -f "${GRADER_ENDPOINTS_FILE}.tmp" "${GRADER_ENDPOINTS_FILE}"
