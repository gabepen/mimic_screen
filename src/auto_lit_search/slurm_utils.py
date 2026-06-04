"""Slurm job introspection for dynamic grader endpoint discovery."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Final

_TERMINAL_JOB_STATES: Final[frozenset[str]] = frozenset(
    {"CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}
)


def scontrol_bin() -> str:
    return os.environ.get("SCONTROL", "scontrol").strip() or "scontrol"


def squeue_bin() -> str:
    return os.environ.get("SQUEUE", "squeue").strip() or "squeue"


def node_from_scontrol_job(raw: str) -> str | None:
    """
    Extract the allocated hostname from `scontrol show job` output.

    Slurm often prints several space-separated Key=Value tokens on one line,
    e.g. ``NodeList=(null) SchedNodeList=phoenix-00``.
    """
    invalid = {"", "none", "(null)", "null", "n/a", "unknown"}

    def _ok(name: str) -> bool:
        return bool(name) and name.lower() not in invalid

    for m in re.finditer(r"\bNodeList=(\S+)", raw):
        node = m.group(1).strip()
        if _ok(node):
            return node
    for m in re.finditer(r"\bSchedNodeList=(\S+)", raw):
        node = m.group(1).strip()
        if _ok(node):
            return node
    return None


def _normalize_squeue_node(name: str) -> str | None:
    """Return a single hostname from squeue %N (may be a simple host or range)."""
    invalid = {"", "none", "(null)", "null", "n/a", "unknown", "(not set)"}
    node = (name or "").strip()
    if not node or node.lower() in invalid:
        return None
    # "phoenix-[00-03]" -> use first host in bracket expansion is cluster-specific;
    # prefer the literal prefix before '[' when present.
    if "[" in node:
        prefix = node.split("[", 1)[0].rstrip("-")
        suffix = node.split("[", 1)[1]
        m = re.match(r"(\d+)", suffix)
        if prefix and m:
            return f"{prefix}{m.group(1)}"
    return node


def get_job_node(job_id: str) -> str | None:
    job_id = str(job_id or "").strip()
    if not job_id:
        return None
    try:
        raw = subprocess.check_output(
            [scontrol_bin(), "show", "job", job_id],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        node = node_from_scontrol_job(raw)
        if node:
            return node
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    try:
        out = subprocess.check_output(
            [squeue_bin(), "-j", job_id, "-h", "-o", "%N"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        line = (out or "").strip().split("\n")[0].strip()
        return _normalize_squeue_node(line)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def get_job_state(job_id: str) -> str | None:
    """Return Slurm job state string (e.g. RUNNING, PENDING) or None if unknown."""
    job_id = str(job_id or "").strip()
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            [squeue_bin(), "-j", job_id, "-h", "-o", "%T"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        state = (out or "").strip().split("\n")[0].strip()
        return state or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def is_terminal_job_state(state: str | None) -> bool:
    if not state:
        return False
    return state.upper() in _TERMINAL_JOB_STATES
