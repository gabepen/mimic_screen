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


def scancel_bin() -> str:
    return os.environ.get("SCANCEL", "scancel").strip() or "scancel"


def scancel_jobs(job_ids: list[str]) -> list[str]:
    """Best-effort cancel Slurm jobs. Returns job ids for which scancel was invoked."""
    cancelled: list[str] = []
    for raw in job_ids:
        jid = str(raw or "").strip()
        if not jid:
            continue
        try:
            subprocess.check_call(
                [scancel_bin(), jid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            cancelled.append(jid)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue
    return cancelled


def grader_scale_down_should_trigger(
    *,
    remaining_packets: int,
    respect_threshold: int,
) -> bool:
    """True when remaining paper packets still needing grading is at/below respect_threshold."""
    if respect_threshold <= 0:
        return False
    return int(remaining_packets) <= int(respect_threshold)


def select_idle_grader_jobs_to_kill(
    *,
    job_specs: list[dict[str, object]],
    inflight_by_url: dict[str, int],
    url_by_port: dict[int, str],
    n_kill: int,
    min_keep: int = 1,
) -> list[str]:
    """
    Choose up to ``n_kill`` grader Slurm job ids that currently have zero inflight work.

    Prefer higher ports (later-started graders). Always leave at least ``min_keep``
    registered endpoints that are not selected for kill.
    """
    if n_kill <= 0:
        return []
    registered_ports = sorted(url_by_port.keys())
    if len(registered_ports) <= min_keep:
        return []

    idle: list[tuple[int, str]] = []
    for spec in job_specs:
        try:
            port = int(spec["port"])  # type: ignore[arg-type]
            jid = str(spec["job_id"] or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not jid:
            continue
        url = url_by_port.get(port)
        if not url:
            continue
        if int(inflight_by_url.get(url, 0) or 0) > 0:
            continue
        idle.append((port, jid))

    idle.sort(key=lambda item: item[0], reverse=True)
    max_kill = min(n_kill, max(0, len(registered_ports) - min_keep), len(idle))
    return [jid for _, jid in idle[:max_kill]]
