"""Shared HTTP helpers for CPU pipeline stage dispatch."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import requests


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    client = session or requests
    r = client.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return data


def poll_until_status(
    fetch_status: Callable[[], Dict[str, Any]],
    *,
    terminal: frozenset[str] = frozenset({"succeeded", "failed"}),
    poll_interval_sec: float = 2.0,
    deadline_monotonic: Optional[float] = None,
) -> Dict[str, Any]:
    while True:
        status = fetch_status()
        s = str(status.get("status") or "").strip().lower()
        if s in terminal:
            return status
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("poll_until_status deadline exceeded")
        time.sleep(max(0.5, poll_interval_sec))
