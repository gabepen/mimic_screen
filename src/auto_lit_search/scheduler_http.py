"""HTTP helpers for CPU scheduler stage dispatch."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from auto_lit_search.env_config import env_positive_float


def synthesis_http_timeout_sec() -> float:
    return env_positive_float("SYNTHESIS_HTTP_TIMEOUT", 1800.0)


def post_run_alignment_graded(
    synthesis_url_base: str,
    payload: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    base = synthesis_url_base.rstrip("/")
    url = f"{base}/run_alignment_graded"
    t = timeout if timeout is not None else synthesis_http_timeout_sec()
    r = requests.post(url, json=payload, timeout=t)
    r.raise_for_status()
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected synthesis response from {url}")
    return out


def wait_health(service: str, base_url: str, timeout: int = 900) -> bool:
    """Minimal health poll (mirrors download_node behavior)."""
    import time

    url = f"{base_url.rstrip('/')}/healthz"
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False
