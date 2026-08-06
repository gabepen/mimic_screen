"""
CPU node script: read search output, download papers per alignment, POST each batch to GPU.
Run inside the lit-download container with env: DATA_ROOT, PAPER_IDS_PATH, GPU_HOST, GPU_API_PORT, OUTPUT_ROOT.
"""

import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from auto_lit_search.collect import download_papers_to_dir
from auto_lit_search.download_manifest import (
    DOWNLOAD_MANIFEST_FILENAME,
    _alignment_paper_ids,
    _emit_download_progress_summary,
    _infer_docling_required_basenames,
    _load_download_manifest,
    _load_idmap,
    _load_search_json,
    _merge_recs_into_manifest,
    _paper_has_usable_text,
    _write_download_manifest_atomic,
    classify_alignment_papers,
    is_alignment_download_complete,
    load_global_outcome_cache,
    record_global_outcomes_from_rows,
    write_alignment_download_complete,
)
from auto_lit_search.graded_request_payload import build_run_alignment_graded_payload
from auto_lit_search.scheduler_http import post_run_alignment_graded
from auto_lit_search.slurm_utils import (
    get_job_node,
    get_job_state,
    grader_scale_down_should_trigger,
    is_terminal_job_state,
    scancel_jobs,
    select_idle_grader_jobs_to_kill,
)

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)

_DEFAULT_GRADER_MAX_TOKENS = 4096


def _grader_max_tokens_env() -> int:
    raw = os.environ.get("GRADER_MAX_TOKENS", str(_DEFAULT_GRADER_MAX_TOKENS))
    try:
        return max(1, int(str(raw).strip() or _DEFAULT_GRADER_MAX_TOKENS))
    except (TypeError, ValueError):
        return _DEFAULT_GRADER_MAX_TOKENS


def _refresh_grader_payload_constraints(payload: Dict[str, Any] | None) -> None:
    """Apply current GRADER_MAX_TOKENS on each grader submit/resume."""
    if not payload or not isinstance(payload, dict):
        return
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
        payload["constraints"] = constraints
    constraints["max_tokens"] = _grader_max_tokens_env()


def _normalize_grader_url(entry: str, default_port: int) -> str:
    """Return http://host:port base URL from a host, host:port, or full URL."""
    raw = (entry or "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if ":" in raw and not raw.startswith("["):
        host, _, port_s = raw.rpartition(":")
        if port_s.isdigit():
            return f"http://{host}:{port_s}"
    return f"http://{raw}:{default_port}"


def _registered_grader_ports(urls: List[str]) -> set[int]:
    """Ports present in normalized grader base URLs (e.g. 9200..9204)."""
    from urllib.parse import urlparse

    ports: set[int] = set()
    for u in urls:
        p = urlparse(u)
        if p.port is not None:
            ports.add(int(p.port))
    return ports


def _prune_grader_pending_specs(
    pending: List[Dict[str, Any]],
    registered_urls: List[str],
) -> int:
    """
    Drop Slurm discovery entries whose API port is already listed in GRADER_URLS.

    Returns the number of specs removed.
    """
    reg_ports = _registered_grader_ports(registered_urls)
    if not reg_ports:
        return 0
    before = len(pending)
    kept = [s for s in pending if int(s["port"]) not in reg_ports]
    pending[:] = kept
    return before - len(kept)


def _resolve_grader_url_bases(grader_host: str, grader_port: int) -> List[str]:
    """Grader endpoints in priority order: GRADER_URLS, GRADER_HOSTS, single --grader-host."""
    seen: set[str] = set()
    out: List[str] = []

    def _add(entry: str) -> None:
        url = _normalize_grader_url(entry, grader_port)
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    raw_urls = os.environ.get("GRADER_URLS", "").strip()
    if raw_urls:
        for part in re.split(r"[;,]", raw_urls):
            _add(part)
        if out:
            return out

    raw_hosts = os.environ.get("GRADER_HOSTS", "").strip()
    if raw_hosts:
        for part in raw_hosts.split(","):
            _add(part)
        if out:
            return out

    if grader_host and str(grader_host).strip():
        _add(str(grader_host).strip())
    return out


def _resolve_gpu_url_bases(gpu_host: str, gpu_port: int) -> List[str]:
    """Synthesis LLM endpoints: GPU_URLS, GPU_HOSTS, single --gpu-host."""
    seen: set[str] = set()
    out: List[str] = []

    def _add(entry: str) -> None:
        url = _normalize_grader_url(entry, gpu_port)
        if url and url not in seen:
            seen.add(url)
            out.append(url)

    raw_urls = os.environ.get("GPU_URLS", "").strip()
    if raw_urls:
        for part in re.split(r"[;,]", raw_urls):
            _add(part)
        if out:
            return out

    raw_hosts = os.environ.get("GPU_HOSTS", "").strip()
    if raw_hosts:
        for part in raw_hosts.split(","):
            _add(part)
        if out:
            return out

    if gpu_host and str(gpu_host).strip():
        _add(str(gpu_host).strip())
    return out


def _parse_gpu_job_specs(gpu_port: int) -> List[Tuple[str, int]]:
    """Return (slurm_job_id, api_port) pairs from GPU_JOB_IDS."""
    raw = os.environ.get("GPU_JOB_IDS", "").strip()
    if not raw:
        return []
    specs: List[Tuple[str, int]] = []
    for i, part in enumerate(re.split(r"[:,]", raw)):
        job_id = part.strip()
        if job_id:
            specs.append((job_id, gpu_port + i))
    return specs


def _registered_gpu_ports(urls: List[str]) -> set[int]:
    ports: set[int] = set()
    for u in urls:
        p = urlparse(u)
        if p.port is not None:
            ports.add(int(p.port))
    return ports


def _prune_gpu_pending_specs(
    pending: List[Dict[str, Any]],
    registered_urls: List[str],
) -> int:
    reg_ports = _registered_gpu_ports(registered_urls)
    if not reg_ports:
        return 0
    before = len(pending)
    kept = [s for s in pending if int(s["port"]) not in reg_ports]
    pending[:] = kept
    return before - len(kept)


def _only_download_progress_log(record: Dict[str, Any]) -> bool:
    return record["extra"].get("download_progress") is True


def _retry_failed_enabled() -> bool:
    return os.environ.get("DOWNLOAD_RETRY_FAILED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _parse_grader_job_specs(grader_port: int) -> List[Tuple[str, int]]:
    """Return (slurm_job_id, api_port) pairs from GRADER_JOB_IDS (colon- or comma-separated)."""
    raw = os.environ.get("GRADER_JOB_IDS", "").strip()
    if not raw:
        return []
    specs: List[Tuple[str, int]] = []
    # Colon is required for Slurm sbatch --export (commas in values are truncated).
    for i, part in enumerate(re.split(r"[:,]", raw)):
        job_id = part.strip()
        if job_id:
            specs.append((job_id, grader_port + i))
    return specs


def _grader_health_ok(grader_url_base: str, timeout: int = 5) -> bool:
    url = f"{grader_url_base.rstrip('/')}/healthz"
    try:
        r = requests.get(url, timeout=(3, timeout))
        return r.status_code == 200
    except Exception:
        return False


def _wait_health(
    service: str,
    gpu_url_base: str,
    timeout: int = 300,
    interval: int = 5,
    *,
    host_resolver=None,
) -> bool:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        base = gpu_url_base
        if host_resolver is not None:
            try:
                resolved = host_resolver()
                if resolved:
                    base = resolved
            except Exception:
                pass
        url = f"{base.rstrip('/')}/healthz"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info(
                    "{} health OK at {} (after {:.0f}s, {} tries)",
                    service,
                    url,
                    time.monotonic() - started,
                    attempt,
                )
                return True
        except Exception as e:
            if attempt == 1 or attempt % 6 == 0:
                logger.info(
                    "Waiting for {} at {} ({:.0f}s / {}s) last_error={!r}",
                    service,
                    url,
                    time.monotonic() - started,
                    timeout,
                    e,
                )
        time.sleep(interval)
    logger.error(
        "Timed out waiting for {} at {} after {}s",
        service,
        gpu_url_base,
        timeout,
    )
    return False


def _submit_grader_async(
    session: requests.Session,
    grader_url_base: str,
    payload: Dict[str, Any],
    submit_timeout: int,
) -> str:
    deadline = time.monotonic() + max(1, submit_timeout)
    while True:
        try:
            r = session.post(
                f"{grader_url_base}/grade_alignment_async",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            out = r.json()
            job_id = str(out.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("grader async submit returned empty job_id")
            return job_id
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is None or resp.status_code != 429:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


def _wait_service_capacity(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    service_name: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
    warn_on_timeout: bool = True,
) -> bool:
    """Wait until remote async queue reports can_accept=true.

    If the capacity endpoint is unavailable/malformed, fail open so older
    server versions still work.
    """
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        try:
            r = session.get(f"{base_url.rstrip('/')}/{endpoint}", timeout=10)
            r.raise_for_status()
            data = r.json()
            can_accept = bool(data.get("can_accept", False))
            if can_accept:
                return True
        except Exception:
            return True
        time.sleep(max(1, poll_interval_seconds))
    if warn_on_timeout:
        logger.warning(
            "{} capacity wait timed out after {}s; attempting submit anyway",
            service_name,
            timeout_seconds,
        )
    return True


def _grader_status_once(
    session: requests.Session,
    grader_url_base: str,
    job_id: str,
) -> Dict[str, Any]:
    r = session.get(
        f"{grader_url_base}/grade_alignment_status/{job_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _submit_docling_async(
    session: requests.Session,
    docling_url_base: str,
    payload: Dict[str, Any],
    submit_timeout: int,
) -> str:
    deadline = time.monotonic() + max(1, submit_timeout)
    sleep_seconds = 2
    while True:
        try:
            r = session.post(
                f"{docling_url_base}/convert_alignment_async",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            out = r.json()
            job_id = str(out.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("docling async submit returned empty job_id")
            return job_id
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
        ):
            if time.monotonic() >= deadline:
                raise
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 2, 15)
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is None or resp.status_code != 429:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(sleep_seconds)
            sleep_seconds = min(sleep_seconds * 2, 15)


def _docling_status_once(
    session: requests.Session,
    docling_url_base: str,
    job_id: str,
) -> Dict[str, Any]:
    r = session.get(
        f"{docling_url_base}/convert_alignment_status/{job_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _canonical_alignment_text_key(fname: str) -> str:
    """Map source-tagged names to one canonical key for dedupe/logging."""
    low = fname.lower()
    if not low.endswith(".txt"):
        return low
    m = re.match(r"^(.*__(?:query|target))(?:__[^.]*)?\.txt$", low)
    if m:
        return f"{m.group(1)}.txt"
    return low


def run(
    paper_ids_path: str,
    data_root: str,
    gpu_host: str,
    gpu_port: int,
    output_root: str,
    instructions: str = "",
    instructions_file: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    request_timeout: int = 600,
    no_cache: bool = False,
    docling_host: str = "",
    docling_port: int = 9100,
    grader_host: str = "",
    grader_port: int = 9200,
    host_rubric_path: str = "",
    microbe_rubric_path: str = "",
) -> None:
    STATE_DOWNLOADING = "DOWNLOADING"
    STATE_DOCLING_PENDING = "DOCLING_PENDING"
    STATE_DOCLING_INFLIGHT = "DOCLING_INFLIGHT"
    STATE_GRADER_READY = "GRADER_READY"
    STATE_GRADER_INFLIGHT = "GRADER_INFLIGHT"
    STATE_SYNTHESIS_READY = "SYNTHESIS_READY"
    STATE_SYNTHESIS_INFLIGHT = "SYNTHESIS_INFLIGHT"
    STATE_DONE = "DONE"
    STATE_FAILED = "FAILED"

    service_health_wait_seconds = int(
        os.environ.get("SERVICE_HEALTH_WAIT_SECONDS", "900")
    )
    service_health_wait_seconds = max(60, service_health_wait_seconds)
    logger.info(
        "download_node: SERVICE_HEALTH_WAIT_SECONDS={} (LLM / Docling / Grader /healthz)",
        service_health_wait_seconds,
    )
    gpu_url_bases = _resolve_gpu_url_bases(gpu_host, gpu_port)
    if not gpu_url_bases and not os.environ.get("GPU_JOB_IDS", "").strip():
        raise RuntimeError(
            "At least one synthesis GPU endpoint is required (GPU_URLS, GPU_HOSTS, "
            "GPU_JOB_IDS, or --gpu-host)"
        )

    gpu_job_specs = _parse_gpu_job_specs(gpu_port)
    gpu_discovery_enabled = bool(gpu_job_specs)
    gpu_pending_specs: List[Dict[str, Any]] = [
        {"job_id": jid, "port": port, "failed": False, "failed_logged": False}
        for jid, port in gpu_job_specs
    ]
    _gpu_pruned = _prune_gpu_pending_specs(gpu_pending_specs, gpu_url_bases)
    if _gpu_pruned:
        logger.info(
            "download_node: {} synthesis GPU Slurm job(s) skipped for discovery "
            "(ports already in GPU_URLS: {})",
            _gpu_pruned,
            sorted(_registered_gpu_ports(gpu_url_bases)),
        )
    _num_synth_nodes_raw = os.environ.get("NUM_SYNTHESIS_NODES", "").strip()
    _num_synth_nodes_want = 0
    if _num_synth_nodes_raw:
        try:
            _num_synth_nodes_want = max(0, int(_num_synth_nodes_raw))
        except ValueError:
            _num_synth_nodes_want = 0
    if gpu_job_specs:
        gpu_target = len(gpu_job_specs)
        if _num_synth_nodes_want and gpu_target < _num_synth_nodes_want:
            logger.warning(
                "download_node: GPU_JOB_IDS lists {} job(s) but NUM_SYNTHESIS_NODES={}",
                gpu_target,
                _num_synth_nodes_want,
            )
            gpu_target = _num_synth_nodes_want
    elif _num_synth_nodes_want:
        gpu_target = _num_synth_nodes_want
    else:
        gpu_target = max(1, len(gpu_url_bases) or 1)

    scheduler_lock = threading.RLock()

    def _register_gpu_url(url: str, job_id: str = "") -> bool:
        base = url.rstrip("/")
        with scheduler_lock:
            if base in gpu_url_bases:
                return False
            gpu_url_bases.append(base)
        p = urlparse(base)
        if p.port is not None:
            gpu_pending_specs[:] = [
                s for s in gpu_pending_specs if int(s["port"]) != int(p.port)
            ]
        logger.info(
            "download_node: synthesis GPU endpoint registered (job_id={}, url={})",
            job_id or "n/a",
            base,
        )
        return True

    def _refresh_gpu_endpoints_from_host_file() -> None:
        path = os.environ.get("GPU_ENDPOINTS_FILE", "").strip()
        if not path or not os.path.isfile(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                url = line.split()[0]
                if url.startswith("http://") or url.startswith("https://"):
                    _register_gpu_url(url, "host_discovery")

    def _refresh_gpu_endpoints() -> None:
        _refresh_gpu_endpoints_from_host_file()
        if not gpu_pending_specs:
            return
        still_pending: List[Dict[str, Any]] = []
        for spec in gpu_pending_specs:
            if spec.get("failed"):
                continue
            jid = str(spec["job_id"])
            port = int(spec["port"])
            node = get_job_node(jid)
            if not node:
                st = get_job_state(jid)
                if is_terminal_job_state(st):
                    if not spec.get("failed_logged"):
                        logger.warning(
                            "Synthesis GPU Slurm job {} entered terminal state {}",
                            jid,
                            st,
                        )
                        spec["failed_logged"] = True
                    spec["failed"] = True
                else:
                    still_pending.append(spec)
                continue
            url = f"http://{node}:{port}".rstrip("/")
            with scheduler_lock:
                already_registered = url in gpu_url_bases
            if already_registered:
                continue
            if not _grader_health_ok(url):
                still_pending.append(spec)
                continue
            if not _register_gpu_url(url, jid):
                still_pending.append(spec)
        gpu_pending_specs[:] = still_pending

    if gpu_discovery_enabled:
        logger.info(
            "download_node: dynamic synthesis GPU discovery enabled ({} Slurm job(s), "
            "{} URL(s) at startup)",
            len(gpu_job_specs),
            len(gpu_url_bases),
        )
        deadline = time.monotonic() + service_health_wait_seconds
        while time.monotonic() < deadline:
            _refresh_gpu_endpoints()
            if any(_grader_health_ok(u) for u in gpu_url_bases):
                break
            time.sleep(5)
        else:
            raise RuntimeError(
                "No synthesis GPU endpoint became healthy within "
                f"{service_health_wait_seconds}s (GPU_JOB_IDS discovery)"
            )
    for synth_url in list(gpu_url_bases):
        logger.info(
            "download_node: probing synthesis LLM GPU health at {}/healthz",
            synth_url.rstrip("/"),
        )
        sys.stdout.flush()
        if not _wait_health(
            "LLM GPU",
            synth_url,
            timeout=service_health_wait_seconds if not gpu_discovery_enabled else min(
                120, service_health_wait_seconds
            ),
        ):
            if gpu_discovery_enabled:
                logger.warning(
                    "Synthesis GPU at {} not healthy yet; will retry via discovery",
                    synth_url,
                )
            else:
                raise RuntimeError(f"GPU node not healthy at {synth_url}")
    if gpu_discovery_enabled:
        _refresh_gpu_endpoints()
    logger.info(
        "download_node: {} synthesis GPU endpoint(s) active: {}",
        len(gpu_url_bases),
        ", ".join(gpu_url_bases),
    )

    docling_url_base = ""
    if docling_host or os.environ.get("DOCLING_JOB_ID", "").strip():
        docling_job_id = os.environ.get("DOCLING_JOB_ID", "").strip()

        def _docling_url_base() -> str:
            host = (docling_host or "").strip()
            if docling_job_id:
                node = get_job_node(docling_job_id)
                if node:
                    host = node
            if not host:
                return ""
            return f"http://{host}:{docling_port}"

        docling_url_base = _docling_url_base()
        logger.info(
            "download_node: probing Docling health at {}/healthz{}",
            docling_url_base.rstrip("/"),
            f" (DOCLING_JOB_ID={docling_job_id})" if docling_job_id else "",
        )
        sys.stdout.flush()
        if not _wait_health(
            "Docling",
            docling_url_base or f"http://{docling_host}:{docling_port}",
            timeout=service_health_wait_seconds,
            host_resolver=_docling_url_base,
        ):
            raise RuntimeError(
                f"Docling node not healthy at {docling_url_base or docling_host}:{docling_port}"
            )
        docling_url_base = _docling_url_base()
    grader_url_bases = _resolve_grader_url_bases(grader_host, grader_port)
    if not grader_url_bases and not os.environ.get("GRADER_JOB_IDS", "").strip():
        raise RuntimeError(
            "At least one grader endpoint is required (GRADER_URLS, GRADER_HOSTS, "
            "GRADER_JOB_IDS, or --grader-host)"
        )
    if not host_rubric_path or not microbe_rubric_path:
        raise RuntimeError(
            "HOST_RUBRIC_PATH and MICROBE_RUBRIC_PATH are required for grading"
        )

    grader_job_specs = _parse_grader_job_specs(grader_port)
    grader_discovery_enabled = bool(grader_job_specs)
    grader_pending_specs: List[Dict[str, Any]] = [
        {"job_id": jid, "port": port, "failed": False, "failed_logged": False}
        for jid, port in grader_job_specs
    ]
    _pruned = _prune_grader_pending_specs(grader_pending_specs, grader_url_bases)
    if _pruned:
        logger.info(
            "download_node: {} grader Slurm job(s) skipped for discovery "
            "(ports already in GRADER_URLS: {})",
            _pruned,
            sorted(_registered_grader_ports(grader_url_bases)),
        )
    _num_grader_nodes_raw = os.environ.get("NUM_GRADER_NODES", "").strip()
    _num_grader_nodes_want = 0
    if _num_grader_nodes_raw:
        try:
            _num_grader_nodes_want = max(0, int(_num_grader_nodes_raw))
        except ValueError:
            _num_grader_nodes_want = 0
    if grader_job_specs:
        grader_target = len(grader_job_specs)
        if _num_grader_nodes_want and grader_target < _num_grader_nodes_want:
            logger.warning(
                "download_node: GRADER_JOB_IDS lists {} job(s) but NUM_GRADER_NODES={}; "
                "check sbatch --export (use colon-separated GRADER_JOB_IDS, not commas)",
                grader_target,
                _num_grader_nodes_want,
            )
            grader_target = _num_grader_nodes_want
    elif _num_grader_nodes_want:
        grader_target = _num_grader_nodes_want
    else:
        grader_target = max(1, len(grader_url_bases) or 1)
    _grader_limit_raw = os.environ.get("GRADER_INFLIGHT_CAP", "").strip()
    grader_inflight_limit: int | None = None
    if _grader_limit_raw:
        try:
            grader_inflight_limit = max(1, int(_grader_limit_raw))
        except ValueError:
            logger.warning(
                "download_node: invalid GRADER_INFLIGHT_CAP={!r}; ignoring",
                _grader_limit_raw,
            )

    grader_url_penalty_until: Dict[str, float] = {}
    grader_url_penalty_seconds = max(
        30, int(os.environ.get("GRADER_URL_PENALTY_SECONDS", "300"))
    )

    def _drop_pending_specs_for_registered_port(port: int) -> None:
        grader_pending_specs[:] = [
            s for s in grader_pending_specs if int(s["port"]) != port
        ]

    def _penalize_grader_url(url: str, *, reason: str = "") -> None:
        base = url.rstrip("/")
        until = time.monotonic() + float(grader_url_penalty_seconds)
        with scheduler_lock:
            prev = grader_url_penalty_until.get(base, 0.0)
            grader_url_penalty_until[base] = max(prev, until)
        if reason:
            logger.warning("Grader endpoint penalized for {}s: {} ({})", grader_url_penalty_seconds, base, reason)

    def _grader_url_is_penalized(url: str) -> bool:
        base = url.rstrip("/")
        with scheduler_lock:
            until = grader_url_penalty_until.get(base, 0.0)
        return until > time.monotonic()

    def _prune_unhealthy_grader_urls(*, log_removals: bool = False) -> int:
        removed = 0
        with scheduler_lock:
            urls = list(grader_url_bases)
        kept: List[str] = []
        for url in urls:
            if _grader_url_is_penalized(url):
                continue
            if _grader_health_ok(url):
                kept.append(url)
            else:
                removed += 1
                if log_removals:
                    logger.warning(
                        "download_node: dropping unhealthy grader endpoint {}",
                        url,
                    )
        with scheduler_lock:
            grader_url_bases[:] = kept
        return removed

    def _register_grader_url(url: str, job_id: str = "") -> bool:
        base = url.rstrip("/")
        if not _grader_health_ok(base):
            return False
        with scheduler_lock:
            if base in grader_url_bases:
                return False
            grader_url_bases.append(base)
        p = urlparse(base)
        if p.port is not None:
            _drop_pending_specs_for_registered_port(int(p.port))
        logger.info(
            "download_node: grader endpoint registered (job_id={}, url={})",
            job_id or "n/a",
            base,
        )
        return True

    def _refresh_grader_endpoints_from_host_file() -> None:
        """Register endpoints discovered by host-side squeue+curl (see grader_endpoints_discover.sh)."""
        path = os.environ.get("GRADER_ENDPOINTS_FILE", "").strip()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        except OSError as e:
            logger.warning(
                "download_node: could not read GRADER_ENDPOINTS_FILE {}: {}",
                path,
                e,
            )
            return
        for line in lines:
            url = line.split()[0]
            if url.startswith("http://") or url.startswith("https://"):
                _register_grader_url(url, "host_discovery")

    def _refresh_grader_endpoints() -> None:
        _refresh_grader_endpoints_from_host_file()
        if not grader_pending_specs:
            return
        still_pending: List[Dict[str, Any]] = []
        for spec in grader_pending_specs:
            if spec.get("failed"):
                continue
            jid = str(spec["job_id"])
            port = int(spec["port"])
            node = get_job_node(jid)
            if not node:
                st = get_job_state(jid)
                if is_terminal_job_state(st):
                    if not spec.get("failed_logged"):
                        logger.warning(
                            "Grader Slurm job {} entered terminal state {}",
                            jid,
                            st,
                        )
                        spec["failed_logged"] = True
                    spec["failed"] = True
                else:
                    still_pending.append(spec)
                continue
            url = f"http://{node}:{port}".rstrip("/")
            with scheduler_lock:
                already_registered = url in grader_url_bases
            if already_registered:
                logger.info(
                    "download_node: grader job {} already served at {}",
                    jid,
                    url,
                )
                continue
            if not _grader_health_ok(url):
                still_pending.append(spec)
                continue
            if not _register_grader_url(url, jid):
                still_pending.append(spec)
        grader_pending_specs[:] = still_pending

    if grader_discovery_enabled:
        logger.info(
            "download_node: dynamic grader discovery enabled ({} Slurm job(s), "
            "{} URL(s) at startup)",
            len(grader_job_specs),
            len(grader_url_bases),
        )
        sys.stdout.flush()
        _refresh_grader_endpoints_from_host_file()
        preconfigured = bool(os.environ.get("GRADER_URLS", "").strip())
        if not grader_url_bases:
            deadline = time.monotonic() + min(120, service_health_wait_seconds)
            while time.monotonic() < deadline:
                _refresh_grader_endpoints()
                if grader_url_bases and any(
                    _grader_health_ok(u) for u in grader_url_bases
                ):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(
                    "No grader endpoint discovered within "
                    f"{min(120, service_health_wait_seconds)}s"
                )
        else:
            healthy_n = sum(1 for u in grader_url_bases if _grader_health_ok(u))
            logger.info(
                "download_node: {} grader URL(s) configured, {} healthy at startup",
                len(grader_url_bases),
                healthy_n,
            )
            sys.stdout.flush()
            if healthy_n == 0:
                deadline = time.monotonic() + service_health_wait_seconds
                logger.info(
                    "download_node: preconfigured grader URL(s) not healthy yet; "
                    "waiting up to {}s with Slurm discovery",
                    service_health_wait_seconds,
                )
                sys.stdout.flush()
                while time.monotonic() < deadline:
                    _refresh_grader_endpoints()
                    healthy_n = sum(1 for u in grader_url_bases if _grader_health_ok(u))
                    if healthy_n > 0:
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"No healthy grader in configured URLs ({len(grader_url_bases)} total) "
                        f"after {service_health_wait_seconds}s"
                    )
            if preconfigured and healthy_n < len(grader_url_bases):
                logger.info(
                    "download_node: skipping serial health wait for {} unhealthy "
                    "preconfigured grader URL(s); background discovery will prune them",
                    len(grader_url_bases) - healthy_n,
                )
        sys.stdout.flush()
    else:
        for grader_url_base in grader_url_bases:
            logger.info(
                "download_node: probing Grader health at {}/healthz",
                grader_url_base.rstrip("/"),
            )
            if not _wait_health(
                "Grader", grader_url_base, timeout=service_health_wait_seconds
            ):
                raise RuntimeError(f"Grader node not healthy at {grader_url_base}")

    if grader_discovery_enabled and grader_pending_specs:
        _refresh_grader_endpoints()
    _prune_unhealthy_grader_urls(log_removals=True)
    logger.info(
        "download_node: {} grader endpoint(s) active: {}",
        len(grader_url_bases),
        ", ".join(grader_url_bases),
    )
    sys.stdout.flush()
    if grader_discovery_enabled and len(grader_url_bases) < grader_target:
        logger.info(
            "download_node: waiting for up to {} more grader endpoint(s) via "
            "GRADER_JOB_IDS discovery",
            grader_target - len(grader_url_bases),
        )

    instructions_text = instructions
    if instructions_file and os.path.isfile(instructions_file):
        with open(instructions_file, "r", encoding="utf-8") as f:
            instructions_text = f.read().strip()

    data = _load_search_json(paper_ids_path)
    logger.info(
        "download_node: loaded search JSON ({} query keys)",
        len(data) if isinstance(data, dict) else 0,
    )
    sys.stdout.flush()
    planned_alignment_ids: set[str] = set()
    if isinstance(data, dict):
        for query_id, alignments in data.items():
            if not isinstance(alignments, list):
                continue
            for al in alignments:
                if not isinstance(al, dict):
                    continue
                target = al.get("target") or ""
                aid = f"{query_id}_{target}".replace("/", "_").replace(" ", "_")
                if _alignment_paper_ids(al):
                    planned_alignment_ids.add(aid)
    logger.info(
        "download_node: {} alignment packet(s) planned from search JSON",
        len(planned_alignment_ids),
    )
    collection_org = os.environ.get("COLLECTION_ORG", "ucsc").strip() or "ucsc"
    collection_auth_scope = (
        os.environ.get("COLLECTION_AUTH_SCOPE", "email_only").strip() or "email_only"
    )
    collector_email = os.environ.get("COLLECTOR_EMAIL", "").strip()
    if (
        collection_org.lower() == "ucsc"
        and collection_auth_scope.lower() == "email_only"
        and not collector_email
    ):
        raise RuntimeError(
            "COLLECTOR_EMAIL is required for UCSC email_only collection mode."
        )

    logger.info(
        "Collection mode org={} scope={} collector_email_set={}",
        collection_org,
        collection_auth_scope,
        "yes" if bool(collector_email) else "no",
    )

    _mw_raw = os.environ.get("COLLECT_MAX_WORKERS", "2").strip() or "2"
    try:
        collect_max_workers = max(1, min(16, int(_mw_raw)))
    except ValueError:
        collect_max_workers = 2
    collect_disable_s2 = os.environ.get(
        "COLLECT_DISABLE_SEMANTIC_SCHOLAR", ""
    ).strip().lower() in ("1", "true", "yes")
    logger.info(
        "Collect parallelism max_workers={} semantic_scholar={}",
        collect_max_workers,
        "off" if collect_disable_s2 else "on",
    )

    # Optional: mapping CSV for richer gene identifiers; provided via env.
    idmap_path = os.environ.get("IDMAP_CSV", "")
    idmap: Dict[str, Dict[str, Any]] = {}
    if idmap_path:
        idmap = _load_idmap(idmap_path)
    papers_base = os.path.join(data_root, "papers")
    logs_base = os.path.join(data_root, "logs")
    run_logs_dir = os.environ.get("RUN_LOGS_DIR", "").strip() or os.path.join(
        output_root, "logs"
    )
    os.makedirs(papers_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(run_logs_dir, exist_ok=True)
    scheduler_state_dir = os.environ.get("SCHEDULER_STATE_DIR", "").strip() or os.path.join(
        output_root, "scheduler_state"
    )
    os.makedirs(scheduler_state_dir, exist_ok=True)

    if os.environ.get("DOWNLOAD_PROGRESS_LOG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        _p_path = (
            os.environ.get("DOWNLOAD_PROGRESS_LOG_PATH", "").strip()
            or os.path.join(run_logs_dir, "download_progress.log")
        )
        try:
            os.makedirs(os.path.dirname(_p_path) or ".", exist_ok=True)
            logger.add(
                _p_path,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                filter=_only_download_progress_log,
                enqueue=True,
            )
        except Exception as e:
            logger.warning("Could not add download_progress log sink: {}", e)

    session = requests.Session()
    # Docling/grader polling runs on a background thread; use a separate Session
    # because requests.Session is not documented as thread-safe.
    scheduler_session = requests.Session()
    # Avoid stale pooled connections through NATs / proxies (intermittent
    # ConnectionResetError on status polls when the lock previously wrapped HTTP).
    scheduler_session.headers["Connection"] = "close"
    pmcid_cache: Dict[str, str | None] = {}
    retry_failed = _retry_failed_enabled()
    global_cache = load_global_outcome_cache(logs_base)
    total_done = 0
    failed_count = 0
    docling_inflight_cap = max(1, int(os.environ.get("DOCLING_INFLIGHT_CAP", "1")))
    logger.info(
        "download_node: {} grader endpoint(s) at startup, discovery target={}, "
        "grader parallelism=one alignment per endpoint{}",
        len(grader_url_bases),
        grader_target,
        (
            f", GRADER_INFLIGHT_CAP={grader_inflight_limit} (manual ceiling)"
            if grader_inflight_limit is not None
            else ""
        ),
    )
    try:
        cluster_respect = max(0, int(os.environ.get("CLUSTER_RESPECT", "0") or "0"))
    except ValueError:
        cluster_respect = 0
    try:
        respect_threshold = max(
            0, int(float(os.environ.get("RESPECT_THRESHOLD", "0") or "0"))
        )
    except ValueError:
        respect_threshold = 0
    grader_scale_down_done = False
    grader_scale_down_file = os.environ.get("GRADER_SCALE_DOWN_FILE", "").strip()
    if not grader_scale_down_file and run_logs_dir:
        grader_scale_down_file = os.path.join(
            run_logs_dir, "grader_scale_down_jobs.txt"
        )
    if cluster_respect > 0 and respect_threshold > 0:
        logger.info(
            "download_node: cluster_respect enabled "
            "(kill_up_to={}, when remaining grade packets <= {})",
            cluster_respect,
            respect_threshold,
        )
    docling_submit_max_attempts = max(
        1, int(os.environ.get("DOCLING_SUBMIT_MAX_ATTEMPTS", "4"))
    )
    scheduler_tick_seconds = max(
        1, int(os.environ.get("SCHEDULER_TICK_SECONDS", "5"))
    )
    # Grader and Docling jobs can legitimately run for hours on large packets.
    # Keep a long watchdog by default; still user-overridable via env.
    stage_watchdog_seconds = max(
        300, int(os.environ.get("STAGE_WATCHDOG_SECONDS", "14400"))
    )

    alignment_states: Dict[str, Dict[str, Any]] = {}
    docling_inflight: Dict[str, Dict[str, Any]] = {}
    grader_inflight: Dict[str, Dict[str, Any]] = {}
    synthesis_inflight: Dict[str, Dict[str, Any]] = {}
    grader_dispatch_rr = 0
    grader_submit_max_attempts = max(
        1, int(os.environ.get("GRADER_SUBMIT_MAX_ATTEMPTS", "12"))
    )

    def _graded_path(alignment_id: str) -> str:
        return os.path.join(output_root, f"{alignment_id}_graded.json")

    def _remaining_grade_packets() -> int:
        """Alignments that still need grading (no graded.json yet, not FAILED)."""
        n = 0
        for aid in planned_alignment_ids:
            if _graded_exists(aid):
                continue
            with scheduler_lock:
                st = alignment_states.get(aid)
                if st and st.get("state") == STATE_FAILED:
                    continue
            n += 1
        return n

    def _results_path(alignment_id: str) -> str:
        return os.path.join(output_root, f"{alignment_id}_results.json")

    def _graded_exists(alignment_id: str) -> bool:
        return os.path.isfile(_graded_path(alignment_id))

    def _outputs_done(alignment_id: str) -> bool:
        return _graded_exists(alignment_id) and os.path.isfile(_results_path(alignment_id))

    def _state_path(alignment_id: str) -> str:
        return os.path.join(scheduler_state_dir, f"{alignment_id}.json")

    def _write_state(alignment_id: str) -> None:
        st = alignment_states.get(alignment_id)
        if not st:
            return
        try:
            with open(_state_path(alignment_id), "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2)
        except Exception as e:
            logger.warning("Could not write scheduler state for {}: {}", alignment_id, e)

    def _reconcile_outputs_into_state() -> None:
        """Fix stale FAILED/INFLIGHT states when outputs already exist on disk."""
        with scheduler_lock:
            for aid, st in alignment_states.items():
                if st.get("state") == STATE_DONE:
                    continue
                prev = str(st.get("state") or "")
                if _outputs_done(aid):
                    st["state"] = STATE_DONE
                    st.pop("last_error", None)
                    st["updated_at"] = time.time()
                    docling_inflight.pop(aid, None)
                    grader_inflight.pop(aid, None)
                    logger.info(
                        "Alignment {}: outputs present on disk; reconciled state {} -> DONE",
                        aid,
                        prev or "UNKNOWN",
                    )
                    _write_state(aid)
                    continue
                if _graded_exists(aid) and not os.path.isfile(_results_path(aid)):
                    if st.get("state") in {
                        STATE_FAILED,
                        STATE_GRADER_INFLIGHT,
                        STATE_GRADER_READY,
                    }:
                        st["state"] = STATE_SYNTHESIS_READY
                        st.pop("last_error", None)
                        st["updated_at"] = time.time()
                        docling_inflight.pop(aid, None)
                        grader_inflight.pop(aid, None)
                        logger.info(
                            "Alignment {}: graded.json present; reconciled state {} -> SYNTHESIS_READY",
                            aid,
                            prev or "UNKNOWN",
                        )
                        _write_state(aid)
                    continue
                if st.get("state") == STATE_FAILED:
                    pd = str(st.get("papers_dir") or "")
                    if (
                        _papers_dir_has_nonempty_txt(pd)
                        and not _graded_exists(aid)
                        and not os.path.isfile(_results_path(aid))
                    ):
                        st["state"] = STATE_GRADER_READY
                        st["updated_at"] = time.time()
                        docling_inflight.pop(aid, None)
                        logger.info(
                            "Alignment {}: FAILED with text on disk; reconciled -> GRADER_READY",
                            aid,
                        )
                        _write_state(aid)

    def _sync_scheduler_states_from_disk() -> None:
        """Pick up state transitions written by offline requeue scripts (no CPU restart)."""
        if not os.path.isdir(scheduler_state_dir):
            return
        with scheduler_lock:
            for aid, st in list(alignment_states.items()):
                if str(st.get("state") or "") != STATE_FAILED:
                    continue
                path = _state_path(aid)
                if not os.path.isfile(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        on_disk = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(on_disk, dict):
                    continue
                disk_state = str(on_disk.get("state") or "")
                if disk_state and disk_state != STATE_FAILED:
                    alignment_states[aid] = on_disk
                    logger.info(
                        "Alignment {}: picked up disk state {} from offline requeue",
                        aid,
                        disk_state,
                    )

    def _required_docling_txt_done(st: Dict[str, Any]) -> bool:
        req = st.get("docling_required_basenames") or []
        if not req:
            return True
        papers_dir = str(st.get("papers_dir") or "")
        for base in req:
            if _paper_has_usable_text(papers_dir, base):
                continue
            txt_path = os.path.join(papers_dir, f"{base}.txt")
            if not (os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0):
                return False
        return True

    def _recompute_docling_required(
        papers_dir: str,
        manifest_map: Dict[Tuple[str, str], Dict[str, Any]] | None = None,
    ) -> List[str]:
        return _infer_docling_required_basenames(papers_dir, manifest_map)

    def _papers_dir_has_nonempty_txt(papers_dir: str) -> bool:
        if not papers_dir or not os.path.isdir(papers_dir):
            return False
        for fname in os.listdir(papers_dir):
            if not fname.endswith(".txt"):
                continue
            p = os.path.join(papers_dir, fname)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return True
        return False

    def _is_grader_ready(st: Dict[str, Any]) -> bool:
        aid = st["alignment_id"]
        if _outputs_done(aid):
            return False
        if _graded_exists(aid):
            return False
        papers_dir = str(st.get("papers_dir") or "")
        if not _papers_dir_has_nonempty_txt(papers_dir):
            return False
        return _required_docling_txt_done(st)

    def _bootstrap_state(alignment_id: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(defaults)
        path = _state_path(alignment_id)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if isinstance(prev, dict):
                    state.update(prev)
            except Exception:
                pass
        _refresh_grader_payload_constraints(state.get("grader_payload"))
        papers_dir = str(state.get("papers_dir") or "")
        if papers_dir and os.path.isdir(papers_dir):
            manifest_path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
            manifest_map = (
                _load_download_manifest(manifest_path)
                if os.path.isfile(manifest_path)
                else None
            )
            state["docling_required_basenames"] = _recompute_docling_required(
                papers_dir, manifest_map
            )
        if _outputs_done(alignment_id):
            state["state"] = STATE_DONE
        elif _graded_exists(alignment_id) and not os.path.isfile(
            _results_path(alignment_id)
        ):
            state["state"] = STATE_SYNTHESIS_READY
        elif _is_grader_ready(state):
            state["state"] = STATE_GRADER_READY
        elif (state.get("docling_required_basenames") or []) and not _required_docling_txt_done(state):
            state["state"] = STATE_DOCLING_PENDING
        return state

    def _poll_inflight() -> None:
        nonlocal total_done, failed_count
        docling_poll: List[Tuple[str, str]] = []
        with scheduler_lock:
            for aid, meta in list(docling_inflight.items()):
                st = alignment_states.get(aid)
                if not st:
                    docling_inflight.pop(aid, None)
                    continue
                if time.monotonic() - float(meta.get("started_monotonic", 0)) > stage_watchdog_seconds:
                    st["state"] = STATE_FAILED
                    st["last_error"] = "docling watchdog timeout"
                    failed_count += 1
                    docling_inflight.pop(aid, None)
                    _write_state(aid)
                    continue
                jid = str(meta.get("job_id") or "").strip()
                if jid:
                    docling_poll.append((aid, jid))

        for aid, job_id in docling_poll:
            try:
                status = _docling_status_once(
                    scheduler_session, docling_url_base, job_id
                )
            except Exception as e:
                with scheduler_lock:
                    meta2 = docling_inflight.get(aid)
                    if not meta2 or str(meta2.get("job_id")) != job_id:
                        continue
                    errs = int(meta2.get("status_errors", 0)) + 1
                    meta2["status_errors"] = errs
                    if errs == 1 or errs % 5 == 0:
                        logger.warning(
                            "Alignment {} docling status poll failed (job_id={}, errors={}): {}",
                            aid,
                            job_id,
                            errs,
                            e,
                        )
                continue

            s = str(status.get("status") or "").strip().lower()
            with scheduler_lock:
                meta2 = docling_inflight.get(aid)
                if not meta2 or str(meta2.get("job_id")) != job_id:
                    continue
                meta2["status_errors"] = 0
                if s not in {"succeeded", "failed"}:
                    continue
                st = alignment_states.get(aid)
                if not st:
                    docling_inflight.pop(aid, None)
                    continue
                docling_inflight.pop(aid, None)
                if s == "failed":
                    err = str(status.get("error") or "docling failed")
                    pd = str(st.get("papers_dir") or "")
                    if _papers_dir_has_nonempty_txt(pd) and not _graded_exists(aid):
                        st["state"] = STATE_GRADER_READY
                        st["last_error"] = err
                        logger.warning(
                            "Alignment {}: docling failed ({}) but papers_dir has "
                            "usable text; proceeding to GRADER_READY",
                            aid,
                            err[:120],
                        )
                    else:
                        st["state"] = STATE_FAILED
                        st["last_error"] = err
                        failed_count += 1
                elif _outputs_done(aid):
                    st["state"] = STATE_DONE
                    total_done += 1
                elif _is_grader_ready(st):
                    st["state"] = STATE_GRADER_READY
                else:
                    pd = str(st.get("papers_dir") or "")
                    missing_txt: List[str] = []
                    for b in st.get("docling_required_basenames") or []:
                        tp = os.path.join(pd, f"{b}.txt")
                        if not (os.path.isfile(tp) and os.path.getsize(tp) > 0):
                            missing_txt.append(b)
                    if _papers_dir_has_nonempty_txt(pd):
                        st["state"] = STATE_GRADER_READY
                        if missing_txt:
                            logger.warning(
                                "Alignment {}: docling succeeded but {} required basenames "
                                "still lack non-empty .txt (sample={}); "
                                "proceeding to grader with partial text",
                                aid,
                                len(missing_txt),
                                missing_txt[:12],
                            )
                    else:
                        st["state"] = STATE_FAILED
                        st["last_error"] = "docling succeeded but package not grader-ready"
                        failed_count += 1
                        logger.error(
                            "Alignment {}: docling finished but not grader-ready "
                            "(expected non-empty {{base}}.txt for each docling_required basename; "
                            "missing count={}, sample={})",
                            aid,
                            len(missing_txt),
                            missing_txt[:12],
                        )
                _write_state(aid)

        grader_poll: List[Tuple[str, str, str]] = []
        with scheduler_lock:
            for aid, meta in list(grader_inflight.items()):
                st = alignment_states.get(aid)
                if not st:
                    grader_inflight.pop(aid, None)
                    continue
                if time.monotonic() - float(meta.get("started_monotonic", 0)) > stage_watchdog_seconds:
                    grader_inflight.pop(aid, None)
                    if _graded_exists(aid):
                        st["state"] = STATE_SYNTHESIS_READY
                        st.pop("last_error", None)
                        logger.info(
                            "Alignment {}: grader watchdog elapsed but graded.json present; "
                            "queued for CPU synthesis",
                            aid,
                        )
                    else:
                        st["state"] = STATE_FAILED
                        st["last_error"] = "grader watchdog timeout"
                        failed_count += 1
                    _write_state(aid)
                    continue
                gj = str(meta.get("job_id") or "").strip()
                gu = str(
                    meta.get("grader_url_base")
                    or st.get("grader_url_base")
                    or ""
                ).strip()
                if gj and gu:
                    grader_poll.append((aid, gj, gu))

        for aid, job_id, poll_grader_url in grader_poll:
            try:
                status = _grader_status_once(
                    scheduler_session, poll_grader_url, job_id
                )
            except Exception:
                continue
            s = str(status.get("status") or "").strip().lower()
            with scheduler_lock:
                meta2 = grader_inflight.get(aid)
                if not meta2 or str(meta2.get("job_id")) != job_id:
                    continue
                if s not in {"succeeded", "failed"}:
                    continue
                st = alignment_states.get(aid)
                if not st:
                    grader_inflight.pop(aid, None)
                    continue
                grader_inflight.pop(aid, None)
                if s == "failed":
                    st["state"] = STATE_FAILED
                    st["last_error"] = str(status.get("error") or "grader failed")
                    failed_count += 1
                elif _outputs_done(aid):
                    st["state"] = STATE_DONE
                    total_done += 1
                elif _graded_exists(aid):
                    st["state"] = STATE_SYNTHESIS_READY
                    st.pop("last_error", None)
                    logger.info(
                        "Alignment {}: grader finished; queued for CPU synthesis",
                        aid,
                    )
                else:
                    st["state"] = STATE_FAILED
                    st["last_error"] = "grader succeeded but graded.json missing"
                    failed_count += 1
                _write_state(aid)

        synth_poll: List[str] = []
        with scheduler_lock:
            for aid, meta in list(synthesis_inflight.items()):
                st = alignment_states.get(aid)
                if not st:
                    synthesis_inflight.pop(aid, None)
                    continue
                thread = meta.get("thread")
                if thread is not None and thread.is_alive():
                    if (
                        time.monotonic() - float(meta.get("started_monotonic", 0))
                        > stage_watchdog_seconds
                    ):
                        st["state"] = STATE_FAILED
                        st["last_error"] = "synthesis watchdog timeout"
                        failed_count += 1
                        synthesis_inflight.pop(aid, None)
                        _write_state(aid)
                    continue
                synth_poll.append(aid)

        for aid in synth_poll:
            with scheduler_lock:
                meta2 = synthesis_inflight.pop(aid, None)
                st = alignment_states.get(aid)
                if not st:
                    continue
                err = (meta2 or {}).get("error")
                if err:
                    st["state"] = STATE_FAILED
                    st["last_error"] = str(err)
                    failed_count += 1
                elif _outputs_done(aid):
                    st["state"] = STATE_DONE
                    total_done += 1
                    st.pop("last_error", None)
                else:
                    st["state"] = STATE_FAILED
                    st["last_error"] = "synthesis finished but results file missing"
                    failed_count += 1
                _write_state(aid)

    def _dispatch_synthesis() -> None:
        def _synthesis_inflight_count_for_url(url: str) -> int:
            return sum(
                1
                for meta in synthesis_inflight.values()
                if str(meta.get("gpu_url_base") or "") == url
            )

        def _pick_gpu_url_for_synthesis() -> str | None:
            with scheduler_lock:
                candidates = [
                    u
                    for u in gpu_url_bases
                    if _synthesis_inflight_count_for_url(u) < 1
                ]
            if not candidates:
                return None
            candidates.sort(key=lambda u: (_synthesis_inflight_count_for_url(u), u))
            return candidates[0]

        max_rounds = max(1, len(gpu_url_bases))
        for _ in range(max_rounds):
            with scheduler_lock:
                if not gpu_url_bases:
                    return
                busy = all(
                    _synthesis_inflight_count_for_url(u) >= 1 for u in gpu_url_bases
                )
                if busy:
                    return
            pending_aid: str | None = None
            with scheduler_lock:
                for aid, st in alignment_states.items():
                    if st.get("state") != STATE_SYNTHESIS_READY:
                        continue
                    pending_aid = aid
                    st["state"] = STATE_SYNTHESIS_INFLIGHT
                    st["synthesis_submitted_at"] = time.time()
                    break
            if not pending_aid:
                return
            synth_url = _pick_gpu_url_for_synthesis()
            if not synth_url:
                with scheduler_lock:
                    st = alignment_states.get(pending_aid)
                    if st and st.get("state") == STATE_SYNTHESIS_INFLIGHT:
                        st["state"] = STATE_SYNTHESIS_READY
                        st.pop("synthesis_submitted_at", None)
                return

            def _synthesis_worker(alignment_id: str, gpu_base: str) -> None:
                err: str | None = None
                try:
                    payload = build_run_alignment_graded_payload(
                        alignment_id,
                        output_root,
                        papers_root=papers_base,
                        instructions=instructions_text,
                    )
                    post_run_alignment_graded(gpu_base, payload)
                except Exception as e:
                    err = f"synthesis failed: {e}"
                    logger.error("Alignment {}: {}", alignment_id, err)
                with scheduler_lock:
                    meta = synthesis_inflight.get(alignment_id)
                    if meta is not None:
                        meta["error"] = err

            worker = threading.Thread(
                target=_synthesis_worker,
                args=(pending_aid, synth_url),
                name=f"lit-synthesis-{pending_aid}",
                daemon=True,
            )
            with scheduler_lock:
                synthesis_inflight[pending_aid] = {
                    "thread": worker,
                    "started_monotonic": time.monotonic(),
                    "gpu_url_base": synth_url,
                }
            logger.info(
                "Alignment {}: synthesis worker started (host={})",
                pending_aid,
                synth_url,
            )
            worker.start()
            with scheduler_lock:
                _write_state(pending_aid)

    def _dispatch_docling() -> None:
        if not docling_url_base:
            return
        with scheduler_lock:
            if len(docling_inflight) >= docling_inflight_cap:
                return
        if not _wait_service_capacity(
            session=scheduler_session,
            base_url=docling_url_base,
            endpoint="docling_capacity",
            service_name="Docling",
            timeout_seconds=1,
            poll_interval_seconds=1,
            warn_on_timeout=False,
        ):
            return
        pending_aid: str | None = None
        pending_payload: Dict[str, Any] | None = None
        submit_attempt = 0
        with scheduler_lock:
            for aid, st in alignment_states.items():
                if st.get("state") != STATE_DOCLING_PENDING:
                    continue
                submit_attempt = int(st.get("docling_submit_attempt", 0)) + 1
                st["docling_submit_attempt"] = submit_attempt
                pending_aid = aid
                pending_payload = st["docling_payload"]
                break
        if not pending_aid or pending_payload is None:
            return
        try:
            job_id = _submit_docling_async(
                session=scheduler_session,
                docling_url_base=docling_url_base,
                payload=pending_payload,
                submit_timeout=30,
            )
        except Exception as e:
            with scheduler_lock:
                st = alignment_states.get(pending_aid)
                if not st or st.get("state") != STATE_DOCLING_PENDING:
                    return
                st["last_error"] = (
                    f"docling submit failed (attempt {submit_attempt}): {e}"
                )
                if submit_attempt >= docling_submit_max_attempts:
                    st["state"] = STATE_FAILED
                    logger.error(
                        "Alignment {}: docling submit exhausted after {} attempts: {}",
                        pending_aid,
                        submit_attempt,
                        e,
                    )
                else:
                    st["state"] = STATE_DOCLING_PENDING
                    logger.warning(
                        "Alignment {}: docling submit failed (attempt {}/{}), will retry: {}",
                        pending_aid,
                        submit_attempt,
                        docling_submit_max_attempts,
                        e,
                    )
                _write_state(pending_aid)
            return

        with scheduler_lock:
            st = alignment_states.get(pending_aid)
            if not st or st.get("state") != STATE_DOCLING_PENDING:
                return
            st["state"] = STATE_DOCLING_INFLIGHT
            st["docling_job_id"] = job_id
            st["docling_submitted_at"] = time.time()
            docling_inflight[pending_aid] = {
                "job_id": job_id,
                "started_monotonic": time.monotonic(),
                "status_errors": 0,
            }
            logger.info(
                "Alignment {}: docling submitted (attempt={}, job_id={}, host={})",
                pending_aid,
                submit_attempt,
                job_id,
                docling_url_base,
            )
            _write_state(pending_aid)

    def _grader_inflight_count_for_url(url: str) -> int:
        return sum(
            1
            for meta in grader_inflight.values()
            if str(meta.get("grader_url_base") or "") == url
        )

    def _pick_grader_url_for_dispatch() -> str | None:
        """Round-robin across grader endpoints with a free inflight slot."""
        nonlocal grader_dispatch_rr
        with scheduler_lock:
            candidates = [
                u
                for u in grader_url_bases
                if _grader_inflight_count_for_url(u) < 1
                and not _grader_url_is_penalized(u)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda u: (_grader_inflight_count_for_url(u), u))
        with scheduler_lock:
            idx = grader_dispatch_rr % len(candidates)
            grader_dispatch_rr += 1
        url = candidates[idx]
        if not _grader_health_ok(url):
            _penalize_grader_url(url, reason="healthz failed before dispatch")
            return None
        return url

    def _next_grader_ready_alignment() -> Tuple[str | None, Dict[str, Any] | None]:
        with scheduler_lock:
            for aid, st in alignment_states.items():
                if st.get("state") != STATE_GRADER_READY:
                    continue
                payload = st.get("grader_payload")
                if payload is None:
                    continue
                return aid, payload
        return None, None

    def _dispatch_grader() -> None:
        """Send GRADER_READY alignments to any endpoint with a free slot and capacity."""
        max_rounds = max(1, len(grader_url_bases))
        for _ in range(max_rounds):
            with scheduler_lock:
                if not grader_url_bases:
                    return
                if (
                    grader_inflight_limit is not None
                    and len(grader_inflight) >= grader_inflight_limit
                ):
                    return
            pending_aid, pending_payload = _next_grader_ready_alignment()
            if not pending_aid or pending_payload is None:
                return
            grader_url_base = _pick_grader_url_for_dispatch()
            if not grader_url_base:
                return
            _refresh_grader_payload_constraints(pending_payload)
            grader_mt = (pending_payload.get("constraints") or {}).get("max_tokens")
            logger.info(
                "Alignment {}: grader submit max_tokens={} host={}",
                pending_aid,
                grader_mt,
                grader_url_base,
            )
            try:
                job_id = _submit_grader_async(
                    session=scheduler_session,
                    grader_url_base=grader_url_base,
                    payload=pending_payload,
                    submit_timeout=30,
                )
            except Exception as e:
                _penalize_grader_url(grader_url_base, reason=f"submit failed: {e}")
                with scheduler_lock:
                    st = alignment_states.get(pending_aid)
                    if st and st.get("state") == STATE_GRADER_READY:
                        attempts = int(st.get("grader_submit_attempts") or 0) + 1
                        st["grader_submit_attempts"] = attempts
                        st["last_error"] = f"grader submit failed: {e}"
                        if attempts >= grader_submit_max_attempts:
                            st["state"] = STATE_FAILED
                            logger.error(
                                "Alignment {}: grader submit exhausted after {} attempts",
                                pending_aid,
                                attempts,
                            )
                        else:
                            logger.warning(
                                "Alignment {}: grader submit failed (attempt {}/{}), "
                                "will retry on another endpoint: {}",
                                pending_aid,
                                attempts,
                                grader_submit_max_attempts,
                                e,
                            )
                        _write_state(pending_aid)
                continue

            with scheduler_lock:
                st = alignment_states.get(pending_aid)
                if not st or st.get("state") != STATE_GRADER_READY:
                    continue
                st["state"] = STATE_GRADER_INFLIGHT
                st["grader_job_id"] = job_id
                st["grader_url_base"] = grader_url_base
                st["grader_submitted_at"] = time.time()
                grader_inflight[pending_aid] = {
                    "job_id": job_id,
                    "grader_url_base": grader_url_base,
                    "started_monotonic": time.monotonic(),
                }
                logger.info(
                    "Alignment {}: async grader job submitted (job_id={}, host={})",
                    pending_aid,
                    job_id,
                    grader_url_base,
                )
                _write_state(pending_aid)

    def _maybe_cluster_respect_scale_down() -> None:
        """Cancel idle grader Slurm jobs once remaining grade packets hit respect_threshold."""
        nonlocal grader_scale_down_done
        if grader_scale_down_done:
            return
        if cluster_respect <= 0 or respect_threshold <= 0:
            return
        remaining = _remaining_grade_packets()
        if not grader_scale_down_should_trigger(
            remaining_packets=remaining,
            respect_threshold=respect_threshold,
        ):
            return

        with scheduler_lock:
            urls = list(grader_url_bases)
            inflight_by_url = {
                u: sum(
                    1
                    for meta in grader_inflight.values()
                    if str(meta.get("grader_url_base") or "") == u
                )
                for u in urls
            }

        url_by_port: Dict[int, str] = {}
        for url in urls:
            try:
                port = urlparse(url).port
            except Exception:
                port = None
            if port is not None:
                url_by_port[int(port)] = url

        kill_ids = select_idle_grader_jobs_to_kill(
            job_specs=list(grader_job_specs),
            inflight_by_url=inflight_by_url,
            url_by_port=url_by_port,
            n_kill=cluster_respect,
            min_keep=1,
        )
        if not kill_ids:
            return

        kill_set = set(kill_ids)
        ports_to_drop = {
            int(spec["port"])
            for spec in grader_job_specs
            if str(spec.get("job_id") or "").strip() in kill_set
        }
        drop_urls = {url_by_port[p] for p in ports_to_drop if p in url_by_port}

        with scheduler_lock:
            grader_url_bases[:] = [u for u in grader_url_bases if u not in drop_urls]
        for spec in grader_job_specs:
            if str(spec.get("job_id") or "").strip() in kill_set:
                spec["failed"] = True
                spec["failed_logged"] = True

        if grader_scale_down_file:
            try:
                os.makedirs(os.path.dirname(grader_scale_down_file) or ".", exist_ok=True)
                tmp = grader_scale_down_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    for jid in kill_ids:
                        f.write(f"{jid}\n")
                os.replace(tmp, grader_scale_down_file)
            except OSError as e:
                logger.warning(
                    "download_node: could not write grader scale-down file {}: {}",
                    grader_scale_down_file,
                    e,
                )

        cancelled = scancel_jobs(kill_ids)
        grader_scale_down_done = True
        logger.info(
            "download_node: cluster_respect scale-down "
            "(remaining_packets={}, threshold={}, "
            "requested_kill={}, scancel_ok={}, remaining_endpoints={})",
            remaining,
            respect_threshold,
            kill_ids,
            cancelled,
            len(grader_url_bases),
        )

    _discovery_status_tick = 0

    def _tick_scheduler() -> None:
        nonlocal _discovery_status_tick
        _sync_scheduler_states_from_disk()
        _reconcile_outputs_into_state()
        n_registered_before = len(grader_url_bases)
        _refresh_grader_endpoints()
        _prune_unhealthy_grader_urls()
        if gpu_discovery_enabled and gpu_pending_specs:
            _refresh_gpu_endpoints()
        if grader_discovery_enabled and grader_pending_specs:
            _discovery_status_tick += 1
            if (
                _discovery_status_tick == 1
                or _discovery_status_tick % 12 == 0
                or len(grader_url_bases) > n_registered_before
            ):
                logger.info(
                    "download_node: grader discovery {}/{} endpoint(s) active, "
                    "{} grading inflight, {} Slurm job(s) still pending",
                    len(grader_url_bases),
                    grader_target,
                    len(grader_inflight),
                    len(grader_pending_specs),
                )
        _poll_inflight()
        _dispatch_docling()
        _dispatch_grader()
        _maybe_cluster_respect_scale_down()
        _dispatch_synthesis()

    def _preload_scheduler_states() -> int:
        """Load saved scheduler JSON so the background tick can dispatch immediately."""
        if not os.path.isdir(scheduler_state_dir):
            return 0
        loaded = 0
        for fname in os.listdir(scheduler_state_dir):
            if not fname.endswith(".json"):
                continue
            aid = fname[: -len(".json")]
            path = _state_path(aid)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    st = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(st, dict):
                continue
            state = str(st.get("state") or "")
            if state == STATE_GRADER_INFLIGHT and _graded_exists(aid):
                st["state"] = STATE_SYNTHESIS_READY
                for k in ("grader_job_id", "grader_url_base", "grader_submitted_at"):
                    st.pop(k, None)
            elif state == STATE_DOCLING_INFLIGHT and _required_docling_txt_done(st):
                st["state"] = STATE_GRADER_READY
                for k in ("docling_job_id", "docling_submitted_at"):
                    st.pop(k, None)
            with scheduler_lock:
                alignment_states[aid] = st
                if st.get("state") == STATE_GRADER_INFLIGHT:
                    gj = str(st.get("grader_job_id") or "").strip()
                    gu = str(st.get("grader_url_base") or "").strip()
                    if gj and gu:
                        grader_inflight[aid] = {
                            "job_id": gj,
                            "grader_url_base": gu,
                            "started_monotonic": time.monotonic(),
                        }
                elif st.get("state") == STATE_DOCLING_INFLIGHT:
                    dj = str(st.get("docling_job_id") or "").strip()
                    if dj:
                        docling_inflight[aid] = {
                            "job_id": dj,
                            "started_monotonic": time.monotonic(),
                            "status_errors": 0,
                        }
            loaded += 1
        return loaded

    scheduler_stop = threading.Event()

    def _bg_scheduler_loop() -> None:
        logger.info(
            "download_node: background Docling/Grader scheduler (tick every {}s); "
            "main thread only registers alignments after each collect",
            scheduler_tick_seconds,
        )
        while not scheduler_stop.is_set():
            try:
                _tick_scheduler()
            except Exception:
                logger.exception("Background scheduler tick failed")
            if scheduler_stop.wait(timeout=float(scheduler_tick_seconds)):
                break
        logger.info("download_node: background scheduler loop exited")

    bg_scheduler_thread = threading.Thread(
        target=_bg_scheduler_loop,
        name="lit-docling-grader-scheduler",
        daemon=True,
    )
    preloaded = _preload_scheduler_states()
    logger.info(
        "download_node: preloaded {} scheduler state(s) from {}",
        preloaded,
        scheduler_state_dir,
    )
    logger.info(
        "download_node: run logs dir={} (shared data logs={})",
        run_logs_dir,
        logs_base,
    )
    sys.stdout.flush()
    bg_scheduler_thread.start()

    for query_id, alignments in data.items():
        if not isinstance(alignments, list):
            continue
        for al in alignments:
            target = al.get("target") or ""
            alignment_id = f"{query_id}_{target}".replace("/", "_").replace(" ", "_")
            paper_ids_src = _alignment_paper_ids(al)
            if not paper_ids_src:
                logger.warning(f"Alignment {alignment_id}: no paper IDs")
                continue
            papers_dir = os.path.join(papers_base, alignment_id)
            os.makedirs(papers_dir, exist_ok=True)
            already_done = _outputs_done(alignment_id)
            if already_done:
                logger.info(
                    "Alignment {} already has graded+results outputs; skipping download/convert/grade",
                    alignment_id,
                )
                recs = []
                has_pdf = False
                n_docling_required = 0
                docling_required_basenames: List[str] = []
            else:
                existing_txt = [
                    fname
                    for fname in os.listdir(papers_dir)
                    if fname.endswith(".txt")
                    and os.path.isfile(os.path.join(papers_dir, fname))
                    and os.path.getsize(os.path.join(papers_dir, fname)) > 0
                ]
                existing_pdf_dir = os.path.join(papers_dir, "pdf")
                existing_pdf = os.path.isdir(existing_pdf_dir) and any(
                    fname.endswith(".pdf")
                    and os.path.isfile(os.path.join(existing_pdf_dir, fname))
                    and os.path.getsize(os.path.join(existing_pdf_dir, fname)) > 0
                    for fname in os.listdir(existing_pdf_dir)
                )
                manifest_path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
                manifest_map = _load_download_manifest(manifest_path)

                collect_skip = (
                    not no_cache
                    and not retry_failed
                    and is_alignment_download_complete(papers_dir)
                )
                recs: List[Any] = []
                plan_stats: Dict[str, int] = {}

                if collect_skip:
                    logger.info(
                        "Alignment {}: download-complete; skipping collect",
                        alignment_id,
                    )
                    _emit_download_progress_summary(
                        alignment_id,
                        paper_ids_src,
                        manifest_map,
                        papers_dir,
                        "before_collect",
                    )
                else:
                    plan = classify_alignment_papers(
                        paper_ids_src,
                        manifest_map,
                        papers_dir,
                        global_cache,
                        retry_failed=retry_failed,
                        no_cache=no_cache,
                    )
                    plan_stats = plan.stats
                    if plan.global_inject:
                        manifest_map = {**manifest_map, **plan.global_inject}
                        _write_download_manifest_atomic(manifest_path, manifest_map)
                    logger.info(
                        "Alignment {}: collect plan satisfied={} terminal_failed={} "
                        "global_skipped={} to_fetch={}",
                        alignment_id,
                        plan.stats.get("satisfied", 0),
                        plan.stats.get("terminal_failed", 0),
                        plan.stats.get("global_skipped", 0),
                        plan.stats.get("to_fetch", 0),
                    )
                    _emit_download_progress_summary(
                        alignment_id,
                        paper_ids_src,
                        manifest_map,
                        papers_dir,
                        "before_collect",
                        classification_stats=plan.stats,
                    )

                    if no_cache:
                        logger.info(
                            "Alignment {}: no_cache re-downloading all {} papers",
                            alignment_id,
                            len(paper_ids_src),
                        )
                        recs = download_papers_to_dir(
                            paper_ids_src,
                            papers_dir,
                            session=session,
                            pmcid_cache=pmcid_cache,
                            no_cache=True,
                            force_pdfs=False,
                            prefer_pdf_text=True,
                            delete_pdf_after_text=True,
                            collection_org=collection_org,
                            auth_scope=collection_auth_scope,
                            collector_email=collector_email or None,
                            max_workers=collect_max_workers,
                            disable_semantic_scholar=collect_disable_s2,
                        )
                        manifest_map = _merge_recs_into_manifest({}, recs)
                        _write_download_manifest_atomic(manifest_path, manifest_map)
                    elif plan.to_fetch:
                        recs = download_papers_to_dir(
                            plan.to_fetch,
                            papers_dir,
                            session=session,
                            pmcid_cache=pmcid_cache,
                            no_cache=no_cache,
                            force_pdfs=False,
                            prefer_pdf_text=True,
                            delete_pdf_after_text=True,
                            collection_org=collection_org,
                            auth_scope=collection_auth_scope,
                            collector_email=collector_email or None,
                            max_workers=collect_max_workers,
                            disable_semantic_scholar=collect_disable_s2,
                        )
                        manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                        _write_download_manifest_atomic(manifest_path, manifest_map)
                        record_global_outcomes_from_rows(
                            logs_base, manifest_map.values(), global_cache
                        )
                    else:
                        logger.info(
                            "Alignment {}: nothing to fetch (satisfied={} terminal_failed={} global_skipped={})",
                            alignment_id,
                            plan.stats.get("satisfied", 0),
                            plan.stats.get("terminal_failed", 0),
                            plan.stats.get("global_skipped", 0),
                        )

                    if not no_cache:
                        write_alignment_download_complete(
                            papers_dir,
                            {
                                "alignment_id": alignment_id,
                                "total_expected": len(paper_ids_src),
                                **plan.stats,
                                "fetched": len(recs),
                                "retry_failed_mode": retry_failed,
                            },
                        )

                _emit_download_progress_summary(
                    alignment_id,
                    paper_ids_src,
                    manifest_map,
                    papers_dir,
                    "after_collect",
                    classification_stats=plan_stats or None,
                )

                if recs or not already_done:
                    docling_required_basenames = _infer_docling_required_basenames(
                        papers_dir, manifest_map
                    )
                    n_docling_required = len(docling_required_basenames)
                    if not recs:
                        has_pdf = existing_pdf or bool(docling_required_basenames)
                    else:
                        has_pdf = any(r.pdf_path for r in recs) or existing_pdf

            query_meta = al.get("query_meta")
            target_meta = al.get("target_meta")
            gene_context: Dict[str, Any] | None = None
            if isinstance(query_meta, dict) or isinstance(target_meta, dict):
                gene_context = {"query": query_meta or {}, "target": target_meta or {}}
            elif idmap:
                key = f"{query_id}|{target}"
                meta = idmap.get(key)
                if meta:
                    gene_context = {
                        "query": meta.get("query_meta") or {},
                        "target": meta.get("target_meta") or {},
                    }

            eval_manifest_path = os.path.join(papers_dir, "docling_eval_manifest.jsonl")
            try:
                if recs:
                    # Manifest must list every basename in docling_required_basenames, not only
                    # rows from this collect. from_disk can add PDFs already on disk that still
                    # need Docling; omitting them leaves the Docling filter incomplete while the
                    # scheduler still waits on those .txt files (no GRADER_READY, no grader POST).
                    pdf_dir_m = os.path.join(papers_dir, "pdf")
                    seen_pdf_bases: set[str] = set()
                    with open(eval_manifest_path, "w", encoding="utf-8") as mf:
                        for rrec in recs:
                            mf.write(
                                json.dumps(
                                    {
                                        "paper_id": rrec.paper_id,
                                        "pdf_path": rrec.pdf_path,
                                        "details": rrec.details or {},
                                    }
                                )
                                + "\n"
                            )
                            pp = rrec.pdf_path
                            if pp:
                                seen_pdf_bases.add(
                                    os.path.splitext(os.path.basename(str(pp)))[0]
                                )
                        for base in docling_required_basenames:
                            if base in seen_pdf_bases:
                                continue
                            pdf_path = os.path.join(pdf_dir_m, f"{base}.pdf")
                            mf.write(
                                json.dumps(
                                    {
                                        "paper_id": base,
                                        "pdf_path": pdf_path,
                                        "details": {"pdf_docling_required": True},
                                    }
                                )
                                + "\n"
                            )
                elif docling_required_basenames:
                    # Resume path: recs is empty but disk still has PDFs needing Docling.
                    # Rewrite manifest so Docling's filter matches docling_required_basenames
                    # (stale JSONL with no pdf_docling_required rows used to break GRADER_READY).
                    pdf_dir_m = os.path.join(papers_dir, "pdf")
                    with open(eval_manifest_path, "w", encoding="utf-8") as mf:
                        for base in docling_required_basenames:
                            pdf_path = os.path.join(pdf_dir_m, f"{base}.pdf")
                            mf.write(
                                json.dumps(
                                    {
                                        "paper_id": base,
                                        "pdf_path": pdf_path,
                                        "details": {"pdf_docling_required": True},
                                    }
                                )
                                + "\n"
                            )
            except Exception as e:
                logger.warning(
                    f"Alignment {alignment_id}: could not write docling manifest: {e}"
                )
                eval_manifest_path = ""

            docling_payload: Dict[str, Any] = {
                "alignment_id": alignment_id,
                "pdf_dir": os.path.join(papers_dir, "pdf"),
                "papers_dir": papers_dir,
                "query": query_id,
                "target_id": target,
                "constraints": {"max_tokens": max_tokens, "temperature": temperature},
                "instructions": instructions_text
                or (
                    "Analyze the paper excerpt for relevance to the genes in this alignment. "
                    "First give a brief justification (2-4 sentences). "
                    "Then output a single line: relevance_score=<float between 0 and 1>."
                ),
                "output_root": output_root,
                "gene_context": gene_context,
                "analysis_host": gpu_host,
                "analysis_port": gpu_port,
                "evaluation_manifest_path": eval_manifest_path or None,
                "call_analysis": False,
            }
            grader_payload: Dict[str, Any] = {
                "alignment_id": alignment_id,
                "papers_dir": papers_dir,
                "query": query_id,
                "target_id": target,
                "constraints": {
                    "max_tokens": _grader_max_tokens_env(),
                    "temperature": temperature,
                },
                "instructions": instructions_text
                or (
                    "Grade each paper with the rubric. This pair was selected for structural "
                    "similarity; infection-naive host pathway overlap can still score positively."
                ),
                "output_root": output_root,
                "host_rubric_path": host_rubric_path,
                "microbe_rubric_path": microbe_rubric_path,
            }
            if gene_context is not None:
                grader_payload["gene_context"] = gene_context

            needs_docling = bool(docling_url_base and n_docling_required > 0 and has_pdf)
            default_state = STATE_DOCLING_PENDING if needs_docling else STATE_GRADER_READY
            state_obj = _bootstrap_state(
                alignment_id,
                {
                    "alignment_id": alignment_id,
                    "state": default_state,
                    "papers_dir": papers_dir,
                    "docling_required_basenames": docling_required_basenames,
                    "docling_payload": docling_payload,
                    "grader_payload": grader_payload,
                    "updated_at": time.time(),
                },
            )
            with scheduler_lock:
                alignment_states[alignment_id] = state_obj
                if state_obj.get("state") == STATE_GRADER_INFLIGHT:
                    gj = str(state_obj.get("grader_job_id") or "").strip()
                    gu = str(
                        state_obj.get("grader_url_base") or grader_url_bases[0]
                    ).strip()
                    if gj and gu:
                        grader_inflight[alignment_id] = {
                            "job_id": gj,
                            "grader_url_base": gu,
                            "started_monotonic": time.monotonic(),
                        }
                _write_state(alignment_id)

    while True:
        with scheduler_lock:
            terminal = {STATE_DONE, STATE_FAILED}
            non_terminal = [
                st for st in alignment_states.values()
                if st.get("state") not in terminal
            ]
            docling_n = len(docling_inflight)
            grader_n = len(grader_inflight)
            synthesis_n = len(synthesis_inflight)
        if not non_terminal and docling_n == 0 and grader_n == 0 and synthesis_n == 0:
            break
        time.sleep(scheduler_tick_seconds)

    scheduler_stop.set()
    join_timeout = max(600.0, 2.0 * float(stage_watchdog_seconds))
    bg_scheduler_thread.join(timeout=join_timeout)
    if bg_scheduler_thread.is_alive():
        logger.error(
            "Background scheduler thread still alive after {:.0f}s join; "
            "skipping final on-main drain to avoid racing a stuck tick",
            join_timeout,
        )
    else:
        try:
            with scheduler_lock:
                _tick_scheduler()
        except Exception:
            logger.exception("Final scheduler drain failed")

    logger.info(
        "Scheduler complete: done={} failed={} total={}",
        total_done,
        failed_count,
        len(alignment_states),
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "CPU download node: download papers, POST to Docling or GPU analysis node"
        )
    )
    p.add_argument("--paper-ids", required=True, help="Search output JSON path")
    p.add_argument("--data-root", required=True, help="Shared data root")
    p.add_argument("--gpu-host", default=os.environ.get("GPU_HOST", ""), help="GPU node hostname")
    p.add_argument("--gpu-port", type=int, default=9000)
    p.add_argument("--docling-host", default="", help="Docling node hostname")
    p.add_argument("--docling-port", type=int, default=9100)
    p.add_argument("--grader-host", default=os.environ.get("GRADER_HOST", ""))
    p.add_argument("--grader-port", type=int, default=int(os.environ.get("GRADER_API_PORT", "9200")))
    p.add_argument("--host-rubric-path", default=os.environ.get("HOST_RUBRIC_PATH", ""))
    p.add_argument("--microbe-rubric-path", default=os.environ.get("MICROBE_RUBRIC_PATH", ""))
    p.add_argument("--output-root", required=True, help="Results output root")
    p.add_argument("--instructions", default="", help="Inline prompt/instructions")
    p.add_argument("--instructions-file", default="", help="Path to prompt/instructions file")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    run(
        paper_ids_path=args.paper_ids,
        data_root=args.data_root,
        gpu_host=args.gpu_host,
        gpu_port=args.gpu_port,
        output_root=args.output_root,
        instructions=args.instructions,
        instructions_file=args.instructions_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        no_cache=args.no_cache,
        docling_host=args.docling_host,
        docling_port=args.docling_port,
        grader_host=args.grader_host,
        grader_port=args.grader_port,
        host_rubric_path=args.host_rubric_path,
        microbe_rubric_path=args.microbe_rubric_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
