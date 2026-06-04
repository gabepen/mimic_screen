"""
CPU node script: read search output, download papers per alignment, POST each batch to GPU.
Run inside the lit-download container with env: DATA_ROOT, PAPER_IDS_PATH, GPU_HOST, GPU_API_PORT, OUTPUT_ROOT.
"""

import csv
import json
import os
import re
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from auto_lit_search.collect import _extract_doi_from_identifier, download_papers_to_dir
from auto_lit_search.slurm_utils import (
    get_job_node,
    get_job_state,
    is_terminal_job_state,
)

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)

DOWNLOAD_MANIFEST_FILENAME = "download_manifest.jsonl"
_DEFAULT_GRADER_MAX_TOKENS = 8192


def _grader_max_tokens_env() -> int:
    raw = os.environ.get("GRADER_MAX_TOKENS", str(_DEFAULT_GRADER_MAX_TOKENS))
    try:
        return max(1, int(str(raw).strip() or _DEFAULT_GRADER_MAX_TOKENS))
    except (TypeError, ValueError):
        return _DEFAULT_GRADER_MAX_TOKENS


def _refresh_grader_payload_constraints(payload: Dict[str, Any] | None) -> None:
    """Apply current GRADER_MAX_TOKENS; persisted scheduler state must not pin 4096."""
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


def _only_download_progress_log(record: Dict[str, Any]) -> bool:
    return record["extra"].get("download_progress") is True


def _paper_pair_key(paper_id: str, source: str) -> Tuple[str, str]:
    return (str(paper_id).strip(), str(source).strip())


def _load_download_manifest(manifest_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not manifest_path or not os.path.isfile(manifest_path):
        return out
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                pid = str(rec.get("paper_id") or "").strip()
                src = str(rec.get("source") or "").strip()
                if not pid:
                    continue
                out[_paper_pair_key(pid, src)] = rec
    except Exception:
        return out
    return out


def _manifest_file_stem(row: Dict[str, Any]) -> str:
    stem = str(row.get("file_stem") or "").strip()
    if stem:
        return stem
    tp = str(row.get("text_path") or "").strip()
    if tp:
        return os.path.splitext(os.path.basename(tp))[0]
    pp = str(row.get("pdf_path") or "").strip()
    if pp:
        return os.path.splitext(os.path.basename(pp))[0]
    return ""


def _manifest_row_satisfied(row: Dict[str, Any], papers_dir: str) -> bool:
    st = str(row.get("status") or "").strip().lower()
    if st == "failed":
        return False
    if st == "skipped":
        return True
    stem = _manifest_file_stem(row)
    if not stem:
        return False
    txt_path = os.path.join(papers_dir, f"{stem}.txt")
    if os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0:
        return True
    if st == "partial":
        pdf_path = os.path.join(papers_dir, "pdf", f"{stem}.pdf")
        return os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0
    return False


def _download_record_to_manifest_row(rec: Any, updated_at: float) -> Dict[str, Any]:
    d = getattr(rec, "details", None) or {}
    doi = d.get("doi") or _extract_doi_from_identifier(getattr(rec, "paper_id", "") or "")
    stem = str(d.get("file_stem") or "").strip()
    if not stem:
        tp = getattr(rec, "text_path", None) or ""
        if tp:
            stem = os.path.splitext(os.path.basename(str(tp)))[0]
    if not stem:
        pp = getattr(rec, "pdf_path", None) or ""
        if pp:
            stem = os.path.splitext(os.path.basename(str(pp)))[0]
    return {
        "paper_id": getattr(rec, "paper_id", "") or "",
        "source": getattr(rec, "source", "") or "",
        "doi": doi or "",
        "file_stem": stem,
        "status": getattr(rec, "status", "") or "",
        "selected_text_source": str(d.get("selected_text_source") or ""),
        "pdf_docling_required": bool(d.get("pdf_docling_required")),
        "text_path": getattr(rec, "text_path", None) or "",
        "pdf_path": getattr(rec, "pdf_path", None) or "",
        "message": getattr(rec, "message", None) or "",
        "updated_at": updated_at,
    }


def _merge_recs_into_manifest(
    existing: Dict[Tuple[str, str], Dict[str, Any]], recs: List[Any]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    merged = dict(existing)
    ts = time.time()
    for rec in recs:
        key = _paper_pair_key(getattr(rec, "paper_id", ""), getattr(rec, "source", ""))
        merged[key] = _download_record_to_manifest_row(rec, ts)
    return merged


def _write_download_manifest_atomic(
    manifest_path: str, rows_by_key: Dict[Tuple[str, str], Dict[str, Any]]
) -> None:
    d = os.path.dirname(manifest_path) or "."
    os.makedirs(d, exist_ok=True)
    keys = sorted(rows_by_key.keys(), key=lambda k: (k[0], k[1]))
    fd, tmp = tempfile.mkstemp(
        prefix=".download_manifest_", suffix=".tmp", dir=d, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wf:
            for k in keys:
                wf.write(json.dumps(rows_by_key[k], ensure_ascii=False) + "\n")
        os.replace(tmp, manifest_path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _emit_download_progress_summary(
    alignment_id: str,
    expected: List[Tuple[str, str]],
    manifest_map: Dict[Tuple[str, str], Dict[str, Any]],
    papers_dir: str,
    phase: str,
) -> List[Tuple[str, str]]:
    total = len(expected)
    satisfied = 0
    missing: List[Tuple[str, str]] = []
    for pid, src in expected:
        key = _paper_pair_key(pid, src)
        row = manifest_map.get(key)
        if row and _manifest_row_satisfied(row, papers_dir):
            satisfied += 1
        else:
            missing.append((pid, src))
    cap = 20
    parts: List[str] = []
    for pid, src in missing[:cap]:
        doi = _extract_doi_from_identifier(pid) or pid
        parts.append(f"{doi}:{src}")
    tail = ""
    if len(missing) > cap:
        tail = f" …(+{len(missing) - cap} more)"
    preview = ",".join(parts) + tail
    logger.bind(download_progress=True).info(
        "alignment_download_summary alignment_id={} phase={} total_expected={} "
        "satisfied={} missing_count={} missing_preview=[{}]",
        alignment_id,
        phase,
        total,
        satisfied,
        len(missing),
        preview,
    )
    return missing


def _load_search_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _alignment_paper_ids(alignment: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return a flat list of (paper_id, source) for this alignment.

    Source is either "query" or "target"; for lp-human controls we also
    carry richer gene metadata alongside the alignment, but this helper
    stays focused on IDs + coarse role.
    """
    out: List[Tuple[str, str]] = []
    for pid in alignment.get("query_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "query"))
    for pid in alignment.get("target_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "target"))
    return out


def _load_idmap(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load mapping CSV with columns like:
        query,target,
        query_entrez_id,target_entrez_id,
        query_gene_name,target_gene_name,
        query_locus_tag,target_locus_tag,
        query_genbank_acc,target_genbank_acc,
        query_common_name,target_common_name
    and return a dict keyed by "query|target" with query/target metadata
    shaped for the GPU gene_context helper.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            target = (row.get("target") or "").strip()
            if not query or not target:
                continue

            def _meta(prefix: str) -> Dict[str, Any]:
                syn_raw = (
                    row.get(f"{prefix}_synonyms")
                    or row.get(f"{prefix}_gene_synonyms")
                    or row.get(f"{prefix}_aliases")
                    or ""
                )
                syns = [s.strip() for s in str(syn_raw).split(",") if s.strip()]
                return {
                    "uniprot_id": (row.get(prefix) or "").strip(),
                    "entrez_id": (row.get(f"{prefix}_entrez_id") or "").strip(),
                    "gene_name": (row.get(f"{prefix}_gene_name") or "").strip(),
                    "locus_tag": (row.get(f"{prefix}_locus_tag") or "").strip(),
                    "genbank_acc": (row.get(f"{prefix}_genbank_acc") or "").strip(),
                    "common_name": (row.get(f"{prefix}_common_name") or "").strip(),
                    "synonyms": syns,
                }

            key = f"{query}|{target}"
            out[key] = {
                "query_meta": _meta("query"),
                "target_meta": _meta("target"),
            }
    return out


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
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _wait_health(
    service: str,
    gpu_url_base: str,
    timeout: int = 300,
    interval: int = 5,
) -> bool:
    url = f"{gpu_url_base.rstrip('/')}/healthz"
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
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
        url,
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
    return False


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
    gpu_url_base = f"http://{gpu_host}:{gpu_port}"
    logger.info(
        "download_node: probing LLM GPU health at {}/healthz",
        gpu_url_base.rstrip("/"),
    )
    sys.stdout.flush()
    if not _wait_health(
        "LLM GPU", gpu_url_base, timeout=service_health_wait_seconds
    ):
        raise RuntimeError(f"GPU node not healthy at {gpu_url_base}")

    docling_url_base = ""
    if docling_host:
        docling_url_base = f"http://{docling_host}:{docling_port}"
        logger.info(
            "download_node: probing Docling health at {}/healthz",
            docling_url_base.rstrip("/"),
        )
        sys.stdout.flush()
        if not _wait_health(
            "Docling", docling_url_base, timeout=service_health_wait_seconds
        ):
            raise RuntimeError(f"Docling node not healthy at {docling_url_base}")
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
    scheduler_lock = threading.Lock()

    def _drop_pending_specs_for_registered_port(port: int) -> None:
        grader_pending_specs[:] = [
            s for s in grader_pending_specs if int(s["port"]) != port
        ]

    def _register_grader_url(url: str, job_id: str = "") -> bool:
        base = url.rstrip("/")
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
        deadline = time.monotonic() + service_health_wait_seconds
        while time.monotonic() < deadline:
            _refresh_grader_endpoints()
            if any(_grader_health_ok(u) for u in grader_url_bases):
                break
            time.sleep(5)
        else:
            raise RuntimeError(
                "No grader endpoint became healthy within "
                f"{service_health_wait_seconds}s (GRADER_JOB_IDS discovery)"
            )
        for grader_url_base in grader_url_bases:
            if not _grader_health_ok(grader_url_base):
                logger.info(
                    "download_node: waiting for Grader health at {}",
                    grader_url_base.rstrip("/"),
                )
                if not _wait_health(
                    "Grader",
                    grader_url_base,
                    timeout=min(120, service_health_wait_seconds),
                    interval=5,
                ):
                    logger.warning(
                        "Grader at {} not healthy yet; will retry via discovery",
                        grader_url_base,
                    )
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

    if grader_discovery_enabled:
        _refresh_grader_endpoints()
    logger.info(
        "download_node: {} grader endpoint(s) active: {}",
        len(grader_url_bases),
        ", ".join(grader_url_bases),
    )
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
    os.makedirs(papers_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)
    scheduler_state_dir = os.path.join(logs_base, "scheduler_state")
    os.makedirs(scheduler_state_dir, exist_ok=True)

    if os.environ.get("DOWNLOAD_PROGRESS_LOG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        _p_path = (
            os.environ.get("DOWNLOAD_PROGRESS_LOG_PATH", "").strip()
            or os.path.join(logs_base, "download_progress.log")
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

    def _outputs_done(alignment_id: str) -> bool:
        # Grader writes *_graded.json and synthesis (GPU node) writes *_results.json.
        graded_path = os.path.join(output_root, f"{alignment_id}_graded.json")
        results_path = os.path.join(output_root, f"{alignment_id}_results.json")
        return os.path.isfile(graded_path) and os.path.isfile(results_path)

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
                if not _outputs_done(aid):
                    continue
                prev = str(st.get("state") or "")
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

    def _required_docling_txt_done(st: Dict[str, Any]) -> bool:
        req = st.get("docling_required_basenames") or []
        if not req:
            return True
        papers_dir = str(st.get("papers_dir") or "")
        for base in req:
            txt_path = os.path.join(papers_dir, f"{base}.txt")
            if not (os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0):
                return False
        return True

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
        if _outputs_done(st["alignment_id"]):
            return False
        papers_dir = str(st.get("papers_dir") or "")
        if not _papers_dir_has_nonempty_txt(papers_dir):
            return False
        return _required_docling_txt_done(st)

    def _infer_docling_required_basenames_from_disk(papers_dir: str) -> List[str]:
        """Best-effort resume helper when a prior run already downloaded artifacts."""
        pdf_dir = os.path.join(papers_dir, "pdf")
        if not os.path.isdir(pdf_dir):
            return []
        txt_basenames = {
            os.path.splitext(_canonical_alignment_text_key(fname))[0]
            for fname in os.listdir(papers_dir)
            if fname.endswith(".txt")
            and os.path.isfile(os.path.join(papers_dir, fname))
            and os.path.getsize(os.path.join(papers_dir, fname)) > 0
        }
        required: List[str] = []
        for fname in os.listdir(pdf_dir):
            if not fname.endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdf_dir, fname)
            if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) <= 0:
                continue
            base = os.path.splitext(fname)[0]
            if base not in txt_basenames:
                required.append(base)
        return sorted(required)

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
        if _outputs_done(alignment_id):
            state["state"] = STATE_DONE
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
                    st["state"] = STATE_FAILED
                    st["last_error"] = str(status.get("error") or "docling failed")
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
                    st["state"] = STATE_FAILED
                    st["last_error"] = "grader watchdog timeout"
                    failed_count += 1
                    grader_inflight.pop(aid, None)
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
                else:
                    st["state"] = STATE_FAILED
                    st["last_error"] = "grader succeeded but results file missing"
                    failed_count += 1
                _write_state(aid)

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
        """Pick a grader endpoint with no inflight alignment and remote queue capacity."""
        with scheduler_lock:
            candidates = [
                u
                for u in grader_url_bases
                if _grader_inflight_count_for_url(u) < 1
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda u: (_grader_inflight_count_for_url(u), u))
        for url in candidates:
            if _wait_service_capacity(
                session=scheduler_session,
                base_url=url,
                endpoint="grader_capacity",
                service_name="Grader",
                timeout_seconds=1,
                poll_interval_seconds=1,
                warn_on_timeout=False,
            ):
                return url
        return None

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
                with scheduler_lock:
                    st = alignment_states.get(pending_aid)
                    if st and st.get("state") == STATE_GRADER_READY:
                        st["state"] = STATE_FAILED
                        st["last_error"] = f"grader submit failed: {e}"
                        _write_state(pending_aid)
                return

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

    _discovery_status_tick = 0

    def _tick_scheduler() -> None:
        nonlocal _discovery_status_tick
        _reconcile_outputs_into_state()
        n_registered_before = len(grader_url_bases)
        _refresh_grader_endpoints()
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
                can_resume_from_disk = bool(existing_txt or existing_pdf) and not no_cache
                manifest_path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
                manifest_map = _load_download_manifest(manifest_path)
                _emit_download_progress_summary(
                    alignment_id,
                    paper_ids_src,
                    manifest_map,
                    papers_dir,
                    "before_collect",
                )
                missing_pre: List[Tuple[str, str]] = []
                for pid, src in paper_ids_src:
                    key = _paper_pair_key(pid, src)
                    row = manifest_map.get(key)
                    if row is None or not _manifest_row_satisfied(row, papers_dir):
                        missing_pre.append((pid, src))

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
                        force_pdfs=True,
                        prefer_pdf_text=True,
                        collection_org=collection_org,
                        auth_scope=collection_auth_scope,
                        collector_email=collector_email or None,
                        max_workers=collect_max_workers,
                        disable_semantic_scholar=collect_disable_s2,
                    )
                    manifest_map = _merge_recs_into_manifest({}, recs)
                    _write_download_manifest_atomic(manifest_path, manifest_map)
                elif missing_pre:
                    logger.info(
                        "Alignment {}: collecting {} missing papers ({} already satisfied)",
                        alignment_id,
                        len(missing_pre),
                        len(paper_ids_src) - len(missing_pre),
                    )
                    recs = download_papers_to_dir(
                        missing_pre,
                        papers_dir,
                        session=session,
                        pmcid_cache=pmcid_cache,
                        no_cache=no_cache,
                        force_pdfs=True,
                        prefer_pdf_text=True,
                        collection_org=collection_org,
                        auth_scope=collection_auth_scope,
                        collector_email=collector_email or None,
                        max_workers=collect_max_workers,
                        disable_semantic_scholar=collect_disable_s2,
                    )
                    manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                    _write_download_manifest_atomic(manifest_path, manifest_map)
                elif can_resume_from_disk:
                    docling_required_basenames = _infer_docling_required_basenames_from_disk(
                        papers_dir
                    )
                    n_docling_required = len(docling_required_basenames)
                    has_pdf = existing_pdf
                    recs = []
                    logger.info(
                        "Alignment {} reusing existing artifacts (txt={} pending_docling={}): download manifest complete",
                        alignment_id,
                        len(existing_txt),
                        n_docling_required,
                    )
                else:
                    logger.info(
                        "Downloading {} papers for {}",
                        len(paper_ids_src),
                        alignment_id,
                    )
                    recs = download_papers_to_dir(
                        paper_ids_src,
                        papers_dir,
                        session=session,
                        pmcid_cache=pmcid_cache,
                        no_cache=no_cache,
                        force_pdfs=True,
                        prefer_pdf_text=True,
                        collection_org=collection_org,
                        auth_scope=collection_auth_scope,
                        collector_email=collector_email or None,
                        max_workers=collect_max_workers,
                        disable_semantic_scholar=collect_disable_s2,
                    )
                    manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                    _write_download_manifest_atomic(manifest_path, manifest_map)

                if recs:
                    has_pdf = any(r.pdf_path for r in recs) or existing_pdf
                    from_recs = {
                        os.path.splitext(os.path.basename(str(r.pdf_path)))[0]
                        for r in recs
                        if ((r.details or {}).get("pdf_docling_required")) and r.pdf_path
                    }
                    from_disk = set(
                        _infer_docling_required_basenames_from_disk(papers_dir)
                    )
                    docling_required_basenames = sorted(from_recs | from_disk)
                    n_docling_required = len(docling_required_basenames)
                elif not already_done:
                    docling_required_basenames = _infer_docling_required_basenames_from_disk(
                        papers_dir
                    )
                    n_docling_required = len(docling_required_basenames)
                    has_pdf = existing_pdf

                _emit_download_progress_summary(
                    alignment_id,
                    paper_ids_src,
                    manifest_map,
                    papers_dir,
                    "after_collect",
                )
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
                "synthesis_host": gpu_host,
                "synthesis_port": gpu_port,
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
        if not non_terminal and docling_n == 0 and grader_n == 0:
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
    p.add_argument("--gpu-host", required=True, help="GPU node hostname")
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
