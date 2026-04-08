"""
CPU node script: read search output, download papers per alignment, POST each batch to GPU.
Run inside the lit-download container with env: DATA_ROOT, PAPER_IDS_PATH, GPU_HOST, GPU_API_PORT, OUTPUT_ROOT.
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from loguru import logger

from auto_lit_search.collect import download_papers_to_dir

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


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
        except requests.exceptions.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is None or resp.status_code != 429:
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


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
    if not grader_host:
        raise RuntimeError("GRADER_HOST is required for two-stage analysis flow")
    if not host_rubric_path or not microbe_rubric_path:
        raise RuntimeError(
            "HOST_RUBRIC_PATH and MICROBE_RUBRIC_PATH are required for grading"
        )
    grader_url_base = f"http://{grader_host}:{grader_port}"
    logger.info(
        "download_node: probing Grader health at {}/healthz",
        grader_url_base.rstrip("/"),
    )
    if not _wait_health(
        "Grader", grader_url_base, timeout=service_health_wait_seconds
    ):
        raise RuntimeError(f"Grader node not healthy at {grader_url_base}")

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

    session = requests.Session()
    pmcid_cache: Dict[str, str | None] = {}
    total_done = 0
    failed_count = 0
    docling_inflight_cap = max(1, int(os.environ.get("DOCLING_INFLIGHT_CAP", "1")))
    grader_inflight_cap = max(1, int(os.environ.get("GRADER_INFLIGHT_CAP", "1")))
    scheduler_tick_seconds = max(
        1, int(os.environ.get("SCHEDULER_TICK_SECONDS", "5"))
    )
    stage_watchdog_seconds = max(
        300, int(os.environ.get("STAGE_WATCHDOG_SECONDS", str(request_timeout)))
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

    def _is_grader_ready(st: Dict[str, Any]) -> bool:
        if _outputs_done(st["alignment_id"]):
            return False
        papers_dir = str(st.get("papers_dir") or "")
        has_text = any(
            fname.endswith(".txt")
            for fname in os.listdir(papers_dir)
            if os.path.isfile(os.path.join(papers_dir, fname))
        )
        return has_text and _required_docling_txt_done(st)

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
        if _outputs_done(alignment_id):
            state["state"] = STATE_DONE
        elif _is_grader_ready(state):
            state["state"] = STATE_GRADER_READY
        elif (state.get("docling_required_basenames") or []) and not _required_docling_txt_done(state):
            state["state"] = STATE_DOCLING_PENDING
        return state

    def _poll_inflight() -> None:
        nonlocal total_done, failed_count
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
            try:
                status = _docling_status_once(session, docling_url_base, meta["job_id"])
            except Exception:
                continue
            s = str(status.get("status") or "").strip().lower()
            if s not in {"succeeded", "failed"}:
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
                st["state"] = STATE_FAILED
                st["last_error"] = "docling succeeded but package not grader-ready"
                failed_count += 1
            _write_state(aid)

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
            try:
                status = _grader_status_once(session, grader_url_base, meta["job_id"])
            except Exception:
                continue
            s = str(status.get("status") or "").strip().lower()
            if s not in {"succeeded", "failed"}:
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
        if len(docling_inflight) >= docling_inflight_cap:
            return
        if not _wait_service_capacity(
            session=session,
            base_url=docling_url_base,
            endpoint="docling_capacity",
            service_name="Docling",
            timeout_seconds=1,
            poll_interval_seconds=1,
            warn_on_timeout=False,
        ):
            return
        for aid, st in alignment_states.items():
            if st.get("state") != STATE_DOCLING_PENDING:
                continue
            try:
                job_id = _submit_docling_async(
                    session=session,
                    docling_url_base=docling_url_base,
                    payload=st["docling_payload"],
                    submit_timeout=30,
                )
                st["state"] = STATE_DOCLING_INFLIGHT
                st["docling_job_id"] = job_id
                st["docling_submitted_at"] = time.time()
                docling_inflight[aid] = {
                    "job_id": job_id,
                    "started_monotonic": time.monotonic(),
                }
                _write_state(aid)
            except Exception as e:
                st["state"] = STATE_FAILED
                st["last_error"] = f"docling submit failed: {e}"
                _write_state(aid)
            break

    def _dispatch_grader() -> None:
        if len(grader_inflight) >= grader_inflight_cap:
            return
        if not _wait_service_capacity(
            session=session,
            base_url=grader_url_base,
            endpoint="grader_capacity",
            service_name="Grader",
            timeout_seconds=1,
            poll_interval_seconds=1,
            warn_on_timeout=False,
        ):
            return
        for aid, st in alignment_states.items():
            if st.get("state") != STATE_GRADER_READY:
                continue
            try:
                job_id = _submit_grader_async(
                    session=session,
                    grader_url_base=grader_url_base,
                    payload=st["grader_payload"],
                    submit_timeout=30,
                )
                st["state"] = STATE_GRADER_INFLIGHT
                st["grader_job_id"] = job_id
                st["grader_submitted_at"] = time.time()
                grader_inflight[aid] = {
                    "job_id": job_id,
                    "started_monotonic": time.monotonic(),
                }
                _write_state(aid)
            except Exception as e:
                st["state"] = STATE_FAILED
                st["last_error"] = f"grader submit failed: {e}"
                _write_state(aid)
            break

    def _tick_scheduler() -> None:
        _poll_inflight()
        _dispatch_docling()
        _dispatch_grader()

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
            logger.info(f"Downloading {len(paper_ids_src)} papers for {alignment_id}")
            recs = download_papers_to_dir(
                paper_ids_src,
                papers_dir,
                session=session,
                pmcid_cache=pmcid_cache,
                no_cache=no_cache,
                force_pdfs=True,
                prefer_pdf_text=False,
                collection_org=collection_org,
                auth_scope=collection_auth_scope,
                collector_email=collector_email or None,
                max_workers=collect_max_workers,
                disable_semantic_scholar=collect_disable_s2,
            )
            has_pdf = any(r.pdf_path for r in recs)
            n_docling_required = sum(
                1 for r in recs if ((r.details or {}).get("pdf_docling_required"))
            )
            docling_required_basenames = sorted(
                {
                    os.path.splitext(os.path.basename(str(r.pdf_path)))[0]
                    for r in recs
                    if ((r.details or {}).get("pdf_docling_required")) and r.pdf_path
                }
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
                "constraints": {"max_tokens": max_tokens, "temperature": temperature},
                "instructions": instructions_text
                or (
                    "Analyze the paper excerpt for relevance to the genes in this alignment. "
                    "First give a brief justification (2-4 sentences). "
                    "Then output a single line: relevance_score=<float between 0 and 1>."
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
            alignment_states[alignment_id] = state_obj
            _write_state(alignment_id)
            _tick_scheduler()

    while True:
        _tick_scheduler()
        terminal = {STATE_DONE, STATE_FAILED}
        non_terminal = [
            st for st in alignment_states.values() if st.get("state") not in terminal
        ]
        if not non_terminal and not docling_inflight and not grader_inflight:
            break
        time.sleep(scheduler_tick_seconds)

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
