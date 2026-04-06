import json
import os
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from loguru import logger

from auto_lit_search.analysis_packet import (
    GradeAlignmentRequest,
    GradedPaper,
    RunAlignmentGradedRequest,
    RunAlignmentResponse,
)

app = FastAPI(title="auto_lit_search Grader node")

TEXT_EXTENSIONS = (".txt",)
MAX_PAPER_CHARS = 120000
_MODEL_ID_CACHE: Dict[str, str] = {}
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}
_ASYNC_QUEUE: "deque[Tuple[str, GradeAlignmentRequest]]" = deque()
_ASYNC_LOCK = threading.Lock()
_ASYNC_WORKER_STARTED = False


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
        return v if v > 0 else default
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
        return v if v >= 1 else default
    except ValueError:
        return default


def _grader_http_read_timeout_sec() -> float:
    """Seconds to wait for vLLM response body per attempt (queue + generation)."""
    for key in ("VLLM_HTTP_READ_TIMEOUT", "VLLM_GRADER_TIMEOUT"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            try:
                v = float(str(raw).strip())
                if v > 0:
                    return v
            except ValueError:
                pass
    return 300.0


def _grader_http_timeout_tuple() -> Tuple[float, float]:
    connect = _env_positive_float("VLLM_HTTP_CONNECT_TIMEOUT", 30.0)
    read = _grader_http_read_timeout_sec()
    return (connect, read)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_paper_files(papers_dir: str) -> List[str]:
    files = sorted(
        f
        for f in os.listdir(papers_dir)
        if os.path.isfile(os.path.join(papers_dir, f))
        and (f.endswith(TEXT_EXTENSIONS) or not f.endswith((".pdf", ".xml")))
    )
    labeled = [
        f
        for f in files
        if "__query" in f.lower() or "__target" in f.lower()
    ]
    return labeled if labeled else files


def _read_text(path: str, max_chars: int = MAX_PAPER_CHARS) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n[truncated]"
    return s


def _load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"rubric file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid rubric JSON {path}: {e}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"rubric must be a JSON object: {path}")
    return data


def _rubric_dimensions(rubric: Dict[str, Any]) -> List[Dict[str, Any]]:
    dims = rubric.get("dimensions")
    if isinstance(dims, list) and dims:
        out: List[Dict[str, Any]] = []
        for d in dims:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "description": str(d.get("description") or "").strip(),
                    "weight": float(d.get("weight") or 1.0),
                }
            )
        if out:
            return out
    # Support rubric schema with `axes` (e.g. legionella_rubric.json).
    # Each axis becomes one dimension in `rubric_dimension_scores`.
    axes = rubric.get("axes")
    if isinstance(axes, list) and axes:
        out_axes: List[Dict[str, Any]] = []
        for a in axes:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id") or "").strip()
            if not aid:
                continue
            out_axes.append(
                {
                    "name": aid,
                    "description": str(a.get("description") or "").strip(),
                    # Axis-level weights are not encoded in a simple scalar in the rubric
                    # file, so keep 1.0 for now and let the model produce calibrated 0..1 values.
                    "weight": float(a.get("weight") or 1.0),
                }
            )
        if out_axes:
            return out_axes
    # Fallback schema: scores: {dim: description}
    scores = rubric.get("scores")
    if isinstance(scores, dict) and scores:
        return [
            {"name": str(k), "description": str(v), "weight": 1.0}
            for k, v in scores.items()
        ]
    return [{"name": "overall_relevance", "description": "overall evidence relevance", "weight": 1.0}]


def _resolve_model_id(base_url: str) -> str:
    cached = _MODEL_ID_CACHE.get(base_url)
    if cached:
        return cached
    models_url = f"{base_url.rstrip('/')}/v1/models"
    r = requests.get(models_url, timeout=30)
    r.raise_for_status()
    data = r.json()
    models = data.get("data") if isinstance(data, dict) else None
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and model.get("id"):
                model_id = str(model["id"]).strip()
                if model_id:
                    _MODEL_ID_CACHE[base_url] = model_id
                    return model_id
    raise RuntimeError(f"Could not resolve model id from {models_url}")


def _post_chat_completion(
    url: str,
    payload: Dict[str, Any],
    timeout: Tuple[float, float],
    root_url: str,
) -> requests.Response:
    r = requests.post(url, json=payload, timeout=timeout)
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    if r.status_code == 404 and configured:
        model_id = str(payload.get("model") or "")
        served_model = _resolve_model_id(root_url)
        if served_model != model_id:
            retry_payload = dict(payload)
            retry_payload["model"] = served_model
            r = requests.post(url, json=retry_payload, timeout=timeout)
    return r


def _call_llm(
    user_content: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    log_context: str = "",
) -> str:
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    root_url = base_url.rstrip("/")
    api_base = root_url
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"
    url = f"{api_base}/chat/completions"
    model_id = configured or _resolve_model_id(root_url)
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    timeout = _grader_http_timeout_tuple()
    max_attempts = _env_positive_int("VLLM_GRADER_HTTP_RETRIES", 3)
    backoff_base = _env_positive_float("VLLM_GRADER_RETRY_BACKOFF_SEC", 45.0)
    backoff_cap = _env_positive_float("VLLM_GRADER_RETRY_BACKOFF_CAP_SEC", 180.0)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            r = _post_chat_completion(url, payload, timeout, root_url)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict):
                    return str(msg.get("content") or "").strip()
            return ""
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt + 1 >= max_attempts:
                raise
            wait = min(backoff_base * (2**attempt), backoff_cap)
            ctx = log_context or "grader"
            logger.warning(
                "Grader LLM {} (attempt {}/{}); sleeping {:.1f}s before retry context={!r}",
                type(e).__name__,
                attempt + 1,
                max_attempts,
                wait,
                ctx,
            )
            time.sleep(wait)
    if last_exc:
        raise last_exc
    return ""


def _extract_paper_role(fname: str) -> Optional[str]:
    lower = fname.lower()
    if "__query" in lower:
        return "query"
    if "__target" in lower:
        return "target"
    return None


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(k)
    return out


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _gene_terms(meta: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    symbol = str(meta.get("gene_name") or "").strip() or fallback_id
    common_name = str(meta.get("common_name") or "").strip()
    syn_keys = [
        "synonyms",
        "gene_synonyms",
        "aliases",
        "alias",
        "name_synonyms",
    ]
    syns: List[str] = []
    for k in syn_keys:
        syns.extend(_as_list(meta.get(k)))
    syns = _dedupe_keep_order(syns)
    syns = [s for s in syns if s.lower() not in {symbol.lower(), common_name.lower()}]
    return {
        "symbol": symbol,
        "common_name": common_name or "none",
        "synonyms": syns,
    }


def _identification_terms_block(req: GradeAlignmentRequest) -> str:
    query_meta = (req.gene_context or {}).get("query") or {}
    target_meta = (req.gene_context or {}).get("target") or {}
    q = _gene_terms(query_meta, req.query)
    t = _gene_terms(target_meta, req.target_id)
    q_syn = ", ".join(q["synonyms"]) if q["synonyms"] else "none"
    t_syn = ", ".join(t["synonyms"]) if t["synonyms"] else "none"
    return (
        "Paper identification terms used in retrieval (prioritize symbol/common name; "
        "use synonyms as alternate mentions):\n"
        f"- Query gene ({req.query}): symbol={q['symbol']}; common_name={q['common_name']}; synonyms={q_syn}\n"
        f"- Target gene ({req.target_id}): symbol={t['symbol']}; common_name={t['common_name']}; synonyms={t_syn}\n"
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def _strip_markdown_json_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if len(lines) < 2:
        return t
    body = lines[1:]
    while body and body[-1].strip() == "```":
        body.pop()
    return "\n".join(body).strip()


def _try_parse_grade_json(raw: str, dims: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    dsc = obj.get("rubric_dimension_scores")
    if not isinstance(dsc, dict):
        return None
    rax = obj.get("rubric_axis_rationales")
    if not isinstance(rax, dict):
        return None
    dim_scores: Dict[str, float] = {}
    axis_rationales: Dict[str, str] = {}
    for d in dims:
        dn = d["name"]
        if dn not in dsc:
            return None
        if dn not in rax:
            return None
        dim_scores[dn] = _safe_float(dsc.get(dn), 0.0)
        rtext = str(rax.get(dn) or "").strip()
        if not rtext:
            return None
        axis_rationales[dn] = rtext
    grade = sum(dim_scores.values()) / max(len(dim_scores), 1)
    return {
        "relevance_grade": grade,
        "rubric_dimension_scores": dim_scores,
        "rubric_axis_rationales": axis_rationales,
        "rationale": str(obj.get("rationale") or "").strip(),
    }


def _parse_grade_output(raw: str, dims: List[Dict[str, Any]]) -> Dict[str, Any]:
    stripped = _strip_markdown_json_fence(raw)
    parsed = _try_parse_grade_json(stripped, dims)
    if parsed is not None:
        return parsed
    dim_scores = {d["name"]: 0.0 for d in dims}
    axr = {d["name"]: "" for d in dims}
    if stripped and dims:
        axr[dims[0]["name"]] = (
            f"[Grader JSON parse failed; raw output excerpt:] {stripped[:2000]}"
        )
    return {
        "relevance_grade": 0.0,
        "rubric_dimension_scores": dim_scores,
        "rubric_axis_rationales": axr,
        "rationale": stripped[:800],
    }


def _grade_single_paper(
    file_path: str,
    req: GradeAlignmentRequest,
    rubric: Dict[str, Any],
    llm_base_url: Optional[str],
) -> GradedPaper:
    fname = os.path.basename(file_path)
    role = _extract_paper_role(fname)
    text = _read_text(file_path)
    dims = _rubric_dimensions(rubric)
    term_block = _identification_terms_block(req)
    dim_lines = "\n".join(
        f"- {d['name']}: {d['description']} (weight={d['weight']})" for d in dims
    )
    gene_focus = (
        "the QUERY gene (pathogen / microbe-side rubric context)"
        if role == "query"
        else (
            "the TARGET gene (host-side rubric context)"
            if role == "target"
            else "the gene implied by this paper's role in the pair below (query vs target)"
        )
    )
    prompt = (
        "Grade using the RUBRIC JSON object below. If `grader_instructions` exists, read it first; "
        "then `system_context`, `evaluation_unit`, `scoring_scale`, and each `axis` with criteria.\n\n"
        "Alignment context (pair-level; the rubric file is per side):\n"
        f"- alignment_id={req.alignment_id}\n"
        f"- query_gene_id={req.query}\n"
        f"- target_gene_id={req.target_id}\n"
        f"- paper_role={role or 'unknown'} (query → microbe rubric; target → host rubric)\n"
        f"- gene_focus_for_this_paper: {gene_focus}\n"
        f"{term_block}\n"
        "OUTPUT (pipeline JSON schema; not part of the rubric file):\n"
        "Return strict JSON only with keys:\n"
        "rubric_dimension_scores (object: each axis id below → number 0..1),\n"
        "rubric_axis_rationales (object: SAME axis ids → string: cite evidence from the excerpt; "
        "explain how you applied that axis’s criteria; no empty strings),\n"
        "rationale (optional string: one-sentence cross-axis takeaway; may be \"\").\n"
        "Do not output relevance_grade; it will be computed as the mean of axis scores.\n\n"
        f"RUBRIC:\n{json.dumps(rubric, ensure_ascii=False)}\n"
        f"rubric_dimension_scores and rubric_axis_rationales must each include exactly these "
        f"axis ids:\n{dim_lines}\n\n"
        f"Paper excerpt:\n{text[:100000]}"
    )
    dim_names = ", ".join(d["name"] for d in dims)
    notes = ""
    raw = ""
    parsed: Dict[str, Any] = {
        "relevance_grade": 0.0,
        "rubric_dimension_scores": {d["name"]: 0.0 for d in dims},
        "rubric_axis_rationales": {d["name"]: "" for d in dims},
        "rationale": "",
    }
    if llm_base_url and text.strip():
        max_tokens = (req.constraints and req.constraints.max_tokens) or 3072
        temperature = (
            (req.constraints and req.constraints.temperature)
            if req.constraints is not None
            else 0.0
        )
        if temperature is None:
            temperature = 0.0
        graded_ok = False
        for attempt in range(2):
            retry_extra = ""
            if attempt:
                bad = _strip_markdown_json_fence(raw)[:1500]
                retry_extra = (
                    "\n\nYour previous reply was not usable (empty, prose, markdown fences, "
                    "or missing required JSON keys). Follow rubric.grader_instructions and the axes, "
                    "then respond with ONLY one JSON object—no other text.\n"
                    "Required keys: rubric_dimension_scores (numbers 0..1), "
                    "rubric_axis_rationales (strings; same keys), optional rationale (string). "
                    f"Axis ids: {dim_names}.\n"
                )
                if bad:
                    retry_extra += f"Invalid earlier reply (excerpt):\n{bad}\n"
            try:
                raw = _call_llm(
                    prompt + retry_extra,
                    llm_base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    log_context=fname,
                )
            except Exception as e:
                notes = str(e)
                logger.warning(f"Grader LLM call failed for {fname}: {e}")
                break
            candidate = _strip_markdown_json_fence(raw)
            maybe = _try_parse_grade_json(candidate, dims)
            if maybe is not None:
                parsed = maybe
                graded_ok = True
                break
            if attempt == 0:
                reason = (
                    "empty model content"
                    if not candidate.strip()
                    else "invalid JSON or missing rubric_dimension_scores / rubric_axis_rationales"
                )
                logger.warning(
                    f"Grader LLM ({fname}): {reason}; resubmitting prompt once with stricter instruction"
                )
        if not graded_ok and raw:
            if not notes:
                logger.warning(
                    f"Grader LLM ({fname}): invalid JSON after retry; "
                    "using default scores and rationale excerpt"
                )
            parsed = _parse_grade_output(raw, dims)
    rdim = parsed["rubric_dimension_scores"]
    grade = sum(float(rdim[k]) for k in rdim) / max(len(rdim), 1)
    return GradedPaper(
        paper_id=fname,
        file_name=fname,
        paper_role=role,
        relevance_grade=_safe_float(grade),
        rubric_dimension_scores=parsed["rubric_dimension_scores"],
        rubric_axis_rationales=parsed.get("rubric_axis_rationales")
        or {d["name"]: "" for d in dims},
        rationale=parsed.get("rationale") or "",
        model_output=raw or None,
        notes=notes or None,
    )


def _grade_alignment_sync(req: GradeAlignmentRequest) -> RunAlignmentResponse:
    if not os.path.isdir(req.papers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"papers_dir does not exist or is not a directory: {req.papers_dir}",
        )
    files = _list_paper_files(req.papers_dir)
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"no files found in papers_dir: {req.papers_dir}",
        )
    host_rubric = _load_json_file(req.host_rubric_path)
    microbe_rubric = _load_json_file(req.microbe_rubric_path)
    llm_base_url = os.environ.get("VLLM_BASE_URL")
    graded: List[GradedPaper] = []
    for fname in files:
        role = _extract_paper_role(fname)
        rubric = microbe_rubric if role == "query" else host_rubric
        graded.append(
            _grade_single_paper(
                file_path=os.path.join(req.papers_dir, fname),
                req=req,
                rubric=rubric,
                llm_base_url=llm_base_url,
            )
        )

    def _parse_fallback_paper(g: GradedPaper) -> bool:
        rax = g.rubric_axis_rationales or {}
        return any(
            str(v).strip().startswith("[Grader JSON parse failed")
            for v in rax.values()
        )

    llm_enabled = bool(llm_base_url and str(llm_base_url).strip())
    n_llm_exceptions = sum(1 for g in graded if (g.notes or "").strip())
    n_without_model_output = sum(1 for g in graded if not (g.model_output or "").strip())
    n_json_parse_fallback = sum(1 for g in graded if _parse_fallback_paper(g))
    n_llm_ok_structured = sum(
        1
        for g in graded
        if (g.model_output or "").strip()
        and not (g.notes or "").strip()
        and not _parse_fallback_paper(g)
    )

    _ensure_dir(req.output_root)
    graded_path = os.path.join(req.output_root, f"{req.alignment_id}_graded.json")
    grading_meta: Dict[str, Any] = {
        "grader_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
        "host_rubric_path": req.host_rubric_path,
        "microbe_rubric_path": req.microbe_rubric_path,
        "graded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_papers": len(graded),
        "llm_enabled": llm_enabled,
        "n_llm_exceptions": n_llm_exceptions,
        "n_without_model_output": n_without_model_output,
        "n_json_parse_fallback": n_json_parse_fallback,
        "n_llm_ok_structured": n_llm_ok_structured,
    }
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "alignment_id": req.alignment_id,
                "graded_papers": [g.dict() for g in graded],
                "grading_meta": grading_meta,
            },
            f,
            indent=2,
        )
    logger.info(
        f"Grader wrote {len(graded)} graded papers for {req.alignment_id} -> {graded_path}"
    )
    logger.info(
        f"Grader summary {req.alignment_id}: n_papers={len(graded)} llm_enabled={llm_enabled} "
        f"n_llm_ok_structured={n_llm_ok_structured} n_llm_exceptions={n_llm_exceptions} "
        f"n_json_parse_fallback={n_json_parse_fallback} "
        f"n_without_model_output={n_without_model_output}"
    )

    synth_payload = RunAlignmentGradedRequest(
        alignment_id=req.alignment_id,
        papers_dir=req.papers_dir,
        query=req.query,
        target_id=req.target_id,
        constraints=req.constraints,
        instructions=req.instructions,
        output_root=req.output_root,
        gene_context=req.gene_context,
        graded_papers=graded,
        grading_meta=grading_meta,
    )
    synthesis_url = f"http://{req.synthesis_host}:{req.synthesis_port}/run_alignment_graded"
    try:
        r = requests.post(synthesis_url, json=synth_payload.dict(), timeout=600)
        r.raise_for_status()
        out = r.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"synthesis request failed to {synthesis_url}: {e}",
        ) from e
    return RunAlignmentResponse(
        status=str(out.get("status") or "ok"),
        alignment_id=req.alignment_id,
        results_path=str(out.get("results_path") or ""),
    )


def _async_queue_max_size() -> int:
    return _env_positive_int("GRADER_ASYNC_MAX_QUEUE", 1)


def _async_poll_interval_sec() -> float:
    return _env_positive_float("GRADER_ASYNC_POLL_INTERVAL_SEC", 2.0)


def _async_worker_loop() -> None:
    while True:
        with _ASYNC_LOCK:
            item = _ASYNC_QUEUE.popleft() if _ASYNC_QUEUE else None
        if item is None:
            time.sleep(0.2)
            continue
        job_id, req = item
        with _ASYNC_LOCK:
            job = _ASYNC_JOBS.get(job_id) or {}
            job["status"] = "running"
            job["started_at"] = time.time()
            _ASYNC_JOBS[job_id] = job
        try:
            out = _grade_alignment_sync(req)
            with _ASYNC_LOCK:
                job = _ASYNC_JOBS.get(job_id) or {}
                job["status"] = "succeeded"
                job["finished_at"] = time.time()
                job["result"] = out.dict()
                _ASYNC_JOBS[job_id] = job
        except Exception as e:
            with _ASYNC_LOCK:
                job = _ASYNC_JOBS.get(job_id) or {}
                job["status"] = "failed"
                job["finished_at"] = time.time()
                job["error"] = str(e)
                _ASYNC_JOBS[job_id] = job


def _ensure_async_worker_started() -> None:
    global _ASYNC_WORKER_STARTED
    with _ASYNC_LOCK:
        if _ASYNC_WORKER_STARTED:
            return
        t = threading.Thread(target=_async_worker_loop, daemon=True)
        t.start()
        _ASYNC_WORKER_STARTED = True


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


@app.post("/grade_alignment", response_model=RunAlignmentResponse)
def grade_alignment(req: GradeAlignmentRequest) -> RunAlignmentResponse:
    return _grade_alignment_sync(req)


@app.get("/grader_capacity")
def grader_capacity() -> Dict[str, Any]:
    _ensure_async_worker_started()
    with _ASYNC_LOCK:
        queue_depth = len(_ASYNC_QUEUE)
        running = sum(1 for v in _ASYNC_JOBS.values() if v.get("status") == "running")
        max_queue = _async_queue_max_size()
    return {
        "status": "ok",
        "can_accept": queue_depth < max_queue,
        "queue_depth": queue_depth,
        "max_queue": max_queue,
        "running_jobs": running,
    }


@app.post("/grade_alignment_async")
def grade_alignment_async(req: GradeAlignmentRequest) -> Dict[str, Any]:
    _ensure_async_worker_started()
    max_queue = _async_queue_max_size()
    with _ASYNC_LOCK:
        if len(_ASYNC_QUEUE) >= max_queue:
            raise HTTPException(
                status_code=429,
                detail=f"grader queue full (queue_depth={len(_ASYNC_QUEUE)} max_queue={max_queue})",
            )
        job_id = uuid.uuid4().hex
        _ASYNC_JOBS[job_id] = {
            "job_id": job_id,
            "alignment_id": req.alignment_id,
            "status": "queued",
            "submitted_at": time.time(),
        }
        _ASYNC_QUEUE.append((job_id, req))
    return {
        "job_id": job_id,
        "alignment_id": req.alignment_id,
        "status": "queued",
        "poll_interval_sec": _async_poll_interval_sec(),
    }


@app.get("/grade_alignment_status/{job_id}")
def grade_alignment_status(job_id: str) -> Dict[str, Any]:
    with _ASYNC_LOCK:
        job = _ASYNC_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
        out = dict(job)
    return out


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GRADER_API_PORT", "9200"))
    uvicorn.run(app, host="0.0.0.0", port=port)
