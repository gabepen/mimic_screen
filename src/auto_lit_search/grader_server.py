import csv
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from loguru import logger

from auto_lit_search.analysis_packet import (
    GradeAlignmentRequest,
    GradedPaper,
    RunAlignmentGradedRequest,
    RunAlignmentResponse,
)
from auto_lit_search.env_config import env_flag
from auto_lit_search.paper_io import (
    DOWNLOAD_MANIFEST_FILENAME,
    MAX_PAPER_CHARS,
    ensure_dir as _ensure_dir,
    extract_paper_role as _extract_paper_role,
    identification_terms_block,
    list_paper_files as _list_paper_files,
    paper_id_by_artifact_basename as _paper_id_by_artifact_basename,
    read_text as _read_text,
)
from auto_lit_search.rubric_scoring import (
    GRADING_SCHEMA_VERSION,
    aggregate_paper_scores,
    criteria_prompt_block,
    normalize_criterion_scores,
    required_flag_ids,
    required_scored_criterion_ids,
    rubric_role_for_paper_role,
)
from auto_lit_search.scheduler_http import post_run_alignment_graded

app = FastAPI(title="auto_lit_search Grader node")
_MODEL_ID_CACHE: Dict[str, str] = {}
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}
_ASYNC_QUEUE: "deque[Tuple[str, GradeAlignmentRequest]]" = deque()
_ASYNC_LOCK = threading.Lock()
_ASYNC_WORKER_STARTED = False
_GRADER_LLM_JSONL_LOCK = threading.Lock()
_MODEL_ID_CACHE_LOCK = threading.Lock()
_DEFAULT_GRADER_MAX_TOKENS = 4096
_DEFAULT_GRADER_PAPER_WORKERS = 3
_GRADER_CRITERION_NOTE_MAX_CHARS = 60
_GRADER_CLAIM_SUMMARY_MAX_CHARS = 150


def _grader_paper_workers() -> int:
    raw = _env_positive_int("GRADER_PAPER_WORKERS", _DEFAULT_GRADER_PAPER_WORKERS)
    return max(1, min(16, raw))


def _grader_max_tokens(req: GradeAlignmentRequest) -> int:
    # Grader GPU env is authoritative; CPU scheduler constraints are refreshed on submit.
    del req
    return _env_positive_int("GRADER_MAX_TOKENS", _DEFAULT_GRADER_MAX_TOKENS)


def _grader_json_output_instructions(rubric: Dict[str, Any], rubric_role: str) -> str:
    scored_ids = required_scored_criterion_ids(rubric)
    flag_ids = required_flag_ids(rubric)
    scored_list = ", ".join(scored_ids)
    flag_line = (
        f"rubric_tags (object: include {', '.join(flag_ids)} when applicable; "
        "values from rubric flag_values)\n"
        if flag_ids
        else ""
    )
    host_meta = (
        "infection_naive (boolean),\n"
        if rubric_role == "host"
        else ""
    )
    return (
        "OUTPUT (pipeline JSON schema v2; not part of the rubric file):\n"
        "Return strict JSON only with keys:\n"
        "criterion_scores (object: each scored criterion id → "
        f"{{score: 0|1|2, note: optional string ≤{_GRADER_CRITERION_NOTE_MAX_CHARS} chars}}),\n"
        "mention_type (string: focal_study | supporting_evidence | incidental_mention | "
        "negative_result | methodological_reference),\n"
        "no_meaningful_mention (boolean),\n"
        f"{host_meta}"
        f"{flag_line}"
        f"claim_summary (optional string ≤{_GRADER_CLAIM_SUMMARY_MAX_CHARS} chars).\n"
        "Required criterion_scores ids:\n"
        f"{scored_list}\n"
        "Do not output rubric_dimension_scores, axis totals, paper_grade, or relevance_grade; "
        "the server computes weighted axis totals and readable grades from criterion_scores.\n"
    )


def _clamp_parsed_metadata(parsed: Dict[str, Any]) -> None:
    cs = parsed.get("criterion_scores")
    if isinstance(cs, dict):
        for crit_id, entry in cs.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("note"):
                entry["note"] = str(entry["note"]).strip()[:_GRADER_CRITERION_NOTE_MAX_CHARS]
    if parsed.get("claim_summary"):
        parsed["claim_summary"] = str(parsed["claim_summary"]).strip()[
            :_GRADER_CLAIM_SUMMARY_MAX_CHARS
        ]


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
    with _MODEL_ID_CACHE_LOCK:
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
                    with _MODEL_ID_CACHE_LOCK:
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
    if r.status_code == 400 and "response_format" in payload:
        # Some vLLM/OpenAI-compatible servers reject response_format. Retry once.
        fallback_payload = dict(payload)
        fallback_payload.pop("response_format", None)
        r = requests.post(url, json=fallback_payload, timeout=timeout)
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    if r.status_code == 404 and configured:
        model_id = str(payload.get("model") or "")
        served_model = _resolve_model_id(root_url)
        if served_model != model_id:
            retry_payload = dict(payload)
            retry_payload["model"] = served_model
            r = requests.post(url, json=retry_payload, timeout=timeout)
    return r


@dataclass
class GraderLLMCallResult:
    content: str
    prompt_char_len: int
    requested_model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    n_choices: int = 0
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    http_status: Optional[int] = None


def _append_grader_llm_jsonl(output_root: str, alignment_id: str, record: Dict[str, Any]) -> None:
    if not output_root or not str(output_root).strip():
        return
    logs_dir = os.path.join(output_root, "logs")
    path = os.path.join(logs_dir, f"{alignment_id}_grader_llm.jsonl")
    row = dict(record)
    row.setdefault("ts_iso", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _GRADER_LLM_JSONL_LOCK:
        _ensure_dir(logs_dir)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _call_llm(
    user_content: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    log_context: str = "",
    *,
    emit_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_static: Optional[Dict[str, Any]] = None,
) -> GraderLLMCallResult:
    prompt_char_len = len(user_content)
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
        "response_format": {"type": "json_object"},
    }
    timeout = _grader_http_timeout_tuple()
    max_attempts = _env_positive_int("VLLM_GRADER_HTTP_RETRIES", 3)
    backoff_base = _env_positive_float("VLLM_GRADER_RETRY_BACKOFF_SEC", 45.0)
    backoff_cap = _env_positive_float("VLLM_GRADER_RETRY_BACKOFF_CAP_SEC", 180.0)

    def _emit_http_event(
        *,
        data: Dict[str, Any],
        http_status: int,
        content: str,
        http_attempt: int,
    ) -> None:
        if emit_event is None:
            return
        choices = data.get("choices") or []
        n_ch = len(choices) if isinstance(choices, list) else 0
        fr: Optional[str] = None
        if n_ch and isinstance(choices[0], dict):
            fr = str(choices[0].get("finish_reason") or "") or None
        ev: Dict[str, Any] = {
            **(log_static or {}),
            "log_context": log_context or None,
            "phase": "chat_completion",
            "prompt_char_len": prompt_char_len,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "requested_model": model_id,
            "response_model": data.get("model"),
            "response_id": data.get("id"),
            "response_created": data.get("created"),
            "system_fingerprint": data.get("system_fingerprint"),
            "finish_reason": fr,
            "usage": data.get("usage"),
            "n_choices": n_ch,
            "http_status": http_status,
            "http_attempt_index": http_attempt,
            "content_char_len": len(content),
            "content_empty": not content.strip(),
        }
        emit_event(ev)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            r = _post_chat_completion(url, payload, timeout, root_url)
            status = int(r.status_code)
            try:
                data = r.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                if emit_event is not None:
                    emit_event(
                        {
                            **(log_static or {}),
                            "log_context": log_context or None,
                            "phase": "chat_completion",
                            "prompt_char_len": prompt_char_len,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "requested_model": model_id,
                            "http_status": status,
                            "http_attempt_index": attempt,
                            "error": str(e),
                            "response_text_excerpt": (r.text or "")[:2000],
                        }
                    )
                raise
            choices = data.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                if isinstance(msg, dict):
                    content = str(msg.get("content") or "").strip()
            _emit_http_event(
                data=data, http_status=status, content=content, http_attempt=attempt
            )
            return GraderLLMCallResult(
                content=content,
                prompt_char_len=prompt_char_len,
                requested_model=model_id,
                finish_reason=(
                    str(choices[0].get("finish_reason") or "").strip() or None
                    if choices and isinstance(choices[0], dict)
                    else None
                ),
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else None,
                n_choices=len(choices) if isinstance(choices, list) else 0,
                response_id=str(data.get("id") or "").strip() or None,
                response_model=str(data.get("model") or "").strip() or None,
                http_status=status,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if emit_event is not None:
                emit_event(
                    {
                        **(log_static or {}),
                        "log_context": log_context or None,
                        "phase": "chat_completion",
                        "prompt_char_len": prompt_char_len,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "requested_model": model_id,
                        "http_attempt_index": attempt,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "will_retry": attempt + 1 < max_attempts,
                    }
                )
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
    return GraderLLMCallResult(
        content="",
        prompt_char_len=prompt_char_len,
        requested_model=model_id,
        http_status=None,
    )


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


def _build_grade_repair_prompt(
    bad_output: str,
    criterion_ids: str,
) -> str:
    excerpt = _strip_markdown_json_fence(bad_output)[:4000]
    return (
        "Your previous response was non-empty but not valid for the required schema.\n"
        "Do NOT re-grade the paper and do NOT add new evidence.\n"
        "Only reformat/recover the previous response into strict JSON.\n\n"
        "Return JSON only with keys:\n"
        "- criterion_scores (object: criterion id -> {score: 0|1|2, note: optional string})\n"
        "- mention_type, no_meaningful_mention, rubric_tags, claim_summary (as applicable)\n\n"
        f"Scored criterion ids must be exactly: {criterion_ids}\n\n"
        "Invalid output to repair:\n"
        f"{excerpt}\n"
    )


def _extract_first_json_object(raw: str) -> Optional[str]:
    """Extract the first balanced JSON object from mixed model text."""
    s = raw.strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _try_parse_grade_json(
    raw: str,
    rubric: Dict[str, Any],
    rubric_role: str,
) -> Optional[Dict[str, Any]]:
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        wrapped = _extract_first_json_object(s)
        if not wrapped:
            return None
        try:
            obj = json.loads(wrapped)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    raw_scores = obj.get("criterion_scores")
    if not isinstance(raw_scores, dict):
        return None
    required = required_scored_criterion_ids(rubric)
    criterion_scores = normalize_criterion_scores(raw_scores)
    for crit_id in required:
        if crit_id not in criterion_scores:
            return None

    no_mention = _parse_bool_field(obj.get("no_meaningful_mention"))
    if no_mention:
        criterion_scores = {
            crit_id: {"score": 0, "note": ""} for crit_id in required
        }

    tags_raw = obj.get("rubric_tags")
    rubric_tags: Dict[str, str] = {}
    if isinstance(tags_raw, dict):
        for k, v in tags_raw.items():
            sv = str(v or "").strip()
            if sv:
                rubric_tags[str(k)] = sv

    agg = aggregate_paper_scores(
        rubric, criterion_scores, rubric_role=rubric_role
    )
    parsed: Dict[str, Any] = {
        **agg,
        "criterion_scores": criterion_scores,
        "mention_type": str(obj.get("mention_type") or "").strip() or None,
        "infection_naive": (
            _parse_bool_field(obj["infection_naive"])
            if "infection_naive" in obj
            else None
        ),
        "no_meaningful_mention": no_mention,
        "claim_summary": str(obj.get("claim_summary") or "").strip(),
        "rubric_tags": rubric_tags,
        "rationale": str(obj.get("claim_summary") or obj.get("rationale") or "").strip(),
    }
    _clamp_parsed_metadata(parsed)
    return parsed


def _parse_grade_output(
    raw: str,
    rubric: Dict[str, Any],
    rubric_role: str,
) -> Dict[str, Any]:
    stripped = _strip_markdown_json_fence(raw)
    parsed = _try_parse_grade_json(stripped, rubric, rubric_role)
    if parsed is not None:
        return parsed
    required = required_scored_criterion_ids(rubric)
    criterion_scores = {crit_id: {"score": 0, "note": ""} for crit_id in required}
    agg = aggregate_paper_scores(
        rubric, criterion_scores, rubric_role=rubric_role
    )
    return {
        **agg,
        "criterion_scores": criterion_scores,
        "mention_type": None,
        "infection_naive": None,
        "no_meaningful_mention": False,
        "claim_summary": "",
        "rubric_tags": {},
        "rationale": (
            f"[Grader JSON parse failed; raw output excerpt:] {stripped[:2000]}"
            if stripped
            else ""
        ),
        "notes_parse_failed": True,
    }


def _grade_single_paper(
    file_path: str,
    req: GradeAlignmentRequest,
    rubric: Dict[str, Any],
    llm_base_url: Optional[str],
    paper_id_by_file: Optional[Dict[str, str]] = None,
) -> Tuple[GradedPaper, Dict[str, int]]:
    fname = os.path.basename(file_path)
    lookup = paper_id_by_file or {}
    paper_id = lookup.get(fname) or os.path.splitext(fname)[0]
    role = _extract_paper_role(fname)
    rubric_role = rubric_role_for_paper_role(role or "")
    text = _read_text(file_path)
    criterion_block = criteria_prompt_block(rubric)
    term_block = identification_terms_block(req.query, req.target_id, req.gene_context)
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
        f"{_grader_json_output_instructions(rubric, rubric_role)}\n"
        f"Scored criteria by axis:\n{criterion_block}\n\n"
        f"RUBRIC:\n{json.dumps(rubric, ensure_ascii=False)}\n\n"
        f"Paper excerpt:\n{text[:100000]}"
    )
    criterion_ids = ", ".join(required_scored_criterion_ids(rubric))
    notes = ""
    raw = ""
    required = required_scored_criterion_ids(rubric)
    zero_scores = {crit_id: {"score": 0, "note": ""} for crit_id in required}
    parsed: Dict[str, Any] = {
        **aggregate_paper_scores(rubric, zero_scores, rubric_role=rubric_role),
        "criterion_scores": zero_scores,
        "mention_type": None,
        "infection_naive": None,
        "no_meaningful_mention": False,
        "claim_summary": "",
        "rationale": "",
        "rubric_tags": {},
    }
    repair_attempted = 0
    repair_succeeded = 0
    regrade_retry_used = 0

    def _emit_llm_row(ev: Dict[str, Any]) -> None:
        row = dict(ev)
        row.setdefault("alignment_id", req.alignment_id)
        row.setdefault("file_name", fname)
        _append_grader_llm_jsonl(req.output_root, req.alignment_id, row)

    excerpt_char_len = len(text)
    excerpt_in_prompt_chars = min(100000, excerpt_char_len)

    if not llm_base_url or not str(llm_base_url).strip():
        _emit_llm_row(
            {
                "call_kind": "skipped_no_vllm_url",
                "excerpt_char_len": excerpt_char_len,
                "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
            }
        )
    elif not text.strip():
        _emit_llm_row(
            {
                "call_kind": "skipped_no_excerpt",
                "excerpt_char_len": excerpt_char_len,
                "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
            }
        )
    if llm_base_url and text.strip():
        max_tokens = _grader_max_tokens(req)
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
                    "Required keys: criterion_scores (each scored criterion id -> "
                    f"{{score: 0|1|2, note: optional ≤{_GRADER_CRITERION_NOTE_MAX_CHARS} chars}}), "
                    "mention_type, no_meaningful_mention. "
                    f"Criterion ids: {criterion_ids}.\n"
                )
                if bad:
                    retry_extra += f"Invalid earlier reply (excerpt):\n{bad}\n"
            try:
                llm_res = _call_llm(
                    prompt + retry_extra,
                    llm_base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    log_context=fname,
                    emit_event=_emit_llm_row,
                    log_static={
                        "call_kind": "grade",
                        "grade_attempt": attempt,
                        "excerpt_char_len": excerpt_char_len,
                        "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
                    },
                )
                raw = llm_res.content
            except Exception as e:
                notes = str(e)
                logger.warning(f"Grader LLM call failed for {fname}: {e}")
                if not isinstance(e, requests.exceptions.RequestException):
                    _emit_llm_row(
                        {
                            "call_kind": "grade",
                            "grade_attempt": attempt,
                            "phase": "client_exception",
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "prompt_char_len": len(prompt + retry_extra),
                            "excerpt_char_len": excerpt_char_len,
                            "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
                        }
                    )
                break
            candidate = _strip_markdown_json_fence(raw)
            maybe = _try_parse_grade_json(candidate, rubric, rubric_role)
            if maybe is not None:
                parsed = maybe
                graded_ok = True
                break
            if attempt == 0:
                if not candidate.strip():
                    regrade_retry_used += 1
                    logger.warning(
                        f"Grader LLM ({fname}): empty model content; "
                        "resubmitting full grading prompt once"
                    )
                    continue
                repair_attempted += 1
                logger.warning(
                    f"Grader LLM ({fname}): non-empty invalid JSON; attempting repair pass "
                    "before full regrade retry"
                )
                repair_prompt = _build_grade_repair_prompt(raw, criterion_ids)
                try:
                    repair_llm = _call_llm(
                        repair_prompt,
                        llm_base_url,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        log_context=f"{fname}::repair",
                        emit_event=_emit_llm_row,
                        log_static={
                            "call_kind": "repair",
                            "grade_attempt": attempt,
                            "excerpt_char_len": excerpt_char_len,
                            "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
                        },
                    )
                    repaired_raw = repair_llm.content
                except Exception as e:
                    logger.warning(f"Grader repair LLM call failed for {fname}: {e}")
                    if not isinstance(e, requests.exceptions.RequestException):
                        _emit_llm_row(
                            {
                                "call_kind": "repair",
                                "grade_attempt": attempt,
                                "phase": "client_exception",
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "prompt_char_len": len(repair_prompt),
                                "excerpt_char_len": excerpt_char_len,
                                "excerpt_in_prompt_chars": excerpt_in_prompt_chars,
                            }
                        )
                    repaired_raw = ""
                repaired_candidate = _strip_markdown_json_fence(repaired_raw)
                repaired = _try_parse_grade_json(repaired_candidate, rubric, rubric_role)
                if repaired is not None:
                    parsed = repaired
                    graded_ok = True
                    repair_succeeded += 1
                    raw = repaired_raw
                    break
                regrade_retry_used += 1
                logger.warning(
                    f"Grader LLM ({fname}): repair pass still invalid; "
                    "resubmitting full grading prompt once with stricter instruction"
                )
        if not graded_ok and raw:
            if not notes:
                logger.warning(
                    f"Grader LLM ({fname}): invalid JSON after retry; "
                    "using default scores and rationale excerpt"
                )
            parsed = _parse_grade_output(raw, rubric, rubric_role)
    if parsed.pop("notes_parse_failed", False) and not notes:
        notes = "grader JSON parse failed"
    paper = GradedPaper(
        paper_id=paper_id,
        file_name=fname,
        paper_role=role,
        grading_schema_version=int(
            parsed.get("grading_schema_version") or GRADING_SCHEMA_VERSION
        ),
        relevance_grade=_safe_float(parsed.get("relevance_grade", 0.0)),
        relevance_sort=int(parsed.get("relevance_sort") or 0),
        paper_grade=str(parsed.get("paper_grade") or ""),
        primary_grade=str(parsed.get("primary_grade") or ""),
        criterion_scores=parsed.get("criterion_scores") or {},
        axis_totals=parsed.get("axis_totals") or {},
        rubric_dimension_scores=parsed.get("rubric_dimension_scores") or {},
        rubric_axis_rationales=parsed.get("rubric_axis_rationales") or {},
        mention_type=parsed.get("mention_type"),
        infection_naive=parsed.get("infection_naive"),
        no_meaningful_mention=bool(parsed.get("no_meaningful_mention")),
        claim_summary=str(parsed.get("claim_summary") or ""),
        rationale=parsed.get("rationale") or "",
        rubric_tags=parsed.get("rubric_tags") or {},
        model_output=raw or None,
        notes=notes or None,
    )
    return paper, {
        "repair_attempted": repair_attempted,
        "repair_succeeded": repair_succeeded,
        "regrade_retry_used": regrade_retry_used,
    }


def _write_rubric_score_csvs(
    alignment_id: str,
    output_root: str,
    graded: List[GradedPaper],
    host_rubric: Dict[str, Any],
    microbe_rubric: Dict[str, Any],
) -> Tuple[str, str]:
    """One CSV per rubric side: rows=papers, columns=axis scores."""
    _ensure_dir(output_root)
    host_path = os.path.join(output_root, f"{alignment_id}_host_rubric_scores.csv")
    microbe_path = os.path.join(output_root, f"{alignment_id}_microbe_rubric_scores.csv")
    host_axes = [d["name"] for d in _rubric_dimensions(host_rubric)]
    microbe_axes = [d["name"] for d in _rubric_dimensions(microbe_rubric)]
    base_cols = [
        "paper_id",
        "file_name",
        "paper_grade",
        "primary_grade",
        "relevance_sort",
        "relevance_grade",
    ]

    def _write(path: str, papers: List[GradedPaper], axis_cols: List[str]) -> None:
        axis_total_cols = [f"{ax}_total" for ax in axis_cols]
        fieldnames = base_cols + axis_total_cols + axis_cols + ["claim_summary"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for gp in papers:
                row: Dict[str, Any] = {
                    "paper_id": gp.paper_id,
                    "file_name": gp.file_name,
                    "paper_grade": gp.paper_grade,
                    "primary_grade": gp.primary_grade,
                    "relevance_sort": gp.relevance_sort,
                    "relevance_grade": gp.relevance_grade,
                    "claim_summary": (gp.claim_summary or gp.rationale or "")[:2000],
                }
                scores = gp.rubric_dimension_scores or {}
                totals = gp.axis_totals or {}
                for ax in axis_cols:
                    total = totals.get(ax) or {}
                    row[f"{ax}_total"] = total.get("label", "")
                    row[ax] = scores.get(ax, "")
                w.writerow(row)

    host_papers = [g for g in graded if g.paper_role == "target"]
    microbe_papers = [g for g in graded if g.paper_role == "query"]
    _write(host_path, host_papers, host_axes)
    _write(microbe_path, microbe_papers, microbe_axes)
    return host_path, microbe_path


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
    paper_id_by_file = _paper_id_by_artifact_basename(req.papers_dir)
    paper_workers = _grader_paper_workers()
    logger.info(
        "Grader {}: grading {} papers with {} parallel workers",
        req.alignment_id,
        len(files),
        paper_workers,
    )

    def _grade_one(fname: str) -> Tuple[str, GradedPaper, Dict[str, int]]:
        role = _extract_paper_role(fname)
        rubric = microbe_rubric if role == "query" else host_rubric
        gp, retry_meta = _grade_single_paper(
            file_path=os.path.join(req.papers_dir, fname),
            req=req,
            rubric=rubric,
            llm_base_url=llm_base_url,
            paper_id_by_file=paper_id_by_file,
        )
        return fname, gp, retry_meta

    graded_by_file: Dict[str, GradedPaper] = {}
    n_repair_attempted = 0
    n_repair_succeeded = 0
    n_regrade_retry_used = 0
    if paper_workers <= 1:
        for fname in files:
            _, gp, retry_meta = _grade_one(fname)
            graded_by_file[fname] = gp
            n_repair_attempted += int(retry_meta.get("repair_attempted", 0))
            n_repair_succeeded += int(retry_meta.get("repair_succeeded", 0))
            n_regrade_retry_used += int(retry_meta.get("regrade_retry_used", 0))
    else:
        with ThreadPoolExecutor(max_workers=paper_workers) as pool:
            futures = {pool.submit(_grade_one, fname): fname for fname in files}
            for fut in as_completed(futures):
                fname = futures[fut]
                try:
                    fname_out, gp, retry_meta = fut.result()
                except Exception as e:
                    logger.exception(
                        "Grader parallel task failed for {} paper {}: {}",
                        req.alignment_id,
                        fname,
                        e,
                    )
                    raise
                graded_by_file[fname_out] = gp
                n_repair_attempted += int(retry_meta.get("repair_attempted", 0))
                n_repair_succeeded += int(retry_meta.get("repair_succeeded", 0))
                n_regrade_retry_used += int(retry_meta.get("regrade_retry_used", 0))
    graded = [graded_by_file[fname] for fname in files]

    def _parse_fallback_paper(g: GradedPaper) -> bool:
        rationale = str(g.rationale or "").strip()
        return rationale.startswith("[Grader JSON parse failed")

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
        "grading_schema_version": GRADING_SCHEMA_VERSION,
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
        "n_repair_attempted": n_repair_attempted,
        "n_repair_succeeded": n_repair_succeeded,
        "n_regrade_retry_used": n_regrade_retry_used,
        "grader_paper_workers": paper_workers,
        "grader_llm_jsonl": os.path.join("logs", f"{req.alignment_id}_grader_llm.jsonl"),
    }
    host_csv, microbe_csv = _write_rubric_score_csvs(
        req.alignment_id,
        req.output_root,
        graded,
        host_rubric,
        microbe_rubric,
    )
    grading_meta["host_rubric_scores_csv"] = os.path.basename(host_csv)
    grading_meta["microbe_rubric_scores_csv"] = os.path.basename(microbe_csv)
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "alignment_id": req.alignment_id,
                "query": req.query,
                "target_id": req.target_id,
                "papers_dir": req.papers_dir,
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
        f"n_without_model_output={n_without_model_output} "
        f"n_repair_attempted={n_repair_attempted} n_repair_succeeded={n_repair_succeeded} "
        f"n_regrade_retry_used={n_regrade_retry_used}"
    )

    skip_synthesis = env_flag("GRADER_SKIP_SYNTHESIS", True)
    if skip_synthesis or not (req.synthesis_host or "").strip():
        return RunAlignmentResponse(
            status="ok",
            alignment_id=req.alignment_id,
            results_path="",
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
    synthesis_url = f"http://{req.synthesis_host}:{req.synthesis_port}"
    try:
        out = post_run_alignment_graded(synthesis_url, synth_payload.dict())
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
def healthz() -> Dict[str, Any]:
    """Ready only when the local vLLM server responds on /v1/models."""
    base = (os.environ.get("VLLM_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return {"status": "ok", "vllm": "disabled", "detail": "VLLM_BASE_URL unset"}
    try:
        resp = requests.get(f"{base}/v1/models", timeout=5)
        if resp.ok:
            data = resp.json()
            if isinstance(data, dict) and data.get("data"):
                return {"status": "ok", "vllm": "ready", "detail": "vllm models available"}
        return {
            "status": "degraded",
            "vllm": "unreachable",
            "detail": (resp.text or "")[:300],
        }
    except Exception as e:
        return {"status": "degraded", "vllm": "unreachable", "detail": str(e)}


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
