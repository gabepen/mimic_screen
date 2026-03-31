import json
import os
import time
from typing import Any, Dict, List, Optional

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


def _call_llm(user_content: str, base_url: str, max_tokens: int, temperature: float) -> str:
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    root_url = base_url.rstrip("/")
    base_url = root_url
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat/completions"
    model_id = configured or _resolve_model_id(root_url)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code == 404 and configured:
        served_model = _resolve_model_id(root_url)
        if served_model != model_id:
            payload["model"] = served_model
            r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return str(msg.get("content") or "").strip()
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


def _parse_grade_output(raw: str, dims: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            dsc = obj.get("rubric_dimension_scores") or {}
            if not isinstance(dsc, dict):
                dsc = {}
            dim_scores: Dict[str, float] = {}
            for d in dims:
                dn = d["name"]
                dim_scores[dn] = _safe_float(dsc.get(dn), 0.0)
            grade = _safe_float(obj.get("relevance_grade"), sum(dim_scores.values()) / max(len(dim_scores), 1))
            return {
                "relevance_grade": grade,
                "rubric_dimension_scores": dim_scores,
                "rationale": str(obj.get("rationale") or "").strip(),
            }
    except Exception:
        pass
    # Fallback to empty parse.
    dim_scores = {d["name"]: 0.0 for d in dims}
    return {
        "relevance_grade": 0.0,
        "rubric_dimension_scores": dim_scores,
        "rationale": raw[:800],
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
    prompt = (
        "You are grading one research paper for relevance to a protein alignment.\n"
        "Return strict JSON only with keys:\n"
        "relevance_grade (0..1 float), rubric_dimension_scores (object dim->0..1), rationale (string).\n\n"
        f"Alignment={req.alignment_id}\n"
        f"Query={req.query}\n"
        f"Target={req.target_id}\n"
        f"Paper role={role or 'unknown'}\n"
        f"{term_block}\n"
        f"Rubric:\n{json.dumps(rubric, ensure_ascii=False)}\n"
        f"Dimensions:\n{dim_lines}\n\n"
        f"Paper excerpt:\n{text[:80000]}"
    )
    notes = ""
    raw = ""
    parsed = {
        "relevance_grade": 0.0,
        "rubric_dimension_scores": {d["name"]: 0.0 for d in dims},
        "rationale": "",
    }
    if llm_base_url and text.strip():
        max_tokens = (req.constraints and req.constraints.max_tokens) or 1024
        temperature = (
            (req.constraints and req.constraints.temperature)
            if req.constraints is not None
            else 0.0
        )
        if temperature is None:
            temperature = 0.0
        try:
            raw = _call_llm(prompt, llm_base_url, max_tokens=max_tokens, temperature=temperature)
            if raw:
                parsed = _parse_grade_output(raw, dims)
        except Exception as e:
            notes = str(e)
            logger.warning(f"Grader LLM call failed for {fname}: {e}")
    return GradedPaper(
        paper_id=fname,
        file_name=fname,
        paper_role=role,
        relevance_grade=_safe_float(parsed["relevance_grade"]),
        rubric_dimension_scores=parsed["rubric_dimension_scores"],
        rationale=parsed["rationale"],
        model_output=raw or None,
        notes=notes or None,
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


@app.post("/grade_alignment", response_model=RunAlignmentResponse)
def grade_alignment(req: GradeAlignmentRequest) -> RunAlignmentResponse:
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

    _ensure_dir(req.output_root)
    graded_path = os.path.join(req.output_root, f"{req.alignment_id}_graded.json")
    grading_meta: Dict[str, Any] = {
        "grader_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
        "host_rubric_path": req.host_rubric_path,
        "microbe_rubric_path": req.microbe_rubric_path,
        "graded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_papers": len(graded),
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GRADER_API_PORT", "9200"))
    uvicorn.run(app, host="0.0.0.0", port=port)
