import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger

from auto_lit_search.analysis_packet import (
    Constraints,
    RunAlignmentGradedRequest,
    RunAlignmentRequest,
    RunAlignmentResponse,
)
from auto_lit_search.env_config import env_flag, env_positive_float
from auto_lit_search.paper_io import (
    ensure_dir,
    identification_terms_block,
    list_paper_files,
    read_text,
)
from auto_lit_search import synthesis_graded

app = FastAPI(title="auto_lit_search GPU node")

MAX_PAPER_CHARS = 120000
_MODEL_ID_CACHE: Dict[str, str] = {}


def _call_llm(
    user_content: str,
    base_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> str:
    import requests

    def _fetch_served_model_id(root: str) -> str:
        cached = _MODEL_ID_CACHE.get(root)
        if cached:
            return cached
        models_url = f"{root}/v1/models"
        mr = requests.get(models_url, timeout=30)
        mr.raise_for_status()
        mdata = mr.json()
        entries = mdata.get("data") if isinstance(mdata, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    mid = str(entry["id"]).strip()
                    if mid:
                        _MODEL_ID_CACHE[root] = mid
                        return mid
        raise RuntimeError(f"Could not resolve model id from {models_url}")

    root_url = base_url.rstrip("/")
    base_url = root_url
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat/completions"
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    if not model:
        model = configured or _fetch_served_model_id(root_url)
    timeout = env_positive_float("SYNTHESIS_LLM_TIMEOUT", 300.0)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code == 404 and configured:
        served_model = _fetch_served_model_id(root_url)
        if served_model != model:
            payload["model"] = served_model
            r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return (msg.get("content") or "").strip()
    return ""


def _write_alignment_results(
    req: RunAlignmentRequest,
    payload: Dict[str, Any],
) -> RunAlignmentResponse:
    ensure_dir(req.output_root)
    result_path = os.path.join(req.output_root, f"{req.alignment_id}_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "GPU node wrote results for {} -> {}",
        req.alignment_id,
        result_path,
    )
    return RunAlignmentResponse(
        status="ok",
        alignment_id=req.alignment_id,
        results_path=result_path,
    )


def _run_alignment_impl(req: RunAlignmentRequest) -> RunAlignmentResponse:
    if not env_flag("GPU_ENABLE_LEGACY_RUN_ALIGNMENT", False):
        raise HTTPException(
            status_code=410,
            detail=(
                "POST /run_alignment is disabled (set GPU_ENABLE_LEGACY_RUN_ALIGNMENT=1). "
                "Production uses grade -> /run_alignment_graded."
            ),
        )
    if not os.path.isdir(req.papers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"papers_dir does not exist or is not a directory: {req.papers_dir}",
        )

    ensure_dir(req.output_root)
    log_path = os.path.join(req.output_root, "logs", f"{req.alignment_id}.log")

    files = list_paper_files(req.papers_dir)
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"no files found in papers_dir: {req.papers_dir}",
        )

    llm_base_url = os.environ.get("VLLM_BASE_URL")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    term_block = identification_terms_block(req.query, req.target_id, req.gene_context)

    papers: List[Dict[str, Any]] = []
    for fname in files:
        fpath = os.path.join(req.papers_dir, fname)
        text = read_text(fpath)
        summary = ""
        relevance_score = None
        fname_lower = fname.lower()
        paper_role = "query" if "__query" in fname_lower else (
            "target" if "__target" in fname_lower else None
        )
        if llm_base_url and text.strip():
            prompt = (
                f"{req.instructions}\n\n{term_block}\n\n"
                f"Paper excerpt:\n{text[:80000]}"
            )
            try:
                raw = _call_llm(prompt, llm_base_url, max_tokens=2048, temperature=0.0)
                summary = raw[:4000]
            except Exception as e:
                summary = f"[LLM error: {e}]"
        papers.append(
            {
                "paper_id": fname,
                "file_name": fname,
                "paper_role": paper_role,
                "analysis": {"summary": summary, "relevance_score": relevance_score},
            }
        )

    payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "papers": papers,
        "meta": {
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
            "constraints": req.constraints.dict() if req.constraints else None,
        },
    }
    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"legacy_run_alignment n_papers={len(papers)}\n")
    return _write_alignment_results(req, payload)


def _run_alignment_graded_impl(req: RunAlignmentGradedRequest) -> RunAlignmentResponse:
    try:
        return synthesis_graded.run_alignment_graded(
            req,
            call_llm=_call_llm,
            write_results=_write_alignment_results,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


@app.post("/run_alignment", response_model=RunAlignmentResponse)
def run_alignment(req: RunAlignmentRequest) -> RunAlignmentResponse:
    return _run_alignment_impl(req)


@app.post("/run_alignment_graded", response_model=RunAlignmentResponse)
def run_alignment_graded(req: RunAlignmentGradedRequest) -> RunAlignmentResponse:
    return _run_alignment_graded_impl(req)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GPU_API_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
