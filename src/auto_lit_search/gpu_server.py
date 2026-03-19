import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

app = FastAPI(title="auto_lit_search GPU node")

TEXT_EXTENSIONS = (".txt",)
MAX_PAPER_CHARS = 120000


class Constraints(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class RunAlignmentRequest(BaseModel):
    alignment_id: str
    papers_dir: str
    query: str
    target_id: str
    constraints: Optional[Constraints] = None
    instructions: str
    output_root: str
    # Optional gene metadata for improving prompts (gene_context["query"], gene_context["target"])
    gene_context: Optional[Dict[str, Any]] = None


class RunAlignmentResponse(BaseModel):
    status: str
    alignment_id: str
    results_path: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_paper_files(papers_dir: str) -> List[str]:
    return sorted(
        f for f in os.listdir(papers_dir)
        if os.path.isfile(os.path.join(papers_dir, f))
        and (f.endswith(TEXT_EXTENSIONS) or not f.endswith((".pdf", ".xml")))
    )


def _read_text(path: str, max_chars: int = MAX_PAPER_CHARS) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n[truncated]"
    return s


def _call_llm(
    user_content: str,
    base_url: str,
    model: str = "default",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    import requests
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return (msg.get("content") or "").strip()
    return ""


def _analyze_paper(
    file_path: str,
    query: str,
    target_id: str,
    instructions: str,
    constraints: Optional[Constraints],
    llm_base_url: Optional[str],
    log_path: Optional[str],
    gene_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stat = os.stat(file_path)
    text = _read_text(file_path)
    summary = ""
    relevance_score = None
    notes = ""

    # Infer which side of the alignment this paper supports from filename suffix.
    # Collect.py writes files as: <pmcid>__query.txt and <pmcid>__target.txt.
    fname = os.path.basename(file_path)
    paper_role: Optional[str] = None
    if fname.endswith("__query.txt"):
        paper_role = "query"
    elif fname.endswith("__target.txt"):
        paper_role = "target"

    query_meta = (gene_context or {}).get("query") or {}
    target_meta = (gene_context or {}).get("target") or {}
    query_gene_name = (query_meta.get("gene_name") or "").strip() or query
    target_gene_name = (target_meta.get("gene_name") or "").strip() or target_id

    def _format_identifiers(meta: Dict[str, Any]) -> str:
        # Keep this compact: only include non-empty identifier fields.
        keys = [
            ("gene_name", "gene_name"),
            ("uniprot_id", "uniprot_id"),
            ("entrez_id", "entrez_id"),
            ("locus_tag", "locus_tag"),
            ("genbank_acc", "genbank_acc"),
            ("common_name", "common_name"),
        ]
        parts: List[str] = []
        for k, label in keys:
            v = meta.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(f"{label}={s}")
        return "; ".join(parts) if parts else "none"

    if paper_role == "query":
        focus_gene_id = query
        focus_gene_name = query_gene_name
        other_gene_id = target_id
        other_gene_name = target_gene_name
    elif paper_role == "target":
        focus_gene_id = target_id
        focus_gene_name = target_gene_name
        other_gene_id = query
        other_gene_name = query_gene_name
    else:
        # Fallback when file naming isn't role-suffixed.
        focus_gene_id = query
        focus_gene_name = query_gene_name
        other_gene_id = target_id
        other_gene_name = target_gene_name

    focus_id_str = _format_identifiers(
        query_meta if paper_role == "query" else (target_meta if paper_role == "target" else query_meta)
    )
    other_id_str = _format_identifiers(
        query_meta if paper_role == "target" else (target_meta if paper_role == "query" else target_meta)
    )

    user_content = (
        f"{instructions}\n\n"
        f"Alignment genes:\n"
        f"- Focus gene: {focus_gene_name} ({focus_gene_id})\n"
        f"- Focus gene identifiers: {focus_id_str}\n"
        f"- Other gene: {other_gene_name} ({other_gene_id})\n"
        f"- Other gene identifiers: {other_id_str}\n"
        f"- Paper role in alignment: {paper_role or 'unknown'}\n\n"
        f"Paper excerpt:\n{text[:80000]}"
    )

    if llm_base_url and text.strip():
        max_tokens = (constraints and constraints.max_tokens) or 4096
        temperature = (constraints and constraints.temperature) if constraints is not None else 0.0
        if temperature is None:
            temperature = 0.0
        try:
            raw = _call_llm(user_content, llm_base_url, max_tokens=max_tokens, temperature=temperature)
            if raw:
                summary = raw
                try:
                    if "relevance" in raw.lower() or "score" in raw.lower():
                        for part in raw.replace(",", " ").split():
                            try:
                                v = float(part)
                                if 0 <= v <= 1:
                                    relevance_score = v
                                    break
                            except ValueError:
                                continue
                except Exception:
                    pass
        except Exception as e:
            notes = str(e)
            logger.warning(f"LLM call failed for {file_path}: {e}")

    if log_path:
        _ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- {os.path.basename(file_path)} ---\n")
            f.write(f"query={query} target_id={target_id}\n")
            f.write(f"inferred_paper_role={paper_role}\n")
            f.write(f"focus_gene_id={focus_gene_id} other_gene_id={other_gene_id}\n")
            # Keep prompt small; excerpt can be huge.
            if text.strip():
                f.write("prompt_preview:\n")
                f.write(user_content[:4000] + ("\n...[truncated]\n" if len(user_content) > 4000 else "\n"))
            f.write(f"summary_len={len(summary)} relevance_score={relevance_score}\n")
            f.write(f"{summary[:2000]}\n\n")

    return {
        "summary": summary,
        "relevance_score": relevance_score,
        "notes": notes,
        "file_size_bytes": stat.st_size,
        "query": query,
        "target_id": target_id,
        "paper_role": paper_role,
        "focus_gene_id": focus_gene_id,
    }


def _run_alignment_impl(req: RunAlignmentRequest) -> RunAlignmentResponse:
    if not os.path.isdir(req.papers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"papers_dir does not exist or is not a directory: {req.papers_dir}",
        )

    _ensure_dir(req.output_root)
    result_path = os.path.join(req.output_root, f"{req.alignment_id}_results.json")
    log_dir = os.path.join(req.output_root, "logs")
    log_path = os.path.join(log_dir, f"{req.alignment_id}.log")

    files = _list_paper_files(req.papers_dir)
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"no files found in papers_dir: {req.papers_dir}",
        )

    # If we have role-labeled inputs (<pmcid>__query.txt / <pmcid>__target.txt),
    # prefer them over any legacy unlabeled .txt files.
    labeled = [
        f
        for f in files
        if f.endswith("__query.txt") or f.endswith("__target.txt")
    ]
    if labeled:
        files = labeled

    llm_base_url = os.environ.get("VLLM_BASE_URL")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    papers: List[Dict[str, Any]] = []
    for fname in files:
        fpath = os.path.join(req.papers_dir, fname)
        fname_base = os.path.basename(fpath)
        analysis = _analyze_paper(
            file_path=fpath,
            query=req.query,
            target_id=req.target_id,
            instructions=req.instructions,
            constraints=req.constraints,
            llm_base_url=llm_base_url,
            log_path=log_path,
            gene_context=req.gene_context,
        )
        papers.append(
            {
                "paper_id": fname,
                "file_name": fname,
                "analysis": analysis,
            }
        )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "papers": papers,
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
            "constraints": req.constraints.dict() if req.constraints else None,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"GPU node wrote results for {req.alignment_id} "
        f"({len(papers)} papers) -> {result_path}"
    )

    return RunAlignmentResponse(
        status="ok",
        alignment_id=req.alignment_id,
        results_path=result_path,
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


@app.post("/run_alignment", response_model=RunAlignmentResponse)
def run_alignment(req: RunAlignmentRequest) -> RunAlignmentResponse:
    return _run_alignment_impl(req)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GPU_API_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

