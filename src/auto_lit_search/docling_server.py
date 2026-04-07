import os
import json
import gc
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

# Headless-friendly defaults for PDF stacks that touch Qt/XCB in some builds.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption


app = FastAPI(title="Docling PDF-to-text node")
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}
_ASYNC_QUEUE: "deque[tuple[str, ConvertAlignmentRequest]]" = deque()
_ASYNC_LOCK = threading.Lock()
_ASYNC_WORKER_STARTED = False


class Constraints(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ConvertAlignmentRequest(BaseModel):
    alignment_id: str
    pdf_dir: str
    papers_dir: str
    query: str
    target_id: str
    constraints: Optional[Constraints] = None
    instructions: str
    output_root: str
    gene_context: Optional[Dict[str, Any]] = None
    analysis_host: str
    analysis_port: int
    evaluation_manifest_path: Optional[str] = None
    call_analysis: bool = True


class ConvertAlignmentResponse(BaseModel):
    status: str
    alignment_id: str
    papers_dir: str
    results_path: Optional[str] = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


_DOC_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(),
    }
)

def _best_effort_free_memory() -> None:
    """
    Docling/pdf stacks can hold onto large CPU/GPU allocations between requests.
    Force Python GC and (when available) clear CUDA caching allocator.
    """
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _extract_text_pypdf(pdf_path: str) -> str:
    try:
        import pypdf  # type: ignore
    except ImportError:
        return ""
    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception:
        return ""
    parts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts)


def _load_docling_required_pdf_basenames(
    manifest_path: Optional[str],
) -> Optional[set[str]]:
    if not manifest_path:
        return None
    if not os.path.isfile(manifest_path):
        logger.warning(
            f"evaluation_manifest_path does not exist, ignoring filter: {manifest_path}"
        )
        return None
    out: set[str] = set()
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
                details = rec.get("details") or {}
                if not details.get("pdf_docling_required"):
                    continue
                p = rec.get("pdf_path")
                if not p:
                    continue
                base = os.path.splitext(os.path.basename(str(p)))[0]
                if base:
                    out.add(base)
    except Exception as e:
        logger.warning(f"Could not parse evaluation manifest {manifest_path}: {e}")
        return None
    logger.info(
        f"Loaded {len(out)} docling-required PDFs from manifest {manifest_path}"
    )
    return out


def _convert_pdfs_to_text(
    pdf_dir: str, papers_dir: str, allowed_pdf_basenames: Optional[set[str]] = None
) -> List[str]:
    if not os.path.isdir(pdf_dir):
        raise HTTPException(
            status_code=400,
            detail=f"pdf_dir does not exist or is not a directory: {pdf_dir}",
        )

    _ensure_dir(papers_dir)

    txt_paths: List[str] = []
    attempted = 0
    for name in sorted(os.listdir(pdf_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(name)[0]
        if allowed_pdf_basenames is not None and base not in allowed_pdf_basenames:
            continue
        pdf_path = os.path.join(pdf_dir, name)
        if not os.path.isfile(pdf_path):
            continue
        txt_path = os.path.join(papers_dir, f"{base}.txt")
        if os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0:
            txt_paths.append(txt_path)
            continue
        attempted += 1
        text = ""
        result = None
        doc = None
        try:
            result = _DOC_CONVERTER.convert(source=pdf_path)
            doc = result.document
            text = doc.export_to_markdown()
        except Exception as e:
            logger.warning(f"Docling failed for {pdf_path}: {e}")
            text = ""
        finally:
            # Ensure large objects are dereferenced between PDFs.
            try:
                del doc
            except Exception:
                pass
            try:
                del result
            except Exception:
                pass
            _best_effort_free_memory()
        if not (text or "").strip():
            fallback = _extract_text_pypdf(pdf_path)
            if fallback.strip():
                text = fallback
                logger.info(
                    f"Used pypdf fallback for {pdf_path} ({len(fallback)} chars)"
                )
            else:
                continue
        try:
            with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception as e:
            logger.warning(f"Could not write text for {pdf_path} -> {txt_path}: {e}")
            continue
        txt_paths.append(txt_path)

    if allowed_pdf_basenames is not None and attempted == 0:
        logger.info("No PDFs selected for Docling conversion from manifest filter")
        return []

    if not txt_paths:
        raise HTTPException(
            status_code=400,
            detail=f"no PDFs converted in pdf_dir: {pdf_dir}",
        )
    return txt_paths


def _docling_chunk_size() -> int:
    raw = os.environ.get("DOCLING_CHUNK_SIZE", "25")
    try:
        n = int(str(raw).strip())
        return max(1, n)
    except Exception:
        return 25


def _docling_async_poll_interval_sec() -> float:
    raw = os.environ.get("DOCLING_ASYNC_POLL_INTERVAL_SEC", "2")
    try:
        v = float(str(raw).strip())
        return max(0.5, v)
    except Exception:
        return 2.0


def _docling_async_max_queue() -> int:
    raw = os.environ.get("DOCLING_ASYNC_MAX_QUEUE", "1")
    try:
        v = int(str(raw).strip())
        return max(1, v)
    except Exception:
        return 1


def _eligible_pdf_basenames(
    pdf_dir: str, allowed_pdf_basenames: Optional[set[str]]
) -> List[str]:
    out: List[str] = []
    for name in sorted(os.listdir(pdf_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(name)[0]
        if allowed_pdf_basenames is not None and base not in allowed_pdf_basenames:
            continue
        pdf_path = os.path.join(pdf_dir, name)
        if os.path.isfile(pdf_path):
            out.append(base)
    return out


def _convert_alignment_sync(req: ConvertAlignmentRequest) -> ConvertAlignmentResponse:
    allowed = _load_docling_required_pdf_basenames(req.evaluation_manifest_path)
    txt_paths = _convert_pdfs_to_text(req.pdf_dir, req.papers_dir, allowed)
    logger.info(
        f"Docling node converted {len(txt_paths)} PDFs for {req.alignment_id} "
        f"into {req.papers_dir}"
    )
    results_path = ""
    if req.call_analysis:
        results_path = _call_analysis_node(req)
    return ConvertAlignmentResponse(
        status="ok",
        alignment_id=req.alignment_id,
        papers_dir=req.papers_dir,
        results_path=results_path or None,
    )


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
            allowed = _load_docling_required_pdf_basenames(req.evaluation_manifest_path)
            basenames = _eligible_pdf_basenames(req.pdf_dir, allowed)
            if not basenames:
                raise HTTPException(
                    status_code=400, detail=f"no PDFs selected in pdf_dir: {req.pdf_dir}"
                )
            chunk_size = _docling_chunk_size()
            chunks = [
                basenames[i : i + chunk_size]
                for i in range(0, len(basenames), chunk_size)
            ]
            converted_total = 0
            for idx, chunk in enumerate(chunks, start=1):
                converted = _convert_pdfs_to_text(
                    req.pdf_dir, req.papers_dir, set(chunk)
                )
                converted_total += len(converted)
                with _ASYNC_LOCK:
                    job = _ASYNC_JOBS.get(job_id) or {}
                    job["chunks_total"] = len(chunks)
                    job["chunks_done"] = idx
                    job["converted_count"] = converted_total
                    _ASYNC_JOBS[job_id] = job
                _best_effort_free_memory()
            results_path = ""
            if req.call_analysis:
                results_path = _call_analysis_node(req)
            with _ASYNC_LOCK:
                job = _ASYNC_JOBS.get(job_id) or {}
                job["status"] = "succeeded"
                job["finished_at"] = time.time()
                job["result"] = ConvertAlignmentResponse(
                    status="ok",
                    alignment_id=req.alignment_id,
                    papers_dir=req.papers_dir,
                    results_path=results_path or None,
                ).dict()
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


def _call_analysis_node(
    req: ConvertAlignmentRequest,
) -> str:
    base_url = f"http://{req.analysis_host}:{req.analysis_port}"
    url = f"{base_url}/run_alignment"

    payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "papers_dir": req.papers_dir,
        "query": req.query,
        "target_id": req.target_id,
        "constraints": (
            req.constraints.dict() if isinstance(req.constraints, Constraints) else None
        ),
        "instructions": req.instructions,
        "output_root": req.output_root,
        "gene_context": req.gene_context,
    }

    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return str(data.get("results_path") or "")


@app.post("/convert_alignment", response_model=ConvertAlignmentResponse)
def convert_alignment(req: ConvertAlignmentRequest) -> ConvertAlignmentResponse:
    return _convert_alignment_sync(req)


@app.get("/docling_capacity")
def docling_capacity() -> Dict[str, Any]:
    _ensure_async_worker_started()
    with _ASYNC_LOCK:
        queue_depth = len(_ASYNC_QUEUE)
        running = sum(1 for v in _ASYNC_JOBS.values() if v.get("status") == "running")
        max_queue = _docling_async_max_queue()
    return {
        "status": "ok",
        "can_accept": queue_depth < max_queue,
        "queue_depth": queue_depth,
        "max_queue": max_queue,
        "running_jobs": running,
    }


@app.post("/convert_alignment_async")
def convert_alignment_async(req: ConvertAlignmentRequest) -> Dict[str, Any]:
    _ensure_async_worker_started()
    max_queue = _docling_async_max_queue()
    with _ASYNC_LOCK:
        if len(_ASYNC_QUEUE) >= max_queue:
            raise HTTPException(
                status_code=429,
                detail=f"docling queue full (queue_depth={len(_ASYNC_QUEUE)} max_queue={max_queue})",
            )
        job_id = uuid.uuid4().hex
        _ASYNC_JOBS[job_id] = {
            "job_id": job_id,
            "alignment_id": req.alignment_id,
            "status": "queued",
            "submitted_at": time.time(),
            "chunks_total": 0,
            "chunks_done": 0,
            "converted_count": 0,
        }
        _ASYNC_QUEUE.append((job_id, req))
    return {
        "job_id": job_id,
        "alignment_id": req.alignment_id,
        "status": "queued",
        "poll_interval_sec": _docling_async_poll_interval_sec(),
    }


@app.get("/convert_alignment_status/{job_id}")
def convert_alignment_status(job_id: str) -> Dict[str, Any]:
    with _ASYNC_LOCK:
        job = _ASYNC_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
        out = dict(job)
    return out


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DOCLING_API_PORT", "9100"))
    uvicorn.run(app, host="0.0.0.0", port=port)

