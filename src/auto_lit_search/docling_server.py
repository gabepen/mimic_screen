import os
import json
import gc
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Union

# Headless-friendly defaults for PDF stacks that touch Qt/XCB in some builds.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

    _PDF_PIPELINE_CLS = ThreadedStandardPdfPipeline
    _PDF_PIPELINE_OPTIONS_CLS = ThreadedPdfPipelineOptions
    _PDF_PIPELINE_KIND = "threaded_standard"
except ImportError:  # pragma: no cover - older docling
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    _PDF_PIPELINE_CLS = StandardPdfPipeline
    _PDF_PIPELINE_OPTIONS_CLS = PdfPipelineOptions
    _PDF_PIPELINE_KIND = "standard"


app = FastAPI(title="Docling PDF-to-text node")
_ASYNC_JOBS: Dict[str, Dict[str, Any]] = {}
_ASYNC_QUEUE: "deque[tuple[str, ConvertAlignmentRequest]]" = deque()
_ASYNC_LOCK = threading.Lock()
_ASYNC_WORKER_STARTED = False

_DOC_CONVERTER: Optional[DocumentConverter] = None
_DOC_CONVERTER_LOCK = threading.Lock()
_DOC_CONVERTER_INFO: Dict[str, Any] = {}


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


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(str(os.environ.get(name, str(default))).strip()))
    except ValueError:
        return default


def _parse_ocr_langs() -> List[str]:
    raw = os.environ.get("DOCLING_OCR_LANGS", "english").strip()
    if not raw:
        return ["english"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def _default_accelerator_device() -> Union[str, AcceleratorDevice]:
    """
    Prefer CUDA when PyTorch sees a GPU unless DOCLING_DEVICE is set (Docling reads this too).
    """
    explicit = os.environ.get("DOCLING_DEVICE", "").strip()
    if explicit:
        return explicit
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return AcceleratorDevice.CUDA
    except Exception:
        pass
    return AcceleratorDevice.AUTO


def _rapidocr_backend() -> str:
    raw = os.environ.get("DOCLING_RAPIDOCR_BACKEND", "torch").strip().lower()
    if raw in ("torch", "onnxruntime", "paddle", "openvino"):
        return raw
    return "torch"


def _make_pdf_format_option(
    device: Union[str, AcceleratorDevice],
    ocr_backend: str,
) -> PdfFormatOption:
    opts = _PDF_PIPELINE_OPTIONS_CLS(
        accelerator_options=AcceleratorOptions(device=device),
        ocr_batch_size=_int_env("DOCLING_OCR_BATCH_SIZE", 8),
        layout_batch_size=_int_env("DOCLING_LAYOUT_BATCH_SIZE", 32),
        table_batch_size=_int_env("DOCLING_TABLE_BATCH_SIZE", 4),
        ocr_options=RapidOcrOptions(
            backend=ocr_backend,  # type: ignore[arg-type]
            lang=_parse_ocr_langs(),
        ),
    )
    return PdfFormatOption(pipeline_cls=_PDF_PIPELINE_CLS, pipeline_options=opts)


def _create_document_converter(
    device: Union[str, AcceleratorDevice],
    ocr_backend: str,
) -> DocumentConverter:
    fmt = _make_pdf_format_option(device, ocr_backend)
    conv = DocumentConverter(format_options={InputFormat.PDF: fmt})
    try:
        conv.initialize_pipeline(InputFormat.PDF)
    except Exception as e:
        logger.warning("Docling initialize_pipeline: {}", e)
    return conv


def _get_document_converter() -> DocumentConverter:
    """
    Lazy singleton: CUDA + RapidOCR torch by default; CPU/onnxruntime fallback if init fails.
    """
    global _DOC_CONVERTER, _DOC_CONVERTER_INFO
    with _DOC_CONVERTER_LOCK:
        if _DOC_CONVERTER is not None:
            return _DOC_CONVERTER
        device = _default_accelerator_device()
        ocr_backend = _rapidocr_backend()
        try:
            _DOC_CONVERTER = _create_document_converter(device, ocr_backend)
            _DOC_CONVERTER_INFO = {
                "accelerator_device": str(device),
                "rapidocr_backend": ocr_backend,
                "pdf_pipeline": _PDF_PIPELINE_KIND,
                "ocr_batch_size": _int_env("DOCLING_OCR_BATCH_SIZE", 8),
                "layout_batch_size": _int_env("DOCLING_LAYOUT_BATCH_SIZE", 32),
            }
            logger.info(
                "Docling converter ready: pipeline={} device={} rapidocr_backend={} "
                "ocr_batch={} layout_batch={}",
                _PDF_PIPELINE_KIND,
                device,
                ocr_backend,
                _DOC_CONVERTER_INFO["ocr_batch_size"],
                _DOC_CONVERTER_INFO["layout_batch_size"],
            )
        except Exception as e:
            logger.error(
                "Docling GPU init failed ({}); falling back to CPU + RapidOCR onnxruntime",
                e,
            )
            _DOC_CONVERTER = _create_document_converter(
                AcceleratorDevice.CPU, "onnxruntime"
            )
            _DOC_CONVERTER_INFO = {
                "accelerator_device": "cpu",
                "rapidocr_backend": "onnxruntime",
                "pdf_pipeline": _PDF_PIPELINE_KIND,
                "init_error": str(e)[:500],
            }
        return _DOC_CONVERTER


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
    try:
        pages = list(reader.pages)
    except Exception:
        return ""
    for page in pages:
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
    if not out:
        # Empty set is not "no filter": it would skip every PDF and fail the job.
        # Stale manifests (e.g. resume without refreshed rows) have this shape.
        logger.warning(
            f"Manifest {manifest_path} has no pdf_docling_required rows; "
            "ignoring filter (convert eligible PDFs in pdf_dir)"
        )
        return None
    logger.info(
        f"Loaded {len(out)} docling-required PDFs from manifest {manifest_path}"
    )
    return out


def _convert_pdfs_to_text(
    pdf_dir: str,
    papers_dir: str,
    allowed_pdf_basenames: Optional[set[str]] = None,
    alignment_id: str = "",
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
            if alignment_id:
                logger.info(
                    "docling_pdf_skip_cached alignment_id={} basename={} reason=non_empty_txt",
                    alignment_id,
                    base,
                )
            continue
        attempted += 1
        text = ""
        result = None
        doc = None
        docling_err = ""
        try:
            result = _get_document_converter().convert(source=pdf_path)
            doc = result.document
            text = (doc.export_to_markdown() or "").strip()
        except Exception as e:
            docling_err = str(e)[:500]
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
        mode = "docling"
        if not text:
            fallback = _extract_text_pypdf(pdf_path)
            if (fallback or "").strip():
                text = fallback.strip()
                mode = "pypdf_fallback"
                if alignment_id:
                    logger.info(
                        "docling_pdf_fallback alignment_id={} basename={} chars={}",
                        alignment_id,
                        base,
                        len(fallback),
                    )
            else:
                if alignment_id:
                    detail = docling_err or "no_markdown_no_pypdf"
                    logger.warning(
                        "docling_pdf_convert_failed alignment_id={} basename={} detail={}",
                        alignment_id,
                        base,
                        detail,
                    )
                continue
        try:
            with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception as e:
            if alignment_id:
                logger.warning(
                    "docling_pdf_write_failed alignment_id={} basename={} error={}",
                    alignment_id,
                    base,
                    str(e)[:300],
                )
            continue
        txt_paths.append(txt_path)
        if alignment_id:
            logger.info(
                "docling_pdf_convert_ok alignment_id={} basename={} mode={} chars={}",
                alignment_id,
                base,
                mode,
                len(text),
            )

    if allowed_pdf_basenames is not None and attempted == 0 and not txt_paths:
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
    txt_paths = _convert_pdfs_to_text(
        req.pdf_dir, req.papers_dir, allowed, req.alignment_id
    )
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
                    req.pdf_dir,
                    req.papers_dir,
                    set(chunk),
                    req.alignment_id,
                )
                converted_total += len(converted)
                logger.info(
                    "docling_chunk_summary alignment_id={} chunk={}/{} "
                    "chunk_pdfs={} txt_paths_returned={} cumulative_txt_paths={}",
                    req.alignment_id,
                    idx,
                    len(chunks),
                    len(chunk),
                    len(converted),
                    converted_total,
                )
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
def healthz() -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok", "detail": "ready"}
    if _DOC_CONVERTER_INFO:
        out["docling"] = dict(_DOC_CONVERTER_INFO)
    return out


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DOCLING_API_PORT", "9100"))
    uvicorn.run(app, host="0.0.0.0", port=port)

