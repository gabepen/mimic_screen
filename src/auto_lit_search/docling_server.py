import os
import json
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption


app = FastAPI(title="Docling PDF-to-text node")


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
        attempted += 1
        try:
            result = _DOC_CONVERTER.convert(source=pdf_path)
            doc = result.document
            text = doc.export_to_markdown()
        except Exception as e:
            logger.warning(f"Docling failed for {pdf_path}: {e}")
            continue
        txt_path = os.path.join(papers_dir, f"{base}.txt")
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


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DOCLING_API_PORT", "9100"))
    uvicorn.run(app, host="0.0.0.0", port=port)

