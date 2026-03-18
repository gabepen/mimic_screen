import os
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


def _convert_pdfs_to_text(pdf_dir: str, papers_dir: str) -> List[str]:
    if not os.path.isdir(pdf_dir):
        raise HTTPException(
            status_code=400,
            detail=f"pdf_dir does not exist or is not a directory: {pdf_dir}",
        )

    _ensure_dir(papers_dir)

    txt_paths: List[str] = []
    for name in sorted(os.listdir(pdf_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, name)
        if not os.path.isfile(pdf_path):
            continue
        try:
            result = _DOC_CONVERTER.convert(source=pdf_path)
            doc = result.document
            text = doc.export_to_markdown()
        except Exception as e:
            logger.warning(f"Docling failed for {pdf_path}: {e}")
            continue
        base = os.path.splitext(name)[0]
        txt_path = os.path.join(papers_dir, f"{base}.txt")
        try:
            with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
        except Exception as e:
            logger.warning(f"Could not write text for {pdf_path} -> {txt_path}: {e}")
            continue
        txt_paths.append(txt_path)

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
    txt_paths = _convert_pdfs_to_text(req.pdf_dir, req.papers_dir)
    logger.info(
        f"Docling node converted {len(txt_paths)} PDFs for {req.alignment_id} "
        f"into {req.papers_dir}"
    )
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

