"""
Collect module for automated literature search pipeline (Module 3 of 4).

Bulk full-text downloader.

Consumes mapping+search results and downloads full texts for all discovered
papers into a shared data directory for later LLM analysis.

Pipeline Interface:
    run(df_or_path, **kwargs) -> pd.DataFrame
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
API_DELAY = 0.35


@dataclass
class DownloadRecord:
    paper_id: str
    source: str
    pmcid: Optional[str]
    pdf_path: Optional[str]
    text_path: Optional[str]
    status: str
    message: Optional[str] = None


def _configure_file_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collect_download_debug.log")
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Collect (download) debug log file: {log_path}")


def _normalize_paper_id(pid: Any) -> Optional[str]:
    if pid is None:
        return None
    s = str(pid).strip()
    if not s or s.lower() == "nan":
        return None
    return s


_EXCLUDED_PUBTYPE_SUBSTRINGS: List[str] = [
    # Notices/corrections that are not the underlying research article.
    "expression-of-concern",
    "retraction",
    "correction",
    "erratum",
    "corrigendum",
    "withdrawn",
    # Editorial-ish / non-research pieces.
    "comment",
    "editorial",
    "letter",
    "abstract",
    "review",
]


def _pubtype_is_excluded(pubtypes: List[Any]) -> bool:
    for pt in pubtypes:
        norm = str(pt).strip().lower().replace(" ", "-")
        if not norm:
            continue
        for sub in _EXCLUDED_PUBTYPE_SUBSTRINGS:
            if sub in norm:
                return True
    return False


def _resolve_to_pmcid(
    paper_id: str,
    session: requests.Session,
    cache: Dict[str, Optional[str]],
    delay: float = API_DELAY,
) -> Optional[str]:
    """
    Resolve an arbitrary paper identifier to a Europe PMC PMCID, if possible.

    Supported inputs:
        - PMC123456 or PMC:123456
        - PMID:123456 or bare numeric PMID
        - DOI (10.xxxx/...)
        - Other IDs resolvable via EXT_ID search.
    """
    if paper_id in cache:
        return cache[paper_id]

    pid = paper_id.strip()
    u = pid.upper()
    search_query: str

    # If the input is already a PMC id, still resolve it through the Europe PMC
    # search API so we can filter out notices (e.g. expression of concern).
    if u.startswith("PMC:"):
        pmcid = u[4:].strip()
        pmcid = f"PMC{pmcid}" if not pmcid.upper().startswith("PMC") else pmcid
        search_query = pmcid
    elif u.startswith("PMC"):
        search_query = u
    else:
        # Try to resolve via EXT_ID search.
        if u.startswith("PMID:"):
            ext_id = u[5:].strip()
        else:
            ext_id = pid
        search_query = f"EXT_ID:{ext_id}"

    time.sleep(delay)
    try:
        resp = session.get(
            EUROPEPMC_SEARCH_URL,
            params={
                "query": search_query,
                "format": "json",
                "resultType": "core",
                # Some DOIs map to multiple PMCID versions/records
                # (e.g. notice + underlying research). We filter the results by
                # pubTypeList to prefer real research articles.
                "pageSize": 20,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"EXT_ID lookup failed for {paper_id}: {e}")
        cache[paper_id] = None
        return None

    results = (data.get("resultList") or {}).get("result") or []
    for rec in results:
        pmcid = rec.get("pmcid")
        if not pmcid:
            continue

        pubtypes = (rec.get("pubTypeList") or {}).get("pubType") or []
        if pubtypes and _pubtype_is_excluded(pubtypes):
            logger.debug(f"Skipping non-research pubType={pubtypes} for {paper_id} -> {pmcid}")
            continue

        pmcid = str(pmcid).strip()
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        cache[paper_id] = pmcid
        return pmcid

    cache[paper_id] = None
    return None


def _fetch_fulltext_pdf(
    pmcid: str,
    session: requests.Session,
    pdf_dir: str,
    timeout: int = 120,
) -> Optional[str]:
    """
    Download full-text PDF for a given PMCID from Europe PMC, if available.

    Saves to pdf_dir and returns local path, or None if not available.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    safe = pmcid.replace("/", "_").replace(":", "_")
    out_path = os.path.join(pdf_dir, f"{safe}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.debug(f"PDF cache hit: {out_path}")
        return out_path

    url = f"{EUROPEPMC_FULLTEXT_BASE}/{pmcid}/fullTextPDF"
    logger.debug(f"Fetching PDF for {pmcid} -> {url}")
    time.sleep(API_DELAY)
    try:
        resp = session.get(url, timeout=timeout)
        # Some responses may be HTML if PDF is not available.
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            logger.debug(
                f"PDF not available for {pmcid} (content-type={content_type!r})"
            )
            return None
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
    except Exception as e:
        logger.debug(f"PDF fetch failed for {pmcid}: {e}")
        return None


def _fetch_fulltext_xml(
    pmcid: str,
    session: requests.Session,
    xml_dir: str,
    timeout: int = 120,
) -> Optional[str]:
    """
    Download full-text XML for a given PMCID from Europe PMC, if available.

    Saves to xml_dir and returns local path, or None if not available.
    """
    os.makedirs(xml_dir, exist_ok=True)
    safe = pmcid.replace("/", "_").replace(":", "_")
    out_path = os.path.join(xml_dir, f"{safe}.xml")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.debug(f"XML cache hit: {out_path}")
        return out_path

    url = f"{EUROPEPMC_FULLTEXT_BASE}/{pmcid}/fullTextXML"
    logger.debug(f"Fetching XML for {pmcid} -> {url}")
    time.sleep(API_DELAY)
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        if not text or len(text) < 500:
            return None
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
        return out_path
    except Exception as e:
        logger.debug(f"XML fetch failed for {pmcid}: {e}")
        return None


def _extract_text_from_xml(xml_path: str) -> str:
    """Very simple XML -> plain text extraction."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return ""

    parts: List[str] = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
    return "\n".join(parts)


def _extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract plain text from a PDF using pypdf, if installed.

    Returns empty string on failure.
    """
    try:
        import pypdf  # type: ignore
    except Exception:
        logger.debug("pypdf not installed; skipping PDF text extraction")
        return ""

    try:
        reader = pypdf.PdfReader(pdf_path)
    except Exception as e:
        logger.debug(f"Failed to open PDF {pdf_path}: {e}")
        return ""

    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t.strip())
    return "\n\n".join(texts)


def download_papers_to_dir(
    paper_ids_with_source: List[Tuple[str, str]],
    output_dir: str,
    session: Optional[requests.Session] = None,
    pmcid_cache: Optional[Dict[str, Optional[str]]] = None,
    no_cache: bool = False,
    force_pdfs: bool = False,
) -> List[DownloadRecord]:
    """
    Download full-text for a list of (paper_id, source) into output_dir.
    Writes output_dir/<safe_id>.txt (and optionally pdf/xml subdirs).
    Returns list of DownloadRecord. Use output_dir as papers_dir for GPU API.
    """
    session = session or requests.Session()
    pmcid_cache = pmcid_cache if pmcid_cache is not None else {}
    os.makedirs(output_dir, exist_ok=True)
    pdf_dir = os.path.join(output_dir, "pdf")
    xml_dir = os.path.join(output_dir, "text_xml")
    text_dir = output_dir

    records: List[DownloadRecord] = []
    for paper_id, source in paper_ids_with_source:
        pmcid = _resolve_to_pmcid(paper_id, session, pmcid_cache)
        if not pmcid:
            records.append(
                DownloadRecord(
                    paper_id=paper_id,
                    source=source,
                    pmcid=None,
                    pdf_path=None,
                    text_path=None,
                    status="skipped",
                    message="no PMCID",
                )
            )
            continue

        pdf_path = None
        text_path = None
        safe = pmcid.replace("/", "_").replace(":", "_")

        # If cached text exists and caching is allowed, use it as a quick path.
        # When force_pdfs=True we still prefer PDFs for downstream Docling.
        if not no_cache:
            candidate_text = os.path.join(text_dir, f"{safe}.txt")
            if os.path.exists(candidate_text) and os.path.getsize(candidate_text) > 0:
                text_path = candidate_text

        # XML extraction is used as a fallback text source (but may be noisy).
        if not text_path:
            xml_path = _fetch_fulltext_xml(pmcid, session, xml_dir)
            if xml_path:
                extracted = _extract_text_from_xml(xml_path)
                if extracted.strip():
                    text_path = os.path.join(text_dir, f"{safe}.txt")
                    with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                        f.write(extracted)

        # PDF fetching is either the normal "no text" path, or always when
        # force_pdfs=True (so Docling can convert PDFs reliably).
        if force_pdfs or not text_path:
            pdf_path = _fetch_fulltext_pdf(pmcid, session, pdf_dir)

            # Only extract PDF text if we still don't have any text_path.
            if pdf_path and not text_path:
                extracted = _extract_text_from_pdf(pdf_path)
                if extracted.strip():
                    text_path = os.path.join(text_dir, f"{safe}.txt")
                    with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                        f.write(extracted)

        status = "ok" if text_path else ("partial" if pdf_path else "failed")
        records.append(
            DownloadRecord(
                paper_id=paper_id,
                source=source,
                pmcid=pmcid,
                pdf_path=pdf_path,
                text_path=text_path,
                status=status,
                message=None if text_path else "no text extracted",
            )
        )
    return records


def _iter_paper_ids_from_search_df(df: pd.DataFrame) -> Iterable[Tuple[str, str]]:
    """
    Yield (paper_id, source) from search results DataFrame.

    Expects columns:
        - query_paper_dois
        - target_paper_dois
    which contain JSON-encoded lists of IDs.
    """
    for _, row in df.iterrows():
        for col, source in (
            ("query_paper_dois", "query"),
            ("target_paper_dois", "target"),
        ):
            if col not in row:
                continue
            val = row[col]
            if isinstance(val, str):
                try:
                    ids = json.loads(val) if val else []
                except Exception:
                    ids = []
            else:
                ids = val or []
            if not isinstance(ids, list):
                continue
            for pid in ids:
                norm = _normalize_paper_id(pid)
                if not norm:
                    continue
                yield norm, source


def run(
    df_or_path,
    data_root: str = "/private/groups/corbettlab/gabe/auto_lit_eval_data",
    batch_size: int = 500,
    max_papers: Optional[int] = None,
    delete_pdf_after_text: bool = False,
    no_cache: bool = False,
) -> pd.DataFrame:
    """
    Bulk full-text downloader for papers discovered in the search module.

    Args:
        df_or_path: DataFrame or path to CSV/JSON produced by the search module.
        data_root: Shared data root (contains pdf/, text/, llm_queue/, logs/, etc.).
        batch_size: Number of papers per manifest batch file.
        max_papers: Optional cap on number of unique papers to process.
        delete_pdf_after_text: If True, delete PDFs after successful text extraction.
        no_cache: If True, ignore any previously downloaded files.

    Returns:
        DataFrame of DownloadRecord rows.
    """
    os.makedirs(data_root, exist_ok=True)
    pdf_dir = os.path.join(data_root, "pdf")
    text_dir = os.path.join(data_root, "text")
    xml_dir = os.path.join(data_root, "text_xml")
    llm_queue_dir = os.path.join(data_root, "llm_queue")
    logs_dir = os.path.join(data_root, "logs")
    for d in (pdf_dir, text_dir, xml_dir, llm_queue_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    _configure_file_logging(logs_dir)

    if isinstance(df_or_path, str):
        path = df_or_path
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = []
            for query_id, alignments in data.items():
                for al in alignments:
                    q_ids = al.get("query_paper_dois", [])
                    t_ids = al.get("target_paper_dois", [])
                    rows.append(
                        {
                            "query": query_id,
                            "target": al.get("target", ""),
                            "query_paper_dois": json.dumps(q_ids),
                            "target_paper_dois": json.dumps(t_ids),
                        }
                    )
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(path)
    else:
        df = df_or_path.copy()

    if df.empty:
        logger.warning("Collect (download): no papers found in input")
        return pd.DataFrame(
            columns=[
                "paper_id",
                "source",
                "pmcid",
                "pdf_path",
                "text_path",
                "status",
                "message",
            ]
        )

    session = requests.Session()
    pmcid_cache: Dict[str, Optional[str]] = {}

    # Build unique paper list.
    unique: Dict[str, str] = {}
    for pid, src in _iter_paper_ids_from_search_df(df):
        if pid not in unique:
            unique[pid] = src

    paper_items: List[Tuple[str, str]] = list(unique.items())
    if max_papers is not None:
        paper_items = paper_items[:max_papers]

    logger.info(
        f"Collect (download): preparing to process {len(paper_items)} unique papers "
        f"(batch_size={batch_size})"
    )

    records: List[DownloadRecord] = []
    batch: List[DownloadRecord] = []
    batch_index = 0

    def flush_batch() -> None:
        nonlocal batch, batch_index
        if not batch:
            return
        batch_index += 1
        manifest_path = os.path.join(
            llm_queue_dir, f"batch_{batch_index:04d}.jsonl"
        )
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                for rec in batch:
                    f.write(
                        json.dumps(
                            {
                                "paper_id": rec.paper_id,
                                "source": rec.source,
                                "pmcid": rec.pmcid,
                                "pdf_path": rec.pdf_path,
                                "text_path": rec.text_path,
                                "status": rec.status,
                            }
                        )
                        + "\n"
                    )
            logger.info(
                f"Wrote manifest {manifest_path} ({len(batch)} papers, batch={batch_index})"
            )
        except Exception as e:
            logger.warning(f"Could not write manifest {manifest_path}: {e}")
        batch = []

    for idx, (paper_id, source) in enumerate(paper_items, start=1):
        pmcid = _resolve_to_pmcid(paper_id, session, pmcid_cache)
        if not pmcid:
            msg = "no PMCID / no full text in Europe PMC"
            rec = DownloadRecord(
                paper_id=paper_id,
                source=source,
                pmcid=None,
                pdf_path=None,
                text_path=None,
                status="skipped",
                message=msg,
            )
            records.append(rec)
            batch.append(rec)
        else:
            pdf_path = None
            text_path = None
            status = "ok"
            message = None

            if not no_cache:
                # Reuse existing files if present.
                safe = pmcid.replace("/", "_").replace(":", "_")
                candidate_pdf = os.path.join(pdf_dir, f"{safe}.pdf")
                candidate_text = os.path.join(text_dir, f"{safe}.txt")
                if os.path.exists(candidate_pdf):
                    pdf_path = candidate_pdf
                if os.path.exists(candidate_text):
                    text_path = candidate_text

            if not text_path:
                # Try XML, then PDF, then cache-only.
                xml_path = _fetch_fulltext_xml(pmcid, session, xml_dir)
                if xml_path:
                    extracted = _extract_text_from_xml(xml_path)
                    if extracted.strip():
                        safe = pmcid.replace("/", "_").replace(":", "_")
                        text_path = os.path.join(text_dir, f"{safe}.txt")
                        try:
                            with open(
                                text_path, "w", encoding="utf-8", errors="replace"
                            ) as f:
                                f.write(extracted)
                        except Exception as e:
                            logger.warning(
                                f"Failed to write extracted XML text for {pmcid}: {e}"
                            )
                            text_path = None

                if not text_path:
                    pdf_path = pdf_path or _fetch_fulltext_pdf(
                        pmcid, session, pdf_dir
                    )
                    if pdf_path:
                        extracted = _extract_text_from_pdf(pdf_path)
                        if extracted.strip():
                            safe = pmcid.replace("/", "_").replace(":", "_")
                            text_path = os.path.join(text_dir, f"{safe}.txt")
                            try:
                                with open(
                                    text_path,
                                    "w",
                                    encoding="utf-8",
                                    errors="replace",
                                ) as f:
                                    f.write(extracted)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to write extracted PDF text for {pmcid}: {e}"
                                )
                                text_path = None

            if text_path is None:
                status = "partial" if pdf_path else "failed"
                message = "no text extracted"

            if delete_pdf_after_text and pdf_path and text_path:
                try:
                    os.remove(pdf_path)
                    logger.debug(f"Deleted PDF after extraction: {pdf_path}")
                    pdf_path = None
                except Exception as e:
                    logger.warning(f"Could not delete PDF {pdf_path}: {e}")

            rec = DownloadRecord(
                paper_id=paper_id,
                source=source,
                pmcid=pmcid,
                pdf_path=pdf_path,
                text_path=text_path,
                status=status,
                message=message,
            )
            records.append(rec)
            batch.append(rec)

        if idx % batch_size == 0:
            flush_batch()

        if idx % 50 == 0:
            logger.info(
                f"Collect (download): processed {idx}/{len(paper_items)} papers "
                f"({(idx/len(paper_items))*100:.1f}%)"
            )

    flush_batch()

    out_df = pd.DataFrame(
        [
            {
                "paper_id": r.paper_id,
                "source": r.source,
                "pmcid": r.pmcid,
                "pdf_path": r.pdf_path,
                "text_path": r.text_path,
                "status": r.status,
                "message": r.message,
            }
            for r in records
        ]
    )

    n_ok = (out_df["status"] == "ok").sum() if not out_df.empty else 0
    logger.info(
        f"Collect (download): {n_ok}/{len(out_df)} papers with extracted text "
        f"({data_root})"
    )
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Module 3 (new): Download full texts for all papers discovered by the "
            "search module and write text files + batch manifests for LLM analysis."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Search output CSV/JSON containing query_paper_dois/target_paper_dois.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV path summarizing download/text extraction status.",
    )
    parser.add_argument(
        "--data-root",
        default="/private/groups/corbettlab/gabe/auto_lit_eval_data",
        help="Shared data root with pdf/, text/, llm_queue/, logs/ (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of papers per manifest batch (default: 500).",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap on number of unique papers to process.",
    )
    parser.add_argument(
        "--delete-pdf-after-text",
        action="store_true",
        help="Delete PDFs after successful text extraction to save space.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any previously downloaded files (re-download everything).",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Collect (download) reading search results: {args.input}")
    result = run(
        args.input,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_papers=args.max_papers,
        delete_pdf_after_text=args.delete_pdf_after_text,
        no_cache=args.no_cache,
    )
    result.to_csv(args.output, index=False)
    logger.info(f"Collect (download) wrote summary CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

