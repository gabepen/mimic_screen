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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

try:
    from .ucsc_paper_collection_tools import (
        get_arxiv_pdf_url,
        get_semantic_scholar_pdf_url,
        get_unpaywall_pdf_url,
        is_ucsc_email,
    )
except ImportError:
    from ucsc_paper_collection_tools import (
        get_arxiv_pdf_url,
        get_semantic_scholar_pdf_url,
        get_unpaywall_pdf_url,
        is_ucsc_email,
    )


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
API_DELAY = 0.35

# Minimum spacing between outbound calls per channel (shared across threads).
_DEFAULT_THROTTLE_INTERVALS_S: Dict[str, float] = {
    "europe_pmc": 0.35,
    "unpaywall": 0.55,
    "arxiv": 3.0,
    "semantic_scholar": 3.5,
    "publisher_pdf": 0.2,
}


class CollectThrottle:
    """
    Per-channel rate limiter: ensures at least `interval` seconds between
    successive waits on the same channel (process-wide for that CollectThrottle).
    """

    def __init__(self, intervals: Optional[Dict[str, float]] = None):
        self._intervals = dict(intervals or _DEFAULT_THROTTLE_INTERVALS_S)
        self._locks: Dict[str, threading.Lock] = {
            k: threading.Lock() for k in self._intervals
        }
        self._next_ok: Dict[str, float] = {k: 0.0 for k in self._intervals}

    def wait(self, channel: str) -> None:
        interval = self._intervals.get(channel)
        if interval is None or interval <= 0:
            return
        lock = self._locks.get(channel)
        if lock is None:
            return
        with lock:
            now = time.monotonic()
            wait_s = self._next_ok[channel] - now
            if wait_s > 0:
                time.sleep(wait_s)
            self._next_ok[channel] = time.monotonic() + interval


@dataclass
class DownloadRecord:
    paper_id: str
    source: str
    pmcid: Optional[str]
    pdf_path: Optional[str]
    text_path: Optional[str]
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class CollectionContext:
    session: requests.Session
    pmcid_cache: Dict[str, Optional[str]]
    pdf_dir: str
    text_dir: str
    xml_dir: str
    no_cache: bool = False
    delete_pdf_after_text: bool = False
    force_pdfs: bool = True
    prefer_pdf_text: bool = True
    throttle: Optional[CollectThrottle] = None
    cache_lock: Optional[threading.Lock] = None
    disable_semantic_scholar: bool = False


class BaseCollectionProvider:
    def resolve_and_fetch(
        self, paper_id: str, source: str, context: CollectionContext
    ) -> DownloadRecord:
        raise NotImplementedError


def _collect_single_record(
    item: Tuple[str, str],
    provider: BaseCollectionProvider,
    pdf_dir: str,
    text_dir: str,
    xml_dir: str,
    pmcid_cache: Dict[str, Optional[str]],
    no_cache: bool,
    delete_pdf_after_text: bool,
    force_pdfs: bool,
    prefer_pdf_text: bool,
    throttle: CollectThrottle,
    cache_lock: Optional[threading.Lock],
    disable_semantic_scholar: bool,
) -> DownloadRecord:
    paper_id, source = item
    session = requests.Session()
    ctx = CollectionContext(
        session=session,
        pmcid_cache=pmcid_cache,
        pdf_dir=pdf_dir,
        text_dir=text_dir,
        xml_dir=xml_dir,
        no_cache=no_cache,
        delete_pdf_after_text=delete_pdf_after_text,
        force_pdfs=force_pdfs,
        prefer_pdf_text=prefer_pdf_text,
        throttle=throttle,
        cache_lock=cache_lock,
        disable_semantic_scholar=disable_semantic_scholar,
    )
    return provider.resolve_and_fetch(paper_id, source, ctx)


class NotImplementedScopeProvider(BaseCollectionProvider):
    def __init__(self, reason: str):
        self.reason = reason

    def resolve_and_fetch(
        self, paper_id: str, source: str, context: CollectionContext
    ) -> DownloadRecord:
        return DownloadRecord(
            paper_id=paper_id,
            source=source,
            pmcid=None,
            pdf_path=None,
            text_path=None,
            status="skipped",
            message=self.reason,
        )


class UCSCEmailOnlyProvider(BaseCollectionProvider):
    def __init__(self, collector_email: str):
        self.collector_email = collector_email

    def resolve_and_fetch(
        self, paper_id: str, source: str, context: CollectionContext
    ) -> DownloadRecord:
        doi = _extract_doi_from_identifier(paper_id)
        title = _extract_title_from_identifier(paper_id)
        source_attempts: Dict[str, Dict[str, Any]] = {
            "europe_pmc": {"attempted": True, "success": False, "artifact": None, "error": None},
            "unpaywall": {"attempted": bool(doi), "success": False, "artifact": None, "error": None},
            "arxiv": {"attempted": bool(doi or title), "success": False, "artifact": None, "error": None},
            "semantic_scholar": {"attempted": bool(doi or title), "success": False, "artifact": None, "error": None},
        }

        pmcid = _resolve_to_pmcid(
            paper_id,
            context.session,
            context.pmcid_cache,
            throttle=context.throttle,
            cache_lock=context.cache_lock,
        )
        xml_text = ""
        xml_path: Optional[str] = None
        pdf_path: Optional[str] = None
        text_path: Optional[str] = None
        safe = (
            f"{(pmcid or paper_id)}__{source}"
            .replace("/", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

        if not context.no_cache:
            candidate_pdf = os.path.join(context.pdf_dir, f"{safe}.pdf")
            candidate_text = os.path.join(context.text_dir, f"{safe}.txt")
            if os.path.exists(candidate_pdf):
                pdf_path = candidate_pdf
            if os.path.exists(candidate_text) and os.path.getsize(candidate_text) > 0:
                text_path = candidate_text

        if pmcid and not text_path:
            xml_path = _fetch_fulltext_xml(
                pmcid,
                context.session,
                context.xml_dir,
                file_stem=safe,
                throttle=context.throttle,
            )
            if xml_path:
                xml_text = _extract_text_from_xml(xml_path)
            source_attempts["europe_pmc"]["success"] = bool(xml_path)
            source_attempts["europe_pmc"]["artifact"] = (
                "xml" if xml_path else None
            )

        if pmcid and (context.force_pdfs or not text_path):
            epmc_pdf = _fetch_fulltext_pdf(
                pmcid,
                context.session,
                context.pdf_dir,
                file_stem=safe,
                throttle=context.throttle,
            )
            if epmc_pdf:
                pdf_path = epmc_pdf
                source_attempts["europe_pmc"]["success"] = True
                source_attempts["europe_pmc"]["artifact"] = "pdf"

        unpaywall_url = None
        if doi:
            if context.throttle:
                context.throttle.wait("unpaywall")
            unpaywall_url = get_unpaywall_pdf_url(
                doi, self.collector_email, context.session
            )
        if unpaywall_url:
            source_attempts["unpaywall"]["artifact"] = "url"
            up_pdf = _download_pdf_from_url(
                unpaywall_url,
                context.session,
                context.pdf_dir,
                f"{safe}__unpaywall",
                throttle=context.throttle,
            )
            if up_pdf:
                source_attempts["unpaywall"]["success"] = True
                source_attempts["unpaywall"]["artifact"] = "pdf"
                pdf_path = pdf_path or up_pdf

        arxiv_url = None
        if doi or title:
            if context.throttle:
                context.throttle.wait("arxiv")
            arxiv_url = get_arxiv_pdf_url(doi, title, context.session)
        if arxiv_url:
            source_attempts["arxiv"]["artifact"] = "url"
            arxiv_pdf = _download_pdf_from_url(
                arxiv_url,
                context.session,
                context.pdf_dir,
                f"{safe}__arxiv",
                throttle=context.throttle,
            )
            if arxiv_pdf:
                source_attempts["arxiv"]["success"] = True
                source_attempts["arxiv"]["artifact"] = "pdf"
                pdf_path = pdf_path or arxiv_pdf

        s2_url = None
        if context.disable_semantic_scholar:
            source_attempts["semantic_scholar"]["attempted"] = False
        elif doi or title:
            if context.throttle:
                context.throttle.wait("semantic_scholar")
            s2_url = get_semantic_scholar_pdf_url(doi, title, context.session)
        if s2_url:
            source_attempts["semantic_scholar"]["artifact"] = "url"
            s2_pdf = _download_pdf_from_url(
                s2_url,
                context.session,
                context.pdf_dir,
                f"{safe}__semantic_scholar",
                throttle=context.throttle,
            )
            if s2_pdf:
                source_attempts["semantic_scholar"]["success"] = True
                source_attempts["semantic_scholar"]["artifact"] = "pdf"
                pdf_path = pdf_path or s2_pdf

        xml_stats = _xml_quality_stats(xml_text)
        xml_pass = bool(xml_stats["quality_pass"])
        selected_text_source = "none"
        pdf_docling_required = False

        if xml_pass and xml_text.strip():
            text_path = os.path.join(context.text_dir, f"{safe}.txt")
            with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(xml_text)
            selected_text_source = "xml"
        elif pdf_path:
            # XML didn't pass minimum quality; leave conversion for Docling stage.
            selected_text_source = "docling_pdf"
            pdf_docling_required = True
        elif text_path:
            selected_text_source = "cached_text"

        if text_path is None and pdf_path and not pdf_docling_required:
            extracted = _extract_text_from_pdf(pdf_path)
            if extracted.strip():
                text_path = os.path.join(context.text_dir, f"{safe}.txt")
                with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(extracted)
                selected_text_source = "pdf_extract"

        status = "ok" if text_path else ("partial" if pdf_path else "failed")
        successful_sources = sorted(
            [k for k, v in source_attempts.items() if v.get("success")]
        )
        overlap_pairs: List[str] = []
        for i in range(len(successful_sources)):
            for j in range(i + 1, len(successful_sources)):
                overlap_pairs.append(
                    f"{successful_sources[i]}__{successful_sources[j]}"
                )
        message = None if text_path else "no text extracted"
        if pdf_docling_required:
            message = "xml below quality threshold; docling conversion required"

        if context.delete_pdf_after_text and pdf_path and text_path:
            try:
                os.remove(pdf_path)
                pdf_path = None
            except Exception as e:
                logger.warning(f"Could not delete PDF {pdf_path}: {e}")

        return DownloadRecord(
            paper_id=paper_id,
            source=source,
            pmcid=pmcid,
            pdf_path=pdf_path,
            text_path=text_path,
            status=status,
            message=message,
            details={
                "source_attempts": source_attempts,
                "successful_sources": successful_sources,
                "n_successful_sources": len(successful_sources),
                "source_overlap_pairs": overlap_pairs,
                "xml_stats": xml_stats,
                "xml_path": xml_path,
                "selected_text_source": selected_text_source,
                "pdf_docling_required": pdf_docling_required,
                "doi": doi,
                "unpaywall_url": unpaywall_url,
                "arxiv_url": arxiv_url,
                "semantic_scholar_url": s2_url,
            },
        )


def _extract_doi_from_identifier(paper_id: str) -> Optional[str]:
    pid = (paper_id or "").strip()
    if not pid:
        return None
    if pid.upper().startswith("DOI:"):
        return pid[4:].strip() or None
    if pid.startswith("10."):
        return pid
    return None


def _extract_title_from_identifier(paper_id: str) -> Optional[str]:
    # Placeholder: collect currently receives IDs only from search output.
    return None


def _download_pdf_from_url(
    pdf_url: str,
    session: requests.Session,
    pdf_dir: str,
    file_stem: str,
    timeout: int = 120,
    throttle: Optional[CollectThrottle] = None,
    request_headers: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        if throttle:
            throttle.wait("pdf_url")
        headers: Dict[str, str] = {"Connection": "close"}
        if request_headers:
            headers.update(request_headers)
        resp = session.get(pdf_url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        content = resp.content or b""
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
            return None
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path
    except Exception as e:
        logger.debug(f"PDF download failed from url={pdf_url!r}: {e}")
        return None


def _xml_quality_stats(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "char_count": 0,
            "line_count": 0,
            "section_hits": 0,
            "noise_ratio": 1.0,
            "quality_score": 0.0,
            "quality_pass": False,
        }
    lower = cleaned.lower()
    section_keywords = [
        "abstract",
        "introduction",
        "method",
        "result",
        "discussion",
        "conclusion",
    ]
    noise_keywords = [
        "expression of concern",
        "retraction",
        "copyright",
        "rights reserved",
    ]
    section_hits = sum(1 for k in section_keywords if k in lower)
    noise_hits = sum(1 for k in noise_keywords if k in lower)
    char_count = len(cleaned)
    line_count = cleaned.count("\n") + 1
    noise_ratio = min(1.0, noise_hits / max(1, section_hits + noise_hits))
    score = (
        min(1.0, char_count / 15000.0) * 0.5
        + min(1.0, section_hits / 4.0) * 0.35
        + max(0.0, 1.0 - noise_ratio) * 0.15
    )
    quality_pass = (char_count >= 2500) and (section_hits >= 2) and (noise_ratio < 0.7)
    return {
        "char_count": char_count,
        "line_count": line_count,
        "section_hits": section_hits,
        "noise_ratio": round(noise_ratio, 4),
        "quality_score": round(score, 4),
        "quality_pass": bool(quality_pass),
    }


def _build_collection_provider(
    collection_org: str,
    auth_scope: str,
    collector_email: Optional[str],
) -> BaseCollectionProvider:
    org = (collection_org or "ucsc").strip().lower()
    scope = (auth_scope or "email_only").strip().lower()
    if scope == "email_password":
        return NotImplementedScopeProvider(
            "auth_scope=email_password not implemented yet"
        )
    if org == "ucsc":
        if not collector_email:
            raise ValueError(
                "collector_email is required for UCSC email_only collection mode"
            )
        if not is_ucsc_email(collector_email):
            logger.warning(
                f"collector_email={collector_email!r} is not @ucsc.edu; continuing in UCSC mode"
            )
        return UCSCEmailOnlyProvider(collector_email=collector_email)
    return NotImplementedScopeProvider(f"collection_org={org!r} not implemented")


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
    # Non-research program/meeting records.
    "meeting-report",
    "conference-abstract",
    "proceedings",
]

_ALLOWED_RESEARCH_PUBTYPE_SUBSTRINGS: List[str] = [
    # Most research articles on Europe PMC.
    "research-article",
    # Some publishers show only this category.
    "journal-article",
]


def _pubtypes_look_like_research(pubtypes: List[Any]) -> bool:
    normed: List[str] = []
    for pt in pubtypes:
        s = str(pt).strip().lower().replace(" ", "-")
        if s:
            normed.append(s)
    if not normed:
        return False
    return any(any(allowed in s for allowed in _ALLOWED_RESEARCH_PUBTYPE_SUBSTRINGS) for s in normed)


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
    throttle: Optional[CollectThrottle] = None,
    cache_lock: Optional[threading.Lock] = None,
) -> Optional[str]:
    """
    Resolve an arbitrary paper identifier to a Europe PMC PMCID, if possible.

    Supported inputs:
        - PMC123456 or PMC:123456
        - PMID:123456 or bare numeric PMID
        - DOI (10.xxxx/...)
        - Other IDs resolvable via EXT_ID search.
    """
    if cache_lock:
        with cache_lock:
            if paper_id in cache:
                return cache[paper_id]
    elif paper_id in cache:
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

    if throttle:
        throttle.wait("europe_pmc")
    else:
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
        if cache_lock:
            with cache_lock:
                cache[paper_id] = None
        else:
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

        # If we have pubType information, require it to look like a research article.
        # This filters out meeting reports / abstracts programs which otherwise
        # produce "abstract list" text that doesn't help model reasoning.
        if pubtypes and not _pubtypes_look_like_research(pubtypes):
            logger.debug(f"Skipping non-research (pubType not article-like) pubType={pubtypes} for {paper_id} -> {pmcid}")
            continue

        pmcid = str(pmcid).strip()
        if not pmcid.upper().startswith("PMC"):
            pmcid = f"PMC{pmcid}"
        if cache_lock:
            with cache_lock:
                cache[paper_id] = pmcid
        else:
            cache[paper_id] = pmcid
        return pmcid

    if cache_lock:
        with cache_lock:
            cache[paper_id] = None
    else:
        cache[paper_id] = None
    return None


def _fetch_fulltext_pdf(
    pmcid: str,
    session: requests.Session,
    pdf_dir: str,
    timeout: int = 120,
    file_stem: Optional[str] = None,
    throttle: Optional[CollectThrottle] = None,
) -> Optional[str]:
    """
    Download full-text PDF for a given PMCID from Europe PMC, if available.

    Saves to pdf_dir and returns local path, or None if not available.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    stem = file_stem or pmcid.replace("/", "_").replace(":", "_")
    out_path = os.path.join(pdf_dir, f"{stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.debug(f"PDF cache hit: {out_path}")
        return out_path

    url = f"{EUROPEPMC_FULLTEXT_BASE}/{pmcid}/fullTextPDF"
    logger.debug(f"Fetching PDF for {pmcid} -> {url}")
    if throttle:
        throttle.wait("europe_pmc")
    else:
        time.sleep(API_DELAY)
    try:
        resp = session.get(url, timeout=timeout)
        # Some responses may be HTML if PDF is not available.
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        # Europe PMC sometimes returns PDFs with a non-standard/misleading
        # Content-Type header, so also detect by magic header.
        content_bytes = resp.content or b""
        is_pdf_by_magic = content_bytes.startswith(b"%PDF")
        if (("pdf" not in content_type) and not is_pdf_by_magic):
            logger.debug(
                f"PDF not available for {pmcid} (content-type={content_type!r})"
            )
            return None
        with open(out_path, "wb") as f:
            f.write(content_bytes)
        return out_path
    except Exception as e:
        logger.debug(f"PDF fetch failed for {pmcid}: {e}")
        return None


def _fetch_fulltext_xml(
    pmcid: str,
    session: requests.Session,
    xml_dir: str,
    timeout: int = 120,
    file_stem: Optional[str] = None,
    throttle: Optional[CollectThrottle] = None,
) -> Optional[str]:
    """
    Download full-text XML for a given PMCID from Europe PMC, if available.

    Saves to xml_dir and returns local path, or None if not available.
    """
    os.makedirs(xml_dir, exist_ok=True)
    stem = file_stem or pmcid.replace("/", "_").replace(":", "_")
    out_path = os.path.join(xml_dir, f"{stem}.xml")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        logger.debug(f"XML cache hit: {out_path}")
        return out_path

    url = f"{EUROPEPMC_FULLTEXT_BASE}/{pmcid}/fullTextXML"
    logger.debug(f"Fetching XML for {pmcid} -> {url}")
    if throttle:
        throttle.wait("europe_pmc")
    else:
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
    prefer_pdf_text: bool = False,
    collection_org: str = "ucsc",
    auth_scope: str = "email_only",
    collector_email: Optional[str] = None,
    delete_pdf_after_text: bool = False,
    max_workers: int = 2,
    disable_semantic_scholar: bool = False,
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

    env_org = os.environ.get("COLLECTION_ORG", "").strip()
    env_scope = os.environ.get("COLLECTION_AUTH_SCOPE", "").strip()
    env_email = os.environ.get("COLLECTOR_EMAIL", "").strip()
    selected_org = env_org or collection_org
    selected_scope = env_scope or auth_scope
    selected_email = env_email or (collector_email or "")

    provider = _build_collection_provider(
        collection_org=selected_org,
        auth_scope=selected_scope,
        collector_email=selected_email or None,
    )
    throttle = CollectThrottle()
    workers = max(1, int(max_workers))
    cache_lock = threading.Lock() if workers > 1 else None

    if workers <= 1:
        context = CollectionContext(
            session=session,
            pmcid_cache=pmcid_cache,
            pdf_dir=pdf_dir,
            text_dir=text_dir,
            xml_dir=xml_dir,
            no_cache=no_cache,
            delete_pdf_after_text=delete_pdf_after_text,
            force_pdfs=force_pdfs,
            prefer_pdf_text=prefer_pdf_text,
            throttle=throttle,
            cache_lock=cache_lock,
            disable_semantic_scholar=disable_semantic_scholar,
        )
        return [
            provider.resolve_and_fetch(pid, src, context)
            for pid, src in paper_ids_with_source
        ]

    worker = partial(
        _collect_single_record,
        provider=provider,
        pdf_dir=pdf_dir,
        text_dir=text_dir,
        xml_dir=xml_dir,
        pmcid_cache=pmcid_cache,
        no_cache=no_cache,
        delete_pdf_after_text=delete_pdf_after_text,
        force_pdfs=force_pdfs,
        prefer_pdf_text=prefer_pdf_text,
        throttle=throttle,
        cache_lock=cache_lock,
        disable_semantic_scholar=disable_semantic_scholar,
    )
    n = len(paper_ids_with_source)
    records: List[Optional[DownloadRecord]] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_i = {
            ex.submit(worker, paper_ids_with_source[i]): i for i in range(n)
        }
        done = 0
        for fut in as_completed(future_to_i):
            i = future_to_i[fut]
            records[i] = fut.result()
            done += 1
            if done % 50 == 0:
                logger.info(
                    f"Collect (download dir): finished {done}/{n} papers "
                    f"({100.0 * done / n:.1f}%)"
                )
    return [r for r in records if r is not None]


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
    collection_org: str = "ucsc",
    auth_scope: str = "email_only",
    collector_email: Optional[str] = None,
    max_workers: int = 2,
    disable_semantic_scholar: bool = False,
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
                "details",
            ]
        )

    pmcid_cache: Dict[str, Optional[str]] = {}
    env_org = os.environ.get("COLLECTION_ORG", "").strip()
    env_scope = os.environ.get("COLLECTION_AUTH_SCOPE", "").strip()
    env_email = os.environ.get("COLLECTOR_EMAIL", "").strip()
    selected_org = env_org or collection_org
    selected_scope = env_scope or auth_scope
    selected_email = env_email or (collector_email or "")
    provider = _build_collection_provider(
        collection_org=selected_org,
        auth_scope=selected_scope,
        collector_email=selected_email or None,
    )
    ew = os.environ.get("COLLECT_MAX_WORKERS", "").strip()
    if ew.isdigit():
        max_workers = max(1, min(16, int(ew)))
    workers = max(1, int(max_workers))
    ess = os.environ.get("COLLECT_DISABLE_SEMANTIC_SCHOLAR", "").strip().lower()
    if ess in ("1", "true", "yes"):
        disable_semantic_scholar = True
    throttle = CollectThrottle()
    cache_lock = threading.Lock() if workers > 1 else None
    logger.info(
        f"Collect (download): max_workers={workers} "
        f"semantic_scholar={'off' if disable_semantic_scholar else 'on'}"
    )

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
                                "details": rec.details or {},
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

    def _log_paper_eval(rec: DownloadRecord) -> None:
        d = rec.details or {}
        xml_stats = d.get("xml_stats") or {}
        logger.debug(
            json.dumps(
                {
                    "event": "paper_eval",
                    "paper_id": rec.paper_id,
                    "role": rec.source,
                    "status": rec.status,
                    "selected_text_source": d.get("selected_text_source"),
                    "pdf_docling_required": d.get("pdf_docling_required"),
                    "successful_sources": d.get("successful_sources", []),
                    "xml_stats": xml_stats,
                },
                ensure_ascii=False,
            )
        )

    n_items = len(paper_items)
    if workers <= 1:
        session = requests.Session()
        context = CollectionContext(
            session=session,
            pmcid_cache=pmcid_cache,
            pdf_dir=pdf_dir,
            text_dir=text_dir,
            xml_dir=xml_dir,
            no_cache=no_cache,
            delete_pdf_after_text=delete_pdf_after_text,
            force_pdfs=True,
            prefer_pdf_text=True,
            throttle=throttle,
            cache_lock=cache_lock,
            disable_semantic_scholar=disable_semantic_scholar,
        )
        for idx, (paper_id, source) in enumerate(paper_items, start=1):
            rec = provider.resolve_and_fetch(paper_id, source, context)
            _log_paper_eval(rec)
            records.append(rec)
            batch.append(rec)
            if idx % batch_size == 0:
                flush_batch()
            if idx % 50 == 0:
                logger.info(
                    f"Collect (download): processed {idx}/{n_items} papers "
                    f"({(idx / n_items) * 100:.1f}%)"
                )
    else:
        worker = partial(
            _collect_single_record,
            provider=provider,
            pdf_dir=pdf_dir,
            text_dir=text_dir,
            xml_dir=xml_dir,
            pmcid_cache=pmcid_cache,
            no_cache=no_cache,
            delete_pdf_after_text=delete_pdf_after_text,
            force_pdfs=True,
            prefer_pdf_text=True,
            throttle=throttle,
            cache_lock=cache_lock,
            disable_semantic_scholar=disable_semantic_scholar,
        )
        slot: List[Optional[DownloadRecord]] = [None] * n_items
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_i = {ex.submit(worker, paper_items[i]): i for i in range(n_items)}
            done = 0
            for fut in as_completed(future_to_i):
                i = future_to_i[fut]
                rec = fut.result()
                slot[i] = rec
                _log_paper_eval(rec)
                done += 1
                if done % 50 == 0:
                    logger.info(
                        f"Collect (download): processed {done}/{n_items} papers "
                        f"({(done / n_items) * 100:.1f}%)"
                    )
        records = [slot[i] for i in range(n_items) if slot[i] is not None]
        for rec in records:
            batch.append(rec)
            if len(batch) >= batch_size:
                flush_batch()

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
                "details": r.details or {},
            }
            for r in records
        ]
    )

    n_ok = (out_df["status"] == "ok").sum() if not out_df.empty else 0
    logger.info(
        f"Collect (download): {n_ok}/{len(out_df)} papers with extracted text "
        f"({data_root})"
    )
    for role in ("query", "target"):
        role_rows = out_df[out_df["source"] == role] if not out_df.empty else out_df
        if role_rows.empty:
            continue
        source_success_counts: Dict[str, int] = {
            "europe_pmc": 0,
            "unpaywall": 0,
            "arxiv": 0,
            "semantic_scholar": 0,
        }
        xml_pass_n = 0
        docling_required_n = 0
        for _, rr in role_rows.iterrows():
            details = rr.get("details") or {}
            attempts = details.get("source_attempts") or {}
            for sname in source_success_counts:
                if (attempts.get(sname) or {}).get("success"):
                    source_success_counts[sname] += 1
            if (details.get("xml_stats") or {}).get("quality_pass"):
                xml_pass_n += 1
            if details.get("pdf_docling_required"):
                docling_required_n += 1
        logger.info(
            f"Collect summary role={role}: n={len(role_rows)} "
            f"source_success={source_success_counts} "
            f"xml_quality_pass={xml_pass_n} "
            f"docling_required={docling_required_n}"
        )
    try:
        summary_by_role: Dict[str, Any] = {}
        for role in ("query", "target"):
            role_rows = out_df[out_df["source"] == role] if not out_df.empty else out_df
            if role_rows.empty:
                continue
            source_success_counts: Dict[str, int] = {
                "europe_pmc": 0,
                "unpaywall": 0,
                "arxiv": 0,
                "semantic_scholar": 0,
            }
            xml_pass_n = 0
            docling_required_n = 0
            for _, rr in role_rows.iterrows():
                details = rr.get("details") or {}
                attempts = details.get("source_attempts") or {}
                for sname in source_success_counts:
                    if (attempts.get(sname) or {}).get("success"):
                        source_success_counts[sname] += 1
                if (details.get("xml_stats") or {}).get("quality_pass"):
                    xml_pass_n += 1
                if details.get("pdf_docling_required"):
                    docling_required_n += 1
            summary_by_role[role] = {
                "n_papers": int(len(role_rows)),
                "source_success_counts": source_success_counts,
                "xml_quality_pass_n": int(xml_pass_n),
                "docling_required_n": int(docling_required_n),
            }
        summary_path = os.path.join(logs_dir, "collect_source_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_by_role, f, indent=2)
        logger.info(f"Wrote source evaluation summary: {summary_path}")
    except Exception as e:
        logger.warning(f"Could not write source evaluation summary: {e}")
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
    parser.add_argument(
        "--collection-org",
        default="ucsc",
        help="Collection organization routing key (default: ucsc).",
    )
    parser.add_argument(
        "--auth-scope",
        default="email_only",
        choices=["email_only", "email_password"],
        help="Authentication scope for collection tools (default: email_only).",
    )
    parser.add_argument(
        "--collector-email",
        default=None,
        help="Collector email identity used by org tools (or env COLLECTOR_EMAIL).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel download threads (1=sequential). Capped at 16. Env COLLECT_MAX_WORKERS overrides.",
    )
    parser.add_argument(
        "--disable-semantic-scholar",
        action="store_true",
        help="Skip Semantic Scholar lookups (reduces 429s). Env COLLECT_DISABLE_SEMANTIC_SCHOLAR=1 also sets this.",
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
        collection_org=args.collection_org,
        auth_scope=args.auth_scope,
        collector_email=args.collector_email,
        max_workers=max(1, args.max_workers),
        disable_semantic_scholar=args.disable_semantic_scholar,
    )
    result.to_csv(args.output, index=False)
    logger.info(f"Collect (download) wrote summary CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

