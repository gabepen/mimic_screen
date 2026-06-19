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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from loguru import logger

try:
    from .ucsc_paper_collection_tools import (
        download_elsevier_article_pdf,
        download_asm_article_pdf,
        download_mdpi_article_pdf,
        download_wiley_tdm_pdf,
        get_arxiv_pdf_url,
        get_elsevier_fulltext_xml,
        get_semantic_scholar_pdf_url,
        get_unpaywall_pdf_url,
        is_asm_primary_doi,
        is_elsevier_primary_doi,
        is_mdpi_primary_doi,
        is_ucsc_email,
        is_wiley_primary_doi,
    )
except ImportError:
    from ucsc_paper_collection_tools import (
        download_elsevier_article_pdf,
        download_asm_article_pdf,
        download_mdpi_article_pdf,
        download_wiley_tdm_pdf,
        get_arxiv_pdf_url,
        get_elsevier_fulltext_xml,
        get_semantic_scholar_pdf_url,
        get_unpaywall_pdf_url,
        is_asm_primary_doi,
        is_elsevier_primary_doi,
        is_mdpi_primary_doi,
        is_ucsc_email,
        is_wiley_primary_doi,
    )


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PMC_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
API_DELAY = 0.35

# PMID from the same Europe PMC search row as pmcid_cache[paper_id]; used for /article/MED/{pmid}/fullText*
_EUROPEPMC_PMID_BY_PAPER_ID: Dict[str, Optional[str]] = {}

# Last Europe PMC `/search` `query` param sent per paper_id (for logs / manifest).
_EUROPEPMC_LAST_SEARCH_QUERY: Dict[str, str] = {}


def _europepmc_note_search_query(
    paper_id: str, search_query: str, cache_lock: Optional[threading.Lock]
) -> None:
    if cache_lock:
        with cache_lock:
            _EUROPEPMC_LAST_SEARCH_QUERY[paper_id] = search_query
    else:
        _EUROPEPMC_LAST_SEARCH_QUERY[paper_id] = search_query


def _europepmc_store_pmid(
    paper_id: str,
    pmid: Optional[str],
    cache_lock: Optional[threading.Lock],
) -> None:
    if cache_lock:
        with cache_lock:
            _EUROPEPMC_PMID_BY_PAPER_ID[paper_id] = pmid
    else:
        _EUROPEPMC_PMID_BY_PAPER_ID[paper_id] = pmid


def _europepmc_get_pmid(
    paper_id: str, cache_lock: Optional[threading.Lock]
) -> Optional[str]:
    if cache_lock:
        with cache_lock:
            return _EUROPEPMC_PMID_BY_PAPER_ID.get(paper_id)
    return _EUROPEPMC_PMID_BY_PAPER_ID.get(paper_id)


def _europepmc_xml_body_is_error(text: str) -> bool:
    t = (text or "").lstrip()
    return t.startswith("<?xml") and "<errorbean>" in t.lower()


def _europepmc_fulltext_urls(pmcid: str, pmid: Optional[str], kind: str) -> List[str]:
    """
    kind is fullTextXML or fullTextPDF.

    Primary URL matches Europe PMC docs / browser examples, e.g.
    https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7096322/fullTextXML
    Then try article/MED and article/PMC fallbacks if needed.
    """
    base = EUROPEPMC_FULLTEXT_BASE
    pid = (pmcid or "").strip()
    if not pid.upper().startswith("PMC"):
        pid = f"PMC{pid}" if pid.isdigit() else pid
    urls: List[str] = [f"{base}/{pid}/{kind}"]
    p = (pmid or "").strip()
    if p.isdigit():
        urls.append(f"{base}/article/MED/{p}/{kind}")
    tail = pid.upper().removeprefix("PMC")
    if tail.isdigit():
        alt = f"{base}/article/PMC/{tail}/{kind}"
        if alt not in urls:
            urls.append(alt)
    return urls

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(hi, int(raw)))
    except ValueError:
        return default


def _phased_download_enabled() -> bool:
    return os.environ.get("COLLECT_PHASED_DOWNLOAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _pmcid_cache_path(output_dir: str) -> str:
    custom = os.environ.get("COLLECT_PMCID_CACHE_PATH", "").strip()
    if custom:
        return custom
    data_root = os.environ.get("DATA_ROOT", "").strip()
    if data_root:
        return os.path.join(data_root, "logs", "pmcid_cache.json")
    # papers/<alignment_id> -> <data_root>/papers/...
    parent = os.path.dirname(os.path.dirname(output_dir))
    return os.path.join(parent, "logs", "pmcid_cache.json")


def load_pmcid_cache(path: str) -> Dict[str, Optional[str]]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "pmcids" in data:
            raw = data.get("pmcids") or {}
        elif isinstance(data, dict):
            raw = data
        else:
            return {}
        return {str(k): (v if v else None) for k, v in raw.items()}
    except Exception as e:
        logger.warning("Could not load PMCID cache {}: {}", path, e)
        return {}


def save_pmcid_cache(path: str, cache: Dict[str, Optional[str]]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pmids = dict(_EUROPEPMC_PMID_BY_PAPER_ID)
        payload = {"pmcids": cache, "pmids": pmids}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning("Could not save PMCID cache {}: {}", path, e)


def _merge_pmcid_cache_from_disk(
    cache: Dict[str, Optional[str]], path: str
) -> None:
    loaded = load_pmcid_cache(path)
    for k, v in loaded.items():
        if k not in cache:
            cache[k] = v
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pmids = data.get("pmids") if isinstance(data, dict) else None
        if isinstance(pmids, dict):
            for k, v in pmids.items():
                if k not in _EUROPEPMC_PMID_BY_PAPER_ID:
                    _EUROPEPMC_PMID_BY_PAPER_ID[k] = v
    except Exception:
        pass


_ASM_THROTTLE_SECONDS = _env_float("ASM_THROTTLE_SECONDS", 6.0)

# Minimum spacing between outbound calls per channel (shared across threads).
_DEFAULT_THROTTLE_INTERVALS_S: Dict[str, float] = {
    "pmc_oa_s3": 0.05,
    "europe_pmc": 0.35,
    "elsevier": 0.55,
    "wiley": 5.0,
    "mdpi": 0.55,
    # ASM Journals: recommended crawler throttling rates (see comments in chat).
    # Set override explicitly with ASM_THROTTLE_SECONDS (default: 6.0s).
    "asm": _ASM_THROTTLE_SECONDS,
    "unpaywall": 0.55,
    "arxiv": 3.0,
    "semantic_scholar": 3.5,
    "pdf_url": 0.2,
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
    disable_pmc_oa_s3: bool = False


def _env_disable_pmc_oa_s3() -> bool:
    return os.environ.get("COLLECT_DISABLE_PMC_OA_S3", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


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
    disable_pmc_oa_s3: bool = False,
) -> DownloadRecord:
    paper_id, source = item
    session = requests.Session()
    # Avoid broken HTTP(S)_PROXY env on some cluster nodes; Europe PMC may still
    # work while publisher APIs (Wiley TDM, Unpaywall redirects, Elsevier) fail.
    session.trust_env = False
    session.headers.setdefault("User-Agent", "auto_lit_search/0.1 (collect)")
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
        disable_pmc_oa_s3=disable_pmc_oa_s3,
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
            "pmc_oa_s3": {
                "attempted": False,
                "success": False,
                "artifact": None,
                "error": None,
            },
            "europe_pmc": {"attempted": True, "success": False, "artifact": None, "error": None},
            "elsevier": {
                "attempted": bool(doi and is_elsevier_primary_doi(doi)),
                "success": False,
                "artifact": None,
                "error": None,
            },
            "wiley": {
                "attempted": False,
                "success": False,
                "artifact": None,
                "error": None,
            },
            "mdpi": {
                "attempted": False,
                "success": False,
                "artifact": None,
                "error": None,
            },
            "asm": {
                "attempted": bool(doi and is_asm_primary_doi(doi)),
                "success": False,
                "artifact": None,
                "error": None,
            },
            "unpaywall": {"attempted": False, "success": False, "artifact": None, "error": None},
            "arxiv": {"attempted": bool(doi or title), "success": False, "artifact": None, "error": None},
            "semantic_scholar": {"attempted": bool(doi or title), "success": False, "artifact": None, "error": None},
        }

        pmcid, epmc_search_query = _resolve_to_pmcid(
            paper_id,
            context.session,
            context.pmcid_cache,
            throttle=context.throttle,
            cache_lock=context.cache_lock,
        )
        epmc_pmid = _europepmc_get_pmid(paper_id, context.cache_lock)
        xml_text = ""
        xml_path: Optional[str] = None
        pdf_path: Optional[str] = None
        text_path: Optional[str] = None
        selected_text_source_hint: Optional[str] = None
        pmc_oa_s3_meta: Dict[str, Any] = {}
        safe = _doi_file_stem(paper_id, doi, source)

        if not context.no_cache:
            candidate_pdf = os.path.join(context.pdf_dir, f"{safe}.pdf")
            candidate_text = os.path.join(context.text_dir, f"{safe}.txt")
            if os.path.exists(candidate_pdf):
                pdf_path = candidate_pdf
            if (
                os.path.exists(candidate_text)
                and os.path.getsize(candidate_text) > 0
            ):
                text_path = candidate_text
                selected_text_source_hint = "cached_text"
            # Channel-suffixed PDFs (e.g. __unpaywall) from prior runs
            if not pdf_path and os.path.isdir(context.pdf_dir):
                prefix = safe + "__"
                for fname in os.listdir(context.pdf_dir):
                    if fname.startswith(prefix) and fname.lower().endswith(".pdf"):
                        pdf_path = os.path.join(context.pdf_dir, fname)
                        break

        if pmcid and not text_path and not context.disable_pmc_oa_s3:
            source_attempts["pmc_oa_s3"]["attempted"] = True
            s3_out = _attempt_pmc_oa_s3(pmcid, context, safe)
            source_attempts["pmc_oa_s3"] = s3_out["attempt"]
            pmc_oa_s3_meta = s3_out.get("metadata_info") or {}
            if s3_out.get("text_path"):
                text_path = s3_out["text_path"]
            if s3_out.get("xml_path"):
                xml_path = s3_out["xml_path"]
            if s3_out.get("xml_text"):
                xml_text = s3_out["xml_text"]
            if s3_out.get("pdf_path"):
                pdf_path = s3_out["pdf_path"]
            if s3_out.get("selected_text_source"):
                selected_text_source_hint = s3_out["selected_text_source"]

        if pmcid and not text_path:
            xml_path = _fetch_fulltext_xml(
                pmcid,
                context.session,
                context.xml_dir,
                file_stem=safe,
                throttle=context.throttle,
                pmid=epmc_pmid,
            )
            if xml_path:
                xml_text = _extract_text_from_xml(xml_path)
            source_attempts["europe_pmc"]["success"] = bool(xml_path)
            source_attempts["europe_pmc"]["artifact"] = (
                "xml" if xml_path else None
            )

        if pmcid and not text_path and not pdf_path:
            epmc_pdf = _fetch_fulltext_pdf(
                pmcid,
                context.session,
                context.pdf_dir,
                file_stem=safe,
                throttle=context.throttle,
                pmid=epmc_pmid,
            )
            if epmc_pdf:
                pdf_path = epmc_pdf
                source_attempts["europe_pmc"]["success"] = True
                source_attempts["europe_pmc"]["artifact"] = "pdf"

        # Elsevier Article API: ScienceDirect (10.1016) and ASBMB journals (10.1074).
        if doi and not text_path and is_elsevier_primary_doi(doi):
            if context.throttle:
                context.throttle.wait("elsevier")
            elsevier_raw = get_elsevier_fulltext_xml(doi, context.session)
            if elsevier_raw:
                els_plain = _extract_text_from_xml_string(elsevier_raw)
                els_stats = _xml_quality_stats(els_plain)
                epmc_stats = _xml_quality_stats(xml_text)
                if els_stats["quality_pass"] and (
                    not epmc_stats["quality_pass"]
                    or len(els_plain) > len(xml_text or "") * 1.05
                ):
                    xml_text = els_plain
                    try:
                        os.makedirs(context.xml_dir, exist_ok=True)
                        els_out = os.path.join(
                            context.xml_dir, f"{safe}__elsevier.xml"
                        )
                        with open(
                            els_out, "w", encoding="utf-8", errors="replace"
                        ) as xf:
                            xf.write(elsevier_raw)
                        xml_path = els_out
                    except Exception as ex:
                        logger.debug(
                            f"Could not save Elsevier XML for {safe}: {ex}"
                        )
                    source_attempts["elsevier"]["success"] = True
                    source_attempts["elsevier"]["artifact"] = "xml"

        # MDPI before Unpaywall: OA PDFs are on mdpi.com; Unpaywall often omits them.
        if doi and is_mdpi_primary_doi(doi) and not text_path and not pdf_path:
            source_attempts["mdpi"]["attempted"] = True
            if context.throttle:
                context.throttle.wait("mdpi")
            mdpi_pdf = download_mdpi_article_pdf(
                doi,
                context.session,
                context.pdf_dir,
                f"{safe}__mdpi",
            )
            if mdpi_pdf:
                source_attempts["mdpi"]["success"] = True
                source_attempts["mdpi"]["artifact"] = "pdf"
                pdf_path = pdf_path or mdpi_pdf

        # ASM before Unpaywall: ASM PDFs are often reachable via doi-derived URL patterns.
        if doi and is_asm_primary_doi(doi) and not text_path and not pdf_path:
            if context.throttle:
                context.throttle.wait("asm")
            asm_pdf = download_asm_article_pdf(
                doi,
                context.session,
                context.pdf_dir,
                f"{safe}__asm",
            )
            if asm_pdf:
                source_attempts["asm"]["success"] = True
                source_attempts["asm"]["artifact"] = "pdf"
                pdf_path = pdf_path or asm_pdf

        unpaywall_url = None
        if doi and not text_path and not pdf_path:
            source_attempts["unpaywall"]["attempted"] = True
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

        if doi and is_wiley_primary_doi(doi) and not text_path and not pdf_path:
            source_attempts["wiley"]["attempted"] = True
            if context.throttle:
                context.throttle.wait("wiley")
            wiley_pdf = download_wiley_tdm_pdf(
                doi,
                context.session,
                context.pdf_dir,
                f"{safe}__wiley_tdm",
            )
            if wiley_pdf:
                source_attempts["wiley"]["success"] = True
                source_attempts["wiley"]["artifact"] = "pdf"
                pdf_path = pdf_path or wiley_pdf

        if doi and is_elsevier_primary_doi(doi) and not text_path and not pdf_path:
            if context.throttle:
                context.throttle.wait("elsevier")
            el_pdf = download_elsevier_article_pdf(
                doi,
                context.session,
                context.pdf_dir,
                f"{safe}__elsevier",
            )
            if el_pdf:
                source_attempts["elsevier"]["success"] = True
                source_attempts["elsevier"]["artifact"] = "pdf"
                pdf_path = pdf_path or el_pdf

        arxiv_url = None
        if not text_path and (doi or title):
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
        elif not text_path and (doi or title):
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
        if text_path:
            selected_text_source = selected_text_source_hint or "cached_text"
            pdf_docling_required = False
        else:
            if pdf_path:
                selected_text_source = "docling_pdf"
                pdf_docling_required = True
            elif xml_pass and xml_text.strip():
                text_path = os.path.join(context.text_dir, f"{safe}.txt")
                with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(xml_text)
                selected_text_source = selected_text_source_hint or "xml"

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

        pdf_path = _finalize_pdf_text_paths(
            pdf_path,
            text_path,
            pdf_docling_required,
            delete_pdf_after_text=context.delete_pdf_after_text,
        )

        retrieval_queries: Dict[str, Any] = {
            "paper_id": paper_id,
            "source_tag": source,
            "doi": doi,
            "title": title,
            "pmc_oa_s3": pmc_oa_s3_meta,
            "europe_pmc": {
                "search_endpoint": EUROPEPMC_SEARCH_URL,
                "search_query": epmc_search_query,
                "pmcid_resolved": pmcid,
                "pmid_from_search": epmc_pmid,
                "fulltext_xml_url_try_order": (
                    _europepmc_fulltext_urls(pmcid, epmc_pmid, "fullTextXML")
                    if pmcid
                    else []
                ),
                "fulltext_pdf_url_try_order": (
                    _europepmc_fulltext_urls(pmcid, epmc_pmid, "fullTextPDF")
                    if pmcid
                    else []
                ),
            },
            "identifiers_used_elsewhere": {
                "elsevier_xml_pdf": doi,
                "unpaywall": doi,
                "wiley_tdm": doi,
                "mdpi": doi,
                "asm": doi,
                "arxiv": {"doi": doi, "title": title},
                "semantic_scholar": {"doi": doi, "title": title},
            },
            "resolved_pdf_urls": {
                "unpaywall": unpaywall_url,
                "arxiv": arxiv_url,
                "semantic_scholar": s2_url,
            },
        }
        logger.info(
            f"paper_text_retrieval paper_id={paper_id!r} doi={doi!r} "
            f"europe_pmc_search_query={epmc_search_query!r} pmcid={pmcid!r} "
            f"pmid={epmc_pmid!r} status={status!r} selected_text_source={selected_text_source!r} "
            f"xml_quality_score={xml_stats.get('quality_score', 0.0)!r} "
            f"pdf_docling_required={pdf_docling_required!r}"
        )
        _channels = ",".join(successful_sources) if successful_sources else "none"
        _primary = selected_text_source or "none"
        logger.info(
            f"paper_download_outcome paper_id={paper_id!r} source={source!r} "
            f"outcome={status!r} primary_source={_primary!r} channels={_channels!r}"
        )

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
                "retrieval_queries": retrieval_queries,
                "file_stem": safe,
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


def _doi_file_stem(paper_id: str, doi: Optional[str], source: str) -> str:
    """On-disk stem: DOI with / -> _ plus alignment role (query|target)."""
    id_base = (doi or _extract_doi_from_identifier(paper_id) or paper_id or "").strip()
    return f"{id_base.replace('/', '_')}__{source.strip()}"


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
            "metadata_line_ratio": 1.0,
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
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    line_count = len(lines)
    metadata_lines = sum(1 for ln in lines if _line_is_xml_noise(ln))
    metadata_ratio = min(1.0, metadata_lines / max(1, line_count))
    noise_ratio = min(1.0, noise_hits / max(1, section_hits + noise_hits))
    score = (
        min(1.0, char_count / 15000.0) * 0.4
        + min(1.0, section_hits / 4.0) * 0.3
        + max(0.0, 1.0 - noise_ratio) * 0.15
        + max(0.0, 1.0 - metadata_ratio) * 0.15
    )
    quality_pass = (
        (char_count >= 2500)
        and (section_hits >= 2)
        and (noise_ratio < 0.7)
        and (metadata_ratio < 0.45)
    )
    return {
        "char_count": char_count,
        "line_count": line_count,
        "section_hits": section_hits,
        "noise_ratio": round(noise_ratio, 4),
        "metadata_line_ratio": round(metadata_ratio, 4),
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
    # Some records use only this category.
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


def _epmc_search_query_for_paper_id(paper_id: str) -> str:
    pid = paper_id.strip()
    u = pid.upper()
    if u.startswith("PMC:"):
        pmcid_norm = u[4:].strip()
        if not pmcid_norm.upper().startswith("PMC"):
            pmcid_norm = f"PMC{pmcid_norm}"
        return pmcid_norm
    if u.startswith("PMC"):
        return u
    if u.startswith("PMID:"):
        ext_id = u[5:].strip()
    else:
        ext_id = pid
    if ext_id.startswith("10."):
        return f"DOI:{ext_id}"
    return f"EXT_ID:{ext_id}"


def _normalize_pmcid_value(pmcid: str) -> str:
    p = str(pmcid or "").strip()
    if not p:
        return ""
    u = p.upper()
    if not u.startswith("PMC"):
        return f"PMC{p}" if p.isdigit() else p
    return u


def _pmcid_from_epmc_record(rec: Dict[str, Any]) -> Optional[Tuple[str, Optional[str]]]:
    pmcid = rec.get("pmcid")
    if not pmcid:
        return None
    pubtypes = (rec.get("pubTypeList") or {}).get("pubType") or []
    if pubtypes and _pubtype_is_excluded(pubtypes):
        return None
    if pubtypes and not _pubtypes_look_like_research(pubtypes):
        return None
    pmcid_norm = _normalize_pmcid_value(str(pmcid))
    pmid_raw = rec.get("pmid") or rec.get("id")
    pmid_str = str(pmid_raw).strip() if pmid_raw is not None else ""
    pmid_out: Optional[str] = pmid_str if pmid_str.isdigit() else None
    return pmcid_norm, pmid_out


def _store_pmcid_resolution(
    paper_id: str,
    pmcid: Optional[str],
    pmid: Optional[str],
    search_query: str,
    cache: Dict[str, Optional[str]],
    cache_lock: Optional[threading.Lock],
) -> None:
    if cache_lock:
        with cache_lock:
            cache[paper_id] = pmcid
    else:
        cache[paper_id] = pmcid
    _europepmc_note_search_query(paper_id, search_query, cache_lock)
    _europepmc_store_pmid(paper_id, pmid, cache_lock)


def _pmcid_idconv_batch(
    paper_ids: List[str],
    session: requests.Session,
    collector_email: str,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """PMC ID Converter: up to 200 IDs per request."""
    out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if not paper_ids:
        return out
    params: Dict[str, str] = {
        "ids": ",".join(paper_ids),
        "format": "json",
        "tool": "auto_lit_search",
    }
    if collector_email:
        params["email"] = collector_email
    try:
        resp = session.get(PMC_IDCONV_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("PMC ID converter batch failed: {}", e)
        return out
    for rec in data.get("records") or []:
        if not isinstance(rec, dict):
            continue
        req_id = str(rec.get("requested-id") or rec.get("doi") or "").strip()
        if not req_id:
            continue
        err = str(rec.get("errmsg") or rec.get("error") or "").strip()
        if err:
            out[req_id] = (None, None)
            continue
        pmcid_raw = rec.get("pmcid")
        if not pmcid_raw:
            out[req_id] = (None, None)
            continue
        pmcid = _normalize_pmcid_value(str(pmcid_raw))
        pmid_raw = rec.get("pmid")
        pmid = str(pmid_raw).strip() if pmid_raw is not None else ""
        pmid_out: Optional[str] = pmid if pmid.isdigit() else None
        out[req_id] = (pmcid, pmid_out)
    return out


def _pmcid_epmc_or_batch(
    paper_ids: List[str],
    session: requests.Session,
    cache: Dict[str, Optional[str]],
    throttle: Optional[CollectThrottle],
    cache_lock: Optional[threading.Lock],
) -> int:
    """Resolve a batch via one Europe PMC OR search. Returns count stored."""
    if not paper_ids:
        return 0
    doi_to_pids: Dict[str, List[str]] = {}
    query_to_pids: Dict[str, List[str]] = {}
    for pid in paper_ids:
        sq = _epmc_search_query_for_paper_id(pid)
        query_to_pids.setdefault(sq, []).append(pid)
        doi = _extract_doi_from_identifier(pid)
        if doi:
            doi_to_pids.setdefault(doi.lower(), []).append(pid)

    or_parts = list(query_to_pids.keys())
    or_query = " OR ".join(or_parts)
    _europepmc_note_search_query(paper_ids[0], or_query, cache_lock)
    if throttle:
        throttle.wait("europe_pmc")
    else:
        time.sleep(API_DELAY)
    try:
        resp = session.get(
            EUROPEPMC_SEARCH_URL,
            params={
                "query": or_query,
                "format": "json",
                "resultType": "core",
                "pageSize": min(1000, max(25, len(paper_ids) * 3)),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("Europe PMC OR batch failed (n={}): {}", len(paper_ids), e)
        return 0

    stored = 0
    assigned: Set[str] = set()
    for rec in (data.get("resultList") or {}).get("result") or []:
        if not isinstance(rec, dict):
            continue
        picked = _pmcid_from_epmc_record(rec)
        if not picked:
            continue
        pmcid, pmid = picked
        targets: List[str] = []
        rec_doi = str(rec.get("doi") or "").strip().lower()
        if rec_doi and rec_doi in doi_to_pids:
            targets.extend(doi_to_pids[rec_doi])
        rec_pmcid = _normalize_pmcid_value(str(rec.get("pmcid") or ""))
        sq = rec_pmcid if rec_pmcid else ""
        if sq and sq in query_to_pids:
            targets.extend(query_to_pids[sq])
        for pid in dict.fromkeys(targets):
            if pid in assigned:
                continue
            _store_pmcid_resolution(
                pid, pmcid, pmid, or_query, cache, cache_lock
            )
            assigned.add(pid)
            stored += 1

    for pid in paper_ids:
        if pid in assigned:
            continue
        if pid not in cache:
            _store_pmcid_resolution(pid, None, None, or_query, cache, cache_lock)
    return stored


def batch_resolve_pmcids(
    paper_ids: List[str],
    cache: Dict[str, Optional[str]],
    session: requests.Session,
    throttle: Optional[CollectThrottle],
    cache_lock: Optional[threading.Lock],
    *,
    collector_email: str = "",
    resolve_workers: int = 8,
    epmc_batch_size: int = 40,
    idconv_batch_size: int = 200,
) -> int:
    """
    Resolve many paper identifiers to PMCIDs before per-paper download.

    Uses PMC ID Converter (200 IDs/request) then Europe PMC OR batches.
    """
    pending: List[str] = []
    for pid in paper_ids:
        norm = _normalize_paper_id(pid)
        if not norm:
            continue
        if norm in cache:
            continue
        u = norm.upper()
        if u.startswith("PMC"):
            pmcid = _normalize_pmcid_value(norm)
            _store_pmcid_resolution(
                norm, pmcid, None, "direct_pmcid", cache, cache_lock
            )
            continue
        pending.append(norm)

    if not pending:
        return 0

    pending = list(dict.fromkeys(pending))
    newly = 0
    idconv_batch_size = max(1, min(200, int(idconv_batch_size)))
    epmc_batch_size = max(1, min(80, int(epmc_batch_size)))
    resolve_workers = max(1, min(32, int(resolve_workers)))

    idconv_chunks = [
        pending[i : i + idconv_batch_size]
        for i in range(0, len(pending), idconv_batch_size)
    ]
    for chunk in idconv_chunks:
        still_need = [p for p in chunk if p not in cache]
        if not still_need:
            continue
        resolved = _pmcid_idconv_batch(still_need, session, collector_email)
        for pid in still_need:
            hit = resolved.get(pid)
            if hit is None:
                doi = _extract_doi_from_identifier(pid)
                if doi:
                    hit = resolved.get(doi)
            if not hit:
                continue
            pmcid, pmid = hit
            if pmcid:
                _store_pmcid_resolution(
                    pid, pmcid, pmid, "pmc_idconv", cache, cache_lock
                )
                newly += 1

    remaining = [p for p in pending if p not in cache]
    epmc_chunks = [
        remaining[i : i + epmc_batch_size]
        for i in range(0, len(remaining), epmc_batch_size)
    ]
    if epmc_chunks:
        workers = min(resolve_workers, len(epmc_chunks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _pmcid_epmc_or_batch,
                    chunk,
                    session,
                    cache,
                    throttle,
                    cache_lock,
                )
                for chunk in epmc_chunks
            ]
            for fut in as_completed(futs):
                newly += int(fut.result() or 0)

    # Final pass: mark unresolved
    for pid in pending:
        if pid not in cache:
            _store_pmcid_resolution(pid, None, None, "unresolved", cache, cache_lock)

    return newly


def _resolve_to_pmcid(
    paper_id: str,
    session: requests.Session,
    cache: Dict[str, Optional[str]],
    delay: float = API_DELAY,
    throttle: Optional[CollectThrottle] = None,
    cache_lock: Optional[threading.Lock] = None,
) -> Tuple[Optional[str], str]:
    """
    Resolve an arbitrary paper identifier to a Europe PMC PMCID, if possible.

    Returns (pmcid_or_None, europe_pmc_search_query) where the query is the
    exact `query` param sent to /search, or ``pmcid_cache_hit`` if served from cache.

    Supported inputs:
        - PMC123456 or PMC:123456
        - PMID:123456 or bare numeric PMID
        - DOI (10.xxxx/...)
        - Other IDs resolvable via EXT_ID search.
    """
    if cache_lock:
        with cache_lock:
            if paper_id in cache:
                qnote = _EUROPEPMC_LAST_SEARCH_QUERY.get(
                    paper_id, "pmcid_cache_hit"
                )
                return cache[paper_id], qnote
    elif paper_id in cache:
        return cache[paper_id], _EUROPEPMC_LAST_SEARCH_QUERY.get(
            paper_id, "pmcid_cache_hit"
        )

    pid = paper_id.strip()
    u = pid.upper()
    search_query: str

    # If the input is already a PMC id, still resolve it through the Europe PMC
    # search API so we can filter out notices (e.g. expression of concern).
    if u.startswith("PMC:"):
        pmcid_norm = u[4:].strip()
        pmcid_norm = (
            f"PMC{pmcid_norm}"
            if not pmcid_norm.upper().startswith("PMC")
            else pmcid_norm
        )
        search_query = pmcid_norm
    elif u.startswith("PMC"):
        search_query = u
    else:
        # Europe PMC: DOI: finds PubMed-linked records; EXT_ID:10.xxx often returns zero hits.
        if u.startswith("PMID:"):
            ext_id = u[5:].strip()
        else:
            ext_id = pid
        if ext_id.startswith("10."):
            search_query = f"DOI:{ext_id}"
        else:
            search_query = f"EXT_ID:{ext_id}"

    _europepmc_note_search_query(paper_id, search_query, cache_lock)

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
        logger.debug(f"Europe PMC lookup failed for {paper_id}: {e}")
        if cache_lock:
            with cache_lock:
                cache[paper_id] = None
        else:
            cache[paper_id] = None
        _europepmc_store_pmid(paper_id, None, cache_lock)
        return None, search_query

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
        pmid_raw = rec.get("pmid") or rec.get("id")
        pmid_str = str(pmid_raw).strip() if pmid_raw is not None else ""
        pmid_out: Optional[str] = pmid_str if pmid_str.isdigit() else None
        if cache_lock:
            with cache_lock:
                cache[paper_id] = pmcid
        else:
            cache[paper_id] = pmcid
        _europepmc_store_pmid(paper_id, pmid_out, cache_lock)
        return pmcid, search_query

    if cache_lock:
        with cache_lock:
            cache[paper_id] = None
    else:
        cache[paper_id] = None
    _europepmc_store_pmid(paper_id, None, cache_lock)
    return None, search_query


def _fetch_fulltext_pdf(
    pmcid: str,
    session: requests.Session,
    pdf_dir: str,
    timeout: int = 120,
    file_stem: Optional[str] = None,
    throttle: Optional[CollectThrottle] = None,
    pmid: Optional[str] = None,
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

    for url in _europepmc_fulltext_urls(pmcid, pmid, "fullTextPDF"):
        logger.debug(f"Fetching PDF for {pmcid} -> {url}")
        if throttle:
            throttle.wait("europe_pmc")
        else:
            time.sleep(API_DELAY)
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code != 200:
                continue
            content_type = (resp.headers.get("Content-Type") or "").lower()
            content_bytes = resp.content or b""
            is_pdf_by_magic = content_bytes.startswith(b"%PDF")
            if (("pdf" not in content_type) and not is_pdf_by_magic):
                continue
            with open(out_path, "wb") as f:
                f.write(content_bytes)
            return out_path
        except Exception as e:
            logger.debug(f"PDF fetch failed {url}: {e}")
    return None


def _fetch_fulltext_xml(
    pmcid: str,
    session: requests.Session,
    xml_dir: str,
    timeout: int = 120,
    file_stem: Optional[str] = None,
    throttle: Optional[CollectThrottle] = None,
    pmid: Optional[str] = None,
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

    for url in _europepmc_fulltext_urls(pmcid, pmid, "fullTextXML"):
        logger.debug(f"Fetching XML for {pmcid} -> {url}")
        if throttle:
            throttle.wait("europe_pmc")
        else:
            time.sleep(API_DELAY)
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code != 200:
                continue
            text = resp.text or ""
            if not text or len(text) < 500 or _europepmc_xml_body_is_error(text):
                continue
            with open(out_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
            return out_path
        except Exception as e:
            logger.debug(f"XML fetch failed {url}: {e}")
    return None


def _normalize_extracted_line(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _line_is_xml_noise(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    lower = s.lower()
    if lower.startswith(("http://", "https://", "doi:", "pmc-")):
        return True
    if lower in {"full-text", "text/xml", "journal", "author", "serial"}:
        return True
    compact = lower.replace(" ", "")
    if compact.isdigit() and len(compact) <= 8:
        return True
    if compact.startswith("1-s2.0-") or compact.startswith("2-s2.0-"):
        return True
    return False


def _extract_text_from_xml_root(root: Any) -> str:
    """Extract likely narrative prose from XML while filtering metadata-heavy lines."""
    prefer_tags = {
        "abstract",
        "body",
        "sec",
        "title",
        "p",
        "paragraph",
        "ce:para",
        "ce:section-title",
    }
    lines: List[str] = []
    seen: Set[str] = set()

    def _tag_name(elem: Any) -> str:
        tag = str(getattr(elem, "tag", "") or "")
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        return tag.lower()

    def _push(text: str) -> None:
        norm = _normalize_extracted_line(text)
        if not norm or _line_is_xml_noise(norm):
            return
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        lines.append(norm)

    preferred_nodes = [e for e in root.iter() if _tag_name(e) in prefer_tags]
    source_nodes = preferred_nodes if preferred_nodes else list(root.iter())
    for elem in source_nodes:
        for chunk in elem.itertext():
            _push(chunk)
    return "\n".join(lines)


def _extract_text_from_xml(xml_path: str) -> str:
    """Extract narrative text from XML file content."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return ""
    return _extract_text_from_xml_root(root)


def _extract_text_from_xml_string(xml: str) -> str:
    """Extract narrative text from an in-memory XML document."""
    import xml.etree.ElementTree as ET

    if not (xml or "").strip():
        return ""
    try:
        root = ET.fromstring(xml.strip())
    except Exception:
        return ""
    return _extract_text_from_xml_root(root)


def _attempt_pmc_oa_s3(
    pmcid: str,
    context: CollectionContext,
    file_stem: str,
) -> Dict[str, Any]:
    """
  Try PMC OA S3 bucket first (metadata + .txt/.xml, optional .pdf).

  Returns dict with xml_text, xml_path, pdf_path, text_path, attempt, metadata_info.
    """
    from auto_lit_search.pmc_oa_s3 import (
        download_pmc_oa_fulltext,
        download_pmc_oa_pdf,
        fetch_pmc_oa_metadata,
        metadata_https_url,
    )

    attempt: Dict[str, Any] = {
        "attempted": True,
        "success": False,
        "artifact": None,
        "error": None,
    }
    metadata_info: Dict[str, Any] = {"metadata_urls_tried": []}
    out: Dict[str, Any] = {
        "xml_text": "",
        "xml_path": None,
        "pdf_path": None,
        "text_path": None,
        "attempt": attempt,
        "selected_text_source": None,
        "metadata_info": metadata_info,
    }
    if context.throttle:
        context.throttle.wait("pmc_oa_s3")
    try:
        meta = fetch_pmc_oa_metadata(pmcid, session=context.session)
    except Exception as e:
        attempt["error"] = str(e)
        return out
    if meta is None:
        for v in range(1, 4):
            metadata_info["metadata_urls_tried"].append(
                metadata_https_url(pmcid, v)
            )
        attempt["error"] = "not_in_oa_bucket"
        return out

    metadata_info["metadata_url"] = metadata_https_url(meta.pmcid, meta.version)
    metadata_info["text_url"] = meta.text_https_url
    metadata_info["xml_url"] = meta.xml_https_url
    metadata_info["pdf_url"] = meta.pdf_https_url

    body, artifact = download_pmc_oa_fulltext(meta, session=context.session)
    if body and artifact:
        if artifact == "txt":
            plain = body
            stats = _xml_quality_stats(plain)
            if stats["quality_pass"]:
                txt_path = os.path.join(context.text_dir, f"{file_stem}.txt")
                with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(plain)
                out["text_path"] = txt_path
                out["xml_text"] = plain
                out["selected_text_source"] = "pmc_oa_s3_txt"
                attempt["success"] = True
                attempt["artifact"] = "txt"
            else:
                out["xml_text"] = plain
                attempt["error"] = "txt_below_quality_threshold"
        elif artifact == "xml":
            plain = _extract_text_from_xml_string(body)
            stats = _xml_quality_stats(plain)
            try:
                os.makedirs(context.xml_dir, exist_ok=True)
                xml_out = os.path.join(
                    context.xml_dir, f"{file_stem}__pmc_oa_s3.xml"
                )
                with open(xml_out, "w", encoding="utf-8", errors="replace") as xf:
                    xf.write(body)
                out["xml_path"] = xml_out
            except Exception as ex:
                logger.debug(f"Could not save PMC OA S3 XML for {file_stem}: {ex}")
            out["xml_text"] = plain
            if stats["quality_pass"]:
                txt_path = os.path.join(context.text_dir, f"{file_stem}.txt")
                with open(txt_path, "w", encoding="utf-8", errors="replace") as tf:
                    tf.write(plain)
                out["text_path"] = txt_path
                attempt["success"] = True
                attempt["artifact"] = "xml"
                out["selected_text_source"] = "pmc_oa_s3_xml"
            else:
                attempt["error"] = "xml_below_quality_threshold"

    if context.force_pdfs and not out["pdf_path"] and not out["text_path"]:
        pdf_bytes = download_pmc_oa_pdf(meta, session=context.session)
        if pdf_bytes:
            os.makedirs(context.pdf_dir, exist_ok=True)
            pdf_out = os.path.join(context.pdf_dir, f"{file_stem}__pmc_oa_s3.pdf")
            with open(pdf_out, "wb") as pf:
                pf.write(pdf_bytes)
            out["pdf_path"] = pdf_out
            if not attempt["success"]:
                attempt["success"] = True
                attempt["artifact"] = "pdf"

    return out


def _record_has_usable_text(rec: Optional[DownloadRecord]) -> bool:
    if rec is None or not rec.text_path:
        return False
    return os.path.isfile(rec.text_path) and os.path.getsize(rec.text_path) > 0


def _finalize_pdf_text_paths(
    pdf_path: Optional[str],
    text_path: Optional[str],
    pdf_docling_required: bool,
    *,
    delete_pdf_after_text: bool = False,
) -> Optional[str]:
    """Drop PDF from record when usable text exists; optionally delete file from disk."""
    if not text_path or pdf_docling_required or not pdf_path:
        return pdf_path
    if delete_pdf_after_text:
        try:
            os.remove(pdf_path)
        except Exception as e:
            logger.warning(f"Could not delete PDF {pdf_path}: {e}")
    return None


def _record_needs_publisher_fallback(rec: Optional[DownloadRecord]) -> bool:
    """Skip publisher APIs when S3/cache already produced text or a PDF for Docling."""
    if _record_has_usable_text(rec):
        return False
    if rec is not None and rec.pdf_path and os.path.isfile(rec.pdf_path):
        return False
    return True


def _collect_s3_record(
    paper_id: str,
    source: str,
    context: CollectionContext,
) -> Optional[DownloadRecord]:
    """S3-only fetch using a pre-populated PMCID cache. Returns record if text/pdf obtained."""
    doi = _extract_doi_from_identifier(paper_id)
    safe = _doi_file_stem(paper_id, doi, source)
    text_path: Optional[str] = None
    pdf_path: Optional[str] = None
    selected_text_source = "none"

    if not context.no_cache:
        candidate_text = os.path.join(context.text_dir, f"{safe}.txt")
        if (
            os.path.exists(candidate_text)
            and os.path.getsize(candidate_text) > 0
        ):
            text_path = candidate_text
            selected_text_source = "cached_text"
        candidate_pdf = os.path.join(context.pdf_dir, f"{safe}.pdf")
        if os.path.exists(candidate_pdf) and os.path.getsize(candidate_pdf) > 0:
            pdf_path = candidate_pdf

    norm_pid = _normalize_paper_id(paper_id) or paper_id
    pmcid = context.pmcid_cache.get(norm_pid)
    if pmcid is None and norm_pid != paper_id:
        pmcid = context.pmcid_cache.get(paper_id)
    epmc_pmid = _europepmc_get_pmid(paper_id, context.cache_lock)
    epmc_search_query = _EUROPEPMC_LAST_SEARCH_QUERY.get(
        paper_id, "pmcid_cache_hit"
    )
    pmc_oa_s3_meta: Dict[str, Any] = {}
    source_attempts: Dict[str, Dict[str, Any]] = {
        "pmc_oa_s3": {
            "attempted": False,
            "success": False,
            "artifact": None,
            "error": None,
        },
        "europe_pmc": {
            "attempted": False,
            "success": False,
            "artifact": None,
            "error": None,
        },
    }

    if text_path:
        xml_stats = _xml_quality_stats("")
        status = "ok"
        return DownloadRecord(
            paper_id=paper_id,
            source=source,
            pmcid=pmcid,
            pdf_path=None,
            text_path=text_path,
            status=status,
            message=None,
            details={
                "source_attempts": source_attempts,
                "selected_text_source": selected_text_source,
                "pdf_docling_required": False,
                "doi": doi,
                "file_stem": safe,
                "xml_stats": xml_stats,
            },
        )

    if not pmcid or context.disable_pmc_oa_s3:
        return None

    s3_out = _attempt_pmc_oa_s3(pmcid, context, safe)
    source_attempts["pmc_oa_s3"] = s3_out["attempt"]
    pmc_oa_s3_meta = s3_out.get("metadata_info") or {}
    if s3_out.get("text_path"):
        text_path = s3_out["text_path"]
        selected_text_source = s3_out.get("selected_text_source") or "pmc_oa_s3_txt"
    if s3_out.get("pdf_path"):
        pdf_path = s3_out["pdf_path"]

    if not text_path and not pdf_path:
        return None

    xml_text = s3_out.get("xml_text") or ""
    if xml_text:
        plain_for_stats = xml_text
    elif text_path:
        try:
            with open(text_path, "r", encoding="utf-8", errors="replace") as tf:
                plain_for_stats = tf.read()
        except Exception:
            plain_for_stats = ""
    else:
        plain_for_stats = ""
    xml_stats = _xml_quality_stats(plain_for_stats)
    pdf_docling_required = bool(not text_path and pdf_path)
    status = "ok" if text_path else "partial"
    message = None if text_path else "xml below quality threshold; docling conversion required"
    if text_path and pdf_path and not pdf_docling_required:
        pdf_path = _finalize_pdf_text_paths(
            pdf_path,
            text_path,
            pdf_docling_required,
            delete_pdf_after_text=context.delete_pdf_after_text,
        )
    successful_sources = sorted(
        [k for k, v in source_attempts.items() if v.get("success")]
    )
    logger.info(
        f"paper_text_retrieval paper_id={paper_id!r} doi={doi!r} "
        f"europe_pmc_search_query={epmc_search_query!r} pmcid={pmcid!r} "
        f"pmid={epmc_pmid!r} status={status!r} selected_text_source={selected_text_source!r} "
        f"xml_quality_score={xml_stats.get('quality_score', 0.0)!r} "
        f"pdf_docling_required={pdf_docling_required!r}"
    )
    _channels = ",".join(successful_sources) if successful_sources else "none"
    logger.info(
        f"paper_download_outcome paper_id={paper_id!r} source={source!r} "
        f"outcome={status!r} primary_source={selected_text_source!r} channels={_channels!r}"
    )
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
            "xml_stats": xml_stats,
            "selected_text_source": selected_text_source,
            "pdf_docling_required": pdf_docling_required,
            "doi": doi,
            "file_stem": safe,
            "retrieval_queries": {
                "pmc_oa_s3": pmc_oa_s3_meta,
                "europe_pmc": {
                    "search_query": epmc_search_query,
                    "pmcid_resolved": pmcid,
                    "pmid_from_search": epmc_pmid,
                },
            },
        },
    )


def _download_papers_phased(
    paper_ids_with_source: List[Tuple[str, str]],
    output_dir: str,
    session: requests.Session,
    pmcid_cache: Dict[str, Optional[str]],
    provider: BaseCollectionProvider,
    *,
    no_cache: bool,
    force_pdfs: bool,
    prefer_pdf_text: bool,
    delete_pdf_after_text: bool,
    disable_semantic_scholar: bool,
    collector_email: str,
    max_workers: int,
) -> List[DownloadRecord]:
    pdf_dir = os.path.join(output_dir, "pdf")
    xml_dir = os.path.join(output_dir, "text_xml")
    text_dir = output_dir
    cache_path = _pmcid_cache_path(output_dir)
    _merge_pmcid_cache_from_disk(pmcid_cache, cache_path)

    throttle_intervals = dict(_DEFAULT_THROTTLE_INTERVALS_S)
    throttle_intervals["pmc_oa_s3"] = _env_float("PMC_OA_S3_THROTTLE_SECONDS", 0.0)
    throttle = CollectThrottle(throttle_intervals)

    s3_workers = _env_int("COLLECT_S3_WORKERS", 32, 1, 64)
    fallback_workers = _env_int("COLLECT_FALLBACK_WORKERS", max_workers, 1, 16)
    resolve_workers = _env_int("COLLECT_PMCID_RESOLVE_WORKERS", 8, 1, 32)
    epmc_batch_size = _env_int("COLLECT_PMCID_EPMC_BATCH_SIZE", 40, 5, 80)
    idconv_batch_size = _env_int("COLLECT_PMCID_IDCONV_BATCH_SIZE", 200, 10, 200)

    unique_ids = list(
        dict.fromkeys(
            _normalize_paper_id(pid) or pid
            for pid, _ in paper_ids_with_source
        )
    )
    cache_lock = threading.Lock()
    t0 = time.monotonic()
    newly = batch_resolve_pmcids(
        unique_ids,
        pmcid_cache,
        session,
        throttle,
        cache_lock,
        collector_email=collector_email,
        resolve_workers=resolve_workers,
        epmc_batch_size=epmc_batch_size,
        idconv_batch_size=idconv_batch_size,
    )
    save_pmcid_cache(cache_path, pmcid_cache)
    resolved_n = sum(1 for pid in unique_ids if pmcid_cache.get(pid))
    logger.info(
        "Collect phased: PMCID batch {} new, {}/{} with PMCID ({:.1f}s)",
        newly,
        resolved_n,
        len(unique_ids),
        time.monotonic() - t0,
    )

    n = len(paper_ids_with_source)
    records: List[Optional[DownloadRecord]] = [None] * n
    need_fallback_set: Set[int] = set()

    def _s3_context() -> CollectionContext:
        s = requests.Session()
        s.trust_env = False
        s.headers.setdefault("User-Agent", "auto_lit_search/0.1 (collect-s3)")
        return CollectionContext(
            session=s,
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
            disable_pmc_oa_s3=False,
        )

    def _s3_job(i: int) -> Tuple[int, Optional[DownloadRecord]]:
        pid, src = paper_ids_with_source[i]
        return i, _collect_s3_record(pid, src, _s3_context())

    t_s3 = time.monotonic()
    s3_done = 0
    with ThreadPoolExecutor(max_workers=s3_workers) as ex:
        futs = {ex.submit(_s3_job, i): i for i in range(n)}
        for fut in as_completed(futs):
            i, rec = fut.result()
            records[i] = rec
            s3_done += 1
            if _record_needs_publisher_fallback(rec):
                need_fallback_set.add(i)
            if s3_done % 50 == 0:
                logger.info(
                    "Collect (S3 phase): finished {}/{} papers ({:.1f}%)",
                    s3_done,
                    n,
                    100.0 * s3_done / n,
                )
    s3_hits = sum(1 for r in records if _record_has_usable_text(r))
    logger.info(
        "Collect phased: S3 phase {} text hits in {:.1f}s (workers={})",
        s3_hits,
        time.monotonic() - t_s3,
        s3_workers,
    )

    need_fallback = sorted(need_fallback_set)
    if need_fallback:
        fallback_worker = partial(
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
            disable_pmc_oa_s3=False,
        )
        t_fb = time.monotonic()
        fb_done = 0
        with ThreadPoolExecutor(max_workers=fallback_workers) as ex:
            futs = {
                ex.submit(fallback_worker, paper_ids_with_source[i]): i
                for i in need_fallback
            }
            for fut in as_completed(futs):
                i = futs[fut]
                records[i] = fut.result()
                fb_done += 1
                if fb_done % 50 == 0:
                    logger.info(
                        "Collect (fallback): finished {}/{} papers",
                        fb_done,
                        len(need_fallback),
                    )
        logger.info(
            "Collect phased: fallback {} papers in {:.1f}s (workers={})",
            len(need_fallback),
            time.monotonic() - t_fb,
            fallback_workers,
        )

    save_pmcid_cache(cache_path, pmcid_cache)
    return [r for r in records if r is not None]


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
    disable_pmc_oa_s3: bool = False,
) -> List[DownloadRecord]:
    """
    Download full-text for a list of (paper_id, source) into output_dir.
    Writes output_dir/<safe_id>.txt (and optionally pdf/xml subdirs).
    Returns list of DownloadRecord. Use output_dir as papers_dir for GPU API.
    """
    session = session or requests.Session()
    session.trust_env = False
    session.headers.setdefault("User-Agent", "auto_lit_search/0.1 (collect)")
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
    workers = max(1, min(16, int(max_workers)))
    if _env_disable_pmc_oa_s3():
        disable_pmc_oa_s3 = True

    if _phased_download_enabled() and not disable_pmc_oa_s3:
        return _download_papers_phased(
            paper_ids_with_source,
            output_dir,
            session,
            pmcid_cache,
            provider,
            no_cache=no_cache,
            force_pdfs=force_pdfs,
            prefer_pdf_text=prefer_pdf_text,
            delete_pdf_after_text=delete_pdf_after_text,
            disable_semantic_scholar=disable_semantic_scholar,
            collector_email=selected_email,
            max_workers=workers,
        )

    throttle = CollectThrottle()
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
            disable_pmc_oa_s3=disable_pmc_oa_s3,
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
        disable_pmc_oa_s3=disable_pmc_oa_s3,
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
    disable_pmc_oa_s3: bool = False,
    retry_failed_from: Optional[str] = None,
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
    if _env_disable_pmc_oa_s3():
        disable_pmc_oa_s3 = True
    throttle = CollectThrottle()
    cache_lock = threading.Lock() if workers > 1 else None
    logger.info(
        f"Collect (download): max_workers={workers} "
        f"semantic_scholar={'off' if disable_semantic_scholar else 'on'} "
        f"pmc_oa_s3={'off' if disable_pmc_oa_s3 else 'on'}"
    )

    def _load_retry_ids(path: str) -> Set[str]:
        ids: Set[str] = set()
        if not path or not os.path.isfile(path):
            return ids
        if path.lower().endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    pid = str(obj.get("paper_id") or obj.get("doi") or "").strip()
                    if pid:
                        ids.add(pid)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    pid = str(line).strip().split(",")[0].strip()
                    if pid and not pid.startswith("#"):
                        ids.add(pid)
        return ids

    retry_ids: Set[str] = set()
    if retry_failed_from:
        retry_ids = _load_retry_ids(retry_failed_from)
        logger.info(
            f"Collect (download): retry-only mode loaded {len(retry_ids)} ids from {retry_failed_from}"
        )

    # Build unique paper list.
    unique: Dict[str, str] = {}
    for pid, src in _iter_paper_ids_from_search_df(df):
        if retry_ids and pid not in retry_ids:
            continue
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
            disable_pmc_oa_s3=disable_pmc_oa_s3,
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
            disable_pmc_oa_s3=disable_pmc_oa_s3,
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
    n_partial = (out_df["status"] == "partial").sum() if not out_df.empty else 0
    n_failed = (out_df["status"] == "failed").sum() if not out_df.empty else 0
    logger.info(
        f"Collect (download) outcome summary: attempted={len(out_df)} "
        f"ok_text={n_ok} partial_pdf_only={n_partial} failed_no_artifact={n_failed} "
        f"({data_root})"
    )
    checked_sources = [
        "pmc_oa_s3",
        "europe_pmc",
        "elsevier",
        "wiley",
        "mdpi",
        "asm",
        "unpaywall",
        "arxiv",
        "semantic_scholar",
    ]

    for role in ("query", "target"):
        role_rows = out_df[out_df["source"] == role] if not out_df.empty else out_df
        if role_rows.empty:
            continue
        source_attempt_counts: Dict[str, int] = {s: 0 for s in checked_sources}
        source_success_counts: Dict[str, int] = {s: 0 for s in checked_sources}
        xml_pass_n = 0
        docling_required_n = 0
        for _, rr in role_rows.iterrows():
            details = rr.get("details") or {}
            attempts = details.get("source_attempts") or {}
            for sname in checked_sources:
                if (attempts.get(sname) or {}).get("attempted"):
                    source_attempt_counts[sname] += 1
                if (attempts.get(sname) or {}).get("success"):
                    source_success_counts[sname] += 1
            if (details.get("xml_stats") or {}).get("quality_pass"):
                xml_pass_n += 1
            if details.get("pdf_docling_required"):
                docling_required_n += 1
        logger.info(
            f"Collect summary role={role}: n={len(role_rows)} "
            f"checked_sources={checked_sources} "
            f"source_attempt_counts={source_attempt_counts} "
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
            source_attempt_counts: Dict[str, int] = {s: 0 for s in checked_sources}
            source_success_counts: Dict[str, int] = {s: 0 for s in checked_sources}
            xml_pass_n = 0
            docling_required_n = 0
            for _, rr in role_rows.iterrows():
                details = rr.get("details") or {}
                attempts = details.get("source_attempts") or {}
                for sname in checked_sources:
                    if (attempts.get(sname) or {}).get("attempted"):
                        source_attempt_counts[sname] += 1
                    if (attempts.get(sname) or {}).get("success"):
                        source_success_counts[sname] += 1
                if (details.get("xml_stats") or {}).get("quality_pass"):
                    xml_pass_n += 1
                if details.get("pdf_docling_required"):
                    docling_required_n += 1
            summary_by_role[role] = {
                "n_papers": int(len(role_rows)),
                "checked_sources": checked_sources,
                "source_attempt_counts": source_attempt_counts,
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

    try:
        failed_rows = out_df[out_df["status"] == "failed"] if not out_df.empty else out_df
        retry_candidates: List[Dict[str, Any]] = []
        for _, rr in failed_rows.iterrows():
            details = rr.get("details") or {}
            doi = str(details.get("doi") or "").strip()
            paper_id = str(rr.get("paper_id") or "").strip()
            retry_candidates.append(
                {
                    "paper_id": paper_id,
                    "doi": doi,
                    "source": str(rr.get("source") or ""),
                    "status": "failed",
                    "message": str(rr.get("message") or ""),
                }
            )
        retry_jsonl = os.path.join(logs_dir, "collect_failed_retry_candidates.jsonl")
        with open(retry_jsonl, "w", encoding="utf-8") as f:
            for row in retry_candidates:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        retry_ids_path = os.path.join(logs_dir, "collect_failed_retry_ids.txt")
        with open(retry_ids_path, "w", encoding="utf-8") as f:
            for row in retry_candidates:
                pid = row.get("doi") or row.get("paper_id")
                if pid:
                    f.write(str(pid).strip() + "\n")
        logger.info(
            f"Wrote retry candidate logs: {retry_jsonl} and {retry_ids_path} (n={len(retry_candidates)})"
        )
    except Exception as e:
        logger.warning(f"Could not write retry candidate logs: {e}")
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
    parser.add_argument(
        "--retry-failed-from",
        default=None,
        help=(
            "Optional retry-only input (jsonl/txt) from prior run failure outputs, "
            "e.g. logs/collect_failed_retry_candidates.jsonl or logs/collect_failed_retry_ids.txt."
        ),
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
        retry_failed_from=args.retry_failed_from,
    )
    result.to_csv(args.output, index=False)
    logger.info(f"Collect (download) wrote summary CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

