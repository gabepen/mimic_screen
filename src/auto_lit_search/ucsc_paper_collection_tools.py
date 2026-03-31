"""
UCSC-specific paper collection helpers.

Small, dependency-light utilities for collection routing in collect.py.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional
from xml.etree import ElementTree as ET

import requests
from loguru import logger


def is_ucsc_email(email: str) -> bool:
    s = (email or "").strip().lower()
    return bool(s) and s.endswith("@ucsc.edu")


def get_unpaywall_pdf_url(
    doi: Optional[str],
    email: str,
    session: requests.Session,
    timeout_s: int = 25,
) -> Optional[str]:
    """
    Return best open PDF URL from Unpaywall for DOI, else None.

    This is an auxiliary DOI coverage path for UCSC email-only mode.
    """
    d = (doi or "").strip()
    if not d:
        return None
    em = (email or "").strip()
    if not em:
        return None

    url = f"https://api.unpaywall.org/v2/{d}"
    try:
        resp = session.get(url, params={"email": em}, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"Unpaywall lookup failed for DOI={d}: {e}")
        return None

    best = (data.get("best_oa_location") or {}).get("url_for_pdf")
    if best:
        return str(best).strip()

    for loc in data.get("oa_locations") or []:
        u = (loc or {}).get("url_for_pdf")
        if u:
            return str(u).strip()

    return None


def get_arxiv_pdf_url(
    doi: Optional[str],
    title: Optional[str],
    session: requests.Session,
    timeout_s: int = 25,
) -> Optional[str]:
    """
    Return arXiv PDF URL if DOI/title maps to an arXiv preprint.
    """
    d = (doi or "").strip()
    t = (title or "").strip()
    if d and "arxiv" in d.lower():
        arxiv_id = d.split("/")[-1].strip()
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if not t:
        return None
    try:
        resp = session.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{t[:100]}"', "start": 0, "max_results": 1},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if not entries:
            return None
        entry_id = entries[0].find(".//{http://www.w3.org/2005/Atom}id")
        if entry_id is None or not entry_id.text:
            return None
        arxiv_id = entry_id.text.split("/")[-1].strip()
        if not arxiv_id:
            return None
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception as e:
        logger.debug(f"arXiv lookup failed for title={t!r} doi={d!r}: {e}")
        return None


def get_semantic_scholar_pdf_url(
    doi: Optional[str],
    title: Optional[str],
    session: requests.Session,
    timeout_s: int = 25,
) -> Optional[str]:
    """
    Return open access PDF URL from Semantic Scholar if available.
    """
    d = (doi or "").strip()
    query = d or (title or "").strip()
    if not query:
        return None

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params: Dict[str, object] = {
        "query": query[:150],
        "fields": "openAccessPdf,title,externalIds",
        "limit": 3,
    }
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                try:
                    delay = min(120.0, float(ra)) if ra else None
                except (TypeError, ValueError):
                    delay = None
                if delay is None:
                    delay = min(60.0, 2.0 ** (attempt + 1))
                logger.debug(
                    f"Semantic Scholar 429 for query={query!r}; sleeping {delay:.1f}s"
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()
            data = resp.json()
            norm_doi = d.lower().strip()
            for paper in data.get("data") or []:
                # If DOI is available from upstream, only accept an exact DOI match
                # from Semantic Scholar to avoid off-topic false positives.
                if norm_doi:
                    ext_ids = paper.get("externalIds") or {}
                    s2_doi = str(ext_ids.get("DOI") or "").lower().strip()
                    if s2_doi != norm_doi:
                        continue
                pdf = (paper.get("openAccessPdf") or {}).get("url")
                if pdf:
                    return str(pdf).strip()
            return None
        except Exception as e:
            logger.debug(
                f"Semantic Scholar lookup failed for query={query!r}: {e}"
            )
            return None

    return None