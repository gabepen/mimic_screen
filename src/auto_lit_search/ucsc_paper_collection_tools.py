"""
UCSC-specific paper collection helpers.

Small, dependency-light utilities for collection routing in collect.py.
"""

from __future__ import annotations

from typing import Optional
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
    query = (doi or "").strip() or (title or "").strip()
    if not query:
        return None
    try:
        resp = session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query[:150],
                "fields": "openAccessPdf,title,externalIds",
                "limit": 3,
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get("data") or []:
            pdf = (paper.get("openAccessPdf") or {}).get("url")
            if pdf:
                return str(pdf).strip()
    except Exception as e:
        logger.debug(f"Semantic Scholar lookup failed for query={query!r}: {e}")
    return None