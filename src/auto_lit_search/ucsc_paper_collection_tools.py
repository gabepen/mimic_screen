"""
UCSC-specific paper collection helpers.

Small, dependency-light utilities for collection routing in collect.py.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote, unquote, urlparse, urlunparse
from uuid import UUID
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


def _read_elsevier_api_key() -> str:
    return (
        os.environ.get("ELS_API_KEY")
        or os.environ.get("ELSEVIER_API_KEY")
        or ""
    ).strip()


def _read_elsevier_insttoken() -> Optional[str]:
    t = (
        os.environ.get("ELS_INSTTOKEN")
        or os.environ.get("ELSEVIER_INSTTOKEN")
        or ""
    ).strip()
    return t or None


def get_elsevier_fulltext_xml(
    doi: Optional[str],
    session: requests.Session,
    timeout_s: int = 90,
) -> Optional[str]:
    """
    Full-text article XML from Elsevier Article Retrieval API (ScienceDirect).

    Uses env: ELS_API_KEY or ELSEVIER_API_KEY; optional ELS_INSTTOKEN /
    ELSEVIER_INSTTOKEN for institutional entitlements.

    Returns raw XML string, or None if not configured, not found, or error.
    """
    d = (doi or "").strip()
    api_key = _read_elsevier_api_key()
    if not d or not api_key:
        return None

    enc = quote(d, safe="")
    url = f"https://api.elsevier.com/content/article/doi/{enc}"
    headers = {
        "Accept": "text/xml, application/xml;q=0.9",
        "X-ELS-APIKey": api_key,
    }
    inst = _read_elsevier_insttoken()
    if inst:
        headers["X-ELS-Insttoken"] = inst

    try:
        resp = session.get(
            url,
            params={"httpAccept": "text/xml", "view": "FULL"},
            headers=headers,
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            logger.debug(
                f"Elsevier article API status={resp.status_code} doi={d!r}"
            )
            return None
        body = resp.text or ""
        low = body.lower()
        if "service-error" in low or "resource_not_found" in low.replace(" ", ""):
            logger.debug(f"Elsevier article API error payload for doi={d!r}")
            return None
        if len(body) < 500:
            return None
        return body
    except Exception as e:
        logger.debug(f"Elsevier full-text fetch failed doi={d!r}: {e}")
        return None


ELSEVIER_ARTICLE_DOI_URL = "https://api.elsevier.com/content/article/doi"


# ScienceDirect (10.1016) plus ASBMB journals hosted on Elsevier (JBC, MCP, etc.).
ELSEVIER_DOI_PREFIXES = ("10.1016/", "10.1074/")


def is_elsevier_primary_doi(doi: Optional[str]) -> bool:
    """True for DOIs on Elsevier-hosted journals (ScienceDirect and JBC/MCP)."""
    d = (doi or "").strip().lower()
    return bool(d) and any(d.startswith(prefix) for prefix in ELSEVIER_DOI_PREFIXES)


def download_elsevier_article_pdf(
    doi: str,
    session: requests.Session,
    pdf_dir: str,
    file_stem: str,
    timeout_s: int = 180,
) -> Optional[str]:
    """
    Download article PDF via Elsevier Article Retrieval API.
    Requires env ELSEVIER_API_KEY or ELS_API_KEY.
    Optional X-ELS-Insttoken: ELSEVIER_INSTTOKEN or ELS_INSTTOKEN.
    """
    key = (os.environ.get("ELSEVIER_API_KEY") or os.environ.get("ELS_API_KEY") or "").strip()
    if not key:
        return None
    inst = (os.environ.get("ELSEVIER_INSTTOKEN") or os.environ.get("ELS_INSTTOKEN") or "").strip()
    d = doi.strip()
    encoded = quote(d, safe="")
    url = f"{ELSEVIER_ARTICLE_DOI_URL}/{encoded}"
    headers: Dict[str, str] = {
        "X-ELS-APIKey": key,
        "Accept": "application/pdf",
        "Connection": "close",
    }
    if inst:
        headers["X-ELS-Insttoken"] = inst
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        resp = session.get(
            url,
            params={"httpAccept": "application/pdf"},
            headers=headers,
            timeout=timeout_s,
        )
        if resp.status_code in (401, 403, 404):
            logger.debug(f"Elsevier article API {resp.status_code} for DOI={d!r}")
            return None
        resp.raise_for_status()
        content = resp.content or b""
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
            logger.debug(f"Elsevier response not PDF for DOI={d!r} ctype={ctype!r}")
            return None
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path
    except Exception as e:
        logger.debug(f"Elsevier PDF download failed for DOI={d!r}: {e}")
        return None


# Wiley Text & Data Mining API (PDF). Token from WOL TDM registration.
# https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining
WILEY_TDM_ARTICLES_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"


def _read_wiley_tdm_token() -> Optional[str]:
    raw = (
        os.environ.get("TDM_API_TOKEN")
        or os.environ.get("WILEY_TDM_API_TOKEN")
        or ""
    ).strip()
    if not raw:
        return None
    try:
        return str(UUID(raw))
    except ValueError:
        logger.debug("Wiley TDM token set but not a valid UUID; skipping Wiley TDM")
        return None


def is_wiley_primary_doi(doi: Optional[str]) -> bool:
    """True for DOIs commonly hosted on Wiley Online Library (TDM-eligible prefix)."""
    d = (doi or "").strip().lower()
    if not d:
        return False
    return d.startswith("10.1002/") or d.startswith("10.1111/")


def download_wiley_tdm_pdf(
    doi: str,
    session: requests.Session,
    pdf_dir: str,
    file_stem: str,
    timeout_s: int = 180,
) -> Optional[str]:
    """
    Download article PDF via Wiley TDM API.

    Env: TDM_API_TOKEN or WILEY_TDM_API_TOKEN (UUID from Wiley).

    Entitlements are tied to the requester's public IP; cluster nodes must be
    allowed like browser access through the library.
    """
    token = _read_wiley_tdm_token()
    if not token:
        return None
    d = unquote((doi or "").strip())
    if not d:
        return None

    enc = quote(d, safe="")
    url = f"{WILEY_TDM_ARTICLES_URL}{enc}"
    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    headers = {
        "Wiley-TDM-Client-Token": token,
        "Accept": "application/pdf",
        "User-Agent": "auto-lit-metrics/1.0 (wiley-tdm)",
    }
    try:
        resp = session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(10, timeout_s),
        )
        if resp.status_code != 200:
            logger.debug(f"Wiley TDM API status={resp.status_code} doi={d!r}")
            return None
        chunks: list[bytes] = []
        for block in resp.iter_content(chunk_size=65536):
            if block:
                chunks.append(block)
        content = b"".join(chunks)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
            logger.debug(f"Wiley TDM response not PDF for doi={d!r} ctype={ctype!r}")
            return None
        with open(out_path, "wb") as f:
            f.write(content)
        return out_path
    except Exception as e:
        logger.debug(f"Wiley TDM PDF download failed for DOI={d!r}: {e}")
        return None


def is_mdpi_primary_doi(doi: Optional[str]) -> bool:
    """True for DOIs on MDPI's main Crossref prefix (open access)."""
    d = (doi or "").strip().lower()
    return bool(d) and d.startswith("10.3390/")


def _browser_like_user_agent() -> str:
    """Used for publisher sites that reject non-browser clients (MDPI, ASM PDF GET)."""
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


_CITATION_PDF_META_RE = re.compile(
    r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)["\']',
    re.I,
)
_CITATION_PDF_META_RE2 = re.compile(
    r'<meta\s+content=["\']([^"\']+)["\']\s+name=["\']citation_pdf_url["\']',
    re.I,
)


def _extract_citation_pdf_url_from_html(html: str) -> Optional[str]:
    """Highwire / Silverchair-style citation PDF link from article HTML."""
    if not html:
        return None
    for rx in (_CITATION_PDF_META_RE, _CITATION_PDF_META_RE2):
        m = rx.search(html)
        if m:
            url = (m.group(1) or "").strip()
            if url.lower().startswith(("http://", "https://")):
                return url
    return None


def _browser_html_fetch_headers(referer: Optional[str] = None) -> Dict[str, str]:
    h: Dict[str, str] = {
        "User-Agent": _browser_like_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h


def _browser_pdf_fetch_headers(referer: str) -> Dict[str, str]:
    ref = (referer or "").strip()
    return {
        "User-Agent": _browser_like_user_agent(),
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": ref,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1",
    }


def _crossref_user_agent() -> str:
    mailto = (
        os.environ.get("COLLECTOR_EMAIL")
        or os.environ.get("CROSSREF_MAILTO")
        or ""
    ).strip()
    base = "auto-lit-metrics/1.0 (MDPI/Crossref resolver)"
    return f"{base} mailto:{mailto}" if mailto else base


def _mdpi_normalize_mdpi_url(url: str) -> str:
    """Force https and bare mdpi.com -> www.mdpi.com (article PDFs live on www)."""
    if not url or "mdpi.com" not in url.lower():
        return url
    p = urlparse(url.strip())
    host = (p.netloc or "").lower()
    if "mdpi.com" not in host:
        return url
    scheme = "https"
    if host == "mdpi.com":
        host = "www.mdpi.com"
    return urlunparse((scheme, host, p.path or "", p.params, p.query, p.fragment))


def _mdpi_pdf_referer(pdf_url: str) -> str:
    u = (pdf_url or "").rstrip("/")
    if re.search(r"/pdf$", u, re.I):
        return re.sub(r"/pdf$", "/", u, count=1, flags=re.I)
    return u + "/" if u else pdf_url


def _mdpi_guess_pdf_from_landing_url(url: str) -> Optional[str]:
    """
    MDPI article pages use /htm or bare .../<journal>/vol/issue/article paths; PDF is .../pdf.

    Journal id is often an ISSN-like token (e.g. 2218-273X), not all digits — older code
    required /\\d+/\\d+/\\d+$/ and never derived /pdf for real MDPI URLs.
    """
    if not url or "mdpi.com" not in url.lower():
        return None
    base = url.split("?")[0].strip()
    parsed = urlparse(base)
    if "mdpi.com" not in (parsed.netloc or "").lower():
        return None
    path = (parsed.path or "").rstrip("/")
    low = path.lower()
    if low.endswith("/pdf"):
        return _mdpi_normalize_mdpi_url(base)
    if low.endswith("/htm") or low.endswith("/html"):
        new_path = re.sub(r"/html?$", "/pdf", path, count=1, flags=re.I)
        return _mdpi_normalize_mdpi_url(
            urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))
        )
    # https://www.mdpi.com/<journal-id>/<vol>/<issue>/<article>
    parts = [p for p in path.split("/") if p]
    if (
        len(parts) >= 4
        and parts[-1].isdigit()
        and parts[-2].isdigit()
        and parts[-3].isdigit()
    ):
        prefix = "/".join(parts)
        pdf_path = f"/{prefix}/pdf"
        return _mdpi_normalize_mdpi_url(
            urlunparse((parsed.scheme, parsed.netloc, pdf_path, "", "", ""))
        )
    return None


def _mdpi_pdf_url_from_doi_org_redirect(
    doi: str,
    session: requests.Session,
    timeout_s: int = 25,
) -> Optional[str]:
    """Follow doi.org to MDPI landing URL, then derive /pdf."""
    d = (doi or "").strip()
    if not d:
        return None
    enc = quote(d, safe="")
    url = f"https://doi.org/{enc}"
    try:
        resp = session.get(
            url,
            allow_redirects=True,
            timeout=timeout_s,
            headers={
                "User-Agent": _browser_like_user_agent(),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            stream=True,
        )
        try:
            final = (resp.url or "").strip()
        finally:
            resp.close()
        return _mdpi_guess_pdf_from_landing_url(_mdpi_normalize_mdpi_url(final))
    except Exception as e:
        logger.debug(f"doi.org redirect for MDPI failed doi={d!r}: {e}")
        return None


def _mdpi_pdf_url_candidates_from_crossref(
    doi: str,
    session: requests.Session,
    timeout_s: int = 25,
) -> List[str]:
    """
    Collect possible MDPI PDF URLs from Crossref (explicit PDF, /htm→/pdf, resource URL).
    """
    d = (doi or "").strip()
    if not d:
        return []
    enc = quote(d, safe="")
    url = f"https://api.crossref.org/works/{enc}"
    try:
        resp = session.get(
            url,
            headers={"User-Agent": _crossref_user_agent()},
            timeout=timeout_s,
        )
        if resp.status_code != 200:
            logger.debug(f"Crossref works status={resp.status_code} doi={d!r}")
            return []
        msg = (resp.json() or {}).get("message") or {}
    except Exception as e:
        logger.debug(f"Crossref MDPI lookup failed doi={d!r}: {e}")
        return []

    seen: set[str] = set()
    out: List[str] = []

    def add(u: Optional[str]) -> None:
        if not u:
            return
        u = _mdpi_normalize_mdpi_url(u.strip())
        if u in seen:
            return
        seen.add(u)
        out.append(u)

    for link in msg.get("link") or []:
        lu = (link.get("URL") or "").strip()
        if not lu or "mdpi.com" not in lu.lower():
            continue
        ctype = (link.get("content-type") or "").lower()
        low = lu.lower().rstrip("/")
        if "pdf" in ctype or low.endswith("/pdf"):
            add(lu)
        guess = _mdpi_guess_pdf_from_landing_url(lu)
        if guess:
            add(guess)

    res = msg.get("resource")
    if isinstance(res, dict):
        prim = res.get("primary")
        if isinstance(prim, dict):
            ru = prim.get("URL")
            if isinstance(ru, str) and "mdpi.com" in ru.lower():
                add(_mdpi_guess_pdf_from_landing_url(ru))

    mu = (msg.get("URL") or "").strip()
    if mu and "mdpi.com" in mu.lower():
        g = _mdpi_guess_pdf_from_landing_url(mu)
        if g:
            add(g)

    return out


def _mdpi_pdf_url_candidates(
    doi: str,
    session: requests.Session,
    timeout_s: int = 25,
) -> List[str]:
    """Ordered PDF URL candidates: Crossref-derived first, then doi.org redirect."""
    seen: set[str] = set()
    out: List[str] = []

    def extend(urls: List[str]) -> None:
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                out.append(u)

    extend(_mdpi_pdf_url_candidates_from_crossref(doi, session, timeout_s))
    alt = _mdpi_pdf_url_from_doi_org_redirect(doi, session, timeout_s)
    if alt and alt not in seen:
        seen.add(alt)
        out.append(alt)
    return out


def download_mdpi_article_pdf(
    doi: str,
    session: requests.Session,
    pdf_dir: str,
    file_stem: str,
    timeout_s: int = 180,
) -> Optional[str]:
    """
    Download MDPI article PDF without Unpaywall: Crossref links + doi.org→mdpi /pdf guess.

    Gold OA MDPI content; no publisher API key.
    Optional COLLECTOR_EMAIL / CROSSREF_MAILTO for polite Crossref.
    """
    d = (doi or "").strip()
    if not d or not is_mdpi_primary_doi(d):
        return None

    candidates = _mdpi_pdf_url_candidates(d, session, timeout_s=min(30, timeout_s))
    if not candidates:
        logger.debug(f"No MDPI PDF URL candidates for doi={d!r}")
        return None

    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Normalize + dedupe after normalization
    seen_c: set[str] = set()
    norm_candidates: List[str] = []
    for u in candidates:
        nu = _mdpi_normalize_mdpi_url(u)
        if nu and nu not in seen_c:
            seen_c.add(nu)
            norm_candidates.append(nu)

    for pdf_url in norm_candidates:
        try:
            ref = _mdpi_pdf_referer(pdf_url)
            htm_url = ref.rstrip("/") + "/htm"
            warm_hdr = _browser_html_fetch_headers(ref)
            for warm in (htm_url, ref):
                try:
                    session.get(
                        warm,
                        headers=warm_hdr,
                        timeout=min(25, timeout_s),
                        allow_redirects=True,
                    )
                except Exception:
                    pass
            hdr = _browser_pdf_fetch_headers(ref)
            resp = session.get(
                pdf_url,
                headers=hdr,
                timeout=timeout_s,
                allow_redirects=True,
            )
            if resp.status_code not in (200, 206):
                logger.debug(
                    f"MDPI PDF GET status={resp.status_code} url={pdf_url!r} doi={d!r}"
                )
                if resp.status_code == 403:
                    try:
                        hresp = session.get(
                            htm_url,
                            headers=warm_hdr,
                            timeout=min(25, timeout_s),
                            allow_redirects=True,
                        )
                        if hresp.status_code == 200:
                            alt_raw = _extract_citation_pdf_url_from_html(
                                hresp.text
                            )
                            alt = (
                                _mdpi_normalize_mdpi_url(alt_raw)
                                if alt_raw
                                else None
                            )
                            retry_urls: List[str] = []
                            if alt:
                                retry_urls.append(alt)
                            if pdf_url not in retry_urls:
                                retry_urls.append(pdf_url)
                            for ru in retry_urls:
                                resp = session.get(
                                    ru,
                                    headers=_browser_pdf_fetch_headers(htm_url),
                                    timeout=timeout_s,
                                    allow_redirects=True,
                                )
                                if resp.status_code in (200, 206):
                                    break
                    except Exception as e:
                        logger.debug(
                            f"MDPI 403 HTML fallback failed url={pdf_url!r} doi={d!r}: {e}"
                        )
                if resp.status_code not in (200, 206):
                    continue
            content = resp.content or b""
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
                logger.debug(
                    f"MDPI non-PDF body url={pdf_url!r} doi={d!r} "
                    f"ctype={ctype!r} head={content[:80]!r}"
                )
                continue
            with open(out_path, "wb") as f:
                f.write(content)
            return out_path
        except Exception as e:
            logger.debug(f"MDPI candidate failed url={pdf_url!r} doi={d!r}: {e}")
            continue
    logger.debug(f"All MDPI PDF candidates failed for doi={d!r}")
    return None


def is_asm_primary_doi(doi: Optional[str]) -> bool:
    """True for DOIs hosted on ASM journals (Science/Medicine prefixes)."""
    d = (doi or "").strip().lower()
    return bool(d) and d.startswith("10.1128/")


def _asm_pdf_candidates(doi: str) -> List[str]:
    """
    ASM / Silverchair PDF URL patterns.

    Try full DOI as one encoded path segment first (common on current asm.org),
    then legacy two-segment prefix/suffix URLs.
    """
    d = (doi or "").strip()
    if not d:
        return []

    if "/" not in d:
        return []
    enc_full = quote(d, safe="")
    prefix, suffix = d.split("/", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix or not suffix:
        return []

    prefix_q = quote(prefix, safe="")
    suffix_q = quote(suffix, safe="")
    return [
        f"https://journals.asm.org/doi/pdf/{enc_full}",
        f"https://journals.asm.org/doi/epdf/{enc_full}",
        f"https://journals.asm.org/doi/pdf/{prefix_q}/{suffix_q}",
        f"https://journals.asm.org/doi/epdf/{prefix_q}/{suffix_q}",
    ]


def _asm_derive_pdf_from_doi_url(final_url: str, timeout_s: int = 25) -> List[str]:
    """
    Small “crawler” fallback:
    - follow doi.org to get the final landing page URL
    - if it includes /doi/<doi>, derive /doi/pdf/<prefix>/<suffix> endpoints
    """
    if not final_url:
        return []
    if "journals.asm.org" not in final_url.lower():
        return []
    if "/doi/" not in final_url:
        return []

    # final URL often looks like: .../doi/<full-doi> (possibly URL-encoded)
    try:
        tail = final_url.split("/doi/", 1)[1].strip().rstrip("/").split("?")[0]
        tail = unquote(tail)
        if not tail:
            return []
        enc_full = quote(tail, safe="")
        out = [
            f"https://journals.asm.org/doi/pdf/{enc_full}",
            f"https://journals.asm.org/doi/epdf/{enc_full}",
        ]
        if "/" in tail:
            prefix, suffix = tail.split("/", 1)
            prefix_q = quote(prefix.strip(), safe="")
            suffix_q = quote(suffix.strip(), safe="")
            out.extend(
                [
                    f"https://journals.asm.org/doi/pdf/{prefix_q}/{suffix_q}",
                    f"https://journals.asm.org/doi/epdf/{prefix_q}/{suffix_q}",
                ]
            )
        return out
    except Exception:
        return []


def download_asm_article_pdf(
    doi: str,
    session: requests.Session,
    pdf_dir: str,
    file_stem: str,
    timeout_s: int = 180,
) -> Optional[str]:
    """
    Download ASM article PDF via DOI-derived endpoints.

    This is a lightweight “webcrawler” approach (no publisher API token),
    relying on predictable ASM URL patterns and a doi.org redirect fallback.
    """
    d = (doi or "").strip()
    if not d or not is_asm_primary_doi(d):
        return None

    if not pdf_dir:
        return None

    os.makedirs(pdf_dir, exist_ok=True)
    out_path = os.path.join(pdf_dir, f"{file_stem}.pdf")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    candidates = _asm_pdf_candidates(d)

    # If direct candidate URLs fail, try a quick doi.org redirect to derive PDFs.
    try:
        resp = session.get(
            f"https://doi.org/{quote(d, safe='')}",
            allow_redirects=True,
            timeout=min(25, timeout_s),
            headers={
                "User-Agent": _browser_like_user_agent(),
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            stream=True,
        )
        try:
            final = (resp.url or "").strip()
        finally:
            resp.close()
        candidates.extend(_asm_derive_pdf_from_doi_url(final))
    except Exception:
        # Not fatal; we'll try whatever candidates we have.
        pass

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: List[str] = []
    for u in candidates:
        if u and u not in seen:
            seen.add(u)
            uniq.append(u)

    doi_landing = f"https://journals.asm.org/doi/{quote(d, safe='/')}"
    warm_h = _browser_html_fetch_headers("https://doi.org/")
    landing_html: Optional[str] = None
    try:
        lr = session.get(
            doi_landing,
            headers=warm_h,
            timeout=min(25, timeout_s),
            allow_redirects=True,
        )
        if lr.status_code == 200:
            landing_html = lr.text
    except Exception:
        pass

    if landing_html:
        front: List[str] = []
        meta_u = _extract_citation_pdf_url_from_html(landing_html)
        if meta_u:
            front.append(meta_u.strip())
        for m in re.finditer(
            r'https://journals\.asm\.org/doi/(?:pdf|epdf)/[^"\'\s<>]+',
            landing_html,
            re.I,
        ):
            u = m.group(0).strip().rstrip(").,;")
            if u and u not in front:
                front.append(u)
        if front:
            seen_asm: set[str] = set()
            merged: List[str] = []
            for u in front + uniq:
                if u and u not in seen_asm:
                    seen_asm.add(u)
                    merged.append(u)
            uniq = merged

    for pdf_url in uniq:
        try:
            r = session.get(
                pdf_url,
                headers=_browser_pdf_fetch_headers(doi_landing),
                timeout=timeout_s,
                allow_redirects=True,
            )
            if r.status_code not in (200, 206):
                logger.debug(
                    f"ASM PDF GET status={r.status_code} url={pdf_url!r} doi={d!r}"
                )
                if r.status_code == 403:
                    try:
                        session.get(
                            doi_landing,
                            headers=warm_h,
                            timeout=min(25, timeout_s),
                            allow_redirects=True,
                        )
                        r = session.get(
                            pdf_url,
                            headers=_browser_pdf_fetch_headers(doi_landing),
                            timeout=timeout_s,
                            allow_redirects=True,
                        )
                    except Exception as e:
                        logger.debug(
                            f"ASM 403 re-warm retry failed url={pdf_url!r} doi={d!r}: {e}"
                        )
                if r.status_code not in (200, 206):
                    continue
            content = r.content or b""
            ctype = (r.headers.get("Content-Type") or "").lower()
            if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
                logger.debug(
                    f"ASM non-PDF body url={pdf_url!r} doi={d!r} "
                    f"ctype={ctype!r} head={content[:80]!r}"
                )
                continue
            with open(out_path, "wb") as f:
                f.write(content)
            return out_path
        except Exception as e:
            logger.debug(f"ASM PDF candidate failed url={pdf_url!r} doi={d!r}: {e}")
            continue

    return None