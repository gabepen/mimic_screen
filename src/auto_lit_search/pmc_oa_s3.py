"""
PMC Open Access article fetch via the public AWS S3 bucket (pmc-oa-opendata).

Bucket: s3://pmc-oa-opendata (us-east-1, anonymous HTTPS, no AWS account required)
Docs: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

PMC_OA_BUCKET = "pmc-oa-opendata"
PMC_OA_HTTPS_BASE = f"https://{PMC_OA_BUCKET}.s3.amazonaws.com"


@dataclass
class PmcOaArticleMetadata:
    pmcid: str
    version: int
    doi: Optional[str]
    text_https_url: Optional[str]
    xml_https_url: Optional[str]
    pdf_https_url: Optional[str]
    raw: Dict[str, Any]


def normalize_pmcid(pmcid: str) -> str:
    pid = (pmcid or "").strip()
    if not pid:
        return ""
    u = pid.upper()
    if u.startswith("PMC:"):
        u = u[4:].strip()
    if not u.startswith("PMC"):
        if u.isdigit():
            u = f"PMC{u}"
        else:
            return pid
    return u


def pmc_version_prefix(pmcid: str, version: int) -> str:
    return f"{normalize_pmcid(pmcid)}.{int(version)}"


def metadata_https_url(pmcid: str, version: int = 1) -> str:
    return f"{PMC_OA_HTTPS_BASE}/metadata/{pmc_version_prefix(pmcid, version)}.json"


def article_object_https_url(pmcid: str, version: int, kind: str) -> str:
    prefix = pmc_version_prefix(pmcid, version)
    ext = kind.lstrip(".")
    return f"{PMC_OA_HTTPS_BASE}/{prefix}/{prefix}.{ext}"


def s3_uri_to_https_url(s3_uri: str) -> str:
    """Convert s3://bucket/key?md5=... to anonymous HTTPS URL."""
    uri = (s3_uri or "").strip()
    if not uri:
        return ""
    if uri.startswith("http://") or uri.startswith("https://"):
        return uri
    if not uri.startswith("s3://"):
        return uri
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        return ""
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    s.headers.setdefault("User-Agent", "auto_lit_search/0.1 (pmc-oa-s3)")
    return s


def fetch_pmc_oa_metadata(
    pmcid: str,
    session: Optional[requests.Session] = None,
    max_version_probe: int = 3,
    timeout_s: int = 30,
) -> Optional[PmcOaArticleMetadata]:
    """
    Load metadata JSON for a PMCID, probing versions 1..max_version_probe.

    Returns None when the article is not in the OA S3 bucket.
    """
    sess = session or _session()
    pid = normalize_pmcid(pmcid)
    if not pid:
        return None

    for version in range(1, max(1, max_version_probe) + 1):
        url = metadata_https_url(pid, version)
        try:
            resp = sess.get(url, timeout=timeout_s)
        except Exception:
            continue
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            continue
        try:
            raw = resp.json()
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        return PmcOaArticleMetadata(
            pmcid=str(raw.get("pmcid") or pid),
            version=int(raw.get("version") or version),
            doi=(str(raw.get("doi")).strip() or None) if raw.get("doi") else None,
            text_https_url=s3_uri_to_https_url(str(raw.get("text_url") or "")) or None,
            xml_https_url=s3_uri_to_https_url(str(raw.get("xml_url") or "")) or None,
            pdf_https_url=s3_uri_to_https_url(str(raw.get("pdf_url") or "")) or None,
            raw=raw,
        )
    return None


def download_pmc_oa_fulltext(
    meta: PmcOaArticleMetadata,
    session: Optional[requests.Session] = None,
    prefer_txt: bool = True,
    timeout_s: int = 120,
) -> Tuple[Optional[str], str]:
    """
    Download full text from S3 HTTPS. Returns (body, artifact_kind).

    Prefers pre-extracted .txt; falls back to JATS XML (returned as raw XML string).
    """
    sess = session or _session()
    order: List[Tuple[str, Optional[str]]] = []
    if prefer_txt:
        order.append(("txt", meta.text_https_url))
        order.append(("xml", meta.xml_https_url))
    else:
        order.append(("xml", meta.xml_https_url))
        order.append(("txt", meta.text_https_url))

    for kind, url in order:
        if not url:
            url = article_object_https_url(meta.pmcid, meta.version, kind)
        try:
            resp = sess.get(url, timeout=timeout_s)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        body = resp.text or ""
        if kind == "txt" and len(body.strip()) < 200:
            continue
        if kind == "xml" and len(body.strip()) < 500:
            continue
        return body, kind
    return None, ""


def download_pmc_oa_pdf(
    meta: PmcOaArticleMetadata,
    session: Optional[requests.Session] = None,
    timeout_s: int = 120,
) -> Optional[bytes]:
    """Download article PDF bytes from S3 when metadata includes pdf_url."""
    sess = session or _session()
    url = meta.pdf_https_url or article_object_https_url(meta.pmcid, meta.version, "pdf")
    if not url:
        return None
    try:
        resp = sess.get(url, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        content = resp.content or b""
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if (not content.startswith(b"%PDF")) and ("pdf" not in ctype):
            return None
        return content
    except Exception:
        return None
