"""
Search module for automated literature search pipeline (Module 2 of 4).

Two-phase search:
  1. UniProt pass: Europe PMC search by UniProt accession citations
     (ACCESSION_ID:ACCESSION AND ACCESSION_TYPE:uniprot) for query and target.
  2. Text search: two-pass Europe PMC TITLE_ABS/BODY search for **both** query and
     target using mapping-row identifiers; merged with UniProt hits (deduped).
     Pass1: locus_tag OR GenBank acc (+ accession stem without version suffix).
     Pass2: gene symbol / description terms plus Entrez-linked synonyms (human:
     NCBI gene_info; all taxa: MyGene alias/symbol merge).

Output columns include per-source DOI lists (europepmc_accession, text_pass1,
text_pass2_*) for query and target.
"""

import argparse
import gzip
import json
import os
import re
import socket
import sys
import time
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import urllib3.util.connection as urllib3_connection
from loguru import logger


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


def _configure_file_logging(output_dir: str) -> None:
    """Add a DEBUG-level file handler in the given output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "search_debug.log")
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Search debug log file: {log_path}")


EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBTATOR3_SEARCH_URLS: List[str] = [
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/",
    "https://ncbi.nlm.nih.gov/research/pubtator3-api/search/",
]
_PUBTATOR_ENABLED: bool = True
_PUBTATOR_DISABLED_REASON: str = ""
NCBI_HUMAN_GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)

_EXCLUDED_PUBTYPE_SUBSTRINGS = (
    "review",
    "case-report",
    "case report",
    "editorial",
    "letter",
    "comment",
    "news",
    "meeting",
    "protocol",
    "preprint",
    "expression-of-concern",
    "expression of concern",
    "retraction",
    "erratum",
    "correction",
    "book-chapter",
    "conference",
)
_ALLOWED_RESEARCH_PUBTYPE_SUBSTRINGS = (
    "research-article",
    "journal article",
    "journal-article",
    "original article",
)


def _set_pubtator_disabled(reason: str) -> None:
    global _PUBTATOR_ENABLED, _PUBTATOR_DISABLED_REASON
    if _PUBTATOR_ENABLED:
        _PUBTATOR_ENABLED = False
        _PUBTATOR_DISABLED_REASON = reason
        logger.warning(f"PubTator disabled for this run: {reason}")


def _probe_pubtator_connectivity(session: requests.Session) -> bool:
    """
    One-time preflight to avoid repeated per-gene timeouts when NCBI is unreachable.
    """
    probe_session = requests.Session()
    probe_session.trust_env = False
    probe_session.headers.update(session.headers)
    probe_session.headers.setdefault("User-Agent", "auto_lit_search/0.1")
    probe_params = {"format": "json", "text": "@GENE_7157", "page": 1}

    last_err: Optional[Exception] = None
    for base_url in PUBTATOR3_SEARCH_URLS:
        try:
            resp = probe_session.get(
                base_url,
                params=probe_params,
                timeout=(8, 20),
                headers={"Connection": "close"},
            )
            resp.raise_for_status()
            _ = resp.json()
            return True
        except Exception as e:
            last_err = e
            logger.debug(f"PubTator preflight failed for host={base_url}: {e}")

    _set_pubtator_disabled(f"preflight connectivity failure: {last_err}")
    return False


def _force_ipv4_resolution() -> None:
    """
    Force urllib3/requests DNS resolution to IPv4.
    Useful on cluster nodes where NCBI resolves to IPv6 but IPv6 routing is broken.
    """
    try:
        urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
        logger.info("Forcing IPv4 DNS resolution for HTTP requests")
    except Exception as e:
        logger.warning(f"Could not force IPv4 DNS resolution: {e}")


def _normalize_uniprot_id(acc: Optional[str]) -> Optional[str]:
    """Return stripped non-empty UniProt accession or None (rejects nan/empty)."""
    s = _normalize_term(acc)
    return s.upper() if s else None


def run_europepmc_crossref(
    uniprot_id: Optional[str],
    session: requests.Session,
    cache: Dict[str, Dict[str, List[str]]],
    delay: float = 0.35,
) -> Dict[str, List[str]]:
    """
    Search Europe PMC for UniProt accession citations (ACCESSION_ID + ACCESSION_TYPE).
    Returns dict with keys "dois", "titles". Uses resultType=core.
    """
    acc = _normalize_uniprot_id(uniprot_id)
    if not acc:
        return {"dois": [], "titles": []}

    if acc in cache:
        logger.debug(f"Cache hit for accession-cite uniprot:{acc}")
        return cache[acc]

    # Europe PMC search syntax:
    # - ACCESSION_ID: find articles containing the accession number
    # - ACCESSION_TYPE: restrict to UniProt accessions
    query = f"ACCESSION_ID:{acc} AND ACCESSION_TYPE:uniprot"
    logger.debug(f"Europe PMC accession-cite uniprot:{acc}")

    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": 200,
    }
    dois: List[str] = []
    titles: List[str] = []

    try:
        time.sleep(delay)
        resp = session.get(EUROPEPMC_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Europe PMC accession-cite failed for uniprot:{acc}: {e}")
        result = {"dois": [], "titles": []}
        cache[acc] = result
        return result

    records = (data.get("resultList") or {}).get("result") or []
    logger.debug(f"accession-cite uniprot:{acc} -> {len(records)} results")

    for rec in records:
        pid = _extract_paper_id(rec)
        if not pid:
            continue
        title = rec.get("title") or ""
        dois.append(pid)
        titles.append(title)

    result = {"dois": dois, "titles": titles}
    cache[acc] = result
    return result


def _normalize_entrez_id(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        if isinstance(val, float) and val != val:  # NaN
            return None
    except Exception:
        pass
    try:
        x = int(val)
        return x if x > 0 else None
    except Exception:
        return None


def _pubtator_gene_to_pmids(
    entrez_gene_id: int,
    session: requests.Session,
    cache: Dict[int, List[str]],
    delay: float = 0.35,
    max_pmids: int = 500,
    connect_timeout_s: int = 10,
    read_timeout_s: int = 90,
    max_retries: int = 3,
) -> List[str]:
    """
    Query PubTator3 semantic search with Entrez gene ID and return PMIDs.
    Uses query format: @GENE_<entrez_id>
    """
    global _PUBTATOR_ENABLED, _PUBTATOR_DISABLED_REASON
    if not _PUBTATOR_ENABLED:
        return []

    if entrez_gene_id in cache:
        return cache[entrez_gene_id]

    query = f"@GENE_{entrez_gene_id}"
    out: List[str] = []
    seen: set[str] = set()
    page = 1
    # Dedicated session for NCBI/PubTator calls. Ignoring proxy env vars
    # avoids flaky proxy/TLS behavior seen on some cluster nodes.
    pubtator_session = requests.Session()
    pubtator_session.trust_env = False
    pubtator_session.headers.update(session.headers)
    pubtator_session.headers.setdefault("User-Agent", "auto_lit_search/0.1")

    while len(out) < max_pmids:
        time.sleep(delay)
        data: Dict[str, Any] = {}
        request_ok = False
        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            for base_url in PUBTATOR3_SEARCH_URLS:
                try:
                    resp = pubtator_session.get(
                        base_url,
                        params={
                            "format": "json",
                            "text": query,
                            "page": page,
                        },
                        timeout=(connect_timeout_s, read_timeout_s),
                        headers={"Connection": "close"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    request_ok = True
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    logger.debug(
                        f"PubTator3 request failed for gene_id={entrez_gene_id} "
                        f"page={page} host={base_url} attempt={attempt}/{max_retries}: {e}"
                    )
                    if isinstance(
                        e,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.SSLError,
                        ),
                    ):
                        _set_pubtator_disabled(
                            f"transport failure on gene_id={entrez_gene_id}: {e}"
                        )
                        cache[entrez_gene_id] = []
                        return []

            if request_ok:
                break
            if attempt < max_retries:
                backoff_s = min(8.0, delay * (2**attempt))
                time.sleep(backoff_s)

        if not request_ok:
            logger.warning(
                f"PubTator3 search failed for gene_id={entrez_gene_id} page={page} "
                f"after {max_retries} retries: {last_err}"
            )
            break

        results = data.get("results") or []
        if not results:
            break

        for rec in results:
            pmid = rec.get("pmid")
            if pmid is None:
                continue
            s = str(pmid).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_pmids:
                break

        total_pages = int(data.get("total_pages") or 0)
        current = int(data.get("current") or page)
        if total_pages <= 0 or current >= total_pages:
            break
        page += 1

    cache[entrez_gene_id] = out
    return out


def _normalize_term(term: Optional[str]) -> Optional[str]:
    """Strip and return term if non-empty; else None."""
    if term is None:
        return None
    s = str(term).strip()
    if not s:
        return None
    if s.lower() == "nan":
        return None
    return s


def _get_pubtype_tokens(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    pubtype_list = rec.get("pubTypeList")
    if isinstance(pubtype_list, dict):
        vals = pubtype_list.get("pubType")
        if isinstance(vals, list):
            out.extend([str(v).strip().lower() for v in vals if str(v).strip()])
        elif vals:
            out.append(str(vals).strip().lower())
    elif pubtype_list:
        out.append(str(pubtype_list).strip().lower())
    pubtype = rec.get("pubType")
    if pubtype:
        out.append(str(pubtype).strip().lower())
    return out


def _is_research_article_record(rec: Dict[str, Any]) -> bool:
    """
    Keep likely full scientific research articles; drop known noisy types.
    """
    pubtypes = _get_pubtype_tokens(rec)
    pubtypes_joined = " | ".join(pubtypes)

    if pubtypes_joined:
        if any(tok in pubtypes_joined for tok in _EXCLUDED_PUBTYPE_SUBSTRINGS):
            return False
        if any(tok in pubtypes_joined for tok in _ALLOWED_RESEARCH_PUBTYPE_SUBSTRINGS):
            return True

    # Fallback heuristic when pubType is missing/inconsistent.
    # Require at least abstract presence to avoid metadata-only records.
    has_abstract = str(rec.get("hasAbstractText", "")).upper() == "Y"
    return has_abstract


def _load_human_gene_name_synonyms(
    session: requests.Session,
    entrez_ids: List[int],
    output_dir: str,
    delay: float = 0.35,
) -> Dict[int, List[str]]:
    """
    Build Entrez->name/synonym map from NCBI Homo sapiens gene_info.
    """
    clean_ids = sorted({int(x) for x in entrez_ids if int(x) > 0})
    if not clean_ids:
        return {}

    cache_path = os.path.join(output_dir, "entrez_gene_name_synonyms_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            out: Dict[int, List[str]] = {}
            for k, v in (cached or {}).items():
                try:
                    gid = int(k)
                except Exception:
                    continue
                if gid in clean_ids and isinstance(v, list):
                    out[gid] = [str(x).strip() for x in v if str(x).strip()]
            if set(clean_ids) <= set(out.keys()):
                return out
        except Exception:
            pass

    time.sleep(delay)
    resp = session.get(NCBI_HUMAN_GENE_INFO_URL, timeout=(10, 90))
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8", errors="ignore")
    gdf = pd.read_csv(StringIO(text), sep="\t", low_memory=False)
    gdf = gdf[gdf["GeneID"].isin(clean_ids)].copy()

    out: Dict[int, List[str]] = {}
    for _, r in gdf.iterrows():
        gid = int(r["GeneID"])
        names: set[str] = set()
        for col in ("Symbol", "Full_name_from_nomenclature_authority"):
            v = _normalize_term(r.get(col))
            if v and v != "-":
                names.add(v)
        syn = r.get("Synonyms")
        if pd.notna(syn) and str(syn).strip() != "-":
            names.update([x.strip() for x in str(syn).split("|") if x.strip() and x.strip() != "-"])
        other = r.get("Other_designations")
        if pd.notna(other) and str(other).strip() != "-":
            names.update([x.strip() for x in str(other).split("|") if x.strip() and x.strip() != "-"])

        out[gid] = sorted(names)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in out.items()}, f, indent=2)
    except Exception:
        pass
    return out


def _load_mygene_synonyms_for_entrez(
    entrez_ids: List[int],
    output_dir: str,
    delay: float = 0.35,
) -> Dict[int, List[str]]:
    """
    Entrez Gene ID -> symbol / alias strings via MyGene.info (all species).

    Fills synonym expansion for microbial and other genes not covered by
    Homo_sapiens.gene_info alone.
    """
    try:
        import mygene
    except ImportError:
        logger.warning(
            "mygene is not installed; skipping MyGene-based synonym expansion "
            "(install mygene or use human-only gene_info synonyms)"
        )
        return {}

    clean = sorted({int(x) for x in entrez_ids if x and int(x) > 0})
    if not clean:
        return {}

    cache_path = os.path.join(output_dir, "entrez_mygene_synonyms_cache.json")
    cached: Dict[int, List[str]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in (raw or {}).items():
                try:
                    gid = int(k)
                except Exception:
                    continue
                if isinstance(v, list):
                    cached[gid] = [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass

    missing = [g for g in clean if g not in cached]
    if missing:
        mg = mygene.MyGeneInfo()
        batch_sz = 900
        for i in range(0, len(missing), batch_sz):
            batch = missing[i : i + batch_sz]
            time.sleep(delay)
            try:
                hits = mg.getgenes(
                    batch,
                    fields="alias,symbol,name,other_names",
                    as_dataframe=False,
                )
            except Exception as e:
                logger.warning(f"MyGene getgenes synonym batch failed: {e}")
                hits = []
            for doc in hits or []:
                if not isinstance(doc, dict):
                    continue
                try:
                    gid = int(doc.get("_id"))
                except Exception:
                    continue
                names: set[str] = set()
                for key in ("symbol", "name"):
                    t = _normalize_term(doc.get(key))
                    if t:
                        names.add(t)
                alias = doc.get("alias")
                if isinstance(alias, list):
                    for x in alias:
                        t = _normalize_term(x)
                        if t:
                            names.add(t)
                elif isinstance(alias, str) and alias.strip():
                    for part in re.split(r"[,;]", alias):
                        t = _normalize_term(part)
                        if t:
                            names.add(t)
                other = doc.get("other_names")
                if isinstance(other, str) and other.strip():
                    for part in re.split(r"[,;]", other):
                        t = _normalize_term(part)
                        if t:
                            names.add(t)
                elif isinstance(other, list):
                    for x in other:
                        t = _normalize_term(x)
                        if t:
                            names.add(t)
                cached[gid] = sorted(names)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({str(k): cached.get(k, []) for k in sorted(cached.keys())}, f, indent=2)
        except Exception:
            pass

    return {g: cached.get(g, []) for g in clean}


def _build_europepmc_text_query_pass1(
    row: pd.Series, prefix: str = "query"
) -> Tuple[Optional[str], List[str]]:
    """
    Text query pass 1 (organism-specific): locus_tag and genbank accession.

    Uses available identifiers and ORs them together (for the given prefix):
        <prefix>_locus_tag, <prefix>_genbank_acc

    Returns:
        (query_string or None if no identifiers, identifiers_used_types)
    """
    id_terms: List[Tuple[str, str]] = []  # (type, value)

    locus_tag = _normalize_term(row.get(f"{prefix}_locus_tag"))
    if locus_tag:
        id_terms.append(("locus_tag", locus_tag))

    genbank_acc = _normalize_term(row.get(f"{prefix}_genbank_acc"))
    if genbank_acc:
        id_terms.append(("genbank_acc", genbank_acc))
        # Europe PMC indexing sometimes omits the version dot; search both forms.
        if "." in genbank_acc:
            stem = genbank_acc.split(".", 1)[0].strip()
            if stem and stem != genbank_acc:
                id_terms.append(("genbank_acc_stem", stem))

    if not id_terms:
        return None, []

    seen_vals: set[str] = set()
    deduped: List[Tuple[str, str]] = []
    for kind, val in id_terms:
        lk = val.lower()
        if lk in seen_vals:
            continue
        seen_vals.add(lk)
        deduped.append((kind, val))
    id_terms = deduped

    or_clauses = []
    for (_kind, val) in id_terms:
        esc = val.replace('"', '\\"')
        or_clauses.append(f'(TITLE_ABS:"{esc}" OR BODY:"{esc}")')
    or_part = " OR ".join(or_clauses)
    query = f"({or_part})"

    id_types_used = [kind for (kind, _val) in id_terms]
    return query, id_types_used


def _build_europepmc_text_query_from_terms(
    id_terms: List[Tuple[str, str]], taxid: Optional[int]
) -> Tuple[Optional[str], List[str]]:
    # Dedupe/clean terms and force title/abstract scoped text search.
    seen_terms: set[str] = set()
    cleaned_terms: List[Tuple[str, str]] = []
    for kind, val in id_terms:
        v = val.strip()
        if not v or v == "-" or len(v) < 2 or len(v) > 120:
            continue
        lk = v.lower()
        if lk in seen_terms:
            continue
        seen_terms.add(lk)
        cleaned_terms.append((kind, v))
    if not cleaned_terms:
        return None, []
    cleaned_terms = cleaned_terms[:30]

    or_clauses = []
    for (_kind, val) in cleaned_terms:
        esc = val.replace('"', '\\"')
        # Use full-text-aware querying: match in title/abstract or full-text body.
        or_clauses.append(f'(TITLE_ABS:"{esc}" OR BODY:"{esc}")')
    or_part = " OR ".join(or_clauses)
    # Do not require HAS_FT at search stage; some paywalled records can still
    # be retrievable downstream through publisher/PMID/DOI routes.
    query = f"({or_part})"
    if taxid is not None:
        query = f"{query} AND ORGANISM_ID:{int(taxid)}"

    id_types_used = [kind for (kind, _val) in cleaned_terms]
    return query, id_types_used


def _build_europepmc_text_query_pass2(
    row: pd.Series,
    taxid: Optional[int],
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Combined pass2 query (base gene/common terms + synonyms).
    """
    id_terms: List[Tuple[str, str]] = []
    gene_name = _normalize_term(row.get(f"{prefix}_gene_name"))
    if gene_name:
        id_terms.append(("gene_name", gene_name))
    common_name = _normalize_term(row.get(f"{prefix}_common_name"))
    if common_name:
        id_terms.append(("common_name", common_name))
    if extra_terms:
        for t in extra_terms:
            tt = _normalize_term(t)
            if tt:
                id_terms.append(("synonym", tt))
    return _build_europepmc_text_query_from_terms(id_terms, taxid)


def _build_europepmc_text_query_pass2_base_only(
    row: pd.Series, taxid: Optional[int], prefix: str = "query"
) -> Tuple[Optional[str], List[str]]:
    id_terms: List[Tuple[str, str]] = []
    gene_name = _normalize_term(row.get(f"{prefix}_gene_name"))
    if gene_name:
        id_terms.append(("gene_name", gene_name))
    common_name = _normalize_term(row.get(f"{prefix}_common_name"))
    if common_name:
        id_terms.append(("common_name", common_name))
    return _build_europepmc_text_query_from_terms(id_terms, taxid)


def _build_europepmc_text_query_pass2_synonym_only(
    taxid: Optional[int], extra_terms: Optional[List[str]] = None
) -> Tuple[Optional[str], List[str]]:
    id_terms: List[Tuple[str, str]] = []
    if extra_terms:
        for t in extra_terms:
            tt = _normalize_term(t)
            if tt:
                id_terms.append(("synonym", tt))
    return _build_europepmc_text_query_from_terms(id_terms, taxid)


def _collect_base_terms_for_pass2(row: pd.Series, prefix: str = "query") -> List[str]:
    out: List[str] = []
    gene_name = _normalize_term(row.get(f"{prefix}_gene_name"))
    common_name = _normalize_term(row.get(f"{prefix}_common_name"))
    if gene_name:
        out.append(gene_name)
    if common_name:
        out.append(common_name)
    return out


def _extract_paper_id(rec: Dict[str, Any]) -> Optional[str]:
    """
    Choose a stable identifier for a Europe PMC record.

    Preference order: DOI > PMID > PMC ID > Europe PMC internal ID.
    """
    doi = rec.get("doi")
    if doi:
        return doi
    pmid = rec.get("pmid")
    if pmid:
        return f"PMID:{pmid}"
    pmcid = rec.get("pmcid")
    if pmcid:
        return pmcid
    # Fallback to Europe PMC internal ID
    rec_id = rec.get("id")
    if rec_id:
        return f"EPMC:{rec_id}"
    return None


def _run_europepmc_search_query(
    query: str,
    session: requests.Session,
    cache: Dict[str, Dict[str, List[str]]],
    delay: float = 0.35,
) -> Dict[str, List[str]]:
    """Run a Europe PMC search for a query string with caching."""
    if query in cache:
        logger.debug(f"Cache hit for query={query!r}")
        return cache[query]

    logger.debug(f"Europe PMC query: {query}")

    params = {
        "query": query,
        "format": "json",
        "pageSize": 200,
    }
    dois: List[str] = []
    titles: List[str] = []

    try:
        resp = session.get(EUROPEPMC_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Europe PMC search failed for query={query!r}: {e}")
        result = {"dois": [], "titles": []}
        cache[query] = result
        return result
    finally:
        time.sleep(delay)

    records = (data.get("resultList") or {}).get("result") or []
    logger.debug(f"Europe PMC returned {len(records)} results for query={query!r}")

    for rec in records:
        if not _is_research_article_record(rec):
            continue
        pid = _extract_paper_id(rec)
        if not pid:
            continue
        title = rec.get("title") or ""
        dois.append(pid)
        titles.append(title)

    result = {"dois": dois, "titles": titles}
    cache[query] = result
    return result


def run_europepmc_search_for_row(
    row: pd.Series,
    taxid: Optional[int],
    session: requests.Session,
    cache: Dict[str, Dict[str, List[str]]],
    delay: float = 0.35,
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a Europe PMC search for a single mapping row.

    Returns dict with keys: "dois", "titles", "pass1_count", "pass2_count".
    """
    row_label = str(row.get(prefix) or row.get(f"{prefix}_id") or "")
    base_terms = _collect_base_terms_for_pass2(row, prefix=prefix)
    syn_terms = [t for t in (extra_terms or []) if _normalize_term(t)]
    logger.debug(
        f"[{prefix}] pass2 term summary for row={row_label!r}: "
        f"base_terms_n={len(base_terms)} base_terms_sample={base_terms[:6]} "
        f"synonym_terms_n={len(syn_terms)} synonym_terms_sample={syn_terms[:6]}"
    )

    q1, q1_types = _build_europepmc_text_query_pass1(row, prefix=prefix)
    q2_base, q2_base_types = _build_europepmc_text_query_pass2_base_only(
        row, taxid, prefix=prefix
    )
    q2_syn, q2_syn_types = _build_europepmc_text_query_pass2_synonym_only(
        taxid, extra_terms=extra_terms
    )

    if not q1 and not q2_base and not q2_syn:
        return {"dois": [], "titles": [], "pass1_count": 0, "pass2_count": 0}

    id_to_title: Dict[str, str] = {}
    pass1_dois: List[str] = []
    pass2_dois: List[str] = []
    pass2_base_dois: List[str] = []
    pass2_synonym_dois: List[str] = []
    pass1_n = 0
    pass2_n = 0
    pass2_base_n = 0
    pass2_synonym_n = 0
    pass2_overlap_n = 0

    if q1:
        logger.debug(f"Europe PMC text pass1 ids={q1_types}")
        r1 = _run_europepmc_search_query(q1, session, cache, delay=delay)
        for pid, title in zip(r1["dois"], r1["titles"]):
            if pid not in id_to_title:
                id_to_title[pid] = title
        pass1_dois = list(r1["dois"])
        pass1_n = len(pass1_dois)

    if q2_base:
        logger.debug(f"Europe PMC text pass2-base ids={q2_base_types} taxid={taxid}")
        logger.debug(f"[{prefix}] pass2-base query: {q2_base}")
        r2_base = _run_europepmc_search_query(q2_base, session, cache, delay=delay)
        pass2_base_dois = list(r2_base["dois"])
        pass2_base_n = len(pass2_base_dois)
        if pass2_base_n == 0 and taxid is not None:
            q2_base_alt, q2_base_alt_types = _build_europepmc_text_query_pass2_base_only(
                row, None, prefix=prefix
            )
            if q2_base_alt:
                logger.debug(
                    f"Europe PMC pass2-base retry without ORGANISM_ID (orig taxid={taxid}) ids={q2_base_alt_types}"
                )
                r2_base_alt = _run_europepmc_search_query(
                    q2_base_alt, session, cache, delay=delay
                )
                pass2_base_dois = list(r2_base_alt["dois"])
                pass2_base_n = len(pass2_base_dois)

    if q2_syn:
        logger.debug(
            f"Europe PMC text pass2-synonym ids={q2_syn_types} taxid={taxid}"
        )
        logger.debug(f"[{prefix}] pass2-synonym query: {q2_syn}")
        r2_syn = _run_europepmc_search_query(q2_syn, session, cache, delay=delay)
        pass2_synonym_dois = list(r2_syn["dois"])
        pass2_synonym_n = len(pass2_synonym_dois)
        if pass2_synonym_n == 0 and taxid is not None:
            q2_syn_alt, q2_syn_alt_types = _build_europepmc_text_query_pass2_synonym_only(
                None, extra_terms=extra_terms
            )
            if q2_syn_alt:
                logger.debug(
                    f"Europe PMC pass2-synonym retry without ORGANISM_ID (orig taxid={taxid}) ids={q2_syn_alt_types}"
                )
                r2_syn_alt = _run_europepmc_search_query(
                    q2_syn_alt, session, cache, delay=delay
                )
                pass2_synonym_dois = list(r2_syn_alt["dois"])
                pass2_synonym_n = len(pass2_synonym_dois)

    pass2_base_set = set(pass2_base_dois)
    pass2_syn_set = set(pass2_synonym_dois)
    pass2_overlap = pass2_base_set & pass2_syn_set
    pass2_overlap_n = len(pass2_overlap)
    pass2_union: List[str] = []
    seen_union: set[str] = set()
    for pid in pass2_base_dois + pass2_synonym_dois:
        if pid in seen_union:
            continue
        seen_union.add(pid)
        pass2_union.append(pid)
    pass2_dois = pass2_union
    pass2_n = len(pass2_dois)

    for pid in pass2_dois:
        if pid not in id_to_title:
            # Prefer base title map first, then synonym map.
            title = ""
            if 'r2_base' in locals():
                idxs = [i for i, x in enumerate(r2_base["dois"]) if x == pid]
                if idxs:
                    title = r2_base["titles"][idxs[0]]
            if not title and 'r2_syn' in locals():
                idxs = [i for i, x in enumerate(r2_syn["dois"]) if x == pid]
                if idxs:
                    title = r2_syn["titles"][idxs[0]]
            if not title and 'r2_base_alt' in locals():
                idxs = [i for i, x in enumerate(r2_base_alt["dois"]) if x == pid]
                if idxs:
                    title = r2_base_alt["titles"][idxs[0]]
            if not title and 'r2_syn_alt' in locals():
                idxs = [i for i, x in enumerate(r2_syn_alt["dois"]) if x == pid]
                if idxs:
                    title = r2_syn_alt["titles"][idxs[0]]
            id_to_title[pid] = title

    merged_dois = list(id_to_title.keys())
    merged_titles = [id_to_title[pid] for pid in merged_dois]
    logger.debug(
        f"[{prefix}] Europe PMC text merged: pass1={pass1_n}, pass2_total={pass2_n}, "
        f"pass2_base={pass2_base_n}, pass2_synonym={pass2_synonym_n}, "
        f"pass2_overlap={pass2_overlap_n}, unique={len(merged_dois)}"
    )
    return {
        "dois": merged_dois,
        "titles": merged_titles,
        "pass1_count": pass1_n,
        "pass2_count": pass2_n,
        "pass1_dois": pass1_dois,
        "pass2_dois": pass2_dois,
        "pass2_base_count": pass2_base_n,
        "pass2_synonym_count": pass2_synonym_n,
        "pass2_overlap_count": pass2_overlap_n,
        "pass2_base_dois": pass2_base_dois,
        "pass2_synonym_dois": pass2_synonym_dois,
        "pass2_overlap_dois": sorted(pass2_overlap),
    }


def _parse_accession_text_overlap(mode: Optional[str]) -> Tuple[bool, bool]:
    """Return (filter_query, filter_target): drop accession-only DOIs without text hit."""
    raw = mode if mode is not None else os.environ.get(
        "AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP", "off"
    )
    v = (raw or "").strip().lower()
    if v in ("", "0", "false", "no", "off", "none"):
        return False, False
    if v in ("1", "true", "yes", "on", "both", "all"):
        return True, True
    if v == "query":
        return True, False
    if v == "target":
        return False, True
    logger.warning(
        "Unknown accession_text_overlap / AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP=%r; using off",
        raw,
    )
    return False, False


def _drop_accession_only_without_text_hit(
    merged_dois: List[str],
    id_to_title: Dict[str, str],
    accession_dois: List[str],
    text_dois: List[str],
) -> Tuple[List[str], List[str], int]:
    """
    Remove DOIs that appear only in the Europe PMC UniProt-accession list and not
    in the text-search result list (pass1 + pass2 union).
    """
    acc_set = set(accession_dois)
    text_set = set(text_dois)
    out_dois: List[str] = []
    n_drop = 0
    for d in merged_dois:
        if d not in acc_set:
            out_dois.append(d)
            continue
        if d in text_set:
            out_dois.append(d)
            continue
        n_drop += 1
    out_titles = [id_to_title[pid] for pid in out_dois]
    return out_dois, out_titles, n_drop


def run(
    df: pd.DataFrame,
    query_id_col: str = "query",
    target_id_col: str = "target",
    taxid_col: Optional[str] = None,
    default_taxid: Optional[int] = None,
    query_taxid: Optional[int] = None,
    target_taxid: Optional[int] = None,
    output_dir: str = ".",
    delay: float = 0.35,
    use_cache: bool = True,
    accession_text_overlap: Optional[str] = None,
) -> pd.DataFrame:
    """
    Two-phase Europe PMC search for each alignment row:
      1. UniProt accession citations
         (ACCESSION_ID:<acc> AND ACCESSION_TYPE:uniprot) for query and target.
      2. Text search using locus_tag / GenBank accession / gene name / common name
         for both query and target, always run and merged with UniProt results.

    Args:
        df: DataFrame from mapping module (must contain query_id_col and
            target_id_col, plus query_* identifier columns for text fallback).
        query_id_col: Column for query protein UniProt ID.
        target_id_col: Column for target protein UniProt ID. Rows missing target
            get empty target_paper_dois/titles.
        taxid_col: Optional column giving per-row taxon ID (used only for text fallback).
        default_taxid: Optional fallback taxon ID when taxid_col is missing/None.
        query_taxid: Optional fixed taxon ID for query organism; if provided,
            query text searches use this instead of per-row taxid.
        target_taxid: Optional fixed taxon ID for target organism; if provided,
            target text searches use this instead of any per-row taxid.
        output_dir: Directory for logs and cache file (search_cache.json).
        use_cache: If True, load cache at start (if exists) and save at end.
        accession_text_overlap: If set, overrides env ``AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP``.
            Values: ``off``, ``query``, ``target``, ``both`` (or ``1``/``true`` via env).
            When enabled for a side, DOIs that appear only in the Europe PMC UniProt-accession
            search and not in the text-search union are dropped from merged outputs.

    Returns a new DataFrame with additional columns:
        - query_paper_dois, query_paper_titles (merged UniProt + text search)
        - target_paper_dois, target_paper_titles (merged UniProt + text search)
    """
    global _PUBTATOR_ENABLED, _PUBTATOR_DISABLED_REASON
    _PUBTATOR_ENABLED = True
    _PUBTATOR_DISABLED_REASON = ""

    force_ipv4_env = os.environ.get("AUTO_LIT_FORCE_IPV4", "1").strip().lower()
    if force_ipv4_env in {"1", "true", "yes", "on"}:
        _force_ipv4_resolution()

    _configure_file_logging(output_dir)

    if query_id_col not in df.columns:
        raise ValueError(f"query_id_col={query_id_col!r} not found in DataFrame columns")
    if target_id_col not in df.columns:
        raise ValueError(f"target_id_col={target_id_col!r} not found in DataFrame columns")

    result_df = df.copy()
    n_rows = len(result_df)
    filter_q_acc, filter_t_acc = _parse_accession_text_overlap(accession_text_overlap)
    if filter_q_acc or filter_t_acc:
        logger.info(
            "Accession-only papers must overlap text search: query={} target={} "
            "(set accession_text_overlap= or AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP)",
            filter_q_acc,
            filter_t_acc,
        )
    logger.info(f"Search module – Entrez/PubTator first, Europe PMC fallback for {n_rows} rows (query col={query_id_col!r}, target col={target_id_col!r})")

    session = requests.Session()
    # NCBI (eutils) may behave better when requests include an explicit UA.
    session.headers.setdefault(
        "User-Agent",
        "auto_lit_search/0.1 (contact: research pipeline; requests to NCBI E-utilities)",
    )
    # Policy: skip PubTator entirely for now; use Europe PMC retrieval only.
    _set_pubtator_disabled("Disabled by policy: Europe PMC-only text search mode")
    cache_path = os.path.join(output_dir, "search_cache.json")
    trace_path = os.path.join(output_dir, "search_trace.jsonl")
    uniprot_cache: Dict[str, Dict[str, List[str]]] = {}
    text_cache: Dict[str, Dict[str, List[str]]] = {}
    pubtator_cache: Dict[int, List[str]] = {}

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            uniprot_cache = loaded.get("uniprot") or {}
            text_cache = loaded.get("text") or {}
            logger.info(f"Loaded cache from {cache_path} ({len(uniprot_cache)} uniprot, {len(text_cache)} text entries)")
        except Exception as e:
            logger.warning(f"Could not load cache from {cache_path}: {e}")

    # Build synonym map from NCBI human gene_info for text fallback expansion.
    # This is best-effort and only applies for IDs present in that file.
    all_entrez_ids: List[int] = []
    if "query_entrez_id" in result_df.columns:
        for x in result_df["query_entrez_id"].dropna().tolist():
            gid = _normalize_entrez_id(x)
            if gid is not None:
                all_entrez_ids.append(gid)
    if "target_entrez_id" in result_df.columns:
        for x in result_df["target_entrez_id"].dropna().tolist():
            gid = _normalize_entrez_id(x)
            if gid is not None:
                all_entrez_ids.append(gid)
    gene_synonyms_by_entrez: Dict[int, List[str]] = {}
    if all_entrez_ids:
        try:
            gene_synonyms_by_entrez = _load_human_gene_name_synonyms(
                session=session,
                entrez_ids=all_entrez_ids,
                output_dir=output_dir,
                delay=delay,
            )
            logger.info(
                f"Loaded synonym sets for {len(gene_synonyms_by_entrez)} Entrez IDs from NCBI gene_info"
            )
            if gene_synonyms_by_entrez:
                sample_items = list(gene_synonyms_by_entrez.items())[:5]
                for gid, names in sample_items:
                    logger.debug(
                        f"Synonym map sample gene_id={gid}: n={len(names)} sample={names[:8]}"
                    )
        except Exception as e:
            logger.warning(f"Could not load NCBI gene synonym map: {e}")

    if all_entrez_ids:
        try:
            mg_syn = _load_mygene_synonyms_for_entrez(
                all_entrez_ids, output_dir=output_dir, delay=delay
            )
            n_mg = 0
            for gid, names in mg_syn.items():
                if not names:
                    continue
                merged = set(gene_synonyms_by_entrez.get(gid, []))
                n_before = len(merged)
                merged.update(names)
                if len(merged) > n_before:
                    n_mg += 1
                gene_synonyms_by_entrez[gid] = sorted(merged)
            logger.info(
                f"MyGene synonym merge: expanded {n_mg} Entrez IDs "
                f"(total IDs with any synonyms: {len(gene_synonyms_by_entrez)})"
            )
        except Exception as e:
            logger.warning(f"MyGene synonym merge failed: {e}")

    query_dois_col: List[str] = []
    query_titles_col: List[str] = []
    target_dois_col: List[str] = []
    target_titles_col: List[str] = []
    query_paper_counts_col: List[Dict[str, int]] = []
    query_paper_ids_by_source_col: List[str] = []
    target_paper_ids_by_source_col: List[str] = []

    query_ids_with_papers: set = set()
    query_ids_seen: set = set()
    target_ids_with_papers: set = set()
    target_ids_seen: set = set()
    rows_query_from_entrez_pubtator = 0
    rows_query_from_europepmc_accession = 0
    rows_query_from_text = 0
    rows_target_from_entrez_pubtator = 0
    rows_target_from_europepmc_accession = 0
    rows_target_from_text = 0

    # Machine-readable per-row trace of the search logic.
    # Each line is a JSON object.
    trace_fh = open(trace_path, "w", encoding="utf-8")

    for idx, row in result_df.iterrows():
        row_taxid: Optional[int] = None
        if taxid_col and taxid_col in result_df.columns:
            val = row.get(taxid_col)
            try:
                row_taxid = int(val) if pd.notna(val) else None
            except Exception:
                row_taxid = None
        elif default_taxid is not None:
            row_taxid = int(default_taxid)

        q_acc = _normalize_uniprot_id(row.get(query_id_col))
        t_acc = _normalize_uniprot_id(row.get(target_id_col))

        # ------------------------------------------------------------
        # Query side: Europe PMC-only retrieval (PubTator disabled)
        # ------------------------------------------------------------
        merged_q_dois: List[str] = []
        merged_q_titles: List[str] = []
        query_paper_counts: Dict[str, int] = {
            "entrez_pubtator": 0,
            "europepmc_accession": 0,
            "text_pass1": 0,
            "text_pass2": 0,
            "text_pass2_base": 0,
            "text_pass2_synonym": 0,
            "text_pass2_overlap": 0,
        }
        query_paper_ids_by_source: Dict[str, List[str]] = {
            "entrez_pubtator": [],
            "europepmc_accession": [],
            "text_pass1": [],
            "text_pass2": [],
            "text_pass2_base": [],
            "text_pass2_synonym": [],
            "text_pass2_overlap": [],
        }

        q_entrez_id = _normalize_entrez_id(row.get("query_entrez_id"))
        q_from_entrez = q_entrez_id is not None
        q_pubtator_used = False
        q_pubtator_empty = False
        if q_entrez_id is not None:
            query_ids_seen.add(str(q_entrez_id))
            q_pubtator_empty = True

        # Fallback: UniProt citation + Europe PMC text search.
        if not merged_q_dois:
            query_res = run_europepmc_crossref(q_acc, session, uniprot_cache)
            if q_acc:
                query_ids_seen.add(q_acc)
                if query_res["dois"]:
                    rows_query_from_europepmc_accession += 1

            query_text_taxid: Optional[int] = (
                query_taxid if query_taxid is not None else row_taxid
            )
            text_res_query = run_europepmc_search_for_row(
                row,
                query_text_taxid,
                session,
                text_cache,
                prefix="query",
                extra_terms=gene_synonyms_by_entrez.get(q_entrez_id or -1, []),
            )
            if text_res_query["dois"]:
                rows_query_from_text += 1

            # Merge UniProt and text results for query (deduplicated by paper ID)
            q_id_to_title: Dict[str, str] = {}
            for pid, title in zip(query_res["dois"], query_res["titles"]):
                if pid not in q_id_to_title:
                    q_id_to_title[pid] = title
            for pid, title in zip(text_res_query["dois"], text_res_query["titles"]):
                if pid not in q_id_to_title:
                    q_id_to_title[pid] = title
            merged_q_dois = list(q_id_to_title.keys())
            merged_q_titles = [q_id_to_title[pid] for pid in merged_q_dois]

            acc_only_drop_q = 0
            if filter_q_acc:
                merged_q_dois, merged_q_titles, acc_only_drop_q = (
                    _drop_accession_only_without_text_hit(
                        merged_q_dois,
                        q_id_to_title,
                        query_res["dois"],
                        text_res_query["dois"],
                    )
                )

            if q_acc and merged_q_dois:
                query_ids_with_papers.add(q_acc)

            query_paper_counts = {
                "entrez_pubtator": 0,
                "europepmc_accession": len(query_res["dois"]),
                "text_pass1": text_res_query.get("pass1_count", 0),
                "text_pass2": text_res_query.get("pass2_count", 0),
                "text_pass2_base": text_res_query.get("pass2_base_count", 0),
                "text_pass2_synonym": text_res_query.get("pass2_synonym_count", 0),
                "text_pass2_overlap": text_res_query.get("pass2_overlap_count", 0),
                "accession_only_dropped": acc_only_drop_q,
            }
            query_paper_ids_by_source = {
                "entrez_pubtator": [],
                "europepmc_accession": list(query_res["dois"]),
                "text_pass1": list(text_res_query.get("pass1_dois", [])),
                "text_pass2": list(text_res_query.get("pass2_dois", [])),
                "text_pass2_base": list(text_res_query.get("pass2_base_dois", [])),
                "text_pass2_synonym": list(
                    text_res_query.get("pass2_synonym_dois", [])
                ),
                "text_pass2_overlap": list(
                    text_res_query.get("pass2_overlap_dois", [])
                ),
            }

        query_dois_col.append(json.dumps(merged_q_dois))
        query_titles_col.append(json.dumps(merged_q_titles))
        query_paper_counts_col.append(query_paper_counts)
        query_paper_ids_by_source_col.append(json.dumps(query_paper_ids_by_source))

        # ------------------------------------------------------------
        # Target side: Europe PMC-only retrieval (PubTator disabled)
        # ------------------------------------------------------------
        merged_t_dois: List[str] = []
        merged_t_titles: List[str] = []
        target_paper_counts: Dict[str, int] = {
            "entrez_pubtator": 0,
            "europepmc_accession": 0,
            "text_pass1": 0,
            "text_pass2": 0,
            "text_pass2_base": 0,
            "text_pass2_synonym": 0,
            "text_pass2_overlap": 0,
        }
        target_paper_ids_by_source: Dict[str, List[str]] = {
            "entrez_pubtator": [],
            "europepmc_accession": [],
            "text_pass1": [],
            "text_pass2": [],
            "text_pass2_base": [],
            "text_pass2_synonym": [],
            "text_pass2_overlap": [],
        }

        t_entrez_id = _normalize_entrez_id(row.get("target_entrez_id"))
        t_from_entrez = t_entrez_id is not None
        t_pubtator_used = False
        t_pubtator_empty = False
        if t_entrez_id is not None:
            target_ids_seen.add(str(t_entrez_id))
            t_pubtator_empty = True

        # Fallback: UniProt citation + Europe PMC text search.
        if not merged_t_dois:
            target_res = run_europepmc_crossref(t_acc, session, uniprot_cache)
            target_text_taxid: Optional[int] = (
                target_taxid if target_taxid is not None else row_taxid
            )
            text_res_target = run_europepmc_search_for_row(
                row,
                target_text_taxid,
                session,
                text_cache,
                prefix="target",
                extra_terms=gene_synonyms_by_entrez.get(t_entrez_id or -1, []),
            )

            if t_acc:
                target_ids_seen.add(t_acc)
                if target_res["dois"]:
                    rows_target_from_europepmc_accession += 1
                if text_res_target["dois"]:
                    rows_target_from_text += 1

            # Merge UniProt and text results for target (deduplicated by paper ID)
            t_id_to_title: Dict[str, str] = {}
            for pid, title in zip(target_res["dois"], target_res["titles"]):
                if pid not in t_id_to_title:
                    t_id_to_title[pid] = title
            for pid, title in zip(text_res_target["dois"], text_res_target["titles"]):
                if pid not in t_id_to_title:
                    t_id_to_title[pid] = title
            merged_t_dois = list(t_id_to_title.keys())
            merged_t_titles = [t_id_to_title[pid] for pid in merged_t_dois]

            acc_only_drop_t = 0
            if filter_t_acc:
                merged_t_dois, merged_t_titles, acc_only_drop_t = (
                    _drop_accession_only_without_text_hit(
                        merged_t_dois,
                        t_id_to_title,
                        target_res["dois"],
                        text_res_target["dois"],
                    )
                )

            if t_acc and merged_t_dois:
                target_ids_with_papers.add(t_acc)

            target_paper_ids_by_source = {
                "entrez_pubtator": [],
                "europepmc_accession": list(target_res["dois"]),
                "text_pass1": list(text_res_target.get("pass1_dois", [])),
                "text_pass2": list(text_res_target.get("pass2_dois", [])),
                "text_pass2_base": list(text_res_target.get("pass2_base_dois", [])),
                "text_pass2_synonym": list(
                    text_res_target.get("pass2_synonym_dois", [])
                ),
                "text_pass2_overlap": list(
                    text_res_target.get("pass2_overlap_dois", [])
                ),
            }
            target_paper_counts = {
                "entrez_pubtator": 0,
                "europepmc_accession": len(target_res["dois"]),
                "text_pass1": text_res_target.get("pass1_count", 0),
                "text_pass2": text_res_target.get("pass2_count", 0),
                "text_pass2_base": text_res_target.get("pass2_base_count", 0),
                "text_pass2_synonym": text_res_target.get("pass2_synonym_count", 0),
                "text_pass2_overlap": text_res_target.get("pass2_overlap_count", 0),
                "accession_only_dropped": acc_only_drop_t,
            }

        target_dois_col.append(json.dumps(merged_t_dois))
        target_titles_col.append(json.dumps(merged_t_titles))
        target_paper_ids_by_source_col.append(json.dumps(target_paper_ids_by_source))

        # Write per-row trace (useful for debugging why a gene got 0 hits).
        trace_obj = {
            "row_idx": idx,
            "query": str(row.get(query_id_col)),
            "target": str(row.get(target_id_col)),
            "query_entrez_id": q_entrez_id,
            "target_entrez_id": t_entrez_id,
            "query_uniprot": q_acc,
            "target_uniprot": t_acc,
            "query_pubtator_used": q_pubtator_used,
            "query_pubtator_empty": q_pubtator_empty,
            "pubtator_enabled": _PUBTATOR_ENABLED,
            "pubtator_disabled_reason": _PUBTATOR_DISABLED_REASON,
            "query_counts": query_paper_counts,
            "accession_text_overlap_filter_query": filter_q_acc,
            "query_dois_n": len(merged_q_dois),
            "query_dois_sample": merged_q_dois[:10],
            "target_pubtator_used": t_pubtator_used,
            "target_pubtator_empty": t_pubtator_empty,
            "target_counts": target_paper_counts,
            "accession_text_overlap_filter_target": filter_t_acc,
            "target_dois_n": len(merged_t_dois),
            "target_dois_sample": merged_t_dois[:10],
        }
        trace_fh.write(json.dumps(trace_obj, ensure_ascii=False) + "\n")

    trace_fh.close()

    result_df["query_paper_dois"] = query_dois_col
    result_df["query_paper_titles"] = query_titles_col
    result_df["target_paper_dois"] = target_dois_col
    result_df["target_paper_titles"] = target_titles_col
    result_df["query_paper_counts"] = query_paper_counts_col
    result_df["query_paper_ids_by_source"] = query_paper_ids_by_source_col
    result_df["target_paper_ids_by_source"] = target_paper_ids_by_source_col

    if use_cache:
        try:
            merged = {"uniprot": {**uniprot_cache}, "text": {**text_cache}}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            logger.info(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save cache to {cache_path}: {e}")

    n_query_with_hits = sum(1 for v in query_dois_col if json.loads(v))
    n_target_with_hits = sum(1 for v in target_dois_col if json.loads(v))
    rows_zero_query_papers = sum(1 for v in query_dois_col if not json.loads(v))
    rows_zero_target_papers = sum(1 for v in target_dois_col if not json.loads(v))

    logger.info(
        f"Query identifiers: {len(query_ids_seen)} unique IDs, {len(query_ids_with_papers)} with ≥1 paper, "
        f"{sum(len(json.loads(v)) for v in query_dois_col)} total query papers"
    )
    logger.info(
        f"Target IDs: {len(target_ids_seen)} unique IDs, {len(target_ids_with_papers)} with ≥1 paper, "
        f"{sum(len(json.loads(v)) for v in target_dois_col)} total target papers"
    )
    logger.info(
        f"Query source: {rows_query_from_entrez_pubtator} rows from Entrez->PubTator, "
        f"{rows_query_from_europepmc_accession} rows from Europe PMC accession, "
        f"{rows_query_from_text} rows from text fallback"
    )
    logger.info(
        f"Target source: {rows_target_from_entrez_pubtator} rows from Entrez->PubTator, "
        f"{rows_target_from_europepmc_accession} rows from Europe PMC accession, "
        f"{rows_target_from_text} rows from text fallback"
    )
    logger.info(
        f"Rows with zero query papers: {rows_zero_query_papers}; with zero target papers: {rows_zero_target_papers}"
    )
    logger.info(f"Search module – {n_query_with_hits}/{n_rows} rows with ≥1 query paper; {n_target_with_hits}/{n_rows} with ≥1 target paper")

    return result_df


def _result_df_to_query_keyed_json(
    result_df: pd.DataFrame,
    query_id_col: str = "query",
    target_id_col: str = "target",
) -> Dict[str, List[Dict[str, Any]]]:
    """Build query-keyed structure for JSON output."""
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in result_df.iterrows():
        q = row.get(query_id_col)
        t = row.get(target_id_col)
        if pd.isna(q) or q is None:
            q = ""
        if pd.isna(t) or t is None:
            t = ""
        q = str(q).strip()
        t = str(t).strip()
        query_dois = json.loads(row["query_paper_dois"]) if isinstance(row["query_paper_dois"], str) else row["query_paper_dois"]
        query_titles = json.loads(row["query_paper_titles"]) if isinstance(row["query_paper_titles"], str) else row["query_paper_titles"]
        target_dois = json.loads(row["target_paper_dois"]) if isinstance(row["target_paper_dois"], str) else row["target_paper_dois"]
        target_titles = json.loads(row["target_paper_titles"]) if isinstance(row["target_paper_titles"], str) else row["target_paper_titles"]
        query_ids_by_source = row.get("query_paper_ids_by_source")
        if isinstance(query_ids_by_source, str):
            query_ids_by_source = json.loads(query_ids_by_source) if query_ids_by_source else {
                "entrez_pubtator": [],
                "europepmc_accession": [],
                "text_pass1": [],
                "text_pass2": [],
                "text_pass2_base": [],
                "text_pass2_synonym": [],
                "text_pass2_overlap": [],
            }
        if query_ids_by_source is None or (isinstance(query_ids_by_source, float) and pd.isna(query_ids_by_source)):
            query_ids_by_source = {
                "entrez_pubtator": [],
                "europepmc_accession": [],
                "text_pass1": [],
                "text_pass2": [],
                "text_pass2_base": [],
                "text_pass2_synonym": [],
                "text_pass2_overlap": [],
            }
        target_ids_by_source = row.get("target_paper_ids_by_source")
        if isinstance(target_ids_by_source, str):
            target_ids_by_source = json.loads(target_ids_by_source) if target_ids_by_source else {
                "entrez_pubtator": [],
                "europepmc_accession": [],
                "text_pass1": [],
                "text_pass2": [],
                "text_pass2_base": [],
                "text_pass2_synonym": [],
                "text_pass2_overlap": [],
            }
        if target_ids_by_source is None or (isinstance(target_ids_by_source, float) and pd.isna(target_ids_by_source)):
            target_ids_by_source = {
                "entrez_pubtator": [],
                "europepmc_accession": [],
                "text_pass1": [],
                "text_pass2": [],
                "text_pass2_base": [],
                "text_pass2_synonym": [],
                "text_pass2_overlap": [],
            }
        counts = row.get("query_paper_counts")
        if isinstance(counts, str):
            counts = json.loads(counts) if counts else {
                "entrez_pubtator": 0,
                "europepmc_accession": 0,
                "text_pass1": 0,
                "text_pass2": 0,
                "text_pass2_base": 0,
                "text_pass2_synonym": 0,
                "text_pass2_overlap": 0,
            }
        if counts is None or (isinstance(counts, float) and pd.isna(counts)):
            counts = {
                "entrez_pubtator": 0,
                "europepmc_accession": 0,
                "text_pass1": 0,
                "text_pass2": 0,
                "text_pass2_base": 0,
                "text_pass2_synonym": 0,
                "text_pass2_overlap": 0,
            }
        entry = {
            "target": t,
            "n_query_papers": len(query_dois),
            "n_target_papers": len(target_dois),
            "query_paper_counts": counts,
            "query_paper_dois": query_dois,
            "query_paper_titles": query_titles,
            "query_paper_ids_by_source": query_ids_by_source,
            "target_paper_dois": target_dois,
            "target_paper_titles": target_titles,
            "target_paper_ids_by_source": target_ids_by_source,
        }
        by_query.setdefault(q, []).append(entry)
    return by_query


def main() -> int:
    """Command-line interface for the search module."""
    parser = argparse.ArgumentParser(
        description="Module 2: Europe PMC two-phase search (CROSS_REF then text fallback) for query and target proteins"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to mapping output CSV (from mapping.run).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output file (JSON or CSV depending on --output-format).",
    )
    parser.add_argument(
        "--query-id-col",
        type=str,
        default="query",
        help="Column name for query UniProt IDs (default: 'query').",
    )
    parser.add_argument(
        "--target-id-col",
        type=str,
        default="target",
        help="Column name for target UniProt IDs (default: 'target').",
    )
    parser.add_argument(
        "--taxid-col",
        type=str,
        default=None,
        help="Optional column name containing per-row taxon IDs.",
    )
    parser.add_argument(
        "--default-taxid",
        type=int,
        default=None,
        help="Optional default taxon ID to use when no per-row taxid is provided.",
    )
    parser.add_argument(
        "--query-taxid",
        type=int,
        default=None,
        help="Optional fixed taxon ID for query organism; overrides per-row taxid for query text searches.",
    )
    parser.add_argument(
        "--target-taxid",
        type=int,
        default=None,
        help="Optional fixed taxon ID for target organism; overrides per-row taxid for target text searches.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for logs and debug files (default: directory of --output).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not load or save search cache.",
    )
    parser.add_argument(
        "--accession-text-overlap",
        type=str,
        default=None,
        choices=["off", "query", "target", "both"],
        help=(
            "Drop DOIs found only via Europe PMC UniProt-accession search unless they "
            "also appear in the text-search union. Default uses env "
            "AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP if set, else off."
        ),
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format: json (query-keyed) or csv (default: json).",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading mapping input: {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 1

    result_df = run(
        df,
        query_id_col=args.query_id_col,
        target_id_col=args.target_id_col,
        taxid_col=args.taxid_col,
        default_taxid=args.default_taxid,
        accession_text_overlap=args.accession_text_overlap,
        query_taxid=args.query_taxid,
        target_taxid=args.target_taxid,
        output_dir=output_dir,
        use_cache=not args.no_cache,
    )

    logger.info(f"Writing search output to: {args.output}")
    try:
        if args.output_format == "json":
            out_data = _result_df_to_query_keyed_json(
                result_df,
                query_id_col=args.query_id_col,
                target_id_col=args.target_id_col,
            )
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=2)
        else:
            result_df.to_csv(args.output, index=False)
    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

