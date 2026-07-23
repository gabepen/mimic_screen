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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

TaxidInput = Optional[Union[int, Sequence[int]]]


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


class RequestGate:
    """Serialize outbound API pacing across threads (min interval between starts)."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if sleep_for:
            time.sleep(sleep_for)


class LockedCache(dict):
    """Dict with a lock for concurrent get/set of cached API responses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def get_if_present(self, key: Any) -> Tuple[bool, Any]:
        with self._lock:
            if key in self:
                return True, self[key]
            return False, None

    def store(self, key: Any, value: Any) -> Any:
        with self._lock:
            existing = self.get(key)
            if existing is not None:
                return existing
            self[key] = value
            return value


_thread_local = threading.local()


def _thread_session(base: Optional[requests.Session] = None) -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        if base is not None:
            sess.headers.update(base.headers)
        else:
            sess.headers.setdefault(
                "User-Agent",
                "auto_lit_search/0.1 (contact: research pipeline; requests to NCBI E-utilities)",
            )
        _thread_local.session = sess
    return sess


def _search_workers_default() -> int:
    raw = os.environ.get("AUTO_LIT_SEARCH_WORKERS", "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


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
    gate: Optional[RequestGate] = None,
) -> Dict[str, List[str]]:
    """
    Search Europe PMC for UniProt accession citations (ACCESSION_ID + ACCESSION_TYPE).
    Returns dict with keys "dois", "titles". Uses resultType=core.
    """
    acc = _normalize_uniprot_id(uniprot_id)
    if not acc:
        return {"dois": [], "titles": []}

    if isinstance(cache, LockedCache):
        hit, cached = cache.get_if_present(acc)
        if hit:
            logger.debug(f"Cache hit for accession-cite uniprot:{acc}")
            return cached
    elif acc in cache:
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
    sess = _thread_session(session)

    try:
        if gate is not None:
            gate.wait()
        else:
            time.sleep(delay)
        resp = sess.get(EUROPEPMC_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Europe PMC accession-cite failed for uniprot:{acc}: {e}")
        result = {"dois": [], "titles": []}
        if isinstance(cache, LockedCache):
            return cache.store(acc, result)
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
    if isinstance(cache, LockedCache):
        return cache.store(acc, result)
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


def _locus_field_is_entrez_junk(val: str, entrez_id: Optional[Any]) -> bool:
    """
    True when a mapping \"locus_tag\" cell is actually an Entrez Gene ID or other
    useless numeric string for Europe PMC quoted search.
    """
    s = (val or "").strip()
    if not s:
        return True
    try:
        f = float(s)
        if f == int(f):
            return True
    except (ValueError, TypeError, OverflowError):
        pass
    if s.isdigit():
        return True
    if entrez_id is not None and str(entrez_id).strip() not in ("", "nan"):
        try:
            ei = int(float(str(entrez_id)))
            if s == str(ei):
                return True
        except (TypeError, ValueError):
            pass
    return False


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
        # Symbol + synonym tokens only (never full names / other designations).
        v = _normalize_term(r.get("Symbol"))
        if v and v != "-":
            names.add(v)
        syn = r.get("Synonyms")
        if pd.notna(syn) and str(syn).strip() != "-":
            names.update(
                [
                    x.strip()
                    for x in str(syn).split("|")
                    if x.strip() and x.strip() != "-"
                ]
            )

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
                    fields="alias,symbol",
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
                # symbol + alias only — never MyGene name / other_names prose
                t = _normalize_term(doc.get("symbol"))
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
    Text query pass 1 (organism-specific): locus_tag and GenBank / RefSeq accession.

    ORs together <prefix>_locus_tag (when not numeric junk) and <prefix>_genbank_acc
    (plus version stem when dotted).

    Returns:
        (query_string or None if no identifiers, identifiers_used_types)
    """
    id_terms: List[Tuple[str, str]] = []  # (type, value)

    locus_tag = _normalize_term(row.get(f"{prefix}_locus_tag"))
    entrez_for_side = row.get(f"{prefix}_entrez_id")
    if locus_tag and not _locus_field_is_entrez_junk(locus_tag, entrez_for_side):
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


def _collect_pass1_terms(
    row: pd.Series, prefix: str = "query"
) -> List[Tuple[str, str]]:
    """Return the exact deduplicated identifiers used by the pass1 text query."""
    id_terms: List[Tuple[str, str]] = []
    locus_tag = _normalize_term(row.get(f"{prefix}_locus_tag"))
    entrez_for_side = row.get(f"{prefix}_entrez_id")
    if locus_tag and not _locus_field_is_entrez_junk(locus_tag, entrez_for_side):
        id_terms.append(("locus_tag", locus_tag))

    genbank_acc = _normalize_term(row.get(f"{prefix}_genbank_acc"))
    if genbank_acc:
        id_terms.append(("genbank_acc", genbank_acc))
        if "." in genbank_acc:
            stem = genbank_acc.split(".", 1)[0].strip()
            if stem and stem != genbank_acc:
                id_terms.append(("genbank_acc_stem", stem))

    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for kind, term in id_terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append((kind, term))
    return out


def _clean_europepmc_text_terms(
    id_terms: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Apply the same filtering, deduplication, and cap used to build pass2 queries."""
    from .search_terms import is_usable_search_term

    seen_terms: set[str] = set()
    cleaned_terms: List[Tuple[str, str]] = []
    for kind, val in id_terms:
        v = val.strip()
        if not v or v == "-" or len(v) < 2 or len(v) > 120:
            continue
        if not is_usable_search_term(v, kind=kind):
            continue
        lk = v.lower()
        if lk in seen_terms:
            continue
        seen_terms.add(lk)
        cleaned_terms.append((kind, v))
    return cleaned_terms[:30]


def _normalize_taxids(taxids: TaxidInput) -> Tuple[int, ...]:
    if taxids is None:
        return ()
    values = [taxids] if isinstance(taxids, int) else list(taxids)
    return tuple(
        dict.fromkeys(int(value) for value in values if int(value) > 0)
    )


def _organism_query_clause(taxids: TaxidInput) -> Optional[str]:
    values = _normalize_taxids(taxids)
    if not values:
        return None
    if len(values) == 1:
        return f"ORGANISM_ID:{values[0]}"
    return "(" + " OR ".join(f"ORGANISM_ID:{value}" for value in values) + ")"


def _normalize_organism_terms(terms: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not terms:
        return ()
    return tuple(
        dict.fromkeys(
            value
            for raw in terms
            if (value := str(raw).strip())
        )
    )


def _organism_text_query_clause(
    organism_terms: Optional[Sequence[str]],
) -> Optional[str]:
    terms = _normalize_organism_terms(organism_terms)
    if not terms:
        return None
    clauses = []
    for term in terms:
        escaped = term.replace('"', '\\"')
        clauses.append(f'(TITLE_ABS:"{escaped}" OR BODY:"{escaped}")')
    return "(" + " OR ".join(clauses) + ")"


def _build_europepmc_text_query_from_terms(
    id_terms: List[Tuple[str, str]],
    taxid: TaxidInput,
    organism_terms: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    # Dedupe/clean terms and force title/abstract scoped text search.
    cleaned_terms = _clean_europepmc_text_terms(id_terms)
    if not cleaned_terms:
        return None, []

    or_clauses = []
    for kind, val in cleaned_terms:
        esc = val.replace('"', '\\"')
        if kind == "gene_name":
            or_clauses.append(f'(TITLE_ABS:"{esc}" OR BODY:"{esc}")')
        else:
            # Curated aliases can still be ambiguous (TIM, FIX, MCL). Requiring
            # title/abstract mention preserves synonym recall without accepting a
            # coincidental full-text-body occurrence as evidence for the gene.
            or_clauses.append(f'TITLE_ABS:"{esc}"')
    or_part = " OR ".join(or_clauses)
    # Do not require HAS_FT at search stage; some paywalled records can still
    # be retrievable downstream through publisher/PMID/DOI routes.
    query = f"({or_part})"
    organism_clause = _organism_query_clause(taxid)
    if organism_clause:
        query = f"{query} AND {organism_clause}"
    organism_text_clause = _organism_text_query_clause(organism_terms)
    if organism_text_clause:
        query = f"{query} AND {organism_text_clause}"

    id_types_used = [kind for (kind, _val) in cleaned_terms]
    return query, id_types_used


def _parse_idmap_gene_aliases(row: pd.Series, prefix: str = "query") -> List[str]:
    raw = row.get(f"{prefix}_gene_aliases")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    return [p.strip() for p in re.split(r"[|;]", s) if p.strip()]


def _pass2_id_terms(
    row: pd.Series,
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Build (kind, term) list for pass2: gene_name + idmap aliases + synonym extras."""
    id_terms: List[Tuple[str, str]] = []
    gene_name = _normalize_term(row.get(f"{prefix}_gene_name"))
    if gene_name:
        id_terms.append(("gene_name", gene_name))
    for alias in _parse_idmap_gene_aliases(row, prefix=prefix):
        tt = _normalize_term(alias)
        if tt:
            id_terms.append(("alias", tt))
    if extra_terms:
        for t in extra_terms:
            tt = _normalize_term(t)
            if tt:
                id_terms.append(("synonym", tt))
    return id_terms


def _build_europepmc_text_query_pass2(
    row: pd.Series,
    taxid: TaxidInput,
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    """Combined pass2 query (gene symbol + aliases + synonym extras)."""
    return _build_europepmc_text_query_from_terms(
        _pass2_id_terms(row, prefix=prefix, extra_terms=extra_terms), taxid
    )


def _build_europepmc_text_query_pass2_base_only(
    row: pd.Series, taxid: TaxidInput, prefix: str = "query"
) -> Tuple[Optional[str], List[str]]:
    return _build_europepmc_text_query_from_terms(
        _pass2_id_terms(row, prefix=prefix, extra_terms=None), taxid
    )


def _build_europepmc_text_query_pass2_synonym_only(
    taxid: TaxidInput, extra_terms: Optional[List[str]] = None
) -> Tuple[Optional[str], List[str]]:
    id_terms: List[Tuple[str, str]] = []
    if extra_terms:
        for t in extra_terms:
            tt = _normalize_term(t)
            if tt:
                id_terms.append(("synonym", tt))
    return _build_europepmc_text_query_from_terms(id_terms, taxid)


def _collect_base_terms_for_pass2(row: pd.Series, prefix: str = "query") -> List[str]:
    from .search_terms import is_usable_search_term

    out: List[str] = []
    seen: set[str] = set()
    for kind, term in _pass2_id_terms(row, prefix=prefix, extra_terms=None):
        if not is_usable_search_term(term, kind=kind):
            continue
        lk = term.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(term)
    return out


def _collect_search_terms_used(
    row: pd.Series,
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
) -> List[str]:
    """Terms that survive usability filters for pass2 (base + synonym extras)."""
    from .search_terms import is_usable_search_term

    out: List[str] = []
    seen: set[str] = set()
    for kind, term in _pass2_id_terms(row, prefix=prefix, extra_terms=extra_terms):
        if not is_usable_search_term(term, kind=kind):
            continue
        lk = term.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(term)
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
    gate: Optional[RequestGate] = None,
) -> Dict[str, List[str]]:
    """Run a Europe PMC search for a query string with caching."""
    if isinstance(cache, LockedCache):
        hit, cached = cache.get_if_present(query)
        if hit:
            logger.debug(f"Cache hit for query={query!r}")
            return cached
    elif query in cache:
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
    sess = _thread_session(session)

    try:
        if gate is not None:
            gate.wait()
        else:
            time.sleep(delay)
        resp = sess.get(EUROPEPMC_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Europe PMC search failed for query={query!r}: {e}")
        result = {
            "dois": [],
            "titles": [],
            "hit_count": 0,
            "n_raw": 0,
            "truncated": False,
            "request_ok": False,
        }
        if isinstance(cache, LockedCache):
            return cache.store(query, result)
        cache[query] = result
        return result

    records = (data.get("resultList") or {}).get("result") or []
    try:
        hit_count = int(data.get("hitCount") or 0)
    except (TypeError, ValueError):
        hit_count = len(records)
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

    result = {
        "dois": dois,
        "titles": titles,
        # Total matches Europe PMC reported vs. how many this single page fetched.
        # hit_count > pageSize means the first-page cap dropped later results.
        "hit_count": hit_count,
        "n_raw": len(records),
        "truncated": hit_count > int(params["pageSize"]),
        "request_ok": True,
    }
    if isinstance(cache, LockedCache):
        return cache.store(query, result)
    cache[query] = result
    return result


def _term_hit_attribution_enabled() -> bool:
    # Default off: per-term re-queries dominate runtime; enable for audit runs.
    value = os.environ.get("AUTO_LIT_TERM_HIT_ATTRIBUTION", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _attribute_result_ids_to_terms(
    result_ids: List[str],
    id_terms: List[Tuple[str, str]],
    *,
    pass_name: str,
    taxid: TaxidInput,
    organism_terms: Optional[Sequence[str]] = None,
    session: requests.Session,
    cache: Dict[str, Dict[str, List[str]]],
    delay: float,
    gate: Optional[RequestGate] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Attribute an existing combined-query result set to individual terms.

    Europe PMC does not identify which OR clause matched a result. Run the same
    TITLE_ABS/BODY query one term at a time, cache it, and intersect each response
    with the already-selected result IDs. This preserves retrieval behavior while
    recording query-engine evidence for each term. Results that cannot be attributed
    (for example because a single-term response hit the API result cap) remain
    explicitly unresolved rather than being assigned a guessed term.
    """
    wanted = set(result_ids)
    if not wanted or not _term_hit_attribution_enabled():
        return {}, sorted(wanted)

    hits: Dict[str, List[Dict[str, Any]]] = {}
    normalized_taxids = _normalize_taxids(taxid)
    normalized_organism_terms = _normalize_organism_terms(organism_terms)
    for kind, term in _clean_europepmc_text_terms(id_terms):
        query, _ = _build_europepmc_text_query_from_terms(
            [(kind, term)],
            taxid,
            organism_terms=normalized_organism_terms,
        )
        if not query:
            continue
        response = _run_europepmc_search_query(
            query, session, cache, delay=delay, gate=gate
        )
        for paper_id in wanted & set(response.get("dois") or []):
            hits.setdefault(paper_id, []).append(
                {
                    "term": term,
                    "kind": kind,
                    "pass": pass_name,
                    "taxids": list(normalized_taxids),
                    "organism_terms": list(normalized_organism_terms),
                    "scope": "TITLE_ABS_OR_BODY",
                }
            )

    return hits, sorted(wanted - set(hits))


def _merge_term_hit_maps(
    destination: Dict[str, List[Dict[str, Any]]],
    source: Dict[str, List[Dict[str, Any]]],
) -> None:
    for paper_id, records in source.items():
        existing = destination.setdefault(paper_id, [])
        seen = {
            (
                str(record.get("term") or "").lower(),
                record.get("kind"),
                record.get("pass"),
                tuple(record.get("taxids") or []),
                tuple(record.get("organism_terms") or []),
            )
            for record in existing
        }
        for record in records:
            key = (
                str(record.get("term") or "").lower(),
                record.get("kind"),
                record.get("pass"),
                tuple(record.get("taxids") or []),
                tuple(record.get("organism_terms") or []),
            )
            if key not in seen:
                seen.add(key)
                existing.append(record)


def _summarize_term_hits(
    paper_term_hits: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    HitKey = Tuple[str, str, str, Tuple[int, ...], Tuple[str, ...]]
    counts: Dict[HitKey, int] = {}
    display: Dict[HitKey, str] = {}
    for records in paper_term_hits.values():
        per_paper: set[HitKey] = set()
        for record in records:
            term = str(record.get("term") or "")
            key = (
                term.lower(),
                str(record.get("kind") or ""),
                str(record.get("pass") or ""),
                tuple(record.get("taxids") or []),
                tuple(record.get("organism_terms") or []),
            )
            display.setdefault(key, term)
            per_paper.add(key)
        for key in per_paper:
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "term": display[key],
            "kind": key[1],
            "pass": key[2],
            "taxids": list(key[3]),
            "organism_terms": list(key[4]),
            "n_papers": count,
        }
        for key, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][2], item[0][0])
        )
    ]


def run_europepmc_search_for_row(
    row: pd.Series,
    taxid: TaxidInput,
    session: requests.Session,
    cache: Dict[str, Dict[str, List[str]]],
    delay: float = 0.35,
    prefix: str = "query",
    extra_terms: Optional[List[str]] = None,
    organism_terms: Optional[Sequence[str]] = None,
    gate: Optional[RequestGate] = None,
) -> Dict[str, Any]:
    """
    Run a Europe PMC search for a single mapping row.

    Returns dict with keys: "dois", "titles", "pass1_count", "pass2_count".
    """
    from .search_terms import filter_terms

    row_label = str(row.get(prefix) or row.get(f"{prefix}_id") or "")
    base_terms = _collect_base_terms_for_pass2(row, prefix=prefix)
    syn_terms = filter_terms(
        [t for t in (extra_terms or []) if _normalize_term(t)],
        kind="synonym",
    )
    search_terms_used = _collect_search_terms_used(
        row, prefix=prefix, extra_terms=extra_terms
    )
    logger.debug(
        f"[{prefix}] pass2 term summary for row={row_label!r}: "
        f"base_terms_n={len(base_terms)} base_terms_sample={base_terms[:6]} "
        f"synonym_terms_n={len(syn_terms)} synonym_terms_sample={syn_terms[:6]} "
        f"search_terms_used={search_terms_used[:12]}"
    )

    q1, q1_types = _build_europepmc_text_query_pass1(row, prefix=prefix)
    q2_base, q2_base_types = _build_europepmc_text_query_pass2_base_only(
        row, taxid, prefix=prefix
    )
    q2_syn, q2_syn_types = _build_europepmc_text_query_pass2_synonym_only(
        taxid, extra_terms=extra_terms
    )
    pass1_terms = _collect_pass1_terms(row, prefix=prefix)
    pass2_base_terms = _clean_europepmc_text_terms(
        _pass2_id_terms(row, prefix=prefix, extra_terms=None)
    )
    pass2_synonym_terms = _clean_europepmc_text_terms(
        [("synonym", term) for term in (extra_terms or []) if _normalize_term(term)]
    )

    if not q1 and not q2_base and not q2_syn:
        return {
            "dois": [],
            "titles": [],
            "pass1_count": 0,
            "pass2_count": 0,
            "search_terms": search_terms_used,
            "paper_term_hits": {},
            "unattributed_term_hit_dois_by_pass": {},
            "organism_fallbacks": [],
        }

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
    pass2_base_effective_taxid = taxid
    pass2_synonym_effective_taxid = taxid
    pass2_base_effective_organism_terms: Tuple[str, ...] = ()
    pass2_synonym_effective_organism_terms: Tuple[str, ...] = ()
    organism_fallbacks: List[Dict[str, Any]] = []
    row_taxids = _normalize_taxids(taxid)
    row_organism_terms = _normalize_organism_terms(organism_terms)

    def _record_organism_fallback(
        pass_name: str,
        terms: List[Tuple[str, str]],
        result: Dict[str, Any],
    ) -> None:
        term_strs = [t for _, t in terms]
        info = {
            "pass": pass_name,
            "prefix": prefix,
            "row_label": row_label,
            "dropped_taxids": list(row_taxids),
            "fallback_scope": "organism_text",
            "organism_terms": list(row_organism_terms),
            "terms": term_strs,
            "hit_count": int(result.get("hit_count") or 0),
            "n_raw": int(result.get("n_raw") or 0),
            "n_kept": len(result.get("dois") or []),
            "truncated": bool(result.get("truncated")),
        }
        organism_fallbacks.append(info)
        logger.warning(
            "[{}] Europe PMC {} fell back from organism taxids to organism-name "
            "text for row={!r}: dropped_taxids={} organism_terms={} terms={} "
            "hit_count={} kept={} truncated={}",
            prefix,
            pass_name,
            row_label,
            info["dropped_taxids"],
            info["organism_terms"],
            info["terms"][:8],
            info["hit_count"],
            info["n_kept"],
            info["truncated"],
        )

    if q1:
        logger.debug(f"Europe PMC text pass1 ids={q1_types}")
        r1 = _run_europepmc_search_query(
            q1, session, cache, delay=delay, gate=gate
        )
        for pid, title in zip(r1["dois"], r1["titles"]):
            if pid not in id_to_title:
                id_to_title[pid] = title
        pass1_dois = list(r1["dois"])
        pass1_n = len(pass1_dois)

    if q2_base:
        logger.debug(f"Europe PMC text pass2-base ids={q2_base_types} taxid={taxid}")
        logger.debug(f"[{prefix}] pass2-base query: {q2_base}")
        r2_base = _run_europepmc_search_query(
            q2_base, session, cache, delay=delay, gate=gate
        )
        pass2_base_dois = list(r2_base["dois"])
        pass2_base_n = len(pass2_base_dois)
        if (
            pass2_base_n == 0
            and row_taxids
            and row_organism_terms
            and bool(r2_base.get("request_ok"))
            and int(r2_base.get("hit_count") or 0) == 0
        ):
            q2_base_alt, q2_base_alt_types = _build_europepmc_text_query_from_terms(
                pass2_base_terms,
                None,
                organism_terms=row_organism_terms,
            )
            if q2_base_alt:
                logger.debug(
                    f"Europe PMC pass2-base retry with organism-name text "
                    f"(orig taxid={taxid}) ids={q2_base_alt_types}"
                )
                r2_base_alt = _run_europepmc_search_query(
                    q2_base_alt, session, cache, delay=delay, gate=gate
                )
                pass2_base_dois = list(r2_base_alt["dois"])
                pass2_base_n = len(pass2_base_dois)
                pass2_base_effective_taxid = None
                pass2_base_effective_organism_terms = row_organism_terms
                _record_organism_fallback("pass2_base", pass2_base_terms, r2_base_alt)

    if q2_syn:
        logger.debug(
            f"Europe PMC text pass2-synonym ids={q2_syn_types} taxid={taxid}"
        )
        logger.debug(f"[{prefix}] pass2-synonym query: {q2_syn}")
        r2_syn = _run_europepmc_search_query(
            q2_syn, session, cache, delay=delay, gate=gate
        )
        pass2_synonym_dois = list(r2_syn["dois"])
        pass2_synonym_n = len(pass2_synonym_dois)
        if (
            pass2_synonym_n == 0
            and row_taxids
            and row_organism_terms
            and bool(r2_syn.get("request_ok"))
            and int(r2_syn.get("hit_count") or 0) == 0
        ):
            q2_syn_alt, q2_syn_alt_types = _build_europepmc_text_query_from_terms(
                pass2_synonym_terms,
                None,
                organism_terms=row_organism_terms,
            )
            if q2_syn_alt:
                logger.debug(
                    f"Europe PMC pass2-synonym retry with organism-name text "
                    f"(orig taxid={taxid}) ids={q2_syn_alt_types}"
                )
                r2_syn_alt = _run_europepmc_search_query(
                    q2_syn_alt, session, cache, delay=delay, gate=gate
                )
                pass2_synonym_dois = list(r2_syn_alt["dois"])
                pass2_synonym_n = len(pass2_synonym_dois)
                pass2_synonym_effective_taxid = None
                pass2_synonym_effective_organism_terms = row_organism_terms
                _record_organism_fallback("pass2_synonym", pass2_synonym_terms, r2_syn_alt)

    paper_term_hits: Dict[str, List[Dict[str, Any]]] = {}
    unattributed_by_pass: Dict[str, List[str]] = {}
    attribution_jobs = (
        ("pass1", pass1_dois, pass1_terms, None, ()),
        (
            "pass2_base",
            pass2_base_dois,
            pass2_base_terms,
            pass2_base_effective_taxid,
            pass2_base_effective_organism_terms,
        ),
        (
            "pass2_synonym",
            pass2_synonym_dois,
            pass2_synonym_terms,
            pass2_synonym_effective_taxid,
            pass2_synonym_effective_organism_terms,
        ),
    )
    for (
        pass_name,
        paper_ids,
        terms,
        effective_taxid,
        effective_organism_terms,
    ) in attribution_jobs:
        attributed, unresolved = _attribute_result_ids_to_terms(
            paper_ids,
            terms,
            pass_name=pass_name,
            taxid=effective_taxid,
            organism_terms=effective_organism_terms,
            session=session,
            cache=cache,
            delay=delay,
            gate=gate,
        )
        _merge_term_hit_maps(paper_term_hits, attributed)
        if unresolved:
            unattributed_by_pass[pass_name] = unresolved

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
        f"pass2_overlap={pass2_overlap_n}, unique={len(merged_dois)}, "
        f"term_attributed={len(paper_term_hits)}, "
        f"term_unattributed={sum(len(v) for v in unattributed_by_pass.values())}"
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
        "search_terms": search_terms_used,
        "paper_term_hits": paper_term_hits,
        "unattributed_term_hit_dois_by_pass": unattributed_by_pass,
        "organism_fallbacks": organism_fallbacks,
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



def _search_one_alignment_row(
    *,
    idx,
    row: pd.Series,
    query_id_col: str,
    target_id_col: str,
    taxid_col: Optional[str],
    default_taxid: Optional[int],
    query_taxid: Optional[int],
    target_taxid: Optional[int],
    query_taxids: Optional[Sequence[int]],
    target_taxids: Optional[Sequence[int]],
    query_organism_terms: Optional[Sequence[str]],
    target_organism_terms: Optional[Sequence[str]],
    result_columns: Sequence[str],
    session: requests.Session,
    uniprot_cache: Dict[str, Dict[str, List[str]]],
    text_cache: Dict[str, Dict[str, List[str]]],
    gene_synonyms_by_entrez: Dict[int, List[str]],
    delay: float,
    gate: Optional[RequestGate],
    filter_q_acc: bool,
    filter_t_acc: bool,
) -> Dict[str, Any]:
    """Search Europe PMC for one alignment row (thread-safe with LockedCache + gate)."""
    row_taxid: Optional[int] = None
    if taxid_col and taxid_col in result_columns:
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
    q_search_terms: List[str] = []
    q_paper_term_hits: Dict[str, List[Dict[str, Any]]] = {}
    q_unattributed_term_hits: Dict[str, List[str]] = {}
    q_organism_fallbacks: List[Dict[str, Any]] = []
    if q_entrez_id is not None:
        q_pubtator_empty = True

    # Fallback: UniProt citation + Europe PMC text search.
    if not merged_q_dois:
        query_res = run_europepmc_crossref(q_acc, session, uniprot_cache, delay=delay, gate=gate)
        query_text_taxid: TaxidInput = (
            query_taxids
            if query_taxids is not None
            else (query_taxid if query_taxid is not None else row_taxid)
        )
        text_res_query = run_europepmc_search_for_row(
            row,
            query_text_taxid,
            session,
            text_cache,
            delay=delay,
            prefix="query",
            extra_terms=gene_synonyms_by_entrez.get(q_entrez_id or -1, []),
            organism_terms=query_organism_terms,
            gate=gate,
        )
        q_search_terms = list(text_res_query.get("search_terms") or [])
        q_paper_term_hits = dict(text_res_query.get("paper_term_hits") or {})
        q_unattributed_term_hits = dict(
            text_res_query.get("unattributed_term_hit_dois_by_pass") or {}
        )
        q_organism_fallbacks = list(
            text_res_query.get("organism_fallbacks") or []
        )
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

    query_dois = json.dumps(merged_q_dois)
    query_titles = json.dumps(merged_q_titles)
    query_paper_counts_out = query_paper_counts
    query_paper_ids_by_source_out = json.dumps(query_paper_ids_by_source)
    query_search_terms_out = json.dumps(q_search_terms)
    query_paper_term_hits_out = json.dumps(q_paper_term_hits)
    query_unattributed_term_hits_out = json.dumps(q_unattributed_term_hits)
    query_organism_fallbacks_out = json.dumps(q_organism_fallbacks)

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
    t_search_terms: List[str] = []
    t_paper_term_hits: Dict[str, List[Dict[str, Any]]] = {}
    t_unattributed_term_hits: Dict[str, List[str]] = {}
    t_organism_fallbacks: List[Dict[str, Any]] = []
    if t_entrez_id is not None:
        t_pubtator_empty = True

    # Fallback: UniProt citation + Europe PMC text search.
    if not merged_t_dois:
        target_res = run_europepmc_crossref(t_acc, session, uniprot_cache, delay=delay, gate=gate)
        target_text_taxid: TaxidInput = (
            target_taxids
            if target_taxids is not None
            else (target_taxid if target_taxid is not None else row_taxid)
        )
        text_res_target = run_europepmc_search_for_row(
            row,
            target_text_taxid,
            session,
            text_cache,
            delay=delay,
            prefix="target",
            extra_terms=gene_synonyms_by_entrez.get(t_entrez_id or -1, []),
            organism_terms=target_organism_terms,
            gate=gate,
        )
        t_search_terms = list(text_res_target.get("search_terms") or [])
        t_paper_term_hits = dict(text_res_target.get("paper_term_hits") or {})
        t_unattributed_term_hits = dict(
            text_res_target.get("unattributed_term_hit_dois_by_pass") or {}
        )
        t_organism_fallbacks = list(
            text_res_target.get("organism_fallbacks") or []
        )

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

    target_dois = json.dumps(merged_t_dois)
    target_titles = json.dumps(merged_t_titles)
    target_paper_ids_by_source_out = json.dumps(target_paper_ids_by_source)
    target_search_terms_out = json.dumps(t_search_terms)
    target_paper_term_hits_out = json.dumps(t_paper_term_hits)
    target_unattributed_term_hits_out = json.dumps(t_unattributed_term_hits)
    target_organism_fallbacks_out = json.dumps(t_organism_fallbacks)

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
        "query_search_terms": q_search_terms,
        "query_paper_term_hits": q_paper_term_hits,
        "query_term_hit_counts": _summarize_term_hits(q_paper_term_hits),
        "query_unattributed_term_hit_dois_by_pass": q_unattributed_term_hits,
        "query_organism_fallbacks": q_organism_fallbacks,
        "accession_text_overlap_filter_query": filter_q_acc,
        "query_dois_n": len(merged_q_dois),
        "query_dois_sample": merged_q_dois[:10],
        "target_pubtator_used": t_pubtator_used,
        "target_pubtator_empty": t_pubtator_empty,
        "target_counts": target_paper_counts,
        "target_search_terms": t_search_terms,
        "target_paper_term_hits": t_paper_term_hits,
        "target_term_hit_counts": _summarize_term_hits(t_paper_term_hits),
        "target_unattributed_term_hit_dois_by_pass": t_unattributed_term_hits,
        "target_organism_fallbacks": t_organism_fallbacks,
        "accession_text_overlap_filter_target": filter_t_acc,
        "target_dois_n": len(merged_t_dois),
        "target_dois_sample": merged_t_dois[:10],
    }

    q_from_accession = bool(locals().get("query_res") and query_res.get("dois"))
    q_from_text = bool(locals().get("text_res_query") and text_res_query.get("dois"))
    t_from_accession = bool(locals().get("target_res") and target_res.get("dois"))
    t_from_text = bool(locals().get("text_res_target") and text_res_target.get("dois"))

    return {
        "query_dois": query_dois,
        "query_titles": query_titles,
        "target_dois": target_dois,
        "target_titles": target_titles,
        "query_paper_counts": query_paper_counts_out,
        "query_paper_ids_by_source": query_paper_ids_by_source_out,
        "target_paper_ids_by_source": target_paper_ids_by_source_out,
        "query_search_terms": query_search_terms_out,
        "target_search_terms": target_search_terms_out,
        "query_paper_term_hits": query_paper_term_hits_out,
        "target_paper_term_hits": target_paper_term_hits_out,
        "query_unattributed_term_hits": query_unattributed_term_hits_out,
        "target_unattributed_term_hits": target_unattributed_term_hits_out,
        "query_organism_fallbacks": query_organism_fallbacks_out,
        "target_organism_fallbacks": target_organism_fallbacks_out,
        "trace": trace_obj,
        "stats": {
            "q_entrez_id": str(q_entrez_id) if q_entrez_id is not None else None,
            "t_entrez_id": str(t_entrez_id) if t_entrez_id is not None else None,
            "q_acc": q_acc,
            "t_acc": t_acc,
            "q_from_accession": q_from_accession,
            "q_from_text": q_from_text,
            "t_from_accession": t_from_accession,
            "t_from_text": t_from_text,
            "q_with_papers": bool(q_acc and merged_q_dois),
            "t_with_papers": bool(t_acc and merged_t_dois),
        },
    }


def run(
    df: pd.DataFrame,
    query_id_col: str = "query",
    target_id_col: str = "target",
    taxid_col: Optional[str] = None,
    default_taxid: Optional[int] = None,
    query_taxid: Optional[int] = None,
    target_taxid: Optional[int] = None,
    query_taxids: Optional[Sequence[int]] = None,
    target_taxids: Optional[Sequence[int]] = None,
    query_organism_terms: Optional[Sequence[str]] = None,
    target_organism_terms: Optional[Sequence[str]] = None,
    output_dir: str = ".",
    delay: float = 0.35,
    use_cache: bool = True,
    accession_text_overlap: Optional[str] = None,
    workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Two-phase Europe PMC search for each alignment row:
      1. UniProt accession citations
         (ACCESSION_ID:<acc> AND ACCESSION_TYPE:uniprot) for query and target.
      2. Text search using locus_tag / GenBank (pass1) and gene name / common name (pass2)
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
        query_taxids: Ordered query-organism taxids searched together. Overrides
            query_taxid when provided.
        target_taxids: Ordered target-organism taxids searched together. Overrides
            target_taxid when provided.
        query_organism_terms: Organism names required by the query-side text
            fallback when taxid-constrained pass2 finds no matches.
        target_organism_terms: Organism names required by the target-side text
            fallback when taxid-constrained pass2 finds no matches.
        output_dir: Directory for logs and cache file (search_cache.json).
        use_cache: If True, load cache at start (if exists) and save at end.
        accession_text_overlap: If set, overrides env ``AUTO_LIT_ACCESSION_REQUIRES_TEXT_OVERLAP``.
            Values: ``off``, ``query``, ``target``, ``both`` (or ``1``/``true`` via env).
            When enabled for a side, DOIs that appear only in the Europe PMC UniProt-accession
            search and not in the text-search union are dropped from merged outputs.
        workers: Parallel alignment-row workers. Default from ``AUTO_LIT_SEARCH_WORKERS`` (8).
            Outbound Europe PMC calls still share a process-wide rate gate (``delay``).

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

    uniprot_cache = LockedCache(uniprot_cache)
    text_cache = LockedCache(text_cache)

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
    query_search_terms_col: List[str] = []
    target_search_terms_col: List[str] = []
    query_paper_term_hits_col: List[str] = []
    target_paper_term_hits_col: List[str] = []
    query_unattributed_term_hits_col: List[str] = []
    target_unattributed_term_hits_col: List[str] = []
    query_organism_fallbacks_col: List[str] = []
    target_organism_fallbacks_col: List[str] = []

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
    gate = RequestGate(delay)
    workers = max(1, int(workers if workers is not None else _search_workers_default()))
    logger.info(
        "Europe PMC search: workers={} delay={}s term_hit_attribution={}",
        workers,
        delay,
        _term_hit_attribution_enabled(),
    )

    row_items = list(result_df.iterrows())
    result_columns = list(result_df.columns)
    row_results: List[Optional[Dict[str, Any]]] = [None] * len(row_items)

    def _job(pos: int, idx, row: pd.Series) -> Tuple[int, Dict[str, Any]]:
        out = _search_one_alignment_row(
            idx=idx,
            row=row,
            query_id_col=query_id_col,
            target_id_col=target_id_col,
            taxid_col=taxid_col,
            default_taxid=default_taxid,
            query_taxid=query_taxid,
            target_taxid=target_taxid,
            query_taxids=query_taxids,
            target_taxids=target_taxids,
            query_organism_terms=query_organism_terms,
            target_organism_terms=target_organism_terms,
            result_columns=result_columns,
            session=session,
            uniprot_cache=uniprot_cache,
            text_cache=text_cache,
            gene_synonyms_by_entrez=gene_synonyms_by_entrez,
            delay=delay,
            gate=gate,
            filter_q_acc=filter_q_acc,
            filter_t_acc=filter_t_acc,
        )
        return pos, out

    if workers == 1:
        for pos, (idx, row) in enumerate(row_items):
            _, out = _job(pos, idx, row)
            row_results[pos] = out
            if (pos + 1) % 50 == 0 or (pos + 1) == n_rows:
                logger.info("Search progress: {}/{}", pos + 1, n_rows)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_job, pos, idx, row)
                for pos, (idx, row) in enumerate(row_items)
            ]
            for fut in as_completed(futures):
                pos, out = fut.result()
                row_results[pos] = out
                done += 1
                if done % 50 == 0 or done == n_rows:
                    logger.info("Search progress: {}/{}", done, n_rows)

    trace_fh = open(trace_path, "w", encoding="utf-8")
    for out in row_results:
        assert out is not None
        query_dois_col.append(out["query_dois"])
        query_titles_col.append(out["query_titles"])
        target_dois_col.append(out["target_dois"])
        target_titles_col.append(out["target_titles"])
        query_paper_counts_col.append(out["query_paper_counts"])
        query_paper_ids_by_source_col.append(out["query_paper_ids_by_source"])
        target_paper_ids_by_source_col.append(out["target_paper_ids_by_source"])
        query_search_terms_col.append(out["query_search_terms"])
        target_search_terms_col.append(out["target_search_terms"])
        query_paper_term_hits_col.append(out["query_paper_term_hits"])
        target_paper_term_hits_col.append(out["target_paper_term_hits"])
        query_unattributed_term_hits_col.append(out["query_unattributed_term_hits"])
        target_unattributed_term_hits_col.append(out["target_unattributed_term_hits"])
        query_organism_fallbacks_col.append(out["query_organism_fallbacks"])
        target_organism_fallbacks_col.append(out["target_organism_fallbacks"])
        st = out["stats"]
        if st["q_entrez_id"] is not None:
            query_ids_seen.add(st["q_entrez_id"])
        if st["t_entrez_id"] is not None:
            target_ids_seen.add(st["t_entrez_id"])
        if st["q_acc"]:
            query_ids_seen.add(st["q_acc"])
            if st["q_with_papers"]:
                query_ids_with_papers.add(st["q_acc"])
        if st["t_acc"]:
            target_ids_seen.add(st["t_acc"])
            if st["t_with_papers"]:
                target_ids_with_papers.add(st["t_acc"])
        if st["q_from_accession"]:
            rows_query_from_europepmc_accession += 1
        if st["q_from_text"]:
            rows_query_from_text += 1
        if st["t_from_accession"]:
            rows_target_from_europepmc_accession += 1
        if st["t_from_text"]:
            rows_target_from_text += 1
        trace_fh.write(json.dumps(out["trace"], ensure_ascii=False) + "\n")

    trace_fh.close()

    result_df["query_paper_dois"] = query_dois_col
    result_df["query_paper_titles"] = query_titles_col
    result_df["target_paper_dois"] = target_dois_col
    result_df["target_paper_titles"] = target_titles_col
    result_df["query_paper_counts"] = query_paper_counts_col
    result_df["query_paper_ids_by_source"] = query_paper_ids_by_source_col
    result_df["target_paper_ids_by_source"] = target_paper_ids_by_source_col
    result_df["query_search_terms"] = query_search_terms_col
    result_df["target_search_terms"] = target_search_terms_col
    result_df["query_paper_term_hits"] = query_paper_term_hits_col
    result_df["target_paper_term_hits"] = target_paper_term_hits_col
    result_df["query_unattributed_term_hit_dois_by_pass"] = (
        query_unattributed_term_hits_col
    )
    result_df["target_unattributed_term_hit_dois_by_pass"] = (
        target_unattributed_term_hits_col
    )
    result_df["query_organism_fallbacks"] = query_organism_fallbacks_col
    result_df["target_organism_fallbacks"] = target_organism_fallbacks_col

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

    fb_events: List[Dict[str, Any]] = []
    for v in query_organism_fallbacks_col + target_organism_fallbacks_col:
        try:
            fb_events.extend(json.loads(v) or [])
        except (TypeError, ValueError):
            continue
    if fb_events:
        n_truncated = sum(1 for e in fb_events if e.get("truncated"))
        hit_counts = sorted((int(e.get("hit_count") or 0) for e in fb_events), reverse=True)
        logger.info(
            "Organism-name text fallback fired {} time(s): {} returned >200 hits "
            "(truncated at pageSize). Largest hit_counts: {}",
            len(fb_events),
            n_truncated,
            hit_counts[:15],
        )
    else:
        logger.info("Organism-name text fallback never fired this run")

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
        def _load_terms(raw):
            if isinstance(raw, str):
                return json.loads(raw) if raw else []
            if isinstance(raw, list):
                return raw
            return []

        def _load_object(raw):
            if isinstance(raw, str):
                return json.loads(raw) if raw else {}
            if isinstance(raw, dict):
                return raw
            return {}

        query_paper_term_hits = _load_object(row.get("query_paper_term_hits"))
        target_paper_term_hits = _load_object(row.get("target_paper_term_hits"))
        entry = {
            "target": t,
            "n_query_papers": len(query_dois),
            "n_target_papers": len(target_dois),
            "query_paper_counts": counts,
            "query_paper_dois": query_dois,
            "query_paper_titles": query_titles,
            "query_paper_ids_by_source": query_ids_by_source,
            "query_search_terms": _load_terms(row.get("query_search_terms")),
            "query_paper_term_hits": query_paper_term_hits,
            "query_term_hit_counts": _summarize_term_hits(query_paper_term_hits),
            "query_unattributed_term_hit_dois_by_pass": _load_object(
                row.get("query_unattributed_term_hit_dois_by_pass")
            ),
            "query_organism_fallbacks": _load_terms(
                row.get("query_organism_fallbacks")
            ),
            "target_paper_dois": target_dois,
            "target_paper_titles": target_titles,
            "target_paper_ids_by_source": target_ids_by_source,
            "target_search_terms": _load_terms(row.get("target_search_terms")),
            "target_paper_term_hits": target_paper_term_hits,
            "target_term_hit_counts": _summarize_term_hits(target_paper_term_hits),
            "target_unattributed_term_hit_dois_by_pass": _load_object(
                row.get("target_unattributed_term_hit_dois_by_pass")
            ),
            "target_organism_fallbacks": _load_terms(
                row.get("target_organism_fallbacks")
            ),
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
        "--query-taxids",
        type=str,
        default=None,
        help="Query-organism taxids searched together ('|' or ',' separated; overrides --query-taxid).",
    )
    parser.add_argument(
        "--target-taxids",
        type=str,
        default=None,
        help="Target-organism taxids searched together ('|' or ',' separated; overrides --target-taxid).",
    )
    parser.add_argument(
        "--query-organism-terms",
        type=str,
        default=None,
        help=(
            "Pipe-separated query-organism names used as a constrained text "
            "fallback when taxid-scoped pass2 has zero matches."
        ),
    )
    parser.add_argument(
        "--target-organism-terms",
        type=str,
        default=None,
        help=(
            "Pipe-separated target-organism names used as a constrained text "
            "fallback when taxid-scoped pass2 has zero matches."
        ),
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
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel alignment-row workers (default: AUTO_LIT_SEARCH_WORKERS or 8). "
            "Europe PMC requests still share a process-wide rate gate."
        ),
    )
    parser.add_argument(
        "--term-hit-attribution",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable per-term Europe PMC re-queries for audit attribution. "
            "Default is off (AUTO_LIT_TERM_HIT_ATTRIBUTION=0)."
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

    if args.term_hit_attribution is not None:
        os.environ["AUTO_LIT_TERM_HIT_ATTRIBUTION"] = (
            "1" if args.term_hit_attribution else "0"
        )

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading mapping input: {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 1

    def _parse_taxid_csv(raw: Optional[str]) -> Optional[List[int]]:
        if raw is None:
            return None
        # Accept | (preferred for Slurm --export) or commas.
        sep = "|" if "|" in raw else ","
        values = [int(part.strip()) for part in raw.split(sep) if part.strip()]
        if not values or any(value <= 0 for value in values):
            parser.error(
                "taxid lists must contain positive integers separated by '|' or ','"
            )
        return list(dict.fromkeys(values))

    def _parse_organism_terms(raw: Optional[str]) -> Optional[List[str]]:
        if raw is None:
            return None
        values = [part.strip() for part in raw.split("|") if part.strip()]
        return list(dict.fromkeys(values)) or None

    result_df = run(
        df,
        query_id_col=args.query_id_col,
        target_id_col=args.target_id_col,
        taxid_col=args.taxid_col,
        default_taxid=args.default_taxid,
        accession_text_overlap=args.accession_text_overlap,
        query_taxid=args.query_taxid,
        target_taxid=args.target_taxid,
        query_taxids=_parse_taxid_csv(args.query_taxids),
        target_taxids=_parse_taxid_csv(args.target_taxids),
        query_organism_terms=_parse_organism_terms(args.query_organism_terms),
        target_organism_terms=_parse_organism_terms(args.target_organism_terms),
        output_dir=output_dir,
        use_cache=not args.no_cache,
        workers=args.workers,
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

