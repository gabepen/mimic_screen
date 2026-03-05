"""
Search module for automated literature search pipeline (Module 2 of 4).

Two-phase search:
  1. UniProt pass: Europe PMC search by UniProt accession citations
     (ACCESSION_ID:ACCESSION AND ACCESSION_TYPE:uniprot) for both query and target
     proteins. Uses resultType=core.
  2. Text fallback: For rows where the query UniProt pass returned no papers, run a
     two-pass text search for the query protein only:
       - pass1 (organism-specific): query_locus_tag OR query_genbank_acc
       - pass2 (ambiguous + organism filter): (query_gene_name OR query_common_name) AND ORGANISM_ID:<taxid>

Output columns:
  - query_paper_dois, query_paper_titles (from UniProt pass when available, else text search)
  - target_paper_dois, target_paper_titles (UniProt pass only; no text search for target)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
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

    if not id_terms:
        return None, []

    or_clauses = [f'"{val}"' for (_kind, val) in id_terms]
    or_part = " OR ".join(or_clauses)
    query = f"({or_part})"

    id_types_used = [kind for (kind, _val) in id_terms]
    return query, id_types_used


def _build_europepmc_text_query_pass2(
    row: pd.Series, taxid: Optional[int], prefix: str = "query"
) -> Tuple[Optional[str], List[str]]:
    """
    Text query pass 2 (ambiguous terms): gene/common name AND organism filter.

    Uses available identifiers and ORs them together (for the given prefix):
        <prefix>_gene_name, <prefix>_common_name

    If taxid is missing, returns None (skips pass 2).
    """
    if taxid is None:
        return None, []

    id_terms: List[Tuple[str, str]] = []  # (type, value)

    gene_name = _normalize_term(row.get(f"{prefix}_gene_name"))
    if gene_name:
        id_terms.append(("gene_name", gene_name))

    common_name = _normalize_term(row.get(f"{prefix}_common_name"))
    if common_name:
        id_terms.append(("common_name", common_name))

    if not id_terms:
        return None, []

    or_clauses = [f'"{val}"' for (_kind, val) in id_terms]
    or_part = " OR ".join(or_clauses)
    query = f"({or_part}) AND ORGANISM_ID:{int(taxid)}"

    id_types_used = [kind for (kind, _val) in id_terms]
    return query, id_types_used


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
) -> Dict[str, Any]:
    """
    Run a Europe PMC search for a single mapping row.

    Returns dict with keys: "dois", "titles", "pass1_count", "pass2_count".
    """
    q1, q1_types = _build_europepmc_text_query_pass1(row, prefix=prefix)
    q2, q2_types = _build_europepmc_text_query_pass2(row, taxid, prefix=prefix)

    if not q1 and not q2:
        return {"dois": [], "titles": [], "pass1_count": 0, "pass2_count": 0}

    id_to_title: Dict[str, str] = {}
    pass1_dois: List[str] = []
    pass2_dois: List[str] = []
    pass1_n = 0
    pass2_n = 0

    if q1:
        logger.debug(f"Europe PMC text pass1 ids={q1_types}")
        r1 = _run_europepmc_search_query(q1, session, cache, delay=delay)
        for pid, title in zip(r1["dois"], r1["titles"]):
            if pid not in id_to_title:
                id_to_title[pid] = title
        pass1_dois = list(r1["dois"])
        pass1_n = len(pass1_dois)

    if q2:
        logger.debug(f"Europe PMC text pass2 ids={q2_types} taxid={taxid}")
        r2 = _run_europepmc_search_query(q2, session, cache, delay=delay)
        for pid, title in zip(r2["dois"], r2["titles"]):
            if pid not in id_to_title:
                id_to_title[pid] = title
        pass2_dois = list(r2["dois"])
        pass2_n = len(pass2_dois)

    merged_dois = list(id_to_title.keys())
    merged_titles = [id_to_title[pid] for pid in merged_dois]
    logger.debug(
        f"Europe PMC text merged: pass1={pass1_n}, pass2={pass2_n}, unique={len(merged_dois)}"
    )
    return {
        "dois": merged_dois,
        "titles": merged_titles,
        "pass1_count": pass1_n,
        "pass2_count": pass2_n,
        "pass1_dois": pass1_dois,
        "pass2_dois": pass2_dois,
    }


def run(
    df: pd.DataFrame,
    query_id_col: str = "query",
    target_id_col: str = "target",
    taxid_col: Optional[str] = None,
    default_taxid: Optional[int] = None,
    target_taxid: Optional[int] = None,
    output_dir: str = ".",
    use_cache: bool = True,
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
        target_taxid: Optional fixed taxon ID for target organism; if provided,
            target text searches use this instead of any per-row taxid.
        output_dir: Directory for logs and cache file (search_cache.json).
        use_cache: If True, load cache at start (if exists) and save at end.

    Returns a new DataFrame with additional columns:
        - query_paper_dois, query_paper_titles (merged UniProt + text search)
        - target_paper_dois, target_paper_titles (merged UniProt + text search)
    """
    _configure_file_logging(output_dir)

    if query_id_col not in df.columns:
        raise ValueError(f"query_id_col={query_id_col!r} not found in DataFrame columns")
    if target_id_col not in df.columns:
        raise ValueError(f"target_id_col={target_id_col!r} not found in DataFrame columns")

    result_df = df.copy()
    n_rows = len(result_df)
    logger.info(f"Search module – two-phase Europe PMC for {n_rows} rows (query col={query_id_col!r}, target col={target_id_col!r})")

    session = requests.Session()
    cache_path = os.path.join(output_dir, "search_cache.json")
    uniprot_cache: Dict[str, Dict[str, List[str]]] = {}
    text_cache: Dict[str, Dict[str, List[str]]] = {}

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            uniprot_cache = loaded.get("uniprot") or {}
            text_cache = loaded.get("text") or {}
            logger.info(f"Loaded cache from {cache_path} ({len(uniprot_cache)} uniprot, {len(text_cache)} text entries)")
        except Exception as e:
            logger.warning(f"Could not load cache from {cache_path}: {e}")

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
    rows_query_from_uniprot = 0
    rows_query_from_text = 0
    rows_target_from_uniprot = 0
    rows_target_from_text = 0

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

        query_res = run_europepmc_crossref(q_acc, session, uniprot_cache)
        if q_acc:
            query_ids_seen.add(q_acc)
            if query_res["dois"]:
                rows_query_from_uniprot += 1

        text_res_query = run_europepmc_search_for_row(
            row, row_taxid, session, text_cache, prefix="query"
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

        if q_acc and merged_q_dois:
            query_ids_with_papers.add(q_acc)

        query_dois_col.append(json.dumps(merged_q_dois))
        query_titles_col.append(json.dumps(merged_q_titles))
        query_paper_counts_col.append({
            "uniprot": len(query_res["dois"]),
            "text_pass1": text_res_query.get("pass1_count", 0),
            "text_pass2": text_res_query.get("pass2_count", 0),
        })
        query_paper_ids_by_source_col.append(json.dumps({
            "uniprot": list(query_res["dois"]),
            "text_pass1": list(text_res_query.get("pass1_dois", [])),
            "text_pass2": list(text_res_query.get("pass2_dois", [])),
        }))

        target_res = run_europepmc_crossref(t_acc, session, uniprot_cache)
        # Use an explicit target_taxid for target-side text search if provided,
        # otherwise fall back to any per-row taxid/default_taxid.
        target_text_taxid: Optional[int] = target_taxid if target_taxid is not None else row_taxid
        text_res_target = run_europepmc_search_for_row(
            row, target_text_taxid, session, text_cache, prefix="target"
        )

        if t_acc:
            target_ids_seen.add(t_acc)
            if target_res["dois"]:
                rows_target_from_uniprot += 1
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

        if t_acc and merged_t_dois:
            target_ids_with_papers.add(t_acc)

        target_dois_col.append(json.dumps(merged_t_dois))
        target_titles_col.append(json.dumps(merged_t_titles))
        target_paper_ids_by_source_col.append(json.dumps({
            "uniprot": list(target_res["dois"]),
            "text_pass1": list(text_res_target.get("pass1_dois", [])),
            "text_pass2": list(text_res_target.get("pass2_dois", [])),
        }))

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
        f"Query UniProt: {len(query_ids_seen)} unique IDs, {len(query_ids_with_papers)} with ≥1 paper, "
        f"{sum(len(json.loads(v)) for v in query_dois_col)} total query papers"
    )
    logger.info(
        f"Target IDs: {len(target_ids_seen)} unique IDs, {len(target_ids_with_papers)} with ≥1 paper, "
        f"{sum(len(json.loads(v)) for v in target_dois_col)} total target papers"
    )
    logger.info(
        f"Query source: {rows_query_from_uniprot} rows from UniProt, {rows_query_from_text} rows from text fallback"
    )
    logger.info(
        f"Target source: {rows_target_from_uniprot} rows from UniProt, {rows_target_from_text} rows from text fallback"
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
            query_ids_by_source = json.loads(query_ids_by_source) if query_ids_by_source else {"uniprot": [], "text_pass1": [], "text_pass2": []}
        if query_ids_by_source is None or (isinstance(query_ids_by_source, float) and pd.isna(query_ids_by_source)):
            query_ids_by_source = {"uniprot": [], "text_pass1": [], "text_pass2": []}
        target_ids_by_source = row.get("target_paper_ids_by_source")
        if isinstance(target_ids_by_source, str):
            target_ids_by_source = json.loads(target_ids_by_source) if target_ids_by_source else {"uniprot": [], "text_pass1": [], "text_pass2": []}
        if target_ids_by_source is None or (isinstance(target_ids_by_source, float) and pd.isna(target_ids_by_source)):
            target_ids_by_source = {"uniprot": [], "text_pass1": [], "text_pass2": []}
        counts = row.get("query_paper_counts")
        if isinstance(counts, str):
            counts = json.loads(counts) if counts else {"uniprot": 0, "text_pass1": 0, "text_pass2": 0}
        if counts is None or (isinstance(counts, float) and pd.isna(counts)):
            counts = {"uniprot": 0, "text_pass1": 0, "text_pass2": 0}
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

