"""
Collect module for automated literature search pipeline (Module 3 of 4).

Validates papers using paper-centric gene-in-paper checks (Europe PMC Annotations)
and PubTator 3.0 relation extraction. Outputs high-confidence paper list for
full-text analysis; papers not validated by either are flagged for future
LLM evaluation (placeholder only).

Lit-OTAR-style validation is done by paper ID: for each PMID we fetch all gene/protein
annotations from Europe PMC; if our gene (UniProt) appears, we treat it as validated.
No Ensembl IDs required, so it works for non-model organisms.

Pipeline Interface:
    run(df_or_path, **kwargs) -> pd.DataFrame
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from loguru import logger

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_ANNOTATIONS_URL = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
PUBTATOR_EXPORT_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
API_DELAY = 0.35
PUBTATOR_DELAY = 0.35
PUBTATOR_BATCH_SIZE = 100


def _configure_file_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collect_debug.log")
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Collect debug log file: {log_path}")


def paper_id_to_pmid(
    paper_id: str,
    session: requests.Session,
    cache: Dict[str, Optional[str]],
    cache_path: Optional[str] = None,
    timeout: int = 30,
) -> Optional[str]:
    """
    Resolve paper_id (DOI, PMID:123, PMC id, etc.) to PMID string.
    Uses cache; if not cached, queries Europe PMC search by EXT_ID and reads pmid.
    Returns None if already PMID (return numeric part), or if resolution fails.
    """
    if not paper_id or not str(paper_id).strip():
        return None
    s = str(paper_id).strip()
    if s.upper().startswith("PMID:"):
        num = s[5:].strip()
        if num.isdigit():
            return num
        return None
    if s.isdigit() and len(s) <= 8:
        return s
    if paper_id in cache:
        return cache[paper_id]
    time.sleep(API_DELAY)
    query = None
    if s.upper().startswith("DOI:"):
        query = f"EXT_ID:{s[4:].strip()}"
    elif s.upper().startswith("PMC"):
        query = f"EXT_ID:{s}"
    elif re.match(r"^10\.\d+/", s):
        query = f"EXT_ID:{s}"
    if not query:
        cache[paper_id] = None
        return None
    try:
        resp = session.get(
            EUROPEPMC_SEARCH_URL,
            params={"query": query, "format": "json", "pageSize": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"Europe PMC PMID lookup failed for {paper_id}: {e}")
        cache[paper_id] = None
        return None
    results = (data.get("resultList") or {}).get("result") or []
    if not results:
        cache[paper_id] = None
        return None
    rec = results[0]
    pmid = rec.get("pmid")
    if pmid is not None:
        pmid_str = str(pmid).strip()
        if pmid_str.isdigit():
            cache[paper_id] = pmid_str
            return pmid_str
    cache[paper_id] = None
    return None


def fetch_pubtator_biocjson(
    pmids: List[str],
    session: requests.Session,
    cache_dir: Optional[str],
    delay: float = PUBTATOR_DELAY,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Fetch PubTator 3.0 biocjson for a list of PMIDs. Batches in chunks of 100.
    Returns dict pmid -> biocjson document (or empty dict for missing/failed).
    """
    if not pmids:
        return {}
    pmids = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
    pmids = list(dict.fromkeys(pmids))
    out: Dict[str, Any] = {}
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for i in range(0, len(pmids), PUBTATOR_BATCH_SIZE):
        batch = pmids[i : i + PUBTATOR_BATCH_SIZE]
        batch_key = ",".join(sorted(batch))
        if cache_dir:
            safe = re.sub(r"[^\w\-]", "_", batch_key)[:200]
            cache_file = os.path.join(cache_dir, f"pubtator_{safe}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, list):
                        for doc in raw:
                            doc_id = doc.get("id") or (doc.get("passages", [{}])[0].get("document", {}).get("id") if doc.get("passages") else None)
                            if not doc_id and doc.get("passages"):
                                infons = doc.get("passages", [{}])[0].get("infons", {})
                                doc_id = infons.get("pmid") or infons.get("article_id")
                            if doc_id:
                                out[str(doc_id)] = doc
                    elif isinstance(raw, dict):
                        if "documents" in raw:
                            for doc in raw["documents"] or []:
                                doc_id = doc.get("id") if isinstance(doc, dict) else None
                                if not doc_id and isinstance(doc, dict) and doc.get("passages"):
                                    infons = doc.get("passages", [{}])[0].get("infons", {})
                                    doc_id = infons.get("pmid") or infons.get("article_id")
                                if doc_id:
                                    out[str(doc_id)] = doc
                        else:
                            doc_id = raw.get("id")
                            if doc_id:
                                out[str(doc_id)] = raw
                except Exception as e:
                    logger.debug(f"PubTator cache read failed {cache_file}: {e}")
                continue
        time.sleep(delay)
        try:
            resp = session.get(
                PUBTATOR_EXPORT_URL,
                params={"pmids": ",".join(batch)},
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.debug(f"PubTator export failed for batch: {e}")
            continue
        if cache_dir and cache_file:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(raw, f, indent=0)
            except Exception as e:
                logger.debug(f"PubTator cache write failed: {e}")
        if isinstance(raw, list):
            for doc in raw:
                doc_id = doc.get("id")
                if not doc_id and doc.get("passages"):
                    infons = doc.get("passages", [{}])[0].get("infons", {})
                    doc_id = infons.get("pmid") or infons.get("article_id")
                if doc_id:
                    out[str(doc_id)] = doc
        elif isinstance(raw, dict):
            if "documents" in raw:
                for doc in raw["documents"] or []:
                    doc_id = doc.get("id") if isinstance(doc, dict) else None
                    if not doc_id and isinstance(doc, dict) and doc.get("passages"):
                        infons = doc.get("passages", [{}])[0].get("infons", {})
                        doc_id = infons.get("pmid") or infons.get("article_id")
                    if doc_id:
                        out[str(doc_id)] = doc
            else:
                doc_id = raw.get("id")
                if doc_id:
                    out[str(doc_id)] = raw
    return out


def _bioc_doc_pmid(doc: Dict[str, Any]) -> Optional[str]:
    """Extract PMID from a BioC JSON document."""
    doc_id = doc.get("id")
    if doc_id:
        return str(doc_id)
    passages = doc.get("passages") or []
    if passages and isinstance(passages[0], dict):
        infons = passages[0].get("infons") or {}
        return infons.get("pmid") or infons.get("article_id")
    return None


def _extract_gene_ids_from_bioc(doc: Dict[str, Any]) -> Set[str]:
    """Collect all Gene entity IDs (NCBI Gene) from annotations and relations."""
    gene_ids: Set[str] = set()
    passages = doc.get("passages") or []
    for p in passages:
        if not isinstance(p, dict):
            continue
        for ann in p.get("annotations") or []:
            if not isinstance(ann, dict):
                continue
            infons = ann.get("infons") or {}
            if infons.get("type") == "Gene" or "Gene" in str(infons.get("type", "")):
                eid = infons.get("id") or ann.get("id")
                if eid:
                    gene_ids.add(str(eid))
        for rel in p.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            nodes = rel.get("nodes") or []
            for node in nodes:
                if isinstance(node, dict):
                    ref = node.get("refid")
                    if ref:
                        for a in p.get("annotations") or []:
                            if isinstance(a, dict) and a.get("id") == ref:
                                inf = a.get("infons") or {}
                                if inf.get("type") == "Gene":
                                    gid = inf.get("id") or a.get("id")
                                    if gid:
                                        gene_ids.add(str(gid))
                                break
    return gene_ids


def gene_in_pubtator_relations(doc: Dict[str, Any], entrez_id: Optional[str]) -> Tuple[bool, List[str]]:
    """
    Return (True, list of relation types) if the gene (by NCBI Gene/Entrez ID)
    appears in any relation in the document; else (False, []).
    """
    if not entrez_id:
        return False, []
    eid = str(entrez_id).strip()
    if not eid.isdigit():
        return False, []
    gene_ids = _extract_gene_ids_from_bioc(doc)
    if eid not in gene_ids:
        return False, []
    relation_types: List[str] = []
    passages = doc.get("passages") or []
    for p in passages:
        if not isinstance(p, dict):
            continue
        for rel in p.get("relations") or []:
            if isinstance(rel, dict):
                inf = rel.get("infons")
                rel_type = inf.get("type") if isinstance(inf, dict) else rel.get("type")
                if rel_type and rel_type not in relation_types:
                    relation_types.append(str(rel_type))
    return True, relation_types


def fetch_annotations_for_pmid(
    pmid: str,
    session: requests.Session,
    cache: Dict[str, List[Dict[str, Any]]],
    cache_path: Optional[str] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Fetch Europe PMC annotations for one article by PMID. Article id format: MED:pmid."""
    if not pmid or not str(pmid).strip().isdigit():
        return []
    p = str(pmid).strip()
    article_id = f"MED:{p}"
    if article_id in cache:
        return cache[article_id]
    time.sleep(API_DELAY)
    try:
        resp = session.get(
            EUROPEPMC_ANNOTATIONS_URL,
            params={"articleIds": article_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug(f"Annotations API failed for {article_id}: {e}")
        cache[article_id] = []
        return []
    anns = data if isinstance(data, list) else data.get("annotations", data.get("result", []))
    if not isinstance(anns, list) and isinstance(data, dict):
        results = data.get("results") or []
        if results and isinstance(results[0], dict):
            anns = results[0].get("annotations") or []
    if not isinstance(anns, list):
        anns = []
    cache[article_id] = anns
    return anns


def _uniprot_from_annotation(ann: Dict[str, Any]) -> Optional[str]:
    """Extract UniProt accession from an annotation body or target URI (Europe PMC JSON-LD)."""
    def from_uri(uri: Any) -> Optional[str]:
        if not isinstance(uri, str):
            return None
        s = uri.upper().strip()
        for prefix in (
            "UNIPROT.ORG/UNIPROT/",
            "HTTP://WWW.UNIPROT.ORG/UNIPROT/",
            "HTTPS://WWW.UNIPROT.ORG/UNIPROT/",
            "HTTP://PURL.UNIPROT.ORG/UNIPROT/",
            "HTTPS://PURL.UNIPROT.ORG/UNIPROT/",
        ):
            if prefix in s:
                rest = s.split(prefix, 1)[1].split("/")[0].split("?")[0].strip()
                if rest and len(rest) <= 12 and rest.isalnum():
                    return rest
        return None

    body = ann.get("body")
    if isinstance(body, str):
        acc = from_uri(body)
        if acc:
            return acc
    if isinstance(body, dict):
        for k in ("id", "@id"):
            acc = from_uri(body.get(k))
            if acc:
                return acc

    target = ann.get("target")
    if isinstance(target, str):
        acc = from_uri(target)
        if acc:
            return acc
    if isinstance(target, dict):
        for item in target.get("items") or []:
            if isinstance(item, dict):
                acc = from_uri(item.get("id"))
                if acc:
                    return acc
        acc = from_uri(target.get("id") or target.get("@id"))
        if acc:
            return acc
    return None


def build_paper_uniprot_sets(
    pmids: List[str],
    session: requests.Session,
    cache: Dict[str, List[Dict[str, Any]]],
    cache_path: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """
    For each PMID, fetch Europe PMC annotations and collect all UniProt IDs
    mentioned in that paper. Returns pmid -> set of UniProt accessions (upper).
    Enables paper-centric lookup: get all genes for a given paper, match by UniProt.
    """
    pmids = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
    pmids = list(dict.fromkeys(pmids))
    out: Dict[str, Set[str]] = {}
    for p in pmids:
        anns = fetch_annotations_for_pmid(p, session, cache, cache_path)
        accs: Set[str] = set()
        for ann in anns:
            acc = _uniprot_from_annotation(ann)
            if acc:
                accs.add(acc.upper())
        out[p] = accs
    return out


def run(
    df_or_path,
    query_id_col: str = "query",
    target_id_col: str = "target",
    output_dir: str = ".",
    verify_side: str = "both",
    no_cache: bool = False,
    litotar_min_score: float = 0.5,
    pubtator_batch_size: int = PUBTATOR_BATCH_SIZE,
    output_llm_queue_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load search results, validate papers via Europe PMC Annotations (gene-in-paper)
    and PubTator, output high-confidence list and optional LLM queue for unvalidated papers.

    Gene-in-paper validation is paper-centric: for each PMID we fetch all gene/protein
    annotations from Europe PMC; if the row's UniProt ID appears, we treat it as
    validated (litotar_score=1.0). No Ensembl IDs required; works for non-model organisms.

    Args:
        df_or_path: DataFrame or path to search output CSV/JSON.
        query_id_col: Column name for query UniProt ID.
        target_id_col: Column name for target UniProt ID.
        output_dir: Directory for logs and caches.
        verify_side: 'query_only', 'target_only', or 'both'.
        no_cache: If True, do not use file caches.
        litotar_min_score: Currently unused (score is 1.0 when gene is annotated in the paper).
        pubtator_batch_size: Max PMIDs per PubTator request (default 100).
        output_llm_queue_path: If set, write papers needing LLM evaluation here (JSON).

    Returns:
        DataFrame with paper_id, query_uniprot_id, target_uniprot_id, side,
        pmids (resolved), litotar_score, validated_by_litotar, validated_by_pubtator,
        verification_method, needs_llm_evaluation.
    """
    _configure_file_logging(output_dir)
    cache_dir = output_dir
    os.makedirs(cache_dir, exist_ok=True)
    pmid_cache_path = None if no_cache else os.path.join(cache_dir, "collect_pmid_cache.json")
    pmid_cache: Dict[str, Optional[str]] = {}
    if pmid_cache_path and os.path.exists(pmid_cache_path):
        try:
            with open(pmid_cache_path, "r", encoding="utf-8") as f:
                pmid_cache = json.load(f)
            logger.info(f"Loaded PMID cache: {len(pmid_cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load PMID cache: {e}")
    annotations_cache: Dict[str, List[Dict[str, Any]]] = {}
    annotations_cache_path = None if no_cache else os.path.join(cache_dir, "collect_annotations_cache.json")
    if annotations_cache_path and os.path.exists(annotations_cache_path):
        try:
            with open(annotations_cache_path, "r", encoding="utf-8") as f:
                annotations_cache = json.load(f)
            logger.info(f"Loaded annotations cache: {len(annotations_cache)} articles")
        except Exception as e:
            logger.warning(f"Could not load annotations cache: {e}")
    pubtator_cache_dir = None if no_cache else os.path.join(cache_dir, "pubtator_biocjson")

    if isinstance(df_or_path, str):
        path = df_or_path
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = []
            for query_id, alignments in data.items():
                for al in alignments:
                    for pid in al.get("query_paper_dois", []):
                        rows.append({
                            **{query_id_col: query_id, target_id_col: al.get("target", "")},
                            "paper_id": pid,
                            "side": "query",
                        })
                    for pid in al.get("target_paper_dois", []):
                        rows.append({
                            **{query_id_col: query_id, target_id_col: al.get("target", "")},
                            "paper_id": pid,
                            "side": "target",
                        })
            df = pd.DataFrame(rows)
        else:
            df = pd.read_csv(path)
    else:
        df = df_or_path.copy()

    if "paper_id" not in df.columns:
        rows_flat = []
        for _, row in df.iterrows():
            q = row.get(query_id_col)
            t = row.get(target_id_col)
            q_dois = row.get("query_paper_dois")
            t_dois = row.get("target_paper_dois")
            if isinstance(q_dois, str):
                q_dois = json.loads(q_dois) if q_dois else []
            if isinstance(t_dois, str):
                t_dois = json.loads(t_dois) if t_dois else []
            for pid in q_dois or []:
                rows_flat.append({**row.to_dict(), "paper_id": pid, "side": "query"})
            for pid in t_dois or []:
                rows_flat.append({**row.to_dict(), "paper_id": pid, "side": "target"})
        df = pd.DataFrame(rows_flat)

    if df.empty:
        logger.warning("No papers to validate")
        out_df = pd.DataFrame(columns=[
            "paper_id", "query_uniprot_id", "target_uniprot_id", "side", "pmid",
            "litotar_score", "validated_by_litotar", "validated_by_pubtator",
            "verification_method", "needs_llm_evaluation", "pubtator_relation_types",
        ])
        return out_df

    session = requests.Session()
    unique_paper_ids = df["paper_id"].dropna().unique().tolist()
    paper_to_pmid: Dict[str, Optional[str]] = {}
    for pid in unique_paper_ids:
        if pid in paper_to_pmid:
            continue
        pmid = paper_id_to_pmid(
            str(pid),
            session,
            pmid_cache,
            cache_path=pmid_cache_path,
        )
        paper_to_pmid[str(pid)] = pmid

    df = df.copy()
    df["pmid"] = df["paper_id"].map(lambda x: paper_to_pmid.get(str(x)))
    pmids_for_pubtator = [p for p in df["pmid"].dropna().unique().tolist() if str(p).strip().isdigit()]
    pubtator_docs = fetch_pubtator_biocjson(
        pmids_for_pubtator,
        session,
        pubtator_cache_dir,
    )
    pmid_to_doc = {}
    for pmid, doc in pubtator_docs.items():
        pid = _bioc_doc_pmid(doc)
        if pid:
            pmid_to_doc[pid] = doc

    unique_pmids = list(dict.fromkeys(
        str(r.get("pmid")).strip() for _, r in df.iterrows()
        if pd.notna(r.get("pmid")) and str(r.get("pmid")).strip().isdigit()
    ))
    pmid_to_uniprot_set: Dict[str, Set[str]] = build_paper_uniprot_sets(
        unique_pmids,
        session,
        annotations_cache,
        annotations_cache_path,
    )
    if annotations_cache_path and annotations_cache:
        try:
            with open(annotations_cache_path, "w", encoding="utf-8") as f:
                json.dump(annotations_cache, f, indent=2)
            logger.info(f"Saved annotations cache: {len(annotations_cache)} articles")
        except Exception as e:
            logger.warning(f"Could not save annotations cache: {e}")

    results: List[Dict[str, Any]] = []
    llm_queue: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for idx, row in df.iterrows():
        paper_id = row.get("paper_id")
        side = row.get("side", "query")
        if verify_side != "both" and verify_side != side:
            continue
        q_acc = str(row.get(query_id_col, "")).strip() or None
        t_acc = str(row.get(target_id_col, "")).strip() or None
        if not q_acc and not t_acc:
            continue
        uniprot_acc = q_acc if side == "query" else t_acc
        pmid = row.get("pmid")
        if isinstance(pmid, float) and pd.isna(pmid):
            pmid = None
        pmid_str = str(pmid).strip() if pmid else None
        if pmid_str and not pmid_str.isdigit():
            pmid_str = None

        q_entrez = row.get("query_entrez_id")
        t_entrez = row.get("target_entrez_id")
        if pd.isna(q_entrez):
            q_entrez = None
        if pd.isna(t_entrez):
            t_entrez = None
        entrez_id = (q_entrez if side == "query" else t_entrez)
        if entrez_id is not None:
            try:
                entrez_id = str(int(float(entrez_id)))
            except (ValueError, TypeError):
                entrez_id = None

        uniprot_upper = str(uniprot_acc).strip().upper() if uniprot_acc else ""
        validated_by_litotar = bool(
            uniprot_upper and pmid_str and uniprot_upper in pmid_to_uniprot_set.get(pmid_str, set())
        )
        litotar_score = 1.0 if validated_by_litotar else None

        validated_by_pubtator = False
        pubtator_relation_types: List[str] = []
        if pmid_str and entrez_id:
            doc = pmid_to_doc.get(pmid_str)
            if doc is not None:
                found, rel_types = gene_in_pubtator_relations(doc, entrez_id)
                if found:
                    validated_by_pubtator = True
                    pubtator_relation_types = rel_types

        verified = validated_by_litotar or validated_by_pubtator
        if validated_by_litotar and validated_by_pubtator:
            method = "both"
        elif validated_by_litotar:
            method = "litotar"
        elif validated_by_pubtator:
            method = "pubtator"
        else:
            method = "none"

        key = (str(paper_id), q_acc or "", t_acc or "", side)
        if key in seen:
            continue
        seen.add(key)

        needs_llm = not verified
        results.append({
            "paper_id": paper_id,
            "query_uniprot_id": q_acc,
            "target_uniprot_id": t_acc,
            "side": side,
            "pmid": pmid_str,
            "litotar_score": litotar_score,
            "validated_by_litotar": validated_by_litotar,
            "validated_by_pubtator": validated_by_pubtator,
            "verification_method": method,
            "needs_llm_evaluation": needs_llm,
            "pubtator_relation_types": pubtator_relation_types if pubtator_relation_types else None,
        })
        if needs_llm:
            llm_queue.append({
                "paper_id": paper_id,
                "query_uniprot_id": q_acc,
                "target_uniprot_id": t_acc,
                "side": side,
                "pmid": pmid_str,
            })

    out_df = pd.DataFrame(results)
    if pmid_cache_path and pmid_cache:
        try:
            with open(pmid_cache_path, "w", encoding="utf-8") as f:
                json.dump(pmid_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save PMID cache: {e}")

    if output_llm_queue_path and llm_queue:
        try:
            out_dir = os.path.dirname(os.path.abspath(output_llm_queue_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_llm_queue_path, "w", encoding="utf-8") as f:
                json.dump(llm_queue, f, indent=2)
            logger.info(f"LLM queue: {output_llm_queue_path} ({len(llm_queue)} papers)")
        except Exception as e:
            logger.warning(f"Could not write LLM queue: {e}")

    n_high = (out_df["validated_by_litotar"] | out_df["validated_by_pubtator"]).sum()
    logger.info(f"Collect: {n_high} high-confidence papers, {len(llm_queue)} need LLM evaluation, from {len(df)} candidate rows")
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Module 3: Validate papers via Europe PMC Annotations and PubTator, output high-confidence list"
    )
    parser.add_argument("-i", "--input", required=True, help="Search output CSV or JSON")
    parser.add_argument("-o", "--output", required=True, help="Output CSV (high-confidence + needs_llm column)")
    parser.add_argument("--query-id-col", default="query")
    parser.add_argument("--target-id-col", default="target")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verify", choices=["query_only", "target_only", "both"], default="both")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--litotar-min-score",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Unused; litotar_score is 1.0 when gene is annotated in the paper.",
    )
    parser.add_argument(
        "--output-llm-queue",
        default=None,
        metavar="PATH",
        help="Write papers needing LLM evaluation to this JSON file.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output))
    logger.info(f"Reading: {args.input}")
    result = run(
        args.input,
        query_id_col=args.query_id_col,
        target_id_col=args.target_id_col,
        output_dir=output_dir,
        verify_side=args.verify,
        no_cache=args.no_cache,
        litotar_min_score=args.litotar_min_score,
        output_llm_queue_path=args.output_llm_queue,
    )
    result.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
