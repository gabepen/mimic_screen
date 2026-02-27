"""
Mapping module for automated literature search pipeline.

This is Module 1 of 4 in the automated literature search pipeline.
Maps UniProt IDs to Entrez Gene IDs and collects fallback identifiers
(gene names, locus tags, GenBank accessions, common/gene descriptions)
for literature searching.

    Step 1: MyGene.info API → Entrez Gene IDs (only trusted source)
    Step 2: UniProt REST API → fallback IDs (gene name, GenBank acc, locus tags)
    Step 3: UniParc (EBI Proteins API) → same for entries removed from UniProtKB
    Step 4: When locus_tag is missing but GenBank acc is present, fetch locus_tag
            from NCBI protein record (canonical source; e.g. Wolbachia wMel aligns
            GenBank acc with the correct locus tag).
    Step 5: Entrez gene esummary → common/description for mapped Gene IDs
            (papers often use the gene description instead of symbol or tag).

We do not use KEGG locus tags: they are often redundant with NCBI locus tags
or are not available for the specific substrain/microbe.

Pipeline Interface:
    run(df, **kwargs) -> DataFrame
"""

import argparse
import json
import os
import re
import sys
import time
import socket

import pandas as pd
from tqdm import tqdm
import mygene
import requests
import requests.packages.urllib3.util.connection as urllib3_cn
from loguru import logger
from lxml import etree

# ---------------------------------------------------------------------------
# Loguru configuration
# ---------------------------------------------------------------------------
# Remove default handler so we can control output
logger.remove()

# INFO+ to stdout (summary messages only)
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)

def _configure_file_logging(output_dir):
    """Add a DEBUG-level file handler in the given output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "mapping_debug.log")
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Debug log file: {log_path}")


# Forcing IPv4 – avoids IPv6-related connection issues (curl -4 equivalent)
def _force_ipv4():
    """Force requests/urllib3 to use IPv4 sockets only.

    This mirrors curl -4 and fixes cases where IPv6 resolution/connectivity
    to NCBI endpoints is broken but IPv4 works.
    """
    try:
        def allowed_gai_family():
            return socket.AF_INET

        urllib3_cn.allowed_gai_family = allowed_gai_family
        logger.debug("Configured urllib3 to use IPv4 only (AF_INET).")
    except Exception as e:
        # If this fails for any reason, fall back silently – requests will
        # use its default behavior.
        logger.debug(f"Could not force IPv4: {e}")


_force_ipv4()


# ============================================================================
# Step 1 – MyGene.info
# ============================================================================
def map_uniprot_to_entrez_mygene(uniprot_ids, cache_file=None, batch_size=1000):
    """
    Maps UniProt IDs to Entrez Gene IDs using MyGene.info API.

    Args:
        uniprot_ids: List of UniProt IDs to map
        cache_file: Optional JSON cache path
        batch_size: IDs per batch (default 1000)

    Returns:
        dict  {uniprot_id: entrez_gene_id | None}
    """
    cache = _load_cache(cache_file)
    ids_to_query = [uid for uid in uniprot_ids if uid not in cache]

    if not ids_to_query:
        logger.info("MyGene.info – all IDs found in cache")
        return cache

    logger.info(f"MyGene.info – querying {len(ids_to_query)} IDs …")
    mg = mygene.MyGeneInfo()
    mapping_results = cache.copy()

    for i in tqdm(range(0, len(ids_to_query), batch_size), desc="Step 1: MyGene.info"):
        batch = ids_to_query[i:i + batch_size]
        try:
            results = mg.querymany(
                batch,
                scopes=["uniprot", "accession", "uniprot.Swiss-Prot", "uniprot.TrEMBL"],
                fields="entrezgene",
                species="all",
                as_dataframe=False,
                returnall=True,
            )

            for result in results.get("out", []):
                qid = result.get("query", "")
                if not qid:
                    continue
                if "entrezgene" in result:
                    eid = result["entrezgene"]
                    if isinstance(eid, list):
                        eid = eid[0] if eid else None
                    mapping_results[qid] = eid
                    logger.debug(f"MyGene mapped {qid} → {eid}")
                else:
                    mapping_results[qid] = None
                    logger.debug(f"MyGene no match for {qid}")

            for nf in results.get("notfound", []):
                if nf not in mapping_results:
                    mapping_results[nf] = None
                    logger.debug(f"MyGene not found: {nf}")

        except Exception as e:
            logger.error(f"MyGene batch error at index {i}: {e}")
            for uid in batch:
                if uid not in mapping_results:
                    mapping_results[uid] = None

    _save_cache(mapping_results, cache_file)
    return mapping_results


# ---------------------------------------------------------------------------
# Lit-search output: columns written to the separate lit-search file
# (Query/Target IDs + these columns only; main results keep analysis columns only)
# ---------------------------------------------------------------------------
LIT_SEARCH_COLUMNS = [
    "query_entrez_id", "target_entrez_id",
    "query_gene_name", "target_gene_name",
    "query_locus_tag", "target_locus_tag",
    "query_genbank_acc", "target_genbank_acc",
    "query_common_name", "target_common_name",
]


def get_lit_search_columns():
    """Return the list of column names that belong in the lit-search output file."""
    return list(LIT_SEARCH_COLUMNS)


# ============================================================================
# Step 2 – UniProt REST API  →  fallback IDs only (no GeneID – see module docstring)
# ============================================================================
_EMPTY_ENTRY = {
    "gene_id": None, "genbank_acc": None, "refseq_acc": None,
    "kegg_locus": None, "gene_name": None, "locus_tag": None, "status": "not_found",
}


def _extract_uniprot_entry(entry):
    """Extract useful identifiers from a single UniProtKB JSON entry."""
    xrefs = entry.get("uniProtKBCrossReferences", [])

    gene_id = None
    for xref in xrefs:
        if xref.get("database") == "GeneID":
            try:
                gene_id = int(xref.get("id", ""))
            except (ValueError, TypeError):
                pass
            break

    genbank_acc = None
    for xref in xrefs:
        if xref.get("database") == "EnsemblBacteria":
            genbank_acc = xref.get("id")
            break
    if not genbank_acc:
        for xref in xrefs:
            if xref.get("database") == "EMBL":
                for prop in xref.get("properties", []):
                    if prop.get("key") == "ProteinId":
                        val = prop.get("value", "")
                        if val and val != "-":
                            genbank_acc = val
                            break
                if genbank_acc:
                    break

    refseq_acc = None
    any_refseq = None
    for xref in xrefs:
        if xref.get("database") == "RefSeq":
            acc = xref.get("id", "")
            if acc.startswith("WP_"):
                refseq_acc = acc
                break
            if any_refseq is None:
                any_refseq = acc
    refseq_acc = refseq_acc or any_refseq

    kegg_locus = None
    for xref in xrefs:
        if xref.get("database") == "KEGG":
            raw = xref.get("id", "")
            kegg_locus = raw.split(":", 1)[1] if ":" in raw else raw
            break

    gene_name = None
    locus_tag = None
    genes = entry.get("genes", [])
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value")
        ordered = genes[0].get("orderedLocusNames", [])
        if ordered:
            locus_tag = ordered[0].get("value")
    if not locus_tag and kegg_locus:
        locus_tag = kegg_locus

    return {
        "gene_id": gene_id,
        "genbank_acc": genbank_acc,
        "refseq_acc": refseq_acc,
        "kegg_locus": kegg_locus,
        "gene_name": gene_name,
        "locus_tag": locus_tag,
        "status": "found",
    }


def map_uniprot_entries(uniprot_ids, cache_file=None, batch_size=25):
    """
    Fetches UniProtKB entries and extracts GeneID + all fallback identifiers
    in a single pass.

    Returns:
        dict  {uniprot_id: {
            "gene_id":     int | None,
            "genbank_acc": str | None,
            "refseq_acc":  str | None,
            "kegg_locus":  str | None,
            "gene_name":   str | None,
            "locus_tag":   str | None,
            "status":      "found" | "not_found" | "error"
        }}
    """
    cache = _load_cache(cache_file)
    ids_to_query = [uid for uid in uniprot_ids if uid not in cache]

    if not ids_to_query:
        logger.info("UniProt entries – all IDs found in cache")
        return cache

    logger.info(f"UniProt entries – querying {len(ids_to_query)} IDs …")
    results = cache.copy()
    base_url = "https://rest.uniprot.org/uniprotkb/stream"

    for i in tqdm(range(0, len(ids_to_query), batch_size), desc="Step 2: UniProt"):
        batch = ids_to_query[i:i + batch_size]
        try:
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            resp = requests.get(base_url, params={
                "query": query,
                "format": "json",
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            found_ids = set()
            for entry in data.get("results", []):
                uid = entry.get("primaryAccession")
                if not uid:
                    continue
                found_ids.add(uid)
                info = _extract_uniprot_entry(entry)
                results[uid] = info

                parts = []
                if info["gene_id"]:
                    parts.append(f"gid={info['gene_id']}")
                if info["genbank_acc"]:
                    parts.append(f"gb={info['genbank_acc']}")
                if info["locus_tag"]:
                    parts.append(f"locus={info['locus_tag']}")
                if info["gene_name"]:
                    parts.append(f"gene={info['gene_name']}")
                logger.debug(f"UniProt {uid}: {', '.join(parts) if parts else 'no usable IDs'}")

            for uid in batch:
                if uid not in found_ids and uid not in results:
                    results[uid] = {**_EMPTY_ENTRY}
                    logger.debug(f"UniProt {uid}: entry not found")

            time.sleep(0.5)

        except Exception as e:
            logger.error(f"UniProt batch error at index {i}: {e}")
            for uid in batch:
                if uid not in results:
                    results[uid] = {**_EMPTY_ENTRY, "status": "error"}

    _save_cache(results, cache_file)
    return results


# ============================================================================
# Step 3 – UniParc (EBI Proteins API) for entries removed from UniProtKB
# ============================================================================
def _extract_uniparc_xrefs(db_references, taxid):
    """
    Extract identifiers from EBI Proteins API UniParc dbReference list,
    filtered to a specific taxon ID.
    """
    gene_name = None
    genbank_acc = None
    refseq_acc = None
    kegg_locus = None
    locus_tag = None

    for xref in db_references:
        props = {p["type"]: p["value"] for p in xref.get("property", [])}
        xref_taxid = props.get("NCBI_taxonomy_id", "")

        if taxid and str(xref_taxid) != str(taxid):
            continue

        db_type = xref.get("type", "")
        xref_id = xref.get("id", "")

        if db_type in ("UniProtKB/TrEMBL", "UniProtKB/Swiss-Prot"):
            if not gene_name:
                gene_name = props.get("gene_name")
            if not locus_tag:
                locus_tag = props.get("locus_tag") or props.get("ordered_locus")

        elif db_type == "RefSeq" and not refseq_acc:
            refseq_acc = xref_id
            if not gene_name:
                gene_name = props.get("gene_name")
            if not locus_tag:
                locus_tag = props.get("locus_tag") or props.get("ordered_locus")

        elif db_type in ("EMBL", "EMBLWGS", "TREMBLNEW") and not genbank_acc:
            if xref_id and xref_id != "-":
                genbank_acc = xref_id
            if not gene_name:
                gene_name = props.get("gene_name")
            if not locus_tag:
                locus_tag = props.get("locus_tag") or props.get("ordered_locus")

        elif db_type == "EnsemblBacteria" and not genbank_acc:
            genbank_acc = xref_id

    # If no explicit locus_tag, use gene_name when it looks like a locus tag
    if not locus_tag and gene_name and re.match(r"^[A-Za-z]{2,}[_\s][0-9]+$", gene_name):
        locus_tag = gene_name

    return {
        "gene_id": None,
        "genbank_acc": genbank_acc,
        "refseq_acc": refseq_acc,
        "kegg_locus": kegg_locus,
        "gene_name": gene_name,
        "locus_tag": locus_tag,
        "status": "found",
    }


def map_uniparc_entries(uniprot_ids, taxid=None, cache_file=None):
    """
    Looks up UniProt IDs in the UniParc archive (EBI Proteins API).
    For entries removed from UniProtKB, UniParc retains the sequence
    and cross-references from all source databases.

    Uses the EBI Proteins API (per-ID) which returns cross-references
    tagged with taxon IDs for organism-specific filtering.

    Returns:
        dict  {uniprot_id: {same fields as map_uniprot_entries} }
    """
    cache = _load_cache(cache_file)
    ids_to_query = [uid for uid in uniprot_ids if uid not in cache]

    if not ids_to_query:
        logger.info("UniParc – all IDs found in cache")
        return cache

    logger.info(f"UniParc – querying {len(ids_to_query)} IDs (taxid={taxid}) …")
    results = cache.copy()
    base_url = "https://www.ebi.ac.uk/proteins/api/uniparc/accession"

    for uid in tqdm(ids_to_query, desc="Step 3: UniParc"):
        try:
            resp = requests.get(
                f"{base_url}/{uid}",
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 404:
                results[uid] = {**_EMPTY_ENTRY}
                logger.debug(f"UniParc {uid}: not found")
                time.sleep(0.2)
                continue
            resp.raise_for_status()
            data = resp.json()

            db_refs = data.get("dbReference", [])
            info = _extract_uniparc_xrefs(db_refs, taxid)
            results[uid] = info

            parts = []
            if info["locus_tag"]:
                parts.append(f"locus={info['locus_tag']}")
            if info["gene_name"]:
                parts.append(f"gene={info['gene_name']}")
            if info["genbank_acc"]:
                parts.append(f"gb={info['genbank_acc']}")
            if info["refseq_acc"]:
                parts.append(f"rs={info['refseq_acc']}")
            logger.debug(f"UniParc {uid}: {', '.join(parts) if parts else 'no xrefs for taxid'}")

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"UniParc error for {uid}: {e}")
            results[uid] = {**_EMPTY_ENTRY, "status": "error"}

    _save_cache(results, cache_file)
    return results



# ============================================================================
# Shared helpers
# ============================================================================
def _load_cache(cache_file):
    """Load a JSON cache file, returning {} on any error."""
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
        logger.debug(f"Loaded {len(cache)} entries from cache {cache_file}")
        return cache
    except Exception as e:
        logger.warning(f"Could not load cache {cache_file}: {e}")
        return {}


def _save_cache(data, cache_file):
    """Persist a dict to a JSON cache file."""
    if not cache_file:
        return
    try:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(data)} entries to cache {cache_file}")
    except Exception as e:
        logger.warning(f"Could not save cache {cache_file}: {e}")


# ============================================================================
# NCBI helpers: locus_tag from GenBank protein, gene description (common name)
# ============================================================================
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_RATE_DELAY = 0.35


def _ncbi_get(operation, params, max_retries=3, timeout=30):
    """GET request to NCBI E-utilities with rate limit and retries.
    Use longer timeout for efetch (batch responses can be large)."""
    time.sleep(NCBI_RATE_DELAY)
    url = f"{NCBI_EUTILS}/{operation}.fcgi"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning(f"NCBI {operation} failed (attempt {attempt + 1}), retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def _batch_genbank_to_locus_tag(accessions, timeout=60):
    """
    Fetch multiple protein records in one efetch and parse locus_tag from each.
    accessions: list of GenBank protein accessions (e.g. from RefSeq WP_, or GenBank).
    Returns dict: accession -> locus_tag or None.
    """
    if not accessions:
        return {}
    # NCBI efetch accepts comma-separated ids for batch
    id_param = ",".join(str(a).strip() for a in accessions)
    try:
        resp = _ncbi_get(
            "efetch",
            {
                "db": "protein",
                "id": id_param,
                "rettype": "gb",
                "retmode": "xml",
            },
            timeout=timeout,
        )
        xml_bytes = resp.content
    except Exception as e:
        logger.debug(f"efetch batch protein failed: {e}")
        return {str(acc).strip(): None for acc in accessions}

    # convert xml to tree
    try:
        genpept_tree = etree.fromstring(xml_bytes)
    except Exception as e:
        logger.debug(f"Failed to parse GenPept XML: {e}")
        return {str(acc).strip(): None for acc in accessions}

    out = {}

    # split sequences from batch xml
    for gbseq in genpept_tree.findall(".//GBSeq"):
        # get genbank accession for matching to accessions list
        accession = gbseq.findtext("./GBSeq_primary-accession")
        if not accession:
            accession = gbseq.findtext("./GBSeq_accession-version")
        if not accession:
            logger.warning(f"No accession found in GenBank record {gbseq.findtext('./GBSeq_locus-name')}")

        # get feature table for parsing locus_tag
        cds_feats = gbseq.xpath(".//GBFeature[GBFeature_key='CDS']")

        locus_tag = None
        if not cds_feats:
            logger.debug(f"No CDS features found for {accession}")
        else:
            if len(cds_feats) > 1:
                logger.warning(f"Multiple CDS features found for {accession} selecting first")
            cds_feat = cds_feats[0]
            for qual in cds_feat.findall("./GBFeature_quals/GBQualifier"):
                if qual.findtext("./GBQualifier_name") == "locus_tag":
                    locus_tag = qual.findtext("./GBQualifier_value")
                    break
            if locus_tag is None:
                logger.debug(f"No locus tag found for {accession}")

        # store by accession and versionless accession (QQL98079.1 -> QQL98079)
        accession = accession.strip()
        out[accession] = locus_tag
        if "." in accession:
            out[accession.split(".", 1)[0]] = locus_tag

    # Shape into accession -> locus_tag dict for every requested accession
    result = {}
    for acc in accessions:
        key = str(acc).strip()
        result[key] = out.get(key)
    return result

def fill_locus_tag_from_genbank(fallback_ids, cache_file=None, batch_size=50):
    """
    For each uid in fallback_ids that has genbank_acc but no locus_tag,
    get locus_tag via batch NCBI protein efetch (one request per batch of accessions).
    Updates fallback_ids in place.
    """
    cache = _load_cache(cache_file) if cache_file else {}
    # Collect accessions we need to look up (no locus_tag, have genbank_acc)
    acc_to_uids = {}  # acc -> list of uid that use this acc
    for uid, fb in fallback_ids.items():
        if fb.get("locus_tag"):
            continue
        acc = fb.get("genbank_acc")
        if not acc or not str(acc).strip():
            continue
        acc = str(acc).strip()
        acc_to_uids.setdefault(acc, []).append(uid)
    # Resolve from cache or batch fetch
    to_fetch = [acc for acc in acc_to_uids if acc not in cache]
    if to_fetch:
        logger.info(f"GenBank→locus_tag: batch fetching {len(to_fetch)} accessions ({batch_size} per request)")
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        batch_result = _batch_genbank_to_locus_tag(batch)
        for acc, locus_tag in batch_result.items():
            cache[acc] = locus_tag
    # Apply to fallback_ids
    updated = 0
    for acc, uids in acc_to_uids.items():
        lt = cache.get(acc)
        if lt:
            for uid in uids:
                fallback_ids[uid]["locus_tag"] = lt
            updated += len(uids)
            logger.debug(f"GenBank {acc} → locus_tag={lt} for {len(uids)} id(s)")
    if cache_file and cache:
        _save_cache(cache, cache_file)
    return updated


def _gene_description_from_entrez(gene_id):
    """Fetch gene description (common name) from NCBI Gene via esummary. Returns str or None."""
    try:
        resp = _ncbi_get("esummary", {"db": "gene", "id": gene_id, "retmode": "json"})
        data = resp.json()
        result = data.get("result", {}).get(str(gene_id), {})
        return result.get("description") or result.get("title") or None
    except Exception as e:
        logger.debug(f"esummary gene for {gene_id}: {e}")
        return None


def fetch_gene_descriptions(gene_ids, cache_file=None):
    """
    For each Entrez Gene ID get the description (common name) from NCBI.
    Returns dict: gene_id (int) -> description string or None.
    """
    gene_ids = [int(g) for g in gene_ids if g is not None]
    gene_ids = list({g for g in gene_ids if g})
    if not gene_ids:
        return {}
    cache = _load_cache(cache_file) if cache_file else {}
    out = {}
    for gid in gene_ids:
        key = str(gid)
        if key in cache:
            out[gid] = cache[key]
        else:
            desc = _gene_description_from_entrez(gid)
            out[gid] = desc
            cache[key] = desc
    if cache_file:
        _save_cache(cache, cache_file)
    return out


def _log_step_summary(step_name, id_mapping, all_ids):
    """Log a one-line INFO summary for a pipeline step."""
    mapped = sum(1 for uid in all_ids if id_mapping.get(uid) is not None)
    remaining = len(all_ids) - mapped
    logger.info(f"{step_name}: {mapped}/{len(all_ids)} mapped, {remaining} remaining")


# ============================================================================
# Pipeline entry point
# ============================================================================
def run(df, query_col="query", target_col="target",
        output_dir=".", cache_dir=None, batch_size=1000, no_cache=False,
        query_taxid=None, target_taxid=None):
    """
    Pipeline function: Maps UniProt IDs → Entrez Gene IDs + fallback identifiers.

    Executes a 3-step cascade:
        1. MyGene.info API → Entrez Gene IDs (only trusted source)
        2. UniProt REST API → fallback IDs (gene name, GenBank acc, locus tags)
        3. UniParc (EBI Proteins API) → same for entries removed from UniProtKB

    Note: UniProt GeneID cross-references are NOT used — they contain
    stale/reused NCBI Gene IDs that map to wrong organisms.

    Args:
        df:           DataFrame with UniProt IDs in *query_col* and *target_col*
        query_col:    Column name for query UniProt IDs  (default: 'query')
        target_col:   Column name for target UniProt IDs (default: 'target')
        output_dir:   Directory for log files and stats   (default: '.')
        cache_dir:    Directory for cache JSON files      (default: output_dir)
        batch_size:   Batch size for MyGene.info          (default: 1000)
        no_cache:     If True, ignore existing cache files (default: False)
        query_taxid:  NCBI taxonomy ID for query organism  (optional)
        target_taxid: NCBI taxonomy ID for target organism (optional)

    Returns:
        DataFrame with only Query/Target IDs and lit-search columns:
            query_entrez_id, target_entrez_id,
            query_gene_name, target_gene_name,
            query_locus_tag, target_locus_tag,
            query_genbank_acc, target_genbank_acc,
            query_common_name, target_common_name
        (Intermediate format for the pipeline; no analysis columns.)
    """
    _configure_file_logging(output_dir)

    if cache_dir is None:
        cache_dir = output_dir
    os.makedirs(cache_dir, exist_ok=True)

    CACHE_FILES = {
        "mygene":    "mygene_cache.json",
        "uniprot":   "uniprot_entries_cache.json",
        "uniparc":   "uniparc_entries_cache.json",
        "genbank_locus": "genbank_locus_tag_cache.json",
        "entrez_description": "entrez_gene_description_cache.json",
    }

    def _cache(key):
        if no_cache:
            return None
        return os.path.join(cache_dir, CACHE_FILES[key])

    if no_cache:
        logger.info("Cache disabled – all steps will query APIs fresh")

    # Separate query and target IDs so we can attach the right taxid later
    query_ids = set()
    target_ids = set()
    if query_col in df.columns:
        query_ids = set(df[query_col].dropna().unique())
    if target_col in df.columns:
        target_ids = set(df[target_col].dropna().unique())
    all_ids = query_ids | target_ids

    logger.info(f"Mapping module – {len(all_ids)} unique UniProt IDs to map")
    if query_taxid:
        logger.info(f"  Query organism taxid:  {query_taxid}")
    if target_taxid:
        logger.info(f"  Target organism taxid: {target_taxid}")

    # Master mapping: uniprot_id → entrez_gene_id
    id_mapping = {uid: None for uid in all_ids}
    # Fallback identifiers collected across Steps 2-3
    fallback_ids = {}  # uid → {"locus_tag": ..., "gene_name": ..., "genbank_acc": ...}

    # ------------------------------------------------------------------
    # Step 1: MyGene.info
    # ------------------------------------------------------------------
    mygene_results = map_uniprot_to_entrez_mygene(
        list(all_ids), cache_file=_cache("mygene"), batch_size=batch_size
    )
    for uid, eid in mygene_results.items():
        if eid is not None:
            id_mapping[uid] = eid
    _log_step_summary("Step 1 (MyGene.info)", id_mapping, all_ids)

    # ------------------------------------------------------------------
    # Step 2: UniProt REST API  →  fallback IDs only
    # (UniProt GeneID xrefs are unreliable — stale/reused NCBI Gene IDs)
    # ------------------------------------------------------------------
    unmapped = [uid for uid in all_ids if id_mapping[uid] is None]
    not_found_in_uniprot = []

    if unmapped:
        uniprot_results = map_uniprot_entries(
            unmapped, cache_file=_cache("uniprot"), batch_size=25
        )
        for uid, info in uniprot_results.items():
            if not isinstance(info, dict):
                continue
            fallback_ids[uid] = {
                "locus_tag": info.get("locus_tag"),
                "gene_name": info.get("gene_name"),
                "genbank_acc": info.get("genbank_acc"),
            }
            if info.get("status") == "not_found":
                not_found_in_uniprot.append(uid)

        n_found = sum(1 for uid in unmapped
                      if isinstance(uniprot_results.get(uid), dict)
                      and uniprot_results[uid].get("status") == "found")
        n_locus = sum(1 for uid in unmapped
                      if isinstance(uniprot_results.get(uid), dict)
                      and uniprot_results[uid].get("locus_tag"))
        n_genbank = sum(1 for uid in unmapped
                        if isinstance(uniprot_results.get(uid), dict)
                        and uniprot_results[uid].get("genbank_acc"))
        logger.info(
            f"Step 2 (UniProt): {n_found}/{len(unmapped)} entries found, "
            f"{n_locus} locus tag, {n_genbank} GenBank acc, "
            f"{len(not_found_in_uniprot)} not in UniProtKB → Step 3"
        )
    else:
        logger.info("Step 2 (UniProt): skipped – no unmapped IDs")

    # ------------------------------------------------------------------
    # Step 3: UniParc  →  for IDs not found in UniProtKB
    # Split by query vs target to use the correct taxid for filtering
    # ------------------------------------------------------------------
    if not_found_in_uniprot:
        nf_query = [uid for uid in not_found_in_uniprot if uid in query_ids]
        nf_target = [uid for uid in not_found_in_uniprot if uid in target_ids]
        # IDs may appear in both query and target; avoid double-querying
        nf_target_only = [uid for uid in nf_target if uid not in query_ids]

        uniparc_results = {}

        if nf_query:
            logger.info(f"Step 3a (UniParc query IDs): {len(nf_query)} IDs, taxid={query_taxid}")
            res_q = map_uniparc_entries(
                nf_query, taxid=query_taxid, cache_file=_cache("uniparc")
            )
            uniparc_results.update(res_q)

        if nf_target_only:
            logger.info(f"Step 3b (UniParc target IDs): {len(nf_target_only)} IDs, taxid={target_taxid}")
            res_t = map_uniparc_entries(
                nf_target_only, taxid=target_taxid, cache_file=_cache("uniparc")
            )
            uniparc_results.update(res_t)

        for uid, info in uniparc_results.items():
            if not isinstance(info, dict):
                continue
            existing = fallback_ids.get(uid, {})
            fallback_ids[uid] = {
                "locus_tag": existing.get("locus_tag") or info.get("locus_tag"),
                "gene_name": existing.get("gene_name") or info.get("gene_name"),
                "genbank_acc": existing.get("genbank_acc") or info.get("genbank_acc"),
            }

        n_uniparc_found = sum(1 for info in uniparc_results.values()
                              if isinstance(info, dict) and info.get("status") == "found")
        logger.info(
            f"Step 3 (UniParc): {n_uniparc_found}/{len(not_found_in_uniprot)} "
            f"entries recovered from archive"
        )
        _log_step_summary("Step 3 (UniParc)", id_mapping, all_ids)
    else:
        logger.info("Step 3 (UniParc): skipped – no missing UniProt entries")

    # ------------------------------------------------------------------
    # Step 4: When locus_tag is missing but GenBank acc present, get from NCBI
    # (GenBank accession consistently coincides with correct locus tag for e.g. wMel)
    # ------------------------------------------------------------------
    n_filled = fill_locus_tag_from_genbank(fallback_ids, cache_file=_cache("genbank_locus"))
    if n_filled:
        logger.info(f"Step 4 (GenBank→locus_tag): filled {n_filled} locus tags from NCBI protein")

    # ------------------------------------------------------------------
    # Step 5: Gene descriptions (common names) from Entrez for literature search
    # ------------------------------------------------------------------
    unique_gene_ids = list({eid for eid in id_mapping.values() if eid is not None})
    gene_descriptions = fetch_gene_descriptions(unique_gene_ids, cache_file=_cache("entrez_description"))
    if gene_descriptions:
        n_with_desc = sum(1 for v in gene_descriptions.values() if v)
        logger.info(f"Step 5 (Entrez description): {n_with_desc}/{len(gene_descriptions)} gene descriptions fetched")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total = len(all_ids)
    mapped = sum(1 for v in id_mapping.values() if v is not None)
    unmapped_final = total - mapped
    pct = 100 * mapped / total if total else 0.0
    logger.info(
        f"Final: {mapped}/{total} mapped ({pct:.1f}%), {unmapped_final} unmapped"
    )

    _save_mapping_stats(id_mapping, fallback_ids, output_dir)

    # ------------------------------------------------------------------
    # Apply mapping to DataFrame
    # ------------------------------------------------------------------
    def _safe_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return None

    result_df = df.copy()
    if query_col in result_df.columns:
        result_df["query_entrez_id"] = result_df[query_col].map(id_mapping)
        result_df["query_gene_name"] = result_df[query_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("gene_name"))
        result_df["query_locus_tag"] = result_df[query_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("locus_tag"))
        result_df["query_genbank_acc"] = result_df[query_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("genbank_acc"))
        result_df["query_common_name"] = result_df["query_entrez_id"].map(
            lambda eid: gene_descriptions.get(_safe_int(eid)) if _safe_int(eid) is not None else None
        )
    if target_col in result_df.columns:
        result_df["target_entrez_id"] = result_df[target_col].map(id_mapping)
        result_df["target_gene_name"] = result_df[target_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("gene_name"))
        result_df["target_locus_tag"] = result_df[target_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("locus_tag"))
        result_df["target_genbank_acc"] = result_df[target_col].map(
            lambda uid: fallback_ids.get(uid, {}).get("genbank_acc"))
        result_df["target_common_name"] = result_df["target_entrez_id"].map(
            lambda eid: gene_descriptions.get(_safe_int(eid)) if _safe_int(eid) is not None else None
        )

    if query_col in result_df.columns:
        mq = result_df["query_entrez_id"].notna().sum()
        logger.info(f"Query column:  {mq}/{len(result_df)} rows mapped")
    if target_col in result_df.columns:
        mt = result_df["target_entrez_id"].notna().sum()
        logger.info(f"Target column: {mt}/{len(result_df)} rows mapped")

    # Return only the lit-search table (query/target IDs + mapping columns)
    id_cols = [c for c in [query_col, target_col] if c in result_df.columns]
    lit_cols = [c for c in LIT_SEARCH_COLUMNS if c in result_df.columns]
    return result_df[id_cols + lit_cols].copy()


# ============================================================================
# Stats output
# ============================================================================
def _save_mapping_stats(id_mapping, fallback_ids, output_dir):
    """Write a JSON summary of mapped / unmapped IDs and available fallbacks."""
    stats_file = os.path.join(output_dir, "mapping_stats.json")

    mapped_ids = {uid: eid for uid, eid in id_mapping.items() if eid is not None}
    unmapped_ids = [uid for uid, eid in id_mapping.items() if eid is None]

    stats = {
        "total_ids": len(id_mapping),
        "mapped_count": len(mapped_ids),
        "unmapped_count": len(unmapped_ids),
        "mapping_rate": len(mapped_ids) / len(id_mapping) if id_mapping else 0.0,
        "mapped_ids": mapped_ids,
        "unmapped_ids": unmapped_ids,
        "fallback_ids": fallback_ids,
    }

    try:
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Mapping stats saved to {stats_file}")
    except Exception as e:
        logger.warning(f"Could not save stats file: {e}")


# ============================================================================
# CLI
# ============================================================================
def main():
    """Command-line interface for the mapping module."""
    parser = argparse.ArgumentParser(
        description="Map UniProt IDs to Entrez Gene IDs (4-step cascade)"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to alignment_analysis results CSV",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to lit-search output CSV (Query/Target IDs + mapping columns; pipeline intermediate)",
    )
    parser.add_argument(
        "-d", "--output-dir", type=str, default=None,
        help="Directory for logs, caches, and stats (default: same as output file)",
    )
    parser.add_argument(
        "--query-col", type=str, default="query",
        help="Column name for query UniProt IDs (default: query)",
    )
    parser.add_argument(
        "--target-col", type=str, default="target",
        help="Column name for target UniProt IDs (default: target)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Batch size for MyGene.info queries (default: 1000)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing cache files and re-query all APIs",
    )
    parser.add_argument(
        "--query-taxid", type=int, default=None,
        help="NCBI taxonomy ID for query organism (e.g. 163164 for Wolbachia wMel)",
    )
    parser.add_argument(
        "--target-taxid", type=int, default=None,
        help="NCBI taxonomy ID for target organism (e.g. 7227 for Drosophila melanogaster)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output))

    logger.info(f"Reading input: {args.input}")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows")
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return 1

    result_df = run(
        df,
        query_col=args.query_col,
        target_col=args.target_col,
        output_dir=output_dir,
        batch_size=args.batch_size,
        no_cache=args.no_cache,
        query_taxid=args.query_taxid,
        target_taxid=args.target_taxid,
    )

    try:
        logger.info(f"Writing lit-search output: {args.output}")
        result_df.to_csv(args.output, index=False)
        logger.info(f"Saved {len(result_df)} rows")
    except Exception as e:
        logger.error(f"Error writing output: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error writing output: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
