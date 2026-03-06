"""
Analysis module for automated literature search pipeline (Module 4 of 4).

Consumes the high-confidence paper list from collect; downloads full texts
where available; performs in-depth analysis (placeholder for now).

Pipeline Interface:
    run(df_or_path, **kwargs) -> pd.DataFrame
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
API_DELAY = 0.35


def _configure_file_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "analysis_debug.log")
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        rotation="10 MB",
    )
    logger.info(f"Analysis debug log file: {log_path}")


def _paper_id_to_pmcid(paper_id: str) -> Optional[str]:
    """Return PMC id suitable for full-text URL (e.g. PMC123 or 123)."""
    if not paper_id:
        return None
    s = str(paper_id).strip()
    if s.upper().startswith("PMC:"):
        return s[4:].strip() or None
    if s.upper().startswith("PMC"):
        return s
    return None


def fetch_fulltext_xml(
    paper_id: str,
    session: requests.Session,
    cache_dir: Optional[str],
    timeout: int = 60,
) -> Optional[str]:
    """
    Fetch full-text XML for a paper from Europe PMC if available.
    Returns local file path if saved, or None. Saves to cache_dir if set.
    """
    pmcid = _paper_id_to_pmcid(paper_id)
    if not pmcid:
        return None
    if cache_dir:
        safe = pmcid.replace("/", "_").replace(":", "_")
        out_path = os.path.join(cache_dir, f"{safe}.xml")
        if os.path.exists(out_path):
            logger.debug(f"Full-text cache hit: {out_path}")
            return out_path
    time.sleep(API_DELAY)
    url = f"{EUROPEPMC_FULLTEXT_BASE}/{pmcid}/fullTextXML"
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        content = resp.text
    except Exception as e:
        logger.debug(f"Full-text fetch failed for {paper_id}: {e}")
        return None
    if not content or len(content) < 500:
        return None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return out_path
    return None


def run(
    df_or_path,
    output_dir: str = ".",
    fulltext_dir: Optional[str] = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """
    Load high-confidence paper list, download full texts, run placeholder analysis.

    Input can be collect output CSV or a DataFrame (e.g. validated-only subset).
    If columns validated_by_litotar/validated_by_pubtator exist, the pipeline
    may pass only validated rows; otherwise all rows are processed.

    Args:
        df_or_path: DataFrame or path to collect output CSV.
        output_dir: Directory for logs and manifest.
        fulltext_dir: Directory to store downloaded full-text XML (default: output_dir/fulltext).
        no_cache: If True, re-download even if file exists.

    Returns:
        DataFrame with original columns plus fulltext_path, fulltext_available,
        and placeholder analysis columns.
    """
    _configure_file_logging(output_dir)
    fulltext_dir = fulltext_dir or os.path.join(output_dir, "fulltext")
    if no_cache:
        pass
    else:
        os.makedirs(fulltext_dir, exist_ok=True)

    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    if df.empty:
        logger.warning("No papers to analyse")
        df["fulltext_path"] = None
        df["fulltext_available"] = False
        return df

    session = requests.Session()
    paper_ids = df["paper_id"].dropna().unique().tolist()
    path_by_id: Dict[str, Optional[str]] = {}
    for pid in paper_ids:
        path = fetch_fulltext_xml(
            pid,
            session,
            fulltext_dir if not no_cache else None,
        )
        path_by_id[str(pid)] = path

    df["fulltext_path"] = df["paper_id"].map(lambda x: path_by_id.get(str(x)))
    df["fulltext_available"] = df["fulltext_path"].notna()

    manifest = []
    for pid in paper_ids:
        manifest.append({
            "paper_id": pid,
            "path": path_by_id.get(str(pid)),
            "available": path_by_id.get(str(pid)) is not None,
        })
    manifest_path = os.path.join(output_dir, "analysis_fulltext_manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Full-text manifest: {manifest_path}")
    except Exception as e:
        logger.warning(f"Could not write manifest: {e}")

    n_avail = df["fulltext_available"].sum()
    logger.info(f"Analysis: {n_avail}/{len(paper_ids)} papers with full text downloaded")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Module 4: Download full texts and run analysis on high-confidence papers"
    )
    parser.add_argument("-i", "--input", required=True, help="Collect output CSV (high-confidence papers)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV with fulltext_path and analysis columns")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fulltext-dir", default=None, help="Directory for full-text XML files")
    parser.add_argument("--no-cache", action="store_true", help="Re-download full texts")
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output))
    logger.info(f"Reading: {args.input}")
    result = run(
        args.input,
        output_dir=output_dir,
        fulltext_dir=args.fulltext_dir,
        no_cache=args.no_cache,
    )
    result.to_csv(args.output, index=False)
    logger.info(f"Wrote {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
