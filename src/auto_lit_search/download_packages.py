"""Download-only entry point for lit-download container (no pydantic / GPU deps)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import requests
from loguru import logger

from auto_lit_search.collect import download_papers_to_dir
from auto_lit_search.download_manifest import (
    DOWNLOAD_MANIFEST_FILENAME,
    _alignment_id_for_pair,
    _alignment_paper_ids,
    _emit_download_progress_summary,
    _infer_docling_required_basenames_from_disk,
    _load_download_manifest,
    _load_search_json,
    _manifest_row_satisfied,
    _merge_recs_into_manifest,
    _paper_pair_key,
    _write_docling_eval_manifest,
    _write_download_manifest_atomic,
    build_paper_identifier_index,
    write_paper_identifier_index,
)

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


def _only_download_progress_log(record: Dict[str, Any]) -> bool:
    return record["extra"].get("download_progress") is True


def _results_path(output_root: str, alignment_id: str) -> str:
    return os.path.join(output_root, f"{alignment_id}_results.json")


def run_download_packages_only(
    paper_ids_path: str,
    data_root: str,
    *,
    output_root: str = "",
    idmap_csv: str = "",
    no_cache: bool = False,
    max_alignments: int | None = None,
    alignment_id: str = "",
    skip_if_results: bool = False,
) -> None:
    """Download per-alignment packages without GPU / Docling / Grader."""
    collection_org = os.environ.get("COLLECTION_ORG", "ucsc").strip() or "ucsc"
    collection_auth_scope = (
        os.environ.get("COLLECTION_AUTH_SCOPE", "email_only").strip() or "email_only"
    )
    collector_email = os.environ.get("COLLECTOR_EMAIL", "").strip()
    if (
        collection_org.lower() == "ucsc"
        and collection_auth_scope.lower() == "email_only"
        and not collector_email
    ):
        raise RuntimeError(
            "COLLECTOR_EMAIL is required for UCSC email_only collection mode."
        )

    _mw_raw = os.environ.get("COLLECT_MAX_WORKERS", "2").strip() or "2"
    try:
        collect_max_workers = max(1, min(16, int(_mw_raw)))
    except ValueError:
        collect_max_workers = 2
    collect_disable_s2 = os.environ.get(
        "COLLECT_DISABLE_SEMANTIC_SCHOLAR", ""
    ).strip().lower() in ("1", "true", "yes")

    if not output_root:
        output_root = os.path.join(data_root, "llm_results")
    papers_base = os.path.join(data_root, "papers")
    logs_base = os.path.join(data_root, "logs")
    os.makedirs(papers_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    if os.environ.get("DOWNLOAD_PROGRESS_LOG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    ):
        progress_path = (
            os.environ.get("DOWNLOAD_PROGRESS_LOG_PATH", "").strip()
            or os.path.join(logs_base, "download_progress.log")
        )
        try:
            os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
            logger.add(
                progress_path,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                filter=_only_download_progress_log,
                enqueue=True,
            )
        except Exception as exc:
            logger.warning("Could not attach download progress log {}: {}", progress_path, exc)

    data = _load_search_json(paper_ids_path)
    session = requests.Session()
    pmcid_cache: Dict[str, str | None] = {}
    processed = 0

    for query_id, alignments in data.items():
        if not isinstance(alignments, list):
            continue
        for al in alignments:
            target = al.get("target") or ""
            aid = _alignment_id_for_pair(query_id, target)
            if alignment_id and aid != alignment_id:
                continue
            paper_ids_src = _alignment_paper_ids(al)
            if not paper_ids_src:
                logger.warning("Alignment {}: no paper IDs", aid)
                continue
            if skip_if_results and os.path.isfile(_results_path(output_root, aid)):
                logger.info("Alignment {}: skipping (results exist)", aid)
                continue

            papers_dir = os.path.join(papers_base, aid)
            os.makedirs(papers_dir, exist_ok=True)
            manifest_path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
            manifest_map = _load_download_manifest(manifest_path)
            _emit_download_progress_summary(
                aid, paper_ids_src, manifest_map, papers_dir, "before_collect"
            )

            existing_txt = [
                fname
                for fname in os.listdir(papers_dir)
                if fname.endswith(".txt")
                and os.path.isfile(os.path.join(papers_dir, fname))
                and os.path.getsize(os.path.join(papers_dir, fname)) > 0
            ]
            existing_pdf_dir = os.path.join(papers_dir, "pdf")
            existing_pdf = os.path.isdir(existing_pdf_dir) and any(
                fname.endswith(".pdf")
                and os.path.isfile(os.path.join(existing_pdf_dir, fname))
                and os.path.getsize(os.path.join(existing_pdf_dir, fname)) > 0
                for fname in os.listdir(existing_pdf_dir)
            )
            can_resume_from_disk = bool(existing_txt or existing_pdf) and not no_cache

            missing_pre: List[Tuple[str, str]] = []
            for pid, src in paper_ids_src:
                key = _paper_pair_key(pid, src)
                row = manifest_map.get(key)
                if row is None or not _manifest_row_satisfied(row, papers_dir):
                    missing_pre.append((pid, src))

            recs = []
            if no_cache:
                logger.info(
                    "Alignment {}: no_cache re-downloading all {} papers",
                    aid,
                    len(paper_ids_src),
                )
                recs = download_papers_to_dir(
                    paper_ids_src,
                    papers_dir,
                    session=session,
                    pmcid_cache=pmcid_cache,
                    no_cache=True,
                    force_pdfs=True,
                    prefer_pdf_text=True,
                    collection_org=collection_org,
                    auth_scope=collection_auth_scope,
                    collector_email=collector_email or None,
                    max_workers=collect_max_workers,
                    disable_semantic_scholar=collect_disable_s2,
                )
                manifest_map = _merge_recs_into_manifest({}, recs)
                _write_download_manifest_atomic(manifest_path, manifest_map)
            elif missing_pre:
                logger.info(
                    "Alignment {}: downloading {} missing papers ({} already on disk)",
                    aid,
                    len(missing_pre),
                    len(paper_ids_src) - len(missing_pre),
                )
                recs = download_papers_to_dir(
                    missing_pre,
                    papers_dir,
                    session=session,
                    pmcid_cache=pmcid_cache,
                    no_cache=no_cache,
                    force_pdfs=True,
                    prefer_pdf_text=True,
                    collection_org=collection_org,
                    auth_scope=collection_auth_scope,
                    collector_email=collector_email or None,
                    max_workers=collect_max_workers,
                    disable_semantic_scholar=collect_disable_s2,
                )
                manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                _write_download_manifest_atomic(manifest_path, manifest_map)
            elif can_resume_from_disk:
                n_docling = len(_infer_docling_required_basenames_from_disk(papers_dir))
                logger.info(
                    "Alignment {} reusing existing artifacts (txt={} pending_docling={})",
                    aid,
                    len(existing_txt),
                    n_docling,
                )
            else:
                logger.info("Alignment {}: downloading {} papers", aid, len(paper_ids_src))
                recs = download_papers_to_dir(
                    paper_ids_src,
                    papers_dir,
                    session=session,
                    pmcid_cache=pmcid_cache,
                    no_cache=no_cache,
                    force_pdfs=True,
                    prefer_pdf_text=True,
                    collection_org=collection_org,
                    auth_scope=collection_auth_scope,
                    collector_email=collector_email or None,
                    max_workers=collect_max_workers,
                    disable_semantic_scholar=collect_disable_s2,
                )
                manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                _write_download_manifest_atomic(manifest_path, manifest_map)

            docling_required = _infer_docling_required_basenames_from_disk(papers_dir)
            if recs:
                try:
                    _write_docling_eval_manifest(papers_dir, recs, docling_required)
                except Exception as exc:
                    logger.warning("Alignment {}: could not write docling manifest: {}", aid, exc)
            elif docling_required:
                try:
                    _write_docling_eval_manifest(papers_dir, [], docling_required)
                except Exception as exc:
                    logger.warning("Alignment {}: could not write docling manifest: {}", aid, exc)

            write_paper_identifier_index(
                papers_dir, build_paper_identifier_index(manifest_map)
            )
            _emit_download_progress_summary(
                aid, paper_ids_src, manifest_map, papers_dir, "after_collect"
            )

            processed += 1
            if max_alignments is not None and processed >= max_alignments:
                logger.info("Reached max_alignments={}", max_alignments)
                return

    logger.info("Download-only complete: processed {} alignment(s)", processed)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Download paper packages only (no GPU pipeline)")
    p.add_argument("--paper-ids", required=True, help="Search output JSON path")
    p.add_argument("--data-root", required=True, help="Shared data root")
    p.add_argument("--output-root", default="", help="LLM results root (for skip-if-results)")
    p.add_argument("--idmap-csv", default=os.environ.get("IDMAP_CSV", ""))
    p.add_argument("--alignment-id", default="", help="Process one alignment only")
    p.add_argument("--max-alignments", type=int, default=None, help="Stop after N alignments")
    p.add_argument("--skip-if-results", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    run_download_packages_only(
        paper_ids_path=args.paper_ids,
        data_root=args.data_root,
        output_root=args.output_root,
        idmap_csv=args.idmap_csv,
        no_cache=args.no_cache,
        max_alignments=args.max_alignments,
        alignment_id=args.alignment_id,
        skip_if_results=args.skip_if_results,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
