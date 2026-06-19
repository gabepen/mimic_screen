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
    _infer_docling_required_basenames,
    _load_download_manifest,
    _load_search_json,
    _merge_recs_into_manifest,
    _write_docling_eval_manifest,
    _write_download_manifest_atomic,
    build_paper_identifier_index,
    classify_alignment_papers,
    is_alignment_download_complete,
    load_global_outcome_cache,
    record_global_outcomes_from_rows,
    write_alignment_download_complete,
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


def _retry_failed_enabled(retry_failed: bool) -> bool:
    if retry_failed:
        return True
    return os.environ.get("DOWNLOAD_RETRY_FAILED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


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
    retry_failed: bool = False,
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
    retry_failed = _retry_failed_enabled(retry_failed)

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
    global_cache = load_global_outcome_cache(logs_base)
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

            if (
                not no_cache
                and not retry_failed
                and is_alignment_download_complete(papers_dir)
            ):
                logger.info("Alignment {}: download-complete marker present; skipping", aid)
                write_paper_identifier_index(
                    papers_dir, build_paper_identifier_index(manifest_map)
                )
                processed += 1
                if max_alignments is not None and processed >= max_alignments:
                    logger.info("Reached max_alignments={}", max_alignments)
                    return
                continue

            plan = classify_alignment_papers(
                paper_ids_src,
                manifest_map,
                papers_dir,
                global_cache,
                retry_failed=retry_failed,
                no_cache=no_cache,
            )
            if plan.global_inject:
                manifest_map = {**manifest_map, **plan.global_inject}
                _write_download_manifest_atomic(manifest_path, manifest_map)

            logger.info(
                "Alignment {}: collect plan satisfied={} terminal_failed={} "
                "global_skipped={} to_fetch={}",
                aid,
                plan.stats.get("satisfied", 0),
                plan.stats.get("terminal_failed", 0),
                plan.stats.get("global_skipped", 0),
                plan.stats.get("to_fetch", 0),
            )
            _emit_download_progress_summary(
                aid,
                paper_ids_src,
                manifest_map,
                papers_dir,
                "before_collect",
                classification_stats=plan.stats,
            )

            recs: List[Any] = []
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
                    force_pdfs=False,
                    prefer_pdf_text=True,
                    collection_org=collection_org,
                    auth_scope=collection_auth_scope,
                    collector_email=collector_email or None,
                    max_workers=collect_max_workers,
                    disable_semantic_scholar=collect_disable_s2,
                )
                manifest_map = _merge_recs_into_manifest({}, recs)
                _write_download_manifest_atomic(manifest_path, manifest_map)
            elif plan.to_fetch:
                recs = download_papers_to_dir(
                    plan.to_fetch,
                    papers_dir,
                    session=session,
                    pmcid_cache=pmcid_cache,
                    no_cache=no_cache,
                    force_pdfs=False,
                    prefer_pdf_text=True,
                    collection_org=collection_org,
                    auth_scope=collection_auth_scope,
                    collector_email=collector_email or None,
                    max_workers=collect_max_workers,
                    disable_semantic_scholar=collect_disable_s2,
                )
                manifest_map = _merge_recs_into_manifest(manifest_map, recs)
                _write_download_manifest_atomic(manifest_path, manifest_map)
                record_global_outcomes_from_rows(logs_base, manifest_map.values(), global_cache)
            else:
                logger.info(
                    "Alignment {}: nothing to fetch (satisfied={} terminal_failed={} global_skipped={})",
                    aid,
                    plan.stats.get("satisfied", 0),
                    plan.stats.get("terminal_failed", 0),
                    plan.stats.get("global_skipped", 0),
                )

            if not no_cache:
                write_alignment_download_complete(
                    papers_dir,
                    {
                        "alignment_id": aid,
                        "total_expected": len(paper_ids_src),
                        **plan.stats,
                        "fetched": len(recs),
                        "retry_failed_mode": retry_failed,
                    },
                )

            docling_required = _infer_docling_required_basenames(
                papers_dir, manifest_map
            )
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
                aid,
                paper_ids_src,
                manifest_map,
                papers_dir,
                "after_collect",
                classification_stats=plan.stats,
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
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run collect on manifest failed/partial-without-pdf rows",
    )
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
        retry_failed=args.retry_failed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
