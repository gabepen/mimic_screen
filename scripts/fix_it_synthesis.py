#!/usr/bin/env python3
"""Re-run synthesis from existing *_graded.json without regrading."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from auto_lit_search.graded_request_payload import build_run_alignment_graded_payload
from auto_lit_search.scheduler_http import post_run_alignment_graded, synthesis_http_timeout_sec
from auto_lit_search.synthesis_validation import (
    needs_synthesis_fix,
    verify_synthesis_results,
)


def _needs_fix(results_path: Path) -> bool:
    return needs_synthesis_fix(results_path)


def _verify_synthesis_results(results_path: Path) -> Tuple[bool, str]:
    return verify_synthesis_results(results_path)


def _discover_alignments(
    output_root: Path,
    only_needs_fix: bool,
    skip_missing_results: bool,
) -> List[str]:
    ids: List[str] = []
    for graded in sorted(output_root.glob("*_graded.json")):
        alignment_id = graded.name[: -len("_graded.json")]
        results = output_root / f"{alignment_id}_results.json"
        if skip_missing_results and not results.is_file():
            continue
        if only_needs_fix and not _needs_fix(results):
            continue
        ids.append(alignment_id)
    return ids


def _process_alignment(
    *,
    idx: int,
    alignment_id: str,
    output_root: Path,
    synthesis_urls: List[str],
    papers_root: Path | None,
    instructions_file: Path | None,
    only_needs_fix: bool,
    skip_missing_results: bool,
    backup: bool,
    dry_run: bool,
    ts: str,
) -> Tuple[str, str, str]:
    synthesis_url = synthesis_urls[idx % len(synthesis_urls)]
    graded_path = output_root / f"{alignment_id}_graded.json"
    if not graded_path.is_file():
        return alignment_id, "skipped", "missing graded.json"
    results_path = output_root / f"{alignment_id}_results.json"
    if skip_missing_results and not results_path.is_file():
        return alignment_id, "skipped", "no results.json (CPU queue)"
    if only_needs_fix and not _needs_fix(results_path):
        return alignment_id, "skipped", "results look ok"
    try:
        payload = build_run_alignment_graded_payload(
            alignment_id,
            output_root,
            papers_root=papers_root,
            instructions_file=instructions_file,
        )
    except Exception as e:
        return alignment_id, "error", f"build request: {e}"
    if dry_run:
        return (
            alignment_id,
            "dry_run",
            f"n_papers={len(payload.get('graded_papers') or [])} url={synthesis_url}",
        )
    if backup:
        for suffix in ("_results.json", "_analysis.json"):
            src = output_root / f"{alignment_id}{suffix}"
            if src.is_file():
                dst = src.with_name(f"{src.name}.bak.{ts}")
                shutil.copy2(src, dst)
    try:
        out = post_run_alignment_graded(
            synthesis_url,
            payload,
            timeout=synthesis_http_timeout_sec(),
        )
        results_path = Path(str(out.get("results_path") or ""))
        if not results_path.is_file():
            results_path = output_root / f"{alignment_id}_results.json"
        ok, detail = _verify_synthesis_results(results_path)
        if not ok:
            return alignment_id, "error", detail
        return alignment_id, "ok", detail
    except Exception as e:
        return alignment_id, "error", str(e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="llm_results directory with *_graded.json",
    )
    p.add_argument(
        "--synthesis-url",
        default="",
        help="GPU synthesis base URL, e.g. http://host:9000 (alias for single --synthesis-urls)",
    )
    p.add_argument(
        "--synthesis-urls",
        default="",
        help="Semicolon-separated synthesis GPU URLs (round-robin, one in-flight per URL)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Max concurrent synthesis requests (default: number of --synthesis-urls)",
    )
    p.add_argument("--papers-root", type=Path, default=None)
    p.add_argument("--instructions-file", type=Path, default=None)
    p.add_argument("--alignments", nargs="*", default=())
    p.add_argument(
        "--alignments-file",
        type=Path,
        default=None,
        help="One alignment_id per line",
    )
    p.add_argument(
        "--only-needs-fix",
        action="store_true",
        help="Skip alignments with scorecard v2 and synthesis_status ok",
    )
    p.add_argument(
        "--skip-missing-results",
        action="store_true",
        help=(
            "Skip graded-only alignments with no *_results.json; leave those for "
            "CPU scheduler SYNTHESIS_READY while this job resynthesizes the v1 backlog"
        ),
    )
    p.add_argument("--backup", action="store_true", help="Backup results/analysis before POST")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    alignment_ids: List[str] = list(args.alignments)
    if args.alignments_file and args.alignments_file.is_file():
        alignment_ids.extend(
            ln.strip()
            for ln in args.alignments_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
    if not alignment_ids:
        alignment_ids = _discover_alignments(
            args.output_root,
            args.only_needs_fix,
            args.skip_missing_results,
        )
    if not alignment_ids:
        print("No alignments to process.", file=sys.stderr)
        return 1

    raw_urls = (args.synthesis_urls or args.synthesis_url or "").strip()
    synthesis_urls = [u.strip() for u in raw_urls.replace(",", ";").split(";") if u.strip()]
    if not synthesis_urls:
        print("Pass --synthesis-url or --synthesis-urls.", file=sys.stderr)
        return 2

    workers = args.workers if args.workers > 0 else len(synthesis_urls)
    workers = max(1, min(workers, len(synthesis_urls)))

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.output_root / f"fix_it_synthesis_report_{ts}.tsv"
    rows: List[str] = ["alignment_id\tstatus\tnotes"]
    rows_lock = threading.Lock()
    done_count = 0
    total = len(alignment_ids)

    print(
        f"fix_it_synthesis: {total} alignments, {workers} workers, "
        f"{len(synthesis_urls)} GPU(s): {'; '.join(synthesis_urls)}",
        file=sys.stderr,
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_alignment,
                idx=idx,
                alignment_id=alignment_id,
                output_root=args.output_root,
                synthesis_urls=synthesis_urls,
                papers_root=args.papers_root,
                instructions_file=args.instructions_file,
                only_needs_fix=args.only_needs_fix,
                skip_missing_results=args.skip_missing_results,
                backup=args.backup,
                dry_run=args.dry_run,
                ts=ts,
            ): alignment_id
            for idx, alignment_id in enumerate(alignment_ids)
        }
        for fut in as_completed(futures):
            alignment_id, status, notes = fut.result()
            with rows_lock:
                rows.append(f"{alignment_id}\t{status}\t{notes}")
                done_count += 1
                print(
                    f"[{done_count}/{total}] {alignment_id}\t{status}\t{notes[:120]}",
                    file=sys.stderr,
                    flush=True,
                )

    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")
    failed = sum(1 for r in rows[1:] if "\terror\t" in r)
    ok_count = sum(1 for r in rows[1:] if "\tok\t" in r)
    print(f"Synthesis ok: {ok_count}, failed: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
