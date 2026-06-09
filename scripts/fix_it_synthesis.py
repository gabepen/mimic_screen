#!/usr/bin/env python3
"""Re-run synthesis from existing *_graded.json without regrading."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List

from auto_lit_search.graded_request import build_run_alignment_graded_request
from auto_lit_search.scheduler_http import post_run_alignment_graded, synthesis_http_timeout_sec


def _needs_fix(results_path: Path) -> bool:
    if not results_path.is_file():
        return True
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    conclusion = data.get("conclusion") or {}
    if not isinstance(conclusion, dict):
        return True
    if conclusion.get("scorecard_version") != "2":
        return True
    if conclusion.get("synthesis_status") in ("error", "grades_only"):
        return True
    synth = data.get("synthesis") or {}
    notes = str(synth.get("notes") or "")
    if "fallback" in notes.lower() or "timeout" in notes.lower():
        return True
    return False


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
        required=True,
        help="GPU synthesis base URL, e.g. http://host:9000",
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

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = args.output_root / f"fix_it_synthesis_report_{ts}.tsv"
    rows: List[str] = ["alignment_id\tstatus\tnotes"]

    for alignment_id in alignment_ids:
        graded_path = args.output_root / f"{alignment_id}_graded.json"
        if not graded_path.is_file():
            rows.append(f"{alignment_id}\tskipped\tmissing graded.json")
            continue
        results_path = args.output_root / f"{alignment_id}_results.json"
        if args.skip_missing_results and not results_path.is_file():
            rows.append(f"{alignment_id}\tskipped\tno results.json (CPU queue)")
            continue
        if args.only_needs_fix:
            if not _needs_fix(results_path):
                rows.append(f"{alignment_id}\tskipped\tresults look ok")
                continue
        try:
            req = build_run_alignment_graded_request(
                alignment_id,
                args.output_root,
                papers_root=args.papers_root,
                instructions_file=args.instructions_file,
            )
        except Exception as e:
            rows.append(f"{alignment_id}\terror\tbuild request: {e}")
            continue
        if args.dry_run:
            rows.append(f"{alignment_id}\tdry_run\tn_papers={len(req.graded_papers)}")
            continue
        if args.backup:
            for suffix in ("_results.json", "_analysis.json"):
                src = args.output_root / f"{alignment_id}{suffix}"
                if src.is_file():
                    dst = src.with_name(f"{src.name}.bak.{ts}")
                    shutil.copy2(src, dst)
        try:
            out = post_run_alignment_graded(
                args.synthesis_url,
                req.dict(),
                timeout=synthesis_http_timeout_sec(),
            )
            rows.append(
                f"{alignment_id}\tok\t{out.get('results_path', '')}"
            )
        except Exception as e:
            rows.append(f"{alignment_id}\terror\t{e}")

    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote report: {report_path}")
    failed = sum(1 for r in rows[1:] if "\terror\t" in r)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
