#!/usr/bin/env python3
"""Incrementally sync stage2 artifacts after a stage1 DOI-set change.

For each alignment in a stage1_doi_delta_*.jsonl (or action list + old/new search):

1. Delete on-disk papers for removed DOIs (query/target sides).
2. Prune matching rows from ``{aid}_graded.json``.
3. Delete synthesis/results/analysis/rubric sidecars.
4. Invalidate download_complete + scheduler DONE state.
5. Keep kept papers and kept graded rows in place.

Does **not** download or grade. After sync, rerun stage2 (or download node) with
``GRADER_REUSE_EXISTING=1`` so only new paper files are graded, then resynthesize.

Example:
  python pipelines/auto_lit/scripts/sync_stage2_paper_sets.py \\
    --data-root /path/to/auto_lit_eval_data \\
    --llm-subdir wol-dros-v1 \\
    --doi-delta search_results/stage1_doi_delta_wol-dros-v1.jsonl \\
    --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _read_doi_delta(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _doi_stem(doi: str) -> str:
    return doi.replace("/", "_")


def _paper_paths_for_dois(
    papers_dir: Path, dois: Iterable[str], role: str
) -> List[Path]:
    """Match ``{doi_stem}__{role}*.txt`` / pdf under papers_dir."""
    out: List[Path] = []
    if not papers_dir.is_dir():
        return out
    pdf_dir = papers_dir / "pdf"
    for doi in dois:
        stem = _doi_stem(doi)
        prefix = f"{stem}__{role}"
        for p in papers_dir.glob(f"{prefix}*.txt"):
            out.append(p)
        if pdf_dir.is_dir():
            for p in pdf_dir.glob(f"{prefix}*.pdf"):
                out.append(p)
    return out


def _prune_graded(graded_path: Path, removed_dois: Set[str], dry_run: bool) -> int:
    if not graded_path.is_file():
        return 0
    data = json.loads(graded_path.read_text(encoding="utf-8"))
    papers = list(data.get("graded_papers") or [])
    removed_stems = {_doi_stem(d) for d in removed_dois}
    kept = []
    n_drop = 0
    for gp in papers:
        fname = str(gp.get("file_name") or "")
        pid = str(gp.get("paper_id") or "")
        drop = False
        if pid in removed_dois or _doi_stem(pid) in removed_stems:
            drop = True
        else:
            for stem in removed_stems:
                if fname.startswith(stem + "__") or fname.startswith(stem):
                    drop = True
                    break
        if drop:
            n_drop += 1
        else:
            kept.append(gp)
    if n_drop and not dry_run:
        data["graded_papers"] = kept
        meta = data.get("grading_meta") or {}
        if isinstance(meta, dict):
            meta["n_papers"] = len(kept)
            meta["pruned_removed_papers"] = n_drop
            data["grading_meta"] = meta
        graded_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return n_drop


def _delete_paths(paths: Iterable[Path], dry_run: bool) -> int:
    n = 0
    for p in paths:
        if not p.exists():
            continue
        print(f"{'DRY ' if dry_run else ''}DELETE {p}")
        if not dry_run:
            p.unlink()
        n += 1
    return n


def _clear_scheduler(paths: Iterable[Path], dry_run: bool) -> int:
    n = 0
    for p in paths:
        if not p.is_file():
            continue
        print(f"{'DRY ' if dry_run else ''}CLEAR_SCHED {p}")
        if not dry_run:
            p.unlink()
        n += 1
    return n


def sync_alignment(
    *,
    data_root: Path,
    llm_subdir: str,
    row: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, int]:
    aid = row["alignment_id"]
    action = row.get("action") or ""
    if action == "noop":
        return {"skipped_noop": 1}

    removed_q = set(row.get("query_removed_dois") or [])
    removed_t = set(row.get("target_removed_dois") or [])
    removed_all = removed_q | removed_t

    papers_dir = data_root / "papers" / aid
    if llm_subdir in (".", "", "flat"):
        llm_dir = data_root / "llm_results"
    else:
        llm_dir = data_root / "llm_results" / llm_subdir
    # Newer layout + legacy scheduler paths
    sched_paths = [
        llm_dir / "scheduler_state" / f"{aid}.json",
        data_root / "logs" / "scheduler_state" / f"{aid}.json",
    ]

    stats = {
        "papers_deleted": 0,
        "grades_pruned": 0,
        "sidecars_deleted": 0,
        "scheduler_cleared": 0,
    }

    del_paths: List[Path] = []
    del_paths.extend(_paper_paths_for_dois(papers_dir, removed_q, "query"))
    del_paths.extend(_paper_paths_for_dois(papers_dir, removed_t, "target"))
    stats["papers_deleted"] = _delete_paths(del_paths, dry_run)

    graded_path = llm_dir / f"{aid}_graded.json"
    n_prune = _prune_graded(graded_path, removed_all, dry_run)
    stats["grades_pruned"] = n_prune
    if n_prune:
        print(f"{'DRY ' if dry_run else ''}PRUNE_GRADED {graded_path} dropped={n_prune}")

    sidecars = [
        llm_dir / f"{aid}_results.json",
        llm_dir / f"{aid}_analysis.json",
        llm_dir / f"{aid}_host_rubric_scores.csv",
        llm_dir / f"{aid}_microbe_rubric_scores.csv",
        papers_dir / "download_complete.json",
    ]
    stats["sidecars_deleted"] = _delete_paths(sidecars, dry_run)
    stats["scheduler_cleared"] = _clear_scheduler(sched_paths, dry_run)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--llm-subdir", required=True, help="e.g. wol-dros-v1 or '.' for flat llm_results")
    ap.add_argument("--doi-delta", required=True, type=Path)
    ap.add_argument(
        "--actions",
        default="prune_and_resynth,add_grade_resynth",
        help="Comma-separated actions to sync (default: both changing actions)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    want = {a.strip() for a in args.actions.split(",") if a.strip()}
    rows = _read_doi_delta(args.doi_delta)
    totals = {
        "alignments": 0,
        "papers_deleted": 0,
        "grades_pruned": 0,
        "sidecars_deleted": 0,
        "scheduler_cleared": 0,
        "skipped_noop": 0,
    }
    llm_subdir = args.llm_subdir

    for row in rows:
        action = row.get("action") or "noop"
        if action not in want:
            if action == "noop":
                totals["skipped_noop"] += 1
            continue
        totals["alignments"] += 1
        stats = sync_alignment(
            data_root=args.data_root,
            llm_subdir=llm_subdir,
            row=row,
            dry_run=args.dry_run,
        )
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v

    print(
        f"sync done dry_run={args.dry_run} alignments={totals['alignments']} "
        f"papers_deleted={totals['papers_deleted']} grades_pruned={totals['grades_pruned']} "
        f"sidecars_deleted={totals['sidecars_deleted']} "
        f"scheduler_cleared={totals['scheduler_cleared']} noop_skipped={totals['skipped_noop']}"
    )
    print(
        "Next: rerun stage2 download/grade/synthesis with GRADER_REUSE_EXISTING=1 "
        "for actioned alignments."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
