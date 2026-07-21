#!/usr/bin/env python3
"""Quarantine old stage2 artifacts for alignments listed for reanalysis.

Moves papers/, llm_results artifacts, and scheduler state aside so stage2 will
re-download/grade/synthesize those pairs.

Example:
  python pipelines/auto_lit/scripts/prepare_reanalysis_queue.py \\
    --data-root /private/groups/corbettlab/gabe/auto_lit_eval_data \\
    --llm-subdir wol-dros-v1 \\
    --alignments-file search_results/reanalysis_alignment_ids_wol-dros-v1.txt \\
    --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _read_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--llm-subdir", required=True, help="e.g. wol-dros-v1 or lp-human-all")
    ap.add_argument("--alignments-file", required=True, type=Path)
    ap.add_argument("--stamp", default=None, help="quarantine stamp (default: UTC now)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ids = _read_ids(args.alignments_file)
    papers = args.data_root / "papers"
    llm = args.data_root / "llm_results" / args.llm_subdir
    sched = args.data_root / "logs" / "scheduler_state"
    quar_root = args.data_root / "reanalysis_quarantine" / f"{args.llm_subdir}_{stamp}"

    moved = 0
    missing = 0
    for aid in ids:
        targets = [
            papers / aid,
            sched / f"{aid}.json",
        ]
        for suffix in (
            "_graded.json",
            "_results.json",
            "_analysis.json",
            "_host_rubric_scores.csv",
            "_microbe_rubric_scores.csv",
        ):
            targets.append(llm / f"{aid}{suffix}")
        # also move known backups next to results
        if llm.is_dir():
            for p in llm.glob(f"{aid}_*"):
                if p not in targets:
                    targets.append(p)

        for src in targets:
            if not src.exists():
                missing += 1
                continue
            rel = src.relative_to(args.data_root)
            dst = quar_root / rel
            print(f"{'DRY ' if args.dry_run else ''}MOVE {src} -> {dst}")
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            moved += 1

    print(f"alignments={len(ids)} moves={moved} missing_paths={missing} quarantine={quar_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
