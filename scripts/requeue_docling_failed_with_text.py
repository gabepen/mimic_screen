#!/usr/bin/env python3
"""Re-queue FAILED alignments that have usable paper text despite docling errors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_state(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _papers_dir_has_nonempty_txt(papers_dir: Path) -> bool:
    if not papers_dir.is_dir():
        return False
    for fname in papers_dir.iterdir():
        if not fname.name.endswith(".txt") or not fname.is_file():
            continue
        if fname.stat().st_size > 0:
            return True
    return False


def requeue_docling_failed_with_text(
    scheduler_dir: Path,
    papers_root: Path,
    *,
    dry_run: bool,
    min_txt_files: int = 1,
) -> Tuple[int, int, int, List[str]]:
    if not scheduler_dir.is_dir():
        raise FileNotFoundError(f"Missing scheduler state dir: {scheduler_dir}")

    requeued = 0
    skipped = 0
    no_text = 0
    ids: List[str] = []
    needles = ("no pdfs converted", "docling")

    for path in sorted(scheduler_dir.glob("*.json")):
        state = _load_state(path)
        if state is None:
            skipped += 1
            continue
        if str(state.get("state") or "") != "FAILED":
            skipped += 1
            continue
        err = str(state.get("last_error") or "").lower()
        if not any(n in err for n in needles):
            skipped += 1
            continue

        aid = path.stem
        papers_dir = papers_root / aid
        if not papers_dir.is_dir():
            papers_dir = Path(str(state.get("papers_dir") or ""))
        n_txt = sum(
            1
            for f in papers_dir.glob("*.txt")
            if f.is_file() and f.stat().st_size > 0
        )
        if n_txt < min_txt_files and not _papers_dir_has_nonempty_txt(papers_dir):
            no_text += 1
            skipped += 1
            continue

        ids.append(aid)
        if dry_run:
            requeued += 1
            continue

        state["state"] = "GRADER_READY"
        state.pop("last_error", None)
        state.pop("docling_job_id", None)
        state.pop("docling_submitted_at", None)
        state["docling_submit_attempt"] = 0
        state["updated_at"] = time.time()
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        requeued += 1

    return requeued, skipped, no_text, ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scheduler-state-dir",
        type=Path,
        default=Path(
            "/private/groups/corbettlab/gabe/auto_lit_eval_data/logs/scheduler_state"
        ),
    )
    p.add_argument(
        "--papers-root",
        type=Path,
        default=Path("/private/groups/corbettlab/gabe/auto_lit_eval_data/papers"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--min-txt-files",
        type=int,
        default=1,
        help="Minimum non-empty .txt files required in papers_dir",
    )
    p.add_argument("--list", action="store_true", help="Print requeued alignment IDs")
    args = p.parse_args()

    requeued, skipped, no_text, ids = requeue_docling_failed_with_text(
        args.scheduler_state_dir,
        args.papers_root,
        dry_run=args.dry_run,
        min_txt_files=max(1, args.min_txt_files),
    )
    mode = "would requeue" if args.dry_run else "requeued"
    print(f"{mode} {requeued} alignment(s); skipped {skipped} (no usable text: {no_text})")
    if args.list and ids:
        for aid in ids:
            print(aid)
    if requeued and not args.dry_run:
        print(
            "Disk state updated. Running CPU job will pick up GRADER_READY on the "
            "next scheduler tick (~5s) if it has the sync/reconcile patch; otherwise "
            "restart the CPU download job.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
