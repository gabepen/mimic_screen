#!/usr/bin/env python3
"""Re-queue alignments that failed on transient grader HTTP submit errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_state(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def requeue_grader_submit_failures(
    scheduler_dir: Path,
    *,
    dry_run: bool,
    include_watchdog: bool,
) -> Tuple[int, int, List[str]]:
    if not scheduler_dir.is_dir():
        raise FileNotFoundError(f"Missing scheduler state dir: {scheduler_dir}")

    requeued = 0
    skipped = 0
    ids: List[str] = []
    needles = ("grader submit failed", "Connection refused", "Max retries exceeded")
    if include_watchdog:
        needles = needles + ("grader watchdog timeout",)

    for path in sorted(scheduler_dir.glob("*.json")):
        state = _load_state(path)
        if state is None:
            skipped += 1
            continue
        if str(state.get("state") or "") != "FAILED":
            skipped += 1
            continue
        err = str(state.get("last_error") or "")
        if not any(n in err for n in needles):
            skipped += 1
            continue
        aid = path.stem
        ids.append(aid)
        if dry_run:
            requeued += 1
            continue
        state["state"] = "GRADER_READY"
        state.pop("last_error", None)
        state.pop("grader_job_id", None)
        state.pop("grader_url_base", None)
        state.pop("grader_submitted_at", None)
        state["grader_submit_attempts"] = 0
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
        requeued += 1

    return requeued, skipped, ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scheduler-state-dir",
        type=Path,
        default=Path(
            "/private/groups/corbettlab/gabe/auto_lit_eval_data/logs/scheduler_state"
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--include-watchdog",
        action="store_true",
        help="Also requeue FAILED alignments with grader watchdog timeout",
    )
    p.add_argument("--list", action="store_true", help="Print requeued alignment IDs")
    args = p.parse_args()

    requeued, skipped, ids = requeue_grader_submit_failures(
        args.scheduler_state_dir,
        dry_run=args.dry_run,
        include_watchdog=args.include_watchdog,
    )
    mode = "would requeue" if args.dry_run else "requeued"
    print(f"{mode} {requeued} alignment(s); skipped {skipped}")
    if args.list and ids:
        for aid in ids:
            print(aid)
    if requeued and not args.dry_run:
        print(
            "Restart the CPU download job so in-memory scheduler state reloads these alignments.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
