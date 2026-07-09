#!/usr/bin/env python3
"""Recompute docling_required_basenames from on-disk text/PDF artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(REPO_SRC))

from auto_lit_search.download_manifest import (  # noqa: E402
    DOWNLOAD_MANIFEST_FILENAME,
    _infer_docling_required_basenames,
    _load_download_manifest,
)
from auto_lit_search.env_config import auto_lit_data_root  # noqa: E402


def recompute_scheduler_docling(
    scheduler_dir: Path,
    papers_root: Path,
    *,
    dry_run: bool,
    revive_docling_pending: bool,
) -> tuple[int, int, int]:
    updated = 0
    to_grader_ready = 0
    skipped = 0
    for path in sorted(scheduler_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        if not isinstance(state, dict):
            skipped += 1
            continue
        aid = path.stem
        papers_dir = str(state.get("papers_dir") or papers_root / aid)
        if not Path(papers_dir).is_dir():
            skipped += 1
            continue
        manifest_path = Path(papers_dir) / DOWNLOAD_MANIFEST_FILENAME
        manifest_map = None
        if manifest_path.is_file():
            manifest_map = _load_download_manifest(str(manifest_path))
        old = state.get("docling_required_basenames") or []
        new = _infer_docling_required_basenames(str(papers_dir), manifest_map)
        old_n = len(old) if isinstance(old, list) else 0
        new_n = len(new)
        new_state = str(state.get("state") or "")
        if revive_docling_pending and new_state in {"DOCLING_PENDING", "DOCLING_INFLIGHT"}:
            has_txt = any(
                f.is_file() and f.stat().st_size > 0
                for f in Path(papers_dir).glob("*.txt")
            )
            if new_n == 0 and has_txt:
                new_state = "GRADER_READY"
                to_grader_ready += 1
        if old_n == new_n and new_state == str(state.get("state") or ""):
            if old == new:
                continue
        if dry_run:
            updated += 1
            continue
        state["docling_required_basenames"] = new
        if new_state != str(state.get("state") or ""):
            state["state"] = new_state
            state.pop("docling_job_id", None)
            state.pop("docling_submitted_at", None)
        path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        updated += 1
    return updated, to_grader_ready, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scheduler-state-dir",
        type=Path,
        default=None,
        help="Scheduler state directory (default: <AUTO_LIT_DATA_ROOT>/logs/scheduler_state).",
    )
    p.add_argument(
        "--papers-root",
        type=Path,
        default=None,
        help="Papers root (default: <AUTO_LIT_DATA_ROOT>/papers).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--revive-docling-pending",
        action="store_true",
        help="Move DOCLING_* alignments with zero PDFs left to GRADER_READY",
    )
    args = p.parse_args()
    data_root = auto_lit_data_root()
    scheduler_state_dir = args.scheduler_state_dir or (
        data_root / "logs" / "scheduler_state"
    )
    papers_root = args.papers_root or (data_root / "papers")
    updated, revived, skipped = recompute_scheduler_docling(
        scheduler_state_dir,
        papers_root,
        dry_run=args.dry_run,
        revive_docling_pending=args.revive_docling_pending,
    )
    mode = "would update" if args.dry_run else "updated"
    print(f"{mode} {updated} scheduler state file(s); revived {revived}; skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
