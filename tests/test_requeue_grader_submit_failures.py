"""Tests for pipelines/auto_lit/scripts/requeue_grader_submit_failures.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "auto_lit" / "scripts"))

from requeue_grader_submit_failures import requeue_grader_submit_failures  # noqa: E402


def test_requeue_grader_submit_failures(tmp_path: Path):
    sched = tmp_path / "scheduler_state"
    sched.mkdir()
    good = sched / "A_B.json"
    good.write_text(
        json.dumps(
            {
                "alignment_id": "A_B",
                "state": "FAILED",
                "last_error": "grader submit failed: Connection refused",
            }
        ),
        encoding="utf-8",
    )
    skip = sched / "C_D.json"
    skip.write_text(
        json.dumps({"alignment_id": "C_D", "state": "FAILED", "last_error": "docling failed"}),
        encoding="utf-8",
    )
    requeued, skipped, ids = requeue_grader_submit_failures(sched, dry_run=False, include_watchdog=False)
    assert requeued == 1
    assert skipped == 1
    assert ids == ["A_B"]
    data = json.loads(good.read_text(encoding="utf-8"))
    assert data["state"] == "GRADER_READY"
    assert "last_error" not in data
