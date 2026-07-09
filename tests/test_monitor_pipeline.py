"""Unit tests for pipelines/auto_lit/scripts/monitor_pipeline.py (pure helpers)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "auto_lit" / "scripts"))

import monitor_pipeline as mp  # noqa: E402


def test_bucket_from_scheduler_state():
    assert mp.bucket_from_scheduler_state("GRADER_READY", has_results=False) == "grader_queue"
    assert mp.bucket_from_scheduler_state("SYNTHESIS_INFLIGHT", has_results=False) == "synthesis"
    assert mp.bucket_from_scheduler_state("FAILED", has_results=False) == "failed"
    assert mp.bucket_from_scheduler_state("GRADER_INFLIGHT", has_results=True) == "done"


def test_classify_alignment_stages():
    now = time.time()
    rec = mp.classify_alignment(
        "A_B",
        scheduler_state=None,
        has_papers_dir=False,
        has_results=False,
        now=now,
    )
    assert rec.stage == "not_started"

    rec = mp.classify_alignment(
        "A_B",
        scheduler_state=None,
        has_papers_dir=True,
        has_results=False,
        now=now,
    )
    assert rec.stage == "collecting"

    rec = mp.classify_alignment(
        "A_B",
        scheduler_state={"state": "GRADER_INFLIGHT", "grader_submitted_at": now - 120},
        has_papers_dir=True,
        has_results=False,
        now=now,
    )
    assert rec.stage == "grading"
    assert rec.age_minutes is not None
    assert rec.age_minutes >= 1.9


def test_parse_grader_discovery_line():
    line = (
        "16:51:41 | INFO    | download_node: grader discovery 7/10 endpoint(s) active, "
        "1 grading inflight, 5 Slurm job(s) still pending"
    )
    parsed = mp.parse_grader_discovery_line(line)
    assert parsed == {
        "active": 7,
        "target": 10,
        "inflight": 1,
        "pending_jobs": 5,
    }


def test_count_recent_artifacts(tmp_path: Path):
    out = tmp_path / "llm_results"
    out.mkdir()
    graded = out / "pair_graded.json"
    graded.write_text("{}", encoding="utf-8")
    now = time.time()
    count = mp.count_recent_artifacts(
        out, "_graded.json", window_seconds=3600, now=now
    )
    assert count == 1


def test_rank_bottlenecks_docling():
    hints = mp.rank_bottlenecks(
        {},
        total=100,
        grader_queue=0,
        grader_inflight=0,
        grader_active=5,
        synthesis_queue=0,
        synthesis_gpus=2,
        docling_queue=12,
        docling_cap=1,
        collecting=0,
        not_started=0,
        done=1,
        results_per_hour=0.2,
    )
    assert any("Docling" in h for h in hints)


def test_build_snapshot_minimal(tmp_path: Path):
    paper_ids = tmp_path / "search.json"
    paper_ids.write_text(
        json.dumps(
            {
                "Q1": [{"target": "T1"}, {"target": "T2"}],
                "Q2": [{"target": "T3"}],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "llm_results"
    output_root.mkdir()
    (output_root / "Q1_T1_results.json").write_text("{}", encoding="utf-8")

    scheduler_dir = tmp_path / "logs" / "scheduler_state"
    scheduler_dir.mkdir(parents=True)
    (scheduler_dir / "Q1_T2.json").write_text(
        json.dumps({"state": "GRADER_READY", "alignment_id": "Q1_T2"}),
        encoding="utf-8",
    )

    snap = mp.build_snapshot(
        data_root=tmp_path,
        output_root=output_root,
        paper_ids_path=paper_ids,
        scheduler_dir=scheduler_dir,
        papers_root=tmp_path / "papers",
        logs_dir=tmp_path / "logs",
        cpu_log=None,
        window_minutes=60,
        stuck_minutes=90,
        docling_inflight_cap=1,
    )
    assert snap.total_alignments == 3
    assert snap.done == 1
    assert snap.stage_counts.get("grader_queue") == 1
    assert snap.stage_counts.get("not_started") == 1


def test_alignment_id_for_pair():
    assert mp.alignment_id_for_pair("Q5/A", "T 1") == "Q5_A_T_1"
