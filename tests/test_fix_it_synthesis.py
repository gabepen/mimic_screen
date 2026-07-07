"""Tests for fix_it_synthesis result validation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.fix_it_synthesis import _needs_fix, _verify_synthesis_results


def _valid_synthesis_text() -> str:
    return (
        "Discussion.\n\nQuick results summary:\n```json\n"
        '{"headline": "Test headline.", "host_exploitation_score": 40, '
        '"query_effector_score": 60, "mimicry_plausibility_score": 50, '
        '"pair_priority_score": 45, "best_host_paper": "10.1/x", '
        '"best_query_paper": "", "main_uncertainties": "None."}\n```\n'
    )


def _write_results(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_needs_fix_on_connection_refused(tmp_path: Path) -> None:
    p = tmp_path / "A_B_results.json"
    _write_results(
        p,
        {
            "conclusion": {"scorecard_version": "2", "synthesis_status": "grades_only"},
            "synthesis": {"notes": "HTTPConnectionPool connection refused"},
        },
    )
    assert _needs_fix(p)


def test_verify_synthesis_results_ok(tmp_path: Path) -> None:
    p = tmp_path / "A_B_results.json"
    _write_results(
        p,
        {
            "conclusion": {"scorecard_version": "2", "synthesis_status": "ok"},
            "synthesis": {"text": _valid_synthesis_text(), "notes": ""},
        },
    )
    ok, detail = _verify_synthesis_results(p)
    assert ok
    assert "A_B_results.json" in detail


def test_verify_synthesis_results_rejects_grades_only(tmp_path: Path) -> None:
    p = tmp_path / "A_B_results.json"
    _write_results(
        p,
        {
            "conclusion": {"scorecard_version": "2", "synthesis_status": "grades_only"},
            "synthesis": {"notes": "empty synthesis output", "text": "fallback"},
        },
    )
    ok, detail = _verify_synthesis_results(p)
    assert not ok
    assert "grades_only" in detail
    assert "reason=" in detail
