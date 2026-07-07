"""Tests for Quick results JSON parsing and validation."""

from __future__ import annotations

from auto_lit_search.synthesis_scorecard import (
    _extract_json_block,
    merge_synthesis_repair,
    parse_llm_scorecard,
    synthesis_output_diagnosis,
    synthesis_output_well_formed,
)
from auto_lit_search.synthesis_validation import classify_synthesis_failure, needs_synthesis_fix


def _valid_synthesis_text() -> str:
    return (
        "Discussion of host and query evidence.\n\n"
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Moderate host pathway overlap with query effector support.",\n'
        '  "host_exploitation_score": 40,\n'
        '  "query_effector_score": 60,\n'
        '  "mimicry_plausibility_score": 50,\n'
        '  "pair_priority_score": 45,\n'
        '  "best_host_paper": "10.1000/example",\n'
        '  "best_query_paper": "",\n'
        '  "main_uncertainties": "Limited direct infection context."\n'
        "}\n"
        "```\n"
    )


def test_parse_llm_scorecard_fenced_json() -> None:
    parsed = parse_llm_scorecard(_valid_synthesis_text())
    assert parsed["host_exploitation_score"] == 40
    assert parsed["query_effector_score"] == 60
    assert parsed["headline"]


def test_parse_llm_scorecard_balanced_json_with_trailing_prose() -> None:
    text = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Nested braces test ok.",\n'
        '  "host_exploitation_score": 10,\n'
        '  "query_effector_score": 20,\n'
        '  "mimicry_plausibility_score": 30,\n'
        '  "pair_priority_score": 25,\n'
        '  "best_host_paper": "10.1/x",\n'
        '  "best_query_paper": "",\n'
        '  "main_uncertainties": "Sparse evidence."\n'
        "}\n"
        "```\n"
        "Extra trailing commentary that should not break parsing.\n"
    )
    obj = _extract_json_block(text)
    assert obj is not None
    assert obj["headline"] == "Nested braces test ok."
    assert synthesis_output_well_formed(text)


def test_classify_parse_error() -> None:
    data = {
        "conclusion": {"synthesis_status": "grades_only", "scorecard_version": "2"},
        "synthesis": {
            "notes": "synthesis missing parseable Quick results JSON after retry",
            "text": "LLM discussion without a valid quick results block.",
        },
    }
    assert classify_synthesis_failure(data) == "parse_error"


def test_needs_fix_rejects_ok_status_with_bad_text(tmp_path) -> None:
    import json

    p = tmp_path / "A_B_results.json"
    p.write_text(
        json.dumps(
            {
                "conclusion": {"scorecard_version": "2", "synthesis_status": "ok"},
                "synthesis": {"text": "No quick results block.", "notes": ""},
            }
        ),
        encoding="utf-8",
    )
    assert needs_synthesis_fix(p)


def test_synthesis_output_diagnosis_missing_header() -> None:
    assert synthesis_output_diagnosis("plain discussion only") == "missing_quick_results_header"


def test_merge_synthesis_repair_attaches_json_footer() -> None:
    discussion = "Host and query evidence discussion without footer."
    repair = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Moderate overlap.",\n'
        '  "host_exploitation_score": 40,\n'
        '  "query_effector_score": 60,\n'
        '  "mimicry_plausibility_score": 50,\n'
        '  "pair_priority_score": 45,\n'
        '  "best_host_paper": "10.1000/example",\n'
        '  "best_query_paper": "",\n'
        '  "main_uncertainties": "Limited evidence."\n'
        "}\n"
        "```\n"
    )
    merged = merge_synthesis_repair(discussion, repair)
    assert merged is not None
    assert merged.startswith(discussion)
    assert synthesis_output_well_formed(merged)
