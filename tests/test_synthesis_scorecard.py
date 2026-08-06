"""Tests for Quick results JSON parsing and validation."""

from __future__ import annotations

from types import SimpleNamespace

from auto_lit_search.synthesis_scorecard import (
    _extract_json_block,
    build_conclusion,
    compute_rubric_scorecard,
    merge_synthesis_repair,
    mimicry_flag_handling_block,
    parse_llm_scorecard,
    quick_summary_prompt_footer,
    score_to_dimension_tier,
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


def test_quick_summary_footer_asks_for_mimicry_score() -> None:
    footer = quick_summary_prompt_footer()
    assert "mimicry_plausibility_score" in footer
    assert "Score mimicry_plausibility from the graded paper evidence" in footer


def test_mimicry_flag_handling_block_from_host_rubric() -> None:
    block = mimicry_flag_handling_block(
        {
            "synthesis_instructions": {
                "mimicry_flag_handling": "Surface mimicry_strong flags explicitly."
            }
        }
    )
    assert "Surface mimicry_strong flags explicitly." in block
    assert mimicry_flag_handling_block(None) == ""
    assert mimicry_flag_handling_block({}) == ""


def _gp(**kwargs):
    defaults = dict(
        paper_id="p1",
        file_name="p1.txt",
        paper_role="query",
        relevance_grade=0.8,
        rubric_dimension_scores={"evidence_quality": 0.9, "system_relevance": 0.8},
        rubric_axis_rationales={},
        rationale="",
        claim_summary="",
        criterion_scores={},
        rubric_tags={},
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_conclusion_uses_synthesis_scores_not_rubric_blend() -> None:
    """Top-level scores must match synthesis Quick results, not grade aggregates."""
    papers = [
        _gp(
            paper_role="query",
            claim_summary="AnkB mimics eukaryotic F-box proteins to hijack SCF.",
            rationale="Dot/Icm F-box effector.",
        ),
        _gp(
            paper_id="h1",
            file_name="h1.txt",
            paper_role="target",
            relevance_grade=0.9,
            rubric_dimension_scores={
                "infection_process_relevance": 0.9,
                "protein_characterisation_quality": 0.9,
            },
            claim_summary="Host actin regulator.",
            # Host mimicry tag would have inflated the old rubric index to 85.
            rubric_tags={"mimicry_potential_flag": "mimicry_strong"},
        ),
    ]
    synth = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Query effector literature; Foldseek host not co-mentioned.",\n'
        '  "host_exploitation_score": 40,\n'
        '  "query_effector_score": 90,\n'
        '  "mimicry_plausibility_score": 20,\n'
        '  "pair_priority_score": 50,\n'
        '  "best_host_paper": "h1",\n'
        '  "best_query_paper": "p1",\n'
        '  "main_uncertainties": "No co-mention of Foldseek host."\n'
        "}\n"
        "```\n"
    )
    conc = build_conclusion(papers, synth, synthesis_status="ok")
    assert conc["mimicry_plausibility"]["score"] == 20
    assert conc["mimicry_plausibility"]["source"] == "synthesis"
    assert conc["query_effector"]["score"] == 90
    assert conc["host_exploitation"]["score"] == 40
    assert conc["pair_priority"]["score"] == 50
    assert conc["synthesis_scores"]["mimicry_plausibility_score"] == 20
    # Rubric diagnostics still available and may disagree.
    assert conc["rubric_indices"]["mimicry_plausibility"]["score"] == 85


def test_build_conclusion_falls_back_to_rubric_when_synthesis_missing() -> None:
    papers = [
        _gp(
            paper_id="h1",
            file_name="h1.txt",
            paper_role="target",
            relevance_grade=0.5,
            rubric_dimension_scores={
                "infection_process_relevance": 0.6,
                "protein_characterisation_quality": 0.7,
            },
            rubric_tags={"mimicry_potential_flag": "none"},
        ),
    ]
    conc = build_conclusion(papers, "", synthesis_status="grades_only")
    assert conc["mimicry_plausibility"]["source"] == "rubric"
    assert conc["synthesis_scores"] == {}
    assert conc["mimicry_plausibility"]["score"] == conc["rubric_indices"][
        "mimicry_plausibility"
    ]["score"]


def test_optional_legacy_blend_still_available() -> None:
    papers = [
        _gp(
            paper_id="h1",
            file_name="h1.txt",
            paper_role="target",
            relevance_grade=0.9,
            rubric_dimension_scores={
                "infection_process_relevance": 0.9,
                "protein_characterisation_quality": 0.9,
            },
            rubric_tags={"mimicry_potential_flag": "mimicry_strong"},
        ),
    ]
    synth = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "x",\n'
        '  "host_exploitation_score": 10,\n'
        '  "query_effector_score": 10,\n'
        '  "mimicry_plausibility_score": 20,\n'
        '  "pair_priority_score": 10,\n'
        '  "best_host_paper": "h1",\n'
        '  "best_query_paper": "",\n'
        '  "main_uncertainties": "n"\n'
        "}\n"
        "```\n"
    )
    conc = build_conclusion(papers, synth, synthesis_status="ok", rubric_weight=0.6)
    # 0.6*85 + 0.4*20 = 51 + 8 = 59
    assert conc["mimicry_plausibility"]["score"] == 59
    assert conc["mimicry_plausibility"]["source"] == "blend"
