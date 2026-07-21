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
    query_asserts_molecular_mimicry,
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


def test_quick_summary_footer_has_mimicry_anchors() -> None:
    footer = quick_summary_prompt_footer()
    assert "80-100: query literature establishes molecular/functional mimicry" in footer
    assert "Do not lower mimicry_plausibility_score" in footer


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


def test_query_mimicry_detected_from_criterion_notes() -> None:
    papers = [
        _gp(
            claim_summary="Translocated effector.",
            criterion_scores={
                "system_context": {
                    "score": 2,
                    "note": "Sec7-domain fold matching eukaryotic Arf GEFs.",
                }
            },
        )
    ]
    assert query_asserts_molecular_mimicry(papers)


def test_query_mimicry_literature_floors_rubric_and_protects_blend() -> None:
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
            relevance_grade=0.4,
            rubric_dimension_scores={"infection_process_relevance": 0.3},
            claim_summary="Host actin regulator, infection-naive.",
        ),
    ]
    assert query_asserts_molecular_mimicry(papers)
    rubric = compute_rubric_scorecard(papers)
    assert rubric["evidence"]["query_mimicry_literature"] is True
    assert rubric["mimicry_plausibility"]["score"] >= 80

    # Conservative LLM pair score must not demote query-side molecular mimicry.
    synth = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Known F-box mimic; Foldseek host not co-mentioned.",\n'
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
    assert conc["mimicry_plausibility"]["score"] >= 80
    assert score_to_dimension_tier(conc["mimicry_plausibility"]["score"]) == "Strong"


def test_no_query_mimicry_language_allows_low_llm_blend() -> None:
    papers = [
        _gp(
            claim_summary="Translocated Dot/Icm effector with phospholipase activity.",
            rationale="Strong effector evidence without mimicry wording.",
        ),
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
    assert not query_asserts_molecular_mimicry(papers)
    synth = (
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "Effector yes; pair mimicry unclear.",\n'
        '  "host_exploitation_score": 50,\n'
        '  "query_effector_score": 90,\n'
        '  "mimicry_plausibility_score": 20,\n'
        '  "pair_priority_score": 55,\n'
        '  "best_host_paper": "h1",\n'
        '  "best_query_paper": "p1",\n'
        '  "main_uncertainties": "No pair link."\n'
        "}\n"
        "```\n"
    )
    conc = build_conclusion(papers, synth, synthesis_status="ok")
    # Without query mimicry literature, LLM can still pull blended mimicry down.
    assert conc["mimicry_plausibility"]["score"] < 80
