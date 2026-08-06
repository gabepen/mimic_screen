"""Tests for deterministic rubric scoring (grading schema v2)."""

import json

import pytest

from auto_lit_search.env_config import rubrics_dir
from auto_lit_search.rubric_scoring import (
    GRADING_SCHEMA_VERSION,
    HOST_PRIMARY_AXIS,
    aggregate_paper_scores,
    compute_axis_totals,
    normalize_criterion_scores,
    primary_axis_for_role,
    required_scored_criterion_ids,
    rubric_role_for_paper_role,
)

_RUBRICS = rubrics_dir()
if not _RUBRICS.is_dir():
    pytest.skip(
        f"Rubrics dir not found: {_RUBRICS} (set AUTO_LIT_DATA_ROOT or HOST_RUBRIC_PATH)",
        allow_module_level=True,
    )


def _load_rubric(name: str):
    return json.loads((_RUBRICS / name).read_text())


def test_normalize_criterion_scores_accepts_int_or_object():
    raw = {
        "functional_characterisation_depth": 2,
        "biochemical_property_definition": {"score": 1, "note": "partial assay"},
        "bad": "x",
    }
    out = normalize_criterion_scores(raw)
    assert out["functional_characterisation_depth"] == {"score": 2, "note": ""}
    assert out["biochemical_property_definition"] == {"score": 1, "note": ""}
    assert "bad" not in out


def test_host_weighted_axis_total_one_high_criterion_at_two():
    host = _load_rubric("host_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(host)
    scores = {cid: {"score": 0, "note": ""} for cid in crit_ids}
    scores["exploitation_process_overlap"] = {"score": 2, "note": "Rab GTPase"}

    axis_totals = compute_axis_totals(host, scores)
    axis2 = axis_totals[HOST_PRIMARY_AXIS]
    assert axis2.score == 4
    assert axis2.max_score == 12
    assert axis2.label == "4/12"


def test_host_primary_grade_and_relevance_sort():
    host = _load_rubric("host_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(host)
    scores = {cid: {"score": 0, "note": ""} for cid in crit_ids}
    scores["exploitation_process_overlap"] = {"score": 2, "note": ""}

    agg = aggregate_paper_scores(host, scores, rubric_role="host")
    assert agg["grading_schema_version"] == GRADING_SCHEMA_VERSION
    assert agg["primary_grade"] == "4/12"
    assert agg["relevance_sort"] == 4
    assert agg["rubric_dimension_scores"][HOST_PRIMARY_AXIS] == 4 / 12
    assert agg["relevance_grade"] > 0.0


def test_host_axis_three_bonus_does_not_reduce_relevance_grade():
    host = _load_rubric("host_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(host)
    scores = {cid: {"score": 0, "note": ""} for cid in crit_ids}
    scores["exploitation_process_overlap"] = {"score": 2, "note": ""}

    with_bonus = aggregate_paper_scores(host, scores, rubric_role="host")["relevance_grade"]
    scores["clinical_cohort_evidence"] = {"score": 2, "note": ""}
    higher = aggregate_paper_scores(host, scores, rubric_role="host")["relevance_grade"]
    assert higher >= with_bonus


def test_microbe_rubric_role_mapping():
    assert rubric_role_for_paper_role("query") == "microbe"
    assert rubric_role_for_paper_role("target") == "host"
    assert primary_axis_for_role("microbe") == "system_relevance"


def test_derive_axis_rationales_scores_only_ignores_notes():
    host = _load_rubric("host_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(host)
    scores = {cid: {"score": 0, "note": ""} for cid in crit_ids}
    scores["exploitation_process_overlap"] = {"score": 2, "note": "Rab GTPase cycling"}
    scores["functional_characterisation_depth"] = {"score": 1, "note": "partial assay"}

    from auto_lit_search.rubric_scoring import derive_axis_rationales_from_criterion_scores

    rax = derive_axis_rationales_from_criterion_scores(host, scores)
    assert HOST_PRIMARY_AXIS in rax
    assert "exploitation_process_overlap=2" in rax[HOST_PRIMARY_AXIS]
    assert "Rab GTPase" not in rax[HOST_PRIMARY_AXIS]
    assert "partial assay" not in rax[HOST_PRIMARY_AXIS]
    agg = aggregate_paper_scores(host, scores, rubric_role="host")
    assert agg["rubric_axis_rationales"][HOST_PRIMARY_AXIS] == rax[HOST_PRIMARY_AXIS]


def test_microbe_paper_grade_readable():
    microbe = _load_rubric("legionella_rubric.json")
    crit_ids = required_scored_criterion_ids(microbe)
    scores = {cid: {"score": 1, "note": ""} for cid in crit_ids}
    agg = aggregate_paper_scores(microbe, scores, rubric_role="microbe")
    assert "/" in agg["paper_grade"]
    score_part, max_part = agg["paper_grade"].split("/")
    assert int(score_part) > 0
    assert int(max_part) > int(score_part)


def test_drosophila_host_primary_grade_uses_configured_axis():
    host = _load_rubric("drosophila_host_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(host)
    scores = {cid: {"score": 0, "note": ""} for cid in crit_ids}
    scores["exploitation_process_overlap"] = {"score": 2, "note": ""}

    agg = aggregate_paper_scores(host, scores, rubric_role="host")
    assert agg["primary_grade"] == "4/12"
    assert agg["relevance_sort"] == 4


def test_wolbachia_microbe_grader_v2_criterion_ids():
    microbe = _load_rubric("wolbachia_wmel_rubric_v1.json")
    crit_ids = required_scored_criterion_ids(microbe)
    scores = {cid: {"score": 1, "note": ""} for cid in crit_ids}
    agg = aggregate_paper_scores(microbe, scores, rubric_role="microbe")
    assert agg["primary_grade"] == "8/16"
    assert agg["grading_schema_version"] == GRADING_SCHEMA_VERSION
