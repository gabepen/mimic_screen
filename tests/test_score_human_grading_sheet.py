"""Tests for human blind-sheet scoring."""

import csv
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from score_human_grading_sheet import (  # noqa: E402
    build_detailed_report,
    discover_human_criterion_columns,
    format_report_markdown,
    llm_aggregate_from_answer_key,
    parse_axis_rationales,
    parse_criterion_scores_from_rationales,
    score_sheet_row,
    _compare_to_answer_key,
)
from auto_lit_search.env_config import resolve_rubric_paths  # noqa: E402
from sample_blind_grading_audit import load_rubric_spec  # noqa: E402


def _legionella_rubric_spec():
    host, microbe = resolve_rubric_paths()
    return load_rubric_spec(host, microbe)


def test_score_sheet_row_matches_aggregate_paper_scores():
    spec = _legionella_rubric_spec()
    row = {
        "sample_id": "audit_001",
        "paper_role": "target",
        "human_exploitation_process_overlap": "2",
    }
    for crit in spec.criteria_for_role("target"):
        row.setdefault(f"human_{crit}", "0")

    scored = score_sheet_row(row, spec)
    assert scored is not None
    assert scored["primary_grade"] == "4/12"
    assert scored["paper_grade_score"] > 0


def test_score_human_grading_sheet_cli(tmp_path):
    spec = _legionella_rubric_spec()
    sheet = tmp_path / "sheet.csv"
    crit_ids = [c.criterion_id for c in spec.criteria]
    fieldnames = ["sample_id", "paper_role"] + [f"human_{c}" for c in crit_ids]
    values = {f"human_{c}": "0" for c in crit_ids}
    values["human_exploitation_process_overlap"] = "2"
    with sheet.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"sample_id": "audit_001", "paper_role": "target", **values})

    out = tmp_path / "scores.json"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "score_human_grading_sheet.py"),
            "--sheet",
            str(sheet),
            "--out",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_scored"] == 1
    assert payload["n_ungraded"] == 0
    assert payload["scores"]["audit_001"]["primary_grade"] == "4/12"


def test_discover_columns_any_order():
    cols = discover_human_criterion_columns(
        ["sample_id", "human_claim_confidence", "doi", "human_experimental_directness"]
    )
    assert cols == ["claim_confidence", "experimental_directness"]


def test_score_sheet_row_shuffled_columns_partial_fill():
    spec = _legionella_rubric_spec()
    row = {
        "sample_id": "audit_002",
        "paper_role": "target",
        "human_exploitation_process_overlap": "2",
        "human_notes": "Rab overlap",
    }
    scored = score_sheet_row(row, spec)
    assert scored is not None
    assert scored["primary_grade"] == "4/12"
    assert scored["criterion_scores"] == {
        "exploitation_process_overlap": {"score": 2, "note": ""},
    }


def test_cli_skips_ungraded_rows(tmp_path):
    spec = _legionella_rubric_spec()
    sheet = tmp_path / "sheet.csv"
    crit_ids = [c.criterion_id for c in spec.criteria]
    fieldnames = ["sample_id", "paper_role"] + list(reversed([f"human_{c}" for c in crit_ids]))
    with sheet.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"sample_id": "audit_empty", "paper_role": "target"})
        writer.writerow(
            {
                "sample_id": "audit_filled",
                "paper_role": "target",
                "human_exploitation_process_overlap": "2",
            }
        )

    out = tmp_path / "scores.json"
    import subprocess

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "score_human_grading_sheet.py"),
            "--sheet",
            str(sheet),
            "--out",
            str(out),
        ],
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 2
    assert payload["n_scored"] == 1
    assert payload["n_ungraded"] == 1
    assert "audit_empty" not in payload["scores"]
    assert payload["scores"]["audit_filled"]["primary_grade"] == "4/12"


def test_parse_criterion_scores_from_rationales():
    rax = {
        "evidence_quality": (
            "claim_confidence=2: Declarative; experimental_directness=2: PaAP deletion; "
            "host_cell_evidence=0: No host cells"
        ),
    }
    scores = parse_criterion_scores_from_rationales(rax)
    assert scores["claim_confidence"]["score"] == 2
    assert scores["claim_confidence"]["note"] == "Declarative"
    assert scores["experimental_directness"]["score"] == 2
    assert scores["host_cell_evidence"]["score"] == 0


def test_parse_axis_rationales_preserves_axis_grouping():
    rax = {
        "evidence_quality": "claim_confidence=2: Declarative; host_cell_evidence=0: None",
        "system_relevance": "strain_relevance=2: Clinical isolate",
    }
    by_axis = parse_axis_rationales(rax)
    assert by_axis["evidence_quality"]["claim_confidence"]["note"] == "Declarative"
    assert by_axis["system_relevance"]["strain_relevance"]["score"] == 2


def test_build_detailed_report_includes_llm_reasoning():
    spec = _legionella_rubric_spec()
    row = {
        "sample_id": "audit_001",
        "doi": "10.1/test",
        "gene_focus_id": "Q1",
        "gene_focus_symbol": "geneA",
        "paper_role": "query",
        "human_notes": "looks weak",
    }
    human_scores = {f"human_{c}": "0" for c in spec.criteria_for_role("query")}
    human_row = {**row, **human_scores}
    human = score_sheet_row(human_row, spec)
    llm_entry = {
        "paper_role": "query",
        "rationale": "Paper about PaAP biofilms.",
        "rubric_axis_rationales": {
            "evidence_quality": (
                "claim_confidence=2: Declarative claim; experimental_directness=2: KO done; "
                "host_cell_evidence=0: No host cells; replication_support=0: None; "
                "translocation_confirmation=0: None"
            ),
            "system_relevance": (
                "conservation_evidence=1: dN/dS noted; host_cell_type_match=0: None; "
                "host_target_system=0: None; lcv_biology_connection=0: None; "
                "strain_relevance=2: PA14 clinical"
            ),
        },
        "rubric_tags": {"novelty_flag": "none"},
    }
    llm_agg = llm_aggregate_from_answer_key(llm_entry, spec)
    comparison = _compare_to_answer_key(human, llm_agg)
    report = build_detailed_report("audit_001", row, human, llm_entry, comparison, spec)
    assert report["llm_claim_summary"] == "Paper about PaAP biofilms."
    axis1 = report["axes"][0]
    claim = next(c for c in axis1["criteria"] if c["criterion_id"] == "claim_confidence")
    assert claim["llm_reasoning"] == "Declarative claim"
    md = format_report_markdown([report], {"sheet": "t.csv", "n_scored": 1, "n_rows": 1})
    assert "Declarative claim" in md
    assert "Paper about PaAP biofilms." in md


def test_llm_aggregate_from_answer_key_rationales():
    spec = _legionella_rubric_spec()
    entry = {
        "paper_role": "query",
        "rubric_axis_rationales": {
            "evidence_quality": (
                "claim_confidence=2; experimental_directness=2; "
                "host_cell_evidence=0; replication_support=0; "
                "translocation_confirmation=0"
            ),
            "system_relevance": (
                "conservation_evidence=1; host_cell_type_match=0; "
                "host_target_system=0; lcv_biology_connection=0; strain_relevance=2"
            ),
        },
    }
    agg = llm_aggregate_from_answer_key(entry, spec)
    assert agg is not None
    assert agg["paper_grade"] == "9/32"
    assert agg["criterion_scores"]["claim_confidence"]["score"] == 2
