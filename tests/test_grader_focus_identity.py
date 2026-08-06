"""Focus-gene identity gate for grader parsing."""

from __future__ import annotations

from auto_lit_search.grader_focus_identity import (
    enforce_focus_identity,
    text_mentions_focus_term,
)
from auto_lit_search.paper_io import focus_terms_for_paper_role


MICROBE_RUBRIC = {
    "axes": [
        {
            "id": "evidence_quality",
            "criteria": [
                {"id": "experimental_directness", "weight": "high"},
                {"id": "host_cell_evidence", "weight": "medium"},
            ],
        }
    ]
}


def test_text_mentions_focus_term_short_and_long():
    assert text_mentions_focus_term("studied pdxH in Legionella", ["pdxH", "Q5ZY37"])
    assert not text_mentions_focus_term("studied nimB resistance", ["pdxH", "Q5ZY37"])
    assert text_mentions_focus_term("gene NO is irrelevant", ["NO"])
    assert not text_mentions_focus_term("none of the above", ["NO"])


def test_enforce_focus_identity_zeros_when_excerpt_lacks_focus():
    parsed = {
        "criterion_scores": {
            "experimental_directness": {"score": 2, "note": "knockout"},
            "host_cell_evidence": {"score": 1, "note": "assay"},
        },
        "mention_type": "focal_study",
        "no_meaningful_mention": False,
        "claim_summary": "nimB confers metronidazole resistance",
        "rationale": "nimB study",
        "rubric_tags": {"novelty_flag": "none"},
    }
    out = enforce_focus_identity(
        parsed,
        excerpt="A paper about Clostridioides difficile nimB and heme.",
        focus_terms=["Q5ZY37", "pdxH"],
        rubric=MICROBE_RUBRIC,
        rubric_role="microbe",
    )
    assert out["no_meaningful_mention"] is True
    assert out["mention_type"] == "incidental_mention"
    assert out["relevance_grade"] == 0.0
    assert out["criterion_scores"]["experimental_directness"]["score"] == 0
    assert out["claim_summary"] == "NO_FOCUS_MENTION"


def test_enforce_focus_identity_zeros_when_claim_omits_focus():
    parsed = {
        "criterion_scores": {
            "experimental_directness": {"score": 2, "note": "PieF KO"},
            "host_cell_evidence": {"score": 2, "note": "macrophage"},
        },
        "mention_type": "focal_study",
        "no_meaningful_mention": False,
        "claim_summary": "PieF inhibits host mRNA deadenylation",
        "rationale": "PieF effector",
        "rubric_tags": {},
    }
    out = enforce_focus_identity(
        parsed,
        excerpt="Legionella effectors include PieF and briefly mention legK1 in a table.",
        focus_terms=["Q5ZVF7", "legK1"],
        rubric=MICROBE_RUBRIC,
        rubric_role="microbe",
    )
    assert out["no_meaningful_mention"] is True
    assert out["criterion_scores"]["experimental_directness"]["score"] == 0


def test_enforce_focus_identity_keeps_scores_when_focus_present():
    parsed = {
        "criterion_scores": {
            "experimental_directness": {"score": 2, "note": "legK1 KO"},
            "host_cell_evidence": {"score": 1, "note": "THP-1"},
        },
        "mention_type": "focal_study",
        "no_meaningful_mention": False,
        "claim_summary": "legK1 is a Dot/Icm kinase effector",
        "rationale": "legK1 study",
        "rubric_tags": {},
    }
    out = enforce_focus_identity(
        parsed,
        excerpt="We deleted legK1 and measured intracellular replication.",
        focus_terms=["Q5ZVF7", "legK1"],
        rubric=MICROBE_RUBRIC,
        rubric_role="microbe",
    )
    assert out["no_meaningful_mention"] is False
    assert out["criterion_scores"]["experimental_directness"]["score"] == 2


def test_focus_terms_for_paper_role():
    ctx = {
        "query": {"gene_name": "pdxH", "common_name": "PNPO", "synonyms": ["AVR58_x"]},
        "target": {"gene_name": "PNPO", "common_name": "pyridoxine oxidase"},
    }
    q = focus_terms_for_paper_role("query", "Q5ZY37", "Q9NVS9", ctx)
    assert "Q5ZY37" in q and "pdxH" in q
    t = focus_terms_for_paper_role("target", "Q5ZY37", "Q9NVS9", ctx)
    assert "Q9NVS9" in t and "PNPO" in t
