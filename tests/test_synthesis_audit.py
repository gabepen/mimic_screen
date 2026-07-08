"""Tests for one-shot synthesis audit prompts and sampling."""

from __future__ import annotations

from auto_lit_search.synthesis_audit import (
    build_one_shot_audit_prompt,
    pair_priority_stratum,
)


def test_build_one_shot_audit_prompt_includes_synonyms():
    gene_context = {
        "query": {
            "gene_name": "legD2",
            "locus_tag": "lpg1234",
            "common_name": "Dot/Icm T4SS effector LegD2",
            "synonyms": ["legD2", "lpg1234"],
        },
        "target": {
            "gene_name": "PHYHD1",
            "common_name": "phytanoyl-CoA dioxygenase domain containing 1",
            "synonyms": ["PHYHD1"],
        },
    }
    prompt = build_one_shot_audit_prompt(
        alignment_id="Q5ZY57_Q5SRE7",
        query_id="Q5ZY57",
        target_id="Q5SRE7",
        gene_context=gene_context,
    )
    assert "structural similarity" in prompt
    assert "legD2" in prompt
    assert "PHYHD1" in prompt
    assert "Quick results summary:" in prompt
    assert "pre-retrieved paper texts" in prompt.lower() or "do not have pre-retrieved" in prompt.lower()


def test_pair_priority_stratum_bins():
    assert pair_priority_stratum(80) == "strong"
    assert pair_priority_stratum(60) == "strong"
    assert pair_priority_stratum(50) == "mid"
    assert pair_priority_stratum(39) == "weak"
