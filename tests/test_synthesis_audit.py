"""Tests for one-shot synthesis audit helpers."""

from __future__ import annotations

import pytest

from auto_lit_search.synthesis_audit import (
    brief_system_preamble,
    build_oneshot_synthesis_audit_prompt,
    pair_priority_stratum,
)


def test_pair_priority_stratum_bins():
    assert pair_priority_stratum(80) == "strong"
    assert pair_priority_stratum(60) == "strong"
    assert pair_priority_stratum(50) == "mid"
    assert pair_priority_stratum(39) == "weak"


def test_brief_system_preamble():
    text = brief_system_preamble("Wolbachia pipientis wMel", "Drosophila melanogaster")
    assert "Wolbachia pipientis wMel" in text
    assert "Drosophila melanogaster" in text
    assert "structural similarity" in text
    assert "Host exploitation" not in text
    assert "Per-paper" not in text


def test_oneshot_prompt_is_brief_research_question():
    prompt = build_oneshot_synthesis_audit_prompt(
        query="Q5ZRZ3",
        target_id="P12345",
        query_organism="Legionella pneumophila",
        target_organism="Homo sapiens",
        gene_context={
            "query": {
                "gene_name": "lepB",
                "common_name": "effector LepB",
                "synonyms": ["lpg2490"],
            },
            "target": {
                "gene_name": "RAB1A",
                "common_name": "Ras-related protein Rab-1A",
                "synonyms": ["RAB1"],
            },
        },
    )
    assert prompt.startswith("Microbe: Legionella pneumophila.")
    assert "Homo sapiens" in prompt
    assert "UniProt ID: Q5ZRZ3" in prompt
    assert "UniProt ID: P12345" in prompt
    assert "common gene name: lepB / effector LepB" in prompt
    assert "common gene name: RAB1A / Ras-related protein Rab-1A" in prompt
    assert "unknown" not in prompt
    assert "structural similarity or mimicry" in prompt
    assert "Alignment:" not in prompt
    assert "Q5ZRZ3_P12345" not in prompt
    assert "Quick results summary" not in prompt
    assert "Host exploitation:" not in prompt
    assert "Research questions for this system" not in prompt
    assert len(prompt.splitlines()) <= 5


def test_display_uses_gene_name_from_idmap_style_meta():
    prompt = build_oneshot_synthesis_audit_prompt(
        query="Q5ZVA9",
        target_id="P30039",
        query_organism="Legionella pneumophila",
        target_organism="Homo sapiens",
        gene_context={
            "query": {"gene_name": "PhzF", "common_name": ""},
            "target": {"gene_name": "PBLD", "common_name": ""},
        },
    )
    assert "common gene name: PhzF" in prompt
    assert "common gene name: PBLD" in prompt
    assert "unknown" not in prompt


def test_oneshot_prompt_requires_organisms():
    with pytest.raises(ValueError):
        build_oneshot_synthesis_audit_prompt(
            query="Q1",
            target_id="P1",
            query_organism="",
            target_organism="B",
        )
