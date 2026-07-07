"""Tests for mention-centered paper excerpts."""

from __future__ import annotations

from auto_lit_search.mention_excerpt import (
    MENTION_SEPARATOR,
    _EXCERPT_MODE_MENTIONS,
    _EXCERPT_MODE_NO_MENTIONS,
    build_mention_excerpt,
    cluster_mention_sites,
    collect_search_terms,
    find_hits,
    format_excerpt_block,
)


def _row(
    *,
    role: str = "target",
    query: str = "P11111",
    target: str = "Q22222",
    gene_context: dict | None = None,
) -> dict:
    if gene_context is None:
        gene_context = {
            "query": {"gene_name": "geneA", "common_name": "ProteinA"},
            "target": {
                "gene_name": "MRPL19",
                "common_name": "mitochondrial",
                "synonyms": ["MRP-L19"],
            },
        }
    return {
        "query": query,
        "target_id": target,
        "paper_role": role,
        "alignment_id": "P11111_Q22222",
        "gene_context": gene_context,
    }


def test_two_mentions_even_split():
    pad = "x" * 3000
    text = (
        f"{pad} First mention of MRPL19 in this paragraph. More text here.\n\n"
        f"{pad} Second mention of MRPL19 appears later in the document."
    )
    row = _row(role="target")
    result = build_mention_excerpt(text, row, max_chars=10000)
    assert result.excerpt_mode == _EXCERPT_MODE_MENTIONS
    assert result.n_mentions == 2
    assert result.budget_per_mention == 5000
    assert MENTION_SEPARATOR in result.excerpt
    assert len(result.excerpt.split(MENTION_SEPARATOR)) == 2


def test_query_role_only_searches_query_terms():
    text = "The pathogen geneA is essential for virulence."
    row = _row(role="query")
    terms = collect_search_terms(row)
    literals = {t.literal.lower() for t in terms}
    assert "genea" in literals
    assert "mrpl19" not in literals


def test_no_mentions_fallback():
    text = "This paper discusses unrelated pathways only. " * 100
    row = _row(role="target")
    result = build_mention_excerpt(text, row, max_chars=10000, no_mention_fallback_chars=500)
    assert result.excerpt_mode == _EXCERPT_MODE_NO_MENTIONS
    assert "not found in text" in result.focus_gene
    assert result.n_mentions == 0


def test_overlapping_synonyms_one_site():
    text = "We analyzed MRPL19 (MRP-L19) expression in cells."
    row = _row(role="target")
    hits = find_hits(text, collect_search_terms(row))
    sites = cluster_mention_sites(hits, cluster_gap=50)
    assert len(sites) == 1
    result = build_mention_excerpt(text, row, max_chars=5000)
    assert result.n_mentions == 1


def test_format_excerpt_block_includes_metadata():
    from types import SimpleNamespace

    row = _row(role="target")
    text = "MRPL19 was knocked down with strong phenotype."
    result = build_mention_excerpt(text, row, max_chars=2000)
    gp = SimpleNamespace(file_name="paper.txt", paper_role="target")
    block = format_excerpt_block(gp, result)
    assert "genes_found_in_text:" in block
    assert "matched_terms:" in block
    assert "MRPL19" in block
