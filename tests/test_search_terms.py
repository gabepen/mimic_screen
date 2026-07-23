"""Tests for search-term usability filters and pass2 query building."""

from __future__ import annotations

import json

import pandas as pd
import auto_lit_search.search as search_module

from auto_lit_search.search_terms import (
    is_usable_search_term,
    sanitize_common_name_for_idmap,
)
from auto_lit_search.search import (
    _build_europepmc_text_query_from_terms,
    _build_europepmc_text_query_pass2_base_only,
    _collect_base_terms_for_pass2,
    _collect_search_terms_used,
    _attribute_result_ids_to_terms,
    _summarize_term_hits,
)
from auto_lit_search.gene_symbols import (
    aliases_excluding_primary,
    first_uniprot_orf_name,
    format_gene_aliases,
    gene_aliases_from_uniprot_gene,
    infer_gene_name_from_protein_description,
    is_symbol_like_token,
    prefer_gene_name_uniprot,
)


def test_rejects_method_word_knockdown():
    assert not is_usable_search_term("knockdown", kind="common_name")
    assert sanitize_common_name_for_idmap("knockdown") is None


def test_keeps_real_short_gene_symbols():
    assert is_usable_search_term("kdn", kind="gene_name")
    assert is_usable_search_term("mip", kind="gene_name")
    assert is_usable_search_term("vipD", kind="gene_name")
    assert is_usable_search_term("legK2", kind="gene_name")
    assert is_usable_search_term("Cs1", kind="gene_name")


def test_rejects_short_ambiguous_and_cs():
    assert not is_usable_search_term("gap", kind="gene_name")
    assert not is_usable_search_term("tpm", kind="gene_name")
    assert not is_usable_search_term("CS", kind="gene_name")
    assert not is_usable_search_term("CS", kind="synonym")


def test_rejects_technical_aliases():
    for term in ("143198_at", "anon-EST:Liang-2.53", "clone 2.53", "ORF1", "11981"):
        assert not is_usable_search_term(term, kind="alias"), term
    assert is_usable_search_term("CG3861", kind="alias")
    assert is_usable_search_term("kdn", kind="alias")


def test_rejects_generic_protein_phrases():
    assert not is_usable_search_term("NfeD family protein", kind="common_name")
    assert not is_usable_search_term(
        "Glyceraldehyde-3-phosphate dehydrogenase", kind="common_name"
    )
    assert not is_usable_search_term("Iron regulatory protein 1B", kind="common_name")


def test_rejects_generic_functional_terms():
    for term in ("ATPase", "peptidase", "homodimer", "methyltransferase"):
        assert not is_usable_search_term(term, kind="gene_name"), term
        assert not is_usable_search_term(term, kind="alias"), term


def test_keeps_cg_and_accessions():
    assert is_usable_search_term("CG3861", kind="synonym")
    assert is_usable_search_term("Dmel_CG3861", kind="locus_tag")
    assert is_usable_search_term("AAF46159.1", kind="genbank_acc")


def test_prefer_uniprot_gene_name_over_mygene():
    assert prefer_gene_name_uniprot("Cs1", "kdn") == "Cs1"
    assert prefer_gene_name_uniprot(None, "kdn") == "kdn"
    assert prefer_gene_name_uniprot("Cs1", None) == "Cs1"


def test_uniprot_aliases_include_kdn_and_cg():
    g0 = {
        "geneName": {"value": "Cs1"},
        "synonyms": [{"value": "kdn"}, {"value": "l(1)G0030"}],
        "orfNames": [{"value": "CG3861"}],
    }
    aliases = gene_aliases_from_uniprot_gene(g0)
    assert "kdn" in aliases
    assert "CG3861" in aliases
    assert "l(1)G0030" in aliases
    cleaned = aliases_excluding_primary(aliases + ["Cs1", "kdn"], "Cs1")
    assert "Cs1" not in cleaned
    assert "kdn" in cleaned
    joined = format_gene_aliases(cleaned)
    assert joined is not None
    assert "kdn" in joined
    assert "CG3861" in joined


def test_pass2_uses_aliases_not_common_name_knockdown():
    row = pd.Series(
        {
            "target": "Q9W401",
            "target_gene_name": "Cs1",
            "target_gene_aliases": "CG3861|kdn|l(1)G0030",
            "target_common_name": "knockdown",
        }
    )
    terms = _collect_base_terms_for_pass2(row, prefix="target")
    assert "Cs1" in terms
    assert "kdn" in terms
    assert "CG3861" in terms
    assert "knockdown" not in [t.lower() for t in terms]

    used = _collect_search_terms_used(
        row, prefix="target", extra_terms=["knockdown", "Csyn"]
    )
    assert "knockdown" not in [t.lower() for t in used]
    assert "Csyn" in used
    assert "Cs1" in used

    q, kinds = _build_europepmc_text_query_pass2_base_only(row, taxid=7227, prefix="target")
    assert q is not None
    assert "knockdown" not in q.lower()
    assert "Cs1" in q
    assert "kdn" in q
    assert "ORGANISM_ID:7227" in q
    assert "gene_name" in kinds
    assert "alias" in kinds
    assert '(TITLE_ABS:"Cs1" OR BODY:"Cs1")' in q
    assert 'TITLE_ABS:"kdn"' in q
    assert 'BODY:"kdn"' not in q


def test_pass2_supports_multiple_organism_taxids():
    row = pd.Series(
        {
            "query": "Q5ZSQ2",
            "query_gene_name": "sidD",
            "query_gene_aliases": "",
        }
    )
    query, _ = _build_europepmc_text_query_pass2_base_only(
        row, taxid=[272624, 446], prefix="query"
    )
    assert query is not None
    assert " AND (ORGANISM_ID:272624 OR ORGANISM_ID:446)" in query


def test_ambiguous_synonyms_are_title_abstract_only():
    q, kinds = _build_europepmc_text_query_from_terms(
        [("gene_name", "TPI1"), ("synonym", "TIM"), ("alias", "TPID")],
        taxid=9606,
    )
    assert q is not None
    assert '(TITLE_ABS:"TPI1" OR BODY:"TPI1")' in q
    assert 'TITLE_ABS:"TIM"' in q
    assert 'BODY:"TIM"' not in q
    assert 'TITLE_ABS:"TPID"' in q
    assert 'BODY:"TPID"' not in q
    assert kinds == ["gene_name", "synonym", "alias"]


def test_symbol_like_token_accepts_real_symbols():
    for tok in ("CG3861", "Mettl5", "RFT1", "IscU", "ApepP", "RomA", "PCNA", "ATIC", "Lgt1"):
        assert is_symbol_like_token(tok), tok


def test_symbol_like_token_rejects_description_words():
    for tok in (
        "homolog",
        "homologue",
        "mitochondrial",
        "helicase",
        "Methyltransferase",
        "Oxidoreductase",
        "Glycosyltransferase",
        "containing",
        "ATPase",
        "homodimer",
        "III",
        "an",
    ):
        assert not is_symbol_like_token(tok), tok


def test_infer_prefers_symbol_over_localization_and_homolog():
    # Trailing ", mitochondrial" must be stripped; remaining word is not symbol-like.
    entry = {
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": "Protein arginine methyltransferase NDUFAF7 homolog, mitochondrial"}
            }
        }
    }
    assert infer_gene_name_from_protein_description(entry) is None

    effector = {
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Glucosyltransferase Lgt1"}}
        }
    }
    assert infer_gene_name_from_protein_description(effector) == "Lgt1"


def test_orf_name_precedence_for_drosophila():
    g0 = {"orfNames": [{"value": "CG14683"}]}
    assert first_uniprot_orf_name(g0) == "CG14683"


def test_from_terms_filters_gap_synonym():
    q, kinds = _build_europepmc_text_query_from_terms(
        [("gene_name", "gap"), ("synonym", "CG3861")],
        taxid=7227,
    )
    assert q is not None
    assert '"gap"' not in q.lower()
    assert "CG3861" in q
    assert "gene_name" not in kinds
    assert "synonym" in kinds


def test_term_hit_attribution_records_exact_matching_term(monkeypatch):
    monkeypatch.setenv("AUTO_LIT_TERM_HIT_ATTRIBUTION", "1")

    def fake_search(query, _session, _cache, delay=0, gate=None):
        if '"Cs1"' in query:
            return {"dois": ["doi:one"], "titles": ["one"]}
        if '"CG3861"' in query:
            return {"dois": ["doi:one", "doi:two"], "titles": ["one", "two"]}
        return {"dois": [], "titles": []}

    monkeypatch.setattr(search_module, "_run_europepmc_search_query", fake_search)
    hits, unresolved = _attribute_result_ids_to_terms(
        ["doi:one", "doi:two", "doi:unresolved"],
        [("gene_name", "Cs1"), ("alias", "CG3861")],
        pass_name="pass2_base",
        taxid=7227,
        session=object(),
        cache={},
        delay=0,
    )

    assert [row["term"] for row in hits["doi:one"]] == ["Cs1", "CG3861"]
    assert hits["doi:one"][0]["taxids"] == [7227]
    assert [row["term"] for row in hits["doi:two"]] == ["CG3861"]
    assert unresolved == ["doi:unresolved"]
    assert _summarize_term_hits(hits) == [
        {
            "term": "CG3861",
            "kind": "alias",
            "pass": "pass2_base",
            "taxids": [7227],
            "organism_terms": [],
            "n_papers": 2,
        },
        {
            "term": "Cs1",
            "kind": "gene_name",
            "pass": "pass2_base",
            "taxids": [7227],
            "organism_terms": [],
            "n_papers": 1,
        },
    ]


def test_taxid_fallback_requires_organism_name_text(monkeypatch):
    monkeypatch.setenv("AUTO_LIT_TERM_HIT_ATTRIBUTION", "1")

    def fake_search(query, _session, _cache, delay=0, gate=None):
        if "ORGANISM_ID" in query:
            return {
                "dois": [],
                "titles": [],
                "hit_count": 0,
                "n_raw": 0,
                "truncated": False,
                "request_ok": True,
            }
        assert 'TITLE_ABS:"Legionella"' in query
        assert "ORGANISM_ID" not in query
        return {
            "dois": ["doi:a", "doi:b"],
            "titles": ["a", "b"],
            "hit_count": 2,
            "n_raw": 2,
            "truncated": False,
            "request_ok": True,
        }

    monkeypatch.setattr(search_module, "_run_europepmc_search_query", fake_search)

    row = pd.Series(
        {
            "query": "Q5ZSQ2",
            "query_gene_name": "sidD",
            "query_gene_aliases": "",
        }
    )
    res = search_module.run_europepmc_search_for_row(
        row,
        [272624, 446],
        session=object(),
        cache={},
        prefix="query",
        delay=0,
        organism_terms=["Legionella pneumophila", "Legionella"],
    )

    fallbacks = res["organism_fallbacks"]
    assert len(fallbacks) >= 1
    base = next(f for f in fallbacks if f["pass"] == "pass2_base")
    assert base["dropped_taxids"] == [272624, 446]
    assert base["fallback_scope"] == "organism_text"
    assert base["organism_terms"] == ["Legionella pneumophila", "Legionella"]
    assert base["hit_count"] == 2
    assert base["truncated"] is False
    assert base["n_kept"] == 2
    assert all(
        hit["organism_terms"] == ["Legionella pneumophila", "Legionella"]
        for hit in res["paper_term_hits"]["doi:a"]
    )


def test_failed_taxid_request_does_not_trigger_fallback(monkeypatch):
    def fake_search(_query, _session, _cache, delay=0, gate=None):
        return {
            "dois": [],
            "titles": [],
            "hit_count": 0,
            "n_raw": 0,
            "truncated": False,
            "request_ok": False,
        }

    monkeypatch.setattr(search_module, "_run_europepmc_search_query", fake_search)
    row = pd.Series({"query": "Q5ZSQ2", "query_gene_name": "sidD"})
    result = search_module.run_europepmc_search_for_row(
        row,
        [272624, 446],
        session=object(),
        cache={},
        prefix="query",
        delay=0,
        organism_terms=["Legionella"],
    )
    assert result["organism_fallbacks"] == []


def test_request_gate_spaces_starts():
    gate = search_module.RequestGate(0.05)
    t0 = search_module.time.monotonic()
    gate.wait()
    gate.wait()
    elapsed = search_module.time.monotonic() - t0
    assert elapsed >= 0.045


def test_locked_cache_get_store_roundtrip():
    cache = search_module.LockedCache()
    hit, _ = cache.get_if_present("q")
    assert hit is False
    cache.store("q", {"dois": ["a"]})
    hit, val = cache.get_if_present("q")
    assert hit is True
    assert val == {"dois": ["a"]}
    # store is first-writer-wins
    assert cache.store("q", {"dois": ["b"]}) == {"dois": ["a"]}


def test_parallel_run_preserves_row_order(monkeypatch):
    monkeypatch.setenv("AUTO_LIT_TERM_HIT_ATTRIBUTION", "0")

    def fake_crossref(uniprot_id, session, cache, delay=0.35, gate=None):
        return {"dois": [f"acc:{uniprot_id}"], "titles": [str(uniprot_id)]}

    def fake_text(row, taxid, session, cache, delay=0.35, prefix="query",
                  extra_terms=None, organism_terms=None, gate=None):
        uid = str(row.get(prefix) or "")
        return {
            "dois": [f"text:{prefix}:{uid}"],
            "titles": [uid],
            "pass1_count": 0,
            "pass2_count": 1,
            "pass1_dois": [],
            "pass2_dois": [f"text:{prefix}:{uid}"],
            "pass2_base_count": 1,
            "pass2_synonym_count": 0,
            "pass2_overlap_count": 0,
            "pass2_base_dois": [f"text:{prefix}:{uid}"],
            "pass2_synonym_dois": [],
            "pass2_overlap_dois": [],
            "search_terms": [],
            "paper_term_hits": {},
            "unattributed_term_hit_dois_by_pass": {},
            "organism_fallbacks": [],
        }

    monkeypatch.setattr(search_module, "run_europepmc_crossref", fake_crossref)
    monkeypatch.setattr(search_module, "run_europepmc_search_for_row", fake_text)
    monkeypatch.setattr(search_module, "_load_human_gene_name_synonyms", lambda **k: {})
    monkeypatch.setattr(search_module, "_load_mygene_synonyms_for_entrez", lambda *a, **k: {})
    monkeypatch.setattr(search_module, "_configure_file_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(search_module, "_force_ipv4_resolution", lambda: None)

    df = pd.DataFrame(
        {
            "query": ["Q1", "Q2", "Q3"],
            "target": ["T1", "T2", "T3"],
        }
    )
    out = search_module.run(df, output_dir="/tmp", use_cache=False, workers=3, delay=0)
    assert list(out["query"]) == ["Q1", "Q2", "Q3"]
    assert [json.loads(x)[0] for x in out["query_paper_dois"]] == [
        "acc:Q1",
        "acc:Q2",
        "acc:Q3",
    ]
