import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "pipelines"
    / "auto_lit"
    / "scripts"
    / "diff_stage1_runs.py"
)
SPEC = importlib.util.spec_from_file_location("diff_stage1_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diff_stage1_runs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diff_stage1_runs)


def test_doi_attribution_precedence_and_unscoped_fallback():
    search = {
        "query_sources": {
            "entrez_pubtator": ["direct"],
            "europepmc_accession": [],
        },
        "query_term_hits": {
            "scoped": [{"pass": "pass2_base", "taxids": [446]}],
            "identifier": [{"pass": "pass1", "taxids": []}],
            "unscoped": [{"pass": "pass2_synonym", "taxids": []}],
        },
        "query_unattributed": {"pass2_base": ["combined-only"]},
        "query_fallbacks": [{"pass": "pass2_base", "n_kept": 12}],
    }

    expected = {
        "direct": "direct_database",
        "scoped": "organism_scoped_text",
        "identifier": "identifier_text",
        "unscoped": "unscoped_text",
        "combined-only": "unscoped_text",
        "unknown": "unattributed",
    }
    for doi, category in expected.items():
        assert diff_stage1_runs._doi_attribution(search, "query", doi) == category


def test_organism_name_fallback_is_still_scoped():
    search = {
        "query_sources": {},
        "query_term_hits": {
            "attributed": [
                {
                    "pass": "pass2_base",
                    "taxids": [],
                    "organism_terms": ["Legionella"],
                }
            ],
        },
        "query_unattributed": {"pass2_base": ["combined-only"]},
        "query_fallbacks": [
            {
                "pass": "pass2_base",
                "n_kept": 2,
                "organism_terms": ["Legionella"],
            }
        ],
    }
    for doi in ("attributed", "combined-only"):
        assert (
            diff_stage1_runs._doi_attribution(search, "query", doi)
            == "organism_scoped_text"
        )


def test_fallback_metrics_distinguish_page_truncation_from_kept_count():
    search = {
        "query_fallbacks": [
            {
                "dropped_taxids": [272624, 446],
                "hit_count": 957,
                "n_kept": 152,
                "truncated": True,
            },
            {
                "dropped_taxids": [272624, 446],
                "hit_count": 3,
                "n_kept": 2,
                "truncated": False,
            },
        ]
    }

    metrics = diff_stage1_runs._fallback_metrics(search, "query")
    assert metrics == {
        "events": 2,
        "retained_events": 2,
        "truncated_events": 1,
        "n_kept": 154,
        "max_hit_count": 957,
        "taxid_sets": "272624,446",
        "organism_term_sets": "",
    }
