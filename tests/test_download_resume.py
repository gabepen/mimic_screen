"""Tests for download resume modes (terminal failures + global cache)."""

from __future__ import annotations

import json
import os
import tempfile

from auto_lit_search.download_manifest import (
    GLOBAL_OUTCOME_CACHE_FILENAME,
    _manifest_row_satisfied,
    _paper_pair_key,
    classify_alignment_papers,
    is_alignment_download_complete,
    load_global_outcome_cache,
    record_global_outcomes_from_rows,
    write_alignment_download_complete,
)


def test_failed_terminal_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        papers_dir = tmp
        manifest_map = {
            _paper_pair_key("10.1/dead", "query"): {
                "paper_id": "10.1/dead",
                "source": "query",
                "status": "failed",
                "file_stem": "dead__query",
                "message": "no text extracted",
            }
        }
        plan = classify_alignment_papers(
            [("10.1/dead", "query")],
            manifest_map,
            papers_dir,
            {},
            retry_failed=False,
            no_cache=False,
        )
        assert plan.to_fetch == []
        assert plan.stats["terminal_failed"] == 1
        assert plan.stats["to_fetch"] == 0


def test_retry_failed_requeues() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_map = {
            _paper_pair_key("10.1/dead", "query"): {
                "paper_id": "10.1/dead",
                "source": "query",
                "status": "failed",
                "file_stem": "dead__query",
            }
        }
        plan = classify_alignment_papers(
            [("10.1/dead", "query")],
            manifest_map,
            tmp,
            {},
            retry_failed=True,
            no_cache=False,
        )
        assert plan.to_fetch == [("10.1/dead", "query")]
        assert plan.stats["to_fetch"] == 1


def test_alignment_complete_skips_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        write_alignment_download_complete(tmp, {"total_expected": 1})
        assert is_alignment_download_complete(tmp) is True


def test_global_cache_skips_failed_doi() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = os.path.join(tmp, "logs")
        os.makedirs(logs_dir)
        global_cache = load_global_outcome_cache(logs_dir)
        record_global_outcomes_from_rows(
            logs_dir,
            [
                {
                    "paper_id": "10.1/dead",
                    "doi": "10.1/dead",
                    "status": "failed",
                    "message": "no text extracted",
                }
            ],
            global_cache,
        )
        plan = classify_alignment_papers(
            [("10.1/dead", "target")],
            {},
            tmp,
            global_cache,
            retry_failed=False,
            no_cache=False,
        )
        assert plan.to_fetch == []
        assert plan.stats["global_skipped"] == 1
        key = _paper_pair_key("10.1/dead", "target")
        assert key in plan.global_inject
        assert plan.global_inject[key]["status"] == "failed"


def test_no_cache_ignores_all() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = os.path.join(tmp, "logs")
        os.makedirs(logs_dir)
        global_cache = load_global_outcome_cache(logs_dir)
        record_global_outcomes_from_rows(
            logs_dir,
            [{"paper_id": "10.1/dead", "status": "failed", "message": "x"}],
            global_cache,
        )
        manifest_map = {
            _paper_pair_key("10.1/dead", "query"): {
                "paper_id": "10.1/dead",
                "source": "query",
                "status": "failed",
            },
            _paper_pair_key("10.1/ok", "query"): {
                "paper_id": "10.1/ok",
                "source": "query",
                "status": "ok",
                "file_stem": "ok__query",
            },
        }
        txt_path = os.path.join(tmp, "ok__query.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("hello")
        manifest_map[_paper_pair_key("10.1/ok", "query")]["text_path"] = txt_path

        plan = classify_alignment_papers(
            [("10.1/dead", "query"), ("10.1/ok", "query"), ("10.1/new", "target")],
            manifest_map,
            tmp,
            global_cache,
            retry_failed=False,
            no_cache=True,
        )
        assert len(plan.to_fetch) == 3
        assert plan.stats["to_fetch"] == 3


def test_partial_with_pdf_still_satisfied() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        papers_dir = tmp
        pdf_dir = os.path.join(papers_dir, "pdf")
        os.makedirs(pdf_dir)
        stem = "partial__query"
        pdf_path = os.path.join(pdf_dir, f"{stem}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4")
        row = {
            "paper_id": "10.1/partial",
            "source": "query",
            "status": "partial",
            "file_stem": stem,
            "pdf_path": pdf_path,
        }
        assert _manifest_row_satisfied(row, papers_dir) is True
        plan = classify_alignment_papers(
            [("10.1/partial", "query")],
            {_paper_pair_key("10.1/partial", "query"): row},
            papers_dir,
            {},
            retry_failed=False,
            no_cache=False,
        )
        assert plan.to_fetch == []
        assert plan.stats["satisfied"] == 1


def test_global_cache_persists_to_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logs_dir = os.path.join(tmp, "logs")
        cache = {}
        record_global_outcomes_from_rows(
            logs_dir,
            [{"paper_id": "10.1/x", "doi": "10.1/x", "status": "failed", "message": "m"}],
            cache,
        )
        path = os.path.join(logs_dir, GLOBAL_OUTCOME_CACHE_FILENAME)
        assert os.path.isfile(path)
        loaded = load_global_outcome_cache(logs_dir)
        assert "10.1/x" in loaded
        with open(path, encoding="utf-8") as f:
            line = json.loads(f.readline())
        assert line["status"] == "failed"


def test_docling_not_required_when_canonical_txt_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = os.path.join(tmp, "pdf")
        os.makedirs(pdf_dir)
        txt_path = os.path.join(tmp, "10.1_a__target.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("already have full text from pmc s3\n")
        pdf_path = os.path.join(pdf_dir, "10.1_a__target__pmc_oa_s3.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4")

        from auto_lit_search.download_manifest import (
            _infer_docling_required_basenames,
            _infer_docling_required_basenames_from_disk,
            _paper_has_usable_text,
        )

        assert _paper_has_usable_text(tmp, "10.1_a__target__pmc_oa_s3")
        assert _infer_docling_required_basenames_from_disk(tmp) == []


def test_manifest_docling_ignores_orphan_s3_pdfs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = os.path.join(tmp, "pdf")
        os.makedirs(pdf_dir)
        txt_path = os.path.join(tmp, "10.1_a__target.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("s3 text already present\n")
        pdf_path = os.path.join(pdf_dir, "10.1_a__target__pmc_oa_s3.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4")

        from auto_lit_search.download_manifest import _infer_docling_required_basenames

        manifest = {
            ("10.1/a", "target"): {
                "paper_id": "10.1/a",
                "source": "target",
                "file_stem": "10.1_a__target",
                "status": "ok",
                "selected_text_source": "pmc_oa_s3_txt",
                "pdf_docling_required": False,
                "text_path": txt_path,
                "pdf_path": pdf_path,
            }
        }
        assert _infer_docling_required_basenames(tmp, manifest) == []


def test_manifest_docling_queues_api_pdf_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_dir = os.path.join(tmp, "pdf")
        os.makedirs(pdf_dir)
        base = "10.1_a__target__unpaywall"
        pdf_path = os.path.join(pdf_dir, f"{base}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4")

        from auto_lit_search.download_manifest import _infer_docling_required_basenames

        manifest = {
            ("10.1/a", "target"): {
                "paper_id": "10.1/a",
                "source": "target",
                "file_stem": base,
                "status": "partial",
                "selected_text_source": "docling_pdf",
                "pdf_docling_required": True,
                "text_path": "",
                "pdf_path": pdf_path,
            }
        }
        assert _infer_docling_required_basenames(tmp, manifest) == [base]
