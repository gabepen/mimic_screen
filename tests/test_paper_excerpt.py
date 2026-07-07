"""Tests for section-aware grader paper excerpts."""

from __future__ import annotations

import os

import pytest

from auto_lit_search.paper_excerpt import (
    build_grader_excerpt,
    build_grader_excerpt_with_meta,
)


def _section_doc(
    *,
    preamble: str = "Title line\nJournal line\n",
    abstract: str = "A" * 200,
    introduction: str = "I" * 5000,
    results: str = "R" * 20000,
    methods: str = "M" * 20000,
    discussion: str = "D" * 15000,
    references: str = "Ref " * 8000,
) -> str:
    return (
        f"{preamble}"
        f"Abstract\n{abstract}\n\n"
        f"Introduction\n{introduction}\n\n"
        f"Results\n{results}\n\n"
        f"Methods\n{methods}\n\n"
        f"Discussion\n{discussion}\n\n"
        f"References\n{references}\n"
    )


def test_short_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "50000")
    text = "Short paper body."
    meta = build_grader_excerpt_with_meta(text)
    assert meta.excerpt == text
    assert meta.section_trim_applied is False


def test_drops_references_when_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "1000")
    monkeypatch.setenv("GRADER_EXCERPT_MAX_CHARS", "80000")
    text = _section_doc(references="Smith et al. 2020\n" * 500)
    excerpt = build_grader_excerpt(text)
    assert "References" not in excerpt
    assert "Smith et al. 2020" not in excerpt
    assert "Results" in excerpt
    assert "Methods" in excerpt


def test_over_budget_trims_introduction_before_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "1000")
    monkeypatch.setenv("GRADER_EXCERPT_MAX_CHARS", "30000")
    text = _section_doc(
        introduction="I" * 25000,
        results="UNIQUE_RESULTS_MARKER " + ("R" * 12000),
        methods="M" * 8000,
        discussion="D" * 8000,
        references="Ref\n" * 2000,
    )
    meta = build_grader_excerpt_with_meta(text)
    assert meta.references_dropped is True
    assert meta.intro_trimmed is True
    assert "UNIQUE_RESULTS_MARKER" in meta.excerpt
    assert "Introduction" not in meta.excerpt or "I" * 100 not in meta.excerpt


def test_nature_order_results_before_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "1000")
    monkeypatch.setenv("GRADER_EXCERPT_MAX_CHARS", "80000")
    text = (
        "Title\n\nAbstract\nShort abstract.\n\n"
        "Results\nNature-order results " + ("x" * 5000) + "\n\n"
        "Discussion\nSome discussion.\n\n"
        "Methods\nNature-order methods " + ("y" * 5000) + "\n\n"
        "References\n1. ref\n"
    )
    excerpt = build_grader_excerpt(text)
    res_pos = excerpt.find("Nature-order results")
    meth_pos = excerpt.find("Nature-order methods")
    assert res_pos != -1
    assert meth_pos != -1
    assert res_pos < meth_pos
    assert "References" not in excerpt


def test_no_headers_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "1000")
    monkeypatch.setenv("GRADER_EXCERPT_MAX_CHARS", "80000")
    text = "no section headers here\n" + ("z" * 90000)
    meta = build_grader_excerpt_with_meta(text)
    assert meta.section_trim_applied is True
    assert "[... omitted ...]" in meta.excerpt
    assert len(meta.excerpt) <= 80000 + len("\n\n[excerpt truncated for grading]")


def test_enforces_80k_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADER_EXCERPT_LONG_THRESHOLD", "1000")
    monkeypatch.setenv("GRADER_EXCERPT_MAX_CHARS", "80000")
    text = _section_doc(
        introduction="I" * 30000,
        results="R" * 40000,
        methods="M" * 40000,
        discussion="D" * 40000,
        references="Ref\n" * 5000,
    )
    meta = build_grader_excerpt_with_meta(text)
    assert meta.excerpt_char_len <= 80000 + len("\n\n[excerpt truncated for grading]")
