"""Unit tests for PMC OA S3 -> Europe PMC REST cascade in collect."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from auto_lit_search.collect import (
    CollectionContext,
    CollectThrottle,
    UCSCEmailOnlyProvider,
    _attempt_pmc_oa_s3,
)


def _context(tmpdir: str, disable_pmc_oa_s3: bool = False) -> CollectionContext:
    pdf_dir = os.path.join(tmpdir, "pdf")
    xml_dir = os.path.join(tmpdir, "text_xml")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    return CollectionContext(
        session=MagicMock(),
        pmcid_cache={},
        pdf_dir=pdf_dir,
        text_dir=tmpdir,
        xml_dir=xml_dir,
        force_pdfs=True,
        throttle=CollectThrottle(),
        disable_pmc_oa_s3=disable_pmc_oa_s3,
    )


def test_attempt_pmc_oa_s3_writes_txt_on_quality_pass() -> None:
    good_text = (
        "Abstract\nIntroduction\nMethods\nResults\nDiscussion\nConclusion\n"
        + ("x" * 3000)
    )
    meta = MagicMock()
    meta.pmcid = "PMC123"
    meta.version = 1
    meta.text_https_url = "https://example/txt"
    meta.xml_https_url = None
    meta.pdf_https_url = None

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _context(tmp)
        with (
            patch(
                "auto_lit_search.pmc_oa_s3.fetch_pmc_oa_metadata",
                return_value=meta,
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_fulltext",
                return_value=(good_text, "txt"),
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_pdf",
                return_value=None,
            ),
        ):
            out = _attempt_pmc_oa_s3("PMC123", ctx, "10.1_test__query")

        assert out["attempt"]["success"] is True
        assert out["attempt"]["artifact"] == "txt"
        assert out["text_path"] is not None
        assert os.path.isfile(out["text_path"])
        assert out["selected_text_source"] == "pmc_oa_s3_txt"


@patch("auto_lit_search.collect._fetch_fulltext_pdf", return_value=None)
@patch("auto_lit_search.collect._fetch_fulltext_xml")
@patch("auto_lit_search.collect._resolve_to_pmcid", return_value=("PMC123", "DOI:10.1/a"))
def test_s3_success_skips_europe_pmc_rest_xml(
    mock_resolve: MagicMock,
    mock_xml: MagicMock,
    mock_pdf: MagicMock,
) -> None:
    good_text = (
        "Abstract\nIntroduction\nMethods\nResults\nDiscussion\nConclusion\n"
        + ("x" * 3000)
    )
    meta = MagicMock()
    meta.pmcid = "PMC123"
    meta.version = 1
    meta.pdf_https_url = None

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _context(tmp)
        provider = UCSCEmailOnlyProvider(collector_email="test@ucsc.edu")
        with (
            patch(
                "auto_lit_search.pmc_oa_s3.fetch_pmc_oa_metadata",
                return_value=meta,
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_fulltext",
                return_value=(good_text, "txt"),
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_pdf",
                return_value=None,
            ),
        ):
            rec = provider.resolve_and_fetch("10.1/a", "query", ctx)

    mock_xml.assert_not_called()
    assert rec.status == "ok"
    assert rec.text_path is not None
    assert (rec.details or {}).get("selected_text_source") == "pmc_oa_s3_txt"
    assert (rec.details or {}).get("source_attempts", {}).get("pmc_oa_s3", {}).get(
        "success"
    )


@patch("auto_lit_search.collect._fetch_fulltext_pdf", return_value=None)
@patch("auto_lit_search.collect._fetch_fulltext_xml", return_value=None)
@patch("auto_lit_search.collect._resolve_to_pmcid", return_value=("PMC123", "DOI:10.1/a"))
def test_s3_miss_falls_back_to_europe_pmc_rest(
    mock_resolve: MagicMock,
    mock_xml: MagicMock,
    mock_pdf: MagicMock,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _context(tmp)
        provider = UCSCEmailOnlyProvider(collector_email="test@ucsc.edu")
        with patch(
            "auto_lit_search.pmc_oa_s3.fetch_pmc_oa_metadata",
            return_value=None,
        ):
            rec = provider.resolve_and_fetch("10.1/a", "query", ctx)

    mock_xml.assert_called_once()
    assert (rec.details or {}).get("source_attempts", {}).get("pmc_oa_s3", {}).get(
        "attempted"
    )


@patch("auto_lit_search.collect._fetch_fulltext_pdf", return_value=None)
@patch("auto_lit_search.collect._fetch_fulltext_xml", return_value=None)
@patch("auto_lit_search.collect._resolve_to_pmcid", return_value=("PMC123", "DOI:10.1/a"))
def test_s3_low_quality_txt_falls_back_to_rest(
    mock_resolve: MagicMock,
    mock_xml: MagicMock,
    mock_pdf: MagicMock,
) -> None:
    meta = MagicMock()
    meta.pmcid = "PMC123"
    meta.version = 1
    meta.pdf_https_url = None

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _context(tmp)
        provider = UCSCEmailOnlyProvider(collector_email="test@ucsc.edu")
        with (
            patch(
                "auto_lit_search.pmc_oa_s3.fetch_pmc_oa_metadata",
                return_value=meta,
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_fulltext",
                return_value=("short", "txt"),
            ),
            patch(
                "auto_lit_search.pmc_oa_s3.download_pmc_oa_pdf",
                return_value=None,
            ),
        ):
            provider.resolve_and_fetch("10.1/a", "query", ctx)

    mock_xml.assert_called_once()


@patch("auto_lit_search.collect._attempt_pmc_oa_s3")
@patch("auto_lit_search.collect._fetch_fulltext_pdf", return_value=None)
@patch("auto_lit_search.collect._fetch_fulltext_xml", return_value=None)
@patch("auto_lit_search.collect._resolve_to_pmcid", return_value=("PMC123", "DOI:10.1/a"))
def test_disable_pmc_oa_s3_skips_s3_attempt(
    mock_resolve: MagicMock,
    mock_xml: MagicMock,
    mock_pdf: MagicMock,
    mock_s3: MagicMock,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _context(tmp, disable_pmc_oa_s3=True)
        provider = UCSCEmailOnlyProvider(collector_email="test@ucsc.edu")
        provider.resolve_and_fetch("10.1/a", "query", ctx)

    mock_s3.assert_not_called()
    mock_xml.assert_called_once()
