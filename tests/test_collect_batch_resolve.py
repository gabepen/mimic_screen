"""Tests for phased PMCID batch resolve and S3-first collect."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from auto_lit_search.collect import (
    CollectThrottle,
    _collect_s3_record,
    _download_papers_phased,
    _pmcid_epmc_or_batch,
    _pmcid_idconv_batch,
    batch_resolve_pmcids,
    load_pmcid_cache,
    save_pmcid_cache,
)
from tests.test_collect_pmc_oa_s3_cascade import _context


def test_pmcid_idconv_batch_parses_records() -> None:
    session = MagicMock()
    session.get.return_value.json.return_value = {
        "records": [
            {
                "requested-id": "10.1/a",
                "pmcid": "PMC111",
                "pmid": 123,
            }
        ]
    }
    session.get.return_value.raise_for_status = MagicMock()
    out = _pmcid_idconv_batch(["10.1/a"], session, "test@ucsc.edu")
    assert out["10.1/a"][0] == "PMC111"
    assert out["10.1/a"][1] == "123"


def test_pmcid_epmc_or_batch_maps_doi() -> None:
    session = MagicMock()
    session.get.return_value.json.return_value = {
        "resultList": {
            "result": [
                {
                    "pmcid": "PMC222",
                    "pmid": "999",
                    "doi": "10.2/b",
                    "pubTypeList": {"pubType": ["research-article"]},
                }
            ]
        }
    }
    session.get.return_value.raise_for_status = MagicMock()
    cache: dict = {}
    n = _pmcid_epmc_or_batch(
        ["10.2/b"],
        session,
        cache,
        CollectThrottle(),
        None,
    )
    assert n == 1
    assert cache["10.2/b"] == "PMC222"


def test_batch_resolve_uses_idconv_then_epmc() -> None:
    cache: dict = {}
    session = MagicMock()

    def fake_idconv(ids, sess, email):
        return {"10.1/a": ("PMC1", "1")}

    def fake_epmc(ids, sess, c, throttle, lock):
        c["10.1/b"] = "PMC2"
        _EUROPEPMC_LAST = None
        return 1

    with (
        patch(
            "auto_lit_search.collect._pmcid_idconv_batch",
            side_effect=fake_idconv,
        ),
        patch(
            "auto_lit_search.collect._pmcid_epmc_or_batch",
            side_effect=fake_epmc,
        ),
    ):
        batch_resolve_pmcids(
            ["10.1/a", "10.1/b"],
            cache,
            session,
            CollectThrottle(),
            None,
            collector_email="t@ucsc.edu",
        )
    assert cache["10.1/a"] == "PMC1"
    assert cache["10.1/b"] == "PMC2"


def test_pmcid_cache_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pmcid_cache.json")
        save_pmcid_cache(path, {"10.1/a": "PMC1", "10.1/b": None})
        loaded = load_pmcid_cache(path)
        assert loaded["10.1/a"] == "PMC1"
        assert loaded["10.1/b"] is None


@patch("auto_lit_search.collect._collect_single_record")
@patch("auto_lit_search.collect.batch_resolve_pmcids", return_value=2)
def test_phased_download_s3_then_fallback(
    mock_batch: MagicMock,
    mock_fallback: MagicMock,
) -> None:
    from auto_lit_search.collect import DownloadRecord, UCSCEmailOnlyProvider

    good_text = "Abstract\n" + ("x" * 3000)
    meta = MagicMock()
    meta.pmcid = "PMC123"
    meta.version = 1
    meta.text_https_url = None
    meta.xml_https_url = None
    meta.pdf_https_url = None

    fallback_rec = DownloadRecord(
        paper_id="10.1/miss",
        source="query",
        pmcid=None,
        pdf_path=None,
        text_path="/tmp/x.txt",
        status="ok",
    )
    mock_fallback.return_value = fallback_rec

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_ROOT"] = tmp
        os.environ["COLLECT_S3_WORKERS"] = "2"
        os.environ["COLLECT_FALLBACK_WORKERS"] = "1"
        ctx = _context(tmp)
        ctx.pmcid_cache = {
            "10.1/hit": "PMC123",
            "10.1/miss": None,
        }
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
            recs = _download_papers_phased(
                [("10.1/hit", "query"), ("10.1/miss", "target")],
                tmp,
                MagicMock(),
                ctx.pmcid_cache,
                provider,
                no_cache=False,
                force_pdfs=True,
                prefer_pdf_text=True,
                delete_pdf_after_text=False,
                disable_semantic_scholar=True,
                collector_email="test@ucsc.edu",
                max_workers=2,
            )
        assert len(recs) == 2
        mock_fallback.assert_called_once()
        hit = next(r for r in recs if r.paper_id == "10.1/hit")
        assert hit.text_path is not None
