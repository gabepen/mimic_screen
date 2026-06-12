"""Unit tests for PMC OA S3 URL helpers."""

from __future__ import annotations

from auto_lit_search.pmc_oa_s3 import (
    article_object_https_url,
    metadata_https_url,
    normalize_pmcid,
    s3_uri_to_https_url,
)


def test_normalize_pmcid() -> None:
    assert normalize_pmcid("PMC12345") == "PMC12345"
    assert normalize_pmcid("pmc:12345") == "PMC12345"
    assert normalize_pmcid("12345") == "PMC12345"


def test_s3_uri_to_https_url() -> None:
    uri = "s3://pmc-oa-opendata/PMC10009402.1/PMC10009402.1.txt?md5=abc"
    assert (
        s3_uri_to_https_url(uri)
        == "https://pmc-oa-opendata.s3.amazonaws.com/PMC10009402.1/PMC10009402.1.txt?md5=abc"
    )


def test_metadata_and_object_urls() -> None:
    assert (
        metadata_https_url("PMC10009402", 1)
        == "https://pmc-oa-opendata.s3.amazonaws.com/metadata/PMC10009402.1.json"
    )
    assert (
        article_object_https_url("PMC10009402", 1, "txt")
        == "https://pmc-oa-opendata.s3.amazonaws.com/PMC10009402.1/PMC10009402.1.txt"
    )
