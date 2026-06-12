"""Tests for multi-grader URL bookkeeping in download_node."""

from auto_lit_search.download_node import (
    _prune_grader_pending_specs,
    _registered_grader_ports,
)


def test_registered_grader_ports() -> None:
    urls = [
        "http://phoenix-00:9200",
        "http://phoenix-01:9201",
        "http://phoenix-02:9202",
    ]
    assert _registered_grader_ports(urls) == {9200, 9201, 9202}


def test_prune_grader_pending_specs() -> None:
    pending = [
        {"job_id": "1", "port": 9200, "failed": False},
        {"job_id": "2", "port": 9201, "failed": False},
        {"job_id": "3", "port": 9202, "failed": False},
    ]
    registered = ["http://phoenix-00:9200", "http://phoenix-01:9201"]
    removed = _prune_grader_pending_specs(pending, registered)
    assert removed == 2
    assert len(pending) == 1
    assert pending[0]["port"] == 9202
