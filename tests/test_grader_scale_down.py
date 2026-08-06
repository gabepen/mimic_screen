"""Unit tests for cluster_respect grader scale-down helpers."""

from auto_lit_search.slurm_utils import (
    grader_scale_down_should_trigger,
    select_idle_grader_jobs_to_kill,
)


def test_trigger_when_remaining_at_or_below_threshold():
    assert not grader_scale_down_should_trigger(
        remaining_packets=21, respect_threshold=20
    )
    assert grader_scale_down_should_trigger(
        remaining_packets=20, respect_threshold=20
    )
    assert grader_scale_down_should_trigger(
        remaining_packets=0, respect_threshold=20
    )


def test_trigger_disabled_when_threshold_zero():
    assert not grader_scale_down_should_trigger(
        remaining_packets=0, respect_threshold=0
    )


def test_select_idle_prefers_higher_ports_and_keeps_one():
    urls = {
        9200: "http://a:9200",
        9201: "http://b:9201",
        9202: "http://c:9202",
        9203: "http://d:9203",
        9204: "http://e:9204",
    }
    specs = [
        {"port": p, "job_id": f"j{p}"}
        for p in (9200, 9201, 9202, 9203, 9204)
    ]
    inflight = {u: 0 for u in urls.values()}
    kill = select_idle_grader_jobs_to_kill(
        job_specs=specs,
        inflight_by_url=inflight,
        url_by_port=urls,
        n_kill=3,
        min_keep=1,
    )
    assert kill == ["j9204", "j9203", "j9202"]


def test_select_skips_busy_and_respects_min_keep():
    urls = {
        9200: "http://a:9200",
        9201: "http://b:9201",
        9202: "http://c:9202",
    }
    specs = [{"port": p, "job_id": f"j{p}"} for p in urls]
    inflight = {
        "http://a:9200": 0,
        "http://b:9201": 2,
        "http://c:9202": 0,
    }
    kill = select_idle_grader_jobs_to_kill(
        job_specs=specs,
        inflight_by_url=inflight,
        url_by_port=urls,
        n_kill=3,
        min_keep=1,
    )
    # Idle high-port first; busy 9201 skipped; keep ≥1 registered endpoint
    assert kill == ["j9202", "j9200"]


def test_select_noop_when_n_kill_zero_or_too_few_endpoints():
    urls = {9200: "http://a:9200"}
    specs = [{"port": 9200, "job_id": "j9200"}]
    inflight = {"http://a:9200": 0}
    assert (
        select_idle_grader_jobs_to_kill(
            job_specs=specs,
            inflight_by_url=inflight,
            url_by_port=urls,
            n_kill=2,
            min_keep=1,
        )
        == []
    )
    assert (
        select_idle_grader_jobs_to_kill(
            job_specs=specs,
            inflight_by_url=inflight,
            url_by_port=urls,
            n_kill=0,
            min_keep=1,
        )
        == []
    )
