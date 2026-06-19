#!/usr/bin/env python3
"""Monitor auto-lit pipeline progress from scheduler state and on-disk artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

STAGE_ORDER: Tuple[str, ...] = (
    "not_started",
    "collecting",
    "docling_queue",
    "docling",
    "grader_queue",
    "grading",
    "synthesis_queue",
    "synthesis",
    "done",
    "failed",
)

INFLIGHT_STAGES = frozenset(
    {"docling", "grading", "synthesis", "collecting", "docling_queue", "grader_queue", "synthesis_queue"}
)

STATE_TO_BUCKET: Dict[str, str] = {
    "DOWNLOADING": "collecting",
    "DOCLING_PENDING": "docling_queue",
    "DOCLING_INFLIGHT": "docling",
    "GRADER_READY": "grader_queue",
    "GRADER_INFLIGHT": "grading",
    "SYNTHESIS_READY": "synthesis_queue",
    "SYNTHESIS_INFLIGHT": "synthesis",
    "DONE": "done",
    "FAILED": "failed",
}

GRADER_DISCOVERY_RE = re.compile(
    r"grader discovery (\d+)/(\d+) endpoint\(s\) active, "
    r"(\d+) grading inflight(?:, (\d+) Slurm job\(s\) still pending)?"
)

DOWNLOAD_COMPLETE = "download_complete.json"
TOTAL_ALIGNMENTS_CACHE = "monitor_total_alignments.txt"


@dataclass
class AlignmentRecord:
    alignment_id: str
    stage: str
    scheduler_state: Optional[str] = None
    last_error: str = ""
    age_minutes: Optional[float] = None
    papers_dir: str = ""


@dataclass
class PipelineSnapshot:
    timestamp: float
    total_alignments: int
    stage_counts: Dict[str, int] = field(default_factory=dict)
    done: int = 0
    failed: int = 0
    in_flight: int = 0
    graded_count: int = 0
    results_count: int = 0
    scheduler_registered: int = 0
    papers_dirs: int = 0
    throughput_graded_per_hour: float = 0.0
    throughput_results_per_hour: float = 0.0
    throughput_registered_per_hour: float = 0.0
    eta_hours: Optional[float] = None
    elapsed_hours: Optional[float] = None
    synthesis_gpu_urls: List[str] = field(default_factory=list)
    grader_urls: List[str] = field(default_factory=list)
    grader_active: Optional[int] = None
    grader_target: Optional[int] = None
    grader_inflight_log: Optional[int] = None
    grader_pending_jobs: Optional[int] = None
    docling_inflight_cap: int = 1
    bottlenecks: List[str] = field(default_factory=list)
    error_clusters: List[Tuple[str, int]] = field(default_factory=list)
    stuck_samples: List[AlignmentRecord] = field(default_factory=list)
    delta_done_per_hour: Optional[float] = None
    delta_graded_per_hour: Optional[float] = None
    cpu_log_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "total_alignments": self.total_alignments,
            "done": self.done,
            "failed": self.failed,
            "in_flight": self.in_flight,
            "graded_count": self.graded_count,
            "results_count": self.results_count,
            "scheduler_registered": self.scheduler_registered,
            "papers_dirs": self.papers_dirs,
            "stage_counts": dict(self.stage_counts),
            "throughput": {
                "graded_per_hour": self.throughput_graded_per_hour,
                "results_per_hour": self.throughput_results_per_hour,
                "registered_per_hour": self.throughput_registered_per_hour,
                "delta_done_per_hour": self.delta_done_per_hour,
                "delta_graded_per_hour": self.delta_graded_per_hour,
            },
            "eta_hours": self.eta_hours,
            "elapsed_hours": self.elapsed_hours,
            "capacity": {
                "synthesis_gpu_urls": self.synthesis_gpu_urls,
                "grader_urls": self.grader_urls,
                "grader_active": self.grader_active,
                "grader_target": self.grader_target,
                "grader_inflight_log": self.grader_inflight_log,
                "grader_pending_jobs": self.grader_pending_jobs,
                "docling_inflight_cap": self.docling_inflight_cap,
            },
            "bottlenecks": self.bottlenecks,
            "error_clusters": [
                {"message_prefix": msg, "count": count}
                for msg, count in self.error_clusters
            ],
            "stuck_samples": [
                {
                    "alignment_id": r.alignment_id,
                    "stage": r.stage,
                    "scheduler_state": r.scheduler_state,
                    "age_minutes": r.age_minutes,
                    "last_error": r.last_error,
                }
                for r in self.stuck_samples
            ],
            "cpu_log_path": self.cpu_log_path,
        }


def alignment_id_for_pair(query_id: str, target: str) -> str:
    return f"{query_id}_{target}".replace("/", "_").replace(" ", "_")


def iter_alignment_ids_from_paper_ids(paper_ids_path: Path) -> Iterable[str]:
    with paper_ids_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return
    for query_id, rows in data.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            target = str(row.get("target") or "").strip()
            if not target:
                continue
            yield alignment_id_for_pair(str(query_id).strip(), target)


def count_alignments_cached(
    paper_ids_path: Path,
    cache_path: Path,
    *,
    force_refresh: bool = False,
) -> int:
    if (
        not force_refresh
        and cache_path.is_file()
        and paper_ids_path.is_file()
        and cache_path.stat().st_mtime >= paper_ids_path.stat().st_mtime
    ):
        try:
            return int(cache_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    total = sum(1 for _ in iter_alignment_ids_from_paper_ids(paper_ids_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(f"{total}\n", encoding="utf-8")
    return total


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def bucket_from_scheduler_state(
    raw_state: str,
    *,
    has_results: bool,
) -> str:
    if has_results:
        return "done"
    state = (raw_state or "").strip().upper()
    if state == "FAILED":
        return "failed"
    return STATE_TO_BUCKET.get(state, "collecting")


def inflight_age_minutes(state: Dict[str, Any], now: float) -> Optional[float]:
    bucket = STATE_TO_BUCKET.get(str(state.get("state") or "").strip().upper(), "")
    if bucket not in {"docling", "grading", "synthesis"}:
        return None
    key = {
        "docling": "docling_submitted_at",
        "grading": "grader_submitted_at",
        "synthesis": "synthesis_submitted_at",
    }[bucket]
    submitted = state.get(key)
    if submitted is None:
        submitted = state.get("updated_at")
    try:
        submitted_f = float(submitted)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - submitted_f) / 60.0)


def classify_alignment(
    alignment_id: str,
    *,
    scheduler_state: Optional[Dict[str, Any]],
    has_papers_dir: bool,
    has_results: bool,
    now: float,
) -> AlignmentRecord:
    last_error = ""
    scheduler_raw = ""
    papers_dir = ""
    age_minutes: Optional[float] = None

    if has_results:
        stage = "done"
    elif scheduler_state is not None:
        scheduler_raw = str(scheduler_state.get("state") or "")
        last_error = str(scheduler_state.get("last_error") or "").strip()
        papers_dir = str(scheduler_state.get("papers_dir") or "")
        stage = bucket_from_scheduler_state(scheduler_raw, has_results=False)
        age_minutes = inflight_age_minutes(scheduler_state, now)
    elif has_papers_dir:
        stage = "collecting"
    else:
        stage = "not_started"

    return AlignmentRecord(
        alignment_id=alignment_id,
        stage=stage,
        scheduler_state=scheduler_raw or None,
        last_error=last_error,
        age_minutes=age_minutes,
        papers_dir=papers_dir,
    )


def count_recent_artifacts(
    output_root: Path,
    suffix: str,
    *,
    window_seconds: float,
    now: float,
) -> int:
    cutoff = now - window_seconds
    count = 0
    if not output_root.is_dir():
        return 0
    for path in output_root.glob(f"*{suffix}"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime >= cutoff:
                count += 1
        except OSError:
            continue
    return count


def count_recent_scheduler_registrations(
    scheduler_dir: Path,
    *,
    window_seconds: float,
    now: float,
) -> int:
    cutoff = now - window_seconds
    count = 0
    if not scheduler_dir.is_dir():
        return 0
    for path in scheduler_dir.glob("*.json"):
        try:
            if path.stat().st_mtime >= cutoff:
                count += 1
        except OSError:
            continue
    return count


def parse_grader_discovery_line(line: str) -> Optional[Dict[str, int]]:
    m = GRADER_DISCOVERY_RE.search(line)
    if not m:
        return None
    out = {
        "active": int(m.group(1)),
        "target": int(m.group(2)),
        "inflight": int(m.group(3)),
    }
    if m.group(4) is not None:
        out["pending_jobs"] = int(m.group(4))
    return out


def read_endpoint_urls(path: Path) -> List[str]:
    if not path.is_file():
        return []
    urls: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            urls.append(line)
    return urls


def tail_find_grader_discovery(cpu_log: Path, *, max_bytes: int = 2_000_000) -> Optional[Dict[str, int]]:
    if not cpu_log.is_file():
        return None
    try:
        size = cpu_log.stat().st_size
        with cpu_log.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        parsed = parse_grader_discovery_line(line)
        if parsed is not None:
            return parsed
    return None


def infer_pipeline_start(
    scheduler_dir: Path,
    cpu_log: Optional[Path],
) -> Optional[float]:
    earliest: Optional[float] = None
    if scheduler_dir.is_dir():
        for path in scheduler_dir.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            earliest = mtime if earliest is None else min(earliest, mtime)
    if cpu_log and cpu_log.is_file():
        try:
            mtime = cpu_log.stat().st_mtime
            # Prefer log birth: use min of first lines timestamp if parseable
            earliest = mtime if earliest is None else min(earliest, mtime)
        except OSError:
            pass
    return earliest


def error_prefix(msg: str, max_len: int = 120) -> str:
    msg = " ".join(msg.split())
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 3] + "..."


def rank_bottlenecks(
    stage_counts: Dict[str, int],
    *,
    total: int,
    grader_queue: int,
    grader_inflight: int,
    grader_active: Optional[int],
    synthesis_queue: int,
    synthesis_gpus: int,
    docling_queue: int,
    docling_cap: int,
    collecting: int,
    not_started: int,
    done: int,
    results_per_hour: float,
) -> List[str]:
    hints: List[Tuple[int, str]] = []
    if total <= 0:
        return []

    def add(score: int, text: str) -> None:
        if score > 0:
            hints.append((score, text))

    if grader_queue >= 5 and (grader_inflight or 0) >= max(1, (grader_active or 1)):
        add(
            grader_queue + grader_inflight,
            f"Grading backlog: {grader_queue} queued, {grader_inflight} inflight "
            f"({grader_active or '?'} grader endpoints active)",
        )
    elif grader_queue >= 10:
        add(grader_queue, f"Grader queue large ({grader_queue} waiting for a grader slot)")

    if docling_queue >= 5 and docling_cap <= 1:
        add(
            docling_queue * 2,
            f"Docling serial bottleneck: {docling_queue} queued (DOCLING_INFLIGHT_CAP={docling_cap})",
        )
    elif docling_queue >= 3:
        add(docling_queue, f"Docling queue building ({docling_queue} pending)")

    if synthesis_queue >= 3 and synthesis_gpus <= 1:
        add(
            synthesis_queue * 2,
            f"Synthesis backlog: {synthesis_queue} queued with only {synthesis_gpus} GPU URL(s)",
        )
    elif synthesis_queue >= 5:
        add(synthesis_queue, f"Synthesis queue large ({synthesis_queue} waiting)")

    backlog_collect = collecting + not_started
    if backlog_collect > max(done, 1) and results_per_hour < 1.0:
        add(
            backlog_collect,
            f"Download/collect still feeding pipeline ({collecting} collecting, "
            f"{not_started} not started)",
        )

    hints.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in hints[:5]]


def build_snapshot(
    *,
    data_root: Path,
    output_root: Path,
    paper_ids_path: Path,
    scheduler_dir: Path,
    papers_root: Path,
    logs_dir: Path,
    cpu_log: Optional[Path],
    window_minutes: float,
    stuck_minutes: float,
    docling_inflight_cap: int,
    force_count_refresh: bool = False,
    prev_snapshot: Optional[PipelineSnapshot] = None,
) -> PipelineSnapshot:
    now = time.time()
    window_seconds = window_minutes * 60.0

    cache_path = logs_dir / TOTAL_ALIGNMENTS_CACHE
    total = count_alignments_cached(
        paper_ids_path, cache_path, force_refresh=force_count_refresh
    )

    scheduler_states: Dict[str, Dict[str, Any]] = {}
    if scheduler_dir.is_dir():
        for path in scheduler_dir.glob("*.json"):
            aid = path.stem
            state = _load_json(path)
            if state is not None:
                scheduler_states[aid] = state

    papers_dirs: set[str] = set()
    if papers_root.is_dir():
        for entry in papers_root.iterdir():
            if entry.is_dir():
                papers_dirs.add(entry.name)

    results_ids: set[str] = set()
    graded_ids: set[str] = set()
    if output_root.is_dir():
        for path in output_root.glob("*_results.json"):
            if path.is_file():
                results_ids.add(path.name[: -len("_results.json")])
        for path in output_root.glob("*_graded.json"):
            if path.is_file():
                graded_ids.add(path.name[: -len("_graded.json")])

    all_ids: set[str] = set()
    if paper_ids_path.is_file():
        all_ids.update(iter_alignment_ids_from_paper_ids(paper_ids_path))
    all_ids.update(scheduler_states.keys())
    all_ids.update(papers_dirs)
    all_ids.update(results_ids)
    all_ids.update(graded_ids)

    records: List[AlignmentRecord] = []
    for aid in all_ids:
        records.append(
            classify_alignment(
                aid,
                scheduler_state=scheduler_states.get(aid),
                has_papers_dir=aid in papers_dirs,
                has_results=aid in results_ids,
                now=now,
            )
        )

    stage_counts = Counter(r.stage for r in records)
    for stage in STAGE_ORDER:
        stage_counts.setdefault(stage, 0)

    done = stage_counts.get("done", 0)
    failed = stage_counts.get("failed", 0)
    in_flight = sum(stage_counts.get(s, 0) for s in STAGE_ORDER if s not in {"done", "failed", "not_started"})

    graded_recent = count_recent_artifacts(
        output_root, "_graded.json", window_seconds=window_seconds, now=now
    )
    results_recent = count_recent_artifacts(
        output_root, "_results.json", window_seconds=window_seconds, now=now
    )
    registered_recent = count_recent_scheduler_registrations(
        scheduler_dir, window_seconds=window_seconds, now=now
    )
    hours_window = window_minutes / 60.0
    throughput_graded = graded_recent / hours_window if hours_window > 0 else 0.0
    throughput_results = results_recent / hours_window if hours_window > 0 else 0.0
    throughput_registered = registered_recent / hours_window if hours_window > 0 else 0.0

    eta_hours: Optional[float] = None
    if throughput_results > 0 and total > done:
        eta_hours = (total - done) / throughput_results

    start = infer_pipeline_start(scheduler_dir, cpu_log)
    elapsed_hours = (now - start) / 3600.0 if start is not None else None

    gpu_urls = read_endpoint_urls(logs_dir / "gpu_endpoints_discovered.txt")
    grader_urls = read_endpoint_urls(logs_dir / "grader_endpoints_discovered.txt")
    grader_log = tail_find_grader_discovery(cpu_log) if cpu_log else None

    error_counter: Counter[str] = Counter()
    stuck: List[AlignmentRecord] = []
    for rec in records:
        if rec.stage == "failed" and rec.last_error:
            error_counter[error_prefix(rec.last_error)] += 1
        if rec.age_minutes is not None and rec.age_minutes >= stuck_minutes:
            stuck.append(rec)
    stuck.sort(key=lambda r: r.age_minutes or 0.0, reverse=True)

    bottlenecks = rank_bottlenecks(
        dict(stage_counts),
        total=total,
        grader_queue=stage_counts.get("grader_queue", 0),
        grader_inflight=stage_counts.get("grading", 0),
        grader_active=(grader_log or {}).get("active"),
        synthesis_queue=stage_counts.get("synthesis_queue", 0),
        synthesis_gpus=max(len(gpu_urls), 1) if gpu_urls else 0,
        docling_queue=stage_counts.get("docling_queue", 0),
        docling_cap=docling_inflight_cap,
        collecting=stage_counts.get("collecting", 0),
        not_started=stage_counts.get("not_started", 0),
        done=done,
        results_per_hour=throughput_results,
    )

    snap = PipelineSnapshot(
        timestamp=now,
        total_alignments=total,
        stage_counts=dict(stage_counts),
        done=done,
        failed=failed,
        in_flight=in_flight,
        graded_count=len(graded_ids),
        results_count=len(results_ids),
        scheduler_registered=len(scheduler_states),
        papers_dirs=len(papers_dirs),
        throughput_graded_per_hour=throughput_graded,
        throughput_results_per_hour=throughput_results,
        throughput_registered_per_hour=throughput_registered,
        eta_hours=eta_hours,
        elapsed_hours=elapsed_hours,
        synthesis_gpu_urls=gpu_urls,
        grader_urls=grader_urls,
        grader_active=(grader_log or {}).get("active"),
        grader_target=(grader_log or {}).get("target"),
        grader_inflight_log=(grader_log or {}).get("inflight"),
        grader_pending_jobs=(grader_log or {}).get("pending_jobs"),
        docling_inflight_cap=docling_inflight_cap,
        bottlenecks=bottlenecks,
        error_clusters=error_counter.most_common(3),
        stuck_samples=stuck[:10],
        cpu_log_path=str(cpu_log) if cpu_log else "",
    )

    if prev_snapshot is not None:
        dt_hours = (snap.timestamp - prev_snapshot.timestamp) / 3600.0
        if dt_hours > 0:
            snap.delta_done_per_hour = (snap.done - prev_snapshot.done) / dt_hours
            snap.delta_graded_per_hour = (
                snap.graded_count - prev_snapshot.graded_count
            ) / dt_hours

    return snap


def find_latest_cpu_log(logs_dir: Path) -> Optional[Path]:
    if not logs_dir.is_dir():
        return None
    candidates = list(logs_dir.glob("auto_lit_cpu_*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def format_dashboard(
    snap: PipelineSnapshot, *, window_minutes: float, stuck_minutes: float = 90.0
) -> str:
    lines: List[str] = []
    ts = datetime.fromtimestamp(snap.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    pct = (100.0 * snap.done / snap.total_alignments) if snap.total_alignments else 0.0
    lines.append(f"Auto-lit pipeline monitor  |  {ts}")
    lines.append("=" * 72)
    lines.append(
        f"Progress: {snap.done}/{snap.total_alignments} done ({pct:.1f}%)  |  "
        f"failed={snap.failed}  in_flight={snap.in_flight}  "
        f"graded={snap.graded_count}  registered={snap.scheduler_registered}"
    )
    if snap.elapsed_hours is not None:
        lines.append(f"Elapsed: {snap.elapsed_hours:.1f}h", )
    if snap.eta_hours is not None:
        lines.append(f"ETA (at {window_minutes:.0f}m results rate): {snap.eta_hours:.1f}h")
    lines.append("")
    lines.append("Stage counts:")
    for stage in STAGE_ORDER:
        count = snap.stage_counts.get(stage, 0)
        if count == 0 and stage not in {"done", "failed", "not_started"}:
            continue
        stage_pct = (100.0 * count / snap.total_alignments) if snap.total_alignments else 0.0
        bar = "#" * min(40, int(stage_pct / 2.5))
        lines.append(f"  {stage:16s} {count:6d}  ({stage_pct:5.1f}%)  {bar}")
    lines.append("")
    lines.append(
        f"Throughput (last {window_minutes:.0f}m): "
        f"graded {snap.throughput_graded_per_hour:.1f}/hr  "
        f"results {snap.throughput_results_per_hour:.1f}/hr  "
        f"registered {snap.throughput_registered_per_hour:.1f}/hr"
    )
    if snap.delta_done_per_hour is not None:
        lines.append(
            f"Watch delta: done {snap.delta_done_per_hour:.1f}/hr  "
            f"graded {snap.delta_graded_per_hour or 0:.1f}/hr"
        )
    lines.append("")
    lines.append("Capacity:")
    lines.append(f"  synthesis GPUs: {len(snap.synthesis_gpu_urls)}  {snap.synthesis_gpu_urls}")
    lines.append(f"  grader URLs:    {len(snap.grader_urls)}  (file discovery)")
    if snap.grader_active is not None:
        pending = (
            f", {snap.grader_pending_jobs} Slurm pending"
            if snap.grader_pending_jobs is not None
            else ""
        )
        lines.append(
            f"  grader log:     {snap.grader_active}/{snap.grader_target} active, "
            f"{snap.grader_inflight_log} grading inflight{pending}"
        )
    lines.append(f"  docling cap:    DOCLING_INFLIGHT_CAP={snap.docling_inflight_cap}")
    if snap.bottlenecks:
        lines.append("")
        lines.append("Bottleneck hints:")
        for hint in snap.bottlenecks:
            lines.append(f"  - {hint}")
    if snap.error_clusters:
        lines.append("")
        lines.append("Top errors:")
        for msg, count in snap.error_clusters:
            lines.append(f"  - ({count}x) {msg}")
    if snap.stuck_samples:
        lines.append("")
        lines.append(f"Stuck inflight (>{stuck_minutes:.0f} min):")
        for rec in snap.stuck_samples:
            err = f"  err={rec.last_error[:60]}..." if rec.last_error else ""
            lines.append(
                f"  {rec.alignment_id:24s} {rec.stage:12s} "
                f"{rec.age_minutes:.0f}m{err}"
            )
    if snap.cpu_log_path:
        lines.append("")
        lines.append(f"CPU log: {snap.cpu_log_path}")
    return "\n".join(lines)


def write_csv(snap: PipelineSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["timestamp", snap.timestamp])
        writer.writerow(["total_alignments", snap.total_alignments])
        writer.writerow(["done", snap.done])
        writer.writerow(["failed", snap.failed])
        writer.writerow(["in_flight", snap.in_flight])
        writer.writerow(["graded_count", snap.graded_count])
        writer.writerow(["results_count", snap.results_count])
        for stage in STAGE_ORDER:
            writer.writerow([f"stage_{stage}", snap.stage_counts.get(stage, 0)])
        writer.writerow(["throughput_graded_per_hour", snap.throughput_graded_per_hour])
        writer.writerow(["throughput_results_per_hour", snap.throughput_results_per_hour])
        writer.writerow(["eta_hours", snap.eta_hours or ""])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/private/groups/corbettlab/gabe/auto_lit_eval_data"),
    )
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--paper-ids", type=Path, default=None)
    p.add_argument("--scheduler-state-dir", type=Path, default=None)
    p.add_argument("--papers-root", type=Path, default=None)
    p.add_argument("--logs-dir", type=Path, default=None)
    p.add_argument("--cpu-log", type=Path, default=None)
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    p.add_argument("--json", dest="json_path", type=Path, default=None)
    p.add_argument("--csv", dest="csv_path", type=Path, default=None)
    p.add_argument("--stuck-minutes", type=float, default=90.0)
    p.add_argument("--window-minutes", type=float, default=60.0)
    p.add_argument(
        "--docling-inflight-cap",
        type=int,
        default=int(os.environ.get("DOCLING_INFLIGHT_CAP", "1")),
    )
    p.add_argument("--refresh-total", action="store_true")
    return p.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    data_root = args.data_root
    logs_dir = args.logs_dir or (data_root / "logs")
    return {
        "data_root": data_root,
        "output_root": args.output_root or (data_root / "llm_results"),
        "paper_ids": args.paper_ids
        or (data_root / "search_results" / "lp-human-all_search.json"),
        "scheduler_dir": args.scheduler_state_dir or (logs_dir / "scheduler_state"),
        "papers_root": args.papers_root or (data_root / "papers"),
        "logs_dir": logs_dir,
        "cpu_log": args.cpu_log or find_latest_cpu_log(logs_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    prev: Optional[PipelineSnapshot] = None
    cache_file = Path(f"/tmp/monitor_pipeline_{os.getuid()}.json")

    def one_pass() -> PipelineSnapshot:
        nonlocal prev
        snap = build_snapshot(
            data_root=paths["data_root"],
            output_root=paths["output_root"],
            paper_ids_path=paths["paper_ids"],
            scheduler_dir=paths["scheduler_dir"],
            papers_root=paths["papers_root"],
            logs_dir=paths["logs_dir"],
            cpu_log=paths["cpu_log"],
            window_minutes=args.window_minutes,
            stuck_minutes=args.stuck_minutes,
            docling_inflight_cap=args.docling_inflight_cap,
            force_count_refresh=args.refresh_total,
            prev_snapshot=prev,
        )
        prev = snap
        if args.watch is not None:
            try:
                cache_file.write_text(json.dumps(snap.to_dict()), encoding="utf-8")
            except OSError:
                pass
        return snap

    if args.watch is not None:
        if cache_file.is_file():
            try:
                prev_data = json.loads(cache_file.read_text(encoding="utf-8"))
                prev = PipelineSnapshot(
                    timestamp=float(prev_data.get("timestamp", 0)),
                    total_alignments=int(prev_data.get("total_alignments", 0)),
                    done=int(prev_data.get("done", 0)),
                    graded_count=int(prev_data.get("graded_count", 0)),
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                prev = None
        while True:
            snap = one_pass()
            if args.json_path:
                args.json_path.write_text(
                    json.dumps(snap.to_dict(), indent=2), encoding="utf-8"
                )
            if args.csv_path:
                write_csv(snap, args.csv_path)
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(
                format_dashboard(
                    snap,
                    window_minutes=args.window_minutes,
                    stuck_minutes=args.stuck_minutes,
                )
            )
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.watch))
    else:
        snap = one_pass()
        if args.json_path:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(
                json.dumps(snap.to_dict(), indent=2), encoding="utf-8"
            )
        if args.csv_path:
            write_csv(snap, args.csv_path)
        print(
            format_dashboard(
                snap,
                window_minutes=args.window_minutes,
                stuck_minutes=args.stuck_minutes,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
