#!/usr/bin/env python3
"""Generate a pipeline status report (markdown + JSON) from scheduler state and artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STATE_TO_BUCKET = {
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


def _count_artifacts(root: Path, suffix: str, hours: float, now: float) -> int:
    cutoff = now - hours * 3600
    return sum(1 for p in root.glob(f"*{suffix}") if p.stat().st_mtime >= cutoff)


def _load_total_alignments(paper_ids_path: Path) -> int:
    if not paper_ids_path.is_file():
        return 0
    data = json.loads(paper_ids_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return 0
    return sum(len(v) for v in data.values() if isinstance(v, list))


def _categorize_failed_error(err: str) -> str:
    el = (err or "").lower()
    if not err:
        return "empty"
    if "watchdog" in el:
        return "watchdog"
    if "no pdfs converted" in el:
        return "docling_no_pdf"
    if "docling" in el:
        return "docling_other"
    if "synthesis" in el or "papers_root" in el or "sidecar" in el:
        return "synthesis"
    if "grader" in el:
        return "grader"
    if "collect" in el or "download" in el or "no text" in el:
        return "collect"
    return "other"


def _find_cpu_log(logs_dir: Path) -> Optional[Path]:
    candidates = sorted(logs_dir.glob("auto_lit_cpu_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _failed_dynamics(cpu_log: Optional[Path], failed_state_files: List[Path], now: float) -> Dict[str, Any]:
    recent_lines = 5000
    out: Dict[str, Any] = {
        "reconcile_failed_to_synth_ready": 0,
        "recent_log_docling_fail_mentions": 0,
        "recent_log_failed_transitions": 0,
        "failed_state_updated_last_2h": 0,
        "failed_state_older_than_2h": 0,
    }
    if cpu_log and cpu_log.is_file():
        lines = cpu_log.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = lines[-recent_lines:]
        out["reconcile_failed_to_synth_ready"] = cpu_log.read_text(
            encoding="utf-8", errors="replace"
        ).count("reconciled state FAILED -> SYNTHESIS_READY")
        out["recent_log_docling_fail_mentions"] = sum(
            1 for l in recent if "no PDFs converted" in l
        )
        out["recent_log_failed_transitions"] = sum(
            1 for l in recent if "-> FAILED" in l or "state=FAILED" in l
        )
    for p in failed_state_files:
        age_h = (now - p.stat().st_mtime) / 3600
        if age_h < 2:
            out["failed_state_updated_last_2h"] += 1
        else:
            out["failed_state_older_than_2h"] += 1
    return out


def build_report(
    data_root: Path,
    output_root: Path,
    paper_ids_path: Path,
) -> Dict[str, Any]:
    now = time.time()
    state_dir = data_root / "logs" / "scheduler_state"
    logs_dir = data_root / "logs"

    state_counts: Counter[str] = Counter()
    queue_buckets: Counter[str] = Counter()
    failed_errors: Counter[str] = Counter()
    failed_cats: Counter[str] = Counter()
    failed_detail = Counter()

    for p in state_dir.glob("*.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        aid = p.stem
        state = str(st.get("state") or "?")
        state_counts[state] += 1
        bucket = STATE_TO_BUCKET.get(state, state.lower())
        queue_buckets[bucket] += 1
        if state != "FAILED":
            continue
        err = str(st.get("last_error") or "").strip()
        failed_errors[err[:150]] += 1
        failed_cats[_categorize_failed_error(err)] += 1
        graded = (output_root / f"{aid}_graded.json").is_file()
        results = (output_root / f"{aid}_results.json").is_file()
        if graded:
            failed_detail["with_graded"] += 1
        if results:
            failed_detail["with_results"] += 1
        if graded and not results:
            failed_detail["with_graded_no_results"] += 1
        if not graded and not results:
            failed_detail["no_graded_no_results"] += 1

    graded_files = list(output_root.glob("*_graded.json"))
    results_files = list(output_root.glob("*_results.json"))
    analysis_files = list(output_root.glob("*_analysis.json"))

    graded_n = len(graded_files)
    results_n = len(results_files)
    total = _load_total_alignments(paper_ids_path)
    registered = sum(state_counts.values())
    failed_n = state_counts.get("FAILED", 0)

    g_rate_1h = _count_artifacts(output_root, "_graded.json", 1, now)
    g_rate_6h = _count_artifacts(output_root, "_graded.json", 6, now) / 6.0
    r_rate_1h = _count_artifacts(output_root, "_results.json", 1, now)
    r_rate_6h = _count_artifacts(output_root, "_results.json", 6, now) / 6.0

    g_rate = max(g_rate_1h, g_rate_6h, 0.1)
    r_rate = max(r_rate_1h, r_rate_6h, 0.1)

    actionable = max(0, total - failed_n)
    remaining_results = max(0, actionable - results_n)
    remaining_grade = max(0, actionable - graded_n)
    synth_backlog = max(0, graded_n - results_n)

    failed_state_paths: List[Path] = []
    for p in state_dir.glob("*.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if st.get("state") == "FAILED":
            failed_state_paths.append(p)
    cpu_log = _find_cpu_log(logs_dir)
    dynamics = _failed_dynamics(cpu_log, failed_state_paths, now)

    hourly_graded: Dict[int, int] = Counter()
    hourly_results: Dict[int, int] = Counter()
    for p in graded_files:
        h = int((now - p.stat().st_mtime) // 3600)
        if h < 24:
            hourly_graded[h] += 1
    for p in results_files:
        h = int((now - p.stat().st_mtime) // 3600)
        if h < 24:
            hourly_results[h] += 1

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "cpu_log": str(cpu_log) if cpu_log else None,
        "total_alignments": total,
        "scheduler_registered": registered,
        "not_started_estimate": max(0, total - registered),
        "state_counts": dict(state_counts),
        "queue_buckets": dict(queue_buckets),
        "artifacts": {
            "graded_json": graded_n,
            "results_json": results_n,
            "analysis_json": len(analysis_files),
        },
        "throughput": {
            "graded_per_hr_1h": g_rate_1h,
            "graded_per_hr_6h": round(g_rate_6h, 1),
            "results_per_hr_1h": r_rate_1h,
            "results_per_hr_6h": round(r_rate_6h, 1),
        },
        "failed": {
            "count": failed_n,
            "pct_of_total": round(100 * failed_n / total, 1) if total else 0,
            "categories": dict(failed_cats),
            "detail": dict(failed_detail),
            "top_errors": failed_errors.most_common(8),
            "dynamics": dynamics,
            "frozen": dynamics.get("recent_log_failed_transitions", 1) == 0
            and dynamics.get("failed_state_updated_last_2h", 1) == 0,
        },
        "backlog": {
            "grader_queue": queue_buckets.get("grader_queue", 0),
            "grading_inflight": queue_buckets.get("grading", 0),
            "synthesis_queue": queue_buckets.get("synthesis_queue", 0),
            "synthesis_inflight": queue_buckets.get("synthesis", 0),
            "graded_awaiting_synthesis": synth_backlog,
        },
        "eta_hours": {
            "synthesis_backlog": round(synth_backlog / r_rate, 1),
            "grading_remaining": round(remaining_grade / g_rate, 1),
            "full_actionable_results": round(remaining_results / r_rate, 1),
        },
        "actionable": {
            "count": actionable,
            "results_so_far": results_n,
            "pct_complete": round(100 * results_n / actionable, 1) if actionable else 0,
        },
        "hourly_last_24h": {
            "graded": {str(k): v for k, v in sorted(hourly_graded.items())},
            "results": {str(k): v for k, v in sorted(hourly_results.items())},
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    a = report["artifacts"]
    t = report["throughput"]
    f = report["failed"]
    b = report["backlog"]
    e = report["eta_hours"]
    act = report["actionable"]
    ts = report["timestamp"]
    total = report["total_alignments"]
    failed_n = f["count"]
    frozen = f.get("frozen", False)

    lines = [
        f"# Pipeline status report ({ts})",
        "",
        f"Total alignments: **{total}** | Registered: **{report['scheduler_registered']}** | "
        f"Not started (est.): **{report['not_started_estimate']}**",
        "",
        "## Headline numbers",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Graded (`*_graded.json`) | **{a['graded_json']}** |",
        f"| Synthesized (`*_results.json`) | **{a['results_json']}** |",
        f"| Analysis (`*_analysis.json`) | **{a['analysis_json']}** |",
        f"| Scheduler DONE | **{report['state_counts'].get('DONE', 0)}** |",
        f"| FAILED | **{failed_n}** ({f['pct_of_total']}%) |",
        f"| Grader queue | **{b['grader_queue']}** |",
        f"| Grading inflight | **{b['grading_inflight']}** |",
        f"| Synthesis queue | **{b['synthesis_queue']}** |",
        f"| Synthesis inflight | **{b['synthesis_inflight']}** |",
        "",
        "## Failures",
        "",
    ]

    if frozen:
        lines.append(
            "**Failure set appears frozen** — no recent FAILED transitions in CPU log; "
            "no FAILED state files updated in last 2h."
        )
    else:
        lines.append("**Failures may still be accumulating** — check CPU log.")
    lines.extend(
        [
            "",
            f"- Reconciled on startup (FAILED→SYNTHESIS_READY): **{f['dynamics'].get('reconcile_failed_to_synth_ready', 0)}**",
            f"- FAILED with graded.json: **{f['detail'].get('with_graded', 0)}**",
            f"- FAILED with results.json: **{f['detail'].get('with_results', 0)}**",
            "",
            "**Categories:**",
            "",
        ]
    )
    for cat, n in sorted(f["categories"].items(), key=lambda x: -x[1]):
        lines.append(f"- {cat}: {n}")
    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Step | 1h | 6h avg |",
            "|------|---:|-------:|",
            f"| Grading | {t['graded_per_hr_1h']}/hr | {t['graded_per_hr_6h']}/hr |",
            f"| Synthesis | {t['results_per_hr_1h']}/hr | {t['results_per_hr_6h']}/hr |",
            f"| Graded awaiting synthesis | **{b['graded_awaiting_synthesis']}** | |",
            "",
            "## Timing estimates (actionable = total − permanent failures)",
            "",
            f"- Actionable alignments: **{act['count']}** ({act['pct_complete']}% have results)",
            f"- Synthesis backlog ETA: **~{e['synthesis_backlog']} h**",
            f"- Grading remaining ETA: **~{e['grading_remaining']} h**",
            f"- Full actionable results ETA: **~{e['full_actionable_results']} h**",
            "",
            f"Permanent docling failures excluded: **{failed_n}** alignments.",
            "",
        ]
    )
    if report.get("cpu_log"):
        lines.append(f"CPU log: `{report['cpu_log']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/private/groups/corbettlab/gabe/auto_lit_eval_data",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="LLM results root (default: DATA_ROOT/llm_results)",
    )
    parser.add_argument(
        "--paper-ids",
        default="",
        help="Search JSON path (default: DATA_ROOT/search_results/lp-human-all_search.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Write report files here (default: DATA_ROOT/logs)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root or data_root / "llm_results")
    paper_ids = Path(
        args.paper_ids or data_root / "search_results" / "lp-human-all_search.json"
    )
    out_dir = Path(args.out_dir or data_root / "logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(data_root, output_root, paper_ids)
    md = render_markdown(report)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    json_path = out_dir / f"pipeline_status_report_{stamp}.json"
    md_path = out_dir / f"pipeline_status_report_{stamp}.md"
    latest_json = out_dir / "pipeline_status_report_latest.json"
    latest_md = out_dir / "pipeline_status_report_latest.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    latest_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
