#!/usr/bin/env python3
"""Summarize grader benchmark timing and baseline score comparison."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _timing_stats(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"count": 0, "mean_sec": 0.0, "median_sec": 0.0, "p90_sec": 0.0}
    return {
        "count": float(len(latencies)),
        "mean_sec": round(statistics.mean(latencies), 3),
        "median_sec": round(statistics.median(latencies), 3),
        "p90_sec": round(_percentile(latencies, 0.9), 3),
    }


def _axis_keys(rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for row in rows:
        for src in (
            row.get("baseline_rubric_dimension_scores") or {},
            row.get("new_rubric_dimension_scores") or {},
        ):
            keys.update(str(k) for k in src.keys())
    return sorted(keys)


def _baseline_latency_from_jsonl(
    jsonl_path: Path,
    file_name: str,
) -> Optional[float]:
    if not jsonl_path.is_file():
        return None
    rows = _read_jsonl(jsonl_path)
    target = [r for r in rows if r.get("file_name") == file_name and r.get("phase") == "chat_completion"]
    if len(target) < 2:
        return None
    gaps: List[float] = []
    for prev, cur in zip(target, target[1:]):
        t0 = prev.get("ts_iso")
        t1 = cur.get("ts_iso")
        if not t0 or not t1:
            continue
        try:
            dt0 = datetime.fromisoformat(str(t0).replace("Z", "+00:00"))
            dt1 = datetime.fromisoformat(str(t1).replace("Z", "+00:00"))
        except ValueError:
            continue
        gap = (dt1 - dt0).total_seconds()
        if 0 < gap < 3600:
            gaps.append(gap)
    if not gaps:
        return None
    return round(statistics.median(gaps), 3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = _read_jsonl(run_dir / "results.jsonl")
    if not results:
        raise SystemExit(f"No results in {run_dir / 'results.jsonl'}")

    def _row_graded_ok(row: Dict[str, Any]) -> bool:
        if "graded_ok" in row:
            return bool(row.get("graded_ok"))
        return row.get("http_status") == 200 and row.get("content_empty") is not True

    graded_rows = [r for r in results if _row_graded_ok(r)]
    latencies = [
        float(r["latency_sec"])
        for r in graded_rows
        if r.get("latency_sec") is not None
    ]
    by_grader: Dict[int, List[float]] = {}
    for row in graded_rows:
        if row.get("latency_sec") is None:
            continue
        idx = int(row.get("grader_index", 0))
        by_grader.setdefault(idx, []).append(float(row["latency_sec"]))

    per_grader = {
        str(idx): _timing_stats(vals) for idx, vals in sorted(by_grader.items())
    }
    pooled = _timing_stats(latencies)
    total_sec = sum(latencies)
    summary = {
        "n_results": len(results),
        "n_http_ok": sum(1 for r in results if r.get("http_status") == 200),
        "n_graded_ok": len(graded_rows),
        "n_llm_failed": sum(1 for r in results if not _row_graded_ok(r)),
        "n_content_empty": sum(1 for r in results if r.get("content_empty")),
        "pooled_latency": pooled,
        "papers_per_sec_pooled": round(len(latencies) / total_sec, 4) if total_sec > 0 else 0.0,
        "per_grader_latency": per_grader,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    timing_rows = []
    for idx, stats in sorted(per_grader.items(), key=lambda x: int(x[0])):
        timing_rows.append({"grader_index": idx, **stats})
    timing_rows.append({"grader_index": "pooled", **pooled})
    with (run_dir / "timing_by_grader.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["grader_index", "count", "mean_sec", "median_sec", "p90_sec"],
        )
        w.writeheader()
        for row in timing_rows:
            w.writerow(row)

    axis_keys = _axis_keys(results)
    compare_fields = [
        "sample_id",
        "alignment_id",
        "file_name",
        "paper_role",
        "grader_index",
        "graded_ok",
        "graded_error",
        "latency_sec",
        "baseline_relevance_grade",
        "new_relevance_grade",
        "relevance_delta",
    ]
    compare_fields.extend(f"baseline_{k}" for k in axis_keys)
    compare_fields.extend(f"new_{k}" for k in axis_keys)
    compare_fields.extend(["baseline_rationale", "new_rationale"])

    baseline_timing_rows: List[Dict[str, Any]] = []
    with (run_dir / "compare.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=compare_fields, extrasaction="ignore")
        w.writeheader()
        paper_by_id = {p["sample_id"]: p for p in manifest.get("papers") or []}
        for row in sorted(results, key=lambda r: str(r.get("sample_id"))):
            base_rel = row.get("baseline_relevance_grade")
            new_rel = row.get("new_relevance_grade")
            delta = None
            if base_rel is not None and new_rel is not None:
                delta = round(float(new_rel) - float(base_rel), 4)
            out: Dict[str, Any] = {
                "sample_id": row.get("sample_id"),
                "alignment_id": row.get("alignment_id"),
                "file_name": row.get("file_name"),
                "paper_role": row.get("paper_role"),
                "grader_index": row.get("grader_index"),
                "graded_ok": _row_graded_ok(row),
                "graded_error": row.get("graded_error") or row.get("error") or "",
                "latency_sec": row.get("latency_sec"),
                "baseline_relevance_grade": base_rel,
                "new_relevance_grade": new_rel,
                "relevance_delta": delta,
                "baseline_rationale": row.get("baseline_rationale"),
                "new_rationale": row.get("new_rationale"),
            }
            base_dims = row.get("baseline_rubric_dimension_scores") or {}
            new_dims = row.get("new_rubric_dimension_scores") or {}
            for k in axis_keys:
                out[f"baseline_{k}"] = base_dims.get(k, "")
                out[f"new_{k}"] = new_dims.get(k, "")
            w.writerow(out)

            manifest_row = paper_by_id.get(row.get("sample_id"), {})
            baseline_jsonl = Path(str(manifest_row.get("baseline_grader_llm_jsonl") or ""))
            approx = _baseline_latency_from_jsonl(baseline_jsonl, str(row.get("file_name")))
            if approx is not None:
                baseline_timing_rows.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "file_name": row.get("file_name"),
                        "baseline_approx_latency_sec": approx,
                        "new_latency_sec": row.get("latency_sec"),
                    }
                )

    if baseline_timing_rows:
        with (run_dir / "timing_vs_baseline.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "file_name",
                    "baseline_approx_latency_sec",
                    "new_latency_sec",
                ],
            )
            w.writeheader()
            w.writerows(baseline_timing_rows)

    print(json.dumps(summary, indent=2))
    return 0 if summary["n_llm_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
