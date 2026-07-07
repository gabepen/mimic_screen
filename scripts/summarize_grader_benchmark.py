#!/usr/bin/env python3
"""Summarize grader benchmark timing and baseline score comparison."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.env_config import resolve_rubric_paths  # noqa: E402

# Axes applicable per paper role, with weighted max totals from rubric JSON.
ROLE_AXES: Dict[str, List[Tuple[str, int, str]]] = {
    "target": [
        ("protein_characterisation_quality", 14, "Host A1 protein characterisation"),
        ("infection_process_relevance", 12, "Host A2 infection/relevance"),
        ("disease_population_relevance", 10, "Host A3 disease/population"),
    ],
    "query": [
        ("evidence_quality", 16, "Microbe A1 evidence quality"),
        ("system_relevance", 16, "Microbe A2 system relevance"),
    ],
}

# ~1 criterion point on a 12-point axis.
SIMILAR_DELTA_THRESHOLD = 0.083


def _criterion_axis_map(rubric_path: Path) -> Dict[str, str]:
    """Map scored criterion id -> axis id from rubric JSON."""
    if not rubric_path.is_file():
        return {}
    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, str] = {}
    for axis in rubric.get("axes") or []:
        axis_id = str(axis.get("id") or "").strip()
        if not axis_id:
            continue
        for crit in axis.get("criteria") or []:
            if not isinstance(crit, dict):
                continue
            crit_id = str(crit.get("id") or "").strip()
            if not crit_id:
                continue
            if str(crit.get("weight") or "medium").lower() == "flag":
                continue
            out[crit_id] = axis_id
    return out


def _resolve_rubric_paths(manifest: Dict[str, Any], run_dir: Path) -> Tuple[Path, Path]:
    host_path: Path | None = None
    microbe_path: Path | None = None
    for paper in manifest.get("papers") or []:
        if not isinstance(paper, dict):
            continue
        meta = paper.get("baseline_grading_meta") or {}
        if meta.get("host_rubric_path"):
            host_path = Path(str(meta["host_rubric_path"]))
        if meta.get("microbe_rubric_path"):
            microbe_path = Path(str(meta["microbe_rubric_path"]))
        break
    for graded_name in ("bench_001_graded.json", "bench_050_graded.json"):
        graded_path = run_dir / graded_name
        if not graded_path.is_file():
            continue
        try:
            data = json.loads(graded_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = data.get("grading_meta") or {}
        if meta.get("host_rubric_path"):
            host_path = Path(str(meta["host_rubric_path"]))
        if meta.get("microbe_rubric_path"):
            microbe_path = Path(str(meta["microbe_rubric_path"]))
        break
    return resolve_rubric_paths(host=host_path, microbe=microbe_path)


def _format_v2_axis_reasoning(
    criterion_scores: Dict[str, Any],
    axis_id: str,
    crit_to_axis: Dict[str, str],
) -> str:
    """Join v2 per-criterion notes for one axis."""
    parts: List[str] = []
    for crit_id in sorted(crit_to_axis.keys()):
        if crit_to_axis.get(crit_id) != axis_id:
            continue
        entry = criterion_scores.get(crit_id)
        if entry is None:
            continue
        if isinstance(entry, dict):
            score = entry.get("score", "?")
            note = str(entry.get("note") or "").strip()
        else:
            score = entry
            note = ""
        if note:
            parts.append(f"{crit_id}={score}: {note}")
        else:
            parts.append(f"{crit_id}={score}")
    return "; ".join(parts)


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


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shift_label(delta: Optional[float]) -> str:
    if delta is None:
        return "missing"
    if abs(delta) < SIMILAR_DELTA_THRESHOLD:
        return "similar"
    return "higher" if delta > 0 else "lower"


def _shift_symbol(delta: Optional[float]) -> str:
    label = _shift_label(delta)
    return {"higher": "↑", "lower": "↓", "similar": "≈", "missing": "?"}.get(label, "?")


def _comparison_text(
    baseline_norm: Optional[float],
    new_label: str,
    new_norm: Optional[float],
    delta: Optional[float],
) -> str:
    base_s = f"{baseline_norm:.3f}" if baseline_norm is not None else "?"
    new_s = new_label if new_label else "?"
    if new_norm is not None:
        new_s = f"{new_label} ({new_norm:.3f})"
    if delta is None:
        return f"{base_s} → {new_s}"
    return f"{base_s} → {new_s} ({delta:+.3f}, {_shift_symbol(delta)})"


def _load_new_graded_fields(row: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Load v2 grade fields from results row or bench_*_graded.json fallback."""
    if row.get("new_criterion_scores"):
        return row
    graded_path = Path(str(row.get("graded_path") or ""))
    if not graded_path.is_file():
        bench_id = row.get("bench_alignment_id") or f"bench_{row.get('sample_id')}"
        graded_path = run_dir / f"{bench_id}_graded.json"
    if not graded_path.is_file():
        return row
    try:
        data = json.loads(graded_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return row
    file_name = str(row.get("file_name") or "")
    for paper in data.get("graded_papers") or []:
        if not isinstance(paper, dict):
            continue
        if paper.get("file_name") == file_name:
            merged = dict(row)
            merged.setdefault("new_paper_grade", paper.get("paper_grade"))
            merged.setdefault("new_primary_grade", paper.get("primary_grade"))
            merged.setdefault("new_relevance_sort", paper.get("relevance_sort"))
            merged.setdefault("new_axis_totals", paper.get("axis_totals"))
            merged.setdefault("new_criterion_scores", paper.get("criterion_scores"))
            merged.setdefault("new_grading_schema_version", paper.get("grading_schema_version"))
            merged.setdefault(
                "new_rubric_dimension_scores",
                paper.get("rubric_dimension_scores"),
            )
            merged.setdefault("new_relevance_grade", paper.get("relevance_grade"))
            merged.setdefault(
                "new_rationale",
                paper.get("claim_summary") or paper.get("rationale"),
            )
            return merged
    return row


def _axis_rows_for_result(
    row: Dict[str, Any],
    graded_ok: bool,
    *,
    run_dir: Path,
    manifest_paper: Optional[Dict[str, Any]] = None,
    host_crit_axis: Dict[str, str],
    microbe_crit_axis: Dict[str, str],
) -> List[Dict[str, Any]]:
    role = str(row.get("paper_role") or "").strip().lower()
    axes = ROLE_AXES.get(role, [])
    if not axes:
        return []

    enriched = _load_new_graded_fields(row, run_dir)
    base_dims = enriched.get("baseline_rubric_dimension_scores") or {}
    new_dims = enriched.get("new_rubric_dimension_scores") or {}
    axis_totals = enriched.get("new_axis_totals") or {}
    manifest_paper = manifest_paper or {}
    baseline_axr = manifest_paper.get("baseline_rubric_axis_rationales") or {}
    if not baseline_axr:
        baseline_axr = enriched.get("baseline_rubric_axis_rationales") or {}
    criterion_scores = enriched.get("new_criterion_scores") or {}
    crit_map = host_crit_axis if role == "target" else microbe_crit_axis

    out: List[Dict[str, Any]] = []
    for axis_id, axis_max, axis_label in axes:
        baseline_norm = _safe_float(base_dims.get(axis_id))
        baseline_approx = None
        if baseline_norm is not None:
            baseline_approx = int(round(baseline_norm * axis_max))

        total = axis_totals.get(axis_id) if isinstance(axis_totals, dict) else None
        if isinstance(total, dict):
            new_score = total.get("score")
            new_max = total.get("max", axis_max)
            new_label = str(total.get("label") or "")
        else:
            new_norm = _safe_float(new_dims.get(axis_id))
            new_score = int(round(new_norm * axis_max)) if new_norm is not None else None
            new_max = axis_max
            new_label = (
                f"{new_score}/{new_max}"
                if new_score is not None and new_max
                else ""
            )

        new_norm = _safe_float(new_dims.get(axis_id))
        if new_norm is None and new_score is not None and new_max:
            new_norm = float(new_score) / float(new_max)

        delta = None
        if baseline_norm is not None and new_norm is not None:
            delta = round(new_norm - baseline_norm, 4)

        out.append(
            {
                "sample_id": row.get("sample_id"),
                "alignment_id": row.get("alignment_id"),
                "file_name": row.get("file_name"),
                "paper_role": role,
                "graded_ok": graded_ok,
                "axis_id": axis_id,
                "axis_label": axis_label,
                "axis_max": axis_max,
                "baseline_norm": baseline_norm,
                "baseline_approx_score": baseline_approx,
                "new_score": new_score,
                "new_max": new_max,
                "new_label": new_label,
                "new_norm": new_norm,
                "delta_norm": delta,
                "shift": _shift_label(delta),
                "comparison": _comparison_text(baseline_norm, new_label, new_norm, delta),
                "baseline_axis_reasoning": str(baseline_axr.get(axis_id) or "").strip(),
                "new_axis_reasoning": _format_v2_axis_reasoning(
                    criterion_scores, axis_id, crit_map
                ),
            }
        )
    return out


def _aggregate_axis_rows(axis_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in axis_rows:
        if not row.get("graded_ok"):
            continue
        key = (str(row.get("paper_role")), str(row.get("axis_id")))
        grouped[key].append(row)

    aggregates: List[Dict[str, Any]] = []
    for (role, axis_id), rows in sorted(grouped.items()):
        deltas = [
            float(r["delta_norm"])
            for r in rows
            if r.get("delta_norm") is not None
        ]
        shifts = [str(r.get("shift") or "") for r in rows if r.get("delta_norm") is not None]
        label = rows[0].get("axis_label") if rows else axis_id
        aggregates.append(
            {
                "paper_role": role,
                "axis_id": axis_id,
                "axis_label": label,
                "n_compared": len(deltas),
                "n_higher": sum(1 for s in shifts if s == "higher"),
                "n_lower": sum(1 for s in shifts if s == "lower"),
                "n_similar": sum(1 for s in shifts if s == "similar"),
                "pct_higher": round(100 * sum(1 for s in shifts if s == "higher") / len(deltas), 1)
                if deltas
                else 0.0,
                "pct_lower": round(100 * sum(1 for s in shifts if s == "lower") / len(deltas), 1)
                if deltas
                else 0.0,
                "mean_delta_norm": round(statistics.mean(deltas), 4) if deltas else None,
                "median_delta_norm": round(statistics.median(deltas), 4) if deltas else None,
                "mean_abs_delta_norm": round(statistics.mean([abs(d) for d in deltas]), 4)
                if deltas
                else None,
            }
        )
    return aggregates


def _readable_row(
    row: Dict[str, Any],
    axis_rows: Sequence[Dict[str, Any]],
    graded_ok: bool,
    run_dir: Path,
) -> Dict[str, Any]:
    enriched = _load_new_graded_fields(row, run_dir)
    base_rel = enriched.get("baseline_relevance_grade")
    new_rel = enriched.get("new_relevance_grade")
    rel_delta = None
    if base_rel is not None and new_rel is not None:
        rel_delta = round(float(new_rel) - float(base_rel), 4)

    out: Dict[str, Any] = {
        "sample_id": row.get("sample_id"),
        "alignment_id": row.get("alignment_id"),
        "file_name": row.get("file_name"),
        "paper_role": row.get("paper_role"),
        "graded_ok": graded_ok,
        "baseline_relevance_grade": base_rel,
        "new_relevance_grade": new_rel,
        "relevance_delta": rel_delta,
        "new_primary_grade": enriched.get("new_primary_grade"),
        "new_paper_grade": enriched.get("new_paper_grade"),
        "new_grading_schema_version": enriched.get("new_grading_schema_version"),
    }
    for ax_row in axis_rows:
        axis_id = str(ax_row.get("axis_id"))
        out[f"{axis_id}_baseline_norm"] = ax_row.get("baseline_norm")
        out[f"{axis_id}_new"] = ax_row.get("new_label")
        out[f"{axis_id}_delta_norm"] = ax_row.get("delta_norm")
        out[f"{axis_id}_shift"] = ax_row.get("shift")
        out[f"{axis_id}_comparison"] = ax_row.get("comparison")
        out[f"{axis_id}_baseline_reasoning"] = ax_row.get("baseline_axis_reasoning")
        out[f"{axis_id}_new_reasoning"] = ax_row.get("new_axis_reasoning")
    return out


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

    paper_by_id = {p["sample_id"]: p for p in manifest.get("papers") or []}

    baseline_timing_rows: List[Dict[str, Any]] = []
    with (run_dir / "compare.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=compare_fields, extrasaction="ignore")
        w.writeheader()
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

    axis_rows: List[Dict[str, Any]] = []
    readable_rows: List[Dict[str, Any]] = []
    host_rubric_path, microbe_rubric_path = _resolve_rubric_paths(manifest, run_dir)
    host_crit_axis = _criterion_axis_map(host_rubric_path)
    microbe_crit_axis = _criterion_axis_map(microbe_rubric_path)
    for row in sorted(results, key=lambda r: str(r.get("sample_id"))):
        ok = _row_graded_ok(row)
        manifest_paper = paper_by_id.get(row.get("sample_id"), {})
        per_axis = _axis_rows_for_result(
            row,
            ok,
            run_dir=run_dir,
            manifest_paper=manifest_paper,
            host_crit_axis=host_crit_axis,
            microbe_crit_axis=microbe_crit_axis,
        )
        axis_rows.extend(per_axis)
        if per_axis:
            readable_rows.append(_readable_row(row, per_axis, ok, run_dir))

    compare_by_axis_fields = [
        "sample_id",
        "alignment_id",
        "file_name",
        "paper_role",
        "graded_ok",
        "axis_id",
        "axis_label",
        "axis_max",
        "baseline_norm",
        "baseline_approx_score",
        "new_score",
        "new_max",
        "new_label",
        "new_norm",
        "delta_norm",
        "shift",
        "comparison",
        "baseline_axis_reasoning",
        "new_axis_reasoning",
    ]
    with (run_dir / "compare_by_axis.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=compare_by_axis_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(axis_rows)

    readable_fields: List[str] = [
        "sample_id",
        "alignment_id",
        "file_name",
        "paper_role",
        "graded_ok",
        "baseline_relevance_grade",
        "new_relevance_grade",
        "relevance_delta",
        "new_primary_grade",
        "new_paper_grade",
        "new_grading_schema_version",
    ]
    for role_axes in ROLE_AXES.values():
        for axis_id, _, _ in role_axes:
            readable_fields.extend(
                [
                    f"{axis_id}_comparison",
                    f"{axis_id}_shift",
                    f"{axis_id}_baseline_norm",
                    f"{axis_id}_new",
                    f"{axis_id}_delta_norm",
                    f"{axis_id}_baseline_reasoning",
                    f"{axis_id}_new_reasoning",
                ]
            )
    with (run_dir / "compare_readable.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=readable_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(readable_rows)

    axis_aggregate = _aggregate_axis_rows(axis_rows)
    aggregate_fields = [
        "paper_role",
        "axis_id",
        "axis_label",
        "n_compared",
        "n_higher",
        "n_lower",
        "n_similar",
        "pct_higher",
        "pct_lower",
        "mean_delta_norm",
        "median_delta_norm",
        "mean_abs_delta_norm",
    ]
    with (run_dir / "axis_aggregate.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=aggregate_fields)
        w.writeheader()
        w.writerows(axis_aggregate)

    notable = sorted(
        [
            r
            for r in axis_rows
            if r.get("graded_ok") and r.get("delta_norm") is not None
        ],
        key=lambda r: abs(float(r["delta_norm"])),
        reverse=True,
    )[:15]
    axis_summary = {
        "description": (
            "Baseline = v1 opaque axis float (0-1). New = v2 deterministic weighted "
            "axis total (score/max). delta_norm = new_norm - baseline_norm. "
            f"shift similar if |delta| < {SIMILAR_DELTA_THRESHOLD}."
        ),
        "n_papers": len(results),
        "n_graded_ok": len(graded_rows),
        "per_axis": axis_aggregate,
        "largest_axis_shifts": notable,
    }
    (run_dir / "axis_comparison_summary.json").write_text(
        json.dumps(axis_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote axis summaries: compare_by_axis.csv, compare_readable.csv, axis_aggregate.csv")
    return 0 if summary["n_llm_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
