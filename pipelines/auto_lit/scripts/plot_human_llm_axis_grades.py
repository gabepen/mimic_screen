#!/usr/bin/env python3
"""Plot human vs LLM per-axis grades from a scored blind-grading pack.

Reads human_grades.json produced by score_human_grading_sheet.py and writes:
  - axis_comparison.csv
  - axis_comparison_query.png
  - axis_comparison_target.png
  - paper_grade_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROLE_AXES: Dict[str, Tuple[str, ...]] = {
    "query": ("evidence_quality", "system_relevance"),
    "target": (
        "protein_characterisation_quality",
        "infection_process_relevance",
        "disease_population_relevance",
    ),
}

AXIS_LABELS = {
    "evidence_quality": "Evidence quality",
    "system_relevance": "System relevance",
    "protein_characterisation_quality": "Protein characterisation",
    "infection_process_relevance": "Infection process",
    "disease_population_relevance": "Disease / population",
}


def _axis_score(totals: Optional[Dict[str, Any]], axis_id: str) -> Optional[Tuple[float, float]]:
    if not isinstance(totals, dict):
        return None
    entry = totals.get(axis_id)
    if not isinstance(entry, dict):
        return None
    try:
        score = float(entry["score"])
        mx = float(entry["max"])
    except (KeyError, TypeError, ValueError):
        return None
    if mx <= 0:
        return None
    return score, mx


def _parse_paper_grade(label: Any) -> Optional[Tuple[float, float]]:
    text = str(label or "").strip()
    if "/" not in text:
        return None
    left, right = text.split("/", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def load_axis_rows(grades: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in grades:
        sample_id = str(rec.get("sample_id") or "")
        role = str(rec.get("paper_role") or "").strip().lower()
        if role not in ROLE_AXES:
            continue
        human_totals = rec.get("human_axis_totals") or {}
        llm_totals = rec.get("llm_axis_totals") or {}
        for axis_id in ROLE_AXES[role]:
            human = _axis_score(human_totals, axis_id)
            llm = _axis_score(llm_totals, axis_id)
            if human is None or llm is None:
                continue
            h_score, h_max = human
            l_score, l_max = llm
            rows.append(
                {
                    "sample_id": sample_id,
                    "paper_role": role,
                    "doi": rec.get("doi") or "",
                    "gene_focus_id": rec.get("gene_focus_id") or "",
                    "axis_id": axis_id,
                    "human_score": h_score,
                    "human_max": h_max,
                    "llm_score": l_score,
                    "llm_max": l_max,
                    "human_norm": h_score / h_max,
                    "llm_norm": l_score / l_max,
                    "delta": h_score - l_score,
                    "delta_norm": (h_score / h_max) - (l_score / l_max),
                }
            )
    return rows


def write_axis_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "sample_id",
        "paper_role",
        "doi",
        "gene_focus_id",
        "axis_id",
        "human_score",
        "human_max",
        "llm_score",
        "llm_max",
        "human_norm",
        "llm_norm",
        "delta",
        "delta_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _scatter_panel(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    title: str,
    xmax: float,
    ylabel: str = "LLM score",
    xlabel: str = "Human score",
) -> None:
    lim = max(xmax, float(np.max(xs)) if len(xs) else 0.0, float(np.max(ys)) if len(ys) else 0.0)
    lim = max(lim, 1.0)
    pad = lim * 0.05
    ax.plot([0, lim], [0, lim], color="#888888", linestyle="--", linewidth=1, zorder=1)
    # slight jitter so overlapping discrete points remain visible
    rng = np.random.default_rng(0)
    jitter = lim * 0.012
    jx = xs + rng.uniform(-jitter, jitter, size=len(xs))
    jy = ys + rng.uniform(-jitter, jitter, size=len(ys))
    ax.scatter(jx, jy, s=36, alpha=0.75, color="#1f4e79", edgecolors="white", linewidths=0.4, zorder=2)
    r = _pearson(xs, ys)
    mae = float(np.mean(np.abs(xs - ys))) if len(xs) else float("nan")
    exact = int(np.sum(xs == ys)) if len(xs) else 0
    r_txt = f"{r:.2f}" if not math.isnan(r) else "n/a"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-pad, lim + pad)
    ax.set_ylim(-pad, lim + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.03,
        0.97,
        f"n={len(xs)}\nr={r_txt}\nMAE={mae:.2f}\nexact={exact}/{len(xs)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def plot_role_axes(rows: Sequence[Dict[str, Any]], role: str, out_path: Path) -> None:
    axes = ROLE_AXES[role]
    role_rows = [r for r in rows if r["paper_role"] == role]
    n = len(axes)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for i, axis_id in enumerate(axes):
        subset = [r for r in role_rows if r["axis_id"] == axis_id]
        xs = np.array([r["human_score"] for r in subset], dtype=float)
        ys = np.array([r["llm_score"] for r in subset], dtype=float)
        xmax = max(
            [float(r["human_max"]) for r in subset]
            + [float(r["llm_max"]) for r in subset]
            + [1.0]
        )
        label = AXIS_LABELS.get(axis_id, axis_id.replace("_", " "))
        _scatter_panel(
            axs[0, i],
            xs,
            ys,
            title=label,
            xmax=float(xmax),
        )
    fig.suptitle(f"Human vs LLM axis grades — {role} papers", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paper_grades(grades: Sequence[Dict[str, Any]], out_path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, role in zip(axs, ("query", "target")):
        xs: List[float] = []
        ys: List[float] = []
        xmax = 1.0
        for rec in grades:
            if str(rec.get("paper_role") or "").lower() != role:
                continue
            human = _parse_paper_grade(rec.get("human_paper_grade"))
            llm = _parse_paper_grade(rec.get("llm_paper_grade"))
            if human is None or llm is None:
                continue
            xs.append(human[0])
            ys.append(llm[0])
            xmax = max(xmax, human[1], llm[1])
        _scatter_panel(
            ax,
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            title=f"{role} paper grade",
            xmax=xmax,
            xlabel="Human total",
            ylabel="LLM total",
        )
    fig.suptitle("Human vs LLM paper grades by role", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="Audit pack directory containing human_grades.json",
    )
    p.add_argument(
        "--grades",
        type=Path,
        default=None,
        help="Override human_grades.json path",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: pack-dir)",
    )
    args = p.parse_args()

    grades_path = args.grades or (args.pack_dir / "human_grades.json")
    out_dir = args.out_dir or args.pack_dir
    if not grades_path.is_file():
        print(f"Not found: {grades_path}", file=sys.stderr)
        return 2

    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    if not isinstance(grades, list) or not grades:
        print("human_grades.json is empty or not a list", file=sys.stderr)
        return 1

    rows = load_axis_rows(grades)
    if not rows:
        print("No comparable axis totals found", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "axis_comparison.csv"
    write_axis_csv(rows, csv_path)
    query_path = out_dir / "axis_comparison_query.png"
    target_path = out_dir / "axis_comparison_target.png"
    paper_path = out_dir / "paper_grade_comparison.png"
    plot_role_axes(rows, "query", query_path)
    plot_role_axes(rows, "target", target_path)
    plot_paper_grades(grades, paper_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {query_path}")
    print(f"Wrote {target_path}")
    print(f"Wrote {paper_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
