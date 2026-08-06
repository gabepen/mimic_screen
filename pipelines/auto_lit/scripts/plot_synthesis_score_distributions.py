#!/usr/bin/env python3
"""Plot synthesis-only score distributions for Legionella and Wolbachia.

Legionella: histogram of mimicry_plausibility for synthesis_status=ok pairs,
with known-mimic (control) scores and their mean overlaid.

Wolbachia: same mimicry_plausibility distribution with upper-percentile
thresholds and gene counts above each threshold.

Also writes summary_stats.json / summary_stats.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


SCORE_DIMS = (
    "host_exploitation_score",
    "query_effector_score",
    "mimicry_plausibility_score",
    "pair_priority_score",
)

DEFAULT_PERCENTILES = (75, 90, 95, 99)


def _num(raw: Any) -> Optional[float]:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def synthesis_only(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if str(r.get("synthesis_status") or "").strip() == "ok"]


def score_values(rows: Sequence[Dict[str, str]], field: str) -> np.ndarray:
    vals = [_num(r.get(field)) for r in rows]
    return np.asarray([v for v in vals if v is not None], dtype=float)


def control_row_stats(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        dim_vals = {d: _num(r.get(d)) for d in SCORE_DIMS}
        present = [v for v in dim_vals.values() if v is not None]
        out.append(
            {
                "query": r.get("query") or "",
                "target": r.get("target") or "",
                "alignment_id": r.get("alignment_id")
                or f"{r.get('query')}_{r.get('target')}",
                "query_description": (r.get("query_description") or "")[:80],
                "synthesis_status": r.get("synthesis_status") or "",
                **{d: dim_vals[d] for d in SCORE_DIMS},
                "mean_synthesis_score": (mean(present) if present else None),
            }
        )
    return out


def summarize_distribution(values: np.ndarray, percentiles: Sequence[int]) -> Dict[str, Any]:
    if len(values) == 0:
        return {"n": 0}
    pct = {f"p{p}": float(np.percentile(values, p)) for p in percentiles}
    upper = {}
    for p in percentiles:
        thr = pct[f"p{p}"]
        upper[f"ge_p{p}"] = {
            "threshold": thr,
            "count": int(np.sum(values >= thr)),
            "fraction": float(np.mean(values >= thr)),
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": pct,
        "upper_tail": upper,
    }


def _short_control_label(row: Dict[str, Any]) -> str:
    """Prefer a readable effector-ish name from the description; else a short phrase."""
    desc = str(row.get("query_description") or "").strip()
    query = str(row.get("query") or "unknown")
    for pat in (
        r"\b(VipD|SidD|AnkX|RomA|DrrA|MesI|Lgt1|MIP|UVR8)\b",
        r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b",
    ):
        m = re.search(pat, desc)
        if m:
            return m.group(1)
    if desc:
        # Fall back to a compact description fragment, not a bare UniProt id.
        words = re.findall(r"[A-Za-z0-9/-]+", desc)
        phrase = " ".join(words[:3])
        if phrase:
            return phrase[:28]
    return query


def _empirical_percentile(values: np.ndarray, score: float) -> float:
    """Percent of background scores strictly below `score` (0-100)."""
    if len(values) == 0:
        return float("nan")
    return float(100.0 * np.mean(values < score))


def plot_legionella_controls(
    full_scores: np.ndarray,
    control_stats: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    controls = [
        c for c in control_stats if c.get("mimicry_plausibility_score") is not None
    ]
    controls = sorted(
        controls, key=lambda c: float(c["mimicry_plausibility_score"]), reverse=True
    )
    ctrl_mim = np.asarray(
        [float(c["mimicry_plausibility_score"]) for c in controls], dtype=float
    )
    mean_mim = float(np.mean(ctrl_mim)) if len(ctrl_mim) else float("nan")

    fig, (ax_hist, ax_ctrl) = plt.subplots(
        2,
        1,
        figsize=(10, 8.2),
        gridspec_kw={"height_ratios": [1.15, 1.6], "hspace": 0.08},
        sharex=True,
    )

    bins = np.linspace(0, 100, 21)
    ax_hist.hist(
        full_scores,
        bins=bins,
        color="#c5d4e0",
        edgecolor="white",
        linewidth=0.6,
        label=f"All synthesis-only pairs (n={len(full_scores)})",
    )
    if len(ctrl_mim):
        ax_hist.axvline(
            mean_mim,
            color="#c0392b",
            linewidth=2.0,
            label=f"Known-mimic mean ({mean_mim:.0f})",
        )
    ax_hist.set_ylabel("Gene pairs")
    ax_hist.set_title(
        "Legionella synthesis-only mimicry scores\n"
        "Background distribution (top) vs labeled known mimics (bottom)"
    )
    ax_hist.legend(frameon=False, loc="upper right")
    ax_hist.set_xlim(0, 100)
    ax_hist.tick_params(labelbottom=False)

    labels = []
    scores = []
    pcts = []
    for c in controls:
        score = float(c["mimicry_plausibility_score"])
        labels.append(_short_control_label(c))
        scores.append(score)
        pcts.append(_empirical_percentile(full_scores, score))

    y = np.arange(len(controls))
    ax_ctrl.hlines(y, 0, scores, color="#9bb0c2", linewidth=1.4, zorder=1)
    ax_ctrl.scatter(
        scores,
        y,
        s=54,
        color="#1f4e79",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    if len(ctrl_mim):
        ax_ctrl.axvline(mean_mim, color="#c0392b", linewidth=1.6, alpha=0.9)

    for yi, score, pct in zip(y, scores, pcts):
        ax_ctrl.text(
            min(score + 1.5, 97),
            yi,
            f"{score:.0f}  (p{pct:.0f})",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax_ctrl.set_yticks(y)
    ax_ctrl.set_yticklabels(labels, fontsize=9)
    ax_ctrl.set_xlabel("Mimicry plausibility score")
    ax_ctrl.set_ylabel("Known mimics")
    ax_ctrl.set_xlim(0, 100)
    ax_ctrl.set_ylim(-0.7, len(controls) - 0.3)
    ax_ctrl.invert_yaxis()
    ax_ctrl.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax_ctrl.set_axisbelow(True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wolbachia_percentiles(
    scores: np.ndarray,
    percentiles: Sequence[int],
    out_path: Path,
) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bins = np.linspace(0, 100, 21)
    ax.hist(
        scores,
        bins=bins,
        color="#a3c1ad",
        edgecolor="white",
        linewidth=0.6,
        label=f"Synthesis-only pairs (n={len(scores)})",
    )

    colors = ["#6c757d", "#2c6e49", "#bc4749", "#6a040f"]
    summary = summarize_distribution(scores, percentiles)
    for i, p in enumerate(percentiles):
        thr = summary["percentiles"][f"p{p}"]
        count = summary["upper_tail"][f"ge_p{p}"]["count"]
        color = colors[i % len(colors)]
        ax.axvline(thr, color=color, linewidth=1.8, linestyle="--")
        ax.text(
            thr,
            ax.get_ylim()[1] * (0.92 - 0.08 * i),
            f"p{p}={thr:.0f}\nn≥={count}",
            color=color,
            fontsize=8,
            ha="left",
            va="top",
            rotation=0,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": color},
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Synthesis-only mimicry plausibility score")
    ax.set_ylabel("Number of gene pairs")
    ax.set_title("Wolbachia: upper percentiles of synthesis-only mimicry plausibility")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def write_summary_md(stats: Dict[str, Any], path: Path) -> None:
    lines: List[str] = ["# Synthesis score distribution summary", ""]
    lp = stats["legionella"]
    lines.append("## Legionella")
    lines.append("")
    lines.append(f"- Full results rows: {lp['n_full']}")
    lines.append(f"- Synthesis-only (`status=ok`): {lp['n_synthesis_only']}")
    lines.append(
        f"- Mimicry distribution: mean={lp['mimicry']['mean']:.1f}, "
        f"median={lp['mimicry']['median']:.1f}"
    )
    lines.append(f"- Known mimics (controls): {lp['n_controls']}")
    lines.append(
        f"- Known-mimic mean mimicry: {lp['control_mean_mimicry']:.1f}"
    )
    lines.append(
        f"- Known-mimic mean of 4 synthesis scores: {lp['control_mean_avg4']:.1f}"
    )
    lines.append("")
    lines.append("| query | target | mimicry | mean(4) | status |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for c in lp["controls"]:
        mim = c["mimicry_plausibility_score"]
        avg = c["mean_synthesis_score"]
        lines.append(
            f"| {c['query']} | {c['target']} | "
            f"{'' if mim is None else f'{mim:.0f}'} | "
            f"{'' if avg is None else f'{avg:.1f}'} | {c['synthesis_status']} |"
        )
    lines.append("")

    wol = stats["wolbachia"]
    lines.append("## Wolbachia")
    lines.append("")
    lines.append(f"- Full results rows: {wol['n_full']}")
    lines.append(f"- Synthesis-only (`status=ok`): {wol['n_synthesis_only']}")
    lines.append(
        f"- Mimicry distribution: mean={wol['mimicry']['mean']:.1f}, "
        f"median={wol['mimicry']['median']:.1f}"
    )
    lines.append("")
    lines.append("| percentile | threshold | genes ≥ threshold | fraction |")
    lines.append("| --- | ---: | ---: | ---: |")
    for p, info in wol["mimicry"]["upper_tail"].items():
        pct = p.replace("ge_p", "p")
        lines.append(
            f"| {pct} | {info['threshold']:.1f} | {info['count']} | {info['fraction']:.1%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--legionella-summary",
        type=Path,
        required=True,
        help="Full Legionella results_summary.csv",
    )
    p.add_argument(
        "--controls",
        type=Path,
        required=True,
        help="Legionella controls with LLM scores CSV",
    )
    p.add_argument(
        "--wolbachia-summary",
        type=Path,
        required=True,
        help="Full Wolbachia results_summary.csv",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
    )
    args = p.parse_args()

    lp_rows = load_csv(args.legionella_summary)
    ctrl_rows = load_csv(args.controls)
    wol_rows = load_csv(args.wolbachia_summary)

    lp_syn = synthesis_only(lp_rows)
    wol_syn = synthesis_only(wol_rows)
    lp_mim = score_values(lp_syn, "mimicry_plausibility_score")
    wol_mim = score_values(wol_syn, "mimicry_plausibility_score")
    control_stats = control_row_stats(ctrl_rows)

    if len(lp_mim) == 0:
        raise SystemExit("No synthesis-only Legionella mimicry scores found")
    if len(wol_mim) == 0:
        raise SystemExit("No synthesis-only Wolbachia mimicry scores found")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    leg_plot = out_dir / "legionella_known_mimics_on_mimicry_distribution.png"
    wol_plot = out_dir / "wolbachia_mimicry_upper_percentiles.png"
    plot_legionella_controls(lp_mim, control_stats, leg_plot)
    wol_mim_summary = plot_wolbachia_percentiles(wol_mim, args.percentiles, wol_plot)

    ctrl_mim_vals = [
        c["mimicry_plausibility_score"]
        for c in control_stats
        if c["mimicry_plausibility_score"] is not None
    ]
    ctrl_avg_vals = [
        c["mean_synthesis_score"]
        for c in control_stats
        if c["mean_synthesis_score"] is not None
    ]

    stats = {
        "legionella": {
            "n_full": len(lp_rows),
            "n_synthesis_only": len(lp_syn),
            "n_controls": len(control_stats),
            "mimicry": summarize_distribution(lp_mim, args.percentiles),
            "control_mean_mimicry": float(mean(ctrl_mim_vals)) if ctrl_mim_vals else None,
            "control_mean_avg4": float(mean(ctrl_avg_vals)) if ctrl_avg_vals else None,
            "control_median_mimicry": float(median(ctrl_mim_vals)) if ctrl_mim_vals else None,
            "controls": control_stats,
            "by_dimension_means": {
                d: float(mean([c[d] for c in control_stats if c[d] is not None]))
                for d in SCORE_DIMS
            },
        },
        "wolbachia": {
            "n_full": len(wol_rows),
            "n_synthesis_only": len(wol_syn),
            "mimicry": wol_mim_summary,
            "host_exploitation": summarize_distribution(
                score_values(wol_syn, "host_exploitation_score"), args.percentiles
            ),
            "pair_priority": summarize_distribution(
                score_values(wol_syn, "pair_priority_score"), args.percentiles
            ),
        },
    }

    json_path = out_dir / "summary_stats.json"
    md_path = out_dir / "summary_stats.md"
    json_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_summary_md(stats, md_path)

    # also dump control table
    ctrl_out = out_dir / "legionella_known_mimic_scores.csv"
    fields = [
        "query",
        "target",
        "alignment_id",
        "query_description",
        "synthesis_status",
        *SCORE_DIMS,
        "mean_synthesis_score",
    ]
    with ctrl_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in control_stats:
            writer.writerow(row)

    print(f"Wrote {leg_plot}")
    print(f"Wrote {wol_plot}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {ctrl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
