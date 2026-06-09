#!/usr/bin/env python3
"""
Build a summary table from auto-lit llm_results (scorecard v2).

Primary sort column: pair_priority_score / pair_tier (A–E).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_lit_search.synthesis_scorecard import (
    build_conclusion,
    graded_papers_from_json,
)


@dataclass
class AlignmentSummaryRow:
    alignment_id: str
    query: str
    target: str
    pair_priority_score: int
    pair_priority_tier: str
    host_exploitation_score: int
    host_exploitation_tier: str
    query_effector_score: int
    query_effector_tier: str
    mimicry_plausibility_score: int
    mimicry_plausibility_tier: str
    headline: str
    best_host_paper: str
    best_query_paper: str
    synthesis_status: str
    n_host_nonzero: int
    n_query_nonzero: int
    main_uncertainties: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "alignment_id": self.alignment_id,
            "query": self.query,
            "target": self.target,
            "pair_priority_score": self.pair_priority_score,
            "pair_priority_tier": self.pair_priority_tier,
            "host_exploitation_score": self.host_exploitation_score,
            "host_exploitation_tier": self.host_exploitation_tier,
            "query_effector_score": self.query_effector_score,
            "query_effector_tier": self.query_effector_tier,
            "mimicry_plausibility_score": self.mimicry_plausibility_score,
            "mimicry_plausibility_tier": self.mimicry_plausibility_tier,
            "headline": self.headline,
            "best_host_paper": self.best_host_paper,
            "best_query_paper": self.best_query_paper,
            "synthesis_status": self.synthesis_status,
            "n_host_nonzero": self.n_host_nonzero,
            "n_query_nonzero": self.n_query_nonzero,
            "main_uncertainties": self.main_uncertainties,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _conclusion_from_results(
    results: dict[str, Any],
    graded_papers: List[GradedPaper],
    synthesis_text: str,
) -> dict[str, Any]:
    raw = results.get("conclusion")
    if isinstance(raw, dict) and raw.get("scorecard_version") == "2":
        return raw
    return build_conclusion(
        graded_papers,
        synthesis_text,
        synthesis_status=str(
            (raw or {}).get("synthesis_status")
            if isinstance(raw, dict) and raw.get("synthesis_status")
            else "grades_only"
        ),
    )


def summarize_alignment(
    results_path: Path, graded_path: Path
) -> Optional[AlignmentSummaryRow]:
    results = _load_json(results_path)
    graded = _load_json(graded_path)

    alignment_id = str(
        results.get("alignment_id") or results_path.stem.replace("_results", "")
    )
    query = str(results.get("query") or "").strip()
    target = str(results.get("target_id") or "").strip()
    if not query or not target:
        parts = alignment_id.split("_", 1)
        if len(parts) == 2:
            query = query or parts[0]
            target = target or parts[1]

    papers = graded_papers_from_json(graded)
    synth = results.get("synthesis") or {}
    synthesis_text = str(synth.get("text") or "") if isinstance(synth, dict) else ""
    conclusion = _conclusion_from_results(results, papers, synthesis_text)
    evidence = conclusion.get("evidence") or {}

    return AlignmentSummaryRow(
        alignment_id=alignment_id,
        query=query,
        target=target,
        pair_priority_score=int(conclusion["pair_priority"]["score"]),
        pair_priority_tier=str(conclusion["pair_priority"]["tier"]),
        host_exploitation_score=int(conclusion["host_exploitation"]["score"]),
        host_exploitation_tier=str(conclusion["host_exploitation"]["tier"]),
        query_effector_score=int(conclusion["query_effector"]["score"]),
        query_effector_tier=str(conclusion["query_effector"]["tier"]),
        mimicry_plausibility_score=int(conclusion["mimicry_plausibility"]["score"]),
        mimicry_plausibility_tier=str(conclusion["mimicry_plausibility"]["tier"]),
        headline=str(conclusion.get("headline") or ""),
        best_host_paper=str(conclusion.get("best_host_paper") or ""),
        best_query_paper=str(conclusion.get("best_query_paper") or ""),
        synthesis_status=str(conclusion.get("synthesis_status") or ""),
        n_host_nonzero=int(evidence.get("n_host_nonzero") or 0),
        n_query_nonzero=int(evidence.get("n_query_nonzero") or 0),
        main_uncertainties=str(conclusion.get("main_uncertainties") or ""),
    )


def build_summary_table(output_root: Path) -> list[AlignmentSummaryRow]:
    rows: list[AlignmentSummaryRow] = []
    for results_path in sorted(output_root.glob("*_results.json")):
        alignment_id = results_path.name[: -len("_results.json")]
        graded_path = output_root / f"{alignment_id}_graded.json"
        if not graded_path.is_file():
            continue
        try:
            row = summarize_alignment(results_path, graded_path)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            print(f"Skipping {alignment_id}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (-r.pair_priority_score, r.alignment_id))
    return rows


def _write_csv(rows: list[AlignmentSummaryRow], out_path: Path) -> None:
    fieldnames = list(rows[0].as_dict().keys()) if rows else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _build_parser() -> argparse.ArgumentParser:
    default_output_root = Path("/private/groups/corbettlab/gabe/auto_lit_eval_data/llm_results")
    parser = argparse.ArgumentParser(
        description="Summarize auto-lit results into scorecard v2 table."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write CSV here (default: <output-root>/results_summary.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root: Path = args.output_root
    if not output_root.is_dir():
        print(f"Output root not found: {output_root}", file=sys.stderr)
        return 2

    rows = build_summary_table(output_root)
    if not rows:
        print(f"No complete alignments found under {output_root}", file=sys.stderr)
        return 1

    out_path = args.out or (output_root / "results_summary.csv")
    _write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
