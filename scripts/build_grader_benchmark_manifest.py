#!/usr/bin/env python3
"""Build a 50-paper manifest for grader-only A5500 benchmark runs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from sample_blind_grading_audit import (
    GRADE_BINS,
    PaperRecord,
    _allocate_bin_counts,
    _bin_pool,
    _sample_without_replacement,
    load_candidate_pool,
)


def _max_dim_score(rec: PaperRecord) -> float:
    if not rec.rubric_dimension_scores:
        return 0.0
    return max(float(v) for v in rec.rubric_dimension_scores.values())


def _is_eligible(rec: PaperRecord) -> bool:
    return rec.relevance_grade > 0.0 and _max_dim_score(rec) > 0.0


def _source_txt_path(rec: PaperRecord) -> str:
    if rec.papers_dir:
        return str(Path(rec.papers_dir) / rec.file_name)
    return ""


def _baseline_llm_log(output_root: Path, alignment_id: str) -> str:
    return str(output_root / "logs" / f"{alignment_id}_grader_llm.jsonl")


def sample_benchmark_papers(
    pool: Sequence[PaperRecord],
    n: int,
    min_eligible: int,
    rng: random.Random,
) -> tuple[List[PaperRecord], Dict[str, Any]]:
    eligible = [p for p in pool if _is_eligible(p)]
    meta: Dict[str, Any] = {
        "pool_size": len(pool),
        "eligible_size": len(eligible),
        "requested": n,
        "min_eligible": min_eligible,
    }
    if len(eligible) < min(min_eligible, n):
        raise SystemExit(
            f"Need at least {min(min_eligible, n)} eligible papers "
            f"(relevance>0 and max axis>0); found {len(eligible)}"
        )

    n = min(n, len(pool))
    n_eligible_target = min(n, max(min_eligible, n))
    n_eligible_target = min(n_eligible_target, len(eligible))

    bin_counts = _allocate_bin_counts(n_eligible_target)
    selected: List[PaperRecord] = []
    picked_keys: set[tuple[str, str]] = set()
    bin_picked: Dict[str, int] = {}

    for name, count in bin_counts.items():
        if count <= 0:
            continue
        candidates = [
            p
            for p in _bin_pool(eligible, name)
            if (p.alignment_id, p.file_name) not in picked_keys
        ]
        picked = _sample_without_replacement(rng, candidates, count)
        selected.extend(picked)
        picked_keys.update((p.alignment_id, p.file_name) for p in picked)
        bin_picked[name] = len(picked)

    if len(selected) < n:
        leftovers = [
            p
            for p in eligible
            if (p.alignment_id, p.file_name) not in picked_keys
        ]
        need = n - len(selected)
        extra = _sample_without_replacement(rng, leftovers, need)
        selected.extend(extra)
        picked_keys.update((p.alignment_id, p.file_name) for p in extra)

    rng.shuffle(selected)
    meta["bin_picked"] = bin_picked
    meta["selected_eligible"] = sum(1 for p in selected if _is_eligible(p))
    meta["selected_total"] = len(selected)
    return selected, meta


def records_to_manifest_rows(
    selected: Sequence[PaperRecord],
    output_root: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(selected, start=1):
        src = _source_txt_path(rec)
        rows.append(
            {
                "sample_id": f"{i:03d}",
                "alignment_id": rec.alignment_id,
                "file_name": rec.file_name,
                "paper_id": rec.doi,
                "paper_role": rec.paper_role,
                "query": rec.query_gene_id,
                "target_id": rec.target_gene_id,
                "papers_dir": rec.papers_dir,
                "source_txt_path": src,
                "baseline_relevance_grade": rec.relevance_grade,
                "baseline_rubric_dimension_scores": rec.rubric_dimension_scores,
                "baseline_rubric_axis_rationales": rec.rubric_axis_rationales,
                "baseline_rationale": rec.rationale,
                "baseline_grader_llm_jsonl": _baseline_llm_log(output_root, rec.alignment_id),
                "baseline_grading_meta": rec.grading_meta,
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True, help="Production llm_results root")
    p.add_argument("--run-dir", required=True, help="Benchmark run directory")
    p.add_argument(
        "--papers-root",
        default="",
        help="Fallback papers root (default: output-root/../papers)",
    )
    p.add_argument("--n-papers", type=int, default=50)
    p.add_argument("--min-eligible", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_root = Path(args.output_root)
    run_dir = Path(args.run_dir)
    papers_root = Path(args.papers_root) if args.papers_root else output_root.parent / "papers"

    pool = load_candidate_pool(output_root, papers_root if papers_root.is_dir() else None)
    rng = random.Random(args.seed)
    selected, sample_meta = sample_benchmark_papers(
        pool, args.n_papers, args.min_eligible, rng
    )

    rows = records_to_manifest_rows(selected, output_root)
    missing_txt = [
        row["sample_id"]
        for row in rows
        if not row["source_txt_path"] or not Path(row["source_txt_path"]).is_file()
    ]
    if missing_txt:
        print(
            f"Warning: {len(missing_txt)} selected papers missing source txt; "
            f"first: {missing_txt[:5]}",
            file=sys.stderr,
        )
        rows = [row for row in rows if Path(row["source_txt_path"]).is_file()]
        if len(rows) < args.min_eligible:
            raise SystemExit(
                f"Too few papers with existing txt files after filtering: {len(rows)}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "papers_root": str(papers_root),
        "seed": args.seed,
        "sampling": sample_meta,
        "papers": rows,
    }
    out_path = run_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} papers to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
