#!/usr/bin/env python3
"""Sample graded papers for manual blind rubric QA.

Writes a human-facing CSV (no LLM grades) and a separate answer-key JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BLIND_CSV_FIELDS = [
    "sample_id",
    "doi",
    "gene_focus_id",
    "paper_role",
    "alignment_id",
    "query_gene_id",
    "target_gene_id",
    "file_name",
    "human_relevant_yes_no",
    "human_relevance_0_1",
    "human_notes",
]

GRADE_BINS: List[Tuple[str, float, float, float]] = [
    ("low", 0.0, 0.25, 0.40),
    ("mid", 0.25, 0.5, 0.40),
    ("high", 0.5, 1.0, 0.20),
]


@dataclass
class PaperRecord:
    alignment_id: str
    query_gene_id: str
    target_gene_id: str
    doi: str
    paper_role: str
    gene_focus_id: str
    file_name: str
    papers_dir: str
    relevance_grade: float
    rubric_dimension_scores: Dict[str, float]
    rubric_axis_rationales: Dict[str, str]
    rubric_tags: Dict[str, str]
    rationale: str
    grading_meta: Dict[str, Any] = field(default_factory=dict)


def _parse_alignment_id(alignment_id: str) -> Tuple[str, str]:
    parts = alignment_id.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return alignment_id, ""


def _gene_focus_id(query_gene_id: str, target_gene_id: str, paper_role: str) -> str:
    role = (paper_role or "").strip().lower()
    if role == "query":
        return query_gene_id
    if role == "target":
        return target_gene_id
    return query_gene_id or target_gene_id


def _load_sidecar_meta(output_root: Path, alignment_id: str) -> Dict[str, Any]:
    for suffix in ("_results.json", "_analysis.json"):
        path = output_root / f"{alignment_id}{suffix}"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _resolve_papers_dir(
    alignment_id: str,
    graded: Dict[str, Any],
    sidecar: Dict[str, Any],
    papers_root: Optional[Path],
) -> str:
    for src in (graded, sidecar):
        raw = str(src.get("papers_dir") or "").strip()
        if raw:
            return raw
    if papers_root is not None:
        return str(papers_root / alignment_id)
    return ""


def load_candidate_pool(
    output_root: Path,
    papers_root: Optional[Path],
) -> List[PaperRecord]:
    seen: set[Tuple[str, str]] = set()
    pool: List[PaperRecord] = []

    for graded_path in sorted(output_root.glob("*_graded.json")):
        alignment_id = graded_path.name[: -len("_graded.json")]
        try:
            graded = json.loads(graded_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: skip {graded_path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(graded, dict):
            continue

        sidecar = _load_sidecar_meta(output_root, alignment_id)
        query_gene_id = str(graded.get("query") or sidecar.get("query") or "").strip()
        target_gene_id = str(
            graded.get("target_id") or sidecar.get("target_id") or ""
        ).strip()
        if not query_gene_id or not target_gene_id:
            q, t = _parse_alignment_id(alignment_id)
            query_gene_id = query_gene_id or q
            target_gene_id = target_gene_id or t

        papers_dir = _resolve_papers_dir(alignment_id, graded, sidecar, papers_root)
        meta = graded.get("grading_meta") if isinstance(graded.get("grading_meta"), dict) else {}

        for row in graded.get("graded_papers") or []:
            if not isinstance(row, dict):
                continue
            file_name = str(row.get("file_name") or "").strip()
            if not file_name:
                continue
            dedupe_key = (alignment_id, file_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            doi = str(row.get("paper_id") or "").strip()
            if not doi:
                stem = Path(file_name).stem.split("__")[0]
                doi = stem.replace("_", "/", 1) if stem.startswith("10.") else stem

            paper_role = str(row.get("paper_role") or "").strip().lower() or "unknown"
            pool.append(
                PaperRecord(
                    alignment_id=alignment_id,
                    query_gene_id=query_gene_id,
                    target_gene_id=target_gene_id,
                    doi=doi,
                    paper_role=paper_role,
                    gene_focus_id=_gene_focus_id(
                        query_gene_id, target_gene_id, paper_role
                    ),
                    file_name=file_name,
                    papers_dir=papers_dir,
                    relevance_grade=float(row.get("relevance_grade") or 0.0),
                    rubric_dimension_scores={
                        str(k): float(v)
                        for k, v in (row.get("rubric_dimension_scores") or {}).items()
                    },
                    rubric_axis_rationales={
                        str(k): str(v)
                        for k, v in (row.get("rubric_axis_rationales") or {}).items()
                    },
                    rubric_tags={
                        str(k): str(v)
                        for k, v in (row.get("rubric_tags") or {}).items()
                    },
                    rationale=str(row.get("rationale") or ""),
                    grading_meta=dict(meta),
                )
            )
    return pool


def _sample_without_replacement(
    rng: random.Random,
    items: Sequence[PaperRecord],
    k: int,
) -> List[PaperRecord]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    return rng.sample(list(items), k)


def _allocate_bin_counts(n_rest: int) -> Dict[str, int]:
    counts = {name: int(n_rest * share) for name, _, _, share in GRADE_BINS}
    remainder = n_rest - sum(counts.values())
    order = sorted(GRADE_BINS, key=lambda x: -x[3])
    i = 0
    while remainder > 0:
        counts[order[i % len(order)][0]] += 1
        remainder -= 1
        i += 1
    return counts


def _bin_pool(pool: Sequence[PaperRecord], name: str) -> List[PaperRecord]:
    for bin_name, lo, hi, _ in GRADE_BINS:
        if bin_name != name:
            continue
        if name == "low":
            return [p for p in pool if 0.0 < p.relevance_grade <= hi]
        return [p for p in pool if lo < p.relevance_grade <= hi]
    return []


def _role_pool(pool: Sequence[PaperRecord], role: str) -> List[PaperRecord]:
    want = role.strip().lower()
    return [p for p in pool if (p.paper_role or "").strip().lower() == want]


def _split_stratum(
    count: int,
    query_left: int,
    target_left: int,
) -> Tuple[int, int]:
    """Split one grade stratum across roles while respecting remaining role budgets."""
    if count <= 0:
        return 0, 0
    q = count // 2
    if count % 2 and query_left > target_left:
        q += 1
    q = min(q, query_left)
    t = min(count - q, target_left)
    q = count - t
    q = min(q, query_left)
    t = count - q
    return q, t


def _allocate_cell_counts(n: int, frac: float) -> Dict[Tuple[str, str], int]:
    """Global grade stratification; each stratum split ~50/50 across query/target."""
    n_query = n // 2
    n_target = n - n_query
    n_zero = min(max(int(round(n * frac)), 0), n)
    n_rest = n - n_zero
    bin_counts = _allocate_bin_counts(n_rest)

    query_left = n_query
    target_left = n_target
    cells: Dict[Tuple[str, str], int] = {}

    strata: List[Tuple[str, int]] = [("zero", n_zero)]
    strata.extend((name, bin_counts[name]) for name, _, _, _ in GRADE_BINS)

    for category, count in strata:
        q, t = _split_stratum(count, query_left, target_left)
        cells[("query", category)] = q
        cells[("target", category)] = t
        query_left -= q
        target_left -= t

    return cells


def _cell_pool(
    pool: Sequence[PaperRecord],
    role: str,
    category: str,
) -> List[PaperRecord]:
    role_pool = _role_pool(pool, role)
    if category == "zero":
        return [p for p in role_pool if p.relevance_grade == 0.0]
    return _bin_pool(role_pool, category)


def sample_records(
    pool: Sequence[PaperRecord],
    n: int,
    rng: random.Random,
    irrelevant_min: float,
    irrelevant_max: float,
) -> Tuple[List[PaperRecord], Dict[str, Any]]:
    if not pool:
        return [], {"warning": "empty pool"}

    n = min(n, len(pool))
    frac = rng.uniform(irrelevant_min, irrelevant_max)
    cells = _allocate_cell_counts(n, frac)

    selected: List[PaperRecord] = []
    remaining_ids: set[Tuple[str, str]] = set()
    cell_picked: Dict[str, int] = {}
    bin_picked: Dict[str, int] = {name: 0 for name, _, _, _ in GRADE_BINS}
    warnings: List[str] = []

    for (role, category), need in cells.items():
        if need <= 0:
            continue
        candidates = [
            p
            for p in _cell_pool(pool, role, category)
            if (p.alignment_id, p.file_name) not in remaining_ids
        ]
        picked = _sample_without_replacement(rng, candidates, need)
        selected.extend(picked)
        remaining_ids.update((p.alignment_id, p.file_name) for p in picked)
        key = f"{role}_{category}"
        cell_picked[key] = len(picked)
        if len(picked) < need:
            warnings.append(f"{key}: only {len(picked)} available (requested {need})")
        if category != "zero":
            bin_picked[category] = bin_picked.get(category, 0) + len(picked)

    if len(selected) < n:
        leftovers = [
            p for p in pool if (p.alignment_id, p.file_name) not in remaining_ids
        ]
        extra = _sample_without_replacement(rng, leftovers, n - len(selected))
        selected.extend(extra)
        if extra:
            warnings.append(f"topped up {len(extra)} from unstratified leftovers")

    n_query = sum(1 for p in selected if p.paper_role == "query")
    n_target = sum(1 for p in selected if p.paper_role == "target")

    stats = {
        "requested_n": n,
        "actual_n": len(selected),
        "requested_n_query": n // 2,
        "requested_n_target": n - (n // 2),
        "n_query": n_query,
        "n_target": n_target,
        "irrelevant_fraction_target": frac,
        "n_zero_target_total": cells.get(("query", "zero"), 0)
        + cells.get(("target", "zero"), 0),
        "n_zero": sum(1 for p in selected if p.relevance_grade == 0.0),
        "n_nonzero": sum(1 for p in selected if p.relevance_grade > 0.0),
        "bin_picked": bin_picked,
        "cell_picked": cell_picked,
        "pool_size": len(pool),
        "pool_query": len(_role_pool(pool, "query")),
        "pool_target": len(_role_pool(pool, "target")),
        "pool_zero": sum(1 for p in pool if p.relevance_grade == 0.0),
        "pool_nonzero": sum(1 for p in pool if p.relevance_grade > 0.0),
    }
    if warnings:
        stats["warning"] = "; ".join(warnings)
    return selected, stats


def write_outputs(
    selected: Sequence[PaperRecord],
    out_dir: Path,
    seed: int,
    stats: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(selected)
    rng = random.Random(seed + 1)
    rng.shuffle(ordered)

    answer_key: Dict[str, Any] = {}
    blind_rows: List[Dict[str, str]] = []

    for i, rec in enumerate(ordered, start=1):
        sample_id = f"audit_{i:03d}"
        blind_rows.append(
            {
                "sample_id": sample_id,
                "doi": rec.doi,
                "gene_focus_id": rec.gene_focus_id,
                "paper_role": rec.paper_role,
                "alignment_id": rec.alignment_id,
                "query_gene_id": rec.query_gene_id,
                "target_gene_id": rec.target_gene_id,
                "file_name": rec.file_name,
                "human_relevant_yes_no": "",
                "human_relevance_0_1": "",
                "human_notes": "",
            }
        )
        answer_key[sample_id] = {
            "doi": rec.doi,
            "gene_focus_id": rec.gene_focus_id,
            "alignment_id": rec.alignment_id,
            "query_gene_id": rec.query_gene_id,
            "target_gene_id": rec.target_gene_id,
            "paper_role": rec.paper_role,
            "file_name": rec.file_name,
            "papers_dir": rec.papers_dir,
            "relevance_grade": rec.relevance_grade,
            "rubric_dimension_scores": rec.rubric_dimension_scores,
            "rubric_axis_rationales": rec.rubric_axis_rationales,
            "rubric_tags": rec.rubric_tags,
            "rationale": rec.rationale,
            "graded_at": rec.grading_meta.get("graded_at"),
            "grader_model": rec.grading_meta.get("grader_model"),
        }

    blind_path = out_dir / "blind_grading_sheet.csv"
    with blind_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = out_dir / "llm_grades_answer_key.json"
    key_path.write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"seed={seed}",
        f"requested_n={stats.get('requested_n')}",
        f"actual_n={stats.get('actual_n')}",
        f"pool_size={stats.get('pool_size')}",
        f"pool_query={stats.get('pool_query')}",
        f"pool_target={stats.get('pool_target')}",
        f"pool_zero={stats.get('pool_zero')}",
        f"pool_nonzero={stats.get('pool_nonzero')}",
        f"requested_n_query={stats.get('requested_n_query')}",
        f"requested_n_target={stats.get('requested_n_target')}",
        f"n_query_in_sample={stats.get('n_query')}",
        f"n_target_in_sample={stats.get('n_target')}",
        f"irrelevant_fraction_target={stats.get('irrelevant_fraction_target'):.4f}",
        f"n_zero_target_total={stats.get('n_zero_target_total')}",
        f"n_zero_in_sample={stats.get('n_zero')}",
        f"n_nonzero_in_sample={stats.get('n_nonzero')}",
        f"bin_picked_low={stats.get('bin_picked', {}).get('low', 0)}",
        f"bin_picked_mid={stats.get('bin_picked', {}).get('mid', 0)}",
        f"bin_picked_high={stats.get('bin_picked', {}).get('high', 0)}",
    ]
    if stats.get("warning"):
        report_lines.append(f"warning={stats['warning']}")
    (out_dir / "sampling_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {blind_path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {out_dir / 'sampling_report.txt'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="llm_results directory with *_graded.json",
    )
    p.add_argument("--n", type=int, required=True, help="Number of papers to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: output-root/blind_grading_audit_TIMESTAMP)",
    )
    p.add_argument(
        "--papers-root",
        type=Path,
        default=None,
        help="Parent of per-alignment paper dirs if papers_dir missing from graded JSON",
    )
    p.add_argument("--irrelevant-min", type=float, default=0.10)
    p.add_argument("--irrelevant-max", type=float, default=0.30)
    args = p.parse_args()

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if args.irrelevant_min < 0 or args.irrelevant_max > 1 or args.irrelevant_min > args.irrelevant_max:
        print("Invalid irrelevant fraction range", file=sys.stderr)
        return 2
    if not args.output_root.is_dir():
        print(f"Not a directory: {args.output_root}", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"blind_grading_audit_{ts}"

    pool = load_candidate_pool(args.output_root, args.papers_root)
    if not pool:
        print("No graded papers found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    selected, stats = sample_records(
        pool,
        args.n,
        rng,
        args.irrelevant_min,
        args.irrelevant_max,
    )
    if not selected:
        print("No samples selected.", file=sys.stderr)
        return 1

    write_outputs(selected, out_dir, args.seed, stats)
    if stats.get("warning"):
        print(f"Warning: {stats['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
