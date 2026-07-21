#!/usr/bin/env python3
"""
Export LLM literature-analysis scores for a query subset.

Primary CSV schema matches ``summarize_results_table.py`` (scorecard v2), keyed by
``query`` / ``target`` so it can be merged into initial alignment tables
(e.g. ``alignment_input/*.csv`` with those same columns).

Examples:
  # Filter by queries in a control alignment CSV; write merge-ready summary
  python pipelines/auto_lit/scripts/export_results_by_query.py \\
    --output-root /path/to/llm_results \\
    --alignment-csv alignment_input/legionella_human_controls.csv \\
    --out results_summary.csv

  # Same, but also left-join LLM columns onto the alignment table
  python pipelines/auto_lit/scripts/export_results_by_query.py \\
    --alignment-csv alignment_input/legionella_human_controls.csv \\
    --merged-out alignment_input/legionella_human_controls_with_llm.csv

  # Query ID list + optional pack copy
  python pipelines/auto_lit/scripts/export_results_by_query.py \\
    --query-ids controls.txt --out results_summary.csv --copy-packs packs/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_SCRIPTS))

from auto_lit_search.env_config import auto_lit_data_root  # noqa: E402

from summarize_results_table import (  # noqa: E402
    AlignmentSummaryRow,
    summarize_alignment,
)

RESULT_SUFFIXES = (
    "_results.json",
    "_graded.json",
    "_analysis.json",
)

# Columns written by summarize_results_table.AlignmentSummaryRow.as_dict()
LLM_SUMMARY_FIELDS = [
    "alignment_id",
    "query",
    "target",
    "pair_priority_score",
    "pair_priority_tier",
    "host_exploitation_score",
    "host_exploitation_tier",
    "query_effector_score",
    "query_effector_tier",
    "mimicry_plausibility_score",
    "mimicry_plausibility_tier",
    "headline",
    "best_host_paper",
    "best_query_paper",
    "synthesis_status",
    "n_host_nonzero",
    "n_query_nonzero",
    "main_uncertainties",
]

# LLM cols appended onto an alignment table (keep alignment query/target)
LLM_MERGE_FIELDS = [c for c in LLM_SUMMARY_FIELDS if c not in ("query", "target")]


def _normalize_id(value: str) -> str:
    return str(value or "").strip()


def alignment_id_for_pair(query: str, target: str) -> str:
    return f"{query}_{target}".replace("/", "_").replace(" ", "_")


def _looks_like_alignment_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except (OSError, csv.Error, StopIteration):
        return False
    cols = {c.strip().lower() for c in header}
    return "query" in cols and "target" in cols


def load_alignment_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        fields = {c.strip(): c for c in reader.fieldnames}
        if "query" not in fields or "target" not in fields:
            raise ValueError(
                f"Alignment CSV must have query and target columns: {path}"
            )
        q_col, t_col = fields["query"], fields["target"]
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {k: ("" if v is None else str(v)) for k, v in raw.items()}
            row[q_col] = _normalize_id(row.get(q_col, ""))
            row[t_col] = _normalize_id(row.get(t_col, ""))
            if row[q_col] and row[t_col]:
                rows.append(row)
        return rows


def load_query_ids(
    *,
    query_ids_file: Optional[Path],
    query_ids: Sequence[str],
    alignment_csv: Optional[Path],
) -> Set[str]:
    out: Set[str] = set()
    for raw in query_ids:
        for part in str(raw).split(","):
            q = _normalize_id(part)
            if q:
                out.add(q)

    if query_ids_file is not None:
        if _looks_like_alignment_csv(query_ids_file):
            for row in load_alignment_rows(query_ids_file):
                q = _normalize_id(row.get("query", ""))
                if q:
                    out.add(q)
        else:
            text = query_ids_file.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                for part in line.replace("\t", ",").split(","):
                    q = _normalize_id(part)
                    if q:
                        out.add(q)

    if alignment_csv is not None:
        for row in load_alignment_rows(alignment_csv):
            q = _normalize_id(row.get("query", ""))
            if q:
                out.add(q)

    return out


def query_from_results_payload(data: dict, alignment_id: str) -> str:
    query = _normalize_id(str(data.get("query") or ""))
    if query:
        return query
    parts = alignment_id.split("_", 1)
    return parts[0] if parts else alignment_id


def iter_alignment_ids(output_root: Path) -> Iterable[str]:
    seen: Set[str] = set()
    for path in sorted(output_root.glob("*_results.json")):
        aid = path.name[: -len("_results.json")]
        if aid not in seen:
            seen.add(aid)
            yield aid
    for path in sorted(output_root.glob("*_analysis.json")):
        aid = path.name[: -len("_analysis.json")]
        if aid not in seen:
            seen.add(aid)
            yield aid
    for path in sorted(output_root.glob("*_graded.json")):
        aid = path.name[: -len("_graded.json")]
        if aid not in seen:
            seen.add(aid)
            yield aid


def resolve_query(output_root: Path, alignment_id: str) -> str:
    for suffix in ("_results.json", "_analysis.json", "_graded.json"):
        path = output_root / f"{alignment_id}{suffix}"
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return query_from_results_payload(data, alignment_id)
    return query_from_results_payload({}, alignment_id)


def candidate_alignment_ids_from_pairs(
    alignment_rows: Sequence[dict[str, str]],
) -> list[tuple[str, str]]:
    """Return (alignment_id, query) for explicit query/target pairs."""
    out: list[tuple[str, str]] = []
    seen: Set[str] = set()
    for row in alignment_rows:
        query = _normalize_id(row.get("query", ""))
        target = _normalize_id(row.get("target", ""))
        if not query or not target:
            continue
        aid = alignment_id_for_pair(query, target)
        if aid in seen:
            continue
        seen.add(aid)
        out.append((aid, query))
    return out


def matching_alignment_ids(
    output_root: Path,
    query_ids: Set[str],
    *,
    pair_candidates: Optional[Sequence[tuple[str, str]]] = None,
) -> list[tuple[str, str]]:
    wanted = {_normalize_id(q) for q in query_ids}
    matches: list[tuple[str, str]] = []

    if pair_candidates is not None:
        for aid, query in pair_candidates:
            if query not in wanted:
                continue
            # Prefer pairs that have at least one result artifact
            if any(
                (output_root / f"{aid}{suffix}").is_file() for suffix in RESULT_SUFFIXES
            ):
                matches.append((aid, query))
        return matches

    for aid in iter_alignment_ids(output_root):
        query = resolve_query(output_root, aid)
        if query in wanted:
            matches.append((aid, query))
    return matches


def summarize_matches(
    output_root: Path, matches: Sequence[tuple[str, str]]
) -> list[AlignmentSummaryRow]:
    rows: list[AlignmentSummaryRow] = []
    for alignment_id, _query in matches:
        results_path = output_root / f"{alignment_id}_results.json"
        graded_path = output_root / f"{alignment_id}_graded.json"
        if not results_path.is_file() or not graded_path.is_file():
            print(
                f"Skipping {alignment_id}: missing results/graded JSON",
                file=sys.stderr,
            )
            continue
        try:
            row = summarize_alignment(results_path, graded_path)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            print(f"Skipping summary for {alignment_id}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (-r.pair_priority_score, r.alignment_id))
    return rows


def write_summary_csv(rows: Sequence[AlignmentSummaryRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.as_dict().get(k, "") for k in LLM_SUMMARY_FIELDS})


def merge_alignment_with_llm(
    alignment_rows: Sequence[dict[str, str]],
    summary_rows: Sequence[AlignmentSummaryRow],
) -> tuple[list[str], list[dict[str, Any]]]:
    by_pair = {
        (_normalize_id(r.query), _normalize_id(r.target)): r for r in summary_rows
    }
    if not alignment_rows:
        return LLM_SUMMARY_FIELDS, [r.as_dict() for r in summary_rows]

    base_fields = list(alignment_rows[0].keys())
    # Preserve alignment column order; append LLM cols not already present
    fieldnames = list(base_fields)
    for col in LLM_MERGE_FIELDS:
        if col not in fieldnames:
            fieldnames.append(col)

    merged: list[dict[str, Any]] = []
    for row in alignment_rows:
        query = _normalize_id(row.get("query", ""))
        target = _normalize_id(row.get("target", ""))
        out = dict(row)
        hit = by_pair.get((query, target))
        if hit is not None:
            d = hit.as_dict()
            for col in LLM_MERGE_FIELDS:
                out[col] = d.get(col, "")
        else:
            for col in LLM_MERGE_FIELDS:
                out.setdefault(col, "")
        merged.append(out)
    return fieldnames, merged


def write_merged_csv(
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pack_paths(output_root: Path, alignment_id: str, *, include_logs: bool) -> list[Path]:
    paths: list[Path] = []
    for suffix in RESULT_SUFFIXES:
        p = output_root / f"{alignment_id}{suffix}"
        if p.is_file():
            paths.append(p)
    if include_logs:
        log_dir = output_root / "logs"
        if log_dir.is_dir():
            for p in sorted(log_dir.glob(f"{alignment_id}_*")):
                if p.is_file():
                    paths.append(p)
    return paths


def copy_pack(
    paths: Sequence[Path],
    output_root: Path,
    out_dir: Path,
) -> int:
    n = 0
    for src in paths:
        rel = src.relative_to(output_root)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def _build_parser() -> argparse.ArgumentParser:
    default_output_root = auto_lit_data_root() / "llm_results"
    p = argparse.ArgumentParser(
        description=(
            "Export scorecard-v2 LLM result rows (summarize_results_table schema) "
            "for selected query IDs, optionally merged onto an alignment CSV."
        )
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Shared llm_results directory (default: $AUTO_LIT_DATA_ROOT/llm_results)",
    )
    p.add_argument(
        "--alignment-csv",
        type=Path,
        default=None,
        help=(
            "Initial alignment table with query/target columns. Used to derive "
            "query IDs / candidate pairs, and as the left table for --merged-out."
        ),
    )
    p.add_argument(
        "--query-ids",
        type=Path,
        default=None,
        help=(
            "Query ID list (one per line) or an alignment CSV with a query column "
            "(# comments OK for text lists)"
        ),
    )
    p.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Query ID (repeatable; comma-separated values allowed)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Write summarize_results_table CSV here "
            "(default: <out-dir>/results_summary.csv if --out-dir set)"
        ),
    )
    p.add_argument(
        "--merged-out",
        type=Path,
        default=None,
        help=(
            "Left-join LLM summary columns onto --alignment-csv "
            "(requires --alignment-csv)"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory for pack copies / default summary path",
    )
    p.add_argument(
        "--copy-packs",
        action="store_true",
        help="Also copy *_results/_graded/_analysis JSON into --out-dir",
    )
    p.add_argument(
        "--include-logs",
        action="store_true",
        help="With --copy-packs, also copy logs/<alignment_id>_* files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches without writing outputs",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root: Path = args.output_root
    if not output_root.is_dir():
        print(f"Output root not found: {output_root}", file=sys.stderr)
        return 2

    alignment_rows: list[dict[str, str]] = []
    if args.alignment_csv is not None:
        if not args.alignment_csv.is_file():
            print(f"Alignment CSV not found: {args.alignment_csv}", file=sys.stderr)
            return 2
        alignment_rows = load_alignment_rows(args.alignment_csv)

    if args.merged_out is not None and not alignment_rows:
        print("--merged-out requires --alignment-csv", file=sys.stderr)
        return 2

    query_ids = load_query_ids(
        query_ids_file=args.query_ids,
        query_ids=args.query_id,
        alignment_csv=args.alignment_csv,
    )
    if not query_ids:
        print(
            "No query IDs provided (--alignment-csv, --query-ids, and/or --query-id)",
            file=sys.stderr,
        )
        return 2

    pair_candidates = (
        candidate_alignment_ids_from_pairs(alignment_rows) if alignment_rows else None
    )
    # If only a query-id list is given (no alignment CSV), scan llm_results by query.
    matches = matching_alignment_ids(
        output_root, query_ids, pair_candidates=pair_candidates
    )
    found_queries = {q for _, q in matches}
    missing = sorted(query_ids - found_queries)
    if missing:
        print(
            f"{len(missing)} query ID(s) had no alignments under {output_root}",
            file=sys.stderr,
        )
        for q in missing[:20]:
            print(f"  missing: {q}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)

    if not matches and not args.merged_out:
        print("No matching alignments found", file=sys.stderr)
        return 1

    summary_rows = summarize_matches(output_root, matches)

    out_path = args.out
    if out_path is None and args.out_dir is not None:
        out_path = args.out_dir / "results_summary.csv"
    if out_path is None and args.merged_out is None and not args.copy_packs:
        out_path = Path("results_summary.csv")

    if args.dry_run:
        print(f"Would summarize {len(summary_rows)} / {len(matches)} alignment(s)")
        if out_path is not None:
            print(f"Would write summary CSV -> {out_path}")
        if args.merged_out is not None:
            print(f"Would write merged alignment CSV -> {args.merged_out}")
        if args.copy_packs:
            if args.out_dir is None:
                print("--copy-packs requires --out-dir", file=sys.stderr)
                return 2
            print(f"Would copy packs -> {args.out_dir}")
        return 0

    if out_path is not None:
        write_summary_csv(summary_rows, out_path)
        print(f"Wrote {len(summary_rows)} LLM summary rows to {out_path}")

    if args.merged_out is not None:
        fieldnames, merged = merge_alignment_with_llm(alignment_rows, summary_rows)
        write_merged_csv(fieldnames, merged, args.merged_out)
        n_hit = sum(1 for r in merged if r.get("pair_priority_score") not in ("", None))
        print(
            f"Wrote merged alignment+LLM table ({n_hit}/{len(merged)} with scores) "
            f"to {args.merged_out}"
        )

    if args.copy_packs:
        if args.out_dir is None:
            print("--copy-packs requires --out-dir", file=sys.stderr)
            return 2
        args.out_dir.mkdir(parents=True, exist_ok=True)
        files_copied = 0
        for alignment_id, _query in matches:
            paths = pack_paths(
                output_root, alignment_id, include_logs=bool(args.include_logs)
            )
            files_copied += copy_pack(paths, output_root, args.out_dir)
        print(f"Copied {files_copied} result files to {args.out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
