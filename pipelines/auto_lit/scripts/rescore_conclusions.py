#!/usr/bin/env python3
"""Rebuild conclusion scorecards from existing graded + synthesis text (no LLM).

Re-applies current synthesis_scorecard rules (synthesis Quick results are
authoritative; rubric_indices stay diagnostic). Does not regrade papers or
rewrite synthesis prose.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Set

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from auto_lit_search.env_config import auto_lit_data_root  # noqa: E402
from auto_lit_search.synthesis_scorecard import (  # noqa: E402
    build_conclusion,
    graded_papers_from_json,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _alignment_ids_from_csv(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "alignment_id" not in reader.fieldnames:
            raise ValueError(f"{path} must have an alignment_id column")
        return {str(row["alignment_id"]).strip() for row in reader if row.get("alignment_id")}


def _iter_alignment_ids(output_root: Path, only: Optional[Set[str]]) -> Iterable[str]:
    for graded in sorted(output_root.glob("*_graded.json")):
        aid = graded.name[: -len("_graded.json")]
        if only is not None and aid not in only:
            continue
        if not (output_root / f"{aid}_results.json").is_file():
            continue
        yield aid


def rescore_alignment(
    output_root: Path,
    alignment_id: str,
    *,
    backup: bool,
    dry_run: bool,
    ts: str,
) -> tuple[str, str, str]:
    graded_path = output_root / f"{alignment_id}_graded.json"
    results_path = output_root / f"{alignment_id}_results.json"
    if not graded_path.is_file() or not results_path.is_file():
        return alignment_id, "skipped", "missing graded/results"

    graded = _load_json(graded_path)
    results = _load_json(results_path)
    papers = graded_papers_from_json(graded)
    syn = results.get("synthesis") or {}
    if isinstance(syn, dict):
        synthesis_text = str(syn.get("text") or syn.get("discussion") or "")
    else:
        synthesis_text = str(syn or "")
    old = results.get("conclusion") if isinstance(results.get("conclusion"), dict) else {}
    status = str(old.get("synthesis_status") or "ok")
    if not synthesis_text.strip():
        status = "grades_only"

    new_conclusion = build_conclusion(
        papers,
        synthesis_text,
        synthesis_status=status,
    )
    old_mim = (old.get("mimicry_plausibility") or {}).get("score")
    new_mim = new_conclusion["mimicry_plausibility"]["score"]
    old_pair = (old.get("pair_priority") or {}).get("score")
    new_pair = new_conclusion["pair_priority"]["score"]
    note = (
        f"mimicry {old_mim}->{new_mim} "
        f"({new_conclusion['mimicry_plausibility']['tier']}); "
        f"pair {old_pair}->{new_pair} "
        f"({new_conclusion['pair_priority']['tier']})"
    )

    if dry_run:
        return alignment_id, "dry_run", note

    if backup and results_path.is_file():
        bak = results_path.with_name(f"{results_path.name}.bak.{ts}")
        if not bak.is_file():
            shutil.copy2(results_path, bak)

    results["conclusion"] = new_conclusion
    _write_json(results_path, results)
    return alignment_id, "updated", note


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-rescore conclusion blocks with current scorecard rules."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=auto_lit_data_root() / "llm_results",
    )
    parser.add_argument(
        "--alignment-ids",
        type=str,
        default="",
        help="Comma-separated alignment_ids (default: all with graded+results).",
    )
    parser.add_argument(
        "--from-summary-csv",
        type=Path,
        default=None,
        help="Restrict to alignment_id values listed in this CSV.",
    )
    parser.add_argument("--backup", action="store_true", help="Write *.bak.<ts> before update.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root: Path = args.output_root
    if not output_root.is_dir():
        print(f"Output root not found: {output_root}", file=sys.stderr)
        return 2

    only: Optional[Set[str]] = None
    if args.from_summary_csv:
        only = _alignment_ids_from_csv(args.from_summary_csv)
    if args.alignment_ids.strip():
        ids = {x.strip() for x in args.alignment_ids.split(",") if x.strip()}
        only = ids if only is None else (only & ids)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n_ok = n_skip = n_err = 0
    for aid in _iter_alignment_ids(output_root, only):
        try:
            alignment_id, status, note = rescore_alignment(
                output_root,
                aid,
                backup=args.backup,
                dry_run=args.dry_run,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"{aid}\terror\t{exc}", file=sys.stderr)
            n_err += 1
            continue
        print(f"{alignment_id}\t{status}\t{note}")
        if status == "updated" or status == "dry_run":
            n_ok += 1
        else:
            n_skip += 1

    print(
        f"Done: processed={n_ok} skipped={n_skip} errors={n_err} dry_run={args.dry_run}",
        file=sys.stderr,
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
