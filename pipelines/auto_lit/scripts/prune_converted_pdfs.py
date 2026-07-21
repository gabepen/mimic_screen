#!/usr/bin/env python3
"""
Delete PDFs that have already been converted to .txt, only for alignments whose
analysis finished end-to-end.

Safety rules (aligned with auto_lit_search.download_node):
  - An alignment is "complete" only when both {alignment_id}_graded.json and
    {alignment_id}_results.json exist under output_root (same as _outputs_done).
  - Alignments with a scheduler state file in a non-terminal active state are
    skipped while analysis is still in progress (Docling / grading may need PDFs).
  - Completed alignments are always eligible, even if a stale scheduler state file
    remains on disk.
  - A PDF is deleted only when papers/<alignment_id>/{stem}.txt exists and is
    non-empty. PDFs without a converted .txt are never touched.
  - Incomplete alignments are never pruned, even if some papers already have .txt.

Default is dry-run; pass --delete to remove files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
from auto_lit_search.download_manifest import (  # noqa: E402
    DOWNLOAD_MANIFEST_FILENAME,
    _canonical_paper_stem,
    _load_download_manifest,
    _paper_has_usable_text,
)
from auto_lit_search.env_config import auto_lit_data_root  # noqa: E402

# Matches download_node.py scheduler states.
ACTIVE_SCHEDULER_STATES = frozenset(
    {
        "DOWNLOADING",
        "DOCLING_PENDING",
        "DOCLING_INFLIGHT",
        "GRADER_READY",
        "GRADER_INFLIGHT",
    }
)
TERMINAL_SCHEDULER_STATES = frozenset({"DONE", "FAILED"})


@dataclass
class AlignmentSummary:
    alignment_id: str
    skipped_reason: str = ""
    pdfs_deleted: int = 0
    bytes_deleted: int = 0
    pdfs_kept_no_txt: int = 0
    pdfs_kept_bytes: int = 0
    manifest_rows_updated: int = 0


@dataclass
class RunSummary:
    alignments_complete: int = 0
    alignments_skipped_incomplete: int = 0
    alignments_skipped_active: int = 0
    alignments_skipped_missing_dir: int = 0
    per_alignment: list[AlignmentSummary] = field(default_factory=list)

    @property
    def total_pdfs(self) -> int:
        return sum(a.pdfs_deleted for a in self.per_alignment)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes_deleted for a in self.per_alignment)


def _alignment_outputs_complete(output_root: Path, alignment_id: str) -> bool:
    graded = output_root / f"{alignment_id}_graded.json"
    results = output_root / f"{alignment_id}_results.json"
    return (
        graded.is_file()
        and results.is_file()
        and graded.stat().st_size > 0
        and results.stat().st_size > 0
    )


def _scheduler_state(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    state = str(data.get("state") or "").strip()
    return state or None


def _scheduler_looks_active(scheduler_state_dir: Path, alignment_id: str) -> bool:
    state = _scheduler_state(scheduler_state_dir / f"{alignment_id}.json")
    if state is None:
        return False
    if state in TERMINAL_SCHEDULER_STATES:
        return False
    return state in ACTIVE_SCHEDULER_STATES


def _clear_deleted_pdfs_from_manifest(
    papers_dir: Path,
    deleted_stems: set[str],
) -> int:
    manifest_path = papers_dir / DOWNLOAD_MANIFEST_FILENAME
    if not deleted_stems or not manifest_path.is_file():
        return 0

    manifest_map = _load_download_manifest(str(manifest_path))
    deleted_canonical = {_canonical_paper_stem(stem) for stem in deleted_stems}
    updated = 0
    for row in manifest_map.values():
        pdf_path = str(row.get("pdf_path") or "")
        if not pdf_path:
            continue
        row_stem = Path(pdf_path).stem
        if (
            row_stem not in deleted_stems
            and _canonical_paper_stem(row_stem) not in deleted_canonical
        ):
            continue
        row["pdf_path"] = None
        row["pdf_docling_required"] = False
        details = row.get("details")
        if isinstance(details, dict):
            details["pdf_docling_required"] = False
        updated += 1

    if updated:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in manifest_map.values():
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def _prune_alignment(
    alignment_id: str,
    papers_dir: Path,
    *,
    delete: bool,
) -> AlignmentSummary:
    summary = AlignmentSummary(alignment_id=alignment_id)
    pdf_dir = papers_dir / "pdf"
    if not pdf_dir.is_dir():
        return summary

    deleted_stems: set[str] = set()
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        if not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
            continue
        size = pdf_path.stat().st_size
        if not _paper_has_usable_text(str(papers_dir), pdf_path.stem):
            summary.pdfs_kept_no_txt += 1
            summary.pdfs_kept_bytes += size
            continue
        summary.pdfs_deleted += 1
        summary.bytes_deleted += size
        if delete:
            pdf_path.unlink()
            deleted_stems.add(pdf_path.stem)

    if delete:
        summary.manifest_rows_updated = _clear_deleted_pdfs_from_manifest(
            papers_dir,
            deleted_stems,
        )
        if pdf_dir.is_dir() and not any(pdf_dir.iterdir()):
            try:
                pdf_dir.rmdir()
            except OSError:
                pass

    return summary


def prune_converted_pdfs(
    papers_root: Path,
    output_root: Path,
    scheduler_state_dir: Path,
    *,
    delete: bool,
    alignment_ids: set[str] | None = None,
) -> RunSummary:
    run = RunSummary()

    if not papers_root.is_dir():
        raise FileNotFoundError(f"Papers directory not found: {papers_root}")

    alignment_dirs = sorted(
        p for p in papers_root.iterdir() if p.is_dir() and (not alignment_ids or p.name in alignment_ids)
    )

    for papers_dir in alignment_dirs:
        alignment_id = papers_dir.name
        item = AlignmentSummary(alignment_id=alignment_id)

        if not _alignment_outputs_complete(output_root, alignment_id):
            if _scheduler_looks_active(scheduler_state_dir, alignment_id):
                item.skipped_reason = (
                    "alignment incomplete and scheduler state is active "
                    "(pipeline may still need PDFs for Docling)"
                )
                run.alignments_skipped_active += 1
            else:
                item.skipped_reason = "alignment not complete (missing graded/results outputs)"
                run.alignments_skipped_incomplete += 1
            run.per_alignment.append(item)
            continue

        pruned = _prune_alignment(alignment_id, papers_dir, delete=delete)
        pruned.skipped_reason = ""
        run.alignments_complete += 1
        run.per_alignment.append(pruned)

    return run


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _print_summary(run: RunSummary, *, delete: bool) -> None:
    mode = "DELETE" if delete else "DRY-RUN"
    print(f"=== prune_converted_pdfs ({mode}) ===")
    print(f"Complete alignments pruned: {run.alignments_complete}")
    print(f"Skipped (incomplete):     {run.alignments_skipped_incomplete}")
    print(f"Skipped (active pipeline):  {run.alignments_skipped_active}")
    print(f"PDFs {'deleted' if delete else 'would delete'}: {run.total_pdfs:,} ({_format_gib(run.total_bytes)})")

    detailed = [
        a
        for a in run.per_alignment
        if a.skipped_reason or a.pdfs_deleted or a.pdfs_kept_no_txt
    ]
    if not detailed:
        print("No alignment directories matched.")
        return

    print()
    for item in detailed:
        if item.skipped_reason:
            print(f"  {item.alignment_id}: SKIP — {item.skipped_reason}")
            continue
        kept = ""
        if item.pdfs_kept_no_txt:
            kept = f", kept {item.pdfs_kept_no_txt} PDF(s) without .txt ({_format_gib(item.pdfs_kept_bytes)})"
        manifest = ""
        if item.manifest_rows_updated:
            manifest = f", updated {item.manifest_rows_updated} manifest row(s)"
        verb = "deleted" if delete else "would delete"
        print(
            f"  {item.alignment_id}: {verb} {item.pdfs_deleted} PDF(s) "
            f"({_format_gib(item.bytes_deleted)}){kept}{manifest}"
        )


def _build_parser() -> argparse.ArgumentParser:
    default_data_root = auto_lit_data_root()
    parser = argparse.ArgumentParser(
        description="Remove converted PDFs for alignments that finished analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --data-root /path/to/auto_lit_eval_data\n"
            "  %(prog)s --data-root /path/to/auto_lit_eval_data --delete\n"
            "  %(prog)s --alignment-id Q5ZT19_Q9NYA1 --delete\n"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=f"Dataset root (default: {default_data_root})",
    )
    parser.add_argument(
        "--papers-root",
        type=Path,
        default=None,
        help="Papers directory (default: <data-root>/papers)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="LLM output directory (default: <data-root>/llm_results)",
    )
    parser.add_argument(
        "--scheduler-state-dir",
        type=Path,
        default=None,
        help="Scheduler state directory (default: <data-root>/logs/scheduler_state)",
    )
    parser.add_argument(
        "--alignment-id",
        action="append",
        dest="alignment_ids",
        metavar="ID",
        help="Limit to one or more alignment IDs (repeatable)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_root: Path = args.data_root
    papers_root = args.papers_root or (data_root / "papers")
    output_root = args.output_root or (data_root / "llm_results")
    scheduler_state_dir = args.scheduler_state_dir or (data_root / "logs" / "scheduler_state")
    alignment_ids = set(args.alignment_ids) if args.alignment_ids else None

    try:
        run = prune_converted_pdfs(
            papers_root,
            output_root,
            scheduler_state_dir,
            delete=args.delete,
            alignment_ids=alignment_ids,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    _print_summary(run, delete=args.delete)
    if not args.delete and run.total_pdfs:
        print("\nRe-run with --delete to remove these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
