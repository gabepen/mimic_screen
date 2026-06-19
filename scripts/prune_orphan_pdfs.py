#!/usr/bin/env python3
"""Delete orphan PDFs when canonical or exact matching text already exists."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_lit_search.download_manifest import (  # noqa: E402
    DOWNLOAD_MANIFEST_FILENAME,
    _canonical_paper_stem,
    _load_download_manifest,
    _paper_has_usable_text,
)


def _iter_alignment_paper_dirs(papers_root: Path) -> list[Path]:
    if not papers_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(papers_root.iterdir()):
        if child.is_dir():
            out.append(child)
    return out


def prune_orphan_pdfs(
    papers_root: str,
    *,
    dry_run: bool = True,
    update_manifest: bool = False,
) -> dict[str, int]:
    root = Path(papers_root)
    deleted = 0
    skipped = 0
    manifest_rows_cleared = 0

    for papers_dir in _iter_alignment_paper_dirs(root):
        pdf_dir = papers_dir / "pdf"
        if not pdf_dir.is_dir():
            continue
        manifest_path = papers_dir / DOWNLOAD_MANIFEST_FILENAME
        manifest_map = (
            _load_download_manifest(str(manifest_path))
            if manifest_path.is_file()
            else {}
        )
        manifest_dirty = False

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            base = pdf_path.stem
            if not _paper_has_usable_text(str(papers_dir), base):
                skipped += 1
                continue
            if dry_run:
                deleted += 1
                continue
            try:
                pdf_path.unlink()
                deleted += 1
            except OSError:
                skipped += 1
                continue

            if not manifest_map:
                continue
            canon = _canonical_paper_stem(base)
            for key, row in manifest_map.items():
                row_pdf = str(row.get("pdf_path") or "")
                if not row_pdf:
                    continue
                row_base = Path(row_pdf).stem
                if row_base == base or _canonical_paper_stem(row_base) == canon:
                    row["pdf_path"] = None
                    details = row.get("details")
                    if isinstance(details, dict):
                        details["pdf_docling_required"] = False
                    manifest_dirty = True
                    manifest_rows_cleared += 1

        if update_manifest and manifest_dirty and not dry_run:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                for row in manifest_map.values():
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "deleted_or_would_delete": deleted,
        "skipped": skipped,
        "manifest_rows_cleared": manifest_rows_cleared,
        "dry_run": int(dry_run),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "papers_root",
        help="Root directory containing per-alignment papers/ subdirs",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete PDFs (default is dry-run)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Clear pdf_path in download_manifest.jsonl for pruned PDFs",
    )
    args = parser.parse_args()
    stats = prune_orphan_pdfs(
        args.papers_root,
        dry_run=not args.apply,
        update_manifest=args.update_manifest,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
