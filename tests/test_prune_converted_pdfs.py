from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "auto_lit" / "scripts"))

from prune_converted_pdfs import prune_converted_pdfs  # noqa: E402


def test_prune_uses_canonical_text_name_and_updates_manifest(tmp_path: Path):
    alignment_id = "A_B"
    papers_root = tmp_path / "papers"
    papers_dir = papers_root / alignment_id
    pdf_dir = papers_dir / "pdf"
    output_root = tmp_path / "llm_results"
    scheduler_dir = tmp_path / "scheduler_state"
    pdf_dir.mkdir(parents=True)
    output_root.mkdir()
    scheduler_dir.mkdir()

    pdf_path = pdf_dir / "paper__query__s3.pdf"
    pdf_path.write_bytes(b"pdf")
    (papers_dir / "paper__query.txt").write_text("converted", encoding="utf-8")
    manifest_path = papers_dir / "download_manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "paper_id": "paper",
                "source": "query",
                "pdf_path": str(pdf_path),
                "pdf_docling_required": True,
                "details": {"pdf_docling_required": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / f"{alignment_id}_graded.json").write_text("{}", encoding="utf-8")
    (output_root / f"{alignment_id}_results.json").write_text("{}", encoding="utf-8")

    result = prune_converted_pdfs(
        papers_root,
        output_root,
        scheduler_dir,
        delete=True,
    )

    assert result.total_pdfs == 1
    assert not pdf_path.exists()
    assert result.per_alignment[0].manifest_rows_updated == 1
    manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_row["pdf_path"] is None
    assert manifest_row["pdf_docling_required"] is False
    assert manifest_row["details"]["pdf_docling_required"] is False
