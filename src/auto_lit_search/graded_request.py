"""Build RunAlignmentGradedRequest from on-disk graded artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_lit_search.analysis_packet import (
    Constraints,
    GradedPaper,
    RunAlignmentGradedRequest,
)
from auto_lit_search.graded_request_payload import (
    build_run_alignment_graded_payload,
    load_graded_json,
)


def build_run_alignment_graded_request(
    alignment_id: str,
    output_root: str | Path,
    *,
    graded_path: Optional[str | Path] = None,
    papers_root: Optional[str | Path] = None,
    instructions: str = "",
    instructions_file: Optional[str | Path] = None,
) -> RunAlignmentGradedRequest:
    payload = build_run_alignment_graded_payload(
        alignment_id,
        output_root,
        graded_path=graded_path,
        papers_root=papers_root,
        instructions=instructions,
        instructions_file=instructions_file,
    )
    constraints_raw = payload.get("constraints")
    constraints: Optional[Constraints] = None
    if isinstance(constraints_raw, dict):
        constraints = Constraints(
            max_tokens=constraints_raw.get("max_tokens"),
            temperature=constraints_raw.get("temperature"),
        )

    graded_papers: List[GradedPaper] = []
    for row in payload.get("graded_papers") or []:
        if not isinstance(row, dict):
            continue
        graded_papers.append(GradedPaper(**row))

    return RunAlignmentGradedRequest(
        alignment_id=str(payload.get("alignment_id") or alignment_id),
        papers_dir=str(payload.get("papers_dir") or ""),
        query=str(payload.get("query") or ""),
        target_id=str(payload.get("target_id") or ""),
        constraints=constraints,
        instructions=str(payload.get("instructions") or ""),
        output_root=str(payload.get("output_root") or output_root),
        gene_context=payload.get("gene_context"),
        graded_papers=graded_papers,
        grading_meta=payload.get("grading_meta")
        if isinstance(payload.get("grading_meta"), dict)
        else {},
    )
