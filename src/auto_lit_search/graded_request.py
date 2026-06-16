"""Build RunAlignmentGradedRequest from on-disk graded artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from auto_lit_search.analysis_packet import (
    Constraints,
    GradedPaper,
    RunAlignmentGradedRequest,
)
from auto_lit_search.rubric_scoring import (
    resolve_axis_rationales,
    rubric_role_for_paper_role,
)


def _load_rubric_json(path: str | Path) -> Dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _rubric_for_paper_role(
    paper_role: str | None,
    host_rubric: Dict[str, Any] | None,
    microbe_rubric: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    role = rubric_role_for_paper_role(paper_role or "")
    return microbe_rubric if role == "microbe" else host_rubric


def load_graded_json(graded_path: Path) -> Dict[str, Any]:
    with graded_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {graded_path}")
    return data


def _meta_from_sidecar(
    alignment_id: str,
    output_root: Path,
    papers_root: Optional[Path],
) -> Dict[str, Any]:
    for name in (f"{alignment_id}_results.json", f"{alignment_id}_analysis.json"):
        path = output_root / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        papers_dir = str(data.get("papers_dir") or "").strip()
        if not papers_dir and papers_root is not None:
            papers_dir = str(papers_root / alignment_id)
        return {
            "query": str(data.get("query") or "").strip(),
            "target_id": str(data.get("target_id") or "").strip(),
            "papers_dir": papers_dir,
            "gene_context": data.get("gene_context"),
            "constraints": data.get("constraints") or data.get("synthesis", {}).get(
                "constraints"
            ),
        }
    if papers_root is not None:
        parts = alignment_id.split("_", 1)
        query = parts[0] if parts else alignment_id
        target_id = parts[1] if len(parts) > 1 else ""
        return {
            "query": query,
            "target_id": target_id,
            "papers_dir": str(papers_root / alignment_id),
            "gene_context": None,
            "constraints": None,
        }
    raise FileNotFoundError(
        f"No results/analysis sidecar for {alignment_id} under {output_root}"
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
    out_root = Path(output_root)
    gpath = Path(graded_path) if graded_path else out_root / f"{alignment_id}_graded.json"
    if not gpath.is_file():
        raise FileNotFoundError(f"Missing graded file: {gpath}")

    graded = load_graded_json(gpath)
    papers_root_p = Path(papers_root) if papers_root else None
    meta = _meta_from_sidecar(alignment_id, out_root, papers_root_p)

    host_rubric = _load_rubric_json(os.environ.get("HOST_RUBRIC_PATH", ""))
    microbe_rubric = _load_rubric_json(os.environ.get("MICROBE_RUBRIC_PATH", ""))

    instr = (instructions or "").strip()
    if instructions_file:
        ip = Path(instructions_file)
        if ip.is_file():
            instr = ip.read_text(encoding="utf-8").strip()
    if not instr:
        instr = (
            "Synthesize host and microbe rubric evidence for this structurally similar pair."
        )

    constraints_raw = meta.get("constraints")
    constraints: Optional[Constraints] = None
    if isinstance(constraints_raw, dict):
        constraints = Constraints(
            max_tokens=constraints_raw.get("max_tokens"),
            temperature=constraints_raw.get("temperature"),
        )

    graded_papers: List[GradedPaper] = []
    for row in graded.get("graded_papers") or []:
        if not isinstance(row, dict):
            continue
        tags = row.get("rubric_tags")
        if not isinstance(tags, dict):
            tags = {}
        criterion_scores = row.get("criterion_scores") or {}
        rubric = _rubric_for_paper_role(row.get("paper_role"), host_rubric, microbe_rubric)
        axis_rationales = resolve_axis_rationales(
            rubric,
            criterion_scores if isinstance(criterion_scores, dict) else {},
            row.get("rubric_axis_rationales") if isinstance(row.get("rubric_axis_rationales"), dict) else {},
        )
        graded_papers.append(
            GradedPaper(
                paper_id=str(row.get("paper_id") or ""),
                file_name=str(row.get("file_name") or ""),
                paper_role=row.get("paper_role"),
                grading_schema_version=int(row.get("grading_schema_version") or 1),
                relevance_grade=float(row.get("relevance_grade") or 0.0),
                relevance_sort=int(row.get("relevance_sort") or 0),
                paper_grade=str(row.get("paper_grade") or ""),
                primary_grade=str(row.get("primary_grade") or ""),
                criterion_scores=criterion_scores,
                axis_totals=row.get("axis_totals") or {},
                rubric_dimension_scores={
                    str(k): float(v)
                    for k, v in (row.get("rubric_dimension_scores") or {}).items()
                },
                rubric_axis_rationales={
                    str(k): str(v) for k, v in axis_rationales.items()
                },
                mention_type=row.get("mention_type"),
                infection_naive=row.get("infection_naive"),
                no_meaningful_mention=bool(row.get("no_meaningful_mention")),
                claim_summary=str(row.get("claim_summary") or ""),
                rationale=str(row.get("rationale") or ""),
                rubric_tags={str(k): str(v) for k, v in tags.items()},
                model_output=row.get("model_output"),
                notes=row.get("notes"),
            )
        )

    grading_meta = graded.get("grading_meta")
    if not isinstance(grading_meta, dict):
        grading_meta = {}

    query = str(graded.get("query") or meta.get("query") or "").strip()
    target_id = str(graded.get("target_id") or meta.get("target_id") or "").strip()
    papers_dir = str(graded.get("papers_dir") or meta.get("papers_dir") or "").strip()

    return RunAlignmentGradedRequest(
        alignment_id=str(graded.get("alignment_id") or alignment_id),
        papers_dir=papers_dir,
        query=query,
        target_id=target_id,
        constraints=constraints,
        instructions=instr,
        output_root=str(out_root),
        gene_context=meta.get("gene_context"),
        graded_papers=graded_papers,
        grading_meta=grading_meta,
    )
