#!/usr/bin/env python3
"""Score a filled human blind grading sheet using the same rubric math as the LLM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.rubric_scoring import (  # noqa: E402
    aggregate_paper_scores,
    compute_axis_totals,
    primary_axis_for_rubric,
    rubric_role_for_paper_role,
)
from sample_blind_grading_audit import (  # noqa: E402
    _DEFAULT_HOST_RUBRIC,
    _DEFAULT_MICROBE_RUBRIC,
    load_rubric_spec,
)

_DEFAULT_OUT_NAME = "human_grading_scores.json"
_HUMAN_COL_PREFIX = "human_"
_HUMAN_NOTES_COL = "human_notes"
_CRIT_SCORE_RE = re.compile(r"([a-z][a-z0-9_]+)=(\d+)")
_CRIT_SPLIT_RE = re.compile(r";\s*(?=[a-z][a-z0-9_]+=\d+)")


def parse_axis_rationales(
    rubric_axis_rationales: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Parse axis_id -> {criterion_id -> {score, note}} from rubric_axis_rationales."""
    by_axis: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for axis_id, axis_text in (rubric_axis_rationales or {}).items():
        crits: Dict[str, Dict[str, Any]] = {}
        for part in _CRIT_SPLIT_RE.split(str(axis_text)):
            part = part.strip()
            if not part:
                continue
            match = re.match(r"([a-z][a-z0-9_]+)=(\d+)(?::\s*(.*))?", part, re.DOTALL)
            if not match:
                continue
            crit_id = match.group(1)
            crits[crit_id] = {
                "score": max(0, min(2, int(match.group(2)))),
                "note": (match.group(3) or "").strip(),
            }
        if crits:
            by_axis[str(axis_id)] = crits
    return by_axis


def parse_criterion_scores_from_rationales(
    rubric_axis_rationales: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Recover LLM criterion 0/1/2 scores and notes from rubric_axis_rationales."""
    scores: Dict[str, Dict[str, Any]] = {}
    for crits in parse_axis_rationales(rubric_axis_rationales).values():
        for crit_id, entry in crits.items():
            scores[crit_id] = {
                "score": entry["score"],
                "note": entry.get("note") or "",
            }
    return scores


def _axis_labels(rubric: Dict[str, Any]) -> Dict[str, str]:
    return {
        str(ax.get("id") or ""): str(ax.get("label") or ax.get("id") or "")
        for ax in (rubric.get("axes") or [])
        if isinstance(ax, dict) and ax.get("id")
    }


def _criterion_labels(rubric: Dict[str, Any]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for ax in rubric.get("axes") or []:
        if not isinstance(ax, dict):
            continue
        for crit in ax.get("criteria") or []:
            if not isinstance(crit, dict) or not crit.get("id"):
                continue
            labels[str(crit["id"])] = str(crit.get("label") or crit["id"])
    return labels


def discover_human_criterion_columns(fieldnames: Optional[List[str]]) -> List[str]:
    """Return criterion ids from human_<criterion_id> headers (any column order)."""
    if not fieldnames:
        return []
    out: List[str] = []
    for col in fieldnames:
        if not col.startswith(_HUMAN_COL_PREFIX) or col == _HUMAN_NOTES_COL:
            continue
        crit_id = col[len(_HUMAN_COL_PREFIX) :].strip()
        if crit_id:
            out.append(crit_id)
    return out


def _read_human_criterion_scores(
    row: Dict[str, str],
    *,
    valid_criterion_ids: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Read filled human_<criterion_id> cells from a row, ignoring column order."""
    scores: Dict[str, Dict[str, Any]] = {}
    for col, raw_value in row.items():
        if not col.startswith(_HUMAN_COL_PREFIX) or col == _HUMAN_NOTES_COL:
            continue
        crit_id = col[len(_HUMAN_COL_PREFIX) :].strip()
        if not crit_id:
            continue
        if valid_criterion_ids is not None and crit_id not in valid_criterion_ids:
            continue
        raw = str(raw_value or "").strip()
        if not raw:
            continue
        try:
            score = max(0, min(2, int(float(raw))))
        except ValueError:
            continue
        scores[crit_id] = {"score": score, "note": ""}
    return scores


def score_sheet_row(
    row: Dict[str, str],
    rubric_spec,
) -> Optional[Dict[str, Any]]:
    paper_role = str(row.get("paper_role") or "").strip()
    if not paper_role:
        return None
    rubric = rubric_spec.rubric_for_role(paper_role)
    valid_ids = set(rubric_spec.criteria_for_role(paper_role))
    criterion_scores = _read_human_criterion_scores(
        row, valid_criterion_ids=valid_ids
    )
    if not criterion_scores:
        return None
    rubric_role = rubric_role_for_paper_role(paper_role)
    agg = aggregate_paper_scores(
        rubric, criterion_scores, rubric_role=rubric_role
    )
    return {
        "sample_id": row.get("sample_id"),
        "paper_role": paper_role,
        "criterion_scores": criterion_scores,
        **agg,
    }


def llm_aggregate_from_answer_key(
    entry: Dict[str, Any],
    rubric_spec,
) -> Optional[Dict[str, Any]]:
    """Build LLM paper_grade/axis_totals using the same math as human scoring."""
    paper_role = str(entry.get("paper_role") or "").strip()
    if not paper_role:
        return None

    rubric = rubric_spec.rubric_for_role(paper_role)
    rubric_role = rubric_role_for_paper_role(paper_role)

    if entry.get("criterion_scores"):
        criterion_scores = {
            str(k): dict(v)
            for k, v in entry["criterion_scores"].items()
            if isinstance(v, dict)
        }
        agg = aggregate_paper_scores(
            rubric, criterion_scores, rubric_role=rubric_role
        )
        agg["criterion_scores"] = criterion_scores
        return agg

    if entry.get("paper_grade") and entry.get("axis_totals"):
        return {
            "paper_grade": entry.get("paper_grade"),
            "primary_grade": entry.get("primary_grade"),
            "axis_totals": entry.get("axis_totals") or {},
            "relevance_grade": entry.get("relevance_grade"),
            "criterion_scores": entry.get("criterion_scores") or {},
        }

    criterion_scores = parse_criterion_scores_from_rationales(
        entry.get("rubric_axis_rationales") or {}
    )
    if criterion_scores:
        agg = aggregate_paper_scores(
            rubric, criterion_scores, rubric_role=rubric_role
        )
        agg["criterion_scores"] = criterion_scores
        agg["llm_source"] = "rubric_axis_rationales"
        return agg

    dim_scores = entry.get("rubric_dimension_scores") or {}
    if not isinstance(dim_scores, dict) or not dim_scores:
        return None

    axis_totals_raw = compute_axis_totals(rubric, {})
    axis_totals: Dict[str, Any] = {}
    total_score = 0
    total_max = 0
    for axis_id, axis_total in axis_totals_raw.items():
        norm = float(dim_scores.get(axis_id) or 0.0)
        score = round(norm * axis_total.max_score)
        axis_totals[axis_id] = {
            "score": score,
            "max": axis_total.max_score,
            "label": f"{score}/{axis_total.max_score}",
        }
        total_score += score
        total_max += axis_total.max_score

    primary_axis_id = primary_axis_for_rubric(rubric, rubric_role)
    primary = axis_totals.get(primary_axis_id) or {}
    return {
        "paper_grade": f"{total_score}/{total_max}",
        "primary_grade": primary.get("label", ""),
        "axis_totals": axis_totals,
        "relevance_grade": entry.get("relevance_grade"),
        "criterion_scores": {},
        "rubric_dimension_scores": dim_scores,
        "_approximated_from_dimension_scores": True,
    }


def _compare_to_answer_key(
    human: Dict[str, Any],
    llm: Dict[str, Any],
) -> Dict[str, Any]:
    human_axis = human.get("axis_totals") or {}
    llm_axis = llm.get("axis_totals") or {}
    axis_comparison: Dict[str, Dict[str, Any]] = {}
    for axis_id in sorted(set(human_axis) | set(llm_axis)):
        h = human_axis.get(axis_id) or {}
        l = llm_axis.get(axis_id) or {}
        h_score = int((h.get("score") if isinstance(h, dict) else 0) or 0)
        l_score = int((l.get("score") if isinstance(l, dict) else 0) or 0)
        axis_comparison[axis_id] = {
            "human": h.get("label") if isinstance(h, dict) else "",
            "llm": l.get("label") if isinstance(l, dict) else "",
            "delta": h_score - l_score,
        }

    human_crit = human.get("criterion_scores") or {}
    llm_crit = llm.get("criterion_scores") or {}
    llm_notes_by_crit = {
        crit_id: str(entry.get("note") or "")
        for crit_id, entry in llm_crit.items()
        if isinstance(entry, dict)
    }
    criterion_comparison: Dict[str, Dict[str, Any]] = {}
    for crit_id in sorted(set(human_crit) | set(llm_crit)):
        h_score = int((human_crit.get(crit_id) or {}).get("score", 0))
        l_score = int((llm_crit.get(crit_id) or {}).get("score", 0))
        if crit_id in human_crit or crit_id in llm_crit:
            criterion_comparison[crit_id] = {
                "human": h_score if crit_id in human_crit else None,
                "llm": l_score if crit_id in llm_crit else None,
                "delta": h_score - l_score,
                "llm_reasoning": llm_notes_by_crit.get(crit_id, ""),
            }

    return {
        "paper_grade": {
            "human": str(human.get("paper_grade") or ""),
            "llm": str(llm.get("paper_grade") or ""),
            "match": str(human.get("paper_grade") or "")
            == str(llm.get("paper_grade") or ""),
        },
        "primary_grade": {
            "human": human.get("primary_grade"),
            "llm": llm.get("primary_grade"),
            "match": human.get("primary_grade") == llm.get("primary_grade"),
        },
        "axis_comparison": axis_comparison,
        "criterion_comparison": criterion_comparison,
        "n_criteria_exact_match": sum(
            1
            for c in criterion_comparison.values()
            if c.get("human") is not None
            and c.get("llm") is not None
            and c.get("delta") == 0
        ),
        "n_criteria_compared": sum(
            1
            for c in criterion_comparison.values()
            if c.get("human") is not None and c.get("llm") is not None
        ),
        "llm_source": (
            "approximated_from_dimension_scores"
            if llm.get("_approximated_from_dimension_scores")
            else llm.get("llm_source", "criterion_scores")
        ),
    }


def build_detailed_report(
    sample_id: str,
    row: Dict[str, str],
    human: Dict[str, Any],
    llm_entry: Dict[str, Any],
    comparison: Dict[str, Any],
    rubric_spec,
) -> Dict[str, Any]:
    paper_role = str(human.get("paper_role") or "")
    rubric = rubric_spec.rubric_for_role(paper_role)
    axis_labels = _axis_labels(rubric)
    crit_labels = _criterion_labels(rubric)
    llm_by_axis = parse_axis_rationales(llm_entry.get("rubric_axis_rationales") or {})
    human_crit = human.get("criterion_scores") or {}
    llm_crit = llm_entry.get("criterion_scores") or {}
    if not llm_crit:
        llm_crit = parse_criterion_scores_from_rationales(
            llm_entry.get("rubric_axis_rationales") or {}
        )

    axes_out: List[Dict[str, Any]] = []
    for ax in rubric.get("axes") or []:
        if not isinstance(ax, dict) or not ax.get("id"):
            continue
        axis_id = str(ax["id"])
        axis_total = (comparison.get("axis_comparison") or {}).get(axis_id, {})
        criteria_out: List[Dict[str, Any]] = []
        for crit in ax.get("criteria") or []:
            if not isinstance(crit, dict) or not crit.get("id"):
                continue
            weight = str(crit.get("weight") or "medium").lower()
            crit_id = str(crit["id"])
            if weight == "flag":
                continue
            llm_part = llm_by_axis.get(axis_id, {}).get(crit_id) or llm_crit.get(crit_id) or {}
            human_part = human_crit.get(crit_id) or {}
            comp = (comparison.get("criterion_comparison") or {}).get(crit_id, {})
            criteria_out.append(
                {
                    "criterion_id": crit_id,
                    "label": crit_labels.get(crit_id, crit_id),
                    "weight": weight,
                    "human_score": human_part.get("score") if crit_id in human_crit else None,
                    "llm_score": llm_part.get("score") if llm_part else None,
                    "delta": comp.get("delta"),
                    "llm_reasoning": str(llm_part.get("note") or comp.get("llm_reasoning") or ""),
                }
            )
        if criteria_out:
            axes_out.append(
                {
                    "axis_id": axis_id,
                    "axis_label": axis_labels.get(axis_id, axis_id),
                    "axis_total": axis_total,
                    "criteria": criteria_out,
                }
            )

    flags: Dict[str, str] = {}
    for ax in rubric.get("axes") or []:
        if not isinstance(ax, dict):
            continue
        for crit in ax.get("criteria") or []:
            if not isinstance(crit, dict):
                continue
            if str(crit.get("weight") or "").lower() != "flag":
                continue
            crit_id = str(crit.get("id") or "")
            tag_val = (llm_entry.get("rubric_tags") or {}).get(crit_id)
            if tag_val:
                flags[crit_id] = str(tag_val)

    return {
        "sample_id": sample_id,
        "doi": row.get("doi"),
        "gene_focus_id": row.get("gene_focus_id"),
        "gene_focus_symbol": row.get("gene_focus_symbol"),
        "paper_role": paper_role,
        "text_path": row.get("text_path"),
        "paper_grade": comparison.get("paper_grade"),
        "primary_grade": comparison.get("primary_grade"),
        "llm_claim_summary": llm_entry.get("rationale"),
        "human_notes": row.get("human_notes"),
        "flags": flags,
        "axes": axes_out,
    }


def format_report_markdown(reports: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines = [
        "# Human vs LLM grading report",
        "",
        f"- Sheet: `{summary.get('sheet', '')}`",
        f"- Graded rows: {summary.get('n_scored', 0)} / {summary.get('n_rows', 0)}",
        f"- Exact paper_grade match: {summary.get('n_exact_paper_grade_match', 'n/a')}",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report.get('sample_id')} — {report.get('doi')}",
                "",
                f"- Gene: `{report.get('gene_focus_id')}` ({report.get('gene_focus_symbol')})",
                f"- Role: {report.get('paper_role')}",
                f"- Paper grade: human **{report['paper_grade']['human']}** vs LLM **{report['paper_grade']['llm']}**",
                f"- Primary grade: human **{report['primary_grade']['human']}** vs LLM **{report['primary_grade']['llm']}**",
                "",
            ]
        )
        if report.get("llm_claim_summary"):
            lines.extend(["**LLM claim summary:**", str(report["llm_claim_summary"]), ""])
        if report.get("human_notes"):
            lines.extend(["**Human notes:**", str(report["human_notes"]), ""])

        for axis in report.get("axes") or []:
            total = axis.get("axis_total") or {}
            lines.extend(
                [
                    f"### {axis.get('axis_label')} (`{axis.get('axis_id')}`)",
                    "",
                    f"Axis total: human **{total.get('human', '?')}** vs LLM **{total.get('llm', '?')}** (Δ {total.get('delta', '?')})",
                    "",
                    "| Criterion | Weight | Human | LLM | Δ | LLM reasoning |",
                    "|-----------|--------|-------|-----|---|---------------|",
                ]
            )
            for crit in axis.get("criteria") or []:
                h = crit.get("human_score")
                l = crit.get("llm_score")
                d = crit.get("delta")
                note = str(crit.get("llm_reasoning") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {crit.get('label')} | {crit.get('weight')} | "
                    f"{h if h is not None else '—'} | {l if l is not None else '—'} | "
                    f"{d if d is not None else '—'} | {note} |"
                )
            lines.append("")

        if report.get("flags"):
            lines.append("**LLM flags:**")
            for flag_id, flag_val in report["flags"].items():
                lines.append(f"- `{flag_id}`: {flag_val}")
            lines.append("")

        if report.get("text_path"):
            lines.extend([f"Paper: `{report['text_path']}`", ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sheet", type=Path, required=True, help="Filled blind_grading_sheet.csv")
    p.add_argument("--host-rubric", type=Path, default=_DEFAULT_HOST_RUBRIC)
    p.add_argument("--microbe-rubric", type=Path, default=_DEFAULT_MICROBE_RUBRIC)
    p.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help="llm_grades_answer_key.json for side-by-side comparison",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output JSON (default: <sheet-dir>/{_DEFAULT_OUT_NAME})",
    )
    p.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Detailed markdown report (default: <sheet-dir>/human_grading_report.md when --answer-key set)",
    )
    args = p.parse_args()

    if not args.sheet.is_file():
        print(f"Not found: {args.sheet}", file=sys.stderr)
        return 2
    if not args.host_rubric.is_file() or not args.microbe_rubric.is_file():
        print("Rubric JSON path not found", file=sys.stderr)
        return 2

    rubric_spec = load_rubric_spec(args.host_rubric, args.microbe_rubric)
    answer_key: Dict[str, Any] = {}
    if args.answer_key:
        if not args.answer_key.is_file():
            print(f"Not found: {args.answer_key}", file=sys.stderr)
            return 2
        answer_key = json.loads(args.answer_key.read_text(encoding="utf-8"))

    out_path = args.out or (args.sheet.parent / _DEFAULT_OUT_NAME)
    report_md_path = args.report_md
    if report_md_path is None and args.answer_key:
        report_md_path = args.sheet.parent / "human_grading_report.md"

    results: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    detailed_reports: List[Dict[str, Any]] = []
    n_rows = 0
    n_ungraded = 0

    with args.sheet.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        criterion_columns = discover_human_criterion_columns(reader.fieldnames)
        for row in reader:
            sample_id = str(row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            n_rows += 1
            scored = score_sheet_row(row, rubric_spec)
            if scored is None:
                n_ungraded += 1
                continue
            results[sample_id] = scored
            llm_entry = answer_key.get(sample_id)
            if isinstance(llm_entry, dict):
                llm_agg = llm_aggregate_from_answer_key(llm_entry, rubric_spec)
                if llm_agg is not None:
                    comparison = _compare_to_answer_key(scored, llm_agg)
                    comparisons[sample_id] = comparison
                    detailed_reports.append(
                        build_detailed_report(
                            sample_id, row, scored, llm_entry, comparison, rubric_spec
                        )
                    )

    payload = {
        "sheet": str(args.sheet),
        "host_rubric": str(args.host_rubric),
        "microbe_rubric": str(args.microbe_rubric),
        "criterion_columns": criterion_columns,
        "n_rows": n_rows,
        "n_scored": len(results),
        "n_ungraded": n_ungraded,
        "scores": results,
    }
    if comparisons:
        payload["llm_comparison"] = comparisons
        payload["detailed_report"] = detailed_reports
        payload["n_exact_paper_grade_match"] = sum(
            1 for c in comparisons.values() if c["paper_grade"]["match"]
        )
        payload["n_exact_primary_grade_match"] = sum(
            1 for c in comparisons.values() if c["primary_grade"]["match"]
        )

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(results)} graded of {n_rows} rows)")
    if report_md_path and detailed_reports:
        md = format_report_markdown(
            detailed_reports,
            {
                "sheet": str(args.sheet),
                "n_rows": n_rows,
                "n_scored": len(results),
                "n_exact_paper_grade_match": payload.get("n_exact_paper_grade_match"),
            },
        )
        report_md_path.write_text(md, encoding="utf-8")
        print(f"Wrote {report_md_path}")
    if answer_key:
        print(f"LLM comparison: {len(comparisons)} of {len(results)} graded rows")
    if comparisons:
        n_match = payload["n_exact_paper_grade_match"]
        print(f"Exact paper_grade match: {n_match}/{len(comparisons)}")
    elif answer_key and results:
        print(
            "Warning: --answer-key provided but no comparisons produced",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
