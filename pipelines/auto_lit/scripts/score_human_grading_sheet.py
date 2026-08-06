#!/usr/bin/env python3
"""Score a filled blind-grading pack and compare to the LLM answer key.

Reads scores.csv (long format: sample_id, criterion_id, human_score) from a pack
produced by sample_blind_grading_audit.py --per-criterion, aggregates with the
same rubric math as the LLM grader, and writes a disagreement report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.env_config import resolve_rubric_paths  # noqa: E402
from auto_lit_search.rubric_scoring import (  # noqa: E402
    aggregate_paper_scores,
    normalize_criterion_scores,
    rubric_role_for_paper_role,
)

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from sample_blind_grading_audit import load_rubric_spec  # noqa: E402


def _parse_score(raw: str) -> Optional[int]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        val = int(float(text))
    except ValueError:
        return None
    return max(0, min(2, val))


def load_human_scores(
    scores_path: Path,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, str]]:
    """Return {sample_id: {criterion_id: {score, note}}} and paper roles."""
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    roles: Dict[str, str] = {}
    with scores_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            crit_id = (row.get("criterion_id") or "").strip()
            if not sample_id or not crit_id:
                continue
            role = (row.get("paper_role") or "").strip()
            if role and sample_id not in roles:
                roles[sample_id] = role
            score = _parse_score(row.get("human_score") or "")
            if score is None:
                continue
            note = (row.get("human_note") or "").strip()[:60]
            by_sample[sample_id][crit_id] = {"score": score, "note": note}
    return dict(by_sample), roles


def _llm_criterion_score(entry: Any) -> Optional[int]:
    if isinstance(entry, dict):
        raw = entry.get("score")
    else:
        raw = entry
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def compare_sample(
    sample_id: str,
    human_scores: Dict[str, Dict[str, Any]],
    answer: Dict[str, Any],
    rubric: Dict[str, Any],
    paper_role: str,
) -> Dict[str, Any]:
    role = rubric_role_for_paper_role(paper_role)
    human_agg = aggregate_paper_scores(
        rubric, normalize_criterion_scores(human_scores), rubric_role=role
    )
    llm_crit = answer.get("criterion_scores") or {}
    diffs: List[Dict[str, Any]] = []
    agreed = 0
    compared = 0
    for crit_id, human_entry in sorted(human_scores.items()):
        human_val = int(human_entry["score"])
        llm_val = _llm_criterion_score(llm_crit.get(crit_id))
        if llm_val is None:
            continue
        compared += 1
        if human_val == llm_val:
            agreed += 1
        else:
            diffs.append(
                {
                    "criterion_id": crit_id,
                    "human": human_val,
                    "llm": llm_val,
                    "delta": human_val - llm_val,
                }
            )
    return {
        "sample_id": sample_id,
        "paper_role": paper_role,
        "doi": answer.get("doi"),
        "gene_focus_id": answer.get("gene_focus_id"),
        "human_paper_grade": human_agg["paper_grade"],
        "llm_paper_grade": answer.get("paper_grade"),
        "human_relevance_grade": human_agg["relevance_grade"],
        "llm_relevance_grade": answer.get("relevance_grade"),
        "n_criteria_scored": len(human_scores),
        "n_compared": compared,
        "n_agreed": agreed,
        "n_disagreed": len(diffs),
        "criterion_diffs": diffs,
        "human_axis_totals": human_agg["axis_totals"],
        "llm_axis_totals": answer.get("axis_totals"),
        "human_criterion_scores": human_scores,
    }


def write_report(results: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = ["# Human vs LLM grading comparison", ""]
    total_compared = sum(r["n_compared"] for r in results)
    total_agreed = sum(r["n_agreed"] for r in results)
    rate = (total_agreed / total_compared) if total_compared else 0.0
    lines.append(f"- Papers scored: {len(results)}")
    lines.append(f"- Criterion pairs compared: {total_compared}")
    lines.append(f"- Exact agreement: {total_agreed}/{total_compared} ({rate:.1%})")
    lines.append("")
    lines.append("## Per paper")
    lines.append("")
    lines.append(
        "| sample_id | role | human grade | LLM grade | agreed | disagreed |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        lines.append(
            f"| {r['sample_id']} | {r['paper_role']} | {r['human_paper_grade']} | "
            f"{r['llm_paper_grade']} | {r['n_agreed']}/{r['n_compared']} | {r['n_disagreed']} |"
        )
    lines.append("")
    disagreed = [r for r in results if r["criterion_diffs"]]
    if disagreed:
        lines.append("## Criterion disagreements")
        lines.append("")
        for r in disagreed:
            lines.append(
                f"### {r['sample_id']} ({r.get('gene_focus_id')}, {r.get('doi')})"
            )
            lines.append("")
            for d in r["criterion_diffs"]:
                lines.append(
                    f"- `{d['criterion_id']}`: human={d['human']} llm={d['llm']} "
                    f"(Δ={d['delta']:+d})"
                )
            lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
        help="Audit pack directory containing scores.csv and llm_grades_answer_key.json",
    )
    p.add_argument("--scores", type=Path, default=None, help="Override scores.csv path")
    p.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help="Override llm_grades_answer_key.json path",
    )
    p.add_argument("--host-rubric", type=Path, default=None)
    p.add_argument("--microbe-rubric", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write human_grades.json + comparison_report.md (default: pack-dir)",
    )
    args = p.parse_args()

    pack = args.pack_dir
    scores_path = args.scores or (pack / "scores.csv")
    key_path = args.answer_key or (pack / "llm_grades_answer_key.json")
    out_dir = args.out_dir or pack

    if not scores_path.is_file():
        print(f"Not found: {scores_path}", file=sys.stderr)
        return 2
    if not key_path.is_file():
        print(f"Not found: {key_path}", file=sys.stderr)
        return 2

    try:
        host_path, microbe_path = resolve_rubric_paths(
            host=args.host_rubric, microbe=args.microbe_rubric
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2
    if not host_path.is_file() or not microbe_path.is_file():
        print("Rubric JSON path not found", file=sys.stderr)
        return 2

    rubric_spec = load_rubric_spec(host_path, microbe_path)
    human_by_sample, roles_from_csv = load_human_scores(scores_path)
    if not human_by_sample:
        print("No filled human_score values found in scores.csv", file=sys.stderr)
        return 1

    answer_key = json.loads(key_path.read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = []
    for sample_id, human_scores in sorted(human_by_sample.items()):
        answer = answer_key.get(sample_id)
        if not isinstance(answer, dict):
            print(f"Warning: {sample_id} missing from answer key", file=sys.stderr)
            continue
        paper_role = (
            roles_from_csv.get(sample_id)
            or str(answer.get("paper_role") or "")
        )
        rubric = rubric_spec.rubric_for_role(paper_role)
        results.append(
            compare_sample(sample_id, human_scores, answer, rubric, paper_role)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    grades_path = out_dir / "human_grades.json"
    grades_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = out_dir / "comparison_report.md"
    write_report(results, report_path)
    print(f"Wrote {grades_path}")
    print(f"Wrote {report_path}")
    n_fill = sum(r["n_criteria_scored"] for r in results)
    print(f"Scored {len(results)} papers / {n_fill} criterion values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
