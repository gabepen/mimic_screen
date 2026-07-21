#!/usr/bin/env python3
"""Render rubric JSON files as human-readable Markdown for manual validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))
from auto_lit_search.env_config import auto_lit_data_root, rubrics_dir  # noqa: E402


def _md_escape(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").strip()


def _render_criterion(crit: Dict[str, Any], indent: str = "") -> List[str]:
    lines = [
        f"{indent}- **{crit.get('label') or crit.get('id')}** (`{crit.get('id')}`)",
        f"{indent}  - weight: {crit.get('weight', 'medium')}",
        f"{indent}  - prompt: {_md_escape(str(crit.get('prompt') or ''))}",
    ]
    for score_key, label in (("score_0", "0"), ("score_1", "1"), ("score_2", "2")):
        val = crit.get(score_key)
        if val:
            lines.append(f"{indent}  - **{label}**: {_md_escape(str(val))}")
    return lines


def render_rubric_markdown(rubric: Dict[str, Any], *, source_path: Path) -> str:
    lines: List[str] = []
    title = rubric.get("rubric_type") or source_path.stem
    lines.append(f"# {title.replace('_', ' ').title()}")
    lines.append("")
    lines.append(f"- Source: `{source_path}`")
    lines.append(f"- Rubric version: {rubric.get('rubric_version', '?')}")
    if rubric.get("companion_rubric"):
        lines.append(f"- Companion rubric: {rubric.get('companion_rubric')}")
    lines.append("")

    ctx = rubric.get("system_context") or {}
    if ctx:
        lines.append("## System context")
        lines.append("")
        for key, val in ctx.items():
            if isinstance(val, list):
                lines.append(f"- **{key}**: {', '.join(str(v) for v in val)}")
            else:
                lines.append(f"- **{key}**: {_md_escape(str(val))}")
        lines.append("")

    scale = rubric.get("scoring_scale") or {}
    if scale:
        lines.append("## Scoring scale (per criterion)")
        lines.append("")
        for score, desc in sorted(scale.items(), key=lambda kv: kv[0]):
            lines.append(f"- **{score}**: {_md_escape(str(desc))}")
        lines.append("")

    instr = rubric.get("grader_instructions") or []
    if instr:
        lines.append("## Grader instructions")
        lines.append("")
        for i, item in enumerate(instr, start=1):
            lines.append(f"{i}. {_md_escape(str(item))}")
        lines.append("")

    cfg = rubric.get("grading_config") or {}
    if cfg:
        lines.append("## Grading config")
        lines.append("")
        for key, val in cfg.items():
            lines.append(f"- **{key}**: {val}")
        lines.append("")

    lines.append("## Axes and criteria")
    lines.append("")
    for axis in rubric.get("axes") or []:
        if not isinstance(axis, dict):
            continue
        lines.append(f"### {axis.get('label') or axis.get('id')}")
        lines.append("")
        lines.append(f"- Axis id: `{axis.get('id')}`")
        if axis.get("description"):
            lines.append(f"- Description: {_md_escape(str(axis.get('description')))}")
        if axis.get("synthesis_instruction"):
            lines.append(
                f"- Synthesis note: {_md_escape(str(axis.get('synthesis_instruction')))}"
            )
        lines.append("")
        for crit in axis.get("criteria") or []:
            if isinstance(crit, dict):
                lines.extend(_render_criterion(crit))
                lines.append("")

    synth = rubric.get("synthesis_instructions") or {}
    if synth:
        lines.append("## Synthesis instructions")
        lines.append("")
        if synth.get("description"):
            lines.append(_md_escape(str(synth.get("description"))))
            lines.append("")
        axis_interp = synth.get("axis_interpretation") or {}
        for key, val in axis_interp.items():
            lines.append(f"- **{key}**: {_md_escape(str(val))}")
        for key in (
            "mimicry_flag_handling",
            "infection_naive_handling",
            "data_sparse_microbe_handling",
            "axis_independence",
        ):
            if synth.get(key):
                lines.append(f"- **{key}**: {_md_escape(str(synth[key]))}")
        lines.append("")

    lines.append("## Human validation scoring")
    lines.append("")
    lines.append(
        "For each paper, score every applicable axis from **0.0 to 1.0** "
        "(matching the LLM pipeline's normalized axis totals). "
        "Use the gene search terms in the audit manifest to locate mentions in the paper text."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_paths(paths: Sequence[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rubric = json.loads(path.read_text(encoding="utf-8"))
        md = render_rubric_markdown(rubric, source_path=path)
        out_path = out_dir / f"{path.stem}_validation_guide.md"
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "rubrics",
        nargs="*",
        type=Path,
        help="Rubric JSON file(s). If omitted, renders all standard rubrics.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <AUTO_LIT_DATA_ROOT>/validation_manifests/rubrics).",
    )
    args = p.parse_args()

    data_root = auto_lit_data_root()
    out_dir = args.out_dir or (data_root / "validation_manifests" / "rubrics")
    if args.rubrics:
        paths = list(args.rubrics)
    else:
        rubric_dir = rubrics_dir()
        paths = sorted(rubric_dir.glob("*.json"))
    if not paths:
        print("No rubric files found.", file=sys.stderr)
        return 1
    for path in paths:
        if not path.is_file():
            print(f"Not found: {path}", file=sys.stderr)
            return 1

    render_paths(paths, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
