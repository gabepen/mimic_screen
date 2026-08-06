#!/usr/bin/env python3
"""Build an LLM-oriented Legionella grader audit pack.

Takes an existing human blind-grading pack (index.csv + scores.csv +
worksheets + rubric_guides + answer key) and writes 5 prompt sets of 4
papers (20 total). Each set gets the same shared instructions/rubrics and
a ready-to-paste PROMPT.md that asks the LLM to return tabular scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SHARED_INSTRUCTIONS = """# LLM grader task

You are filling out a Legionella literature rubric for a small set of papers.
Each paper is graded for ONE gene focus only (given below). Use only the paper
text provided in this prompt. Do not use outside knowledge of the paper, and
do not invent experiments that are not described.

## Scoring rules

- Score every listed criterion for that paper with an integer **0**, **1**, or **2**.
- Scale:
  - **0** = absent — no evidence for this criterion for this gene in this paper
  - **1** = partial — indirect, inferred, or with significant caveats
  - **2** = direct — explicitly and unambiguously demonstrated
- Use the criterion prompt and score_0 / score_1 / score_2 definitions in the
  rubric section. Host (target) papers use the host rubric; query (microbe)
  papers use the Legionella microbe rubric.
- Judge only this paper × this gene. Most host papers will not mention
  Legionella; host relevance is inferred from functional overlap with Legionella
  biology, not from infection wording.
- If the gene is not meaningfully discussed, score criteria as 0 and note that
  in `evidence_note`.

## Required output

Return **only** a markdown table (no prose before or after except an optional
one-line note if a paper text is unusable). One row per criterion per paper.

Columns (exact header):

```
set_id | sample_id | doi | paper_role | gene_focus_id | criterion_id | score | evidence_note
```

Rules for the table:

- `set_id`: the set id given in this prompt (e.g. `set_01`)
- `sample_id`: exact id from the paper blocks (e.g. `audit_001`)
- `doi`: exact DOI string from the paper block
- `paper_role`: `query` or `target`
- `gene_focus_id`: UniProt id from the paper block
- `criterion_id`: exact criterion id from that paper's "Criteria to score" list
- `score`: `0`, `1`, or `2` only
- `evidence_note`: ≤60 characters citing the paper (quote fragment / section);
  use `-` if none

Include every criterion listed for every paper in this set. Do not add extra
criteria. Do not omit rows. Do not compute axis totals or overall grades.
"""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _role_bucket(role: str) -> str:
    r = (role or "").strip().lower()
    return "query" if r in ("query", "microbe") else "target"


def pick_papers(
    index_rows: Sequence[Dict[str, str]],
    n: int,
    n_sets: int,
    seed: int,
) -> List[List[Dict[str, str]]]:
    if n % n_sets != 0:
        raise ValueError(f"n={n} must be divisible by n_sets={n_sets}")
    per_set = n // n_sets
    by_role: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        by_role[_role_bucket(row.get("paper_role", ""))].append(row)

    rng = random.Random(seed)
    for role in by_role:
        rng.shuffle(by_role[role])

    n_query = n // 2
    n_target = n - n_query
    if len(by_role["query"]) < n_query or len(by_role["target"]) < n_target:
        raise ValueError(
            f"Need {n_query} query and {n_target} target; have "
            f"{len(by_role['query'])} / {len(by_role['target'])}"
        )

    selected = by_role["query"][:n_query] + by_role["target"][:n_target]
    rng.shuffle(selected)

    # Prefer 2 query + 2 target per set of 4 when possible.
    q = [r for r in selected if _role_bucket(r["paper_role"]) == "query"]
    t = [r for r in selected if _role_bucket(r["paper_role"]) == "target"]
    sets: List[List[Dict[str, str]]] = []
    qi = ti = 0
    q_per = per_set // 2
    t_per = per_set - q_per
    for _ in range(n_sets):
        chunk = q[qi : qi + q_per] + t[ti : ti + t_per]
        qi += q_per
        ti += t_per
        rng.shuffle(chunk)
        sets.append(chunk)
    leftover = q[qi:] + t[ti:]
    if leftover:
        raise RuntimeError("internal split error: leftover papers")
    return sets


def _load_paper_text(text_path: str, max_chars: int) -> Tuple[str, bool]:
    path = Path(text_path)
    if not path.is_file():
        return f"[MISSING PAPER TEXT: {text_path}]", True
    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED — remaining text omitted for prompt size]\n"
        truncated = True
    return text, truncated


def _criteria_block(sample_id: str, score_rows: Sequence[Dict[str, str]]) -> str:
    lines = ["Criteria to score (ids must appear in your table):", ""]
    for r in score_rows:
        if r["sample_id"] != sample_id:
            continue
        lines.append(
            f"- `{r['criterion_id']}` — {r.get('label') or r['criterion_id']} "
            f"(axis=`{r.get('axis_id') or ''}`, weight=`{r.get('weight') or ''}`)"
        )
    lines.append("")
    return "\n".join(lines)


def build_set_prompt(
    set_id: str,
    papers: Sequence[Dict[str, str]],
    score_rows: Sequence[Dict[str, str]],
    host_guide: str,
    microbe_guide: str,
    worksheets_dir: Path,
    max_chars: int,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    parts: List[str] = [
        f"# Legionella LLM rubric grading — {set_id}",
        "",
        f"This prompt is identical in structure to the other sets; only the papers differ.",
        f"**set_id for your table rows: `{set_id}`**",
        "",
        SHARED_INSTRUCTIONS.strip(),
        "",
        "---",
        "",
        "# Host rubric (use for `paper_role=target`)",
        "",
        host_guide.strip(),
        "",
        "---",
        "",
        "# Microbe / Legionella rubric (use for `paper_role=query`)",
        "",
        microbe_guide.strip(),
        "",
        "---",
        "",
        f"# Papers in {set_id} ({len(papers)} papers)",
        "",
    ]

    template_rows: List[Dict[str, str]] = []
    warnings: List[str] = []

    for paper in papers:
        sid = paper["sample_id"]
        worksheet = worksheets_dir / f"{sid}.md"
        worksheet_body = ""
        if worksheet.is_file():
            # Drop the human-oriented "How to score" / scores.csv pointers; keep criteria defs.
            raw = worksheet.read_text(encoding="utf-8")
            worksheet_body = raw
        text, truncated = _load_paper_text(paper.get("text_path") or "", max_chars)
        if truncated:
            warnings.append(f"{set_id}/{sid}: truncated paper text to {max_chars} chars")
        if text.startswith("[MISSING"):
            warnings.append(f"{set_id}/{sid}: missing text at {paper.get('text_path')}")

        parts.extend(
            [
                f"## Paper `{sid}`",
                "",
                f"- DOI: `{paper.get('doi') or ''}`",
                f"- paper_role: `{paper.get('paper_role') or ''}`",
                f"- gene_focus_id: `{paper.get('gene_focus_id') or ''}`",
                f"- gene_focus_symbol: `{paper.get('gene_focus_symbol') or ''}`",
                f"- gene_focus_common_name: `{paper.get('gene_focus_common_name') or ''}`",
                f"- alignment_id: `{paper.get('alignment_id') or ''}`",
                f"- search_terms: {paper.get('gene_focus_search_terms') or paper.get('gene_focus_id') or ''}",
                "",
                _criteria_block(sid, score_rows).rstrip(),
                "",
                "### Criterion definitions for this paper",
                "",
                worksheet_body.strip() if worksheet_body else "_No worksheet found._",
                "",
                "### Paper text",
                "",
                "```",
                text.rstrip(),
                "```",
                "",
                "---",
                "",
            ]
        )

        for r in score_rows:
            if r["sample_id"] != sid:
                continue
            template_rows.append(
                {
                    "set_id": set_id,
                    "sample_id": sid,
                    "doi": paper.get("doi") or "",
                    "paper_role": paper.get("paper_role") or "",
                    "gene_focus_id": paper.get("gene_focus_id") or "",
                    "criterion_id": r["criterion_id"],
                    "axis_id": r.get("axis_id") or "",
                    "weight": r.get("weight") or "",
                    "label": r.get("label") or "",
                    "score": "",
                    "evidence_note": "",
                }
            )

    parts.extend(
        [
            "# Output reminder",
            "",
            "Respond with the markdown table only:",
            "",
            "```",
            "set_id | sample_id | doi | paper_role | gene_focus_id | criterion_id | score | evidence_note",
            f"{set_id} | <sample_id> | <doi> | <query|target> | <gene> | <criterion_id> | <0|1|2> | <note or ->",
            "```",
            "",
            f"Fill every criterion for every paper in `{set_id}` "
            f"({len(template_rows)} rows total).",
            "",
        ]
    )
    return "\n".join(parts), template_rows, warnings


def build_pack(
    source_pack: Path,
    out_dir: Path,
    n: int,
    n_sets: int,
    seed: int,
    max_chars: int,
) -> None:
    index_rows = _read_csv(source_pack / "index.csv")
    score_rows = _read_csv(source_pack / "scores.csv")
    answer_key_path = source_pack / "llm_grades_answer_key.json"
    answer_key: Dict[str, Any] = {}
    if answer_key_path.is_file():
        answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))

    guides_dir = source_pack / "rubric_guides"
    host_guide = (guides_dir / "host_rubric_v1_guide.md").read_text(encoding="utf-8")
    microbe_guide = (guides_dir / "legionella_rubric_guide.md").read_text(encoding="utf-8")

    sets = pick_papers(index_rows, n=n, n_sets=n_sets, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = out_dir / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "INSTRUCTIONS.md").write_text(SHARED_INSTRUCTIONS.strip() + "\n", encoding="utf-8")
    shutil.copy2(guides_dir / "host_rubric_v1_guide.md", shared / "host_rubric_v1_guide.md")
    shutil.copy2(guides_dir / "legionella_rubric_guide.md", shared / "legionella_rubric_guide.md")

    # Optional JSON rubrics from answer key paths or data root sibling.
    for sample in answer_key.values():
        for key, name in (
            ("host_rubric_path", "host_rubric_v1.json"),
            ("microbe_rubric_path", "legionella_rubric.json"),
        ):
            src = Path(str(sample.get(key) or ""))
            if src.is_file():
                shutil.copy2(src, shared / name)
        break

    master_rows: List[Dict[str, str]] = []
    pack_answer: Dict[str, Any] = {}
    all_warnings: List[str] = []
    scores_by_sample = score_rows

    for i, papers in enumerate(sets, start=1):
        set_id = f"set_{i:02d}"
        set_dir = out_dir / set_id
        papers_dir = set_dir / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        prompt, template_rows, warnings = build_set_prompt(
            set_id=set_id,
            papers=papers,
            score_rows=scores_by_sample,
            host_guide=host_guide,
            microbe_guide=microbe_guide,
            worksheets_dir=source_pack / "worksheets",
            max_chars=max_chars,
        )
        all_warnings.extend(warnings)
        (set_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")
        _write_csv(
            set_dir / "scores_template.csv",
            template_rows,
            [
                "set_id",
                "sample_id",
                "doi",
                "paper_role",
                "gene_focus_id",
                "criterion_id",
                "axis_id",
                "weight",
                "label",
                "score",
                "evidence_note",
            ],
        )

        manifest_rows: List[Dict[str, str]] = []
        for paper in papers:
            sid = paper["sample_id"]
            src_text = Path(paper.get("text_path") or "")
            dest = papers_dir / f"{sid}.txt"
            if src_text.is_file():
                shutil.copy2(src_text, dest)
            manifest = {
                "set_id": set_id,
                "sample_id": sid,
                "doi": paper.get("doi") or "",
                "paper_role": paper.get("paper_role") or "",
                "gene_focus_id": paper.get("gene_focus_id") or "",
                "gene_focus_symbol": paper.get("gene_focus_symbol") or "",
                "gene_focus_common_name": paper.get("gene_focus_common_name") or "",
                "alignment_id": paper.get("alignment_id") or "",
                "query_gene_id": paper.get("query_gene_id") or "",
                "target_gene_id": paper.get("target_gene_id") or "",
                "gene_focus_search_terms": paper.get("gene_focus_search_terms") or "",
                "source_text_path": paper.get("text_path") or "",
                "pack_text_path": str(dest),
                "n_criteria": str(sum(1 for r in scores_by_sample if r["sample_id"] == sid)),
            }
            manifest_rows.append(manifest)
            master_rows.append(manifest)
            if sid in answer_key:
                pack_answer[sid] = dict(answer_key[sid])
                pack_answer[sid]["set_id"] = set_id

        _write_csv(
            set_dir / "papers_manifest.csv",
            manifest_rows,
            list(manifest_rows[0].keys()) if manifest_rows else ["set_id", "sample_id", "doi"],
        )

        # Compact prompt without embedded full texts (for tool-using agents).
        compact = [
            f"# Legionella LLM rubric grading — {set_id} (compact)",
            "",
            "Same task as `PROMPT.md`, but paper bodies live under `papers/`.",
            "Read each `papers/<sample_id>.txt` before scoring.",
            "",
            SHARED_INSTRUCTIONS.strip(),
            "",
            "Rubrics: see `../shared/host_rubric_v1_guide.md` and "
            "`../shared/legionella_rubric_guide.md`.",
            "",
            f"**set_id:** `{set_id}`",
            "",
            "## Papers",
            "",
        ]
        for m in manifest_rows:
            compact.append(
                f"- `{m['sample_id']}` DOI=`{m['doi']}` role=`{m['paper_role']}` "
                f"gene=`{m['gene_focus_id']}` ({m['gene_focus_symbol']}) "
                f"text=`papers/{m['sample_id']}.txt` "
                f"criteria={m['n_criteria']}"
            )
            compact.append(f"  search_terms: {m['gene_focus_search_terms']}")
        compact.extend(
            [
                "",
                "Use `scores_template.csv` as the row checklist. Return the markdown "
                "table specified in the instructions.",
                "",
            ]
        )
        (set_dir / "PROMPT_compact.md").write_text("\n".join(compact) + "\n", encoding="utf-8")

    _write_csv(
        out_dir / "master_index.csv",
        master_rows,
        list(master_rows[0].keys()) if master_rows else ["set_id", "sample_id", "doi"],
    )
    (out_dir / "answer_key.json").write_text(
        json.dumps(pack_answer, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    role_counts = defaultdict(int)
    for r in master_rows:
        role_counts[_role_bucket(r["paper_role"])] += 1
    report = [
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"source_pack={source_pack}",
        f"seed={seed}",
        f"n_papers={n}",
        f"n_sets={n_sets}",
        f"papers_per_set={n // n_sets}",
        f"n_query={role_counts['query']}",
        f"n_target={role_counts['target']}",
        f"max_paper_chars={max_chars}",
        "warnings:",
    ]
    report.extend(f"  - {w}" for w in all_warnings) if all_warnings else report.append("  - none")
    (out_dir / "sampling_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    readme = f"""# Legionella LLM grader audit pack

20 papers in **5 sets of 4**, each with the **same instructions and rubrics**.
Only the papers differ across sets.

## How to run an LLM grader

1. Open `set_XX/PROMPT.md` (self-contained: instructions + rubrics + paper texts).
2. Paste into an LLM.
3. Collect the returned markdown table (or fill `set_XX/scores_template.csv`).

If the full prompt is too large for the model context, use `PROMPT_compact.md`
plus the files under `set_XX/papers/` and `shared/*_guide.md`.

## Layout

| Path | Purpose |
| --- | --- |
| `shared/INSTRUCTIONS.md` | Shared task + tabular output schema |
| `shared/*_guide.md` / `*.json` | Host + Legionella rubrics |
| `set_01` … `set_05` | Four papers each |
| `set_XX/PROMPT.md` | Ready-to-paste full prompt |
| `set_XX/PROMPT_compact.md` | Same task without embedded full texts |
| `set_XX/papers_manifest.csv` | DOIs / gene focus / paths |
| `set_XX/scores_template.csv` | Empty score rows to fill |
| `master_index.csv` | All 20 papers with set assignment |
| `answer_key.json` | Pipeline LLM grades (do not show the grader) |

## Expected table

```
set_id | sample_id | doi | paper_role | gene_focus_id | criterion_id | score | evidence_note
set_01 | audit_001 | 10.1038/... | query | Q5ZY37 | experimental_directness | 2 | knockout of pdxH
```

Source pack: `{source_pack}`
Seed: `{seed}`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {out_dir}")
    print(f"Sets: {n_sets} × {n // n_sets} papers")
    for w in all_warnings:
        print(f"Warning: {w}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-pack",
        type=Path,
        required=True,
        help="Existing per-criterion blind pack (index.csv, scores.csv, worksheets, guides)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=20, help="Total papers (default 20)")
    p.add_argument("--n-sets", type=int, default=5, help="Number of prompt sets (default 5)")
    p.add_argument("--seed", type=int, default=30)
    p.add_argument(
        "--max-paper-chars",
        type=int,
        default=120_000,
        help="Truncate each embedded paper in PROMPT.md (0 = no truncate)",
    )
    args = p.parse_args()
    if not (args.source_pack / "index.csv").is_file():
        print(f"Missing index.csv in {args.source_pack}", flush=True)
        return 2
    build_pack(
        source_pack=args.source_pack,
        out_dir=args.out_dir,
        n=args.n,
        n_sets=args.n_sets,
        seed=args.seed,
        max_chars=args.max_paper_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
