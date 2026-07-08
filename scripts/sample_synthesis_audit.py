#!/usr/bin/env python3
"""Sample alignment pairs for one-shot synthesis audit.

Writes a CSV with full literature-only prompts (no papers), per-sample prompt files,
and an answer key with pipeline synthesis conclusions for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.download_manifest import _load_idmap  # noqa: E402
from auto_lit_search.paper_io import gene_terms  # noqa: E402
from auto_lit_search.synthesis_audit import (  # noqa: E402
    build_one_shot_audit_prompt,
    pair_priority_stratum,
)
from auto_lit_search.env_config import resolve_rubric_paths  # noqa: E402

STRATUM_BINS: List[Tuple[str, int, int, float]] = [
    ("weak", 0, 39, 0.40),
    ("mid", 40, 59, 0.20),
    ("strong", 60, 100, 0.40),
]

AUDIT_CSV_FIELDS = [
    "sample_id",
    "alignment_id",
    "query_gene_id",
    "target_gene_id",
    "query_symbol",
    "target_symbol",
    "query_common_name",
    "target_common_name",
    "query_search_terms",
    "target_search_terms",
    "pathogen",
    "host",
    "evidence_stratum",
    "pair_priority_score",
    "pair_priority_tier",
    "host_exploitation_score",
    "query_effector_score",
    "mimicry_plausibility_score",
    "pipeline_headline",
    "prompt_path",
    "research_response",
    "research_notes",
]


@dataclass
class AlignmentRecord:
    alignment_id: str
    query_gene_id: str
    target_gene_id: str
    pair_priority_score: int
    pair_priority_tier: str
    host_exploitation_score: int
    host_exploitation_tier: str
    query_effector_score: int
    query_effector_tier: str
    mimicry_plausibility_score: int
    mimicry_plausibility_tier: str
    headline: str
    main_uncertainties: str
    synthesis_status: str
    synthesis_text: str
    conclusion: Dict[str, Any]
    gene_context: Dict[str, Any] = field(default_factory=dict)
    query_symbol: str = ""
    target_symbol: str = ""
    query_common_name: str = ""
    target_common_name: str = ""
    query_search_terms: str = ""
    target_search_terms: str = ""
    prompt: str = ""

    @property
    def stratum(self) -> str:
        return pair_priority_stratum(self.pair_priority_score)


def _format_search_terms(meta: Dict[str, Any], fallback_id: str) -> str:
    gt = gene_terms(meta, fallback_id)
    parts: List[str] = []
    for val in (
        fallback_id,
        gt["symbol"],
        gt["common_name"] if gt["common_name"] != "none" else "",
        str(meta.get("entrez_id") or "").strip(),
        str(meta.get("locus_tag") or "").strip(),
        str(meta.get("genbank_acc") or "").strip(),
    ):
        if val and val.lower() not in {p.lower() for p in parts}:
            parts.append(val)
    for syn in gt["synonyms"]:
        if syn.lower() not in {p.lower() for p in parts}:
            parts.append(syn)
    return "; ".join(parts)


def _gene_context_for_alignment(
    query_gene_id: str,
    target_gene_id: str,
    idmap: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    key = f"{query_gene_id}|{target_gene_id}"
    if key in idmap:
        row = idmap[key]
        return {
            "query": dict(row.get("query_meta") or {}),
            "target": dict(row.get("target_meta") or {}),
        }
    return {"query": {}, "target": {}}


def _attach_gene_fields(
    rec: AlignmentRecord,
    idmap: Dict[str, Dict[str, Any]],
) -> None:
    ctx = _gene_context_for_alignment(rec.query_gene_id, rec.target_gene_id, idmap)
    rec.gene_context = ctx
    q_meta = ctx.get("query") or {}
    t_meta = ctx.get("target") or {}
    q_gt = gene_terms(q_meta, rec.query_gene_id)
    t_gt = gene_terms(t_meta, rec.target_gene_id)
    rec.query_symbol = q_gt["symbol"]
    rec.target_symbol = t_gt["symbol"]
    rec.query_common_name = q_gt["common_name"]
    rec.target_common_name = t_gt["common_name"]
    rec.query_search_terms = _format_search_terms(q_meta, rec.query_gene_id)
    rec.target_search_terms = _format_search_terms(t_meta, rec.target_gene_id)


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _conclusion_from_results(
    results: Dict[str, Any],
    graded_papers: List[Any],
    synthesis_text: str,
) -> Dict[str, Any]:
    raw = results.get("conclusion")
    if isinstance(raw, dict) and raw.get("scorecard_version") == "2":
        return raw
    return build_conclusion(
        graded_papers,
        synthesis_text,
        synthesis_status=str(
            (raw or {}).get("synthesis_status")
            if isinstance(raw, dict) and raw.get("synthesis_status")
            else "grades_only"
        ),
    )


def load_alignment_pool(output_root: Path) -> List[AlignmentRecord]:
    pool: List[AlignmentRecord] = []
    for results_path in sorted(output_root.glob("*_results.json")):
        alignment_id = results_path.name[: -len("_results.json")]
        graded_path = output_root / f"{alignment_id}_graded.json"
        if not graded_path.is_file():
            continue
        try:
            results = _load_json(results_path)
            graded = _load_json(graded_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: skip {alignment_id}: {exc}", file=sys.stderr)
            continue

        query = str(results.get("query") or graded.get("query") or "").strip()
        target = str(
            results.get("target_id") or graded.get("target_id") or ""
        ).strip()
        if not query or not target:
            parts = alignment_id.split("_", 1)
            if len(parts) == 2:
                query = query or parts[0]
                target = target or parts[1]

        papers = graded_papers_from_json(graded)
        synth = results.get("synthesis") or {}
        synthesis_text = str(synth.get("text") or "") if isinstance(synth, dict) else ""
        conclusion = _conclusion_from_results(results, papers, synthesis_text)

        pool.append(
            AlignmentRecord(
                alignment_id=alignment_id,
                query_gene_id=query,
                target_gene_id=target,
                pair_priority_score=int(conclusion["pair_priority"]["score"]),
                pair_priority_tier=str(conclusion["pair_priority"]["tier"]),
                host_exploitation_score=int(conclusion["host_exploitation"]["score"]),
                host_exploitation_tier=str(conclusion["host_exploitation"]["tier"]),
                query_effector_score=int(conclusion["query_effector"]["score"]),
                query_effector_tier=str(conclusion["query_effector"]["tier"]),
                mimicry_plausibility_score=int(conclusion["mimicry_plausibility"]["score"]),
                mimicry_plausibility_tier=str(conclusion["mimicry_plausibility"]["tier"]),
                headline=str(conclusion.get("headline") or ""),
                main_uncertainties=str(conclusion.get("main_uncertainties") or ""),
                synthesis_status=str(conclusion.get("synthesis_status") or ""),
                synthesis_text=synthesis_text,
                conclusion=conclusion,
            )
        )
    return pool


def _stratum_pool(pool: Sequence[AlignmentRecord], stratum: str) -> List[AlignmentRecord]:
    return [rec for rec in pool if rec.stratum == stratum]


def _allocate_stratum_counts(n: int) -> Dict[str, int]:
    counts = {name: int(n * share) for name, _, _, share in STRATUM_BINS}
    remainder = n - sum(counts.values())
    order = sorted(STRATUM_BINS, key=lambda x: -x[3])
    i = 0
    while remainder > 0:
        counts[order[i % len(order)][0]] += 1
        remainder -= 1
        i += 1
    return counts


def _sample_without_replacement(
    rng: random.Random,
    items: Sequence[AlignmentRecord],
    k: int,
) -> List[AlignmentRecord]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    return rng.sample(list(items), k)


def sample_alignments(
    pool: Sequence[AlignmentRecord],
    n: int,
    rng: random.Random,
) -> Tuple[List[AlignmentRecord], Dict[str, Any]]:
    if not pool:
        return [], {"warning": "empty pool"}

    n = min(n, len(pool))
    counts = _allocate_stratum_counts(n)
    selected: List[AlignmentRecord] = []
    seen: set[str] = set()
    stratum_picked: Dict[str, int] = {}
    warnings: List[str] = []

    for stratum, need in counts.items():
        if need <= 0:
            continue
        candidates = [
            rec for rec in _stratum_pool(pool, stratum) if rec.alignment_id not in seen
        ]
        picked = _sample_without_replacement(rng, candidates, need)
        selected.extend(picked)
        seen.update(rec.alignment_id for rec in picked)
        stratum_picked[stratum] = len(picked)
        if len(picked) < need:
            warnings.append(f"{stratum}: only {len(picked)} available (requested {need})")

    if len(selected) < n:
        leftovers = [rec for rec in pool if rec.alignment_id not in seen]
        extra = _sample_without_replacement(rng, leftovers, n - len(selected))
        selected.extend(extra)
        if extra:
            warnings.append(f"topped up {len(extra)} from unstratified leftovers")

    stats = {
        "requested_n": n,
        "actual_n": len(selected),
        "stratum_requested": counts,
        "stratum_picked": stratum_picked,
        "pool_size": len(pool),
        "pool_strong": len(_stratum_pool(pool, "strong")),
        "pool_mid": len(_stratum_pool(pool, "mid")),
        "pool_weak": len(_stratum_pool(pool, "weak")),
        "n_strong": sum(1 for rec in selected if rec.stratum == "strong"),
        "n_mid": sum(1 for rec in selected if rec.stratum == "mid"),
        "n_weak": sum(1 for rec in selected if rec.stratum == "weak"),
    }
    if warnings:
        stats["warning"] = "; ".join(warnings)
    return selected, stats


def _rubric_system_context(
    host_rubric: Dict[str, Any],
    microbe_rubric: Dict[str, Any],
) -> Tuple[str, str, str]:
    microbe_ctx = microbe_rubric.get("system_context") or {}
    host_ctx = host_rubric.get("system_context") or {}
    pathogen = str(
        microbe_ctx.get("pathogen")
        or host_ctx.get("infection_agent")
        or "Legionella pneumophila"
    )
    host = str(host_ctx.get("host") or microbe_ctx.get("host") or "Homo sapiens")
    interaction = str(microbe_ctx.get("interaction_type") or "").strip()
    research_q = str(microbe_ctx.get("research_question") or "").strip()
    blurbs = [x for x in (interaction, research_q) if x]
    return pathogen, host, " ".join(blurbs)


def write_outputs(
    selected: Sequence[AlignmentRecord],
    out_dir: Path,
    seed: int,
    stats: Dict[str, Any],
    *,
    pathogen: str,
    host: str,
    interaction_blurb: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    ordered = list(selected)
    rng = random.Random(seed + 1)
    rng.shuffle(ordered)

    answer_key: Dict[str, Any] = {}
    rows: List[Dict[str, str]] = []

    for i, rec in enumerate(ordered, start=1):
        sample_id = f"synth_audit_{i:03d}"
        prompt = build_one_shot_audit_prompt(
            alignment_id=rec.alignment_id,
            query_id=rec.query_gene_id,
            target_id=rec.target_gene_id,
            gene_context=rec.gene_context,
            pathogen_name=pathogen,
            host_name=host,
            interaction_blurb=interaction_blurb,
        )
        rec.prompt = prompt
        prompt_rel = f"prompts/{sample_id}.txt"
        (out_dir / prompt_rel).write_text(prompt + "\n", encoding="utf-8")

        rows.append(
            {
                "sample_id": sample_id,
                "alignment_id": rec.alignment_id,
                "query_gene_id": rec.query_gene_id,
                "target_gene_id": rec.target_gene_id,
                "query_symbol": rec.query_symbol,
                "target_symbol": rec.target_symbol,
                "query_common_name": rec.query_common_name,
                "target_common_name": rec.target_common_name,
                "query_search_terms": rec.query_search_terms,
                "target_search_terms": rec.target_search_terms,
                "pathogen": pathogen,
                "host": host,
                "evidence_stratum": rec.stratum,
                "pair_priority_score": str(rec.pair_priority_score),
                "pair_priority_tier": rec.pair_priority_tier,
                "host_exploitation_score": str(rec.host_exploitation_score),
                "query_effector_score": str(rec.query_effector_score),
                "mimicry_plausibility_score": str(rec.mimicry_plausibility_score),
                "pipeline_headline": rec.headline,
                "prompt_path": prompt_rel,
                "research_response": "",
                "research_notes": "",
            }
        )
        answer_key[sample_id] = {
            "alignment_id": rec.alignment_id,
            "query_gene_id": rec.query_gene_id,
            "target_gene_id": rec.target_gene_id,
            "query_symbol": rec.query_symbol,
            "target_symbol": rec.target_symbol,
            "query_common_name": rec.query_common_name,
            "target_common_name": rec.target_common_name,
            "query_search_terms": rec.query_search_terms,
            "target_search_terms": rec.target_search_terms,
            "gene_context": rec.gene_context,
            "evidence_stratum": rec.stratum,
            "pipeline_conclusion": rec.conclusion,
            "pipeline_headline": rec.headline,
            "pipeline_main_uncertainties": rec.main_uncertainties,
            "pipeline_synthesis_status": rec.synthesis_status,
            "pipeline_synthesis_text": rec.synthesis_text,
            "prompt_path": prompt_rel,
        }

    sheet_path = out_dir / "synthesis_audit_sheet.csv"
    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    key_path = out_dir / "pipeline_synthesis_answer_key.json"
    key_path.write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    guide_lines = [
        "One-shot synthesis audit",
        "",
        "Goal: compare research-assistant LLM one-shot literature judgments against",
        "the full pipeline synthesis (paper retrieval + rubric grading + synthesis).",
        "",
        "For each row in synthesis_audit_sheet.csv:",
        "1. Send the prompt in prompts/<sample_id>.txt to a research LLM (one shot).",
        "2. Paste the full model response into research_response.",
        "3. Optional notes in research_notes.",
        "",
        "Gene synonyms and search terms are embedded in each prompt via",
        "identification_terms_block (symbol, common_name, synonyms).",
        "",
        "Compare filled responses to pipeline_synthesis_answer_key.json:",
        "- pipeline_conclusion scores and headline",
        "- pair_priority_score / host_exploitation / query_effector / mimicry_plausibility",
        "",
        "Strata (evidence_stratum): strong=pair_priority>=60, mid=40-59, weak<=39.",
        "",
        f"pathogen={pathogen}",
        f"host={host}",
    ]
    (out_dir / "synthesis_audit_guide.txt").write_text(
        "\n".join(guide_lines) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"seed={seed}",
        f"requested_n={stats.get('requested_n')}",
        f"actual_n={stats.get('actual_n')}",
        f"pool_size={stats.get('pool_size')}",
        f"pool_strong={stats.get('pool_strong')}",
        f"pool_mid={stats.get('pool_mid')}",
        f"pool_weak={stats.get('pool_weak')}",
        f"n_strong={stats.get('n_strong')}",
        f"n_mid={stats.get('n_mid')}",
        f"n_weak={stats.get('n_weak')}",
        f"stratum_requested={json.dumps(stats.get('stratum_requested', {}))}",
        f"stratum_picked={json.dumps(stats.get('stratum_picked', {}))}",
    ]
    if stats.get("warning"):
        report_lines.append(f"warning={stats['warning']}")
    (out_dir / "sampling_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {sheet_path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {prompts_dir}/ ({len(rows)} prompts)")
    print(f"Wrote {out_dir / 'synthesis_audit_guide.txt'}")
    print(f"Wrote {out_dir / 'sampling_report.txt'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="llm_results directory with *_results.json and *_graded.json",
    )
    p.add_argument("--n", type=int, required=True, help="Number of alignment pairs to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: output-root/synthesis_audit_TIMESTAMP)",
    )
    p.add_argument(
        "--idmap-csv",
        type=Path,
        default=None,
        help="Stage-1 idmap CSV for gene synonyms (recommended)",
    )
    p.add_argument("--host-rubric", type=Path, default=None)
    p.add_argument("--microbe-rubric", type=Path, default=None)
    args = p.parse_args()

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if not args.output_root.is_dir():
        print(f"Not a directory: {args.output_root}", file=sys.stderr)
        return 2

    try:
        host_rubric_path, microbe_rubric_path = resolve_rubric_paths(
            host=args.host_rubric,
            microbe=args.microbe_rubric,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    idmap: Dict[str, Dict[str, Any]] = {}
    if args.idmap_csv:
        if not args.idmap_csv.is_file():
            print(f"Not found: {args.idmap_csv}", file=sys.stderr)
            return 2
        idmap = _load_idmap(str(args.idmap_csv))
        if not idmap:
            print(f"Warning: no rows loaded from {args.idmap_csv}", file=sys.stderr)

    host_rubric = json.loads(host_rubric_path.read_text(encoding="utf-8"))
    microbe_rubric = json.loads(microbe_rubric_path.read_text(encoding="utf-8"))
    pathogen, host, interaction_blurb = _rubric_system_context(host_rubric, microbe_rubric)

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"synthesis_audit_{ts}"

    pool = load_alignment_pool(args.output_root)
    if not pool:
        print("No complete alignments found.", file=sys.stderr)
        return 1

    if idmap:
        for rec in pool:
            _attach_gene_fields(rec, idmap)

    rng = random.Random(args.seed)
    selected, stats = sample_alignments(pool, args.n, rng)
    if not selected:
        print("No samples selected.", file=sys.stderr)
        return 1

    write_outputs(
        selected,
        out_dir,
        args.seed,
        stats,
        pathogen=pathogen,
        host=host,
        interaction_blurb=interaction_blurb,
    )
    if stats.get("warning"):
        print(f"Warning: {stats['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
