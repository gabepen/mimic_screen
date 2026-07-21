#!/usr/bin/env python3
"""
Sample a blind synthesis audit pack: stratified alignments + one-shot prompts.

Generates ~N alignments (default 20) mixed across strong / mid / weak pipeline
pair_priority evidence. Each sample gets a short research-LLM prompt with
organism labels and UniProt / gene names from the stage1 idmap (no papers).
Pipeline scores stay in an answer key CSV.

Example:
  python pipelines/auto_lit/scripts/sample_synthesis_audit.py \\
    --output-root /path/to/llm_results/wol-dros-v1 \\
    --idmap /path/to/search_results/wol-dros-v1_idmap.csv \\
    --instructions pipelines/auto_lit/prompts/wol_dros_instructions.txt \\
    --query-organism "Wolbachia pipientis wMel" \\
    --target-organism "Drosophila melanogaster" \\
    --n 20 --seed 28 \\
    --out-dir /path/to/validation_manifests/wmel_synth_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from auto_lit_search.download_manifest import _load_idmap  # noqa: E402
from auto_lit_search.env_config import auto_lit_data_root  # noqa: E402
from auto_lit_search.paper_io import gene_terms  # noqa: E402
from auto_lit_search.synthesis_audit import (  # noqa: E402
    EVIDENCE_STRATA,
    build_oneshot_synthesis_audit_prompt,
    pair_priority_stratum,
)


@dataclass
class AlignmentRecord:
    alignment_id: str
    query: str
    target: str
    pair_priority_score: int
    pair_priority_tier: str
    host_exploitation_score: int
    query_effector_score: int
    mimicry_plausibility_score: int
    headline: str
    synthesis_status: str
    stratum: str
    gene_context: Dict[str, Any]
    results_path: str

    @property
    def query_terms(self) -> Dict[str, Any]:
        meta = (self.gene_context or {}).get("query") or {}
        return gene_terms(meta, self.query)

    @property
    def target_terms(self) -> Dict[str, Any]:
        meta = (self.gene_context or {}).get("target") or {}
        return gene_terms(meta, self.target)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _pair_fields(results: dict[str, Any], alignment_id: str) -> tuple[str, str]:
    query = str(results.get("query") or "").strip()
    target = str(results.get("target_id") or "").strip()
    if not query or not target:
        parts = alignment_id.split("_", 1)
        if len(parts) == 2:
            query = query or parts[0]
            target = target or parts[1]
    return query, target


def _gene_context_has_names(gene_context: Dict[str, Any]) -> bool:
    for side in ("query", "target"):
        meta = gene_context.get(side) or {}
        if not isinstance(meta, dict):
            continue
        if str(meta.get("gene_name") or "").strip() or str(
            meta.get("common_name") or ""
        ).strip():
            return True
    return False


def gene_context_from_idmap(
    idmap: Dict[str, Dict[str, Any]], query: str, target: str
) -> Dict[str, Any]:
    meta = idmap.get(f"{query}|{target}") or {}
    return {
        "query": dict(meta.get("query_meta") or {}),
        "target": dict(meta.get("target_meta") or {}),
    }


def resolve_gene_context(
    results_gene_context: Any,
    *,
    query: str,
    target: str,
    idmap: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(results_gene_context, dict) and _gene_context_has_names(
        results_gene_context
    ):
        return {
            "query": dict(results_gene_context.get("query") or {}),
            "target": dict(results_gene_context.get("target") or {}),
        }
    return gene_context_from_idmap(idmap, query, target)


def guess_idmap_path(output_root: Path) -> Optional[Path]:
    search = auto_lit_data_root() / "search_results"
    name = output_root.name
    candidates = [search / f"{name}_idmap.csv"]
    if name == "llm_results":
        candidates.append(search / "lp-human-all_idmap.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_candidate_pool(
    output_root: Path, *, idmap: Dict[str, Dict[str, Any]]
) -> List[AlignmentRecord]:
    pool: List[AlignmentRecord] = []
    for results_path in sorted(output_root.glob("*_results.json")):
        alignment_id = results_path.name[: -len("_results.json")]
        try:
            results = _load_json(results_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        conclusion = results.get("conclusion") or {}
        if not isinstance(conclusion, dict):
            continue
        pp = conclusion.get("pair_priority") or {}
        he = conclusion.get("host_exploitation") or {}
        qe = conclusion.get("query_effector") or {}
        mp = conclusion.get("mimicry_plausibility") or {}
        try:
            pair_score = int(pp.get("score"))
        except (TypeError, ValueError):
            continue
        query, target = _pair_fields(results, alignment_id)
        if not query or not target:
            continue
        gene_context = resolve_gene_context(
            results.get("gene_context"),
            query=query,
            target=target,
            idmap=idmap,
        )
        pool.append(
            AlignmentRecord(
                alignment_id=alignment_id,
                query=query,
                target=target,
                pair_priority_score=pair_score,
                pair_priority_tier=str(pp.get("tier") or ""),
                host_exploitation_score=int(he.get("score") or 0),
                query_effector_score=int(qe.get("score") or 0),
                mimicry_plausibility_score=int(mp.get("score") or 0),
                headline=str(conclusion.get("headline") or ""),
                synthesis_status=str(conclusion.get("synthesis_status") or ""),
                stratum=pair_priority_stratum(pair_score),
                gene_context=gene_context,
                results_path=str(results_path),
            )
        )
    return pool


def _allocate_stratum_counts(n: int) -> Dict[str, int]:
    base = n // 3
    counts = {s: base for s in EVIDENCE_STRATA}
    rem = n - base * 3
    for s in ("strong", "weak", "mid"):
        if rem <= 0:
            break
        counts[s] += 1
        rem -= 1
    return counts


def _sample_without_replacement(
    items: Sequence[AlignmentRecord], k: int, rng: random.Random
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
) -> tuple[List[AlignmentRecord], Dict[str, Any]]:
    by_stratum: Dict[str, List[AlignmentRecord]] = {s: [] for s in EVIDENCE_STRATA}
    for rec in pool:
        by_stratum.setdefault(rec.stratum, []).append(rec)

    want = _allocate_stratum_counts(n)
    chosen: List[AlignmentRecord] = []
    for stratum, k in want.items():
        chosen.extend(
            _sample_without_replacement(by_stratum.get(stratum) or [], k, rng)
        )

    shortfall = n - len(chosen)
    if shortfall > 0:
        chosen_ids = {c.alignment_id for c in chosen}
        leftover = [r for r in pool if r.alignment_id not in chosen_ids]
        chosen.extend(_sample_without_replacement(leftover, shortfall, rng))

    rng.shuffle(chosen)
    meta = {
        "requested_n": n,
        "sampled_n": len(chosen),
        "requested_by_stratum": want,
        "sampled_by_stratum": {
            s: sum(1 for c in chosen if c.stratum == s) for s in EVIDENCE_STRATA
        },
        "pool_by_stratum": {s: len(by_stratum.get(s) or []) for s in EVIDENCE_STRATA},
        "pool_n": len(pool),
    }
    return chosen, meta


def _terms_cells(terms: Dict[str, Any]) -> Dict[str, str]:
    syns = terms.get("synonyms") or []
    return {
        "symbol": str(terms.get("symbol") or ""),
        "common_name": str(terms.get("common_name") or ""),
        "synonyms": "; ".join(str(s) for s in syns),
    }


def write_audit_pack(
    samples: Sequence[AlignmentRecord],
    out_dir: Path,
    *,
    instructions: str,
    query_organism: str,
    target_organism: str,
    meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    blind_rows: List[Dict[str, str]] = []
    key_rows: List[Dict[str, Any]] = []

    for i, rec in enumerate(samples, start=1):
        sample_id = f"S{i:02d}"
        prompt = build_oneshot_synthesis_audit_prompt(
            query=rec.query,
            target_id=rec.target,
            query_organism=query_organism,
            target_organism=target_organism,
            gene_context=rec.gene_context,
        )
        prompt_rel = f"prompts/{sample_id}.txt"
        (prompts_dir / f"{sample_id}.txt").write_text(prompt, encoding="utf-8")

        q = _terms_cells(rec.query_terms)
        t = _terms_cells(rec.target_terms)
        blind_rows.append(
            {
                "sample_id": sample_id,
                "alignment_id": rec.alignment_id,
                "query": rec.query,
                "target": rec.target,
                "query_organism": query_organism,
                "target_organism": target_organism,
                "query_symbol": q["symbol"],
                "query_common_name": q["common_name"],
                "query_synonyms": q["synonyms"],
                "target_symbol": t["symbol"],
                "target_common_name": t["common_name"],
                "target_synonyms": t["synonyms"],
                "prompt_path": prompt_rel,
                "external_support": "",
                "external_rationale": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "sample_id": sample_id,
                "alignment_id": rec.alignment_id,
                "query": rec.query,
                "target": rec.target,
                "stratum": rec.stratum,
                "pair_priority_score": rec.pair_priority_score,
                "pair_priority_tier": rec.pair_priority_tier,
                "host_exploitation_score": rec.host_exploitation_score,
                "query_effector_score": rec.query_effector_score,
                "mimicry_plausibility_score": rec.mimicry_plausibility_score,
                "headline": rec.headline,
                "synthesis_status": rec.synthesis_status,
                "results_path": rec.results_path,
                "prompt_path": prompt_rel,
            }
        )

    _write_csv(out_dir / "blind_synthesis_sheet.csv", blind_rows)
    _write_csv(out_dir / "answer_key.csv", key_rows)
    (out_dir / "sampling_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "README.txt").write_text(
        "Blind synthesis audit\n"
        "=====================\n\n"
        "blind_synthesis_sheet.csv — sample table + empty external_support/rationale\n"
        "answer_key.csv — pipeline scorecard scores for later comparison\n"
        "prompts/SXX.txt — brief research-LLM question (organism + UniProt / gene names)\n"
        "sampling_meta.json — stratum allocation and pool sizes\n\n"
        "Pass each prompts/SXX.txt to an external research LLM. Fill "
        "external_support (yes|partial|no|unknown) and external_rationale.\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_instructions(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty instructions file: {path}")
    return text


def _build_parser() -> argparse.ArgumentParser:
    default_root = auto_lit_data_root() / "llm_results"
    p = argparse.ArgumentParser(
        description="Sample stratified blind synthesis audit pack (no-paper prompts)."
    )
    p.add_argument("--output-root", type=Path, default=default_root)
    p.add_argument("--n", type=int, default=20, help="Number of alignments (default 20)")
    p.add_argument("--seed", type=int, default=28)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for sheet, answer key, and prompts",
    )
    p.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="System-specific synthesis instructions (kept in sampling meta)",
    )
    p.add_argument(
        "--query-organism",
        required=True,
        help='Microbe/query organism label, e.g. "Wolbachia pipientis wMel"',
    )
    p.add_argument(
        "--target-organism",
        required=True,
        help='Host/target organism label, e.g. "Drosophila melanogaster"',
    )
    p.add_argument(
        "--idmap",
        type=Path,
        default=None,
        help=(
            "Stage1 idmap CSV with query_gene_name/target_gene_name "
            "(default: guess from output-root under search_results/)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if not args.output_root.is_dir():
        print(f"Output root not found: {args.output_root}", file=sys.stderr)
        return 2
    if not args.instructions.is_file():
        print(f"Instructions file not found: {args.instructions}", file=sys.stderr)
        return 2

    query_organism = str(args.query_organism).strip()
    target_organism = str(args.target_organism).strip()
    if not query_organism or not target_organism:
        print("--query-organism and --target-organism must be non-empty", file=sys.stderr)
        return 2

    idmap_path = args.idmap or guess_idmap_path(args.output_root)
    if idmap_path is None or not idmap_path.is_file():
        print(
            "Idmap CSV not found. Pass --idmap "
            "(e.g. $AUTO_LIT_DATA_ROOT/search_results/lp-human-all_idmap.csv)",
            file=sys.stderr,
        )
        return 2
    idmap = _load_idmap(str(idmap_path))
    if not idmap:
        print(f"Idmap empty or unreadable: {idmap_path}", file=sys.stderr)
        return 2

    pool = load_candidate_pool(args.output_root, idmap=idmap)
    if not pool:
        print(f"No scored *_results.json under {args.output_root}", file=sys.stderr)
        return 1

    named = sum(1 for r in pool if _gene_context_has_names(r.gene_context))
    if named == 0:
        print(
            f"Idmap loaded ({idmap_path}) but no alignments got gene names; "
            "check query|target keys match results",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    samples, meta = sample_alignments(pool, args.n, rng)
    meta["seed"] = args.seed
    meta["output_root"] = str(args.output_root)
    meta["instructions"] = str(args.instructions)
    meta["idmap"] = str(idmap_path)
    meta["query_organism"] = query_organism
    meta["target_organism"] = target_organism
    meta["pool_with_gene_names"] = named

    instructions = _load_instructions(args.instructions)
    write_audit_pack(
        samples,
        args.out_dir,
        instructions=instructions,
        query_organism=query_organism,
        target_organism=target_organism,
        meta=meta,
    )
    print(
        f"Wrote {len(samples)} samples to {args.out_dir} "
        f"(pool={meta['pool_n']}; named={named}; by_stratum={meta['sampled_by_stratum']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
