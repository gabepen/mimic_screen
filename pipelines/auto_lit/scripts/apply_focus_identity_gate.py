#!/usr/bin/env python3
"""Offline focus-identity triage on existing *_graded.json (no grader LLM).

Re-applies ``enforce_focus_identity`` using paper text + claim_summary + idmap
aliases. Papers that fail the gate are zeroed and marked
``no_meaningful_mention=true`` so synthesis will drop them.

This catches clear off-gene papers (focus absent from excerpt/claim). It does
NOT catch family/homolog confusions where the focus symbol is still present in
the claim (those need a real regrade with the new prompt).

Example (Legionella):
  PYTHONPATH=src python pipelines/auto_lit/scripts/apply_focus_identity_gate.py \\
    --output-root $AUTO_LIT_DATA_ROOT/llm_results \\
    --idmap-csv $AUTO_LIT_DATA_ROOT/search_results/lp-human-all_idmap.csv \\
    --from-summary-csv $AUTO_LIT_DATA_ROOT/inputs/lp-human-all_alignments.csv \\
    --backup --write-changed-ids /tmp/lp_focus_gate_changed.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from auto_lit_search.download_manifest import _load_idmap  # noqa: E402
from auto_lit_search.grader_focus_identity import enforce_focus_identity  # noqa: E402
from auto_lit_search.paper_excerpt import build_grader_excerpt_with_meta  # noqa: E402
from auto_lit_search.paper_io import focus_terms_for_paper_role  # noqa: E402
from auto_lit_search.rubric_scoring import rubric_role_for_paper_role  # noqa: E402


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _alignment_ids_from_query_target_csv(path: Path) -> Set[str]:
    ids: Set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        if "alignment_id" in fields:
            key = fields["alignment_id"]
            for row in reader:
                aid = str(row.get(key) or "").strip()
                if aid:
                    ids.add(aid)
            return ids
        qk = fields.get("query")
        tk = fields.get("target")
        if not qk or not tk:
            raise ValueError(f"{path} needs alignment_id or query+target columns")
        for row in reader:
            q = str(row.get(qk) or "").strip()
            t = str(row.get(tk) or "").strip()
            if q and t:
                ids.add(f"{q}_{t}")
    return ids


def _gene_context_for(
    alignment_id: str,
    query: str,
    target_id: str,
    idmap: Dict[str, Dict[str, Any]],
    graded: Dict[str, Any],
) -> Dict[str, Any]:
    existing = graded.get("gene_context")
    if isinstance(existing, dict) and (existing.get("query") or existing.get("target")):
        return existing
    key = f"{query}|{target_id}"
    meta = idmap.get(key) or idmap.get(alignment_id) or {}
    return {
        "query": meta.get("query_meta") or {},
        "target": meta.get("target_meta") or {},
    }


def _load_rubric(path: Optional[str], cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not path:
        return {}
    if path in cache:
        return cache[path]
    p = Path(path)
    if not p.is_file():
        cache[path] = {}
        return cache[path]
    cache[path] = json.loads(p.read_text(encoding="utf-8"))
    return cache[path]


def _paper_changed(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    if bool(before.get("no_meaningful_mention")) != bool(after.get("no_meaningful_mention")):
        return True
    if float(before.get("relevance_grade") or 0) != float(after.get("relevance_grade") or 0):
        return True
    bcs = before.get("criterion_scores") or {}
    acs = after.get("criterion_scores") or {}
    if set(bcs) != set(acs):
        return True
    for k, bv in bcs.items():
        bscore = bv.get("score") if isinstance(bv, dict) else bv
        av = acs.get(k)
        ascore = av.get("score") if isinstance(av, dict) else av
        if int(bscore or 0) != int(ascore or 0):
            return True
    return False


def triage_alignment(
    graded_path: Path,
    *,
    idmap: Dict[str, Dict[str, Any]],
    rubric_cache: Dict[str, Dict[str, Any]],
    backup: bool,
    dry_run: bool,
    ts: str,
) -> Tuple[str, str, Dict[str, Any]]:
    graded = _load_json(graded_path)
    alignment_id = str(graded.get("alignment_id") or graded_path.name[: -len("_graded.json")])
    query = str(graded.get("query") or "").strip()
    target_id = str(graded.get("target_id") or graded.get("target") or "").strip()
    papers_dir = Path(str(graded.get("papers_dir") or ""))
    if not query or not target_id:
        parts = alignment_id.split("_", 1)
        if len(parts) == 2:
            query, target_id = parts[0], parts[1]
    gene_context = _gene_context_for(alignment_id, query, target_id, idmap, graded)
    meta = graded.get("grading_meta") if isinstance(graded.get("grading_meta"), dict) else {}
    host_rubric = _load_rubric(str(meta.get("host_rubric_path") or ""), rubric_cache)
    microbe_rubric = _load_rubric(str(meta.get("microbe_rubric_path") or ""), rubric_cache)

    papers = graded.get("graded_papers") or []
    if not isinstance(papers, list):
        return alignment_id, "error", {"error": "graded_papers missing"}

    n_forced = 0
    n_checked = 0
    examples: List[str] = []
    new_papers: List[Dict[str, Any]] = []
    for row in papers:
        if not isinstance(row, dict):
            continue
        n_checked += 1
        role = str(row.get("paper_role") or "").strip().lower()
        rubric_role = rubric_role_for_paper_role(role)
        rubric = microbe_rubric if rubric_role == "microbe" else host_rubric
        if not rubric:
            new_papers.append(row)
            continue
        focus_terms = focus_terms_for_paper_role(role, query, target_id, gene_context)
        fname = str(row.get("file_name") or "").strip()
        text_path = papers_dir / fname if papers_dir and fname else None
        excerpt = ""
        if text_path and text_path.is_file():
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            excerpt = build_grader_excerpt_with_meta(text).excerpt
        parsed = {
            "criterion_scores": row.get("criterion_scores") or {},
            "mention_type": row.get("mention_type"),
            "no_meaningful_mention": bool(row.get("no_meaningful_mention")),
            "claim_summary": row.get("claim_summary") or "",
            "rationale": row.get("rationale") or "",
            "rubric_tags": row.get("rubric_tags") or {},
            "infection_naive": row.get("infection_naive"),
        }
        after = enforce_focus_identity(
            parsed,
            excerpt=excerpt,
            focus_terms=focus_terms,
            rubric=rubric,
            rubric_role=rubric_role,
        )
        updated = dict(row)
        for key in (
            "criterion_scores",
            "mention_type",
            "no_meaningful_mention",
            "claim_summary",
            "rationale",
            "rubric_tags",
            "relevance_grade",
            "relevance_sort",
            "paper_grade",
            "primary_grade",
            "axis_totals",
            "rubric_dimension_scores",
            "rubric_axis_rationales",
            "grading_schema_version",
        ):
            if key in after:
                updated[key] = after[key]
        if _paper_changed(row, updated):
            n_forced += 1
            if len(examples) < 5:
                examples.append(fname or updated.get("paper_id") or "?")
        new_papers.append(updated)

    info = {
        "n_papers": n_checked,
        "n_forced_nmm": n_forced,
        "examples": examples,
    }
    if n_forced == 0:
        return alignment_id, "unchanged", info

    if dry_run:
        return alignment_id, "would_update", info

    if backup:
        bak = graded_path.with_name(f"{graded_path.name}.bak.{ts}")
        if not bak.is_file():
            shutil.copy2(graded_path, bak)
    graded["graded_papers"] = new_papers
    if not graded.get("gene_context"):
        graded["gene_context"] = gene_context
    gate_meta = graded.get("grading_meta") if isinstance(graded.get("grading_meta"), dict) else {}
    gate_meta = dict(gate_meta)
    gate_meta["focus_identity_offline_gate"] = {
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_forced_nmm": n_forced,
    }
    graded["grading_meta"] = gate_meta
    graded_path.write_text(
        json.dumps(graded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return alignment_id, "updated", info


def _iter_alignment_ids(output_root: Path, only: Optional[Set[str]]) -> Iterable[str]:
    for graded in sorted(output_root.glob("*_graded.json")):
        aid = graded.name[: -len("_graded.json")]
        if only is not None and aid not in only:
            continue
        yield aid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--idmap-csv", type=Path, required=True)
    p.add_argument(
        "--from-summary-csv",
        type=Path,
        default=None,
        help="Restrict to alignment_id or query/target pairs in this CSV",
    )
    p.add_argument("--alignments-file", type=Path, default=None)
    p.add_argument("--backup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--write-changed-ids",
        type=Path,
        default=None,
        help="Write alignment_ids that changed (for fix_it_synthesis --alignments-file)",
    )
    args = p.parse_args()

    if not args.output_root.is_dir():
        print(f"Not a directory: {args.output_root}", file=sys.stderr)
        return 2
    if not args.idmap_csv.is_file():
        print(f"Not found: {args.idmap_csv}", file=sys.stderr)
        return 2

    only: Optional[Set[str]] = None
    if args.from_summary_csv:
        only = _alignment_ids_from_query_target_csv(args.from_summary_csv)
    if args.alignments_file:
        ids = {
            ln.strip()
            for ln in args.alignments_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
        only = ids if only is None else (only & ids)

    idmap = _load_idmap(str(args.idmap_csv))
    rubric_cache: Dict[str, Dict[str, Any]] = {}
    ts = time.strftime("%Y%m%d_%H%M%S")
    changed: List[str] = []
    n_unchanged = n_updated = n_err = 0

    for aid in _iter_alignment_ids(args.output_root, only):
        graded_path = args.output_root / f"{aid}_graded.json"
        try:
            alignment_id, status, info = triage_alignment(
                graded_path,
                idmap=idmap,
                rubric_cache=rubric_cache,
                backup=args.backup,
                dry_run=args.dry_run,
                ts=ts,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{aid}\terror\t{exc}", file=sys.stderr)
            n_err += 1
            continue
        note = (
            f"forced_nmm={info.get('n_forced_nmm')}/{info.get('n_papers')} "
            f"examples={','.join(info.get('examples') or [])}"
        )
        print(f"{alignment_id}\t{status}\t{note}")
        if status in ("updated", "would_update"):
            n_updated += 1
            changed.append(alignment_id)
        elif status == "unchanged":
            n_unchanged += 1

    if args.write_changed_ids is not None:
        args.write_changed_ids.parent.mkdir(parents=True, exist_ok=True)
        args.write_changed_ids.write_text(
            "\n".join(changed) + ("\n" if changed else ""), encoding="utf-8"
        )
        print(f"Wrote {len(changed)} ids -> {args.write_changed_ids}", file=sys.stderr)

    print(
        f"Done: changed={n_updated} unchanged={n_unchanged} errors={n_err} "
        f"dry_run={args.dry_run}",
        file=sys.stderr,
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
