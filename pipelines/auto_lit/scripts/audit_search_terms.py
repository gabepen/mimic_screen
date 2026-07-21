#!/usr/bin/env python3
"""Audit idmap gene/common terms against search-term usability rules.

Example:
  python pipelines/auto_lit/scripts/audit_search_terms.py \\
    --idmap /path/to/wol-dros-v1_idmap.csv \\
    --search-json /path/to/wol-dros-v1_search.json \\
    --out /path/to/search_results/term_audit_wol-dros-v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auto_lit_search.search_terms import (  # noqa: E402
    is_usable_search_term,
    normalize_search_term,
    term_reject_reasons,
)


def _load_search_counts(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    for q, entries in data.items():
        if not isinstance(entries, list):
            continue
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            t = str(ent.get("target") or "")
            key = f"{q}_{t}" if t else str(q)
            out[key] = {
                "n_query_papers": int(ent.get("n_query_papers") or 0),
                "n_target_papers": int(ent.get("n_target_papers") or 0),
            }
    return out


def _flag_side(gene: Optional[str], common: Optional[str], locus: Optional[str]) -> List[str]:
    flags: List[str] = []
    g = normalize_search_term(gene)
    c = normalize_search_term(common)
    loc = normalize_search_term(locus)
    if g:
        reasons = term_reject_reasons(g, kind="gene_name")
        if reasons:
            flags.append(f"gene_name:{','.join(reasons)}")
        elif not is_usable_search_term(g, kind="gene_name"):
            flags.append("gene_name:rejected")
    if c:
        reasons = term_reject_reasons(c, kind="common_name")
        if reasons:
            flags.append(f"common_name:{','.join(reasons)}")
    if g and loc and g.lower() == loc.lower() and "_rs" in g.lower():
        flags.append("rs_locus_as_gene_name")
    return flags


def audit_idmap(
    idmap_path: Path,
    search_counts: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    with idmap_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            t = (row.get("target") or "").strip()
            aid = f"{q}_{t}" if q and t else q or t
            counts = search_counts.get(aid, {})
            q_flags = _flag_side(
                row.get("query_gene_name"),
                row.get("query_common_name"),
                row.get("query_locus_tag"),
            )
            t_flags = _flag_side(
                row.get("target_gene_name"),
                row.get("target_common_name"),
                row.get("target_locus_tag"),
            )
            usable_q_gene = is_usable_search_term(row.get("query_gene_name"), kind="gene_name")
            usable_q_common = is_usable_search_term(
                row.get("query_common_name"), kind="common_name"
            )
            usable_t_gene = is_usable_search_term(row.get("target_gene_name"), kind="gene_name")
            usable_t_common = is_usable_search_term(
                row.get("target_common_name"), kind="common_name"
            )
            rows_out.append(
                {
                    "alignment_id": aid,
                    "query": q,
                    "target": t,
                    "query_gene_name": row.get("query_gene_name") or "",
                    "query_common_name": row.get("query_common_name") or "",
                    "target_gene_name": row.get("target_gene_name") or "",
                    "target_common_name": row.get("target_common_name") or "",
                    "n_query_papers": counts.get("n_query_papers", ""),
                    "n_target_papers": counts.get("n_target_papers", ""),
                    "query_gene_usable": usable_q_gene,
                    "query_common_usable": usable_q_common,
                    "target_gene_usable": usable_t_gene,
                    "target_common_usable": usable_t_common,
                    "query_flags": ";".join(q_flags),
                    "target_flags": ";".join(t_flags),
                    "any_bad_term": bool(q_flags or t_flags),
                }
            )
    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idmap", required=True, type=Path)
    ap.add_argument("--search-json", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    counts = _load_search_counts(args.search_json)
    rows = audit_idmap(args.idmap, counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "alignment_id",
        "any_bad_term",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_bad = sum(1 for r in rows if r["any_bad_term"])
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Rows with rejected/flagged terms: {n_bad}")
    # Top flagged common names
    bad_commons: Dict[str, int] = {}
    for r in rows:
        for side in ("query_common_name", "target_common_name"):
            val = (r.get(side) or "").strip()
            usable_key = (
                "query_common_usable" if side.startswith("query") else "target_common_usable"
            )
            if val and not r.get(usable_key):
                bad_commons[val] = bad_commons.get(val, 0) + 1
    if bad_commons:
        print("Top rejected common_name values:")
        for name, n in sorted(bad_commons.items(), key=lambda x: -x[1])[:20]:
            print(f"  {n:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
