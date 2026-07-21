#!/usr/bin/env python3
"""Compare archived vs new stage1 idmap/search outputs.

Emits term diffs, per-DOI add/remove lists, and stage2 action lists:
  - noop: identical DOI sets (term-only changes still logged)
  - prune_and_resynth: removals only
  - add_grade_resynth: additions (with or without removals)

Example:
  python pipelines/auto_lit/scripts/diff_stage1_runs.py \\
    --dataset wol-dros-v1 \\
    --old-idmap search_results/wol-dros-v1_idmap.csv.20260716 \\
    --new-idmap search_results/wol-dros-v1_idmap.csv \\
    --old-search search_results/wol-dros-v1_search.json.20260716 \\
    --new-search search_results/wol-dros-v1_search.json \\
    --out-dir search_results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


IDMAP_TERM_COLS = [
    "query_gene_name",
    "query_gene_aliases",
    "query_common_name",
    "target_gene_name",
    "target_gene_aliases",
    "target_common_name",
]

IDMAP_COMPARE_COLS = IDMAP_TERM_COLS + [
    "query_locus_tag",
    "query_genbank_acc",
    "target_locus_tag",
    "target_genbank_acc",
]


def _load_idmap(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("query") or "").strip()
            t = (row.get("target") or "").strip()
            if not q or not t:
                continue
            out[f"{q}_{t}"] = {k: (row.get(k) or "").strip() for k in row}
    return out


def _load_search(path: Path) -> Dict[str, Dict[str, Any]]:
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
            aid = f"{q}_{t}" if t else str(q)
            q_dois = set(ent.get("query_paper_dois") or [])
            t_dois = set(ent.get("target_paper_dois") or [])
            out[aid] = {
                "n_query_papers": int(ent.get("n_query_papers") or 0),
                "n_target_papers": int(ent.get("n_target_papers") or 0),
                "query_dois": q_dois,
                "target_dois": t_dois,
                "query_search_terms": list(ent.get("query_search_terms") or []),
                "target_search_terms": list(ent.get("target_search_terms") or []),
                "query_sources": dict(ent.get("query_paper_ids_by_source") or {}),
                "target_sources": dict(ent.get("target_paper_ids_by_source") or {}),
                "query_term_hits": dict(ent.get("query_paper_term_hits") or {}),
                "target_term_hits": dict(ent.get("target_paper_term_hits") or {}),
                "query_unattributed": dict(
                    ent.get("query_unattributed_term_hit_dois_by_pass") or {}
                ),
                "target_unattributed": dict(
                    ent.get("target_unattributed_term_hit_dois_by_pass") or {}
                ),
                "query_fallbacks": list(ent.get("query_organism_fallbacks") or []),
                "target_fallbacks": list(ent.get("target_organism_fallbacks") or []),
            }
    return out


def _norm_terms(vals: Any) -> List[str]:
    if vals is None:
        return []
    if isinstance(vals, str):
        parts = [p.strip() for p in vals.replace(";", "|").split("|") if p.strip()]
        return sorted({p.lower(): p for p in parts}.values(), key=str.lower)
    if isinstance(vals, (list, tuple)):
        parts = [str(x).strip() for x in vals if str(x).strip()]
        return sorted({p.lower(): p for p in parts}.values(), key=str.lower)
    return []


def _terms_key(terms: List[str]) -> str:
    return "|".join(sorted({t.lower() for t in terms if t}))


def _doi_delta(old: Set[str], new: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    kept = old & new
    added = new - old
    removed = old - new
    return kept, added, removed


ATTRIBUTION_CLASSES = (
    "direct_database",
    "organism_scoped_text",
    "identifier_text",
    "unscoped_text",
    "unattributed",
)


def _doi_attribution(search: Dict[str, Any], side: str, doi: str) -> str:
    sources = search.get(f"{side}_sources") or {}
    direct = set(sources.get("entrez_pubtator") or []) | set(
        sources.get("europepmc_accession") or []
    )
    if doi in direct:
        return "direct_database"

    hits = search.get(f"{side}_term_hits") or {}
    records = hits.get(doi) or []
    if any(
        record.get("taxids") or record.get("organism_terms")
        for record in records
        if isinstance(record, dict)
    ):
        return "organism_scoped_text"
    if any(
        record.get("pass") == "pass1" for record in records if isinstance(record, dict)
    ):
        return "identifier_text"
    if any(
        str(record.get("pass") or "").startswith("pass2")
        for record in records
        if isinstance(record, dict)
    ):
        return "unscoped_text"

    unattributed = search.get(f"{side}_unattributed") or {}
    fallback_passes = {
        str(event.get("pass") or "")
        for event in search.get(f"{side}_fallbacks") or []
        if int(event.get("n_kept") or 0) > 0
    }
    for pass_name, dois in unattributed.items():
        if doi in (dois or []) and pass_name in fallback_passes:
            return "unscoped_text"
    return "unattributed"


def _attribution_counts(
    search: Dict[str, Any], side: str, dois: Set[str]
) -> Counter[str]:
    return Counter(_doi_attribution(search, side, doi) for doi in dois)


def _fallback_metrics(search: Dict[str, Any], side: str) -> Dict[str, Any]:
    events = search.get(f"{side}_fallbacks") or []
    retained = [event for event in events if int(event.get("n_kept") or 0) > 0]
    truncated = [event for event in events if bool(event.get("truncated"))]
    taxid_sets = sorted(
        {
            ",".join(str(value) for value in (event.get("dropped_taxids") or []))
            for event in events
        }
        - {""}
    )
    return {
        "events": len(events),
        "retained_events": len(retained),
        "truncated_events": len(truncated),
        "n_kept": sum(int(event.get("n_kept") or 0) for event in events),
        "max_hit_count": max(
            (int(event.get("hit_count") or 0) for event in events), default=0
        ),
        "taxid_sets": ";".join(taxid_sets),
    }


def _action_for(added_q: Set[str], removed_q: Set[str], added_t: Set[str], removed_t: Set[str]) -> str:
    added = bool(added_q or added_t)
    removed = bool(removed_q or removed_t)
    if not added and not removed:
        return "noop"
    if added:
        return "add_grade_resynth"
    return "prune_and_resynth"


def diff_runs(
    old_idmap: Dict[str, Dict[str, str]],
    new_idmap: Dict[str, Dict[str, str]],
    old_search: Dict[str, Dict[str, Any]],
    new_search: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
    aids = sorted(set(old_idmap) | set(new_idmap) | set(old_search) | set(new_search))
    summary_rows: List[Dict[str, Any]] = []
    doi_rows: List[Dict[str, Any]] = []
    actions: Dict[str, List[str]] = {
        "noop": [],
        "prune_and_resynth": [],
        "add_grade_resynth": [],
    }

    for aid in aids:
        o_id = old_idmap.get(aid, {})
        n_id = new_idmap.get(aid, {})
        o_s = old_search.get(aid, {})
        n_s = new_search.get(aid, {})

        id_changed_cols = [
            c for c in IDMAP_COMPARE_COLS if (o_id.get(c) or "") != (n_id.get(c) or "")
        ]

        old_q_terms = _norm_terms(o_s.get("query_search_terms")) or _norm_terms(
            o_id.get("query_gene_name")
        ) + _norm_terms(o_id.get("query_gene_aliases")) + _norm_terms(
            o_id.get("query_common_name")
        )
        new_q_terms = _norm_terms(n_s.get("query_search_terms")) or _norm_terms(
            n_id.get("query_gene_name")
        ) + _norm_terms(n_id.get("query_gene_aliases"))
        old_t_terms = _norm_terms(o_s.get("target_search_terms")) or _norm_terms(
            o_id.get("target_gene_name")
        ) + _norm_terms(o_id.get("target_gene_aliases")) + _norm_terms(
            o_id.get("target_common_name")
        )
        new_t_terms = _norm_terms(n_s.get("target_search_terms")) or _norm_terms(
            n_id.get("target_gene_name")
        ) + _norm_terms(n_id.get("target_gene_aliases"))

        terms_changed = (_terms_key(old_q_terms) != _terms_key(new_q_terms)) or (
            _terms_key(old_t_terms) != _terms_key(new_t_terms)
        )

        o_qd: Set[str] = set(o_s.get("query_dois") or set())
        n_qd: Set[str] = set(n_s.get("query_dois") or set())
        o_td: Set[str] = set(o_s.get("target_dois") or set())
        n_td: Set[str] = set(n_s.get("target_dois") or set())
        k_q, a_q, r_q = _doi_delta(o_qd, n_qd)
        k_t, a_t, r_t = _doi_delta(o_td, n_td)
        action = _action_for(a_q, r_q, a_t, r_t)
        actions[action].append(aid)
        q_added_attr = _attribution_counts(n_s, "query", a_q)
        t_added_attr = _attribution_counts(n_s, "target", a_t)
        q_fallback = _fallback_metrics(n_s, "query")
        t_fallback = _fallback_metrics(n_s, "target")

        summary_rows.append(
            {
                "alignment_id": aid,
                "action": action,
                "terms_changed": terms_changed,
                "idmap_changed_cols": ",".join(id_changed_cols),
                "old_query_gene_name": o_id.get("query_gene_name", ""),
                "new_query_gene_name": n_id.get("query_gene_name", ""),
                "old_query_gene_aliases": o_id.get("query_gene_aliases", ""),
                "new_query_gene_aliases": n_id.get("query_gene_aliases", ""),
                "old_query_common_name": o_id.get("query_common_name", ""),
                "new_query_common_name": n_id.get("query_common_name", ""),
                "old_target_gene_name": o_id.get("target_gene_name", ""),
                "new_target_gene_name": n_id.get("target_gene_name", ""),
                "old_target_gene_aliases": o_id.get("target_gene_aliases", ""),
                "new_target_gene_aliases": n_id.get("target_gene_aliases", ""),
                "old_target_common_name": o_id.get("target_common_name", ""),
                "new_target_common_name": n_id.get("target_common_name", ""),
                "old_query_search_terms": "|".join(old_q_terms),
                "new_query_search_terms": "|".join(new_q_terms),
                "old_target_search_terms": "|".join(old_t_terms),
                "new_target_search_terms": "|".join(new_t_terms),
                "n_query_papers_old": len(o_qd),
                "n_query_papers_new": len(n_qd),
                "n_target_papers_old": len(o_td),
                "n_target_papers_new": len(n_td),
                "query_kept": len(k_q),
                "query_added": len(a_q),
                "query_removed": len(r_q),
                "target_kept": len(k_t),
                "target_added": len(a_t),
                "target_removed": len(r_t),
                **{
                    f"query_added_{key}": q_added_attr[key]
                    for key in ATTRIBUTION_CLASSES
                },
                **{
                    f"target_added_{key}": t_added_attr[key]
                    for key in ATTRIBUTION_CLASSES
                },
                "query_fallback_events": q_fallback["events"],
                "query_fallback_retained_events": q_fallback["retained_events"],
                "query_fallback_truncated_events": q_fallback["truncated_events"],
                "query_fallback_n_kept": q_fallback["n_kept"],
                "query_fallback_max_hit_count": q_fallback["max_hit_count"],
                "query_fallback_dropped_taxid_sets": q_fallback["taxid_sets"],
                "target_fallback_events": t_fallback["events"],
                "target_fallback_retained_events": t_fallback["retained_events"],
                "target_fallback_truncated_events": t_fallback["truncated_events"],
                "target_fallback_n_kept": t_fallback["n_kept"],
                "target_fallback_max_hit_count": t_fallback["max_hit_count"],
                "target_fallback_dropped_taxid_sets": t_fallback["taxid_sets"],
            }
        )
        doi_rows.append(
            {
                "alignment_id": aid,
                "action": action,
                "query_kept_dois": sorted(k_q),
                "query_added_dois": sorted(a_q),
                "query_removed_dois": sorted(r_q),
                "target_kept_dois": sorted(k_t),
                "target_added_dois": sorted(a_t),
                "target_removed_dois": sorted(r_t),
            }
        )

    return summary_rows, doi_rows, actions


def _write_md(
    path: Path,
    dataset: str,
    rows: List[Dict[str, Any]],
    actions: Dict[str, List[str]],
    *,
    old_idmap_path: Path,
    new_idmap_path: Path,
    old_search_path: Path,
    new_search_path: Path,
) -> None:
    n = len(rows)
    n_terms = sum(1 for r in rows if r["terms_changed"])
    query_old = sum(int(r["n_query_papers_old"]) for r in rows)
    query_new = sum(int(r["n_query_papers_new"]) for r in rows)
    target_old = sum(int(r["n_target_papers_old"]) for r in rows)
    target_new = sum(int(r["n_target_papers_new"]) for r in rows)
    query_added = sum(int(r["query_added"]) for r in rows)
    query_removed = sum(int(r["query_removed"]) for r in rows)
    target_added = sum(int(r["target_added"]) for r in rows)
    target_removed = sum(int(r["target_removed"]) for r in rows)

    def _mtime(value: Path) -> str:
        return datetime.fromtimestamp(
            value.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds")

    def _fallback_summary(side: str) -> Dict[str, Any]:
        return {
            "alignments": sum(
                int(r[f"{side}_fallback_events"]) > 0 for r in rows
            ),
            "retained_alignments": sum(
                int(r[f"{side}_fallback_retained_events"]) > 0 for r in rows
            ),
            "truncated_alignments": sum(
                int(r[f"{side}_fallback_truncated_events"]) > 0 for r in rows
            ),
            "events": sum(int(r[f"{side}_fallback_events"]) for r in rows),
            "truncated_events": sum(
                int(r[f"{side}_fallback_truncated_events"]) for r in rows
            ),
            "n_kept": sum(int(r[f"{side}_fallback_n_kept"]) for r in rows),
            "max_hit_count": max(
                (int(r[f"{side}_fallback_max_hit_count"]) for r in rows), default=0
            ),
            "taxid_sets": sorted(
                {
                    value
                    for r in rows
                    for value in str(
                        r[f"{side}_fallback_dropped_taxid_sets"]
                    ).split(";")
                    if value
                }
            ),
        }

    query_fallback = _fallback_summary("query")
    target_fallback = _fallback_summary("target")
    lines = [
        f"# Stage1 diff: {dataset}",
        "",
        "## Artifact provenance",
        "",
        f"- Old idmap: `{old_idmap_path}` (mtime UTC: {_mtime(old_idmap_path)})",
        f"- New idmap: `{new_idmap_path}` (mtime UTC: {_mtime(new_idmap_path)})",
        f"- Old search: `{old_search_path}` (mtime UTC: {_mtime(old_search_path)})",
        f"- New search: `{new_search_path}` (mtime UTC: {_mtime(new_search_path)})",
        "",
        "## DOI-set changes",
        "",
        f"- Alignments compared: {n}",
        f"- Term changes: {n_terms}",
        f"- noop (DOI set unchanged): {len(actions['noop'])}",
        f"- prune_and_resynth: {len(actions['prune_and_resynth'])}",
        f"- add_grade_resynth: {len(actions['add_grade_resynth'])}",
        f"- Query papers: {query_old} → {query_new} ({query_new - query_old:+d})",
        f"- Target papers: {target_old} → {target_new} ({target_new - target_old:+d})",
        f"- Query DOI churn: +{query_added} / -{query_removed}",
        f"- Target DOI churn: +{target_added} / -{target_removed}",
        "",
        "## Attribution of newly added papers",
        "",
        "| side | direct database | organism-scoped text | identifier text | unscoped text | unattributed |",
        "|---|---:|---:|---:|---:|---:|",
        "| query | "
        + " | ".join(
            str(sum(int(r[f"query_added_{key}"]) for r in rows))
            for key in ATTRIBUTION_CLASSES
        )
        + " |",
        "| target | "
        + " | ".join(
            str(sum(int(r[f"target_added_{key}"]) for r in rows))
            for key in ATTRIBUTION_CLASSES
        )
        + " |",
        "",
        "These categories are mutually exclusive. `unscoped text` means the DOI was "
        "attributed to a pass2 term after the organism constraint was dropped. "
        "`unattributed` means the combined query returned the DOI but the audit "
        "queries could not assign it to a specific direct/scoped/identifier source.",
        "",
        "## Organism-fallback risk in the new search",
        "",
        "| side | alignments fallback fired | alignments retained fallback papers | "
        "alignments with truncated fallback | fallback events | truncated events | "
        "sum n_kept | largest raw hit count | dropped taxid sets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| query | {query_fallback['alignments']} | "
        f"{query_fallback['retained_alignments']} | "
        f"{query_fallback['truncated_alignments']} | "
        f"{query_fallback['events']} | {query_fallback['truncated_events']} | "
        f"{query_fallback['n_kept']} | {query_fallback['max_hit_count']} | "
        f"{', '.join(query_fallback['taxid_sets']) or 'none'} |",
        f"| target | {target_fallback['alignments']} | "
        f"{target_fallback['retained_alignments']} | "
        f"{target_fallback['truncated_alignments']} | "
        f"{target_fallback['events']} | {target_fallback['truncated_events']} | "
        f"{target_fallback['n_kept']} | {target_fallback['max_hit_count']} | "
        f"{', '.join(target_fallback['taxid_sets']) or 'none'} |",
        "",
        "`truncated` means Europe PMC reported more than the 200-record page size. "
        "`n_kept` is after filtering and can therefore be below 200.",
        "",
        "## Sample alignments with term or paper changes",
        "",
    ]
    sample = [r for r in rows if r["terms_changed"] or r["action"] != "noop"][:40]
    if not sample:
        lines.append("_None_")
    else:
        lines.append("| alignment_id | action | terms_changed | Δquery | Δtarget |")
        lines.append("|---|---|---|---:|---:|")
        for r in sample:
            dq = int(r["n_query_papers_new"]) - int(r["n_query_papers_old"])
            dt = int(r["n_target_papers_new"]) - int(r["n_target_papers_old"])
            lines.append(
                f"| `{r['alignment_id']}` | {r['action']} | {r['terms_changed']} "
                f"| {dq:+d} | {dt:+d} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--old-idmap", required=True, type=Path)
    ap.add_argument("--new-idmap", required=True, type=Path)
    ap.add_argument("--old-search", required=True, type=Path)
    ap.add_argument("--new-search", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    old_id = _load_idmap(args.old_idmap)
    new_id = _load_idmap(args.new_idmap)
    old_s = _load_search(args.old_search)
    new_s = _load_search(args.new_search)
    summary_rows, doi_rows, actions = diff_runs(old_id, new_id, old_s, new_s)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"stage1_diff_{args.dataset}.csv"
    terms_path = args.out_dir / f"stage1_term_diff_{args.dataset}.csv"
    doi_path = args.out_dir / f"stage1_doi_delta_{args.dataset}.jsonl"
    md_path = args.out_dir / f"stage1_diff_{args.dataset}.md"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["alignment_id"]
        )
        w.writeheader()
        w.writerows(summary_rows)

    term_fields = [
        "alignment_id",
        "terms_changed",
        "action",
        "old_query_gene_name",
        "new_query_gene_name",
        "old_query_gene_aliases",
        "new_query_gene_aliases",
        "old_query_common_name",
        "new_query_common_name",
        "old_target_gene_name",
        "new_target_gene_name",
        "old_target_gene_aliases",
        "new_target_gene_aliases",
        "old_target_common_name",
        "new_target_common_name",
        "old_query_search_terms",
        "new_query_search_terms",
        "old_target_search_terms",
        "new_target_search_terms",
    ]
    with terms_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=term_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in term_fields})

    with doi_path.open("w", encoding="utf-8") as f:
        for row in doi_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for action, ids in actions.items():
        p = args.out_dir / f"stage2_action_{action}_{args.dataset}.txt"
        p.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    # Convenience: all alignments needing any stage2 work
    changed = actions["prune_and_resynth"] + actions["add_grade_resynth"]
    (args.out_dir / f"reanalysis_alignment_ids_{args.dataset}.txt").write_text(
        "\n".join(changed) + ("\n" if changed else ""),
        encoding="utf-8",
    )

    _write_md(
        md_path,
        args.dataset,
        summary_rows,
        actions,
        old_idmap_path=args.old_idmap,
        new_idmap_path=args.new_idmap,
        old_search_path=args.old_search,
        new_search_path=args.new_search,
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {terms_path}")
    print(f"Wrote {doi_path}")
    print(f"Wrote {md_path}")
    for action, ids in actions.items():
        print(f"  {action}: {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
