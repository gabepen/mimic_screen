#!/usr/bin/env python3
"""Sample graded papers for manual blind rubric QA.

Writes a human-facing CSV (no LLM grades) and a separate answer-key JSON.
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
from auto_lit_search.env_config import resolve_rubric_paths  # noqa: E402
from auto_lit_search.paper_io import gene_terms  # noqa: E402
from auto_lit_search.rubric_scoring import rubric_criteria  # noqa: E402

BLIND_CSV_FIELDS = [
    "sample_id",
    "doi",
    "gene_focus_id",
    "paper_role",
    "alignment_id",
    "query_gene_id",
    "target_gene_id",
    "file_name",
    "human_relevant_yes_no",
    "human_relevance_0_1",
    "human_notes",
]

BLIND_CSV_BASE_FIELDS = [
    "sample_id",
    "doi",
    "gene_focus_id",
    "paper_role",
    "alignment_id",
    "query_gene_id",
    "target_gene_id",
    "file_name",
    "text_path",
    "papers_dir",
    "gene_focus_symbol",
    "gene_focus_common_name",
    "gene_focus_search_terms",
    "query_search_terms",
    "target_search_terms",
    "applicable_criteria",
]

IDENTIFIER_FIELDS = [
    "text_path",
    "gene_focus_symbol",
    "gene_focus_common_name",
    "gene_focus_search_terms",
    "query_search_terms",
    "target_search_terms",
]

GRADE_BINS: List[Tuple[str, float, float, float]] = [
    ("low", 0.0, 0.25, 0.40),
    ("mid", 0.25, 0.5, 0.40),
    ("high", 0.5, 1.0, 0.20),
]


@dataclass
class PaperRecord:
    alignment_id: str
    query_gene_id: str
    target_gene_id: str
    doi: str
    paper_role: str
    gene_focus_id: str
    file_name: str
    papers_dir: str
    relevance_grade: float
    rubric_dimension_scores: Dict[str, float]
    rubric_axis_rationales: Dict[str, str]
    rubric_tags: Dict[str, str]
    criterion_scores: Dict[str, Dict[str, Any]]
    axis_totals: Dict[str, Any]
    paper_grade: str
    primary_grade: str
    rationale: str
    grading_meta: Dict[str, Any] = field(default_factory=dict)
    text_path: str = ""
    gene_focus_symbol: str = ""
    gene_focus_common_name: str = ""
    gene_focus_search_terms: str = ""
    query_search_terms: str = ""
    target_search_terms: str = ""
    gene_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricCriterion:
    criterion_id: str
    axis_id: str
    weight: str
    paper_roles: Tuple[str, ...]


@dataclass
class RubricSpec:
    host_path: Path
    microbe_path: Path
    host_version: str
    microbe_version: str
    host_rubric: Dict[str, Any]
    microbe_rubric: Dict[str, Any]
    criteria: Tuple[RubricCriterion, ...]

    def rubric_for_role(self, paper_role: str) -> Dict[str, Any]:
        role = (paper_role or "").strip().lower()
        if role in ("query", "microbe"):
            return self.microbe_rubric
        return self.host_rubric

    def criteria_for_role(self, paper_role: str) -> Tuple[str, ...]:
        role = (paper_role or "").strip().lower()
        if role in ("query", "microbe"):
            return tuple(
                c.criterion_id
                for c in self.criteria
                if "query" in c.paper_roles or "microbe" in c.paper_roles
            )
        return tuple(
            c.criterion_id
            for c in self.criteria
            if "target" in c.paper_roles or "host" in c.paper_roles
        )


def _load_rubric_criteria(
    path: Path, paper_roles: Tuple[str, ...]
) -> Tuple[str, Dict[str, Any], Tuple[RubricCriterion, ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = str(data.get("rubric_version") or "")
    scored, _ = rubric_criteria(data)
    criteria = tuple(
        RubricCriterion(
            criterion_id=crit.id,
            axis_id=crit.axis_id,
            weight=crit.weight,
            paper_roles=paper_roles,
        )
        for crit in scored
    )
    return version, data, criteria


def load_rubric_spec(host_rubric: Path, microbe_rubric: Path) -> RubricSpec:
    host_version, host_data, host_criteria = _load_rubric_criteria(
        host_rubric, ("target", "host")
    )
    microbe_version, microbe_data, microbe_criteria = _load_rubric_criteria(
        microbe_rubric, ("query", "microbe")
    )
    seen: set[str] = set()
    merged: List[RubricCriterion] = []
    for crit in host_criteria + microbe_criteria:
        if crit.criterion_id in seen:
            continue
        seen.add(crit.criterion_id)
        merged.append(crit)
    return RubricSpec(
        host_path=host_rubric,
        microbe_path=microbe_rubric,
        host_version=host_version,
        microbe_version=microbe_version,
        host_rubric=host_data,
        microbe_rubric=microbe_data,
        criteria=tuple(merged),
    )


def blind_csv_fieldnames(
    per_criterion: bool,
    rubric_spec: Optional[RubricSpec],
    *,
    include_identifiers: bool,
) -> List[str]:
    if per_criterion and rubric_spec is not None:
        fields = list(BLIND_CSV_BASE_FIELDS)
        for crit in rubric_spec.criteria:
            fields.append(f"human_{crit.criterion_id}")
        fields.append("human_notes")
        return fields
    fields = list(BLIND_CSV_FIELDS)
    if include_identifiers:
        insert_at = fields.index("file_name") + 1
        for name in reversed(IDENTIFIER_FIELDS):
            if name not in fields:
                fields.insert(insert_at, name)
        if "papers_dir" not in fields:
            fields.insert(insert_at, "papers_dir")
    return fields


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
    alignment_id: str,
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


def _attach_identifiers(
    rec: PaperRecord,
    *,
    idmap: Dict[str, Dict[str, Any]],
) -> None:
    ctx = _gene_context_for_alignment(
        rec.alignment_id, rec.query_gene_id, rec.target_gene_id, idmap
    )
    rec.gene_context = ctx
    rec.query_search_terms = _format_search_terms(ctx.get("query") or {}, rec.query_gene_id)
    rec.target_search_terms = _format_search_terms(
        ctx.get("target") or {}, rec.target_gene_id
    )
    role = (rec.paper_role or "").strip().lower()
    if role in ("query", "microbe"):
        focus_meta = ctx.get("query") or {}
        focus_id = rec.query_gene_id
    else:
        focus_meta = ctx.get("target") or {}
        focus_id = rec.target_gene_id
    gt = gene_terms(focus_meta, focus_id)
    rec.gene_focus_symbol = gt["symbol"]
    rec.gene_focus_common_name = gt["common_name"]
    rec.gene_focus_search_terms = _format_search_terms(focus_meta, focus_id)
    if rec.papers_dir and rec.file_name:
        rec.text_path = str(Path(rec.papers_dir) / rec.file_name)


def _parse_alignment_id(alignment_id: str) -> Tuple[str, str]:
    parts = alignment_id.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return alignment_id, ""


def _gene_focus_id(query_gene_id: str, target_gene_id: str, paper_role: str) -> str:
    role = (paper_role or "").strip().lower()
    if role == "query":
        return query_gene_id
    if role == "target":
        return target_gene_id
    return query_gene_id or target_gene_id


def _load_sidecar_meta(output_root: Path, alignment_id: str) -> Dict[str, Any]:
    for suffix in ("_results.json", "_analysis.json"):
        path = output_root / f"{alignment_id}{suffix}"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _resolve_papers_dir(
    alignment_id: str,
    graded: Dict[str, Any],
    sidecar: Dict[str, Any],
    papers_root: Optional[Path],
) -> str:
    for src in (graded, sidecar):
        raw = str(src.get("papers_dir") or "").strip()
        if raw:
            return raw
    if papers_root is not None:
        return str(papers_root / alignment_id)
    return ""


def load_candidate_pool(
    output_root: Path,
    papers_root: Optional[Path],
    idmap: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PaperRecord]:
    seen: set[Tuple[str, str]] = set()
    pool: List[PaperRecord] = []

    for graded_path in sorted(output_root.glob("*_graded.json")):
        alignment_id = graded_path.name[: -len("_graded.json")]
        try:
            graded = json.loads(graded_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: skip {graded_path.name}: {e}", file=sys.stderr)
            continue
        if not isinstance(graded, dict):
            continue

        sidecar = _load_sidecar_meta(output_root, alignment_id)
        query_gene_id = str(graded.get("query") or sidecar.get("query") or "").strip()
        target_gene_id = str(
            graded.get("target_id") or sidecar.get("target_id") or ""
        ).strip()
        if not query_gene_id or not target_gene_id:
            q, t = _parse_alignment_id(alignment_id)
            query_gene_id = query_gene_id or q
            target_gene_id = target_gene_id or t

        papers_dir = _resolve_papers_dir(alignment_id, graded, sidecar, papers_root)
        meta = graded.get("grading_meta") if isinstance(graded.get("grading_meta"), dict) else {}

        for row in graded.get("graded_papers") or []:
            if not isinstance(row, dict):
                continue
            file_name = str(row.get("file_name") or "").strip()
            if not file_name:
                continue
            dedupe_key = (alignment_id, file_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            doi = str(row.get("paper_id") or "").strip()
            if not doi:
                stem = Path(file_name).stem.split("__")[0]
                doi = stem.replace("_", "/", 1) if stem.startswith("10.") else stem

            paper_role = str(row.get("paper_role") or "").strip().lower() or "unknown"
            pool.append(
                PaperRecord(
                    alignment_id=alignment_id,
                    query_gene_id=query_gene_id,
                    target_gene_id=target_gene_id,
                    doi=doi,
                    paper_role=paper_role,
                    gene_focus_id=_gene_focus_id(
                        query_gene_id, target_gene_id, paper_role
                    ),
                    file_name=file_name,
                    papers_dir=papers_dir,
                    relevance_grade=float(row.get("relevance_grade") or 0.0),
                    rubric_dimension_scores={
                        str(k): float(v)
                        for k, v in (row.get("rubric_dimension_scores") or {}).items()
                    },
                    rubric_axis_rationales={
                        str(k): str(v)
                        for k, v in (row.get("rubric_axis_rationales") or {}).items()
                    },
                    rubric_tags={
                        str(k): str(v)
                        for k, v in (row.get("rubric_tags") or {}).items()
                    },
                    criterion_scores={
                        str(k): dict(v)
                        for k, v in (row.get("criterion_scores") or {}).items()
                        if isinstance(v, dict)
                    },
                    axis_totals=dict(row.get("axis_totals") or {}),
                    paper_grade=str(row.get("paper_grade") or ""),
                    primary_grade=str(row.get("primary_grade") or ""),
                    rationale=str(row.get("rationale") or ""),
                    grading_meta=dict(meta),
                )
            )
    if idmap:
        for rec in pool:
            _attach_identifiers(rec, idmap=idmap)
    else:
        for rec in pool:
            if rec.papers_dir and rec.file_name:
                rec.text_path = str(Path(rec.papers_dir) / rec.file_name)
    return pool


def _sample_without_replacement(
    rng: random.Random,
    items: Sequence[PaperRecord],
    k: int,
) -> List[PaperRecord]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    return rng.sample(list(items), k)


def _allocate_bin_counts(n_rest: int) -> Dict[str, int]:
    counts = {name: int(n_rest * share) for name, _, _, share in GRADE_BINS}
    remainder = n_rest - sum(counts.values())
    order = sorted(GRADE_BINS, key=lambda x: -x[3])
    i = 0
    while remainder > 0:
        counts[order[i % len(order)][0]] += 1
        remainder -= 1
        i += 1
    return counts


def _bin_pool(pool: Sequence[PaperRecord], name: str) -> List[PaperRecord]:
    for bin_name, lo, hi, _ in GRADE_BINS:
        if bin_name != name:
            continue
        if name == "low":
            return [p for p in pool if 0.0 < p.relevance_grade <= hi]
        return [p for p in pool if lo < p.relevance_grade <= hi]
    return []


def _role_pool(pool: Sequence[PaperRecord], role: str) -> List[PaperRecord]:
    want = role.strip().lower()
    return [p for p in pool if (p.paper_role or "").strip().lower() == want]


def _split_stratum(
    count: int,
    query_left: int,
    target_left: int,
) -> Tuple[int, int]:
    """Split one grade stratum across roles while respecting remaining role budgets."""
    if count <= 0:
        return 0, 0
    q = count // 2
    if count % 2 and query_left > target_left:
        q += 1
    q = min(q, query_left)
    t = min(count - q, target_left)
    q = count - t
    q = min(q, query_left)
    t = count - q
    return q, t


def _allocate_cell_counts(n: int, frac: float) -> Dict[Tuple[str, str], int]:
    """Global grade stratification; each stratum split ~50/50 across query/target."""
    n_query = n // 2
    n_target = n - n_query
    n_zero = min(max(int(round(n * frac)), 0), n)
    n_rest = n - n_zero
    bin_counts = _allocate_bin_counts(n_rest)

    query_left = n_query
    target_left = n_target
    cells: Dict[Tuple[str, str], int] = {}

    strata: List[Tuple[str, int]] = [("zero", n_zero)]
    strata.extend((name, bin_counts[name]) for name, _, _, _ in GRADE_BINS)

    for category, count in strata:
        q, t = _split_stratum(count, query_left, target_left)
        cells[("query", category)] = q
        cells[("target", category)] = t
        query_left -= q
        target_left -= t

    return cells


def _cell_pool(
    pool: Sequence[PaperRecord],
    role: str,
    category: str,
) -> List[PaperRecord]:
    role_pool = _role_pool(pool, role)
    if category == "zero":
        return [p for p in role_pool if p.relevance_grade == 0.0]
    return _bin_pool(role_pool, category)


def sample_records(
    pool: Sequence[PaperRecord],
    n: int,
    rng: random.Random,
    irrelevant_min: float,
    irrelevant_max: float,
) -> Tuple[List[PaperRecord], Dict[str, Any]]:
    if not pool:
        return [], {"warning": "empty pool"}

    n = min(n, len(pool))
    frac = rng.uniform(irrelevant_min, irrelevant_max)
    cells = _allocate_cell_counts(n, frac)

    selected: List[PaperRecord] = []
    remaining_ids: set[Tuple[str, str]] = set()
    cell_picked: Dict[str, int] = {}
    bin_picked: Dict[str, int] = {name: 0 for name, _, _, _ in GRADE_BINS}
    warnings: List[str] = []

    for (role, category), need in cells.items():
        if need <= 0:
            continue
        candidates = [
            p
            for p in _cell_pool(pool, role, category)
            if (p.alignment_id, p.file_name) not in remaining_ids
        ]
        picked = _sample_without_replacement(rng, candidates, need)
        selected.extend(picked)
        remaining_ids.update((p.alignment_id, p.file_name) for p in picked)
        key = f"{role}_{category}"
        cell_picked[key] = len(picked)
        if len(picked) < need:
            warnings.append(f"{key}: only {len(picked)} available (requested {need})")
        if category != "zero":
            bin_picked[category] = bin_picked.get(category, 0) + len(picked)

    if len(selected) < n:
        leftovers = [
            p for p in pool if (p.alignment_id, p.file_name) not in remaining_ids
        ]
        extra = _sample_without_replacement(rng, leftovers, n - len(selected))
        selected.extend(extra)
        if extra:
            warnings.append(f"topped up {len(extra)} from unstratified leftovers")

    n_query = sum(1 for p in selected if p.paper_role == "query")
    n_target = sum(1 for p in selected if p.paper_role == "target")

    stats = {
        "requested_n": n,
        "actual_n": len(selected),
        "requested_n_query": n // 2,
        "requested_n_target": n - (n // 2),
        "n_query": n_query,
        "n_target": n_target,
        "irrelevant_fraction_target": frac,
        "n_zero_target_total": cells.get(("query", "zero"), 0)
        + cells.get(("target", "zero"), 0),
        "n_zero": sum(1 for p in selected if p.relevance_grade == 0.0),
        "n_nonzero": sum(1 for p in selected if p.relevance_grade > 0.0),
        "bin_picked": bin_picked,
        "cell_picked": cell_picked,
        "pool_size": len(pool),
        "pool_query": len(_role_pool(pool, "query")),
        "pool_target": len(_role_pool(pool, "target")),
        "pool_zero": sum(1 for p in pool if p.relevance_grade == 0.0),
        "pool_nonzero": sum(1 for p in pool if p.relevance_grade > 0.0),
    }
    if warnings:
        stats["warning"] = "; ".join(warnings)
    return selected, stats


def _identifier_row_fields(rec: PaperRecord) -> Dict[str, str]:
    return {
        "text_path": rec.text_path,
        "papers_dir": rec.papers_dir,
        "gene_focus_symbol": rec.gene_focus_symbol,
        "gene_focus_common_name": rec.gene_focus_common_name,
        "gene_focus_search_terms": rec.gene_focus_search_terms,
        "query_search_terms": rec.query_search_terms,
        "target_search_terms": rec.target_search_terms,
    }


def write_outputs(
    selected: Sequence[PaperRecord],
    out_dir: Path,
    seed: int,
    stats: Dict[str, Any],
    *,
    per_criterion: bool = False,
    rubric_spec: Optional[RubricSpec] = None,
    include_identifiers: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(selected)
    rng = random.Random(seed + 1)
    rng.shuffle(ordered)

    answer_key: Dict[str, Any] = {}
    blind_rows: List[Dict[str, str]] = []
    fieldnames = blind_csv_fieldnames(
        per_criterion,
        rubric_spec,
        include_identifiers=include_identifiers,
    )

    for i, rec in enumerate(ordered, start=1):
        sample_id = f"audit_{i:03d}"
        if per_criterion and rubric_spec is not None:
            applicable = rubric_spec.criteria_for_role(rec.paper_role)
            row: Dict[str, str] = {
                "sample_id": sample_id,
                "doi": rec.doi,
                "gene_focus_id": rec.gene_focus_id,
                "paper_role": rec.paper_role,
                "alignment_id": rec.alignment_id,
                "query_gene_id": rec.query_gene_id,
                "target_gene_id": rec.target_gene_id,
                "file_name": rec.file_name,
                **_identifier_row_fields(rec),
                "applicable_criteria": ";".join(applicable),
                "human_notes": "",
            }
            for crit in rubric_spec.criteria:
                row[f"human_{crit.criterion_id}"] = ""
            blind_rows.append(row)
        else:
            row = {
                "sample_id": sample_id,
                "doi": rec.doi,
                "gene_focus_id": rec.gene_focus_id,
                "paper_role": rec.paper_role,
                "alignment_id": rec.alignment_id,
                "query_gene_id": rec.query_gene_id,
                "target_gene_id": rec.target_gene_id,
                "file_name": rec.file_name,
                "human_relevant_yes_no": "",
                "human_relevance_0_1": "",
                "human_notes": "",
            }
            if include_identifiers:
                row.update(_identifier_row_fields(rec))
            blind_rows.append(row)
        answer_key[sample_id] = {
            "doi": rec.doi,
            "gene_focus_id": rec.gene_focus_id,
            "alignment_id": rec.alignment_id,
            "query_gene_id": rec.query_gene_id,
            "target_gene_id": rec.target_gene_id,
            "paper_role": rec.paper_role,
            "file_name": rec.file_name,
            "papers_dir": rec.papers_dir,
            "text_path": rec.text_path,
            "gene_focus_symbol": rec.gene_focus_symbol,
            "gene_focus_common_name": rec.gene_focus_common_name,
            "gene_focus_search_terms": rec.gene_focus_search_terms,
            "query_search_terms": rec.query_search_terms,
            "target_search_terms": rec.target_search_terms,
            "gene_context": rec.gene_context,
            "relevance_grade": rec.relevance_grade,
            "rubric_dimension_scores": rec.rubric_dimension_scores,
            "rubric_axis_rationales": rec.rubric_axis_rationales,
            "rubric_tags": rec.rubric_tags,
            "criterion_scores": rec.criterion_scores,
            "axis_totals": rec.axis_totals,
            "paper_grade": rec.paper_grade,
            "primary_grade": rec.primary_grade,
            "rationale": rec.rationale,
            "graded_at": rec.grading_meta.get("graded_at"),
            "grader_model": rec.grading_meta.get("grader_model"),
            "grading_schema_version": rec.grading_meta.get("grading_schema_version"),
            "host_rubric_path": rec.grading_meta.get("host_rubric_path"),
            "microbe_rubric_path": rec.grading_meta.get("microbe_rubric_path"),
        }
        if per_criterion and rubric_spec is not None:
            answer_key[sample_id]["applicable_criteria"] = list(
                rubric_spec.criteria_for_role(rec.paper_role)
            )
            answer_key[sample_id]["rubric_spec"] = {
                "host_rubric_path": str(rubric_spec.host_path),
                "host_rubric_version": rubric_spec.host_version,
                "microbe_rubric_path": str(rubric_spec.microbe_path),
                "microbe_rubric_version": rubric_spec.microbe_version,
            }

    blind_path = out_dir / "blind_grading_sheet.csv"
    with blind_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = out_dir / "llm_grades_answer_key.json"
    key_path.write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if per_criterion and rubric_spec is not None:
        guide_lines = [
            "Per-criterion human scoring sheet",
            "Fill human_<criterion_id> with 0, 1, or 2 for each applicable criterion.",
            "Leave blank for criteria not listed in applicable_criteria for that row.",
            "Flag criteria (mimicry_potential_flag, novelty_flag) are not scored here.",
            "Open text_path and search for gene_focus_search_terms while grading.",
            "Run score_human_grading_sheet.py on the filled CSV to compute paper_grade",
            "and axis totals with the same math as the LLM grader.",
            "",
            f"host_rubric={rubric_spec.host_path} (v{rubric_spec.host_version})",
            f"microbe_rubric={rubric_spec.microbe_path} (v{rubric_spec.microbe_version})",
            "",
        ]
        for crit in rubric_spec.criteria:
            roles = ", ".join(crit.paper_roles)
            guide_lines.append(
                f"{crit.criterion_id}\troles={roles}\taxis={crit.axis_id}\tweight={crit.weight}"
            )
        (out_dir / "rubric_criterion_guide.txt").write_text(
            "\n".join(guide_lines) + "\n",
            encoding="utf-8",
        )

    report_lines = [
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"seed={seed}",
        f"requested_n={stats.get('requested_n')}",
        f"actual_n={stats.get('actual_n')}",
        f"pool_size={stats.get('pool_size')}",
        f"pool_query={stats.get('pool_query')}",
        f"pool_target={stats.get('pool_target')}",
        f"pool_zero={stats.get('pool_zero')}",
        f"pool_nonzero={stats.get('pool_nonzero')}",
        f"requested_n_query={stats.get('requested_n_query')}",
        f"requested_n_target={stats.get('requested_n_target')}",
        f"n_query_in_sample={stats.get('n_query')}",
        f"n_target_in_sample={stats.get('n_target')}",
        f"irrelevant_fraction_target={stats.get('irrelevant_fraction_target'):.4f}",
        f"n_zero_target_total={stats.get('n_zero_target_total')}",
        f"n_zero_in_sample={stats.get('n_zero')}",
        f"n_nonzero_in_sample={stats.get('n_nonzero')}",
        f"bin_picked_low={stats.get('bin_picked', {}).get('low', 0)}",
        f"bin_picked_mid={stats.get('bin_picked', {}).get('mid', 0)}",
        f"bin_picked_high={stats.get('bin_picked', {}).get('high', 0)}",
    ]
    if stats.get("warning"):
        report_lines.append(f"warning={stats['warning']}")
    (out_dir / "sampling_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {blind_path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {out_dir / 'sampling_report.txt'}")
    if per_criterion and rubric_spec is not None:
        print(f"Wrote {out_dir / 'rubric_criterion_guide.txt'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="llm_results directory with *_graded.json",
    )
    p.add_argument("--n", type=int, required=True, help="Number of papers to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: output-root/blind_grading_audit_TIMESTAMP)",
    )
    p.add_argument(
        "--papers-root",
        type=Path,
        default=None,
        help="Parent of per-alignment paper dirs if papers_dir missing from graded JSON",
    )
    p.add_argument("--irrelevant-min", type=float, default=0.10)
    p.add_argument("--irrelevant-max", type=float, default=0.30)
    p.add_argument(
        "--per-criterion",
        action="store_true",
        help="Blind CSV with human_<criterion_id> columns (0-2 integers; requires rubric paths)",
    )
    p.add_argument(
        "--per-axis",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--host-rubric",
        type=Path,
        default=None,
        help="Host/target-side rubric JSON (e.g. host_rubric_v1.json)",
    )
    p.add_argument(
        "--microbe-rubric",
        type=Path,
        default=None,
        help="Query/microbe-side rubric JSON (e.g. legionella_rubric.json)",
    )
    p.add_argument(
        "--idmap-csv",
        type=Path,
        default=None,
        help="Stage-1 idmap CSV with query/target gene identifiers (recommended)",
    )
    args = p.parse_args()

    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if args.irrelevant_min < 0 or args.irrelevant_max > 1 or args.irrelevant_min > args.irrelevant_max:
        print("Invalid irrelevant fraction range", file=sys.stderr)
        return 2
    if not args.output_root.is_dir():
        print(f"Not a directory: {args.output_root}", file=sys.stderr)
        return 2

    per_criterion = args.per_criterion or args.per_axis
    rubric_spec: Optional[RubricSpec] = None
    if per_criterion:
        try:
            host_rubric, microbe_rubric = resolve_rubric_paths(
                host=args.host_rubric,
                microbe=args.microbe_rubric,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        if not host_rubric.is_file() or not microbe_rubric.is_file():
            print("Rubric JSON path not found", file=sys.stderr)
            return 2
        rubric_spec = load_rubric_spec(host_rubric, microbe_rubric)

    idmap: Dict[str, Dict[str, Any]] = {}
    if args.idmap_csv:
        if not args.idmap_csv.is_file():
            print(f"Not found: {args.idmap_csv}", file=sys.stderr)
            return 2
        idmap = _load_idmap(str(args.idmap_csv))
        if not idmap:
            print(f"Warning: no rows loaded from {args.idmap_csv}", file=sys.stderr)

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = args.output_root / f"blind_grading_audit_{ts}"

    pool = load_candidate_pool(args.output_root, args.papers_root, idmap=idmap or None)
    if not pool:
        print("No graded papers found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    selected, stats = sample_records(
        pool,
        args.n,
        rng,
        args.irrelevant_min,
        args.irrelevant_max,
    )
    if not selected:
        print("No samples selected.", file=sys.stderr)
        return 1

    write_outputs(
        selected,
        out_dir,
        args.seed,
        stats,
        per_criterion=per_criterion,
        rubric_spec=rubric_spec,
        include_identifiers=bool(idmap),
    )
    if stats.get("warning"):
        print(f"Warning: {stats['warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
