"""Deterministic rubric aggregation from criterion-level 0/1/2 scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

GRADING_SCHEMA_VERSION = 2

HOST_PRIMARY_AXIS = "infection_process_relevance"
MICROBE_PRIMARY_AXIS = "system_relevance"
HOST_BONUS_AXIS = "disease_population_relevance"

_CRITERION_NOTE_MAX_CHARS = 60


@dataclass(frozen=True)
class CriterionSpec:
    id: str
    axis_id: str
    weight: str


@dataclass(frozen=True)
class AxisTotal:
    score: int
    max_score: int

    @property
    def label(self) -> str:
        return f"{self.score}/{self.max_score}"

    @property
    def norm(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return self.score / self.max_score


def weight_multiplier(weight: str) -> int:
    w = (weight or "medium").lower()
    if w == "flag":
        return 0
    return 2 if w == "high" else 1


def rubric_criteria(rubric: Dict[str, Any]) -> Tuple[List[CriterionSpec], List[CriterionSpec]]:
    scored: List[CriterionSpec] = []
    flags: List[CriterionSpec] = []
    for ax in rubric.get("axes") or []:
        axis_id = str(ax.get("id") or "").strip()
        if not axis_id:
            continue
        for crit in ax.get("criteria") or []:
            if not isinstance(crit, dict):
                continue
            crit_id = str(crit.get("id") or "").strip()
            if not crit_id:
                continue
            weight = str(crit.get("weight") or "medium").lower()
            spec = CriterionSpec(id=crit_id, axis_id=axis_id, weight=weight)
            if weight == "flag":
                flags.append(spec)
            else:
                scored.append(spec)
    return scored, flags


def required_scored_criterion_ids(rubric: Dict[str, Any]) -> List[str]:
    scored, _ = rubric_criteria(rubric)
    return [c.id for c in scored]


def required_flag_ids(rubric: Dict[str, Any]) -> List[str]:
    _, flags = rubric_criteria(rubric)
    return [c.id for c in flags]


def rubric_role_for_paper_role(paper_role: str) -> str:
    return "microbe" if (paper_role or "").strip().lower() == "query" else "host"


def primary_axis_for_role(rubric_role: str) -> str:
    return HOST_PRIMARY_AXIS if rubric_role == "host" else MICROBE_PRIMARY_AXIS


def criterion_id_to_axis_map(rubric: Dict[str, Any]) -> Dict[str, str]:
    scored, _ = rubric_criteria(rubric)
    return {c.id: c.axis_id for c in scored}


def derive_axis_rationales_from_criterion_scores(
    rubric: Dict[str, Any],
    criterion_scores: Dict[str, Dict[str, Any]],
    *,
    max_axis_chars: int = 700,
) -> Dict[str, str]:
    """Roll v2 per-criterion notes into per-axis strings for synthesis prompts."""
    crit_to_axis = criterion_id_to_axis_map(rubric)
    by_axis: Dict[str, List[str]] = {}
    for crit_id in sorted(crit_to_axis.keys()):
        axis_id = crit_to_axis[crit_id]
        entry = criterion_scores.get(crit_id)
        if entry is None:
            continue
        score = entry.get("score", "?")
        note = str(entry.get("note") or "").strip()
        if note:
            part = f"{crit_id}={score}: {note}"
        else:
            part = f"{crit_id}={score}"
        by_axis.setdefault(axis_id, []).append(part)
    return {
        axis_id: "; ".join(parts)[:max_axis_chars]
        for axis_id, parts in by_axis.items()
        if parts
    }


def resolve_axis_rationales(
    rubric: Dict[str, Any] | None,
    criterion_scores: Dict[str, Dict[str, Any]],
    existing: Dict[str, str] | None,
    *,
    max_axis_chars: int = 700,
) -> Dict[str, str]:
    if existing:
        out = {str(k): str(v) for k, v in existing.items() if str(v).strip()}
        if out:
            return out
    if not rubric or not criterion_scores:
        return {}
    return derive_axis_rationales_from_criterion_scores(
        rubric, criterion_scores, max_axis_chars=max_axis_chars
    )


def normalize_criterion_scores(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize LLM output to {id: {score: int, note: str}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        crit_id = str(key).strip()
        if not crit_id:
            continue
        if isinstance(value, dict):
            score_raw = value.get("score")
            note = str(value.get("note") or "").strip()[:_CRITERION_NOTE_MAX_CHARS]
        else:
            score_raw = value
            note = ""
        try:
            score = int(score_raw)
        except (TypeError, ValueError):
            continue
        out[crit_id] = {"score": max(0, min(2, score)), "note": note}
    return out


def compute_axis_totals(
    rubric: Dict[str, Any], criterion_scores: Dict[str, Dict[str, Any]]
) -> Dict[str, AxisTotal]:
    scored, _ = rubric_criteria(rubric)
    by_axis: Dict[str, List[CriterionSpec]] = {}
    for crit in scored:
        by_axis.setdefault(crit.axis_id, []).append(crit)

    totals: Dict[str, AxisTotal] = {}
    for axis_id, crits in by_axis.items():
        score = 0
        max_score = 0
        for crit in crits:
            mult = weight_multiplier(crit.weight)
            max_score += 2 * mult
            raw = criterion_scores.get(crit.id) or {}
            crit_score = int(raw.get("score") or 0)
            score += crit_score * mult
        totals[axis_id] = AxisTotal(score=score, max_score=max_score)
    return totals


def compute_relevance_grade(dim_scores: Dict[str, float], rubric_role: str) -> float:
    if not dim_scores:
        return 0.0
    if (
        rubric_role == "host"
        and HOST_BONUS_AXIS in dim_scores
        and len(dim_scores) > 1
    ):
        base_keys = [k for k in dim_scores if k != HOST_BONUS_AXIS]
        base = sum(float(dim_scores[k]) for k in base_keys) / len(base_keys)
        bonus = float(dim_scores[HOST_BONUS_AXIS]) / len(dim_scores)
        return max(0.0, min(1.0, base + bonus))
    return max(0.0, min(1.0, sum(float(v) for v in dim_scores.values()) / len(dim_scores)))


def aggregate_paper_scores(
    rubric: Dict[str, Any],
    criterion_scores: Dict[str, Dict[str, Any]],
    *,
    rubric_role: str,
) -> Dict[str, Any]:
    axis_totals = compute_axis_totals(rubric, criterion_scores)
    total_score = sum(t.score for t in axis_totals.values())
    total_max = sum(t.max_score for t in axis_totals.values())

    primary_axis = primary_axis_for_role(rubric_role)
    primary = axis_totals.get(primary_axis, AxisTotal(score=0, max_score=0))
    dim_scores = {axis_id: total.norm for axis_id, total in axis_totals.items()}

    return {
        "grading_schema_version": GRADING_SCHEMA_VERSION,
        "axis_totals": {
            axis_id: {"score": total.score, "max": total.max_score, "label": total.label}
            for axis_id, total in axis_totals.items()
        },
        "paper_grade": f"{total_score}/{total_max}",
        "paper_grade_score": total_score,
        "paper_grade_max": total_max,
        "primary_grade": primary.label,
        "primary_grade_score": primary.score,
        "primary_grade_max": primary.max_score,
        "relevance_sort": primary.score,
        "rubric_dimension_scores": dim_scores,
        "relevance_grade": compute_relevance_grade(dim_scores, rubric_role),
        "rubric_axis_rationales": derive_axis_rationales_from_criterion_scores(
            rubric, criterion_scores
        ),
    }


def criteria_prompt_block(rubric: Dict[str, Any]) -> str:
    lines: List[str] = []
    for ax in rubric.get("axes") or []:
        axis_id = str(ax.get("id") or "").strip()
        label = str(ax.get("label") or axis_id).strip()
        lines.append(f"Axis {axis_id} ({label}):")
        for crit in ax.get("criteria") or []:
            if not isinstance(crit, dict):
                continue
            crit_id = str(crit.get("id") or "").strip()
            weight = str(crit.get("weight") or "medium").lower()
            if not crit_id:
                continue
            if weight == "flag":
                lines.append(f"  - {crit_id} [flag → rubric_tags, not scored]")
            else:
                lines.append(f"  - {crit_id} [weight={weight}]")
    return "\n".join(lines)
