"""Rubric-derived indices and structured synthesis reporting (no legacy qualifiers)."""

from __future__ import annotations

import json
import re
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from auto_lit_search.env_config import env_positive_float

# Pair priority tiers (A = strongest literature support for review)
PAIR_TIERS = ("A", "B", "C", "D", "E")
# Per-dimension evidence tiers
DIMENSION_TIERS = ("Strong", "Moderate", "Limited", "Weak", "None")

HOST_EXPLOITATION_AXIS = "infection_process_relevance"
HOST_CHAR_AXIS = "protein_characterisation_quality"
QUERY_EVIDENCE_AXES = ("evidence_quality", "system_relevance")
MIMICRY_TAG_KEYS = ("mimicry_potential_flag", "mimicry_flag")
MIMICRY_STRONG_VALUES = frozenset(
    {"mimicry_strong", "strong", "mimicry_possible", "possible"}
)


def _axis_score(gp: Any, axis: str) -> float:
    scores = getattr(gp, "rubric_dimension_scores", None) or {}
    try:
        return float(scores.get(axis, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _max_axis_score(gp: Any) -> float:
    scores = getattr(gp, "rubric_dimension_scores", None) or {}
    if not scores:
        return 0.0
    return max(float(v) for v in scores.values())


def _role(gp: Any) -> str:
    return (getattr(gp, "paper_role", None) or "").strip().lower()


def _relevance_grade(gp: Any) -> float:
    try:
        return float(getattr(gp, "relevance_grade", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _paper_id(gp: Any) -> str:
    return str(getattr(gp, "paper_id", None) or getattr(gp, "file_name", "") or "")


def graded_papers_from_json(graded: Dict[str, Any]) -> List[Any]:
    """Load graded rows for scorecard without pydantic."""
    out: List[Any] = []
    for row in graded.get("graded_papers") or []:
        if not isinstance(row, dict):
            continue
        tags = row.get("rubric_tags")
        if not isinstance(tags, dict):
            tags = {}
        out.append(
            SimpleNamespace(
                paper_id=str(row.get("paper_id") or ""),
                file_name=str(row.get("file_name") or ""),
                paper_role=row.get("paper_role"),
                relevance_grade=float(row.get("relevance_grade") or 0.0),
                rubric_dimension_scores={
                    str(k): float(v)
                    for k, v in (row.get("rubric_dimension_scores") or {}).items()
                },
                rubric_axis_rationales={
                    str(k): str(v)
                    for k, v in (row.get("rubric_axis_rationales") or {}).items()
                },
                rationale=str(row.get("rationale") or ""),
                rubric_tags={str(k): str(v) for k, v in tags.items()},
            )
        )
    return out


def score_to_dimension_tier(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Limited"
    if score >= 20:
        return "Weak"
    return "None"


def score_to_pair_tier(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "E"


def _clamp_score(x: float) -> int:
    return max(0, min(100, int(round(x))))


def _dim_entry(score: int, source: str = "rubric") -> Dict[str, Any]:
    return {
        "score": score,
        "tier": score_to_dimension_tier(score),
        "source": source,
    }


def _top_papers_by_role(
    papers: List[Any],
    role: str,
    key_fn: Any,
    n: int = 3,
) -> List[Any]:
    pool = [gp for gp in papers if _role(gp) == role]
    return sorted(pool, key=key_fn, reverse=True)[:n]


def compute_rubric_scorecard(graded_papers: List[Any]) -> Dict[str, Any]:
    """Deterministic evidence indices from grader outputs."""
    host = [gp for gp in graded_papers if _role(gp) == "target"]
    query = [gp for gp in graded_papers if _role(gp) == "query"]

    host_exploit_scores = [
        _axis_score(gp, HOST_EXPLOITATION_AXIS)
        for gp in host
        if _relevance_grade(gp) > 0 or _axis_score(gp, HOST_EXPLOITATION_AXIS) > 0
    ]
    host_char_scores = [_axis_score(gp, HOST_CHAR_AXIS) for gp in host if _relevance_grade(gp) > 0]
    query_scores: List[float] = []
    for gp in query:
        if _relevance_grade(gp) <= 0:
            continue
        query_scores.append(
            max(_axis_score(gp, a) for a in QUERY_EVIDENCE_AXES)
            if any(a in (gp.rubric_dimension_scores or {}) for a in QUERY_EVIDENCE_AXES)
            else _max_axis_score(gp)
        )

    def _index_from_scores(vals: List[float], default: float = 0.0) -> int:
        if not vals:
            return 0
        # Blend max (peak evidence) and mean (breadth)
        peak = max(vals)
        avg = mean(vals)
        return _clamp_score(100.0 * (0.65 * peak + 0.35 * avg))

    host_exploitation_score = _index_from_scores(host_exploit_scores)
    host_char_score = _index_from_scores(host_char_scores)
    query_effector_score = _index_from_scores(query_scores)

    mimicry_score = 0
    for gp in host:
        tags = getattr(gp, "rubric_tags", None) or {}
        if not isinstance(tags, dict):
            continue
        for key in MIMICRY_TAG_KEYS:
            val = str(tags.get(key) or "").strip().lower()
            if val in ("mimicry_strong", "strong"):
                mimicry_score = max(mimicry_score, 85)
            elif val in ("mimicry_possible", "possible"):
                mimicry_score = max(mimicry_score, 55)
        if mimicry_score == 0 and _axis_score(gp, HOST_CHAR_AXIS) >= 0.5:
            mimicry_score = max(mimicry_score, 35)

    w_host = env_positive_float("SCORECARD_WEIGHT_HOST", 0.40)
    w_query = env_positive_float("SCORECARD_WEIGHT_QUERY", 0.35)
    w_mimicry = env_positive_float("SCORECARD_WEIGHT_MIMICRY", 0.15)
    w_depth = env_positive_float("SCORECARD_WEIGHT_DEPTH", 0.10)
    n_host_nz = sum(1 for gp in host if _relevance_grade(gp) > 0)
    n_query_nz = sum(1 for gp in query if _relevance_grade(gp) > 0)
    depth_bonus = min(25, 5 * (n_host_nz ** 0.5) + 5 * (n_query_nz ** 0.5))

    pair_raw = (
        w_host * host_exploitation_score
        + w_query * query_effector_score
        + w_mimicry * mimicry_score
        + w_depth * depth_bonus
    )
    pair_priority_score = _clamp_score(pair_raw)

    best_host = _top_papers_by_role(
        graded_papers,
        "target",
        lambda g: (_axis_score(g, HOST_EXPLOITATION_AXIS), _relevance_grade(g)),
    )
    best_query = _top_papers_by_role(
        graded_papers,
        "query",
        lambda g: (
            max(_axis_score(g, a) for a in QUERY_EVIDENCE_AXES),
            _relevance_grade(g),
        ),
    )

    def _best_paper_id(pool: List[Any]) -> str:
        if not pool:
            return ""
        return _paper_id(pool[0])

    top_host_ax = ""
    top_host_ax_score = 0.0
    if best_host:
        gp0 = best_host[0]
        ax_scores = gp0.rubric_dimension_scores or {}
        if ax_scores:
            top_host_ax, top_host_ax_score = max(
                ((k, float(v)) for k, v in ax_scores.items()),
                key=lambda kv: kv[1],
            )

    return {
        "host_exploitation": _dim_entry(host_exploitation_score),
        "host_characterisation": _dim_entry(host_char_score),
        "query_effector": _dim_entry(query_effector_score),
        "mimicry_plausibility": _dim_entry(mimicry_score),
        "pair_priority": {
            "score": pair_priority_score,
            "tier": score_to_pair_tier(pair_priority_score),
            "source": "rubric",
        },
        "evidence": {
            "n_host_papers": len(host),
            "n_query_papers": len(query),
            "n_host_nonzero": n_host_nz,
            "n_query_nonzero": n_query_nz,
            "n_host_ge_0_5": sum(1 for gp in host if _relevance_grade(gp) >= 0.5),
            "n_query_ge_0_5": sum(1 for gp in query if _relevance_grade(gp) >= 0.5),
            "max_host_relevance_grade": max((_relevance_grade(gp) for gp in host), default=0.0),
            "max_query_relevance_grade": max((_relevance_grade(gp) for gp in query), default=0.0),
            "top_host_axis": top_host_ax,
            "top_host_axis_score": round(top_host_ax_score, 3),
        },
        "best_host_paper": _best_paper_id(best_host),
        "best_query_paper": _best_paper_id(best_query),
    }


def _parse_score_line(text: str, key: str) -> Optional[int]:
    m = re.search(rf"{re.escape(key)}\s*:\s*(\d{{1,3}})", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return _clamp_score(float(m.group(1)))
    except ValueError:
        return None


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    for pattern in (
        r"```json\s*(\{.*?\})\s*```",
        r"Quick results summary:\s*(\{.*\})\s*(?:\n|$)",
    ):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_llm_scorecard(synthesis_text: str) -> Dict[str, Any]:
    """Parse LLM Quick results summary into partial scorecard fields."""
    text = synthesis_text or ""
    out: Dict[str, Any] = {}
    obj = _extract_json_block(text)
    if obj:
        out["headline"] = str(obj.get("headline") or "").strip()
        for key in (
            "host_exploitation_score",
            "query_effector_score",
            "mimicry_plausibility_score",
            "pair_priority_score",
        ):
            if key in obj:
                try:
                    out[key] = _clamp_score(float(obj[key]))
                except (TypeError, ValueError):
                    pass
        out["best_host_paper"] = str(obj.get("best_host_paper") or "").strip()
        out["best_query_paper"] = str(obj.get("best_query_paper") or "").strip()
        out["main_uncertainties"] = str(
            obj.get("main_uncertainties") or obj.get("main_conflicts_uncertainties") or ""
        ).strip()
        return out

    out["headline"] = ""
    m = re.search(r"headline:\s*(.+)", text, re.IGNORECASE)
    if m:
        out["headline"] = m.group(1).strip()

    for key in (
        "host_exploitation_score",
        "query_effector_score",
        "mimicry_plausibility_score",
        "pair_priority_score",
    ):
        v = _parse_score_line(text, key)
        if v is not None:
            out[key] = v

    m = re.search(r"best_host_paper:\s*(.+)", text, re.IGNORECASE)
    if m:
        out["best_host_paper"] = m.group(1).strip()
    m = re.search(r"best_query_paper:\s*(.+)", text, re.IGNORECASE)
    if m:
        out["best_query_paper"] = m.group(1).strip()
    m = re.search(
        r"main_uncertainties:\s*(.+)|Main conflicts / uncertainties:\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if m:
        out["main_uncertainties"] = (m.group(1) or m.group(2) or "").strip()
    return out


def _blend_scores(rubric: int, llm: Optional[int], rubric_weight: float = 0.6) -> int:
    if llm is None:
        return rubric
    w = max(0.0, min(1.0, rubric_weight))
    return _clamp_score(w * rubric + (1.0 - w) * llm)


def build_conclusion(
    graded_papers: List[Any],
    synthesis_text: str,
    *,
    synthesis_status: str,
    rubric_weight: Optional[float] = None,
) -> Dict[str, Any]:
    rubric = compute_rubric_scorecard(graded_papers)
    llm = parse_llm_scorecard(synthesis_text) if synthesis_status == "ok" else {}
    rw = rubric_weight if rubric_weight is not None else env_positive_float(
        "SCORECARD_RUBRIC_BLEND_WEIGHT", 0.6
    )

    host_score = _blend_scores(
        rubric["host_exploitation"]["score"],
        llm.get("host_exploitation_score"),
        rw,
    )
    query_score = _blend_scores(
        rubric["query_effector"]["score"],
        llm.get("query_effector_score"),
        rw,
    )
    mimicry_score = _blend_scores(
        rubric["mimicry_plausibility"]["score"],
        llm.get("mimicry_plausibility_score"),
        rw,
    )
    pair_score = _blend_scores(
        rubric["pair_priority"]["score"],
        llm.get("pair_priority_score"),
        rw,
    )

    headline = str(llm.get("headline") or "").strip()
    if not headline:
        headline = _auto_headline(host_score, query_score, mimicry_score, pair_score)

    best_host = str(llm.get("best_host_paper") or rubric.get("best_host_paper") or "")
    best_query = str(llm.get("best_query_paper") or rubric.get("best_query_paper") or "")
    uncertainties = str(llm.get("main_uncertainties") or "").strip()
    if synthesis_status == "grades_only" and not uncertainties:
        uncertainties = "Conclusion derived from rubric grades only (synthesis LLM unavailable)."

    return {
        "scorecard_version": "2",
        "synthesis_status": synthesis_status,
        "headline": headline,
        "host_exploitation": {
            "score": host_score,
            "tier": score_to_dimension_tier(host_score),
        },
        "query_effector": {
            "score": query_score,
            "tier": score_to_dimension_tier(query_score),
        },
        "mimicry_plausibility": {
            "score": mimicry_score,
            "tier": score_to_dimension_tier(mimicry_score),
        },
        "pair_priority": {
            "score": pair_score,
            "tier": score_to_pair_tier(pair_score),
        },
        "best_host_paper": best_host,
        "best_query_paper": best_query,
        "main_uncertainties": uncertainties,
        "rubric_indices": rubric,
        "evidence": rubric.get("evidence") or {},
    }


def _auto_headline(host: int, query: int, mimicry: int, pair: int) -> str:
    tier = score_to_pair_tier(pair)
    return (
        f"Pair priority {tier} ({pair}/100): host exploitation {host}/100, "
        f"query effector {query}/100, mimicry plausibility {mimicry}/100."
    )


def synthesis_output_well_formed(synthesis_text: str) -> bool:
    text = (synthesis_text or "").strip()
    if not text or "Quick results summary:" not in text:
        return False
    llm = parse_llm_scorecard(synthesis_text)
    return (
        llm.get("host_exploitation_score") is not None
        and llm.get("query_effector_score") is not None
        and bool(str(llm.get("headline") or "").strip())
    )


def quick_summary_prompt_footer() -> str:
    return (
        "END WITH THIS EXACT SECTION HEADER AND A JSON OBJECT (no other keys):\n\n"
        "Quick results summary:\n"
        "```json\n"
        "{\n"
        '  "headline": "<=25 words: host + query evidence in plain language>",\n'
        '  "host_exploitation_score": <0-100 integer>,\n'
        '  "query_effector_score": <0-100 integer>,\n'
        '  "mimicry_plausibility_score": <0-100 integer>,\n'
        '  "pair_priority_score": <0-100 integer; overall review priority>,\n'
        '  "best_host_paper": "<paper_id>",\n'
        '  "best_query_paper": "<paper_id or empty>",\n'
        '  "main_uncertainties": "<1-3 sentences>"\n'
        "}\n"
        "```\n\n"
        "Score guide: 0=no support, 40=limited/indirect, 60=moderate pathway overlap, "
        "80=strong direct evidence, 100=exceptional/multiple papers. "
        "pair_priority weights host exploitation, query effector biology, mimicry tags, "
        "and evidence depth. Do not use categorical labels like Some/High.\n\n"
    )


def quick_summary_retry_suffix() -> str:
    return (
        "\n\nYour previous answer lacked a parseable Quick results summary JSON block. "
        "Reply with plain text discussion, then end with the exact header and JSON schema "
        "shown in the instructions.\n"
    )


def format_fallback_discussion(
    alignment_id: str,
    n_graded: int,
    conclusion: Dict[str, Any],
) -> str:
    pp = conclusion["pair_priority"]
    return (
        "Synthesis used rubric-derived scorecard only (LLM output missing or invalid).\n"
        f"Alignment {alignment_id}: {n_graded} graded papers.\n\n"
        f"Scorecard: pair_priority={pp['score']} ({pp['tier']}); "
        f"host_exploitation={conclusion['host_exploitation']['score']}; "
        f"query_effector={conclusion['query_effector']['score']}; "
        f"mimicry_plausibility={conclusion['mimicry_plausibility']['score']}.\n\n"
        f"Quick results summary:\n```json\n{json.dumps(_quick_json_from_conclusion(conclusion), indent=2)}\n```\n"
    )


def _quick_json_from_conclusion(conclusion: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "headline": conclusion.get("headline") or "",
        "host_exploitation_score": conclusion["host_exploitation"]["score"],
        "query_effector_score": conclusion["query_effector"]["score"],
        "mimicry_plausibility_score": conclusion["mimicry_plausibility"]["score"],
        "pair_priority_score": conclusion["pair_priority"]["score"],
        "best_host_paper": conclusion.get("best_host_paper") or "",
        "best_query_paper": conclusion.get("best_query_paper") or "",
        "main_uncertainties": conclusion.get("main_uncertainties") or "",
    }
