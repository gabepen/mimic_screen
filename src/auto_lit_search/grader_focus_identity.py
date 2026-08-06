"""Focus-gene identity gate for grader outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from auto_lit_search.rubric_scoring import (
    CLAIM_SUMMARY_MAX_CHARS,
    aggregate_paper_scores,
    required_scored_criterion_ids,
)


def text_mentions_focus_term(text: str, terms: Sequence[str]) -> bool:
    """True if any focus identification term appears in text."""
    if not text or not terms:
        return False
    lower = text.lower()
    for term in terms:
        t = str(term or "").strip()
        if not t or t.lower() == "none":
            continue
        if len(t) <= 3:
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])",
                text,
                flags=re.IGNORECASE,
            ):
                return True
        elif t.lower() in lower:
            return True
    return False


def focus_identity_prompt_block(focus_terms: Sequence[str], gene_focus_label: str) -> str:
    term_list = ", ".join(focus_terms) if focus_terms else "(no aliases provided)"
    return (
        "FOCUS-GENE IDENTITY GATE (do this before scoring):\n"
        f"- This scorecard is ONLY about {gene_focus_label}.\n"
        f"- Focus identification terms: {term_list}\n"
        "- If the excerpt does not discuss this focus gene (id/symbol/alias above), "
        "set no_meaningful_mention=true, mention_type=incidental_mention, "
        "claim_summary exactly NO_FOCUS_MENTION, and score EVERY criterion 0.\n"
        "- Do NOT transfer evidence from homologs, paralogs, family members, or "
        "similarly named genes (e.g. LegK2/LegK7 for legK1; human ASL for bacterial "
        "argH; a paper's protagonist gene that is not the focus).\n"
        "- claim_summary MUST begin with the focus gene id or symbol when evidence "
        "exists; otherwise use NO_FOCUS_MENTION.\n"
    )


def enforce_focus_identity(
    parsed: Dict[str, Any],
    *,
    excerpt: str,
    focus_terms: Sequence[str],
    rubric: Dict[str, Any],
    rubric_role: str,
) -> Dict[str, Any]:
    """Zero grades when the excerpt/claim is not about the focus gene."""
    required = required_scored_criterion_ids(rubric)
    claim = str(parsed.get("claim_summary") or "").strip()
    claim_upper = claim.upper()
    force = False
    reason = ""
    if claim_upper.startswith("NO_FOCUS_MENTION"):
        force = True
        reason = "claim_summary=NO_FOCUS_MENTION"
    elif focus_terms and not text_mentions_focus_term(excerpt, focus_terms):
        force = True
        reason = "focus_gene_absent_from_excerpt"
    elif (
        claim
        and focus_terms
        and not claim_upper.startswith("NO_FOCUS_MENTION")
        and not text_mentions_focus_term(claim, focus_terms)
    ):
        force = True
        reason = "claim_summary_omits_focus_gene"

    if not force and not bool(parsed.get("no_meaningful_mention")):
        return parsed

    if not force and bool(parsed.get("no_meaningful_mention")):
        reason = reason or "model_no_meaningful_mention"

    criterion_scores = {
        crit_id: {"score": 0, "note": ""}
        for crit_id in required
    }
    agg = aggregate_paper_scores(
        rubric, criterion_scores, rubric_role=rubric_role
    )
    new_claim = claim if claim_upper.startswith("NO_FOCUS_MENTION") else (
        claim or "NO_FOCUS_MENTION"
    )
    if force and not claim_upper.startswith("NO_FOCUS_MENTION"):
        new_claim = "NO_FOCUS_MENTION"
    rationale = (
        f"[focus identity gate: {reason}] "
        + str(parsed.get("rationale") or claim or "")
    ).strip()[:CLAIM_SUMMARY_MAX_CHARS]
    return {
        **parsed,
        **agg,
        "criterion_scores": criterion_scores,
        "no_meaningful_mention": True,
        "mention_type": "incidental_mention",
        "claim_summary": new_claim[:CLAIM_SUMMARY_MAX_CHARS],
        "rationale": rationale,
        "rubric_tags": {},
    }
