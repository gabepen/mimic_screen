"""One-shot synthesis audit prompts (no pre-retrieved papers)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from auto_lit_search.paper_io import identification_terms_block
from auto_lit_search.synthesis_scorecard import quick_summary_prompt_footer

EVIDENCE_STRATA = ("strong", "mid", "weak")

STRONG_SCORE_MIN = 60
WEAK_SCORE_MAX = 39


def synthesis_pair_context_block() -> str:
    return (
        "Pair context (apply to all synthesis steps):\n"
        "- This query–target pair was selected for structural similarity; do not expect "
        "papers to report explicit mimicry or direct query–target interaction.\n"
        "- Semi-positive conclusions are appropriate when the target shows strong "
        "host-side rubric support (e.g. infection/symbiosis/interaction-relevant axes) "
        "and/or the query shows effector, secretion, persistence, or host-targeting "
        "evidence—even if genes are never named together.\n"
        "- Separate literature support for manipulation or symbiont-relevant potential "
        "from proof of mimicry.\n\n"
    )


def _system_context_block(
    *,
    pathogen_name: str,
    host_name: str,
    interaction_blurb: str,
) -> str:
    lines = [
        "System context:",
        f"- Pathogen (query/microbe side): {pathogen_name}",
        f"- Host (target side): {host_name}",
    ]
    if interaction_blurb.strip():
        lines.append(f"- Interaction context: {interaction_blurb.strip()}")
    lines.append(
        "- Host papers in the full pipeline are often generic cell-biology studies; "
        "relevance is inferred from functional overlap with pathogen exploitation, "
        "not necessarily infection wording."
    )
    return "\n".join(lines) + "\n\n"


def build_one_shot_audit_prompt(
    *,
    alignment_id: str,
    query_id: str,
    target_id: str,
    gene_context: Optional[Dict[str, Any]],
    pathogen_name: str = "Legionella pneumophila",
    host_name: str = "Homo sapiens",
    interaction_blurb: str = (
        "Obligate intracellular pathogen; Dot/Icm Type IV secretion system delivers "
        "effectors into host cells."
    ),
) -> str:
    """Literature-only one-shot prompt aligned with pipeline synthesis tone."""
    term_block = identification_terms_block(query_id, target_id, gene_context)
    return (
        "You are a biomedical research assistant performing a one-shot literature assessment.\n\n"
        "Task: The microbe (query) and host (target) proteins below were paired because they "
        "share detectable structural similarity. You do NOT have pre-retrieved paper texts. "
        "Using published literature you know (PubMed, reviews, canonical effector catalogs), "
        "determine whether there is literature support for the microbe gene product manipulating "
        "or exploiting host biology in this infection context.\n\n"
        f"{_system_context_block(pathogen_name=pathogen_name, host_name=host_name, interaction_blurb=interaction_blurb)}"
        f"{synthesis_pair_context_block()}"
        f"{term_block}\n"
        "Bridge host (target) exploitation evidence with query (microbe) effector evidence. "
        "Do not require co-mention of both genes in the same paper.\n\n"
        "Research questions:\n"
        f"1. Host protein ({target_id}): Is there literature that this host gene/product is "
        "exploited, regulated, or functionally relevant during Legionella infection or "
        "analogous macrophage/epithelial processes?\n"
        f"2. Microbe protein ({query_id}): Is there literature that this pathogen gene/product "
        "acts as a secreted effector, Dot/Icm substrate, or host-manipulation factor?\n"
        "3. Given structural similarity alone, how plausible is mimicry or functional convergence?\n\n"
        "Instruction: Write a running discussion (plain text, not JSON) that states which "
        "host-side and microbe-side patterns drive confidence or uncertainty. Weight host "
        "infection-process relevance and microbe secretion/effector biology even when direct "
        "pair evidence is absent. Cite supporting papers by DOI or PMID when known.\n\n"
        "Assign integer scores 0-100 (not categorical Some/High labels) for each dimension.\n"
        f"{quick_summary_prompt_footer()}"
        f"Alignment: {alignment_id}\n"
        f"Query (microbe)={query_id}\n"
        f"Target (host)={target_id}\n"
    )


def pair_priority_stratum(score: int) -> str:
    if score >= STRONG_SCORE_MIN:
        return "strong"
    if score <= WEAK_SCORE_MAX:
        return "weak"
    return "mid"
