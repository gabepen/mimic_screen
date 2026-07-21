"""One-shot synthesis audit prompts (brief research-LLM input, no papers)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from auto_lit_search.paper_io import gene_terms

EVIDENCE_STRATA = ("strong", "mid", "weak")

STRONG_SCORE_MIN = 60
WEAK_SCORE_MAX = 39


def pair_priority_stratum(score: int | float | None) -> str:
    """Bin pipeline pair_priority score into evidence strata for audit sampling."""
    if score is None:
        return "weak"
    s = float(score)
    if s >= STRONG_SCORE_MIN:
        return "strong"
    if s <= WEAK_SCORE_MAX:
        return "weak"
    return "mid"


def brief_system_preamble(query_organism: str, target_organism: str) -> str:
    """One-line organism framing for an external research LLM."""
    query_org = str(query_organism or "").strip()
    target_org = str(target_organism or "").strip()
    if not query_org or not target_org:
        raise ValueError("query_organism and target_organism are required")
    return (
        f"Microbe: {query_org}. Host: {target_org}. "
        "These two proteins were paired for structural similarity.\n\n"
    )


def _display_name(terms: Dict[str, Any], uniprot_id: str) -> str:
    """Prefer gene symbol / common name from idmap; never echo the UniProt accession."""
    symbol = str(terms.get("symbol") or "").strip()
    common = str(terms.get("common_name") or "").strip()
    if common.lower() in {"", "none"}:
        common = ""
    if symbol and symbol == uniprot_id:
        symbol = ""
    if symbol and common and symbol.lower() != common.lower():
        return f"{symbol} / {common}"
    if common:
        return common
    if symbol:
        return symbol
    return "unknown"


def build_oneshot_synthesis_audit_prompt(
    *,
    query: str,
    target_id: str,
    query_organism: str,
    target_organism: str,
    gene_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Brief research-LLM question with organism + UniProt / gene names only."""
    q_meta = (gene_context or {}).get("query") or {}
    t_meta = (gene_context or {}).get("target") or {}
    q_terms = gene_terms(q_meta if isinstance(q_meta, dict) else {}, query)
    t_terms = gene_terms(t_meta if isinstance(t_meta, dict) else {}, target_id)

    q_name = _display_name(q_terms, query)
    t_name = _display_name(t_terms, target_id)

    return (
        brief_system_preamble(query_organism, target_organism)
        + f"Is there literature evidence that this microbe protein "
        f"(UniProt ID: {query}; common gene name: {q_name}) could be an effector of "
        f"this host protein (UniProt ID: {target_id}; common gene name: {t_name}) "
        f"through structural similarity or mimicry?\n"
    )
