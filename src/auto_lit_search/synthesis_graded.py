"""Graded-paper synthesis (v2): selective excerpts, top-K per role, slim analysis artifacts."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from auto_lit_search.analysis_packet import (
    GradedPaper,
    RunAlignmentGradedRequest,
    RunAlignmentResponse,
)
from auto_lit_search.env_config import env_positive_float, env_positive_int
from auto_lit_search.paper_io import (
    ensure_dir,
    identification_terms_block,
    read_text,
)
from auto_lit_search.rubric_scoring import (
    resolve_axis_rationales,
    rubric_role_for_paper_role,
)
from auto_lit_search.synthesis_scorecard import (
    build_conclusion,
    format_fallback_discussion,
    quick_summary_prompt_footer,
    quick_summary_retry_suffix,
    synthesis_output_well_formed,
)


def max_axis_score(gp: Any) -> float:
    scores = getattr(gp, "rubric_dimension_scores", None) or {}
    if not scores:
        return 0.0
    return max(float(v) for v in scores.values())


def paper_kept_for_synthesis(gp: Any, min_axis_score: float) -> bool:
    if float(gp.relevance_grade) > 0.0:
        return True
    return max_axis_score(gp) >= min_axis_score


def _role_key(gp: Any) -> str:
    role = (getattr(gp, "paper_role", None) or "").strip().lower()
    return role if role in {"query", "target"} else "other"


def _sort_papers_by_relevance(papers: List[Any]) -> List[Any]:
    return sorted(
        papers,
        key=lambda g: (-float(g.relevance_grade), -max_axis_score(g), g.file_name),
    )


def _select_top_k_per_role(
    kept: List[Any],
    top_k_host: int,
    top_k_query: int,
    *,
    excerpt_k_host: int | None = None,
    excerpt_k_query: int | None = None,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Return (batch_pool, excerpt_pool, host_pool, query_pool).

    batch_pool is top-K host + top-K query (role-sorted), not all kept papers.
    """
    by_role: Dict[str, List[Any]] = {"target": [], "query": [], "other": []}
    for gp in kept:
        by_role.setdefault(_role_key(gp), []).append(gp)

    host_pool = _sort_papers_by_relevance(by_role.get("target") or [])[:top_k_host]
    query_pool = _sort_papers_by_relevance(by_role.get("query") or [])[:top_k_query]
    other = _sort_papers_by_relevance(by_role.get("other") or [])
    other_cap = max(0, min(5, top_k_host // 5))
    batch_pool = host_pool + query_pool + other[:other_cap]

    exh_host = top_k_host if excerpt_k_host is None else excerpt_k_host
    exh_query = top_k_query if excerpt_k_query is None else excerpt_k_query
    excerpt_pool = host_pool[:exh_host] + query_pool[:exh_query]
    if other_cap:
        excerpt_pool.extend(other[: min(other_cap, 3)])

    return batch_pool, excerpt_pool, host_pool, query_pool


def _track_label(role: str) -> str:
    if role == "target":
        return "host"
    if role == "query":
        return "query"
    return "other"


def _tags_line(gp: Any) -> str:
    tags = getattr(gp, "rubric_tags", None) or {}
    if not tags:
        return ""
    bits = [f"{k}={v}" for k, v in sorted(tags.items()) if str(v).strip()]
    return f"    • rubric_tags: {', '.join(bits)}\n" if bits else ""


def _load_rubric_json(path: str) -> Dict[str, Any] | None:
    path = (path or "").strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _rubric_for_paper(
    gp: Any,
    host_rubric: Dict[str, Any] | None,
    microbe_rubric: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    role = rubric_role_for_paper_role(getattr(gp, "paper_role", None) or "")
    return microbe_rubric if role == "microbe" else host_rubric


def _axis_rationales_for_paper(
    gp: Any,
    host_rubric: Dict[str, Any] | None,
    microbe_rubric: Dict[str, Any] | None,
) -> Dict[str, str]:
    existing = getattr(gp, "rubric_axis_rationales", None) or {}
    criterion_scores = getattr(gp, "criterion_scores", None) or {}
    rubric = _rubric_for_paper(gp, host_rubric, microbe_rubric)
    return resolve_axis_rationales(
        rubric,
        criterion_scores if isinstance(criterion_scores, dict) else {},
        existing if isinstance(existing, dict) else {},
    )


def _paper_metadata_lines(
    gp: Any,
    per_axis_cap: int,
    host_rubric: Dict[str, Any] | None = None,
    microbe_rubric: Dict[str, Any] | None = None,
) -> str:
    scores = gp.rubric_dimension_scores or {}
    rax = _axis_rationales_for_paper(gp, host_rubric, microbe_rubric)
    order = sorted(scores.keys())
    parts: List[str] = [
        f"- {gp.file_name} role={gp.paper_role or 'unknown'} "
        f"paper_id={gp.paper_id} aggregate_axis_score={gp.relevance_grade:.3f}"
    ]
    parts.append(_tags_line(gp).rstrip())
    for ax in order:
        sc = float(scores.get(ax, 0.0))
        why = (rax.get(ax) or "")[:per_axis_cap]
        parts.append(f"    • {ax}: score={sc:.3f} | grader_reasoning: {why}")
    if (gp.rationale or "").strip():
        parts.append(f"    • cross_axis_note: {gp.rationale.strip()[:400]}")
    return "\n".join(p for p in parts if p)


def _excerpt_block(
    gp: Any,
    papers_dir: str,
    excerpt_chars: int,
) -> str:
    path = os.path.join(papers_dir, gp.file_name)
    if not os.path.isfile(path):
        return f"[excerpt missing: {gp.file_name}]"
    text = read_text(path, max_chars=excerpt_chars)
    return (
        f"\n--- Excerpt: {gp.file_name} (role={gp.paper_role or 'unknown'}) ---\n"
        f"{text}\n--- end excerpt ---\n"
    )


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        k = x.strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(k)
    return out


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


def _chunk_items(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _synthesis_pair_context_block() -> str:
    return (
        "Pair context (apply to all synthesis steps):\n"
        "- This query–target pair was selected for structural similarity; do not expect "
        "papers to report explicit mimicry or direct query–target interaction.\n"
        "- Semi-positive conclusions are appropriate when the target shows virulence, host "
        "manipulation, or Legionella exploitation-pathway relevance (host Axis 2), and/or "
        "the query shows effector / secretion / host-targeting evidence—even if genes are "
        "never named together.\n"
        "- Separate literature support for manipulation potential from proof of mimicry.\n\n"
    )


def _fallback_batch_summary_from_papers(
    paper_summaries: List[Dict[str, Any]],
    *,
    max_chars: int = 2500,
) -> str:
    parts: List[str] = []
    for ps in paper_summaries:
        fn = str(ps.get("file_name") or "").strip()
        summary = str(ps.get("summary") or "").strip()
        if fn and summary:
            parts.append(f"{fn}: {summary[:300]}")
    text = " ".join(parts)
    return text[:max_chars] if text else "No batch summary available."


def _summarize_batch_fallback(
    batch: List[Any],
    host_rubric: Dict[str, Any] | None = None,
    microbe_rubric: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    paper_summaries: List[Dict[str, Any]] = []
    memory_updates: List[str] = []
    for gp in batch:
        rax = _axis_rationales_for_paper(gp, host_rubric, microbe_rubric)
        top_axes = sorted(
            (gp.rubric_dimension_scores or {}).items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )[:2]
        axis_bits = []
        for ax, sc in top_axes:
            why = str(rax.get(ax) or "").strip()
            why = why[:220] if why else "no rationale provided"
            axis_bits.append(f"{ax}={float(sc):.3f} ({why})")
        summary = (
            f"Fallback summary from grader rationale for {gp.file_name}: "
            + "; ".join(axis_bits)
        )[:700]
        paper_summaries.append(
            {
                "file_name": gp.file_name,
                "summary": summary,
                "important_points": axis_bits[:3],
                "confidence_notes": "Derived from grader outputs due to batch-summary parse failure.",
            }
        )
        memory_updates.extend(axis_bits[:2])
    batch_summary = _fallback_batch_summary_from_papers(paper_summaries)
    return {
        "paper_summaries": paper_summaries,
        "memory_updates": _dedupe_keep_order(memory_updates),
        "batch_summary": batch_summary,
    }


def _parse_batch_summary_output(
    raw: str,
    batch: List[Any],
    host_rubric: Dict[str, Any] | None = None,
    microbe_rubric: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    stripped = (raw or "").strip()
    if not stripped:
        return _summarize_batch_fallback(batch, host_rubric, microbe_rubric)
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        body = lines[1:]
        while body and body[-1].strip() == "```":
            body.pop()
        stripped = "\n".join(body).strip()
    try:
        obj = json.loads(stripped)
    except Exception:
        return _summarize_batch_fallback(batch, host_rubric, microbe_rubric)
    if not isinstance(obj, dict):
        return _summarize_batch_fallback(batch, host_rubric, microbe_rubric)
    expected = {gp.file_name for gp in batch}
    items = obj.get("paper_summaries")
    if not isinstance(items, list):
        return _summarize_batch_fallback(batch, host_rubric, microbe_rubric)
    parsed_items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        file_name = str(it.get("file_name") or "").strip()
        if not file_name or file_name not in expected or file_name in seen:
            continue
        summary = str(it.get("summary") or "").strip()
        if not summary:
            continue
        parsed_items.append(
            {
                "file_name": file_name,
                "summary": summary[:1200],
                "important_points": _as_list(it.get("important_points"))[:6],
                "confidence_notes": str(it.get("confidence_notes") or "").strip()[:500],
            }
        )
        seen.add(file_name)
    if len(parsed_items) != len(batch):
        return _summarize_batch_fallback(batch, host_rubric, microbe_rubric)
    memory_updates = _as_list(obj.get("memory_updates"))[:20]
    batch_summary = str(obj.get("batch_summary") or "").strip()
    if not batch_summary:
        batch_summary = _fallback_batch_summary_from_papers(parsed_items)
    return {
        "paper_summaries": parsed_items,
        "memory_updates": _dedupe_keep_order(memory_updates),
        "batch_summary": batch_summary[:2500],
    }


def _run_role_batch_chain(
    role: str,
    papers: List[Any],
    *,
    alignment_id: str,
    instructions: str,
    term_block: str,
    host_rubric: Dict[str, Any] | None,
    microbe_rubric: Dict[str, Any] | None,
    batch_size: int,
    per_axis_cap: int,
    prior_summary_max_chars: int,
    max_tokens: int,
    temperature: float,
    call_llm: Any,
    llm_base_url: Optional[str],
) -> Tuple[str, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stateful batch summarization for one role track (host=target or query)."""
    track = _track_label(role)
    running_summary = ""
    memory_points: List[str] = []
    all_paper_summaries: List[Dict[str, Any]] = []
    batch_outputs: List[Dict[str, Any]] = []
    batches = _chunk_items(papers, batch_size) if papers else []

    for batch_idx, batch in enumerate(batches, start=1):
        prior = (
            running_summary[:prior_summary_max_chars]
            if running_summary
            else "(none — first batch on this track)"
        )
        batch_lines = [
            _paper_metadata_lines(gp, per_axis_cap, host_rubric, microbe_rubric)
            for gp in batch
        ]
        batch_prompt = (
            f"{instructions}\n\n"
            f"{_synthesis_pair_context_block()}"
            f"{term_block}\n\n"
            f"Stateful {track} track: summarize batch {batch_idx}/{len(batches)} "
            f"({len(papers)} papers on this track).\n"
            "Focus only on papers in this batch; they share the same gene role "
            f"({track}). Do not require direct query–target co-mention.\n"
            "Return strict JSON only with keys:\n"
            "- paper_summaries: array of objects with keys "
            "(file_name, summary, important_points, confidence_notes)\n"
            "- memory_updates: array of short strings for cross-paper memory on this track\n"
            "- batch_summary: cumulative narrative rollup (<=400 words) integrating "
            "the prior batch summary and this batch for the next round\n\n"
            f"Prior batch summary for {track} track:\n{prior}\n\n"
            f"Prior memory points ({track} track):\n"
            f"{json.dumps(memory_points[-60:], ensure_ascii=False)}\n\n"
            f"Batch papers:\n" + "\n".join(batch_lines)
        )
        batch_raw = ""
        if llm_base_url:
            try:
                batch_raw = call_llm(
                    batch_prompt,
                    llm_base_url,
                    min(max_tokens, 3000),
                    temperature,
                )
            except Exception as e:
                logger.warning(
                    "Synthesis {} track batch failed for {} batch {}: {}",
                    track,
                    alignment_id,
                    batch_idx,
                    e,
                )
        parsed_batch = _parse_batch_summary_output(
            batch_raw, batch, host_rubric, microbe_rubric
        )
        all_paper_summaries.extend(parsed_batch["paper_summaries"])
        memory_points = _dedupe_keep_order(
            memory_points + parsed_batch.get("memory_updates", [])
        )[-200:]
        running_summary = str(parsed_batch.get("batch_summary") or "").strip()
        if not running_summary:
            running_summary = _fallback_batch_summary_from_papers(
                parsed_batch["paper_summaries"],
                max_chars=prior_summary_max_chars,
            )
        running_summary = running_summary[:prior_summary_max_chars]
        batch_outputs.append(
            {
                "batch_index": batch_idx,
                "track": track,
                "role": role,
                "paper_files": [gp.file_name for gp in batch],
                "memory_updates": parsed_batch.get("memory_updates", []),
                "batch_summary": running_summary,
            }
        )

    return running_summary, memory_points, all_paper_summaries, batch_outputs


def run_alignment_graded(
    req: RunAlignmentGradedRequest,
    *,
    call_llm: Any,
    write_results: Any,
) -> RunAlignmentResponse:
    """Execute synthesis v2. call_llm(prompt, base_url, max_tokens, temperature) -> str."""
    if not os.path.isdir(req.papers_dir):
        raise ValueError(f"papers_dir does not exist: {req.papers_dir}")

    ensure_dir(req.output_root)
    log_dir = os.path.join(req.output_root, "logs")
    log_path = os.path.join(log_dir, f"{req.alignment_id}_synthesis.log")
    llm_base_url = os.environ.get("VLLM_BASE_URL")
    max_tokens = (req.constraints and req.constraints.max_tokens) or 4096
    temperature = (
        (req.constraints and req.constraints.temperature)
        if req.constraints is not None
        else 0.0
    )
    if temperature is None:
        temperature = 0.0

    min_axis_score = env_positive_float("SYNTHESIS_MIN_AXIS_SCORE", 0.25)
    top_k_host = env_positive_int("SYNTHESIS_TOP_K_HOST", 40)
    top_k_query = env_positive_int("SYNTHESIS_TOP_K_QUERY", 40)
    excerpt_k_host = env_positive_int("SYNTHESIS_EXCERPT_TOP_K_HOST", top_k_host)
    excerpt_k_query = env_positive_int("SYNTHESIS_EXCERPT_TOP_K_QUERY", top_k_query)
    excerpt_chars = env_positive_int("SYNTHESIS_EXCERPT_CHARS", 12000)
    prior_summary_max_chars = env_positive_int("SYNTHESIS_PRIOR_SUMMARY_MAX_CHARS", 2500)
    filtered_rule = (
        f"relevance_grade > 0.0 OR max(axis_score) >= {min_axis_score}"
    )

    sorted_graded = _sort_papers_by_relevance(list(req.graded_papers))
    kept_for_synthesis = [
        gp for gp in sorted_graded if paper_kept_for_synthesis(gp, min_axis_score)
    ]
    filtered_out = [
        gp for gp in sorted_graded if not paper_kept_for_synthesis(gp, min_axis_score)
    ]
    batch_pool, excerpt_pool, host_pool, query_pool = _select_top_k_per_role(
        kept_for_synthesis,
        top_k_host,
        top_k_query,
        excerpt_k_host=min(excerpt_k_host, top_k_host),
        excerpt_k_query=min(excerpt_k_query, top_k_query),
    )

    host_rubric = _load_rubric_json(os.environ.get("HOST_RUBRIC_PATH", ""))
    microbe_rubric = _load_rubric_json(os.environ.get("MICROBE_RUBRIC_PATH", ""))

    per_axis_cap = 700
    grading_meta = req.grading_meta or {}
    term_block = identification_terms_block(req.query, req.target_id, req.gene_context)
    batch_size_raw = int(os.environ.get("SYNTHESIS_BATCH_SIZE", "5") or "5")
    batch_size = max(3, min(8, batch_size_raw))

    host_running_summary, host_memory, host_paper_summaries, host_batch_outputs = (
        _run_role_batch_chain(
            "target",
            host_pool,
            alignment_id=req.alignment_id,
            instructions=req.instructions,
            term_block=term_block,
            host_rubric=host_rubric,
            microbe_rubric=microbe_rubric,
            batch_size=batch_size,
            per_axis_cap=per_axis_cap,
            prior_summary_max_chars=prior_summary_max_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            call_llm=call_llm,
            llm_base_url=llm_base_url,
        )
    )
    query_running_summary, query_memory, query_paper_summaries, query_batch_outputs = (
        _run_role_batch_chain(
            "query",
            query_pool,
            alignment_id=req.alignment_id,
            instructions=req.instructions,
            term_block=term_block,
            host_rubric=host_rubric,
            microbe_rubric=microbe_rubric,
            batch_size=batch_size,
            per_axis_cap=per_axis_cap,
            prior_summary_max_chars=prior_summary_max_chars,
            max_tokens=max_tokens,
            temperature=temperature,
            call_llm=call_llm,
            llm_base_url=llm_base_url,
        )
    )

    all_paper_summaries = host_paper_summaries + query_paper_summaries
    important_points_memory = _dedupe_keep_order(host_memory + query_memory)[-200:]

    excerpt_sections = ""
    if excerpt_pool and os.path.isdir(req.papers_dir):
        parts = [
            _excerpt_block(gp, req.papers_dir, excerpt_chars) for gp in excerpt_pool
        ]
        excerpt_sections = (
            "\nTop-paper text excerpts (verify grader claims; pair-level bridge):\n"
            + "".join(parts)
        )

    host_batch_count = len(host_batch_outputs)
    query_batch_count = len(query_batch_outputs)

    synth_prompt = (
        f"{req.instructions}\n\n"
        f"{_synthesis_pair_context_block()}"
        f"{term_block}\n\n"
        "You are in final pair-level synthesis. Bridge host (target) exploitation evidence "
        "with query (microbe) effector evidence. Do not require co-mention of both genes.\n\n"
        f"Host track summary ({len(host_pool)} papers, {host_batch_count} batches):\n"
        f"{host_running_summary or '(no host papers in synthesis pool)'}\n\n"
        f"Query track summary ({len(query_pool)} papers, {query_batch_count} batches):\n"
        f"{query_running_summary or '(no query papers in synthesis pool)'}\n\n"
        "Instruction: Write a running discussion (plain text, not JSON) that references "
        "which host and query axis patterns drive confidence or uncertainty. "
        "Weight host infection_process_relevance and microbe system_relevance even when "
        "aggregate relevance_grade is low. Use rubric_tags (mimicry_potential_flag, "
        "novelty_flag) when present.\n\n"
        "Assign integer scores 0-100 (not categorical Some/High labels) for each dimension.\n"
        f"{quick_summary_prompt_footer()}"
        f"Alignment: {req.alignment_id}\n"
        f"Query={req.query}\n"
        f"Target={req.target_id}\n"
        f"Grading meta: {json.dumps(grading_meta, ensure_ascii=False)}\n"
        f"Synthesis filtering: kept={len(kept_for_synthesis)} filtered_out={len(filtered_out)} "
        f"batch_pool={len(batch_pool)} host_pool={len(host_pool)} query_pool={len(query_pool)} "
        f"excerpt_pool={len(excerpt_pool)} top_k_host={top_k_host} top_k_query={top_k_query} "
        f"rule={filtered_rule}\n"
        f"Cross-track memory points:\n"
        f"{json.dumps(important_points_memory, ensure_ascii=False)}\n"
        + excerpt_sections
    )

    synthesis_retry_suffix = quick_summary_retry_suffix()

    synthesis_text = ""
    notes = ""
    if llm_base_url:
        synth_ok = False
        for attempt in range(2):
            extra = ""
            if attempt:
                bad = synthesis_text.strip()[:2500]
                extra = synthesis_retry_suffix
                if bad:
                    extra += f"\n\nEarlier attempt (invalid or incomplete):\n{bad}\n"
            try:
                synthesis_text = call_llm(
                    synth_prompt + extra,
                    llm_base_url,
                    max_tokens,
                    temperature,
                )
            except Exception as e:
                notes = str(e)
                logger.warning("Synthesis LLM failed for {}: {}", req.alignment_id, e)
                break
            if synthesis_text.strip() and synthesis_output_well_formed(synthesis_text):
                synth_ok = True
                break
        if not synth_ok and synthesis_text.strip() and not notes:
            logger.warning(
                "Synthesis output ill-formed for {}; using fallback",
                req.alignment_id,
            )

    synth_needs_fallback = not synthesis_text.strip() or (
        bool(llm_base_url) and not notes and not synthesis_output_well_formed(synthesis_text)
    )
    synthesis_status = "ok"
    if synth_needs_fallback:
        if notes:
            notes = f"{notes}; synthesis fallback applied"
            synthesis_status = "error"
        elif not synthesis_text.strip():
            notes = "empty synthesis output"
            synthesis_status = "error"
        else:
            notes = "synthesis missing parseable Quick results JSON after retry"
            synthesis_status = "error"
        conclusion = build_conclusion(
            batch_pool,
            "",
            synthesis_status="grades_only",
        )
        synthesis_text = format_fallback_discussion(
            req.alignment_id,
            len(batch_pool),
            conclusion,
        )
    else:
        conclusion = build_conclusion(
            batch_pool,
            synthesis_text,
            synthesis_status="ok",
        )

    if log_path:
        ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"alignment={req.alignment_id}\n")
            f.write(f"n_graded={len(req.graded_papers)}\n")
            f.write(
                f"n_kept={len(kept_for_synthesis)} n_filtered={len(filtered_out)} "
                f"batch_pool={len(batch_pool)} host_pool={len(host_pool)} "
                f"query_pool={len(query_pool)} excerpt_pool={len(excerpt_pool)} "
                f"rule={filtered_rule}\n"
            )
            f.write(f"synthesis_len={len(synthesis_text)} notes={notes}\n")

    graded_path = os.path.join(req.output_root, f"{req.alignment_id}_graded.json")
    analysis_payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "graded_path": graded_path,
        "grading_meta": grading_meta,
        "conclusion": conclusion,
        "synthesis": {
            "text": synthesis_text,
            "notes": notes,
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
            "constraints": req.constraints.dict() if req.constraints else None,
            "filter_rule": filtered_rule,
            "filtered_out_count": len(filtered_out),
            "kept_count": len(kept_for_synthesis),
            "batch_pool_count": len(batch_pool),
            "host_pool_count": len(host_pool),
            "query_pool_count": len(query_pool),
            "excerpt_pool_count": len(excerpt_pool),
            "top_k_host": top_k_host,
            "top_k_query": top_k_query,
            "batch_size": batch_size,
            "host_batch_count": host_batch_count,
            "query_batch_count": query_batch_count,
            "host_running_summary": host_running_summary,
            "query_running_summary": query_running_summary,
            "paper_summaries": all_paper_summaries,
            "batch_outputs_host": host_batch_outputs,
            "batch_outputs_query": query_batch_outputs,
            "important_points_memory": important_points_memory,
            "filtered_out_papers": [gp.file_name for gp in filtered_out],
            "excerpt_papers": [gp.file_name for gp in excerpt_pool],
        },
        "meta": {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "graded_synthesis_v2",
        },
    }
    analysis_path = os.path.join(req.output_root, f"{req.alignment_id}_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    final_payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "analysis_path": analysis_path,
        "conclusion": conclusion,
        "synthesis": {
            "text": synthesis_text,
            "notes": notes,
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
        },
        "meta": {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "final_conclusion",
        },
    }
    return write_results(req, final_payload)
