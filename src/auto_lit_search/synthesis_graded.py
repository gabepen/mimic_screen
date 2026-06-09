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


def _select_top_k_per_role(
    kept: List[Any],
    top_k_host: int,
    top_k_query: int,
) -> Tuple[List[Any], List[Any]]:
    """Return (batch_pool, excerpt_pool). batch_pool may equal kept; excerpt_pool is top-K per role."""
    by_role: Dict[str, List[Any]] = {"target": [], "query": [], "other": []}
    for gp in kept:
        by_role.setdefault(_role_key(gp), []).append(gp)

    excerpt: List[Any] = []
    for role, cap in (("target", top_k_host), ("query", top_k_query)):
        pool = by_role.get(role) or []
        excerpt.extend(pool[:cap])
    other = by_role.get("other") or []
    excerpt.extend(other[: max(0, min(5, top_k_host // 5))])
    excerpt_names = {gp.file_name for gp in excerpt}
    batch_pool = kept
    return batch_pool, [gp for gp in kept if gp.file_name in excerpt_names]


def _tags_line(gp: Any) -> str:
    tags = getattr(gp, "rubric_tags", None) or {}
    if not tags:
        return ""
    bits = [f"{k}={v}" for k, v in sorted(tags.items()) if str(v).strip()]
    return f"    • rubric_tags: {', '.join(bits)}\n" if bits else ""


def _paper_metadata_lines(gp: Any, per_axis_cap: int) -> str:
    scores = gp.rubric_dimension_scores or {}
    rax = gp.rubric_axis_rationales or {}
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


def _summarize_batch_fallback(batch: List[Any]) -> Dict[str, Any]:
    paper_summaries: List[Dict[str, Any]] = []
    memory_updates: List[str] = []
    for gp in batch:
        rax = gp.rubric_axis_rationales or {}
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
    return {
        "paper_summaries": paper_summaries,
        "memory_updates": _dedupe_keep_order(memory_updates),
    }


def _parse_batch_summary_output(raw: str, batch: List[Any]) -> Dict[str, Any]:
    stripped = (raw or "").strip()
    if not stripped:
        return _summarize_batch_fallback(batch)
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        body = lines[1:]
        while body and body[-1].strip() == "```":
            body.pop()
        stripped = "\n".join(body).strip()
    try:
        obj = json.loads(stripped)
    except Exception:
        return _summarize_batch_fallback(batch)
    if not isinstance(obj, dict):
        return _summarize_batch_fallback(batch)
    expected = {gp.file_name for gp in batch}
    items = obj.get("paper_summaries")
    if not isinstance(items, list):
        return _summarize_batch_fallback(batch)
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
        return _summarize_batch_fallback(batch)
    memory_updates = _as_list(obj.get("memory_updates"))[:20]
    return {
        "paper_summaries": parsed_items,
        "memory_updates": _dedupe_keep_order(memory_updates),
    }


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
    top_k_host = env_positive_int("SYNTHESIS_TOP_K_HOST", 25)
    top_k_query = env_positive_int("SYNTHESIS_TOP_K_QUERY", 10)
    excerpt_chars = env_positive_int("SYNTHESIS_EXCERPT_CHARS", 12000)
    filtered_rule = (
        f"relevance_grade > 0.0 OR max(axis_score) >= {min_axis_score}"
    )

    sorted_graded = sorted(
        req.graded_papers,
        key=lambda g: (-float(g.relevance_grade), -max_axis_score(g), g.file_name),
    )
    kept_for_synthesis = [
        gp for gp in sorted_graded if paper_kept_for_synthesis(gp, min_axis_score)
    ]
    filtered_out = [
        gp for gp in sorted_graded if not paper_kept_for_synthesis(gp, min_axis_score)
    ]
    batch_pool, excerpt_pool = _select_top_k_per_role(
        kept_for_synthesis, top_k_host, top_k_query
    )

    per_axis_cap = 700
    grading_meta = req.grading_meta or {}
    term_block = identification_terms_block(req.query, req.target_id, req.gene_context)
    batch_size_raw = int(os.environ.get("SYNTHESIS_BATCH_SIZE", "5") or "5")
    batch_size = max(3, min(5, batch_size_raw))
    batches = _chunk_items(batch_pool, batch_size) if batch_pool else []

    all_paper_summaries: List[Dict[str, Any]] = []
    important_points_memory: List[str] = []
    batch_outputs: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(batches, start=1):
        batch_lines = [_paper_metadata_lines(gp, per_axis_cap) for gp in batch]
        batch_prompt = (
            f"{req.instructions}\n\n"
            f"{_synthesis_pair_context_block()}"
            f"{term_block}\n\n"
            f"Stateful synthesis step: summarize batch {batch_idx}/{len(batches)}.\n"
            "Use prior memory points to keep continuity across batches.\n"
            "Do not require direct query–target co-mention; summarize pathway and "
            "effector relevance from rubric axes and rubric_tags when present.\n"
            "Return strict JSON only with keys:\n"
            "- paper_summaries: array of objects with keys "
            "(file_name, summary, important_points, confidence_notes)\n"
            "- memory_updates: array of short strings for cross-paper memory\n\n"
            f"Prior memory points:\n"
            f"{json.dumps(important_points_memory[-60:], ensure_ascii=False)}\n\n"
            f"Batch papers:\n" + "\n".join(batch_lines[:200])
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
                    "Synthesis batch failed for {} batch {}: {}",
                    req.alignment_id,
                    batch_idx,
                    e,
                )
        parsed_batch = _parse_batch_summary_output(batch_raw, batch)
        all_paper_summaries.extend(parsed_batch["paper_summaries"])
        important_points_memory = _dedupe_keep_order(
            important_points_memory + parsed_batch.get("memory_updates", [])
        )[-200:]
        batch_outputs.append(
            {
                "batch_index": batch_idx,
                "paper_files": [gp.file_name for gp in batch],
                "memory_updates": parsed_batch.get("memory_updates", []),
            }
        )

    summary_lines: List[str] = []
    for ps in all_paper_summaries:
        pts = "; ".join(ps.get("important_points") or [])
        conf = ps.get("confidence_notes") or ""
        summary_lines.append(
            f"- {ps.get('file_name')}: {ps.get('summary')} "
            f"| important_points={pts} | confidence_notes={conf}"
        )

    excerpt_sections = ""
    if excerpt_pool and os.path.isdir(req.papers_dir):
        parts = [
            _excerpt_block(gp, req.papers_dir, excerpt_chars) for gp in excerpt_pool
        ]
        excerpt_sections = (
            "\nTop-paper text excerpts (verify grader claims; pair-level bridge):\n"
            + "".join(parts)
        )

    synth_prompt = (
        f"{req.instructions}\n\n"
        f"{_synthesis_pair_context_block()}"
        f"{term_block}\n\n"
        "You are in final synthesis stage. Use accumulated per-paper summaries, "
        "memory points, and top-paper excerpts below.\n\n"
        "Instruction: Write a running discussion (plain text, not JSON) that references "
        "which summarized papers and axis patterns drive confidence or uncertainty. "
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
        f"batch_pool={len(batch_pool)} excerpt_pool={len(excerpt_pool)} "
        f"top_k_host={top_k_host} top_k_query={top_k_query} rule={filtered_rule}\n"
        f"Stateful memory points:\n{json.dumps(important_points_memory, ensure_ascii=False)}\n\n"
        "Per-paper summaries:\n"
        + ("\n".join(summary_lines[:500]) if summary_lines else "- none")
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
            kept_for_synthesis,
            "",
            synthesis_status="grades_only",
        )
        synthesis_text = format_fallback_discussion(
            req.alignment_id,
            len(kept_for_synthesis),
            conclusion,
        )
    else:
        conclusion = build_conclusion(
            kept_for_synthesis,
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
                f"excerpt_pool={len(excerpt_pool)} rule={filtered_rule}\n"
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
            "excerpt_pool_count": len(excerpt_pool),
            "top_k_host": top_k_host,
            "top_k_query": top_k_query,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "paper_summaries": all_paper_summaries,
            "batch_outputs": batch_outputs,
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
