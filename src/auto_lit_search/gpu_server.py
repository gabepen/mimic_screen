import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from loguru import logger

from auto_lit_search.analysis_packet import (
    Constraints,
    RunAlignmentGradedRequest,
    RunAlignmentRequest,
    RunAlignmentResponse,
)

app = FastAPI(title="auto_lit_search GPU node")

TEXT_EXTENSIONS = (".txt",)
MAX_PAPER_CHARS = 120000
_MODEL_ID_CACHE: Dict[str, str] = {}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_paper_files(papers_dir: str) -> List[str]:
    return sorted(
        f for f in os.listdir(papers_dir)
        if os.path.isfile(os.path.join(papers_dir, f))
        and (f.endswith(TEXT_EXTENSIONS) or not f.endswith((".pdf", ".xml")))
    )


def _read_text(path: str, max_chars: int = MAX_PAPER_CHARS) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n[truncated]"
    return s


def _call_llm(
    user_content: str,
    base_url: str,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    import requests

    def _fetch_served_model_id(root: str) -> str:
        cached = _MODEL_ID_CACHE.get(root)
        if cached:
            return cached
        models_url = f"{root}/v1/models"
        mr = requests.get(models_url, timeout=30)
        mr.raise_for_status()
        mdata = mr.json()
        entries = mdata.get("data") if isinstance(mdata, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("id"):
                    mid = str(entry["id"]).strip()
                    if mid:
                        _MODEL_ID_CACHE[root] = mid
                        return mid
        raise RuntimeError(f"Could not resolve model id from {models_url}")

    root_url = base_url.rstrip("/")
    base_url = root_url
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat/completions"
    configured = (os.environ.get("VLLM_MODEL_NAME") or "").strip()
    if not model:
        if configured:
            model = configured
        else:
            model = _fetch_served_model_id(root_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code == 404 and configured:
        # Common case: env has a friendly name (e.g. qwen3), but vLLM serves a path model id.
        served_model = _fetch_served_model_id(root_url)
        if served_model != model:
            payload["model"] = served_model
            r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return (msg.get("content") or "").strip()
    return ""


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
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
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _gene_terms(meta: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    symbol = str(meta.get("gene_name") or "").strip() or fallback_id
    common_name = str(meta.get("common_name") or "").strip()
    syn_keys = [
        "synonyms",
        "gene_synonyms",
        "aliases",
        "alias",
        "name_synonyms",
    ]
    syns: List[str] = []
    for k in syn_keys:
        syns.extend(_as_list(meta.get(k)))
    syns = _dedupe_keep_order(syns)
    syns = [s for s in syns if s.lower() not in {symbol.lower(), common_name.lower()}]
    return {
        "symbol": symbol,
        "common_name": common_name or "none",
        "synonyms": syns,
    }


def _identification_terms_block(
    query: str,
    target_id: str,
    gene_context: Optional[Dict[str, Any]],
) -> str:
    query_meta = (gene_context or {}).get("query") or {}
    target_meta = (gene_context or {}).get("target") or {}
    q = _gene_terms(query_meta, query)
    t = _gene_terms(target_meta, target_id)
    q_syn = ", ".join(q["synonyms"]) if q["synonyms"] else "none"
    t_syn = ", ".join(t["synonyms"]) if t["synonyms"] else "none"
    return (
        "Paper identification terms used in retrieval (prioritize symbol/common name; "
        "use synonyms as alternate mentions):\n"
        f"- Query gene ({query}): symbol={q['symbol']}; common_name={q['common_name']}; synonyms={q_syn}\n"
        f"- Target gene ({target_id}): symbol={t['symbol']}; common_name={t['common_name']}; synonyms={t_syn}\n"
    )


def _analyze_paper(
    file_path: str,
    query: str,
    target_id: str,
    instructions: str,
    constraints: Optional[Constraints],
    llm_base_url: Optional[str],
    log_path: Optional[str],
    gene_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stat = os.stat(file_path)
    text = _read_text(file_path)
    summary = ""
    relevance_score = None
    notes = ""

    # Infer side from filename markers.
    # Accept both strict files (<pmcid>__query.txt / <pmcid>__target.txt) and
    # source-tagged variants like <doi>__target__unpaywall.txt.
    fname = os.path.basename(file_path)
    paper_role: Optional[str] = None
    fname_lower = fname.lower()
    if "__query" in fname_lower:
        paper_role = "query"
    elif "__target" in fname_lower:
        paper_role = "target"

    query_meta = (gene_context or {}).get("query") or {}
    target_meta = (gene_context or {}).get("target") or {}
    term_block = _identification_terms_block(query, target_id, gene_context)
    query_gene_name = (query_meta.get("gene_name") or "").strip() or query
    target_gene_name = (target_meta.get("gene_name") or "").strip() or target_id

    def _format_identifiers(meta: Dict[str, Any]) -> str:
        # Keep this compact: only include non-empty identifier fields.
        keys = [
            ("gene_name", "gene_name"),
            ("uniprot_id", "uniprot_id"),
            ("entrez_id", "entrez_id"),
            ("locus_tag", "locus_tag"),
            ("genbank_acc", "genbank_acc"),
            ("common_name", "common_name"),
        ]
        parts: List[str] = []
        for k, label in keys:
            v = meta.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(f"{label}={s}")
        return "; ".join(parts) if parts else "none"

    if paper_role == "query":
        focus_gene_id = query
        focus_gene_name = query_gene_name
        other_gene_id = target_id
        other_gene_name = target_gene_name
    elif paper_role == "target":
        focus_gene_id = target_id
        focus_gene_name = target_gene_name
        other_gene_id = query
        other_gene_name = query_gene_name
    else:
        # Fallback when file naming isn't role-suffixed.
        focus_gene_id = query
        focus_gene_name = query_gene_name
        other_gene_id = target_id
        other_gene_name = target_gene_name

    focus_id_str = _format_identifiers(
        query_meta if paper_role == "query" else (target_meta if paper_role == "target" else query_meta)
    )
    other_id_str = _format_identifiers(
        query_meta if paper_role == "target" else (target_meta if paper_role == "query" else target_meta)
    )

    user_content = (
        f"{instructions}\n\n"
        f"{term_block}\n"
        f"Alignment genes:\n"
        f"- Focus gene: {focus_gene_name} ({focus_gene_id})\n"
        f"- Focus gene identifiers: {focus_id_str}\n"
        f"- Other gene: {other_gene_name} ({other_gene_id})\n"
        f"- Other gene identifiers: {other_id_str}\n"
        f"- Paper role in alignment: {paper_role or 'unknown'}\n\n"
        f"Paper excerpt:\n{text[:80000]}"
    )

    if llm_base_url and text.strip():
        max_tokens = (constraints and constraints.max_tokens) or 4096
        temperature = (constraints and constraints.temperature) if constraints is not None else 0.0
        if temperature is None:
            temperature = 0.0
        try:
            raw = _call_llm(user_content, llm_base_url, max_tokens=max_tokens, temperature=temperature)
            if raw:
                summary = raw
                try:
                    if "relevance" in raw.lower() or "score" in raw.lower():
                        for part in raw.replace(",", " ").split():
                            try:
                                v = float(part)
                                if 0 <= v <= 1:
                                    relevance_score = v
                                    break
                            except ValueError:
                                continue
                except Exception:
                    pass
        except Exception as e:
            notes = str(e)
            logger.warning(f"LLM call failed for {file_path}: {e}")

    if log_path:
        _ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- {os.path.basename(file_path)} ---\n")
            f.write(f"query={query} target_id={target_id}\n")
            f.write(f"inferred_paper_role={paper_role}\n")
            f.write(f"focus_gene_id={focus_gene_id} other_gene_id={other_gene_id}\n")
            # Keep prompt small; excerpt can be huge.
            if text.strip():
                f.write("prompt_preview:\n")
                f.write(user_content[:4000] + ("\n...[truncated]\n" if len(user_content) > 4000 else "\n"))
            f.write(f"summary_len={len(summary)} relevance_score={relevance_score}\n")
            f.write(f"{summary[:2000]}\n\n")

    return {
        "summary": summary,
        "relevance_score": relevance_score,
        "notes": notes,
        "file_size_bytes": stat.st_size,
        "query": query,
        "target_id": target_id,
        "paper_role": paper_role,
        "focus_gene_id": focus_gene_id,
    }


def _write_alignment_results(
    req: RunAlignmentRequest,
    payload: Dict[str, Any],
) -> RunAlignmentResponse:
    _ensure_dir(req.output_root)
    result_path = os.path.join(req.output_root, f"{req.alignment_id}_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        f"GPU node wrote results for {req.alignment_id} "
        f"({len(payload.get('papers', []))} papers) -> {result_path}"
    )
    return RunAlignmentResponse(
        status="ok",
        alignment_id=req.alignment_id,
        results_path=result_path,
    )


def _fallback_synthesis_text(
    req: RunAlignmentGradedRequest,
    graded_override: Optional[List[Any]] = None,
) -> str:
    graded = graded_override if graded_override is not None else (req.graded_papers or [])
    if not graded:
        return (
            "No graded papers were available for synthesis.\n\n"
            "Quick results summary:\n"
            "- Likelihood of host manipulation/mimicry (0..1): 0.0\n"
            "- Best supporting paper(s): none\n"
            "- Main conflicts / uncertainties: No evidence was available."
        )
    top = sorted(graded, key=lambda g: g.relevance_grade, reverse=True)[:5]
    best_files = ", ".join(g.file_name for g in top if g.relevance_grade > 0)
    if not best_files:
        best_files = "none"
    max_grade = max(float(g.relevance_grade) for g in graded)
    mean_grade = sum(float(g.relevance_grade) for g in graded) / max(1, len(graded))
    likelihood = max_grade * 0.7 + mean_grade * 0.3
    return (
        "Synthesis fallback generated because the LLM returned empty output.\n"
        f"Processed {len(graded)} graded papers for {req.alignment_id}. "
        f"Max relevance grade={max_grade:.3f}, mean relevance grade={mean_grade:.3f}.\n\n"
        "Quick results summary:\n"
        f"- Likelihood of host manipulation/mimicry (0..1): {likelihood:.3f}\n"
        f"- Best supporting paper(s): {best_files}\n"
        "- Main conflicts / uncertainties: The synthesis model response was empty, "
        "so this conclusion is a conservative heuristic from rubric grades."
    )


def _parse_quick_results_summary(synthesis_text: str) -> Dict[str, Any]:
    text = synthesis_text or ""
    likelihood = None
    best_support = ""
    conflicts = ""
    m = re.search(
        r"Likelihood of host manipulation/mimicry \(0\.\.1\):\s*([0-9]*\.?[0-9]+)",
        text,
    )
    if m:
        try:
            likelihood = float(m.group(1))
        except Exception:
            likelihood = None
    m = re.search(r"Best supporting paper\(s\):\s*(.+)", text)
    if m:
        best_support = m.group(1).strip()
    m = re.search(r"Main conflicts / uncertainties:\s*(.+)", text)
    if m:
        conflicts = m.group(1).strip()
    return {
        "likelihood_host_manipulation_mimicry": likelihood,
        "best_supporting_papers": best_support,
        "main_conflicts_uncertainties": conflicts,
    }


def _synthesis_output_well_formed(synthesis_text: str) -> bool:
    text = (synthesis_text or "").strip()
    if not text or "Quick results summary:" not in text:
        return False
    quick = _parse_quick_results_summary(text)
    return quick.get("likelihood_host_manipulation_mimicry") is not None


def _chunk_items(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


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
    return {"paper_summaries": paper_summaries, "memory_updates": _dedupe_keep_order(memory_updates)}


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
    seen: set = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        file_name = str(it.get("file_name") or "").strip()
        if not file_name or file_name not in expected or file_name in seen:
            continue
        summary = str(it.get("summary") or "").strip()
        if not summary:
            continue
        important_points = _as_list(it.get("important_points"))[:6]
        confidence_notes = str(it.get("confidence_notes") or "").strip()
        parsed_items.append(
            {
                "file_name": file_name,
                "summary": summary[:1200],
                "important_points": important_points,
                "confidence_notes": confidence_notes[:500],
            }
        )
        seen.add(file_name)
    if len(parsed_items) != len(batch):
        return _summarize_batch_fallback(batch)
    memory_updates = _as_list(obj.get("memory_updates"))[:20]
    return {"paper_summaries": parsed_items, "memory_updates": _dedupe_keep_order(memory_updates)}


def _run_alignment_impl(req: RunAlignmentRequest) -> RunAlignmentResponse:
    if not os.path.isdir(req.papers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"papers_dir does not exist or is not a directory: {req.papers_dir}",
        )

    _ensure_dir(req.output_root)
    log_dir = os.path.join(req.output_root, "logs")
    log_path = os.path.join(log_dir, f"{req.alignment_id}.log")

    files = _list_paper_files(req.papers_dir)
    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"no files found in papers_dir: {req.papers_dir}",
        )

    # If we have role-labeled inputs (including source-tagged variants like
    # <doi>__target__unpaywall.txt), prefer them over legacy unlabeled .txt files.
    # prefer them over any legacy unlabeled .txt files.
    labeled = [
        f
        for f in files
        if "__query" in f.lower() or "__target" in f.lower()
    ]
    if labeled:
        files = labeled

    llm_base_url = os.environ.get("VLLM_BASE_URL")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    papers: List[Dict[str, Any]] = []
    for fname in files:
        fpath = os.path.join(req.papers_dir, fname)
        fname_base = os.path.basename(fpath)
        analysis = _analyze_paper(
            file_path=fpath,
            query=req.query,
            target_id=req.target_id,
            instructions=req.instructions,
            constraints=req.constraints,
            llm_base_url=llm_base_url,
            log_path=log_path,
            gene_context=req.gene_context,
        )
        papers.append(
            {
                "paper_id": fname,
                "file_name": fname,
                "analysis": analysis,
            }
        )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "papers": papers,
        "meta": {
            "started_at": started_at,
            "finished_at": finished_at,
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
            "constraints": req.constraints.dict() if req.constraints else None,
        },
    }

    return _write_alignment_results(req, payload)


def _run_alignment_graded_impl(
    req: RunAlignmentGradedRequest,
) -> RunAlignmentResponse:
    if not os.path.isdir(req.papers_dir):
        raise HTTPException(
            status_code=400,
            detail=f"papers_dir does not exist or is not a directory: {req.papers_dir}",
        )
    _ensure_dir(req.output_root)
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

    # Filter synthesis inputs to non-zero aggregate relevance.
    filtered_rule = "relevance_grade > 0.0"
    sorted_graded = sorted(
        req.graded_papers,
        key=lambda g: (-float(g.relevance_grade), g.file_name),
    )
    kept_for_synthesis = [gp for gp in sorted_graded if float(gp.relevance_grade) > 0.0]
    filtered_out = [gp for gp in sorted_graded if float(gp.relevance_grade) <= 0.0]

    # Per-paper axis evidence lines used for batch summarization.
    _per_axis_cap = 700
    grading_meta = req.grading_meta or {}
    term_block = _identification_terms_block(req.query, req.target_id, req.gene_context)
    batch_size_raw = int(os.environ.get("SYNTHESIS_BATCH_SIZE", "5") or "5")
    batch_size = max(3, min(5, batch_size_raw))
    batches = _chunk_items(kept_for_synthesis, batch_size) if kept_for_synthesis else []

    all_paper_summaries: List[Dict[str, Any]] = []
    important_points_memory: List[str] = []
    batch_outputs: List[Dict[str, Any]] = []

    for batch_idx, batch in enumerate(batches, start=1):
        batch_lines: List[str] = []
        for gp in batch:
            scores = gp.rubric_dimension_scores or {}
            rax = gp.rubric_axis_rationales or {}
            order = sorted(scores.keys())
            parts: List[str] = [
                f"- {gp.file_name} role={gp.paper_role or 'unknown'} "
                f"aggregate_axis_score={gp.relevance_grade:.3f}"
            ]
            for ax in order:
                sc = float(scores.get(ax, 0.0))
                why = (rax.get(ax) or "")[:_per_axis_cap]
                parts.append(f"    • {ax}: score={sc:.3f} | grader_reasoning: {why}")
            if (gp.rationale or "").strip():
                parts.append(f"    • cross_axis_note: {gp.rationale.strip()[:400]}")
            batch_lines.append("\n".join(parts))

        batch_prompt = (
            f"{req.instructions}\n\n"
            f"{term_block}\n\n"
            f"Stateful synthesis step: summarize batch {batch_idx}/{len(batches)}.\n"
            "Use prior memory points to keep continuity across batches.\n"
            "Return strict JSON only with keys:\n"
            "- paper_summaries: array of objects with keys "
            "(file_name, summary, important_points, confidence_notes)\n"
            "- memory_updates: array of short strings for cross-paper memory\n\n"
            f"Prior memory points:\n{json.dumps(important_points_memory[-60:], ensure_ascii=False)}\n\n"
            f"Batch papers:\n" + "\n".join(batch_lines[:200])
        )
        batch_raw = ""
        if llm_base_url:
            try:
                batch_raw = _call_llm(
                    batch_prompt,
                    llm_base_url,
                    max_tokens=min(max_tokens, 3000),
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(
                    f"Synthesis batch LLM call failed for {req.alignment_id} batch {batch_idx}: {e}"
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

    synth_prompt = (
        f"{req.instructions}\n\n"
        f"{term_block}\n\n"
        "You are in final synthesis stage. Use ONLY the accumulated per-paper summaries and "
        "stateful memory points below to produce the final conclusion.\n\n"
        "Instruction: Write a running discussion (plain text, not JSON) that references "
        "which summarized papers and axis patterns drive confidence or uncertainty.\n\n"
        "END WITH THIS EXACT SECTION HEADER:\n"
        "Quick results summary:\n"
        "- Likelihood of host manipulation/mimicry (0..1): <float>\n"
        "- Best supporting paper(s): <paper file_name(s)>\n"
        "- Main conflicts / uncertainties: <1-3 sentences>\n\n"
        f"Alignment: {req.alignment_id}\n"
        f"Query={req.query}\n"
        f"Target={req.target_id}\n"
        f"Grading meta: {json.dumps(grading_meta, ensure_ascii=False)}\n"
        f"Synthesis filtering: kept={len(kept_for_synthesis)} filtered_out={len(filtered_out)} "
        f"rule={filtered_rule}\n"
        f"Stateful memory points:\n{json.dumps(important_points_memory, ensure_ascii=False)}\n\n"
        "Per-paper summaries:\n"
        + ("\n".join(summary_lines[:500]) if summary_lines else "- none")
    )
    synthesis_retry_suffix = (
        "\n\nYour previous answer did not include a parseable end section. "
        "Reply with plain text only (not JSON). Keep the running discussion, then end with "
        "this exact header and bullet lines (replace bracketed parts):\n\n"
        "Quick results summary:\n"
        "- Likelihood of host manipulation/mimicry (0..1): <float>\n"
        "- Best supporting paper(s): <paper file_name(s)>\n"
        "- Main conflicts / uncertainties: <1-3 sentences>\n"
    )
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
                synthesis_text = _call_llm(
                    synth_prompt + extra,
                    llm_base_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                notes = str(e)
                logger.warning(f"Synthesis LLM call failed for {req.alignment_id}: {e}")
                break
            if synthesis_text.strip() and _synthesis_output_well_formed(synthesis_text):
                synth_ok = True
                break
            if attempt == 0:
                reason = (
                    "empty synthesis response"
                    if not synthesis_text.strip()
                    else "missing Quick results summary or unparseable likelihood line"
                )
                logger.warning(
                    f"Synthesis LLM ({req.alignment_id}): {reason}; resubmitting prompt once"
                )
        if not synth_ok and synthesis_text.strip() and not notes:
            logger.warning(
                f"Synthesis LLM ({req.alignment_id}): output still ill-formed after retry; "
                "using fallback text"
            )
    synth_needs_fallback = not synthesis_text.strip() or (
        bool(llm_base_url)
        and not notes
        and not _synthesis_output_well_formed(synthesis_text)
    )
    if synth_needs_fallback:
        if notes:
            notes = f"{notes}; synthesis fallback applied"
        elif not synthesis_text.strip():
            notes = "empty synthesis output"
        else:
            notes = (
                "synthesis missing parseable Quick results summary after retry; used fallback"
            )
        synthesis_text = _fallback_synthesis_text(req, graded_override=kept_for_synthesis)
    if log_path:
        _ensure_dir(os.path.dirname(log_path))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"alignment={req.alignment_id}\n")
            f.write(f"n_graded={len(req.graded_papers)}\n")
            f.write(
                f"n_kept_for_synthesis={len(kept_for_synthesis)} "
                f"n_filtered_out={len(filtered_out)} rule={filtered_rule}\n"
            )
            f.write(
                f"prompt_preview={synth_prompt[:3500]}"
                + ("\n...[truncated]\n" if len(synth_prompt) > 3500 else "\n")
            )
            f.write(f"synthesis_len={len(synthesis_text)} notes={notes}\n")

    analysis_payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "graded_papers": [gp.dict() for gp in req.graded_papers],
        "grading_meta": grading_meta,
        "synthesis": {
            "text": synthesis_text,
            "notes": notes,
            "llm_model": os.environ.get("VLLM_MODEL_NAME", "unknown"),
            "constraints": req.constraints.dict() if req.constraints else None,
            "filter_rule": filtered_rule,
            "filtered_out_count": len(filtered_out),
            "kept_count": len(kept_for_synthesis),
            "batch_size": batch_size,
            "batch_count": len(batches),
            "paper_summaries": all_paper_summaries,
            "batch_outputs": batch_outputs,
            "important_points_memory": important_points_memory,
            "filtered_out_papers": [gp.file_name for gp in filtered_out],
        },
        "meta": {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": "graded_synthesis",
        },
    }
    analysis_path = os.path.join(req.output_root, f"{req.alignment_id}_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    quick = _parse_quick_results_summary(synthesis_text)
    final_payload: Dict[str, Any] = {
        "alignment_id": req.alignment_id,
        "query": req.query,
        "target_id": req.target_id,
        "papers_dir": req.papers_dir,
        "analysis_path": analysis_path,
        "conclusion": quick,
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
    return _write_alignment_results(req, final_payload)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "detail": "ready"}


@app.post("/run_alignment", response_model=RunAlignmentResponse)
def run_alignment(req: RunAlignmentRequest) -> RunAlignmentResponse:
    return _run_alignment_impl(req)


@app.post("/run_alignment_graded", response_model=RunAlignmentResponse)
def run_alignment_graded(req: RunAlignmentGradedRequest) -> RunAlignmentResponse:
    return _run_alignment_graded_impl(req)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GPU_API_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

