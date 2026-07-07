"""Mention-centered paper excerpts for synthesis (and shared tooling)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from auto_lit_search.paper_io import gene_terms, meta_search_literals

MENTION_SEPARATOR = "\n\n--- mention context ---\n\n"
_EXCERPT_MODE_MENTIONS = "mentions"
_EXCERPT_MODE_NO_MENTIONS = "no_mentions"
_PARA_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]*(?=\n|[A-Z0-9\"'])")
_ABSTRACT_HEADING = re.compile(r"^abstract\s*$", re.IGNORECASE)


@dataclass
class SearchTerm:
    canonical_id: str
    side: str  # query | target
    literal: str


@dataclass
class RawHit:
    start: int
    end: int
    canonical_id: str
    side: str
    literal: str


@dataclass
class MentionSite:
    start: int
    end: int
    canonical_ids: List[str] = field(default_factory=list)
    sides: List[str] = field(default_factory=list)
    literals: List[str] = field(default_factory=list)


@dataclass
class TextWindow:
    start: int
    end: int


@dataclass
class MentionExcerptResult:
    excerpt: str
    focus_gene: str
    matched_canonical_ids: List[str]
    matched_terms: List[str]
    n_mentions: int
    budget_per_mention: int
    excerpt_mode: str
    expected_gene_id: str = ""


def _expected_gene_id(row: Dict[str, Any]) -> str:
    role = (row.get("paper_role") or "").strip().lower()
    query_id = str(row.get("query") or "").strip()
    target_id = str(row.get("target_id") or "").strip()
    if role == "query":
        return query_id
    if role == "target":
        return target_id
    return f"{query_id} or {target_id}"


def collect_search_terms(row: Dict[str, Any]) -> List[SearchTerm]:
    role = (row.get("paper_role") or "").strip().lower()
    query_id = str(row.get("query") or "").strip()
    target_id = str(row.get("target_id") or "").strip()
    gene_context = row.get("gene_context") or {}

    sides: List[Tuple[str, str, Dict[str, Any]]] = []
    if role == "query":
        sides.append(("query", query_id, gene_context.get("query") or {}))
    elif role == "target":
        sides.append(("target", target_id, gene_context.get("target") or {}))
    else:
        sides.append(("query", query_id, gene_context.get("query") or {}))
        sides.append(("target", target_id, gene_context.get("target") or {}))

    terms: List[SearchTerm] = []
    seen: set[Tuple[str, str, str]] = set()
    for side, canonical_id, meta in sides:
        if not canonical_id:
            continue
        for literal in meta_search_literals(meta, canonical_id):
            if len(literal) < 2:
                continue
            key = (side, canonical_id, literal.lower())
            if key in seen:
                continue
            seen.add(key)
            terms.append(SearchTerm(canonical_id=canonical_id, side=side, literal=literal))
    terms.sort(key=lambda t: (-len(t.literal), t.literal.lower()))
    return terms


def _word_boundary_pattern(literal: str) -> Optional[re.Pattern[str]]:
    if re.search(r"[^A-Za-z0-9_-]", literal):
        return re.compile(re.escape(literal), re.IGNORECASE)
    if len(literal) < 2:
        return None
    return re.compile(rf"\b{re.escape(literal)}\b", re.IGNORECASE)


def find_hits(text: str, terms: Sequence[SearchTerm]) -> List[RawHit]:
    if not text or not terms:
        return []
    covered: List[Tuple[int, int]] = []
    hits: List[RawHit] = []

    def _overlaps_existing(s: int, e: int) -> bool:
        for cs, ce in covered:
            if not (e <= cs or s >= ce):
                return True
        return False

    for term in terms:
        pat = _word_boundary_pattern(term.literal)
        if pat is None:
            continue
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps_existing(s, e):
                continue
            covered.append((s, e))
            hits.append(RawHit(s, e, term.canonical_id, term.side, term.literal))

    hits.sort(key=lambda h: h.start)
    return hits


def cluster_mention_sites(hits: Sequence[RawHit], cluster_gap: int) -> List[MentionSite]:
    if not hits:
        return []
    sorted_hits = sorted(hits, key=lambda h: h.start)
    sites: List[MentionSite] = []
    cur_start = sorted_hits[0].start
    cur_end = sorted_hits[0].end
    cur_ids: List[str] = [sorted_hits[0].canonical_id]
    cur_sides: List[str] = [sorted_hits[0].side]
    cur_literals: List[str] = [sorted_hits[0].literal]

    for h in sorted_hits[1:]:
        if h.start <= cur_end + cluster_gap:
            cur_end = max(cur_end, h.end)
            if h.canonical_id not in cur_ids:
                cur_ids.append(h.canonical_id)
            if h.side not in cur_sides:
                cur_sides.append(h.side)
            if h.literal not in cur_literals:
                cur_literals.append(h.literal)
        else:
            sites.append(
                MentionSite(
                    start=cur_start,
                    end=cur_end,
                    canonical_ids=cur_ids,
                    sides=cur_sides,
                    literals=cur_literals,
                )
            )
            cur_start, cur_end = h.start, h.end
            cur_ids, cur_sides, cur_literals = [h.canonical_id], [h.side], [h.literal]

    sites.append(
        MentionSite(
            start=cur_start,
            end=cur_end,
            canonical_ids=cur_ids,
            sides=cur_sides,
            literals=cur_literals,
        )
    )
    return sites


def _extract_abstract_snippet(text: str, max_chars: int = 1200) -> str:
    lines = text.splitlines()
    start_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _ABSTRACT_HEADING.match(line.strip()):
            start_idx = i + 1
            break
    if start_idx is None:
        return ""
    parts: List[str] = []
    total = 0
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped and parts:
            break
        if stripped and re.match(
            r"^(introduction|keywords|background)\s*$", stripped, re.IGNORECASE
        ):
            break
        parts.append(line)
        total += len(line) + 1
        if total >= max_chars:
            break
    block = "\n".join(parts).strip()
    if len(block) > max_chars:
        block = block[:max_chars] + "\n[... abstract truncated ...]"
    return block


def _paragraph_boundaries(text: str) -> List[int]:
    bounds = [0]
    for m in _PARA_BREAK.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    return sorted(set(bounds))


def _sentence_boundary_before(text: str, pos: int, limit: int) -> Optional[int]:
    lo = max(0, pos - limit)
    chunk = text[lo:pos]
    best = None
    for m in _SENTENCE_END.finditer(chunk):
        best = lo + m.end()
    if best is not None:
        return best
    nl = chunk.rfind("\n\n")
    if nl >= 0:
        return lo + nl + 2
    return None


def _sentence_boundary_after(text: str, pos: int, limit: int) -> Optional[int]:
    hi = min(len(text), pos + limit)
    chunk = text[pos:hi]
    m = _SENTENCE_END.search(chunk)
    if m:
        return pos + m.end()
    nl = chunk.find("\n\n")
    if nl >= 0:
        return pos + nl + 2
    return None


def _snap_start(text: str, start: int, slack: int = 200) -> int:
    paras = _paragraph_boundaries(text)
    for p in paras:
        if abs(p - start) <= slack and p < start:
            return p
    sb = _sentence_boundary_before(text, start, slack)
    if sb is not None and start - sb <= slack:
        return sb
    return start


def _snap_end(text: str, end: int, slack: int = 200) -> int:
    paras = _paragraph_boundaries(text)
    for p in paras:
        if abs(p - end) <= slack and p > end:
            return p
    se = _sentence_boundary_after(text, end, slack)
    if se is not None and se - end <= slack:
        return se
    return end


def _expand_window(
    text: str,
    site: MentionSite,
    budget: int,
    *,
    doc_start_frac: float = 0.10,
) -> TextWindow:
    n = len(text)
    center = (site.start + site.end) // 2
    mention_len = site.end - site.start

    before_limit = budget // 2
    after_limit = budget - before_limit

    at_doc_start = center < int(n * doc_start_frac)
    no_boundary_before = _sentence_boundary_before(text, center, before_limit) is None

    if at_doc_start or no_boundary_before:
        win_start = max(0, site.start - min(200, budget // 10))
        win_end = min(n, site.end + max(budget - mention_len, budget - (center - win_start)))
    else:
        win_start = max(0, center - before_limit)
        win_end = min(n, center + after_limit)

    if win_end - win_start > budget:
        excess = (win_end - win_start) - budget
        trim_left = excess // 2
        trim_right = excess - trim_left
        if center - win_start < win_end - center:
            trim_left = min(trim_left, max(0, center - site.start))
            trim_right = excess - trim_left
        else:
            trim_right = min(trim_right, max(0, win_end - site.end))
            trim_left = excess - trim_right
        win_start += trim_left
        win_end -= trim_right

    win_start = _snap_start(text, win_start)
    win_end = _snap_end(text, win_end)

    if win_end <= win_start:
        win_end = min(n, win_start + budget)
    if win_end - win_start > budget:
        win_end = min(n, win_start + budget)

    return TextWindow(start=win_start, end=win_end)


def _shrink_windows_to_cap(
    text: str,
    windows: List[TextWindow],
    max_chars: int,
    sites: Sequence[MentionSite],
) -> List[TextWindow]:
    total = sum(w.end - w.start for w in windows)
    if total <= max_chars:
        return windows

    if len(windows) == 1:
        w = windows[0]
        return [TextWindow(w.start, min(w.start + max_chars, w.end))]

    while sum(w.end - w.start for w in windows) > max_chars and len(windows) > 1:
        max_gap = -1
        drop_i = len(windows) - 1
        for i in range(len(windows) - 1):
            gap = windows[i + 1].start - windows[i].end
            if gap > max_gap:
                max_gap = gap
                drop_i = i + 1 if gap >= (windows[i + 1].end - windows[i + 1].start) else i
        windows.pop(drop_i)

    total = sum(w.end - w.start for w in windows)
    if total <= max_chars:
        return windows

    scale = max_chars / total
    shrunk: List[TextWindow] = []
    for w in windows:
        length = w.end - w.start
        new_len = max(100, int(length * scale))
        center = (w.start + w.end) // 2
        ns = max(0, center - new_len // 2)
        ne = min(len(text), ns + new_len)
        shrunk.append(TextWindow(ns, ne))
    merged: List[TextWindow] = []
    for w in sorted(shrunk, key=lambda x: x.start):
        if merged and w.start <= merged[-1].end:
            merged[-1] = TextWindow(merged[-1].start, max(merged[-1].end, w.end))
        else:
            merged.append(w)
    return merged


def _format_focus_gene(
    row: Dict[str, Any],
    sites: Sequence[MentionSite],
    *,
    expected_id: str,
) -> Tuple[str, List[str]]:
    role = (row.get("paper_role") or "").strip().lower()
    query_id = str(row.get("query") or "").strip()
    target_id = str(row.get("target_id") or "").strip()

    if not sites:
        return f"{expected_id} (not found in text)", []

    id_order: List[str] = []
    for site in sorted(sites, key=lambda s: s.start):
        for cid in site.canonical_ids:
            if cid not in id_order:
                id_order.append(cid)

    if role in ("query", "target"):
        if len(id_order) == 1:
            return id_order[0], id_order
        return f"({', '.join(id_order)})", id_order

    query_ids = list(dict.fromkeys(cid for s in sites for cid in s.canonical_ids if cid == query_id))
    target_ids = list(dict.fromkeys(cid for s in sites for cid in s.canonical_ids if cid == target_id))

    if query_ids and target_ids:
        parts = []
        if query_ids:
            q = query_ids[0] if len(query_ids) == 1 else f"({', '.join(query_ids)})"
            parts.append(f"query:{q}")
        if target_ids:
            t = target_ids[0] if len(target_ids) == 1 else f"({', '.join(target_ids)})"
            parts.append(f"target:{t}")
        return f"({', '.join(parts)})", id_order

    if len(id_order) == 1:
        return id_order[0], id_order
    return f"({', '.join(id_order)})", id_order


def _assemble_excerpt(text: str, windows: Sequence[TextWindow]) -> str:
    if not windows:
        return ""
    parts = [text[w.start : w.end] for w in windows]
    return MENTION_SEPARATOR.join(parts)


def build_mention_excerpt(
    text: str,
    row: Dict[str, Any],
    *,
    max_chars: int = 3000,
    mention_cluster_gap: int = 50,
    min_mention_chars: int = 400,
    max_mention_chars: int = 8000,
    max_sites: int = 4,
    no_mention_fallback_chars: int = 1500,
    prepend_abstract_on_miss: bool = True,
) -> MentionExcerptResult:
    expected = _expected_gene_id(row)
    terms = collect_search_terms(row)
    hits = find_hits(text, terms)
    sites = cluster_mention_sites(hits, mention_cluster_gap)
    if max_sites > 0 and len(sites) > max_sites:
        sites = sites[:max_sites]

    if not sites:
        fallback = text[:no_mention_fallback_chars]
        if prepend_abstract_on_miss:
            abstract = _extract_abstract_snippet(text)
            if abstract:
                fallback = f"[Abstract]\n{abstract}\n\n[Head fallback]\n{fallback}"
        if len(text) > no_mention_fallback_chars:
            fallback += "\n\n[... no gene mentions found; head fallback ...]"
        return MentionExcerptResult(
            excerpt=fallback[:max_chars],
            focus_gene=f"{expected} (not found in text)",
            matched_canonical_ids=[],
            matched_terms=[],
            n_mentions=0,
            budget_per_mention=0,
            excerpt_mode=_EXCERPT_MODE_NO_MENTIONS,
            expected_gene_id=expected,
        )

    n = len(sites)
    budget_per = max(min_mention_chars, min(max_mention_chars, max_chars // n))

    windows: List[TextWindow] = []
    for site in sites:
        w = _expand_window(text, site, budget_per)
        if w.end - w.start > budget_per:
            w = TextWindow(w.start, min(len(text), w.start + budget_per))
        windows.append(w)

    total = sum(w.end - w.start for w in windows)
    if total > max_chars:
        windows = _shrink_windows_to_cap(text, windows, max_chars, sites)

    excerpt = _assemble_excerpt(text, windows)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n\n[... excerpt capped ...]"

    focus_gene, matched_ids = _format_focus_gene(row, sites, expected_id=expected)
    matched_literals = list(dict.fromkeys(h.literal for h in hits))

    return MentionExcerptResult(
        excerpt=excerpt,
        focus_gene=focus_gene,
        matched_canonical_ids=matched_ids,
        matched_terms=matched_literals,
        n_mentions=n,
        budget_per_mention=budget_per,
        excerpt_mode=_EXCERPT_MODE_MENTIONS,
        expected_gene_id=expected,
    )


def graded_paper_row(
    gp: Any,
    *,
    query_id: str,
    target_id: str,
    gene_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query_id,
        "target_id": target_id,
        "paper_role": getattr(gp, "paper_role", None) or "",
        "gene_context": gene_context or {},
    }


def build_synthesis_mention_excerpt(
    text: str,
    gp: Any,
    *,
    query_id: str,
    target_id: str,
    gene_context: Optional[Dict[str, Any]],
    max_chars: int,
    **kwargs: Any,
) -> MentionExcerptResult:
    row = graded_paper_row(
        gp, query_id=query_id, target_id=target_id, gene_context=gene_context
    )
    return build_mention_excerpt(text, row, max_chars=max_chars, **kwargs)


def format_excerpt_block(
    gp: Any,
    result: MentionExcerptResult,
) -> str:
    role = getattr(gp, "paper_role", None) or "unknown"
    matched = ", ".join(result.matched_terms[:12]) if result.matched_terms else "none"
    mode_line = (
        f"excerpt_mode: {result.excerpt_mode}"
        + (f" ({result.n_mentions} sites)" if result.n_mentions else "")
    )
    return (
        f"\n--- Excerpt: {gp.file_name} (role={role}) ---\n"
        f"genes_found_in_text: {result.focus_gene}\n"
        f"matched_terms: {matched}\n"
        f"{mode_line}\n"
        f"{result.excerpt}\n--- end excerpt ---\n"
    )
