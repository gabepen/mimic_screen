"""Section-aware paper excerpt builder for grader prompts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_TRUNCATION_MARKER = "\n\n[excerpt truncated for grading]"
_OMITTED_MARKER = "\n\n[... omitted ...]\n\n"

_SECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"^abstract\s*$", re.IGNORECASE)),
    ("introduction", re.compile(r"^introduction\s*$", re.IGNORECASE)),
    ("results", re.compile(r"^results\s*$", re.IGNORECASE)),
    ("methods", re.compile(r"^(methods|materials\s+and\s+methods)\s*$", re.IGNORECASE)),
    ("discussion", re.compile(r"^discussion\s*$", re.IGNORECASE)),
    ("references", re.compile(r"^(references|bibliography)\s*$", re.IGNORECASE)),
)

_EVIDENCE_KINDS = ("results", "methods", "discussion")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return default


def _long_threshold() -> int:
    return _env_positive_int("GRADER_EXCERPT_LONG_THRESHOLD", 50000)


def _max_chars() -> int:
    return _env_positive_int("GRADER_EXCERPT_MAX_CHARS", 80000)


def _preamble_chars() -> int:
    return _env_positive_int("GRADER_EXCERPT_PREAMBLE_CHARS", 5000)


@dataclass(frozen=True)
class _SectionSpan:
    kind: str
    start: int
    end: int

    def text(self, full: str) -> str:
        return full[self.start : self.end]


@dataclass
class ExcerptMeta:
    excerpt: str
    raw_char_len: int
    excerpt_char_len: int
    section_trim_applied: bool
    references_dropped: bool
    intro_trimmed: bool


def _classify_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    for kind, pattern in _SECTION_PATTERNS:
        if pattern.match(stripped):
            return kind
    return None


def _detect_sections(text: str) -> List[_SectionSpan]:
    sections: List[_SectionSpan] = []
    line_start = 0
    for line in text.splitlines(keepends=True):
        kind = _classify_line(line)
        if kind is not None:
            sections.append(_SectionSpan(kind=kind, start=line_start, end=len(text)))
            if len(sections) > 1:
                sections[-2] = _SectionSpan(
                    kind=sections[-2].kind,
                    start=sections[-2].start,
                    end=line_start,
                )
        line_start += len(line)
    if sections:
        sections[-1] = _SectionSpan(
            kind=sections[-1].kind,
            start=sections[-1].start,
            end=len(text),
        )
    return sections


def _join_parts(parts: List[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned)


def _trim_tail(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 0:
        return ""
    return text[:max_len].rstrip()


def _preamble_slice(text: str, sections: List[_SectionSpan]) -> str:
    limit = _preamble_chars()
    abstract = next((s for s in sections if s.kind == "abstract"), None)
    if abstract is not None:
        return text[: abstract.end].strip()[:limit]
    first_start = sections[0].start if sections else len(text)
    return text[: min(first_start, limit)].strip()


def _compose_from_sections(text: str, sections: List[_SectionSpan]) -> Tuple[str, bool, bool]:
    references_dropped = any(s.kind == "references" for s in sections)
    evidence = [s for s in sections if s.kind in _EVIDENCE_KINDS]
    intro = next((s for s in sections if s.kind == "introduction"), None)

    preamble = _preamble_slice(text, sections)
    parts: List[str] = [preamble]
    for sec in sorted(evidence, key=lambda s: s.start):
        parts.append(sec.text(text))
    if intro is not None:
        parts.append(intro.text(text))

    composed = _join_parts(parts)
    intro_trimmed = False

    if len(composed) <= _max_chars():
        return composed, references_dropped, intro_trimmed

    parts = [preamble]
    for sec in sorted(evidence, key=lambda s: s.start):
        parts.append(sec.text(text))
    composed = _join_parts(parts)
    if intro is not None:
        intro_trimmed = True

    if len(composed) <= _max_chars():
        return composed, references_dropped, intro_trimmed

    preamble_budget = max(0, _max_chars() - len(_join_parts([s.text(text) for s in evidence])))
    preamble = _trim_tail(preamble, preamble_budget)
    parts = [preamble] + [sec.text(text) for sec in sorted(evidence, key=lambda s: s.start)]
    composed = _join_parts(parts)
    intro_trimmed = True

    if len(composed) <= _max_chars():
        return composed, references_dropped, intro_trimmed

    discussion = next((s for s in sections if s.kind == "discussion"), None)
    methods = next((s for s in sections if s.kind == "methods"), None)
    results = next((s for s in sections if s.kind == "results"), None)

    def _with_evidence(
        discussion_text: str,
        methods_text: str,
        results_text: str,
    ) -> str:
        chunks = [preamble]
        if results_text:
            chunks.append(results_text)
        if methods_text:
            chunks.append(methods_text)
        if discussion_text:
            chunks.append(discussion_text)
        return _join_parts(chunks)

    disc_txt = discussion.text(text) if discussion else ""
    meth_txt = methods.text(text) if methods else ""
    res_txt = results.text(text) if results else ""
    composed = _with_evidence(disc_txt, meth_txt, res_txt)

    if len(composed) > _max_chars() and discussion:
        overhead = len(_with_evidence("", meth_txt, res_txt))
        disc_txt = _trim_tail(disc_txt, max(0, _max_chars() - overhead))
        composed = _with_evidence(disc_txt, meth_txt, res_txt)

    if len(composed) > _max_chars() and methods:
        overhead = len(_with_evidence(disc_txt, "", res_txt))
        meth_txt = _trim_tail(meth_txt, max(0, _max_chars() - overhead))
        composed = _with_evidence(disc_txt, meth_txt, res_txt)

    return composed, references_dropped, intro_trimmed


def _fallback_excerpt(text: str) -> Tuple[str, bool, bool]:
    head = 55000
    tail = 25000
    if len(text) <= head + tail:
        excerpt = text
    else:
        excerpt = text[:head] + _OMITTED_MARKER + text[-tail:]
    references_dropped = "references" in excerpt.lower()
    intro_trimmed = False

    if len(excerpt) > _max_chars():
        excerpt = excerpt[: _max_chars()].rstrip()
        intro_trimmed = True
    return excerpt, references_dropped, intro_trimmed


def build_grader_excerpt_with_meta(text: str) -> ExcerptMeta:
    raw_len = len(text)
    if raw_len <= _long_threshold():
        return ExcerptMeta(
            excerpt=text,
            raw_char_len=raw_len,
            excerpt_char_len=raw_len,
            section_trim_applied=False,
            references_dropped=False,
            intro_trimmed=False,
        )

    sections = _detect_sections(text)
    if sections:
        excerpt, references_dropped, intro_trimmed = _compose_from_sections(text, sections)
    else:
        excerpt, references_dropped, intro_trimmed = _fallback_excerpt(text)

    trim_applied = len(excerpt) < raw_len or references_dropped or intro_trimmed
    if trim_applied and _TRUNCATION_MARKER not in excerpt:
        if len(excerpt) + len(_TRUNCATION_MARKER) > _max_chars():
            excerpt = _trim_tail(excerpt, _max_chars() - len(_TRUNCATION_MARKER))
        excerpt = excerpt.rstrip() + _TRUNCATION_MARKER

    return ExcerptMeta(
        excerpt=excerpt,
        raw_char_len=raw_len,
        excerpt_char_len=len(excerpt),
        section_trim_applied=trim_applied,
        references_dropped=references_dropped,
        intro_trimmed=intro_trimmed,
    )


def build_grader_excerpt(text: str) -> str:
    return build_grader_excerpt_with_meta(text).excerpt


def excerpt_meta_dict(meta: ExcerptMeta) -> Dict[str, object]:
    return {
        "excerpt_char_len": meta.raw_char_len,
        "excerpt_out_char_len": meta.excerpt_char_len,
        "section_trim_applied": meta.section_trim_applied,
        "references_dropped": meta.references_dropped,
        "intro_trimmed": meta.intro_trimmed,
    }
