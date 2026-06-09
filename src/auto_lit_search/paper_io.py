"""Shared paper directory listing, deduplication, and text reads."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

TEXT_EXTENSIONS = (".txt",)
MAX_PAPER_CHARS = 120000
DOWNLOAD_MANIFEST_FILENAME = "download_manifest.jsonl"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dedupe_keep_order(items: List[str]) -> List[str]:
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


def as_list(v: Any) -> List[str]:
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


def extract_paper_role(fname: str) -> Optional[str]:
    low = fname.lower()
    if "__query" in low:
        return "query"
    if "__target" in low:
        return "target"
    return None


def list_paper_files(papers_dir: str) -> List[str]:
    files = sorted(
        f
        for f in os.listdir(papers_dir)
        if os.path.isfile(os.path.join(papers_dir, f))
        and (f.endswith(TEXT_EXTENSIONS) or not f.endswith((".pdf", ".xml")))
    )
    labeled = [f for f in files if "__query" in f.lower() or "__target" in f.lower()]
    selected = labeled if labeled else files
    if not labeled:
        return selected

    grouped: Dict[str, List[str]] = {}
    for f in selected:
        low = f.lower()
        m = re.match(r"^(.*__(?:query|target))(?:__[^.]*)?\.txt$", low)
        key = f"{m.group(1)}.txt" if m else low
        grouped.setdefault(key, []).append(f)

    deduped: List[str] = []
    for key in sorted(grouped.keys()):
        candidates = grouped[key]
        canonical = [
            c
            for c in candidates
            if c.lower().endswith("__query.txt") or c.lower().endswith("__target.txt")
        ]
        suffixed = [c for c in candidates if c not in canonical]
        if suffixed:
            deduped.append(sorted(suffixed)[0])
        elif canonical:
            deduped.append(sorted(canonical)[0])
        else:
            deduped.append(sorted(candidates)[0])
    return deduped


def read_text(path: str, max_chars: int = MAX_PAPER_CHARS) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        s = f.read()
    if len(s) > max_chars:
        s = s[:max_chars] + "\n\n[truncated]"
    return s


def paper_id_by_artifact_basename(papers_dir: str) -> Dict[str, str]:
    path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = str(rec.get("doi") or rec.get("paper_id") or "").strip()
                if not pid:
                    continue
                for key in ("text_path", "pdf_path"):
                    p = str(rec.get(key) or "").strip()
                    if not p:
                        continue
                    base = os.path.basename(p)
                    out[base] = pid
                    if base.lower().endswith(".pdf"):
                        out.setdefault(f"{os.path.splitext(base)[0]}.txt", pid)
    except OSError:
        pass
    return out


def gene_terms(meta: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
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
        syns.extend(as_list(meta.get(k)))
    syns = dedupe_keep_order(syns)
    syns = [s for s in syns if s.lower() not in {symbol.lower(), common_name.lower()}]
    return {
        "symbol": symbol,
        "common_name": common_name or "none",
        "synonyms": syns,
    }


def identification_terms_block(
    query: str,
    target_id: str,
    gene_context: Optional[Dict[str, Any]],
) -> str:
    query_meta = (gene_context or {}).get("query") or {}
    target_meta = (gene_context or {}).get("target") or {}
    q = gene_terms(query_meta, query)
    t = gene_terms(target_meta, target_id)
    q_syn = ", ".join(q["synonyms"]) if q["synonyms"] else "none"
    t_syn = ", ".join(t["synonyms"]) if t["synonyms"] else "none"
    return (
        "Paper identification terms used in retrieval (prioritize symbol/common name; "
        "use synonyms as alternate mentions):\n"
        f"- Query gene ({query}): symbol={q['symbol']}; common_name={q['common_name']}; synonyms={q_syn}\n"
        f"- Target gene ({target_id}): symbol={t['symbol']}; common_name={t['common_name']}; synonyms={t_syn}\n"
    )
