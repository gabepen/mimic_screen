"""Shared helpers for download-only and download_node collect/manifest paths."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

from auto_lit_search.collect import _extract_doi_from_identifier

DOWNLOAD_MANIFEST_FILENAME = "download_manifest.jsonl"
IDENTIFIER_INDEX_FILENAME = "identifier_index.json"
DOWNLOAD_COMPLETE_FILENAME = "download_complete.json"
GLOBAL_OUTCOME_CACHE_FILENAME = "global_paper_outcome_cache.jsonl"


@dataclass
class AlignmentPaperClassification:
    to_fetch: List[Tuple[str, str]] = field(default_factory=list)
    global_inject: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)


def _canonical_alignment_text_key(fname: str) -> str:
    low = fname.lower()
    if not low.endswith(".txt"):
        return low
    m = re.match(r"^(.*__(?:query|target))(?:__[^.]*)?\.txt$", low)
    if m:
        return f"{m.group(1)}.txt"
    return low


def _canonical_paper_stem(name_stem: str) -> str:
    """Collapse channel-tagged artifact stems to query/target paper stem."""
    low = name_stem.lower()
    m = re.match(r"^(.*__(?:query|target))(?:__.*)?$", low)
    if m:
        return m.group(1)
    return low


def _paper_has_usable_text(papers_dir: str, pdf_basename: str) -> bool:
    """True when a non-empty .txt exists for this PDF (exact or canonical query/target stem)."""
    stem = pdf_basename
    if stem.lower().endswith(".pdf"):
        stem = os.path.splitext(stem)[0]
    exact = os.path.join(papers_dir, f"{stem}.txt")
    if os.path.isfile(exact) and os.path.getsize(exact) > 0:
        return True
    canon = _canonical_paper_stem(stem)
    if not os.path.isdir(papers_dir):
        return False
    for fname in os.listdir(papers_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(papers_dir, fname)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            continue
        if _canonical_paper_stem(os.path.splitext(fname)[0]) == canon:
            return True
    return False


def _paper_pair_key(paper_id: str, source: str) -> Tuple[str, str]:
    return (str(paper_id).strip(), str(source).strip())


def _alignment_id_for_pair(query_id: str, target: str) -> str:
    return f"{query_id}_{target}".replace("/", "_").replace(" ", "_")


def _load_search_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _alignment_paper_ids(alignment: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for pid in alignment.get("query_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "query"))
    for pid in alignment.get("target_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "target"))
    return out


def _load_idmap(csv_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            target = (row.get("target") or "").strip()
            if not query or not target:
                continue

            def _meta(prefix: str) -> Dict[str, Any]:
                syn_raw = (
                    row.get(f"{prefix}_synonyms")
                    or row.get(f"{prefix}_gene_synonyms")
                    or row.get(f"{prefix}_gene_aliases")
                    or row.get(f"{prefix}_aliases")
                    or ""
                )
                syns = [s.strip() for s in str(syn_raw).split(",") if s.strip()]
                return {
                    "uniprot_id": (row.get(prefix) or "").strip(),
                    "entrez_id": (row.get(f"{prefix}_entrez_id") or "").strip(),
                    "gene_name": (row.get(f"{prefix}_gene_name") or "").strip(),
                    "locus_tag": (row.get(f"{prefix}_locus_tag") or "").strip(),
                    "genbank_acc": (row.get(f"{prefix}_genbank_acc") or "").strip(),
                    "common_name": (row.get(f"{prefix}_common_name") or "").strip(),
                    "synonyms": syns,
                }

            key = f"{query}|{target}"
            out[key] = {
                "query_meta": _meta("query"),
                "target_meta": _meta("target"),
            }
    return out


def _load_download_manifest(manifest_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not manifest_path or not os.path.isfile(manifest_path):
        return out
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                pid = str(rec.get("paper_id") or "").strip()
                src = str(rec.get("source") or "").strip()
                if not pid:
                    continue
                out[_paper_pair_key(pid, src)] = rec
    except Exception:
        return out
    return out


def _manifest_file_stem(row: Dict[str, Any]) -> str:
    stem = str(row.get("file_stem") or "").strip()
    if stem:
        return stem
    tp = str(row.get("text_path") or "").strip()
    if tp:
        return os.path.splitext(os.path.basename(tp))[0]
    pp = str(row.get("pdf_path") or "").strip()
    if pp:
        return os.path.splitext(os.path.basename(pp))[0]
    return ""


def _outcome_cache_key(paper_id: str) -> str:
    doi = _extract_doi_from_identifier(paper_id)
    key = (doi or paper_id or "").strip().lower()
    return key


def _manifest_row_is_terminal_failed(row: Dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "failed"


def _manifest_row_retry_candidate(row: Dict[str, Any], papers_dir: str) -> bool:
    st = str(row.get("status") or "").strip().lower()
    if st == "failed":
        return True
    if st == "partial":
        stem = _manifest_file_stem(row)
        if not stem:
            return True
        pdf_path = os.path.join(papers_dir, "pdf", f"{stem}.pdf")
        return not (
            os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0
        )
    return False


def load_global_outcome_cache(logs_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(logs_dir, GLOBAL_OUTCOME_CACHE_FILENAME)
    out: Dict[str, Dict[str, Any]] = {}
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
                except Exception:
                    continue
                pid = str(rec.get("paper_id") or rec.get("doi") or "").strip()
                key = _outcome_cache_key(pid)
                if key:
                    out[key] = rec
    except Exception:
        return out
    return out


def _append_global_outcome_cache_entries(
    logs_dir: str, entries: List[Dict[str, Any]]
) -> None:
    if not entries:
        return
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, GLOBAL_OUTCOME_CACHE_FILENAME)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def record_global_outcomes_from_rows(
    logs_dir: str,
    rows: Iterable[Dict[str, Any]],
    global_cache: Dict[str, Dict[str, Any]],
) -> None:
    new_entries: List[Dict[str, Any]] = []
    ts = time.time()
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "failed":
            continue
        paper_id = str(row.get("paper_id") or "").strip()
        key = _outcome_cache_key(paper_id)
        if not key or key in global_cache:
            continue
        entry = {
            "paper_id": paper_id,
            "doi": row.get("doi") or _extract_doi_from_identifier(paper_id) or paper_id,
            "status": "failed",
            "message": row.get("message") or "",
            "updated_at": ts,
        }
        global_cache[key] = entry
        new_entries.append(entry)
    _append_global_outcome_cache_entries(logs_dir, new_entries)


def is_alignment_download_complete(papers_dir: str) -> bool:
    return os.path.isfile(os.path.join(papers_dir, DOWNLOAD_COMPLETE_FILENAME))


def write_alignment_download_complete(
    papers_dir: str, summary: Dict[str, Any]
) -> None:
    os.makedirs(papers_dir, exist_ok=True)
    path = os.path.join(papers_dir, DOWNLOAD_COMPLETE_FILENAME)
    payload = dict(summary)
    payload.setdefault("completed_at", time.time())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def synthetic_failed_manifest_row(
    paper_id: str, source: str, cache_row: Dict[str, Any]
) -> Dict[str, Any]:
    doi = cache_row.get("doi") or _extract_doi_from_identifier(paper_id) or paper_id
    return {
        "paper_id": paper_id,
        "source": source,
        "doi": doi,
        "file_stem": "",
        "status": "failed",
        "selected_text_source": "",
        "pdf_docling_required": False,
        "text_path": "",
        "pdf_path": "",
        "message": cache_row.get("message") or "global_outcome_cache_failed",
        "updated_at": float(cache_row.get("updated_at") or time.time()),
        "from_global_cache": True,
    }


def classify_alignment_papers(
    expected: List[Tuple[str, str]],
    manifest_map: Dict[Tuple[str, str], Dict[str, Any]],
    papers_dir: str,
    global_cache: Dict[str, Dict[str, Any]],
    *,
    retry_failed: bool = False,
    no_cache: bool = False,
) -> AlignmentPaperClassification:
    to_fetch: List[Tuple[str, str]] = []
    global_inject: Dict[Tuple[str, str], Dict[str, Any]] = {}
    stats = {
        "satisfied": 0,
        "terminal_failed": 0,
        "global_skipped": 0,
        "to_fetch": 0,
    }

    if no_cache:
        to_fetch = list(expected)
        stats["to_fetch"] = len(to_fetch)
        return AlignmentPaperClassification(
            to_fetch=to_fetch, global_inject=global_inject, stats=stats
        )

    for pid, src in expected:
        key = _paper_pair_key(pid, src)
        row = manifest_map.get(key)
        if row and _manifest_row_satisfied(row, papers_dir):
            stats["satisfied"] += 1
            continue
        if retry_failed and row and _manifest_row_retry_candidate(row, papers_dir):
            to_fetch.append((pid, src))
            continue
        if row and _manifest_row_is_terminal_failed(row) and not retry_failed:
            stats["terminal_failed"] += 1
            continue
        cache_key = _outcome_cache_key(pid)
        cache_row = global_cache.get(cache_key) if cache_key else None
        if (
            cache_row
            and str(cache_row.get("status") or "").strip().lower() == "failed"
            and not retry_failed
        ):
            stats["global_skipped"] += 1
            if row is None:
                global_inject[key] = synthetic_failed_manifest_row(pid, src, cache_row)
            continue
        to_fetch.append((pid, src))

    stats["to_fetch"] = len(to_fetch)
    return AlignmentPaperClassification(
        to_fetch=to_fetch, global_inject=global_inject, stats=stats
    )


def _manifest_row_satisfied(row: Dict[str, Any], papers_dir: str) -> bool:
    st = str(row.get("status") or "").strip().lower()
    if st == "failed":
        return False
    if st == "skipped":
        return True
    stem = _manifest_file_stem(row)
    if not stem:
        return False
    txt_path = os.path.join(papers_dir, f"{stem}.txt")
    if os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0:
        return True
    if st == "partial":
        pdf_path = os.path.join(papers_dir, "pdf", f"{stem}.pdf")
        return os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0
    return False


def _download_record_to_manifest_row(rec: Any, updated_at: float) -> Dict[str, Any]:
    d = getattr(rec, "details", None) or {}
    doi = d.get("doi") or _extract_doi_from_identifier(getattr(rec, "paper_id", "") or "")
    stem = str(d.get("file_stem") or "").strip()
    if not stem:
        tp = getattr(rec, "text_path", None) or ""
        if tp:
            stem = os.path.splitext(os.path.basename(str(tp)))[0]
    if not stem:
        pp = getattr(rec, "pdf_path", None) or ""
        if pp:
            stem = os.path.splitext(os.path.basename(str(pp)))[0]
    return {
        "paper_id": getattr(rec, "paper_id", "") or "",
        "source": getattr(rec, "source", "") or "",
        "doi": doi or "",
        "file_stem": stem,
        "status": getattr(rec, "status", "") or "",
        "selected_text_source": str(d.get("selected_text_source") or ""),
        "pdf_docling_required": bool(d.get("pdf_docling_required")),
        "text_path": getattr(rec, "text_path", None) or "",
        "pdf_path": getattr(rec, "pdf_path", None) or "",
        "message": getattr(rec, "message", None) or "",
        "updated_at": updated_at,
    }


def _merge_recs_into_manifest(
    existing: Dict[Tuple[str, str], Dict[str, Any]], recs: List[Any]
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    merged = dict(existing)
    ts = time.time()
    for rec in recs:
        key = _paper_pair_key(getattr(rec, "paper_id", ""), getattr(rec, "source", ""))
        merged[key] = _download_record_to_manifest_row(rec, ts)
    return merged


def _write_download_manifest_atomic(
    manifest_path: str, rows_by_key: Dict[Tuple[str, str], Dict[str, Any]]
) -> None:
    d = os.path.dirname(manifest_path) or "."
    os.makedirs(d, exist_ok=True)
    keys = sorted(rows_by_key.keys(), key=lambda k: (k[0], k[1]))
    fd, tmp = tempfile.mkstemp(
        prefix=".download_manifest_", suffix=".tmp", dir=d, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as wf:
            for k in keys:
                wf.write(json.dumps(rows_by_key[k], ensure_ascii=False) + "\n")
        os.replace(tmp, manifest_path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _emit_download_progress_summary(
    alignment_id: str,
    expected: List[Tuple[str, str]],
    manifest_map: Dict[Tuple[str, str], Dict[str, Any]],
    papers_dir: str,
    phase: str,
    classification_stats: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, str]]:
    total = len(expected)
    satisfied = 0
    missing: List[Tuple[str, str]] = []
    for pid, src in expected:
        key = _paper_pair_key(pid, src)
        row = manifest_map.get(key)
        if row and _manifest_row_satisfied(row, papers_dir):
            satisfied += 1
        else:
            missing.append((pid, src))
    cap = 20
    parts: List[str] = []
    for pid, src in missing[:cap]:
        doi = _extract_doi_from_identifier(pid) or pid
        parts.append(f"{doi}:{src}")
    tail = ""
    if len(missing) > cap:
        tail = f" …(+{len(missing) - cap} more)"
    preview = ",".join(parts) + tail
    extra = ""
    if classification_stats:
        extra = (
            f" terminal_failed={classification_stats.get('terminal_failed', 0)}"
            f" global_skipped={classification_stats.get('global_skipped', 0)}"
            f" to_fetch={classification_stats.get('to_fetch', 0)}"
        )
    logger.bind(download_progress=True).info(
        "alignment_download_summary alignment_id={} phase={} total_expected={} "
        "satisfied={} missing_count={}{} missing_preview=[{}]",
        alignment_id,
        phase,
        total,
        satisfied,
        len(missing),
        extra,
        preview,
    )
    return missing


def _manifest_pdf_basename(row: Dict[str, Any]) -> str:
    pp = str(row.get("pdf_path") or "").strip()
    if pp:
        return os.path.splitext(os.path.basename(pp))[0]
    stem = str(row.get("file_stem") or "").strip()
    if stem:
        return stem
    return ""


def _manifest_row_needs_docling(row: Dict[str, Any], papers_dir: str) -> bool:
    """True when manifest says this paper still needs PDF→text conversion."""
    if not row.get("pdf_docling_required"):
        st = str(row.get("status") or "").strip().lower()
        if st != "partial":
            return False
    base = _manifest_pdf_basename(row)
    if not base:
        return False
    if _paper_has_usable_text(papers_dir, base):
        return False
    pdf_path = os.path.join(papers_dir, "pdf", f"{base}.pdf")
    return os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0


def _infer_docling_required_from_manifest(
    manifest_map: Dict[Tuple[str, str], Dict[str, Any]],
    papers_dir: str,
) -> List[str]:
    required: List[str] = []
    seen: set[str] = set()
    for row in manifest_map.values():
        if not _manifest_row_needs_docling(row, papers_dir):
            continue
        base = _manifest_pdf_basename(row)
        if not base or base in seen:
            continue
        seen.add(base)
        required.append(base)
    return sorted(required)


def _infer_docling_required_basenames(
    papers_dir: str,
    manifest_map: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> List[str]:
    """
    Papers needing Docling: manifest rows flagged pdf_docling_required (or partial
    with PDF, no text). When a manifest exists, do not treat orphan S3 PDFs on disk
    as docling work — only API-fallback PDFs recorded in the manifest.
    """
    if manifest_map is None:
        manifest_path = os.path.join(papers_dir, DOWNLOAD_MANIFEST_FILENAME)
        manifest_map = (
            _load_download_manifest(manifest_path)
            if os.path.isfile(manifest_path)
            else {}
        )
    if manifest_map:
        return _infer_docling_required_from_manifest(manifest_map, papers_dir)
    return _infer_docling_required_basenames_from_disk(papers_dir)


def _infer_docling_required_basenames_from_disk(papers_dir: str) -> List[str]:
    pdf_dir = os.path.join(papers_dir, "pdf")
    if not os.path.isdir(pdf_dir):
        return []
    txt_canonical_stems: set[str] = set()
    if os.path.isdir(papers_dir):
        for fname in os.listdir(papers_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(papers_dir, fname)
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                continue
            txt_canonical_stems.add(
                _canonical_paper_stem(os.path.splitext(fname)[0])
            )
    required: List[str] = []
    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        if not os.path.isfile(pdf_path) or os.path.getsize(pdf_path) <= 0:
            continue
        base = os.path.splitext(fname)[0]
        if _canonical_paper_stem(base) in txt_canonical_stems:
            continue
        required.append(base)
    return sorted(required)


def _write_docling_eval_manifest(
    papers_dir: str,
    recs: List[Any],
    docling_required_basenames: List[str],
) -> None:
    eval_manifest_path = os.path.join(papers_dir, "docling_eval_manifest.jsonl")
    pdf_dir_m = os.path.join(papers_dir, "pdf")
    seen_pdf_bases: set[str] = set()
    with open(eval_manifest_path, "w", encoding="utf-8") as mf:
        for rrec in recs:
            mf.write(
                json.dumps(
                    {
                        "paper_id": rrec.paper_id,
                        "pdf_path": rrec.pdf_path,
                        "details": rrec.details or {},
                    }
                )
                + "\n"
            )
            pp = rrec.pdf_path
            if pp:
                seen_pdf_bases.add(os.path.splitext(os.path.basename(str(pp)))[0])
        for base in docling_required_basenames:
            if base in seen_pdf_bases:
                continue
            pdf_path = os.path.join(pdf_dir_m, f"{base}.pdf")
            mf.write(
                json.dumps(
                    {
                        "paper_id": base,
                        "pdf_path": pdf_path,
                        "details": {"pdf_docling_required": True},
                    }
                )
                + "\n"
            )


def build_paper_identifier_index(
    manifest_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    papers: List[Dict[str, Any]] = []
    for (paper_id, source), row in sorted(manifest_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        papers.append(
            {
                "paper_id": paper_id,
                "source": source,
                "doi": row.get("doi") or _extract_doi_from_identifier(paper_id) or paper_id,
                "file_stem": row.get("file_stem") or "",
                "status": row.get("status") or "",
                "text_path": row.get("text_path") or "",
                "pdf_path": row.get("pdf_path") or "",
                "selected_text_source": row.get("selected_text_source") or "",
                "pdf_docling_required": bool(row.get("pdf_docling_required")),
            }
        )
    return {"papers": papers}


def write_paper_identifier_index(papers_dir: str, index: Dict[str, Any]) -> None:
    path = os.path.join(papers_dir, IDENTIFIER_INDEX_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")
