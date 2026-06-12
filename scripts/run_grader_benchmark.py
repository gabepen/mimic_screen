#!/usr/bin/env python3
"""Re-grade benchmark papers against live grader endpoints and record timing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        raise SystemExit(f"Invalid manifest: {path}")
    return data


def _prepare_work_dir(run_dir: Path, sample_id: str, source_txt: str, file_name: str) -> Path:
    work = run_dir / "work" / sample_id
    work.mkdir(parents=True, exist_ok=True)
    link = work / file_name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.abspath(source_txt))
    return work


def _read_graded_paper(graded_path: Path, file_name: str) -> Optional[Dict[str, Any]]:
    if not graded_path.is_file():
        return None
    data = json.loads(graded_path.read_text(encoding="utf-8"))
    for row in data.get("graded_papers") or []:
        if isinstance(row, dict) and row.get("file_name") == file_name:
            return row
    return None


def _graded_ok(graded_path: Path, file_name: str) -> tuple[bool, Optional[str]]:
    if not graded_path.is_file():
        return False, "graded file missing"
    data = json.loads(graded_path.read_text(encoding="utf-8"))
    row = _read_graded_paper(graded_path, file_name)
    if not row:
        return False, "graded paper row missing"
    meta = data.get("grading_meta") or {}
    if row.get("model_output") is None:
        notes = str(row.get("notes") or "").strip()
        if notes:
            return False, notes[:500]
        n_exc = int(meta.get("n_llm_exceptions") or 0)
        if n_exc > 0:
            return False, f"n_llm_exceptions={n_exc}"
        return False, "model_output is null"
    if int(meta.get("n_llm_ok_structured") or 0) < 1:
        return False, "n_llm_ok_structured=0"
    return True, None


def _read_latest_llm_row(log_path: Path, file_name: str) -> Optional[Dict[str, Any]]:
    if not log_path.is_file():
        return None
    last: Optional[Dict[str, Any]] = None
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("file_name") == file_name and row.get("phase") == "chat_completion":
                last = row
    return last


def _grade_one(
    paper: Dict[str, Any],
    grader_url: str,
    grader_index: int,
    run_dir: Path,
    host_rubric_path: str,
    microbe_rubric_path: str,
    instructions: str,
    timeout: float,
) -> Dict[str, Any]:
    sample_id = str(paper["sample_id"])
    file_name = str(paper["file_name"])
    source_txt = str(paper["source_txt_path"])
    alignment_id = f"bench_{sample_id}"
    work_dir = _prepare_work_dir(run_dir, sample_id, source_txt, file_name)

    payload = {
        "alignment_id": alignment_id,
        "papers_dir": str(work_dir),
        "query": paper["query"],
        "target_id": paper["target_id"],
        "instructions": instructions,
        "output_root": str(run_dir),
        "host_rubric_path": host_rubric_path,
        "microbe_rubric_path": microbe_rubric_path,
        "synthesis_host": "",
        "synthesis_port": 9000,
        "constraints": {"temperature": 0.0},
    }

    url = f"{grader_url.rstrip('/')}/grade_alignment"
    started = time.perf_counter()
    err: Optional[str] = None
    http_status: Optional[int] = None
    response_body: Any = None
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        http_status = resp.status_code
        if resp.ok:
            response_body = resp.json()
        else:
            err = resp.text[:2000]
    except Exception as e:
        err = str(e)
    latency_sec = time.perf_counter() - started

    graded_path = run_dir / f"{alignment_id}_graded.json"
    new_row = _read_graded_paper(graded_path, file_name)
    llm_log = run_dir / "logs" / f"{alignment_id}_grader_llm.jsonl"
    llm_row = _read_latest_llm_row(llm_log, file_name)
    graded_ok, graded_err = _graded_ok(graded_path, file_name)

    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "alignment_id": paper["alignment_id"],
        "bench_alignment_id": alignment_id,
        "file_name": file_name,
        "paper_id": paper.get("paper_id"),
        "paper_role": paper.get("paper_role"),
        "grader_url": grader_url,
        "grader_index": grader_index,
        "latency_sec": round(latency_sec, 3),
        "http_status": http_status,
        "error": err,
        "baseline_relevance_grade": paper.get("baseline_relevance_grade"),
        "baseline_rubric_dimension_scores": paper.get("baseline_rubric_dimension_scores"),
        "new_relevance_grade": (new_row or {}).get("relevance_grade"),
        "new_rubric_dimension_scores": (new_row or {}).get("rubric_dimension_scores"),
        "new_rationale": (new_row or {}).get("rationale"),
        "baseline_rationale": paper.get("baseline_rationale"),
        "usage": (llm_row or {}).get("usage"),
        "content_empty": (llm_row or {}).get("content_empty"),
        "graded_ok": graded_ok,
        "graded_error": graded_err,
        "graded_path": str(graded_path),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--grader-urls", required=True, help="Semicolon-separated grader base URLs")
    p.add_argument("--host-rubric-path", required=True)
    p.add_argument("--microbe-rubric-path", required=True)
    p.add_argument("--instructions-file", required=True)
    p.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("VLLM_HTTP_READ_TIMEOUT", "1200")),
    )
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    run_dir = Path(manifest["run_dir"])
    papers: List[Dict[str, Any]] = list(manifest["papers"])
    grader_urls = [u.strip() for u in args.grader_urls.split(";") if u.strip()]
    if not grader_urls:
        raise SystemExit("No grader URLs provided")

    instructions = Path(args.instructions_file).read_text(encoding="utf-8")
    results_path = run_dir / "results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    assignments: List[tuple[Dict[str, Any], str, int]] = []
    for i, paper in enumerate(papers):
        url = grader_urls[i % len(grader_urls)]
        idx = grader_urls.index(url)
        assignments.append((paper, url, idx))

    max_workers = max(1, min(args.max_workers, len(grader_urls)))
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _grade_one,
                paper,
                url,
                idx,
                run_dir,
                args.host_rubric_path,
                args.microbe_rubric_path,
                instructions,
                args.timeout,
            ): paper["sample_id"]
            for paper, url, idx in assignments
        }
        for fut in as_completed(futures):
            sample_id = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"sample_id": sample_id, "error": str(e), "latency_sec": None}
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{row.get('sample_id')}: "
                f"grader={row.get('grader_index')} "
                f"latency={row.get('latency_sec')}s "
                f"status={row.get('http_status')} "
                f"graded_ok={row.get('graded_ok')}"
            )

    rows.sort(key=lambda r: str(r.get("sample_id")))
    n_graded_ok = sum(1 for r in rows if r.get("graded_ok"))
    summary = {
        "n_papers": len(rows),
        "n_http_ok": sum(1 for r in rows if r.get("http_status") == 200),
        "n_graded_ok": n_graded_ok,
        "n_llm_failed": sum(1 for r in rows if not r.get("graded_ok")),
        "results_jsonl": str(results_path),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if n_graded_ok < len(rows):
        failed = [r for r in rows if not r.get("graded_ok")]
        print(
            f"ERROR: {len(failed)}/{len(rows)} papers did not produce LLM grades",
            file=sys.stderr,
        )
        for row in failed[:5]:
            print(
                f"  {row.get('sample_id')}: {row.get('graded_error') or row.get('error')}",
                file=sys.stderr,
            )
    return 0 if n_graded_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
