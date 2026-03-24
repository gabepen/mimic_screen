"""
CPU node script: read search output, download papers per alignment, POST each batch to GPU.
Run inside the lit-download container with env: DATA_ROOT, PAPER_IDS_PATH, GPU_HOST, GPU_API_PORT, OUTPUT_ROOT.
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from loguru import logger

from auto_lit_search.collect import download_papers_to_dir

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)


def _load_search_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _alignment_paper_ids(alignment: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return a flat list of (paper_id, source) for this alignment.

    Source is either "query" or "target"; for lp-human controls we also
    carry richer gene metadata alongside the alignment, but this helper
    stays focused on IDs + coarse role.
    """
    out: List[Tuple[str, str]] = []
    for pid in alignment.get("query_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "query"))
    for pid in alignment.get("target_paper_dois") or []:
        if pid and str(pid).strip():
            out.append((str(pid).strip(), "target"))
    return out


def _load_idmap(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load mapping CSV with columns like:
        query,target,
        query_entrez_id,target_entrez_id,
        query_gene_name,target_gene_name,
        query_locus_tag,target_locus_tag,
        query_genbank_acc,target_genbank_acc,
        query_common_name,target_common_name
    and return a dict keyed by "query|target" with query/target metadata
    shaped for the GPU gene_context helper.
    """
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

            def _meta(prefix: str) -> Dict[str, str]:
                return {
                    "uniprot_id": (row.get(prefix) or "").strip(),
                    "entrez_id": (row.get(f"{prefix}_entrez_id") or "").strip(),
                    "gene_name": (row.get(f"{prefix}_gene_name") or "").strip(),
                    "locus_tag": (row.get(f"{prefix}_locus_tag") or "").strip(),
                    "genbank_acc": (row.get(f"{prefix}_genbank_acc") or "").strip(),
                    "common_name": (row.get(f"{prefix}_common_name") or "").strip(),
                }

            key = f"{query}|{target}"
            out[key] = {
                "query_meta": _meta("query"),
                "target_meta": _meta("target"),
            }
    return out


def _wait_health(
    service: str,
    gpu_url_base: str,
    timeout: int = 300,
    interval: int = 5,
) -> bool:
    url = f"{gpu_url_base.rstrip('/')}/healthz"
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info(
                    "{} health OK at {} (after {:.0f}s, {} tries)",
                    service,
                    url,
                    time.monotonic() - started,
                    attempt,
                )
                return True
        except Exception as e:
            if attempt == 1 or attempt % 6 == 0:
                logger.info(
                    "Waiting for {} at {} ({:.0f}s / {}s) last_error={!r}",
                    service,
                    url,
                    time.monotonic() - started,
                    timeout,
                    e,
                )
        time.sleep(interval)
    logger.error(
        "Timed out waiting for {} at {} after {}s",
        service,
        url,
        timeout,
    )
    return False


def run(
    paper_ids_path: str,
    data_root: str,
    gpu_host: str,
    gpu_port: int,
    output_root: str,
    instructions: str = "",
    instructions_file: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    request_timeout: int = 600,
    no_cache: bool = False,
    docling_host: str = "",
    docling_port: int = 9100,
    grader_host: str = "",
    grader_port: int = 9200,
    host_rubric_path: str = "",
    microbe_rubric_path: str = "",
) -> None:
    gpu_url_base = f"http://{gpu_host}:{gpu_port}"
    logger.info(
        "download_node: probing LLM GPU health at {}/healthz",
        gpu_url_base.rstrip("/"),
    )
    sys.stdout.flush()
    if not _wait_health("LLM GPU", gpu_url_base, timeout=300):
        raise RuntimeError(f"GPU node not healthy at {gpu_url_base}")

    docling_url_base = ""
    if docling_host:
        docling_url_base = f"http://{docling_host}:{docling_port}"
        logger.info(
            "download_node: probing Docling health at {}/healthz",
            docling_url_base.rstrip("/"),
        )
        sys.stdout.flush()
        if not _wait_health("Docling", docling_url_base, timeout=300):
            raise RuntimeError(f"Docling node not healthy at {docling_url_base}")
    if not grader_host:
        raise RuntimeError("GRADER_HOST is required for two-stage analysis flow")
    if not host_rubric_path or not microbe_rubric_path:
        raise RuntimeError(
            "HOST_RUBRIC_PATH and MICROBE_RUBRIC_PATH are required for grading"
        )
    grader_url_base = f"http://{grader_host}:{grader_port}"
    logger.info(
        "download_node: probing Grader health at {}/healthz",
        grader_url_base.rstrip("/"),
    )
    if not _wait_health("Grader", grader_url_base, timeout=300):
        raise RuntimeError(f"Grader node not healthy at {grader_url_base}")

    instructions_text = instructions
    if instructions_file and os.path.isfile(instructions_file):
        with open(instructions_file, "r", encoding="utf-8") as f:
            instructions_text = f.read().strip()

    data = _load_search_json(paper_ids_path)
    collection_org = os.environ.get("COLLECTION_ORG", "ucsc").strip() or "ucsc"
    collection_auth_scope = (
        os.environ.get("COLLECTION_AUTH_SCOPE", "email_only").strip() or "email_only"
    )
    collector_email = os.environ.get("COLLECTOR_EMAIL", "").strip()
    if (
        collection_org.lower() == "ucsc"
        and collection_auth_scope.lower() == "email_only"
        and not collector_email
    ):
        raise RuntimeError(
            "COLLECTOR_EMAIL is required for UCSC email_only collection mode."
        )

    logger.info(
        "Collection mode org={} scope={} collector_email_set={}",
        collection_org,
        collection_auth_scope,
        "yes" if bool(collector_email) else "no",
    )

    _mw_raw = os.environ.get("COLLECT_MAX_WORKERS", "2").strip() or "2"
    try:
        collect_max_workers = max(1, min(16, int(_mw_raw)))
    except ValueError:
        collect_max_workers = 2
    collect_disable_s2 = os.environ.get(
        "COLLECT_DISABLE_SEMANTIC_SCHOLAR", ""
    ).strip().lower() in ("1", "true", "yes")
    logger.info(
        "Collect parallelism max_workers={} semantic_scholar={}",
        collect_max_workers,
        "off" if collect_disable_s2 else "on",
    )

    # Optional: mapping CSV for richer gene identifiers; provided via env.
    idmap_path = os.environ.get("IDMAP_CSV", "")
    idmap: Dict[str, Dict[str, Any]] = {}
    if idmap_path:
        idmap = _load_idmap(idmap_path)
    papers_base = os.path.join(data_root, "papers")
    os.makedirs(papers_base, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    session = requests.Session()
    pmcid_cache: Dict[str, str | None] = {}
    total = 0
    for query_id, alignments in data.items():
        if not isinstance(alignments, list):
            continue
        for al in alignments:
            target = al.get("target") or ""
            alignment_id = f"{query_id}_{target}".replace("/", "_").replace(" ", "_")
            paper_ids_src = _alignment_paper_ids(al)
            if not paper_ids_src:
                logger.warning(f"Alignment {alignment_id}: no paper IDs")
                continue

            papers_dir = os.path.join(papers_base, alignment_id)
            os.makedirs(papers_dir, exist_ok=True)

            logger.info(f"Downloading {len(paper_ids_src)} papers for {alignment_id}")
            recs = download_papers_to_dir(
                paper_ids_src,
                papers_dir,
                session=session,
                pmcid_cache=pmcid_cache,
                no_cache=no_cache,
                force_pdfs=True,
                prefer_pdf_text=False,
                collection_org=collection_org,
                auth_scope=collection_auth_scope,
                collector_email=collector_email or None,
                max_workers=collect_max_workers,
                disable_semantic_scholar=collect_disable_s2,
            )
            has_text = any(r.text_path for r in recs)
            has_pdf = any(r.pdf_path for r in recs)
            n_docling_required = sum(
                1 for r in recs if ((r.details or {}).get("pdf_docling_required"))
            )
            n_text = sum(1 for r in recs if r.text_path)
            n_pdf = sum(1 for r in recs if r.pdf_path)
            n_ok = sum(1 for r in recs if r.status == "ok")
            n_failed = sum(1 for r in recs if r.status == "failed")
            logger.info(
                f"Alignment {alignment_id}: downloaded_papers={len(recs)} "
                f"text_files={n_text} pdf_files={n_pdf} "
                f"docling_required={n_docling_required} ok={n_ok} failed={n_failed}"
            )

            gene_context: Dict[str, Any] | None = None
            # Prefer inline meta if present, otherwise fall back to ID map env.
            query_meta = al.get("query_meta")
            target_meta = al.get("target_meta")
            if isinstance(query_meta, dict) or isinstance(target_meta, dict):
                gene_context = {
                    "query": query_meta or {},
                    "target": target_meta or {},
                }
            elif idmap:
                key = f"{query_id}|{target}"
                meta = idmap.get(key)
                if meta:
                    gene_context = {
                        "query": meta.get("query_meta") or {},
                        "target": meta.get("target_meta") or {},
                    }

            # Convert only PDFs selected over XML through Docling.
            if docling_url_base and n_docling_required > 0 and has_pdf:
                pdf_dir = os.path.join(papers_dir, "pdf")
                eval_manifest_path = os.path.join(
                    papers_dir, "docling_eval_manifest.jsonl"
                )
                try:
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
                except Exception as e:
                    logger.warning(
                        f"Alignment {alignment_id}: could not write docling manifest: {e}"
                    )
                    eval_manifest_path = ""

                docling_payload: Dict[str, Any] = {
                    "alignment_id": alignment_id,
                    "pdf_dir": pdf_dir,
                    "papers_dir": papers_dir,
                    "query": query_id,
                    "target_id": target,
                    "constraints": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    "instructions": instructions_text
                    or (
                        "Analyze the paper excerpt for relevance to the genes in this alignment. "
                        "First give a brief justification (2-4 sentences). "
                        "Then output a single line: relevance_score=<float between 0 and 1>."
                    ),
                    "output_root": output_root,
                    "gene_context": gene_context,
                    "analysis_host": gpu_host,
                    "analysis_port": gpu_port,
                    "evaluation_manifest_path": eval_manifest_path or None,
                    "call_analysis": False,
                }
                try:
                    r = session.post(
                        f"{docling_url_base}/convert_alignment",
                        json=docling_payload,
                        timeout=request_timeout,
                    )
                    r.raise_for_status()
                    out = r.json()
                    logger.info(
                        f"Alignment {alignment_id}: docling_status={out.get('status')} "
                        f"papers_dir={out.get('papers_dir')} (analysis deferred to GPU step)"
                    )
                except requests.RequestException as e:
                    logger.error(f"Alignment {alignment_id}: Docling request failed: {e}")
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        try:
                            logger.error(
                                f"Alignment {alignment_id}: Docling response: {resp.text[:800]}"
                            )
                        except Exception:
                            pass

            if not any(
                fname.endswith(".txt")
                for fname in os.listdir(papers_dir)
                if os.path.isfile(os.path.join(papers_dir, fname))
            ):
                logger.warning(
                    f"Alignment {alignment_id}: no text files available after collection/conversion; skipping GPU"
                )
                continue

            # Final handoff: CPU -> grader -> synthesis GPU.
            payload: Dict[str, Any] = {
                "alignment_id": alignment_id,
                "papers_dir": papers_dir,
                "query": query_id,
                "target_id": target,
                "constraints": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                "instructions": instructions_text
                or (
                    "Analyze the paper excerpt for relevance to the genes in this alignment. "
                    "First give a brief justification (2-4 sentences). "
                    "Then output a single line: relevance_score=<float between 0 and 1>."
                ),
                "output_root": output_root,
                "host_rubric_path": host_rubric_path,
                "microbe_rubric_path": microbe_rubric_path,
                "synthesis_host": gpu_host,
                "synthesis_port": gpu_port,
            }
            if gene_context is not None:
                payload["gene_context"] = gene_context

            try:
                r = session.post(
                    f"{grader_url_base}/grade_alignment",
                    json=payload,
                    timeout=request_timeout,
                )
                r.raise_for_status()
                out = r.json()
                logger.info(
                    f"Alignment {alignment_id}: graded+synthesis {out.get('status')} -> {out.get('results_path')}"
                )
                total += 1
            except requests.RequestException as e:
                logger.error(f"Alignment {alignment_id}: grader request failed: {e}")
                raise

    logger.info(f"Submitted {total} alignments to GPU node.")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "CPU download node: download papers, POST to Docling or GPU analysis node"
        )
    )
    p.add_argument("--paper-ids", required=True, help="Search output JSON path")
    p.add_argument("--data-root", required=True, help="Shared data root")
    p.add_argument("--gpu-host", required=True, help="GPU node hostname")
    p.add_argument("--gpu-port", type=int, default=9000)
    p.add_argument("--docling-host", default="", help="Docling node hostname")
    p.add_argument("--docling-port", type=int, default=9100)
    p.add_argument("--grader-host", default=os.environ.get("GRADER_HOST", ""))
    p.add_argument("--grader-port", type=int, default=int(os.environ.get("GRADER_API_PORT", "9200")))
    p.add_argument("--host-rubric-path", default=os.environ.get("HOST_RUBRIC_PATH", ""))
    p.add_argument("--microbe-rubric-path", default=os.environ.get("MICROBE_RUBRIC_PATH", ""))
    p.add_argument("--output-root", required=True, help="Results output root")
    p.add_argument("--instructions", default="", help="Inline prompt/instructions")
    p.add_argument("--instructions-file", default="", help="Path to prompt/instructions file")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    run(
        paper_ids_path=args.paper_ids,
        data_root=args.data_root,
        gpu_host=args.gpu_host,
        gpu_port=args.gpu_port,
        output_root=args.output_root,
        instructions=args.instructions,
        instructions_file=args.instructions_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        no_cache=args.no_cache,
        docling_host=args.docling_host,
        docling_port=args.docling_port,
        grader_host=args.grader_host,
        grader_port=args.grader_port,
        host_rubric_path=args.host_rubric_path,
        microbe_rubric_path=args.microbe_rubric_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
