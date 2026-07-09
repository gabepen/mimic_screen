#!/usr/bin/env python3
"""
Submit Slurm jobs for the auto_lit_search pipeline (Docling + Grader + LLM GPUs, then CPU download).

Usage:
  python run_pipeline_slurm.py --config pipelines/auto_lit/configs/lp_human_stage2.yaml
  python run_pipeline_slurm.py --paper-ids <search_output.json> [options]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.env_config import auto_lit_pipeline_root, repo_root as get_repo_root, slurm_mail_user
from auto_lit_search.pipeline_config import (  # noqa: E402
    Stage2Config,
    apply_runtime_env,
    load_stage2_config,
)

_PUBLISHER_CPU_ENV_KEYS: tuple[str, ...] = (
    "ELSEVIER_API_KEY",
    "ELS_API_KEY",
    "ELSEVIER_INSTTOKEN",
    "ELS_INSTTOKEN",
    "TDM_API_TOKEN",
    "WILEY_TDM_API_TOKEN",
    "CROSSREF_MAILTO",
)


def _publisher_env_from_os() -> dict[str, str]:
    out: dict[str, str] = {}
    for k in _PUBLISHER_CPU_ENV_KEYS:
        v = os.environ.get(k, "").strip()
        if v:
            out[k] = v
    return out


def _wait_healthz(
    service: str,
    host: str,
    port: int,
    *,
    job_id: str | None = None,
    timeout: int = 900,
    interval: int = 5,
) -> tuple[bool, str]:
    deadline = time.monotonic() + max(60, timeout)
    current_host = (host or "").strip()
    started = time.monotonic()
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if job_id:
            node = _try_get_node_name(job_id)
            if node:
                current_host = node
        if not current_host:
            time.sleep(interval)
            continue
        url = f"http://{current_host}:{port}/healthz"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    print(
                        f"{service} health OK at {url} "
                        f"(after {time.monotonic() - started:.0f}s, {attempt} tries)"
                    )
                    return True, current_host
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == 1 or attempt % 12 == 0:
                print(
                    f"Waiting for {service} at {url} "
                    f"({time.monotonic() - started:.0f}s / {timeout}s): {e!r}",
                    file=sys.stderr,
                )
        time.sleep(interval)
    print(
        f"Timed out waiting for {service} /healthz on {current_host}:{port} after {timeout}s",
        file=sys.stderr,
    )
    return False, current_host


def _sbatch(
    script_path: str,
    env: dict,
    dependency: str | None = None,
    dependency_kind: Literal["afterok", "after"] = "afterok",
    log_path: str | None = None,
) -> str:
    cmd = ["sbatch", "--parsable"]
    mail_user = slurm_mail_user()
    if mail_user:
        cmd.extend(["--mail-type=BEGIN,END,FAIL", f"--mail-user={mail_user}"])
    if dependency:
        cmd.extend([f"--dependency={dependency_kind}:{dependency}"])
    if log_path:
        cmd.extend(["--output", log_path])
    export_pairs = [f"{k}={v}" for k, v in env.items() if v is not None]
    if export_pairs:
        cmd.append("--export=ALL," + ",".join(export_pairs))
    cmd.append(script_path)
    out = subprocess.check_output(cmd, text=True)
    return out.strip().split(";")[0]


def _scontrol_show_job(job_id: str) -> str:
    return subprocess.check_output(
        ["scontrol", "show", "job", job_id],
        text=True,
    )


def _node_from_scontrol_job(raw: str) -> str | None:
    invalid = {"", "none", "(null)", "null", "n/a", "unknown"}

    def _ok(name: str) -> bool:
        return bool(name) and name.lower() not in invalid

    for m in re.finditer(r"\bNodeList=(\S+)", raw):
        node = m.group(1).strip()
        if _ok(node):
            return node
    for m in re.finditer(r"\bSchedNodeList=(\S+)", raw):
        node = m.group(1).strip()
        if _ok(node):
            return node
    return None


def _try_get_node_name(job_id: str) -> str | None:
    try:
        raw = _scontrol_show_job(job_id)
        return _node_from_scontrol_job(raw)
    except Exception:
        return None


def _get_node_name(job_id: str, max_wait: int = 12 * 60 * 60) -> str | None:
    node = _try_get_node_name(job_id)
    if node:
        return node
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        try:
            raw = _scontrol_show_job(job_id)
            node = _node_from_scontrol_job(raw)
            if node:
                return node
            st = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )
            if "RUNNING" in (st.stdout or ""):
                continue
        except Exception:
            pass
        time.sleep(5)
    return None


def _defaults_from_stage2(cfg: Stage2Config) -> dict[str, Any]:
    return {
        "paper_ids": str(cfg.paper_ids_json),
        "data_root": str(cfg.data_root),
        "output_root": str(cfg.output_root),
        "model_dir": str(cfg.cluster.model_dir),
        "grader_model_dir": str(cfg.cluster.grader_model_dir),
        "gpu_image": cfg.cluster.gpu_image,
        "docling_image": cfg.cluster.docling_image,
        "grader_image": cfg.cluster.grader_image,
        "cpu_image": cfg.cluster.cpu_image,
        "gpu_script": str(cfg.slurm.gpu_script),
        "docling_script": str(cfg.slurm.docling_script),
        "grader_script": str(cfg.slurm.grader_script),
        "cpu_script": str(cfg.slurm.cpu_script),
        "gpu_port": cfg.slurm.gpu_port,
        "docling_port": cfg.slurm.docling_port,
        "grader_port": cfg.slurm.grader_port,
        "num_grader_nodes": cfg.slurm.num_grader_nodes,
        "num_synthesis_nodes": cfg.slurm.num_synthesis_nodes,
        "instructions_file": str(cfg.instructions_file),
        "idmap_csv": str(cfg.idmap_csv),
        "host_rubric_path": str(cfg.host_rubric),
        "microbe_rubric_path": str(cfg.microbe_rubric),
        "collection_org": cfg.collection.org,
        "collection_auth_scope": cfg.collection.auth_scope,
        "collector_email": cfg.collection.collector_email,
        "collect_max_workers": cfg.collection.max_workers,
        "collect_disable_semantic_scholar": cfg.collection.disable_semantic_scholar,
        "no_wait": cfg.slurm.no_wait,
        "repo_root": str(cfg.cluster.repo_root),
        "pipeline_root": str(cfg.cluster.pipeline_root),
        "logs_root": str(cfg.cluster.logs_dir),
    }


def run_pipeline(args: argparse.Namespace) -> int:
    grader_model_dir = args.grader_model_dir or args.model_dir

    _mw_env = os.environ.get("COLLECT_MAX_WORKERS", "").strip()
    if _mw_env.isdigit():
        args.collect_max_workers = max(1, min(16, int(_mw_env)))
    if os.environ.get("COLLECT_DISABLE_SEMANTIC_SCHOLAR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        args.collect_disable_semantic_scholar = True

    repo_root_path = getattr(args, "repo_root", None) or str(get_repo_root())
    pipeline_root = getattr(args, "pipeline_root", None) or str(auto_lit_pipeline_root())
    output_root = args.output_root or os.path.join(args.data_root, "llm_results")
    logs_root = getattr(args, "logs_root", None) or os.path.join(args.data_root, "logs")
    gpu_script = args.gpu_script or os.path.join(pipeline_root, "slurm", "gpu_llm_node.slurm")
    docling_script = args.docling_script or os.path.join(
        pipeline_root, "slurm", "gpu_docling_node.slurm"
    )
    grader_script = args.grader_script or os.path.join(
        pipeline_root, "slurm", "gpu_grader_node.slurm"
    )
    cpu_script = args.cpu_script or os.path.join(pipeline_root, "slurm", "cpu_download_node.slurm")

    gpu_image = args.gpu_image or os.path.join(pipeline_root, "containers", "lit-llm-0.1.0.sif")
    docling_image = args.docling_image or os.path.join(
        pipeline_root, "containers", "docling-0.1.0.sif"
    )
    grader_image = args.grader_image or gpu_image
    cpu_image = args.cpu_image or os.path.join(pipeline_root, "containers", "lit-download.sif")

    def _image_ok(path: str) -> bool:
        return os.path.isfile(path) or path.startswith("docker://") or path.startswith(
            "library://"
        )

    if not _image_ok(gpu_image):
        print(f"GPU image not found: {gpu_image}. Pass --gpu-image.", file=sys.stderr)
        return 1
    if not _image_ok(cpu_image):
        print(f"CPU image not found: {cpu_image}. Pass --cpu-image.", file=sys.stderr)
        return 1
    if not _image_ok(grader_image):
        print(f"Grader image not found: {grader_image}. Pass --grader-image.", file=sys.stderr)
        return 1

    for label, script in (
        ("GPU", gpu_script),
        ("Docling", docling_script),
        ("CPU", cpu_script),
        ("Grader", grader_script),
    ):
        if not os.path.isfile(script):
            print(f"{label} script not found: {script}", file=sys.stderr)
            return 1

    if args.num_grader_nodes < 1:
        print("--num-grader-nodes must be >= 1", file=sys.stderr)
        return 1
    if args.num_synthesis_nodes < 1:
        print("--num-synthesis-nodes must be >= 1", file=sys.stderr)
        return 1
    if args.num_grader_nodes > 1 and args.grader_port + args.num_grader_nodes - 1 > 65535:
        print("grader-port + num-grader-nodes - 1 exceeds 65535", file=sys.stderr)
        return 1
    if args.num_synthesis_nodes > 1 and args.gpu_port + args.num_synthesis_nodes - 1 > 65535:
        print("gpu-port + num-synthesis-nodes - 1 exceeds 65535", file=sys.stderr)
        return 1
    if not os.path.isdir(grader_model_dir):
        print(f"Grader model dir not found: {grader_model_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.paper_ids):
        print(f"Paper IDs file not found: {args.paper_ids}", file=sys.stderr)
        return 1
    if not args.host_rubric_path or not os.path.isfile(args.host_rubric_path):
        print(f"Host rubric file not found: {args.host_rubric_path}", file=sys.stderr)
        return 1
    if not args.microbe_rubric_path or not os.path.isfile(args.microbe_rubric_path):
        print(f"Microbe rubric file not found: {args.microbe_rubric_path}", file=sys.stderr)
        return 1

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    docling_env = {
        "DATA_ROOT": args.data_root,
        "OUTPUT_ROOT": output_root,
        "DOCLING_API_PORT": str(args.docling_port),
        "DOCLING_IMAGE": docling_image,
        "REPO_ROOT": repo_root_path,
        "PIPELINE_ROOT": pipeline_root,
    }
    docling_log = os.path.join(logs_root, "auto_lit_docling_%j.log")
    docling_job_id = _sbatch(docling_script, docling_env, log_path=docling_log)
    print(f"Submitted Docling GPU job: {docling_job_id}")

    gpu_job_ids: list[str] = []
    for i in range(args.num_synthesis_nodes):
        gpu_port_i = args.gpu_port + i
        gpu_env = {
            "DATA_ROOT": args.data_root,
            "MODEL_DIR": args.model_dir,
            "OUTPUT_ROOT": output_root,
            "GPU_API_PORT": str(gpu_port_i),
            "GPU_IMAGE": gpu_image,
            "REPO_ROOT": repo_root_path,
        "PIPELINE_ROOT": pipeline_root,
            "HOST_RUBRIC_PATH": os.path.abspath(args.host_rubric_path),
            "MICROBE_RUBRIC_PATH": os.path.abspath(args.microbe_rubric_path),
        }
        gpu_log = os.path.join(logs_root, f"auto_lit_gpu_{i}_%j.log")
        gpu_job_id = _sbatch(gpu_script, gpu_env, log_path=gpu_log)
        gpu_job_ids.append(gpu_job_id)
        print(
            f"Submitted LLM GPU job {i + 1}/{args.num_synthesis_nodes}: "
            f"{gpu_job_id} (port {gpu_port_i})"
        )

    _grader_optional_env_keys = (
        "GRADER_MAX_TOKENS",
        "GRADER_PAPER_WORKERS",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_GPU_MEMORY_UTILIZATION",
    )

    grader_job_ids: list[str] = []
    for i in range(args.num_grader_nodes):
        grader_port_i = args.grader_port + i
        grader_env = {
            "DATA_ROOT": args.data_root,
            "MODEL_DIR": grader_model_dir,
            "OUTPUT_ROOT": output_root,
            "GRADER_API_PORT": str(grader_port_i),
            "GRADER_IMAGE": grader_image,
            "REPO_ROOT": repo_root_path,
        "PIPELINE_ROOT": pipeline_root,
            "GRADER_SKIP_SYNTHESIS": "1",
        }
        for key in _grader_optional_env_keys:
            val = os.environ.get(key, "").strip()
            if val:
                grader_env[key] = val
        grader_log = os.path.join(logs_root, f"auto_lit_grader_{i}_%j.log")
        grader_job_id = _sbatch(grader_script, grader_env, log_path=grader_log)
        grader_job_ids.append(grader_job_id)
        print(
            f"Submitted Grader GPU job {i + 1}/{args.num_grader_nodes}: "
            f"{grader_job_id} (port {grader_port_i})"
        )

    gpu_host = None
    docling_host = None
    grader_host = None
    gpu_urls: list[str] = []
    grader_urls: list[str] = []
    if not args.no_wait:
        print(
            f"Waiting for first LLM GPU job ({gpu_job_ids[0]}) to get a node name "
            f"({len(gpu_job_ids)} synthesis job(s) submitted; others discovered at runtime)..."
        )
        gpu_host = _get_node_name(gpu_job_ids[0])
        if not gpu_host:
            print(
                "Could not get GPU node name; submit CPU job manually with GPU_URLS / GPU_JOB_IDS set.",
                file=sys.stderr,
            )
        else:
            gpu_urls.append(f"http://{gpu_host}:{args.gpu_port}")
            print(f"LLM GPU node 1 (initial): {gpu_host}:{args.gpu_port}")

        print("Waiting for Docling GPU job to run and get node name...")
        docling_host = _get_node_name(docling_job_id)
        if not docling_host:
            print(
                "Could not get Docling node name; submit CPU job manually with DOCLING_HOST set.",
                file=sys.stderr,
            )
        else:
            print(f"Docling node (initial): {docling_host}")
            _health_wait = int(
                os.environ.get("SERVICE_HEALTH_WAIT_SECONDS", "900").strip() or "900"
            )
            _doc_ok, docling_host = _wait_healthz(
                "Docling",
                docling_host,
                args.docling_port,
                job_id=docling_job_id,
                timeout=_health_wait,
            )
            if not _doc_ok:
                print(
                    "Docling /healthz not ready yet; CPU job will retry using DOCLING_JOB_ID.",
                    file=sys.stderr,
                )
            else:
                print(f"Docling node (healthy): {docling_host}")

        print(
            f"Waiting for first Grader GPU job ({grader_job_ids[0]}) to get a node name "
            f"({len(grader_job_ids)} grader job(s) submitted; others discovered at runtime)..."
        )
        host = _get_node_name(grader_job_ids[0])
        port = args.grader_port
        if not host:
            print(
                f"Could not get Grader node name for job {grader_job_ids[0]}; "
                "submit CPU job manually with GRADER_URLS / GRADER_JOB_IDS set.",
                file=sys.stderr,
            )
        else:
            grader_urls.append(f"http://{host}:{port}")
            print(f"Grader node 1 (initial): {host}:{port}")
            grader_host = host
    else:
        print(
            "Not waiting for GPU/Docling nodes. Set GPU_HOST, DOCLING_HOST, and GRADER_URLS when "
            "submitting CPU job manually."
        )

    if (
        args.collection_org.strip().lower() == "ucsc"
        and args.collection_auth_scope.strip().lower() == "email_only"
        and not args.collector_email.strip()
    ):
        print(
            "COLLECTOR_EMAIL is required for UCSC email_only collection mode.",
            file=sys.stderr,
        )
        return 2

    cpu_env = {
        "DATA_ROOT": args.data_root,
        "PAPER_IDS_PATH": os.path.abspath(args.paper_ids),
        "OUTPUT_ROOT": output_root,
        "GPU_API_PORT": str(args.gpu_port),
        "CPU_IMAGE": cpu_image,
        "REPO_ROOT": repo_root_path,
        "PIPELINE_ROOT": pipeline_root,
        "GPU_HOST": gpu_host or "",
        "GPU_URLS": ";".join(gpu_urls),
        "GPU_JOB_IDS": ":".join(gpu_job_ids),
        "NUM_SYNTHESIS_NODES": str(args.num_synthesis_nodes),
        "DOCLING_HOST": docling_host or "",
        "DOCLING_JOB_ID": docling_job_id,
        "DOCLING_API_PORT": str(args.docling_port),
        "GRADER_HOST": grader_host or "",
        "GRADER_API_PORT": str(args.grader_port),
        "GRADER_URLS": ";".join(grader_urls),
        "GRADER_JOB_IDS": ":".join(grader_job_ids),
        "NUM_GRADER_NODES": str(args.num_grader_nodes),
        "HOST_RUBRIC_PATH": os.path.abspath(args.host_rubric_path),
        "MICROBE_RUBRIC_PATH": os.path.abspath(args.microbe_rubric_path),
        "COLLECTION_ORG": args.collection_org,
        "COLLECTION_AUTH_SCOPE": args.collection_auth_scope,
        "COLLECTOR_EMAIL": args.collector_email,
        "COLLECT_MAX_WORKERS": str(max(1, min(16, int(args.collect_max_workers)))),
        "COLLECT_DISABLE_SEMANTIC_SCHOLAR": (
            "1" if args.collect_disable_semantic_scholar else ""
        ),
        "RUN_LOGS_DIR": os.path.join(output_root, "logs"),
        "SCHEDULER_STATE_DIR": os.path.join(output_root, "scheduler_state"),
    }
    if args.instructions_file and os.path.isfile(args.instructions_file):
        cpu_env["INSTRUCTIONS_FILE"] = os.path.abspath(args.instructions_file)
    if args.idmap_csv:
        cpu_env["IDMAP_CSV"] = os.path.abspath(args.idmap_csv)

    cpu_env.update(_publisher_env_from_os())

    gpu_hosts_ready = len(gpu_urls) >= 1 or bool(gpu_job_ids)
    grader_hosts_ready = len(grader_urls) >= 1 or bool(grader_job_ids)
    if gpu_host and docling_host and grader_hosts_ready and gpu_hosts_ready:
        dep_ids = [gpu_job_ids[0], docling_job_id, grader_job_ids[0]]
        dep = ":".join(dep_ids)
        cpu_log = os.path.join(logs_root, "auto_lit_cpu_%j.log")
        cpu_job_id = _sbatch(
            cpu_script,
            cpu_env,
            dependency=dep,
            dependency_kind="after",
            log_path=cpu_log,
        )
        print(f"Submitted CPU job (after GPU+Docling+first Grader start): {cpu_job_id}")
    else:
        print(
            "GPU and/or Docling and/or Grader node name not available. Submit the CPU job manually "
            "after all GPU jobs are RUNNING:"
        )
        all_gpu_jobs = ",".join([*gpu_job_ids, docling_job_id, *grader_job_ids])
        print(f"  squeue -j {all_gpu_jobs}   # then note the NODELIST values")
        export_str = (
            f"DATA_ROOT={args.data_root},"
            f"PAPER_IDS_PATH={os.path.abspath(args.paper_ids)},"
            f"OUTPUT_ROOT={output_root},"
            f"GPU_HOST=<LLM_NODELIST>,"
            f"GPU_URLS=<http://host0:{args.gpu_port}>,"
            f"GPU_JOB_IDS=<job_id:job_id:...>,"
            f"NUM_SYNTHESIS_NODES={args.num_synthesis_nodes},"
            f"GPU_API_PORT={args.gpu_port},"
            f"DOCLING_HOST=<DOCLING_NODELIST>,"
            f"DOCLING_API_PORT={args.docling_port},"
            f"GRADER_URLS=<http://host0:{args.grader_port}>,"
            f"GRADER_JOB_IDS=<job_id:job_id:...>,"
            f"GRADER_HOST=<FIRST_GRADER_HOST>,"
            f"GRADER_API_PORT={args.grader_port},"
            f"NUM_GRADER_NODES={args.num_grader_nodes},"
            f"HOST_RUBRIC_PATH={os.path.abspath(args.host_rubric_path)},"
            f"MICROBE_RUBRIC_PATH={os.path.abspath(args.microbe_rubric_path)},"
            f"COLLECTION_ORG={args.collection_org},"
            f"COLLECTION_AUTH_SCOPE={args.collection_auth_scope},"
            f"COLLECTOR_EMAIL={args.collector_email},"
            f"COLLECT_MAX_WORKERS={max(1, min(16, int(args.collect_max_workers)))},"
            f"COLLECT_DISABLE_SEMANTIC_SCHOLAR={'1' if args.collect_disable_semantic_scholar else ''},"
            f"CPU_IMAGE={cpu_image},"
            f"REPO_ROOT={repo_root_path}"
        )
        if args.instructions_file and os.path.isfile(args.instructions_file):
            export_str += f",INSTRUCTIONS_FILE={os.path.abspath(args.instructions_file)}"
        if args.idmap_csv:
            export_str += f",IDMAP_CSV={os.path.abspath(args.idmap_csv)}"
        _extra = _publisher_env_from_os()
        if _extra:
            export_str += "," + ",".join(f"{k}={v}" for k, v in _extra.items())
        first_grader = grader_job_ids[0] if grader_job_ids else "<GRADER_JOB_ID>"
        print(
            f"  sbatch --dependency=after:{gpu_job_ids[0]}:{docling_job_id}:{first_grader} "
            f"--export=ALL,{export_str} {cpu_script}"
        )

    return 0


def _build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Submit Docling GPU, LLM GPU, and CPU Slurm jobs for "
            "auto_lit_search three-node pipeline"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Stage2 YAML config (sets defaults; CLI flags override)",
    )
    p.add_argument(
        "--paper-ids",
        default=None,
        help="Search output JSON (from search.py or lp_alignments)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Shared data root",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Results dir (default: data_root/llm_results)",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Path to model weights for synthesis LLM (vLLM on gpu_llm_node)",
    )
    p.add_argument(
        "--grader-model-dir",
        default=None,
        help="Path to grader model weights (default: same as --model-dir)",
    )
    p.add_argument("--gpu-image", default=None)
    p.add_argument("--docling-image", default=None)
    p.add_argument("--grader-image", default=None)
    p.add_argument("--cpu-image", default=None)
    p.add_argument("--gpu-script", default=None)
    p.add_argument("--docling-script", default=None)
    p.add_argument("--grader-script", default=None)
    p.add_argument("--cpu-script", default=None)
    p.add_argument("--gpu-port", type=int, default=None)
    p.add_argument("--docling-port", type=int, default=None)
    p.add_argument("--grader-port", type=int, default=None)
    p.add_argument("--num-grader-nodes", type=int, default=None)
    p.add_argument("--num-synthesis-nodes", type=int, default=None)
    p.add_argument("--instructions-file", default=None)
    p.add_argument("--idmap-csv", default=None)
    p.add_argument("--host-rubric-path", default=None)
    p.add_argument("--microbe-rubric-path", default=None)
    p.add_argument("--collection-org", default=None)
    p.add_argument(
        "--collection-auth-scope",
        default=None,
        choices=["email_only", "email_password"],
    )
    p.add_argument("--collector-email", default=None)
    p.add_argument("--collect-max-workers", type=int, default=None)
    p.add_argument("--collect-disable-semantic-scholar", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    if defaults:
        p.set_defaults(**defaults)
    return p


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    defaults: dict[str, Any] = {
        "data_root": os.environ.get("AUTO_LIT_DATA_ROOT", "").strip() or None,
        "gpu_port": 9000,
        "docling_port": 9100,
        "grader_port": 9200,
        "num_grader_nodes": int(os.environ.get("NUM_GRADER_NODES", "1")),
        "num_synthesis_nodes": int(os.environ.get("NUM_SYNTHESIS_NODES", "1")),
        "collection_org": os.environ.get("COLLECTION_ORG", "ucsc"),
        "collection_auth_scope": os.environ.get("COLLECTION_AUTH_SCOPE", "email_only"),
        "collector_email": os.environ.get("COLLECTOR_EMAIL", ""),
        "host_rubric_path": os.environ.get("HOST_RUBRIC_PATH", ""),
        "microbe_rubric_path": os.environ.get("MICROBE_RUBRIC_PATH", ""),
        "collect_max_workers": 2,
        "collect_disable_semantic_scholar": False,
        "no_wait": False,
    }

    if pre_args.config:
        try:
            cfg = load_stage2_config(pre_args.config)
        except (OSError, ValueError, RuntimeError) as e:
            print(str(e), file=sys.stderr)
            return 2
        apply_runtime_env(cfg.runtime_env)
        defaults.update(_defaults_from_stage2(cfg))

    p = _build_parser(defaults)
    args = p.parse_args()

    if not args.paper_ids:
        print("--paper-ids is required (or pass --config with stage2.paper_ids_json)", file=sys.stderr)
        return 2
    if not args.data_root:
        print("--data-root is required (or pass --config with data_root)", file=sys.stderr)
        return 2
    if not args.model_dir:
        print("--model-dir is required (or pass --config with cluster.model_dir)", file=sys.stderr)
        return 2

    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
