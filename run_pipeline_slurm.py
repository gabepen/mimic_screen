#!/usr/bin/env python3
"""
Submit Slurm jobs for the auto_lit_search pipeline (Docling + Grader + LLM GPUs, then CPU download).

Typically invoked via ``literature_analysis_launcher.sh``, which sets paths and optional secrets
in the environment; this script forwards publisher tokens into the CPU job when present.

Usage: python run_pipeline_slurm.py --paper-ids <search_output.json> [options]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Literal

# Forwarded into the CPU/download job if set in the environment when launching.
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


def _sbatch(
    script_path: str,
    env: dict,
    dependency: str | None = None,
    dependency_kind: Literal["afterok", "after"] = "afterok",
    log_path: str | None = None,
) -> str:
    cmd = ["sbatch", "--parsable"]
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
    """
    Extract the allocated hostname from `scontrol show job` output.

    Slurm often prints several space-separated Key=Value tokens on one line,
    e.g. ``NodeList=(null) SchedNodeList=phoenix-00``.  Taking ``split('=', 1)[1]``
    for ``NodeList=`` would incorrectly return ``(null) SchedNodeList=phoenix-00``.
    We take the first token after ``NodeList=``; if that is unset, use
    ``SchedNodeList=``.
    """
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
    """Return allocated node name if the job already has one, else None."""
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


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Submit Docling GPU, LLM GPU, and CPU Slurm jobs for "
            "auto_lit_search three-node pipeline"
        ),
    )
    p.add_argument(
        "--paper-ids",
        required=True,
        help="Search output JSON (from search.py or lp_alignments)",
    )
    p.add_argument(
        "--data-root",
        default="/private/groups/corbettlab/gabe/auto_lit_eval_data",
        help="Shared data root",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Results dir (default: data_root/llm_results)",
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to model weights for synthesis LLM (vLLM on gpu_llm_node)",
    )
    p.add_argument(
        "--grader-model-dir",
        default=None,
        help="Path to grader model weights (default: same as --model-dir)",
    )
    p.add_argument(
        "--gpu-image",
        default=None,
        help="GPU container (path to .sif or docker://user/image:tag)",
    )
    p.add_argument(
        "--docling-image",
        default=None,
        help="Docling container (path to .sif or docker://user/image:tag)",
    )
    p.add_argument(
        "--grader-image",
        default=None,
        help="Grader container (path to .sif or docker://user/image:tag)",
    )
    p.add_argument(
        "--cpu-image",
        default=None,
        help="CPU container (path to .sif or docker://user/image:tag)",
    )
    p.add_argument(
        "--gpu-script",
        default=None,
        help="Path to gpu_llm_node.slurm",
    )
    p.add_argument(
        "--docling-script",
        default=None,
        help="Path to gpu_docling_node.slurm",
    )
    p.add_argument(
        "--grader-script",
        default=None,
        help="Path to gpu_grader_node.slurm",
    )
    p.add_argument(
        "--cpu-script",
        default=None,
        help="Path to cpu_download_node.slurm",
    )
    p.add_argument("--gpu-port", type=int, default=9000)
    p.add_argument(
        "--docling-port",
        type=int,
        default=9100,
        help="Port for Docling GPU service",
    )
    p.add_argument(
        "--grader-port",
        type=int,
        default=9200,
        help="Base port for grader GPU services (instance i uses grader-port + i)",
    )
    p.add_argument(
        "--num-grader-nodes",
        type=int,
        default=int(os.environ.get("NUM_GRADER_NODES", "1")),
        help="Number of parallel grader GPU jobs (one alignment per grader at a time)",
    )
    p.add_argument(
        "--num-synthesis-nodes",
        type=int,
        default=int(os.environ.get("NUM_SYNTHESIS_NODES", "1")),
        help="Number of parallel synthesis LLM GPU jobs (ports gpu-port + i)",
    )
    p.add_argument(
        "--instructions-file",
        default="",
        help="Path to prompt/instructions file for GPU",
    )
    p.add_argument(
        "--idmap-csv",
        default="",
        help="Path to id mapping CSV (query/target identifiers for prompts)",
    )
    p.add_argument(
        "--host-rubric-path",
        default=os.environ.get("HOST_RUBRIC_PATH", ""),
        help="Path to host rubric JSON.",
    )
    p.add_argument(
        "--microbe-rubric-path",
        default=os.environ.get("MICROBE_RUBRIC_PATH", ""),
        help="Path to microbe rubric JSON.",
    )
    p.add_argument(
        "--collection-org",
        default=os.environ.get("COLLECTION_ORG", "ucsc"),
        help="Collection org routing key (default: %(default)s).",
    )
    p.add_argument(
        "--collection-auth-scope",
        default=os.environ.get("COLLECTION_AUTH_SCOPE", "email_only"),
        choices=["email_only", "email_password"],
        help="Collection auth scope (default: %(default)s).",
    )
    p.add_argument(
        "--collector-email",
        default=os.environ.get("COLLECTOR_EMAIL", ""),
        help="Collector identity email (required for UCSC email_only mode).",
    )
    p.add_argument(
        "--collect-max-workers",
        type=int,
        default=2,
        help="Parallel paper download threads on CPU job (1-16, default 2).",
    )
    p.add_argument(
        "--collect-disable-semantic-scholar",
        action="store_true",
        help="Skip Semantic Scholar during collection (fewer 429s).",
    )
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for GPU node to be RUNNING before submitting CPU job",
    )
    args = p.parse_args()
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

    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_root = args.output_root or os.path.join(args.data_root, "llm_results")
    logs_root = os.path.join(args.data_root, "logs")
    gpu_script = args.gpu_script or os.path.join(repo_root, "slurm", "gpu_llm_node.slurm")
    docling_script = args.docling_script or os.path.join(
        repo_root, "slurm", "gpu_docling_node.slurm"
    )
    grader_script = args.grader_script or os.path.join(
        repo_root, "slurm", "gpu_grader_node.slurm"
    )
    cpu_script = args.cpu_script or os.path.join(
        repo_root, "slurm", "cpu_download_node.slurm"
    )

    gpu_image = args.gpu_image or os.path.join(
        repo_root, "containers", "lit-llm-0.1.0.sif"
    )
    docling_image = args.docling_image or os.path.join(
        repo_root, "containers", "docling-0.1.0.sif"
    )
    grader_image = args.grader_image or gpu_image
    cpu_image = args.cpu_image or os.path.join(repo_root, "containers", "lit-download.sif")
    if not os.path.isfile(gpu_image) and not (gpu_image.startswith("docker://") or gpu_image.startswith("library://")):
        print(f"GPU image not found: {gpu_image}. Pass --gpu-image.", file=sys.stderr)
        return 1
    if not os.path.isfile(cpu_image) and not (cpu_image.startswith("docker://") or cpu_image.startswith("library://")):
        print(f"CPU image not found: {cpu_image}. Pass --cpu-image.", file=sys.stderr)
        return 1
    if not os.path.isfile(grader_image) and not (
        grader_image.startswith("docker://") or grader_image.startswith("library://")
    ):
        print(
            f"Grader image not found: {grader_image}. Pass --grader-image.",
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(gpu_script):
        print(f"GPU script not found: {gpu_script}", file=sys.stderr)
        return 1
    if not os.path.isfile(docling_script):
        print(f"Docling script not found: {docling_script}", file=sys.stderr)
        return 1
    if not os.path.isfile(cpu_script):
        print(f"CPU script not found: {cpu_script}", file=sys.stderr)
        return 1
    if not os.path.isfile(grader_script):
        print(f"Grader script not found: {grader_script}", file=sys.stderr)
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
        print(
            f"Host rubric file not found: {args.host_rubric_path}",
            file=sys.stderr,
        )
        return 1
    if not args.microbe_rubric_path or not os.path.isfile(args.microbe_rubric_path):
        print(
            f"Microbe rubric file not found: {args.microbe_rubric_path}",
            file=sys.stderr,
        )
        return 1

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)

    # Launch Docling GPU node.
    docling_env = {
        "DATA_ROOT": args.data_root,
        "OUTPUT_ROOT": output_root,
        "DOCLING_API_PORT": str(args.docling_port),
        "DOCLING_IMAGE": docling_image,
        "REPO_ROOT": repo_root,
    }
    docling_log = os.path.join(logs_root, "auto_lit_docling_%j.log")
    docling_job_id = _sbatch(docling_script, docling_env, log_path=docling_log)
    print(f"Submitted Docling GPU job: {docling_job_id}")

    # Launch one or more synthesis LLM GPU nodes (distinct API ports when colocated).
    gpu_job_ids: list[str] = []
    for i in range(args.num_synthesis_nodes):
        gpu_port_i = args.gpu_port + i
        gpu_env = {
            "DATA_ROOT": args.data_root,
            "MODEL_DIR": args.model_dir,
            "OUTPUT_ROOT": output_root,
            "GPU_API_PORT": str(gpu_port_i),
            "GPU_IMAGE": gpu_image,
            "REPO_ROOT": repo_root,
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

    # Launch one or more Grader GPU nodes (distinct API ports when colocated on one host).
    grader_job_ids: list[str] = []
    for i in range(args.num_grader_nodes):
        grader_port_i = args.grader_port + i
        grader_env = {
            "DATA_ROOT": args.data_root,
            "MODEL_DIR": grader_model_dir,
            "OUTPUT_ROOT": output_root,
            "GRADER_API_PORT": str(grader_port_i),
            "GRADER_IMAGE": grader_image,
            "REPO_ROOT": repo_root,
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
            print(f"Docling node: {docling_host}")

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
        "REPO_ROOT": repo_root,
        "GPU_HOST": gpu_host or "",
        "GPU_URLS": ";".join(gpu_urls),
        "GPU_JOB_IDS": ":".join(gpu_job_ids),
        "NUM_SYNTHESIS_NODES": str(args.num_synthesis_nodes),
        "DOCLING_HOST": docling_host or "",
        "DOCLING_API_PORT": str(args.docling_port),
        "GRADER_HOST": grader_host or "",
        "GRADER_API_PORT": str(args.grader_port),
        # Semicolon-separated: commas break sbatch --export and are ambiguous in URLs.
        "GRADER_URLS": ";".join(grader_urls),
        # Colon-separated: sbatch --export splits on commas and would truncate job lists.
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
        print(
            f"Submitted CPU job (after GPU+Docling+first Grader start): {cpu_job_id}"
        )
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
            f"REPO_ROOT={repo_root}"
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


if __name__ == "__main__":
    sys.exit(main())
