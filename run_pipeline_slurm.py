#!/usr/bin/env python3
"""
Launch the two-node Slurm pipeline: GPU LLM node first, then CPU download node.
Usage: python run_pipeline_slurm.py --paper-ids <search_output.json> [options]
"""

import argparse
import os
import re
import subprocess
import sys
import time


def _sbatch(
    script_path: str,
    env: dict,
    dependency: str | None = None,
    log_path: str | None = None,
) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.extend(["--dependency=afterok:" + dependency])
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


def _get_node_name(job_id: str, max_wait: int = 12 * 60 * 60) -> str | None:
    invalid_nodes = {"", "none", "(null)", "null", "n/a", "unknown"}
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        try:
            raw = _scontrol_show_job(job_id)
            for line in raw.splitlines():
                if line.strip().startswith("NodeList="):
                    node = line.split("=", 1)[1].strip()
                    if node and node.lower() not in invalid_nodes:
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
        help="Path to model weights (for vLLM)",
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
        "--no-wait",
        action="store_true",
        help="Do not wait for GPU node to be RUNNING before submitting CPU job",
    )
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    output_root = args.output_root or os.path.join(args.data_root, "llm_results")
    logs_root = os.path.join(args.data_root, "logs")
    gpu_script = args.gpu_script or os.path.join(repo_root, "slurm", "gpu_llm_node.slurm")
    docling_script = args.docling_script or os.path.join(
        repo_root, "slurm", "gpu_docling_node.slurm"
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
    cpu_image = args.cpu_image or os.path.join(repo_root, "containers", "lit-download.sif")
    if not os.path.isfile(gpu_image) and not (gpu_image.startswith("docker://") or gpu_image.startswith("library://")):
        print(f"GPU image not found: {gpu_image}. Pass --gpu-image.", file=sys.stderr)
        return 1
    if not os.path.isfile(cpu_image) and not (cpu_image.startswith("docker://") or cpu_image.startswith("library://")):
        print(f"CPU image not found: {cpu_image}. Pass --cpu-image.", file=sys.stderr)
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
    if not os.path.isfile(args.paper_ids):
        print(f"Paper IDs file not found: {args.paper_ids}", file=sys.stderr)
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

    # Launch LLM GPU node.
    gpu_env = {
        "DATA_ROOT": args.data_root,
        "MODEL_DIR": args.model_dir,
        "OUTPUT_ROOT": output_root,
        "GPU_API_PORT": str(args.gpu_port),
        "GPU_IMAGE": gpu_image,
        "REPO_ROOT": repo_root,
    }

    gpu_log = os.path.join(logs_root, "auto_lit_gpu_%j.log")
    gpu_job_id = _sbatch(gpu_script, gpu_env, log_path=gpu_log)
    print(f"Submitted LLM GPU job: {gpu_job_id}")

    gpu_host = None
    docling_host = None
    if not args.no_wait:
        print("Waiting for LLM GPU job to run and get node name...")
        gpu_host = _get_node_name(gpu_job_id)
        if not gpu_host:
            print(
                "Could not get GPU node name; submit CPU job manually with GPU_HOST set.",
                file=sys.stderr,
            )
        else:
            print(f"LLM GPU node: {gpu_host}")

        print("Waiting for Docling GPU job to run and get node name...")
        docling_host = _get_node_name(docling_job_id)
        if not docling_host:
            print(
                "Could not get Docling node name; submit CPU job manually with DOCLING_HOST set.",
                file=sys.stderr,
            )
        else:
            print(f"Docling node: {docling_host}")
    else:
        print(
            "Not waiting for GPU/Docling nodes. Set GPU_HOST and DOCLING_HOST when "
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
        "DOCLING_HOST": docling_host or "",
        "DOCLING_API_PORT": str(args.docling_port),
        "COLLECTION_ORG": args.collection_org,
        "COLLECTION_AUTH_SCOPE": args.collection_auth_scope,
        "COLLECTOR_EMAIL": args.collector_email,
    }
    if args.instructions_file and os.path.isfile(args.instructions_file):
        cpu_env["INSTRUCTIONS_FILE"] = os.path.abspath(args.instructions_file)
    if args.idmap_csv:
        cpu_env["IDMAP_CSV"] = os.path.abspath(args.idmap_csv)

    if gpu_host and docling_host:
        dep = f"{gpu_job_id}:{docling_job_id}"
        cpu_log = os.path.join(logs_root, "auto_lit_cpu_%j.log")
        cpu_job_id = _sbatch(cpu_script, cpu_env, dependency=dep, log_path=cpu_log)
        print(f"Submitted CPU job (after GPU+Docling): {cpu_job_id}")
    else:
        print(
            "GPU and/or Docling node name not available. Submit the CPU job manually "
            "after both GPU jobs are RUNNING:"
        )
        print(f"  squeue -j {gpu_job_id},{docling_job_id}   # then note the NODELIST values")
        export_str = (
            f"DATA_ROOT={args.data_root},"
            f"PAPER_IDS_PATH={os.path.abspath(args.paper_ids)},"
            f"OUTPUT_ROOT={output_root},"
            f"GPU_HOST=<LLM_NODELIST>,"
            f"GPU_API_PORT={args.gpu_port},"
            f"DOCLING_HOST=<DOCLING_NODELIST>,"
            f"DOCLING_API_PORT={args.docling_port},"
            f"COLLECTION_ORG={args.collection_org},"
            f"COLLECTION_AUTH_SCOPE={args.collection_auth_scope},"
            f"COLLECTOR_EMAIL={args.collector_email},"
            f"CPU_IMAGE={cpu_image},"
            f"REPO_ROOT={repo_root}"
        )
        if args.instructions_file and os.path.isfile(args.instructions_file):
            export_str += f",INSTRUCTIONS_FILE={os.path.abspath(args.instructions_file)}"
        if args.idmap_csv:
            export_str += f",IDMAP_CSV={os.path.abspath(args.idmap_csv)}"
        print(
            f"  sbatch --dependency=afterok:{gpu_job_id}:{docling_job_id} "
            f"--export=ALL,{export_str} {cpu_script}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
