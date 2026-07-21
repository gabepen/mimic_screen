#!/usr/bin/env python3
"""Submit CPU download/analysis only (after GPU services are up) from a stage2 config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from run_pipeline_slurm import _publisher_env_from_os  # noqa: E402

from auto_lit_search.env_config import slurm_mail_user  # noqa: E402
from auto_lit_search.pipeline_config import (  # noqa: E402
    Stage2Config,
    apply_runtime_env,
    load_stage2_config,
)


def _sbatch_cpu(script: Path, env: dict[str, str], log_path: Path) -> str:
    export_pairs = [f"{k}={v}" for k, v in env.items()]
    cmd = ["sbatch", "--parsable"]
    mail_user = slurm_mail_user()
    if mail_user:
        cmd.extend(["--mail-type=BEGIN,END,FAIL", f"--mail-user={mail_user}"])
    cmd.extend(
        [
            "--output",
            str(log_path),
            "--export=ALL," + ",".join(export_pairs),
            str(script),
        ]
    )
    out = subprocess.check_output(cmd, text=True)
    return out.strip().split(";")[0]


def wait_for_job_node(job_id: str, label: str, max_wait: int = 43200) -> str:
    start = time.time()
    print(
        f"Waiting for job {job_id} ({label}) to be RUNNING with an allocated node "
        f"(max {max_wait}s)...",
        file=sys.stderr,
    )
    while time.time() - start < max_wait:
        try:
            state = subprocess.check_output(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                text=True,
            ).strip()
            node = subprocess.check_output(
                ["squeue", "-j", job_id, "-h", "-o", "%N"],
                text=True,
            ).strip()
            if state == "RUNNING" and node:
                print(f"{label} job {job_id} -> {node}", file=sys.stderr)
                return node
            if state in {
                "COMPLETING",
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
            }:
                raise RuntimeError(f"Job {job_id} is {state}; cannot get a live node")
        except subprocess.CalledProcessError:
            pass
        time.sleep(5)
    raise RuntimeError(f"Timeout waiting for job {job_id}")


def _url_host(url: str) -> str:
    import re

    m = re.match(r"https?://([^:/]+)", url.strip())
    return m.group(1) if m else ""


def build_cpu_env(
    cfg: Stage2Config,
    *,
    gpu_host: str,
    gpu_urls: str,
    gpu_job_ids: str,
    num_synthesis_nodes: int,
    docling_host: str,
    docling_job_id: str,
    grader_host: str,
    grader_urls: str,
    grader_job_ids: str,
    num_grader_nodes: int,
) -> dict[str, str]:
    coll = cfg.collection
    env = {
        "DATA_ROOT": str(cfg.data_root),
        "PAPER_IDS_PATH": str(cfg.paper_ids_json),
        "OUTPUT_ROOT": str(cfg.output_root),
        "GPU_HOST": gpu_host,
        "GPU_API_PORT": str(cfg.slurm.gpu_port),
        "GPU_URLS": gpu_urls.replace(",", ";"),
        "GPU_JOB_IDS": gpu_job_ids,
        "NUM_SYNTHESIS_NODES": str(num_synthesis_nodes),
        "DOCLING_HOST": docling_host,
        "DOCLING_JOB_ID": docling_job_id,
        "DOCLING_API_PORT": str(cfg.slurm.docling_port),
        "GRADER_URLS": grader_urls.replace(",", ";"),
        "GRADER_JOB_IDS": grader_job_ids,
        "NUM_GRADER_NODES": str(num_grader_nodes) if num_grader_nodes else "",
        "GRADER_HOST": grader_host,
        "GRADER_API_PORT": str(cfg.slurm.grader_port),
        "HOST_RUBRIC_PATH": str(cfg.host_rubric),
        "MICROBE_RUBRIC_PATH": str(cfg.microbe_rubric),
        "COLLECTION_ORG": coll.org,
        "COLLECTION_AUTH_SCOPE": coll.auth_scope,
        "COLLECTOR_EMAIL": coll.collector_email,
        "COLLECT_MAX_WORKERS": str(max(1, min(16, coll.max_workers))),
        "COLLECT_DISABLE_SEMANTIC_SCHOLAR": "1" if coll.disable_semantic_scholar else "",
        "CPU_IMAGE": cfg.cluster.cpu_image,
        "REPO_ROOT": str(cfg.cluster.repo_root),
        "PIPELINE_ROOT": str(cfg.cluster.pipeline_root),
        "RUN_LOGS_DIR": str(cfg.output_root / "logs"),
        "SCHEDULER_STATE_DIR": str(cfg.output_root / "scheduler_state"),
    }
    if cfg.instructions_file.is_file():
        env["INSTRUCTIONS_FILE"] = str(cfg.instructions_file)
    if cfg.idmap_csv.is_file():
        env["IDMAP_CSV"] = str(cfg.idmap_csv)

    for key in (
        "SERVICE_HEALTH_WAIT_SECONDS",
        "DOCLING_INFLIGHT_CAP",
        "GRADER_INFLIGHT_CAP",
        "SCHEDULER_TICK_SECONDS",
        "STAGE_WATCHDOG_SECONDS",
    ):
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val

    env.update(_publisher_env_from_os())
    return env


def run_cpu_manual_from_jobs(
    cfg: Stage2Config,
    synth_jobs: List[str],
    docling_job: str,
    grader_jobs: List[str],
) -> int:
    num_synth = len(synth_jobs)
    gpu_urls_list: list[str] = []
    for i, jid in enumerate(synth_jobs):
        host = wait_for_job_node(jid, f"LLM-{i + 1}")
        gpu_urls_list.append(f"http://{host}:{cfg.slurm.gpu_port + i}")
    gpu_urls = ";".join(gpu_urls_list)
    gpu_host = _url_host(gpu_urls_list[0])
    docling_host = wait_for_job_node(docling_job, "Docling")
    grader_job_ids = ":".join(grader_jobs)
    num_grader = len(grader_jobs)
    grader_host = wait_for_job_node(grader_jobs[0], "Grader-1")

    discovered = cfg.cluster.logs_dir / "grader_endpoints_discovered.txt"
    grader_urls = ""
    if discovered.is_file():
        lines = [
            ln.strip()
            for ln in discovered.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("http")
        ]
        if lines:
            grader_urls = ";".join(lines)
            grader_host = _url_host(lines[0])
            print(
                f"Loaded GRADER_URLS from {discovered} ({len(lines)} endpoints)",
                file=sys.stderr,
            )

    return _submit_cpu(
        cfg,
        gpu_host=gpu_host,
        gpu_urls=gpu_urls,
        gpu_job_ids=":".join(synth_jobs),
        num_synthesis_nodes=num_synth,
        docling_host=docling_host,
        docling_job_id=docling_job,
        grader_host=grader_host,
        grader_urls=grader_urls,
        grader_job_ids=grader_job_ids,
        num_grader_nodes=num_grader,
    )


def _submit_cpu(
    cfg: Stage2Config,
    *,
    gpu_host: str,
    gpu_urls: str,
    gpu_job_ids: str,
    num_synthesis_nodes: int,
    docling_host: str,
    docling_job_id: str,
    grader_host: str,
    grader_urls: str,
    grader_job_ids: str,
    num_grader_nodes: int,
) -> int:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "scheduler_state").mkdir(parents=True, exist_ok=True)

    cpu_env = build_cpu_env(
        cfg,
        gpu_host=gpu_host,
        gpu_urls=gpu_urls,
        gpu_job_ids=gpu_job_ids,
        num_synthesis_nodes=num_synthesis_nodes,
        docling_host=docling_host,
        docling_job_id=docling_job_id,
        grader_host=grader_host,
        grader_urls=grader_urls,
        grader_job_ids=grader_job_ids,
        num_grader_nodes=num_grader_nodes,
    )

    cpu_script = cfg.slurm.cpu_script
    log_path = cfg.cluster.logs_dir / "auto_lit_cpu_%j.log"
    cfg.cluster.logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"OUTPUT_ROOT={cfg.output_root}", file=sys.stderr)
    if gpu_urls:
        print(f"Synthesis GPU endpoints: {gpu_urls}", file=sys.stderr)
    if grader_urls:
        print(f"Grader endpoints: {grader_urls}", file=sys.stderr)

    job_id = _sbatch_cpu(cpu_script, cpu_env, log_path)
    print(f"Submitted CPU job: {job_id}")
    print(f"CPU log: {cfg.cluster.logs_dir / f'auto_lit_cpu_{job_id}.log'}", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Stage2 YAML config path")
    p.add_argument(
        "--gpu-urls",
        default=None,
        help="Semicolon-separated synthesis GPU URLs (explicit endpoint mode)",
    )
    p.add_argument("--gpu-job-ids", default=None, help="Colon-separated synthesis Slurm job IDs")
    p.add_argument("--docling-host", default=None)
    p.add_argument("--docling-job-id", default=None)
    p.add_argument("--grader-urls", default=None)
    p.add_argument("--grader-job-ids", default=None)
    p.add_argument("--grader-host", default=None)
    p.add_argument(
        "job_ids",
        nargs="*",
        help="Positional: N synthesis GPU jobs, docling job, then grader job(s)",
    )
    args = p.parse_args()

    try:
        cfg = load_stage2_config(args.config)
    except (OSError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 2
    apply_runtime_env(cfg.runtime_env)

    explicit = any(
        [
            args.gpu_urls,
            args.gpu_job_ids,
            args.docling_host,
            args.docling_job_id,
            args.grader_urls,
            args.grader_job_ids,
            args.grader_host,
        ]
    )

    try:
        if explicit:
            gpu_urls = args.gpu_urls or ""
            gpu_job_ids = args.gpu_job_ids or ""
            docling_host = args.docling_host or ""
            docling_job_id = args.docling_job_id or ""
            grader_urls = args.grader_urls or ""
            grader_job_ids = args.grader_job_ids or ""
            grader_host = args.grader_host or ""

            if not docling_host:
                print("--docling-host is required in explicit endpoint mode", file=sys.stderr)
                return 2
            if not gpu_urls and not gpu_job_ids:
                print("Pass --gpu-urls or --gpu-job-ids", file=sys.stderr)
                return 2
            if not grader_urls and not grader_host and not grader_job_ids:
                print("Pass --grader-urls, --grader-host, or --grader-job-ids", file=sys.stderr)
                return 2

            gpu_host = _url_host(gpu_urls.split(";")[0]) if gpu_urls else ""
            if not grader_host and grader_urls:
                grader_host = _url_host(grader_urls.split(";")[0])
            num_synth = cfg.slurm.num_synthesis_nodes
            if gpu_urls:
                num_synth = len([u for u in gpu_urls.split(";") if u.strip()])
            num_grader = cfg.slurm.num_grader_nodes
            if grader_urls:
                num_grader = len([u for u in grader_urls.split(";") if u.strip()])

            return _submit_cpu(
                cfg,
                gpu_host=gpu_host,
                gpu_urls=gpu_urls,
                gpu_job_ids=gpu_job_ids,
                num_synthesis_nodes=num_synth,
                docling_host=docling_host,
                docling_job_id=docling_job_id,
                grader_host=grader_host,
                grader_urls=grader_urls,
                grader_job_ids=grader_job_ids,
                num_grader_nodes=num_grader,
            )

        if not args.job_ids:
            p.error(
                "Pass Slurm job IDs (synthesis..., docling, grader...) or use explicit --gpu-urls mode"
            )

        num_synth = cfg.slurm.num_synthesis_nodes
        min_args = num_synth + 2
        if len(args.job_ids) < min_args:
            print(
                f"Need at least {min_args} job IDs: {num_synth} synthesis GPU(s), "
                "docling, then grader(s)",
                file=sys.stderr,
            )
            return 2

        synth_jobs = args.job_ids[:num_synth]
        docling_job = args.job_ids[num_synth]
        grader_jobs = args.job_ids[num_synth + 1 :]
        return run_cpu_manual_from_jobs(cfg, synth_jobs, docling_job, grader_jobs)

    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
