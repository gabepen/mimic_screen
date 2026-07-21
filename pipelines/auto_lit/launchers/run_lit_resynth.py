#!/usr/bin/env python3
"""Re-run synthesis only from existing *_graded.json using a stage2 YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.env_config import slurm_mail_user
from auto_lit_search.pipeline_config import (  # noqa: E402
    Stage2Config,
    apply_runtime_env,
    load_stage2_config,
)


def _sbatch_mail_args() -> list[str]:
    mail_user = slurm_mail_user()
    if not mail_user:
        return []
    return ["--mail-type=BEGIN,END,FAIL", f"--mail-user={mail_user}"]


def _sbatch(
    script_path: Path,
    env: dict[str, str],
    *,
    dependency: str | None = None,
    dependency_kind: Literal["afterok", "after"] = "afterok",
    log_path: Path | None = None,
) -> str:
    cmd = ["sbatch", "--parsable", *_sbatch_mail_args()]
    if dependency:
        cmd.append(f"--dependency={dependency_kind}:{dependency}")
    if log_path:
        cmd.extend(["--output", str(log_path)])
    export_pairs = [f"{k}={v}" for k, v in env.items() if v is not None]
    if export_pairs:
        cmd.append("--export=ALL," + ",".join(export_pairs))
    cmd.append(str(script_path))
    out = subprocess.check_output(cmd, text=True)
    return out.strip().split(";")[0]


def _wait_for_gpu_health(url: str, max_wait: int = 3600) -> bool:
    health_url = f"{url.rstrip('/')}/healthz"
    start = time.time()
    print(f"Waiting for {health_url} (max {max_wait}s)...", file=sys.stderr)
    while time.time() - start < max_wait:
        try:
            with urllib.request.urlopen(health_url, timeout=10) as resp:
                if resp.status == 200:
                    print(f"Synthesis ready: {url}", file=sys.stderr)
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    print(f"Timeout waiting for {url}", file=sys.stderr)
    return False


def _wait_for_job_node(job_id: str, label: str, max_wait: int = 3600) -> str | None:
    start = time.time()
    while time.time() - start < max_wait:
        try:
            node = subprocess.check_output(
                ["squeue", "-j", job_id, "-h", "-o", "%N"],
                text=True,
            ).strip()
            if node and node not in {"(null)", "(Resources)", "(Priority)"}:
                print(f"GPU {label} (job {job_id}) on {node}", file=sys.stderr)
                return node
        except subprocess.CalledProcessError:
            pass
        time.sleep(5)
    return None


def submit_synthesis_gpus(cfg: Stage2Config, num_nodes: int) -> tuple[str, int]:
    repo = cfg.cluster.repo_root
    pipeline = cfg.cluster.pipeline_root
    gpu_script = pipeline / "slurm" / "gpu_llm_node.slurm"
    logs_dir = cfg.cluster.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    gpu_port_base = cfg.slurm.gpu_port

    job_ids: list[str] = []
    urls: list[str] = []
    for i in range(num_nodes):
        port = gpu_port_base + i
        gpu_env = {
            "DATA_ROOT": str(cfg.data_root),
            "MODEL_DIR": str(cfg.cluster.model_dir),
            "OUTPUT_ROOT": str(cfg.output_root),
            "GPU_API_PORT": str(port),
            "GPU_IMAGE": cfg.cluster.gpu_image,
            "REPO_ROOT": str(repo),
            "PIPELINE_ROOT": str(pipeline),
            "HOST_RUBRIC_PATH": str(cfg.host_rubric),
            "MICROBE_RUBRIC_PATH": str(cfg.microbe_rubric),
        }
        for key in (
            "SYNTHESIS_EXCERPT_TOP_K_HOST",
            "SYNTHESIS_EXCERPT_TOP_K_QUERY",
            "SYNTHESIS_MENTION_EXCERPT_MAX_CHARS",
            "SYNTHESIS_MENTION_TOTAL_CHARS",
            "SYNTHESIS_MENTION_MAX_SITES",
            "SYNTHESIS_MENTION_NO_HIT_FALLBACK_CHARS",
            "SYNTHESIS_FINAL_MAX_TOKENS",
            "SYNTHESIS_CONTEXT_MARGIN",
            "SYNTHESIS_ENABLE_REPAIR_PASS",
            "SYNTHESIS_MAX_ATTEMPTS",
        ):
            val = __import__("os").environ.get(key, "").strip()
            if val:
                gpu_env[key] = val

        log_path = logs_dir / f"auto_lit_resynth_gpu_{i}_%j.log"
        job_id = _sbatch(gpu_script, gpu_env, log_path=log_path)
        job_ids.append(job_id)
        print(f"Submitted resynth synthesis GPU {i}: job {job_id} port {port}", file=sys.stderr)

    for i, job_id in enumerate(job_ids):
        port = gpu_port_base + i
        node = _wait_for_job_node(job_id, str(i))
        if node:
            urls.append(f"http://{node}:{port}")

    if not urls:
        raise RuntimeError("No synthesis GPU nodes became available")

    gpu_urls = ";".join(urls)
    print(f"Using {len(urls)} synthesis GPU URL(s): {gpu_urls}", file=sys.stderr)
    for url in urls:
        if not _wait_for_gpu_health(url, 3600):
            raise RuntimeError(f"Synthesis GPU not healthy: {url}")
    return gpu_urls, len(urls)


def submit_resynth_driver(
    cfg: Stage2Config,
    *,
    gpu_urls: str,
    gpu_workers: int,
    resynth_all: bool,
    after_job: str | None = None,
) -> str:
    repo = cfg.cluster.repo_root
    pipeline = cfg.cluster.pipeline_root
    driver_script = pipeline / "slurm" / "fix_it_synthesis.slurm"
    logs_dir = cfg.cluster.logs_dir
    label = cfg.dataset.replace("/", "_")

    driver_env = {
        "DATA_ROOT": str(cfg.data_root),
        "OUTPUT_ROOT": str(cfg.output_root),
        "SYNTHESIS_URLS": gpu_urls,
        "WORKERS": str(gpu_workers),
        "CPU_IMAGE": cfg.cluster.cpu_image,
        "REPO_ROOT": str(repo),
        "PIPELINE_ROOT": str(pipeline),
        "PAPERS_ROOT": str(cfg.data_root / "papers"),
        "INSTRUCTIONS_FILE": str(cfg.instructions_file),
        "RESYNTH_ALL": "1" if resynth_all else "0",
    }
    log_path = logs_dir / f"auto_lit_resynth_driver_{label}_%j.log"
    dep = after_job if after_job else None
    job_id = _sbatch(driver_script, driver_env, dependency=dep, log_path=log_path)
    print(f"Submitted resynth driver ({label}): Slurm job {job_id}", file=sys.stderr)
    print(f"Driver log: {logs_dir / f'auto_lit_resynth_driver_{label}_{job_id}.log'}", file=sys.stderr)
    return job_id


def run_resynth(
    cfg: Stage2Config,
    *,
    failed_only: bool,
    reuse_gpu_urls: str | None,
    num_synthesis_nodes: int | None,
) -> int:
    n_nodes = num_synthesis_nodes or cfg.slurm.num_synthesis_nodes
    resynth_all = not failed_only

    if reuse_gpu_urls:
        gpu_urls = reuse_gpu_urls
        gpu_workers = len([u for u in reuse_gpu_urls.split(";") if u.strip()])
        print(f"Reusing synthesis GPU URLs ({gpu_workers}): {gpu_urls}", file=sys.stderr)
        for url in gpu_urls.split(";"):
            if url.strip() and not _wait_for_gpu_health(url.strip(), 600):
                print(f"Warning: GPU not healthy: {url}", file=sys.stderr)
    else:
        gpu_urls, gpu_workers = submit_synthesis_gpus(cfg, n_nodes)

    submit_resynth_driver(
        cfg,
        gpu_urls=gpu_urls,
        gpu_workers=gpu_workers,
        resynth_all=resynth_all,
    )
    print("Resynth jobs submitted. Monitor with: squeue -u $USER", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Stage2 YAML config path")
    p.add_argument(
        "--failed-only",
        action="store_true",
        help="Resynth only alignments that need fix (--only-needs-fix)",
    )
    p.add_argument(
        "--reuse-gpu-urls",
        default=None,
        help="Semicolon-separated synthesis GPU URLs (skip GPU sbatch)",
    )
    p.add_argument(
        "--num-synthesis-nodes",
        type=int,
        default=None,
        help="Override config slurm.num_synthesis_nodes",
    )
    args = p.parse_args()

    try:
        cfg = load_stage2_config(args.config)
    except (OSError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 2
    apply_runtime_env(cfg.runtime_env)

    try:
        return run_resynth(
            cfg,
            failed_only=args.failed_only,
            reuse_gpu_urls=args.reuse_gpu_urls,
            num_synthesis_nodes=args.num_synthesis_nodes,
        )
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
