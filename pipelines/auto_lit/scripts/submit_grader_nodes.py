#!/usr/bin/env python3
"""Submit one or more grader-only Slurm GPU jobs (no synthesis/docling/CPU)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from auto_lit_search.slurm_utils import (  # noqa: E402
    get_job_node,
    get_job_state,
    is_terminal_job_state,
)


def _sbatch(script_path: str, env: dict[str, str], log_path: str) -> str:
    cmd = ["sbatch", "--parsable", "--output", log_path]
    export_pairs = [f"{k}={v}" for k, v in env.items() if v is not None and v != ""]
    if export_pairs:
        cmd.append("--export=ALL," + ",".join(export_pairs))
    cmd.append(script_path)
    out = subprocess.check_output(cmd, text=True)
    return out.strip().split(";")[0]


def _sacct_job_summary(job_id: str) -> tuple[str | None, str | None, str | None]:
    """Return (state, exit_code, node) from sacct for a finished job."""
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "-n", "-P", "-o", "State,ExitCode,NodeList"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None, None
    fallback: tuple[str | None, str | None, str | None] | None = None
    for line in (out or "").strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        state, exit_code, node = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not state:
            continue
        row = state, exit_code or None, node or None
        if node:
            return row
        fallback = fallback or row
    return fallback or (None, None, None)


def _job_failed_message(job_id: str, logs_root: str, index: int) -> str:
    log_glob = os.path.join(logs_root, f"auto_lit_grader_{index}_{job_id}.log")
    sacct_state, exit_code, _node = _sacct_job_summary(job_id)
    parts = [f"grader job {job_id} is not running"]
    if sacct_state:
        parts.append(f"state={sacct_state}")
    if exit_code:
        parts.append(f"exit={exit_code}")
    parts.append(f"see {log_glob}")
    return " (" + ", ".join(parts[1:]) + ")" if len(parts) > 1 else ""


def _get_node_name(
    job_id: str,
    max_wait: int,
    *,
    logs_root: str,
    index: int,
    label: str = "",
) -> str | None:
    start = time.monotonic()
    last_status = ""
    last_heartbeat = start
    while time.monotonic() - start < max_wait:
        node = get_job_node(job_id)
        if node:
            return node

        state = get_job_state(job_id)
        if is_terminal_job_state(state):
            detail = _job_failed_message(job_id, logs_root, index)
            print(
                f"Grader job {job_id} entered terminal state {state}{detail}",
                file=sys.stderr,
            )
            return None

        sacct_state, exit_code, sacct_node = _sacct_job_summary(job_id)
        if sacct_state and sacct_state.upper() in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
        }:
            if sacct_state.upper() == "COMPLETED" and exit_code in (None, "0:0", "0"):
                return sacct_node
            detail = _job_failed_message(job_id, logs_root, index)
            print(
                f"Grader job {job_id} finished as {sacct_state} exit={exit_code}{detail}",
                file=sys.stderr,
            )
            return None

        status = state or sacct_state or "unknown"
        now = time.monotonic()
        if status != last_status:
            prefix = f"{label}: " if label else ""
            print(f"{prefix}Waiting for grader job {job_id} (state={status})")
            last_status = status
            last_heartbeat = now
        elif now - last_heartbeat >= 60:
            elapsed = int(now - start)
            prefix = f"{label}: " if label else ""
            print(
                f"{prefix}Still waiting for grader job {job_id} "
                f"(state={status}, elapsed={elapsed}s)"
            )
            last_heartbeat = now
        time.sleep(5)

    print(
        f"Timed out waiting for node on grader job {job_id}{_job_failed_message(job_id, logs_root, index)}",
        file=sys.stderr,
    )
    return None


def _healthz_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/healthz", timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read().decode("utf-8"))
            if not isinstance(body, dict):
                return False
            if body.get("status") != "ok":
                return False
            vllm = body.get("vllm", "ready")
            return vllm in ("ready", "disabled")
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False


def wait_for_grader_urls(
    job_ids: list[str],
    base_port: int,
    max_wait: int,
    *,
    logs_root: str,
) -> list[str]:
    n = len(job_ids)
    print(
        f"Waiting for {n} grader job(s) to get nodes and pass /healthz "
        f"(benchmark will not start until all are ready)",
        flush=True,
    )
    urls: list[str] = []
    for i, job_id in enumerate(job_ids):
        port = base_port + i
        label = f"Grader {i + 1}/{n}"
        node = _get_node_name(
            job_id,
            max_wait=max_wait,
            logs_root=logs_root,
            index=i,
            label=label,
        )
        if not node:
            continue
        url = f"http://{node}:{port}"
        print(f"{label}: node {node}, waiting for /healthz on port {port}...", flush=True)
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if _healthz_ok(url):
                urls.append(url)
                print(f"{label} ready: {url} (job {job_id})", flush=True)
                if len(urls) < n:
                    print(
                        f"{len(urls)}/{n} graders ready; "
                        f"benchmark has not started yet",
                        flush=True,
                    )
                break
            time.sleep(5)
        else:
            print(f"Timed out waiting for {url} (job {job_id})", file=sys.stderr)
    return urls


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--grader-model-dir", required=True)
    p.add_argument("--grader-script", required=True)
    p.add_argument("--grader-image", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--logs-root", required=True)
    p.add_argument("--num-nodes", type=int, default=4)
    p.add_argument("--grader-port", type=int, default=9200)
    p.add_argument("--wait-seconds", type=int, default=3600)
    p.add_argument("--wait", action="store_true", help="Wait for /healthz on all graders")
    p.add_argument("--json-out", default="", help="Write job ids + urls JSON here")
    args = p.parse_args()

    if args.num_nodes < 1:
        print("--num-nodes must be >= 1", file=sys.stderr)
        return 1
    if not os.path.isfile(args.grader_script):
        print(f"Grader script not found: {args.grader_script}", file=sys.stderr)
        return 1

    os.makedirs(args.logs_root, exist_ok=True)
    job_ids: list[str] = []
    for i in range(args.num_nodes):
        port = args.grader_port + i
        env = {
            "DATA_ROOT": args.data_root,
            "MODEL_DIR": args.grader_model_dir,
            "OUTPUT_ROOT": args.output_root,
            "GRADER_API_PORT": str(port),
            "GRADER_IMAGE": args.grader_image,
            "REPO_ROOT": args.repo_root,
            "GRADER_SKIP_SYNTHESIS": "1",
        }
        for key in (
            "GRADER_MAX_TOKENS",
            "GRADER_PAPER_WORKERS",
            "VLLM_MAX_NUM_SEQS",
            "VLLM_MAX_MODEL_LEN",
            "VLLM_GPU_MEMORY_UTILIZATION",
        ):
            val = os.environ.get(key, "").strip()
            if val:
                env[key] = val
        log_path = os.path.join(args.logs_root, f"auto_lit_grader_{i}_%j.log")
        job_id = _sbatch(args.grader_script, env, log_path=log_path)
        job_ids.append(job_id)
        print(f"Submitted grader {i + 1}/{args.num_nodes}: {job_id} port {port}")

    urls: list[str] = []
    if args.wait:
        urls = wait_for_grader_urls(
            job_ids,
            args.grader_port,
            args.wait_seconds,
            logs_root=args.logs_root,
        )
        if len(urls) < args.num_nodes:
            print(
                f"Only {len(urls)}/{args.num_nodes} graders healthy",
                file=sys.stderr,
            )
            for i, job_id in enumerate(job_ids):
                log_path = os.path.join(
                    args.logs_root,
                    f"auto_lit_grader_{i}_{job_id}.log",
                )
                if os.path.isfile(log_path):
                    print(f"  grader {i} log: {log_path}", file=sys.stderr)
            return 2

    payload: dict[str, Any] = {
        "job_ids": job_ids,
        "grader_urls": urls,
        "grader_port_base": args.grader_port,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
