#!/usr/bin/env python3
"""Submit stage1 Slurm jobs (mapping + EPMC search) from a YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from auto_lit_search.env_config import slurm_mail_user
from auto_lit_search.pipeline_config import Stage1Config, load_stage1_config  # noqa: E402


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


def run_stage1(cfg: Stage1Config) -> int:
    repo = cfg.cluster.repo_root
    pipeline = cfg.cluster.pipeline_root
    mapping_script = pipeline / "slurm" / "mapping_node.slurm"
    search_script = pipeline / "slurm" / "search_node.slurm"
    logs_dir = cfg.cluster.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.search_output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.run_mapping and not cfg.alignments_csv.is_file():
        print(f"Input alignment CSV not found: {cfg.alignments_csv}", file=sys.stderr)
        return 1

    if cfg.run_search and not cfg.run_mapping and not cfg.idmap_csv.is_file():
        print(f"Mapping output not found: {cfg.idmap_csv}", file=sys.stderr)
        print("Run mapping first or use --mapping-only with a completed idmap.", file=sys.stderr)
        return 1

    mapping_export = {
        "REPO_ROOT": str(repo),
        "PIPELINE_ROOT": str(pipeline),
        "INPUT_CSV": str(cfg.alignments_csv),
        "OUTPUT_IDMAP_CSV": str(cfg.idmap_csv),
        "OUTPUT_DIR": str(cfg.search_output_dir),
        "QUERY_TAXID": str(cfg.query_taxid),
        "TARGET_TAXID": str(cfg.target_taxid),
        "QUERY_COL": cfg.query_col,
        "TARGET_COL": cfg.target_col,
        "CONDA_ENV": cfg.cluster.conda_env,
        "MAMBA_BIN": str(cfg.cluster.mamba_bin),
        "NO_CACHE": "1" if cfg.no_cache else "",
    }

    search_export = {
        "REPO_ROOT": str(repo),
        "PIPELINE_ROOT": str(pipeline),
        "INPUT_IDMAP_CSV": str(cfg.idmap_csv),
        "OUTPUT_SEARCH_JSON": str(cfg.search_json),
        "OUTPUT_DIR": str(cfg.search_output_dir),
        "QUERY_TAXID": str(cfg.query_taxid),
        "TARGET_TAXID": str(cfg.target_taxid),
        "QUERY_COL": cfg.query_col,
        "TARGET_COL": cfg.target_col,
        "CONDA_ENV": cfg.cluster.conda_env,
        "MAMBA_BIN": str(cfg.cluster.mamba_bin),
        "NO_CACHE": "1" if cfg.no_cache else "",
        "ACCESSION_TEXT_OVERLAP": cfg.accession_text_overlap,
    }

    mapping_job_id = ""
    if cfg.run_mapping:
        mapping_log = logs_dir / "auto_lit_mapping_%j.log"
        mapping_job_id = _sbatch(
            mapping_script,
            mapping_export,
            log_path=mapping_log,
        )
        print(f"Submitted mapping job: {mapping_job_id}")
        print(f"  input:  {cfg.alignments_csv}")
        print(f"  output: {cfg.idmap_csv}")
        print(f"  log:    {logs_dir / f'auto_lit_mapping_{mapping_job_id}.log'}")

    if cfg.run_search:
        search_log = logs_dir / "auto_lit_search_%j.log"
        dep = mapping_job_id if mapping_job_id else None
        search_job_id = _sbatch(
            search_script,
            search_export,
            dependency=dep,
            log_path=search_log,
        )
        print(f"Submitted search job: {search_job_id}")
        print(f"  input:  {cfg.idmap_csv}")
        print(f"  output: {cfg.search_json}")
        print(f"  log:    {logs_dir / f'auto_lit_search_{search_job_id}.log'}")
        if mapping_job_id:
            print(f"  depends on mapping job: {mapping_job_id}")

    print()
    print("Stage 1 outputs (for stage2 config):")
    print(f"  paper_ids_json: {cfg.search_json}")
    print(f"  idmap_csv:      {cfg.idmap_csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Stage1 YAML config path")
    p.add_argument("--mapping-only", action="store_true", help="Run mapping only")
    p.add_argument("--search-only", action="store_true", help="Run search only (requires idmap)")
    args = p.parse_args()

    if args.mapping_only and args.search_only:
        print("Cannot use --mapping-only and --search-only together", file=sys.stderr)
        return 2

    try:
        cfg = load_stage1_config(
            args.config,
            mapping_only=args.mapping_only,
            search_only=args.search_only,
        )
    except (OSError, ValueError, RuntimeError) as e:
        print(str(e), file=sys.stderr)
        return 2

    return run_stage1(cfg)


if __name__ == "__main__":
    sys.exit(main())
