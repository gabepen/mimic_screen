# Auto-lit pipeline

Runnable cluster pipeline for literature search, download, grading, and synthesis.

| Path | Purpose |
|------|---------|
| [`launchers/`](launchers/) | Shell/Python entrypoints (`run_lit_stage1.sh`, etc.) |
| [`configs/`](configs/) | YAML dataset + cluster configs |
| [`slurm/`](slurm/) | Slurm job scripts |
| [`scripts/`](scripts/) | Ops/maintenance CLIs (monitor, requeue, prune, …) |
| [`containers/`](containers/) | Dockerfiles for GPU/CPU images |
| [`prompts/`](prompts/) | System-specific synthesis research questions (no shared default) |
| [`run_pipeline_slurm.py`](run_pipeline_slurm.py) | Stage2 Slurm submitter |

**Library code** lives in [`src/auto_lit_search/`](../../src/auto_lit_search/) (importable Python modules).

## Quick start

From the **mimic_screen repo root**:

```bash
# Conda env for stage 1 Slurm jobs (once)
mamba create -n auto_lit python=3.12 -y
mamba activate auto_lit
pip install -e ".[auto-lit]"

export AUTO_LIT_DATA_ROOT=/path/to/auto_lit_eval_data
export COLLECTOR_EMAIL='you@ucsc.edu'
export MAMBA_BIN=/path/to/mamba
export MODEL_DIR=/path/to/model-weights
export GRADER_MODEL_DIR=/path/to/grader-model-weights

./pipelines/auto_lit/launchers/run_lit_stage1.sh pipelines/auto_lit/configs/lp_human_stage1.yaml
./pipelines/auto_lit/launchers/run_lit_stage2.sh pipelines/auto_lit/configs/lp_human_stage2.yaml
```

See [`configs/README.md`](configs/README.md) for full config documentation.

For symbol+alias search reruns and incremental stage2 sync after term fixes, see [`docs/symbol_alias_stage1_rerun.md`](docs/symbol_alias_stage1_rerun.md).
