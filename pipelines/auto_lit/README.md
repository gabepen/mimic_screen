# Auto-lit pipeline

Runnable cluster pipeline for literature search, download, grading, and synthesis.

| Path | Purpose |
|------|---------|
| [`launchers/`](launchers/) | Shell/Python entrypoints (`run_lit_stage1.sh`, etc.) |
| [`configs/`](configs/) | YAML dataset + cluster configs |
| [`slurm/`](slurm/) | Slurm job scripts |
| [`scripts/`](scripts/) | Ops/maintenance CLIs (monitor, requeue, prune, …) |
| [`containers/`](containers/) | Dockerfiles for GPU/CPU images |
| [`prompts/`](prompts/) | Synthesis instruction templates |
| [`run_pipeline_slurm.py`](run_pipeline_slurm.py) | Stage2 Slurm submitter |

**Library code** lives in [`src/auto_lit_search/`](../../src/auto_lit_search/) (importable Python modules).

## Quick start

From the **mimic_screen repo root**:

```bash
pip install -e .   # optional; makes auto_lit_search importable

export AUTO_LIT_DATA_ROOT=/path/to/auto_lit_eval_data
export COLLECTOR_EMAIL='you@ucsc.edu'
export MAMBA_BIN=/path/to/mamba
export MODEL_DIR=/path/to/model-weights
export GRADER_MODEL_DIR=/path/to/grader-model-weights

./pipelines/auto_lit/launchers/run_lit_stage1.sh pipelines/auto_lit/configs/lp_human_stage1.yaml
./pipelines/auto_lit/launchers/run_lit_stage2.sh pipelines/auto_lit/configs/lp_human_stage2.yaml
```

See [`configs/README.md`](configs/README.md) for full config documentation.
