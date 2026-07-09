# Auto-lit pipeline configs

YAML configs drive stage1 (mapping + EPMC search) and stage2 (GPU + CPU analysis).
Requires **PyYAML**: `pip install pyyaml` (or install in your `evorate` conda env).

## Setup

1. Export required environment variables (no hardcoded paths in committed configs):

   ```bash
   export AUTO_LIT_DATA_ROOT=/path/to/auto_lit_eval_data
   export COLLECTOR_EMAIL='you@ucsc.edu'
   export MAMBA_BIN=/path/to/mamba
   export MODEL_DIR=/path/to/model-weights
   export GRADER_MODEL_DIR=/path/to/grader-model-weights
   ```

   Optional: `export SLURM_MAIL_USER="$COLLECTOR_EMAIL"` for job failure notifications.

2. Copy [`cluster_ucsc.yaml`](cluster_ucsc.yaml) if you need site-specific container images or conda env name.

3. Optional publisher tokens in repo-root `.env.publishers` (gitignored):

   ```bash
   export ELS_API_KEY='...'
   export TDM_API_TOKEN='...'
   ```

## Run

From the **mimic_screen repo root**:

```bash
# Stage 1
./pipelines/auto_lit/launchers/run_lit_stage1.sh pipelines/auto_lit/configs/lp_human_stage1.yaml

# Stage 2 (full pipeline)
./pipelines/auto_lit/launchers/run_lit_stage2.sh pipelines/auto_lit/configs/lp_human_stage2.yaml

# CPU only (after GPUs are up)
./pipelines/auto_lit/launchers/run_lit_cpu_manual.sh pipelines/auto_lit/configs/lp_human_stage2.yaml <gpu_jobs...> <docling_job> <grader_jobs...>

# Resynthesis
./pipelines/auto_lit/launchers/run_lit_resynth.sh pipelines/auto_lit/configs/lp_human_stage2.yaml
./pipelines/auto_lit/launchers/run_lit_resynth.sh pipelines/auto_lit/configs/lp_human_stage2.yaml --failed-only

# Monitor
./pipelines/auto_lit/launchers/monitor_lit_pipeline.sh pipelines/auto_lit/configs/lp_human_stage2.yaml --watch 30
```

## Config layout

| File | Purpose |
|------|---------|
| `cluster_ucsc.yaml` | Shared cluster paths (models, containers, mamba) |
| `*_stage1.yaml` | Alignments CSV, taxids, search output dir |
| `*_stage2.yaml` | Rubrics, instructions, Slurm scale, collection |

Stage2 derives `paper_ids_json` and `idmap_csv` from `dataset`:

- `{data_root}/search_results/{dataset}_search.json`
- `{data_root}/search_results/{dataset}_idmap.csv`

Override with explicit `stage2.paper_ids_json` / `stage2.idmap_csv` if needed.

## New dataset

1. Duplicate `lp_human_stage1.yaml` / `lp_human_stage2.yaml`.
2. Set `dataset`, `alignments_csv`, taxids, rubrics, `output_root`.
3. Place alignment CSV under `{AUTO_LIT_DATA_ROOT}/inputs/`.
