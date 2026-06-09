# Automated Literature Search Pipeline

This package provides a 4-module pipeline for automated literature searches based on protein alignment analysis results.

## Pipeline Structure

The pipeline consists of 4 modules that process data sequentially:

1. **Mapping Module** - Converts UniProt IDs to Entrez Gene IDs
2. **Module 2** - [TBD]
3. **Module 3** - [TBD]
4. **Module 4** - [TBD]

Each module provides a standardized `run(df, **kwargs) -> DataFrame` interface for pipeline integration.

## Module 1: Mapping

The `mapping` module converts Protein IDs (UniProt) or Accession numbers to Entrez Gene IDs using the MyGene.info API.

#### Usage

**As part of the pipeline (recommended):**
```python
from auto_lit_search.mapping import run as mapping_run
import pandas as pd

# Load your alignment results
df = pd.read_csv('alignment_results.csv')

# Run mapping module (Module 1)
df = mapping_run(df, cache_file='entrez_mapping_cache.json')

# df now has query_entrez_id and target_entrez_id columns
# Pass to next module in pipeline...
```

**Direct function call:**
```python
from auto_lit_search.mapping import run
import pandas as pd

# Load your alignment results
df = pd.read_csv('alignment_results.csv')

# Run mapping
df = run(df, cache_file='entrez_mapping_cache.json')

# Save results
df.to_csv('alignment_results_with_entrez.csv', index=False)
```

**As a command-line script:**
```bash
python -m auto_lit_search.mapping \
    -i alignment_results.csv \
    -o alignment_results_with_entrez.csv \
    -c entrez_mapping_cache.json
```

#### Command-line Arguments

- `-i, --input`: Path to alignment_analysis.py results CSV file (required)
- `-o, --output`: Path to output CSV file with Entrez Gene ID columns added (required)
- `-c, --cache`: Path to JSON cache file for mappings (default: entrez_mapping_cache.json)
- `--query-col`: Name of column containing query UniProt IDs (default: query)
- `--target-col`: Name of column containing target UniProt IDs (default: target)
- `--batch-size`: Number of IDs to query per batch (default: 1000)

#### Features

- **Batch processing**: Queries MyGene.info API in batches to handle large datasets efficiently
- **Caching**: Saves mappings to a JSON cache file to avoid redundant API calls
- **Progress tracking**: Shows progress bars for long-running operations
- **Error handling**: Gracefully handles API errors and missing mappings

#### Output

The script adds two new columns to the input CSV:
- `query_entrez_id`: Entrez Gene ID for the query protein (UniProt ID)
- `target_entrez_id`: Entrez Gene ID for the target protein (UniProt ID)

If a mapping is not found, the value will be `None` (displayed as empty in CSV).

#### Dependencies

This module requires the `mygene` Python package:
```bash
pip install mygene
```

Or if using conda:
```bash
conda install -c bioconda mygene
```

## CPU-owned queue scheduler (download_node)

The CPU node owns orchestration state and keeps stage queues locally:

1. download package on CPU
2. if PDF conversion required, submit async Docling work
3. when package is fully converted, mark grader-ready
4. submit async grading when grader has capacity (writes `*_graded.json` only)
5. CPU dispatches synthesis to the LLM GPU node (`POST /run_alignment_graded`)

Docling, grader, and synthesis are independent stages. Slow grading does not block
Docling submission, except for each stage's own inflight cap.

### Restart semantics

Scheduler state is persisted under:

- `DATA_ROOT/logs/scheduler_state/<alignment_id>.json`

On restart, the CPU job reconstructs state from persisted files + output files:

- `<alignment>_results.json` + `<alignment>_graded.json` => `DONE`
- `<alignment>_graded.json` without results => `SYNTHESIS_READY` (CPU will re-dispatch synthesis)
- required Docling outputs present + no graded file => `GRADER_READY`
- missing required Docling outputs => `DOCLING_PENDING`

### Synthesis scorecard (v2)

Synthesis no longer emits `None` / `Minimal` / `Some` / `High`. Each alignment gets a structured `conclusion` in `*_results.json`:

| Field | Meaning |
|-------|---------|
| `pair_priority.score` / `tier` | Overall review priority (0–100; tiers A–E) |
| `host_exploitation` | Host manipulation / infection-pathway support |
| `query_effector` | Query effector / secretion evidence |
| `mimicry_plausibility` | Mimicry plausibility (rubric tags + host characterisation) |
| `headline` | Short plain-language summary |
| `synthesis_status` | `ok` (LLM + JSON), `grades_only` (rubric fallback), or `error` |

Regenerate summaries after resynthesis:

```bash
PYTHONPATH=src python3 scripts/summarize_results_table.py \
  --output-root /path/to/llm_results
```

Writes `results_summary.csv` sorted by `pair_priority_score`.

Backfill existing grades without regrading:

```bash
python3 scripts/fix_it_synthesis.py \
  --output-root /path/to/llm_results \
  --synthesis-url http://GPU_HOST:9000 \
  --instructions-file prompts/default_instructions.txt \
  --only-needs-fix --backup
```

### Staggered multi-grader discovery

The CPU job starts after the **first** grader is allocated (LLM and Docling must still be up). Additional grader GPU jobs register at runtime via `GRADER_JOB_IDS`. Each registered endpoint can run **one** alignment at a time; with five healthy endpoints, up to five alignments grade in parallel.

| Variable | Meaning |
|----------|---------|
| `GRADER_JOB_IDS` | **Colon-separated** Slurm job IDs (`33863201:33863202:…`). Do not use commas — `sbatch --export` truncates comma-separated values. |
| `GRADER_URLS` | Initial grader URL(s); separate multiple with `;` (not `,`) when passing through Slurm export. More URLs are appended via discovery. |
| `NUM_GRADER_NODES` | Expected grader count (for discovery status logs). |
| `GRADER_INFLIGHT_CAP` | Optional manual ceiling on total alignments grading at once; default is no global cap (only one job per endpoint). |

`download_node` polls `scontrol`/`squeue` each scheduler tick (`cpu_download_node.slurm` bind-mounts Slurm CLI + `libslurm` when `GRADER_JOB_IDS` is set). Watch for `grader endpoint registered` in the CPU log.

Without `GRADER_JOB_IDS`, all URLs in `GRADER_URLS` must be healthy at startup (static multi-grader mode).

### Minimal runtime knobs

| Variable | Default | Meaning |
|----------|---------|---------|
| `SERVICE_HEALTH_WAIT_SECONDS` | `900` | Max wait for LLM/Docling/Grader `/healthz` startup checks. |
| `DOCLING_CHUNK_SIZE` | `25` | PDFs per Docling conversion chunk. |
| `DOCLING_INFLIGHT_CAP` | `1` | Max async Docling jobs inflight from CPU scheduler. |
| `GRADER_INFLIGHT_CAP` | unset | Optional max alignments grading at once; default uses one slot per active endpoint. |
| `SCHEDULER_TICK_SECONDS` | `5` | Scheduler poll/admission tick cadence. |
| `STAGE_WATCHDOG_SECONDS` | `3600` | Fail inflight stage jobs that exceed this runtime. |
| `GRADER_SKIP_SYNTHESIS` | `1` | Grader writes graded JSON only; CPU owns synthesis dispatch. |
| `SYNTHESIS_HTTP_TIMEOUT` | `1800` | CPU → synthesis POST read timeout (seconds). |
| `SYNTHESIS_TOP_K_HOST` | `25` | Top target papers receiving text excerpts in final synthesis. |
| `SYNTHESIS_TOP_K_QUERY` | `10` | Top query papers receiving text excerpts. |
| `SYNTHESIS_EXCERPT_CHARS` | `12000` | Max chars of paper text per excerpted paper. |
| `SCORECARD_RUBRIC_BLEND_WEIGHT` | `0.6` | Blend rubric vs LLM integer scores in `conclusion`. |
| `SCORECARD_WEIGHT_HOST` | `0.40` | Rubric pair-priority weight: host exploitation. |
| `SCORECARD_WEIGHT_QUERY` | `0.35` | Rubric pair-priority weight: query effector. |
| `SCORECARD_WEIGHT_MIMICRY` | `0.15` | Rubric pair-priority weight: mimicry plausibility. |
| `SCORECARD_WEIGHT_DEPTH` | `0.10` | Rubric pair-priority weight: evidence depth bonus. |
| `GPU_ENABLE_LEGACY_RUN_ALIGNMENT` | `0` | Set `1` to allow deprecated `POST /run_alignment`. |
| `DOCLING_ENABLE_CALL_ANALYSIS` | `0` | Set `1` to allow Docling → legacy analysis node calls. |

### Grader -> vLLM settings

`grader_server` calls `VLLM_BASE_URL` (`/v1/chat/completions`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_BASE_URL` | (required) | Base URL of the local vLLM server. |
| `VLLM_MODEL_NAME` | auto from `/v1/models` | Served model id override. |
| `VLLM_HTTP_READ_TIMEOUT` | `300` | Per-attempt read timeout inside grader->vLLM call. |

