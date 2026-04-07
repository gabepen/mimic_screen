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

## Grader node (`grader_server`) and vLLM HTTP settings

The grader calls `VLLM_BASE_URL` (OpenAI-compatible `/v1/chat/completions`). Tune timeouts when vLLM is busy (queue + generation can exceed a few minutes).

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_BASE_URL` | (required) | Base URL of the vLLM server (e.g. `http://127.0.0.1:8000`). |
| `VLLM_MODEL_NAME` | auto from `/v1/models` | Must match a served model id; wrong names cause 404s and empty replies. |
| `VLLM_HTTP_CONNECT_TIMEOUT` | `30` | TCP connect timeout (seconds). |
| `VLLM_HTTP_READ_TIMEOUT` | `300` | Per-attempt read timeout (seconds). Alias: `VLLM_GRADER_TIMEOUT`. For heavy runs try `900`–`1800`. |
| `VLLM_GRADER_HTTP_RETRIES` | `3` | Attempts per LLM call on `ReadTimeout` / `ConnectionError` (exponential backoff between attempts). |
| `VLLM_GRADER_RETRY_BACKOFF_SEC` | `45` | Base sleep (seconds) before retry; actual wait is `min(base * 2**attempt, cap)`. |
| `VLLM_GRADER_RETRY_BACKOFF_CAP_SEC` | `180` | Max sleep between retries (seconds). |

Example for a loaded GPU host:

```bash
export VLLM_HTTP_READ_TIMEOUT=1200
export VLLM_GRADER_HTTP_RETRIES=4
export VLLM_GRADER_RETRY_BACKOFF_SEC=60
```

