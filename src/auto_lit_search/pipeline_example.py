"""
Example of how to use the automated literature search pipeline.

Mapping outputs only the lit-search table (Query/Target IDs + mapping columns);
this is the default intermediate for the full pipeline. Search adds PMID columns to it.
"""

import os
import pandas as pd
from auto_lit_search.mapping import run as mapping_run
from auto_lit_search.search import run as search_run


def run_pipeline(input_csv, output_csv, query_col="query", target_col="target"):
    """
    Run mapping then search. Single output file: lit-search table (intermediate format).

    Args:
        input_csv: Path to alignment_analysis results CSV
        output_csv: Path to lit-search output (Query/Target IDs + mapping + search columns)
        query_col: Column name for query UniProt IDs
        target_col: Column name for target UniProt IDs
    """
    print("Loading input data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    out_dir = os.path.dirname(os.path.abspath(output_csv))

    print("\n=== Module 1: Mapping ===")
    lit_df = mapping_run(df, output_dir=out_dir, cache_dir=out_dir,
                         query_col=query_col, target_col=target_col)
    lit_df.to_csv(output_csv, index=False)
    print(f"Saved lit-search table: {output_csv} ({len(lit_df)} rows)")

    print("\n=== Module 2: Search ===")
    lit_df = search_run(lit_df, query_col=query_col, output_dir=out_dir, cache_dir=out_dir)
    lit_df.to_csv(output_csv, index=False)
    print(f"Pipeline complete. Output: {output_csv}")

    return lit_df


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline_example.py <input_csv> <output_csv>")
        print("  output_csv = lit-search output (Query/Target IDs + mapping + search columns)")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    run_pipeline(input_csv, output_csv)


