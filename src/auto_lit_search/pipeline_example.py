"""
Example of how to use the automated literature search pipeline.

Mapping -> Search -> Collect (optional) -> Analysis (optional).
Mapping outputs the lit-search table; Search adds paper columns and writes JSON.
Collect verifies gene mentions and writes high-confidence paper list.
Analysis downloads full texts and runs placeholder analysis.
"""

import json
import os
import pandas as pd
from auto_lit_search.mapping import run as mapping_run
from auto_lit_search.search import run as search_run, _result_df_to_query_keyed_json
from auto_lit_search.collect import run as collect_run
from auto_lit_search.analysis import run as analysis_run


def run_pipeline(
    input_csv,
    output_path,
    query_col="query",
    target_col="target",
    run_collect=True,
    run_analysis=True,
):
    """
    Run mapping, search, and optionally collect and analysis.

    Args:
        input_csv: Path to alignment_analysis results CSV
        output_path: Path to output JSON (query-keyed search results)
        query_col: Column name for query UniProt IDs
        target_col: Column name for target UniProt IDs
        run_collect: If True, run collect after search and write high-confidence CSV
        run_analysis: If True, run analysis on collect output and write analysis CSV
    """
    print("Loading input data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    base = output_path.replace(".json", "").rsplit(".", 1)[0] if ".json" in output_path else output_path.replace(".json", "")
    high_confidence_path = f"{base}_high_confidence_papers.csv"
    analysis_output_path = f"{base}_analysis.csv"

    print("\n=== Module 1: Mapping ===")
    lit_df = mapping_run(df, output_dir=out_dir, cache_dir=out_dir,
                         query_col=query_col, target_col=target_col)
    lit_df.to_csv(output_path.replace(".json", "_mapping.csv"), index=False)
    print(f"Saved mapping table ({len(lit_df)} rows)")

    print("\n=== Module 2: Search ===")
    lit_df = search_run(
        lit_df,
        query_id_col=query_col,
        target_id_col=target_col,
        output_dir=out_dir,
    )
    out_data = _result_df_to_query_keyed_json(
        lit_df, query_id_col=query_col, target_id_col=target_col
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Search output: {output_path}")

    if run_collect:
        print("\n=== Module 3: Collect ===")
        llm_queue_path = f"{base}_papers_for_llm.json"
        collect_df = collect_run(
            lit_df,
            query_id_col=query_col,
            target_id_col=target_col,
            output_dir=out_dir,
            output_llm_queue_path=llm_queue_path,
        )
        collect_df.to_csv(high_confidence_path, index=False)
        print(f"High-confidence papers: {high_confidence_path} ({len(collect_df)} rows)")

        if run_analysis and len(collect_df) > 0:
            print("\n=== Module 4: Analysis ===")
            analysis_input = high_confidence_path
            if "validated_by_litotar" in collect_df.columns and "validated_by_pubtator" in collect_df.columns:
                validated = collect_df[collect_df["validated_by_litotar"] | collect_df["validated_by_pubtator"]]
                if len(validated) > 0:
                    analysis_input = validated
            analysis_df = analysis_run(
                analysis_input,
                output_dir=out_dir,
            )
            analysis_df.to_csv(analysis_output_path, index=False)
            print(f"Analysis output: {analysis_output_path}")

    return lit_df


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pipeline_example.py <input_csv> <output_json> [--no-collect] [--no-analysis]")
        print("  output_json = path for query-keyed search results JSON")
        print("  --no-collect  skip collect (verification) step")
        print("  --no-analysis skip analysis (full-text) step")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_path = sys.argv[2]
    if not output_path.endswith(".json"):
        output_path = output_path.rstrip("/") + ".json"
    run_collect = "--no-collect" not in sys.argv
    run_analysis = "--no-analysis" not in sys.argv

    run_pipeline(input_csv, output_path, run_collect=run_collect, run_analysis=run_analysis)


