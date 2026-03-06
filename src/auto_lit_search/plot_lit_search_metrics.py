import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_TYPES: List[Tuple[str, str, str]] = [
    ("entrez_id", "query_entrez_id", "target_entrez_id"),
    ("gene_name", "query_gene_name", "target_gene_name"),
    ("locus_tag", "query_locus_tag", "target_locus_tag"),
    ("genbank_acc", "query_genbank_acc", "target_genbank_acc"),
    ("common_name", "query_common_name", "target_common_name"),
]


def _nonempty_count(series: pd.Series) -> int:
    if series is None:
        return 0
    s = series.dropna().astype(str).str.strip()
    return (s != "").sum()


def _plot_id_completeness(mapping_csv: str, output_path: str) -> None:
    df = pd.read_csv(mapping_csv)

    id_labels: List[str] = []
    query_counts: List[int] = []
    target_counts: List[int] = []

    for label, q_col, t_col in ID_TYPES:
        if q_col not in df.columns and t_col not in df.columns:
            continue
        id_labels.append(label)
        query_counts.append(_nonempty_count(df[q_col]) if q_col in df.columns else 0)
        target_counts.append(_nonempty_count(df[t_col]) if t_col in df.columns else 0)

    if not id_labels:
        raise ValueError("No expected ID columns found in mapping CSV.")

    n_rows = len(df)
    x = np.arange(len(id_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, query_counts, width, label="query")
    ax.bar(x + width / 2, target_counts, width, label="target")

    ax.set_xticks(x)
    ax.set_xticklabels(id_labels)
    ax.set_ylabel("Count of rows with ID")
    ax.set_title(f"ID completeness by type (query vs target, n = {n_rows} rows)")
    ax.axhline(y=n_rows, color="gray", linestyle="--", linewidth=1, label="n rows")
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _detect_search_format(path: str, explicit: str | None) -> str:
    if explicit:
        return explicit

    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return "json"

    try:
        with open(path, "r", encoding="utf-8") as f:
            first_non_ws = ""
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
        if first_non_ws in ("{", "["):
            return "json"
    except OSError:
        pass

    return "csv"


def _load_search_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict] = []
    for query_id, entries in data.items():
        if not entries:
            continue
        first = entries[0]
        counts = first.get("query_paper_counts") or {}
        rows.append(
            {
                "query": query_id,
                "n_query_papers": first.get("n_query_papers", 0),
                "uniprot": counts.get("uniprot", 0),
                "text_pass1": counts.get("text_pass1", 0),
                "text_pass2": counts.get("text_pass2", 0),
            }
        )

    return pd.DataFrame(rows)


def _safe_parse_counts(val: object) -> Dict[str, int]:
    if isinstance(val, dict):
        return {
            "uniprot": int(val.get("uniprot", 0)),
            "text_pass1": int(val.get("text_pass1", 0)),
            "text_pass2": int(val.get("text_pass2", 0)),
        }
    if not isinstance(val, str) or not val:
        return {"uniprot": 0, "text_pass1": 0, "text_pass2": 0}

    text = val.strip()
    try:
        return _safe_parse_counts(json.loads(text))
    except Exception:
        import ast

        try:
            obj = ast.literal_eval(text)
            return _safe_parse_counts(obj)
        except Exception:
            return {"uniprot": 0, "text_pass1": 0, "text_pass2": 0}


def _load_search_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("Search CSV must contain a 'query' column.")

    if "n_query_papers" in df.columns:
        n_papers = df["n_query_papers"]
    elif "query_paper_dois" in df.columns:
        n_papers = df["query_paper_dois"].apply(
            lambda v: len(json.loads(v)) if isinstance(v, str) and v else 0
        )
    else:
        n_papers = pd.Series([0] * len(df), index=df.index)

    counts = df["query_paper_counts"].apply(_safe_parse_counts)
    counts_df = pd.DataFrame(list(counts))

    df_counts = pd.DataFrame(
        {
            "query": df["query"].astype(str),
            "n_query_papers": n_papers.astype(int),
            "uniprot": counts_df["uniprot"].astype(int),
            "text_pass1": counts_df["text_pass1"].astype(int),
            "text_pass2": counts_df["text_pass2"].astype(int),
        }
    )

    df_first = df_counts.groupby("query", as_index=False).first()
    return df_first


def _load_search_results(path: str, fmt: str | None) -> pd.DataFrame:
    fmt_resolved = _detect_search_format(path, fmt)
    if fmt_resolved == "json":
        return _load_search_json(path)
    if fmt_resolved == "csv":
        return _load_search_csv(path)
    raise ValueError(f"Unsupported search results format: {fmt_resolved}")


def _plot_papers_by_gene(
    search_path: str,
    output_path: str,
    search_format: str | None,
) -> None:
    df = _load_search_results(search_path, search_format)
    if df.empty:
        raise ValueError("Search results contain no rows.")

    papers = np.sort(df["n_query_papers"].to_numpy())[::-1]
    n_genes = len(papers)
    rank = np.arange(1, n_genes + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(rank, papers, where="mid", color="black", linewidth=0.7)
    ax.fill_between(rank, papers, alpha=0.25)
    ax.set_xlabel("Gene rank (by paper count)")
    ax.set_ylabel("Papers per gene")
    ax.set_title(f"Distribution of papers per gene (n = {n_genes} genes)")
    ax.set_ylim(-0.5, None)
    ax.set_xlim(0, n_genes + 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot metrics for the automated literature search pipeline."
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        required=True,
        help="Path to mapping output CSV (for ID completeness plot).",
    )
    parser.add_argument(
        "--search-results",
        type=str,
        required=True,
        help="Path to search output file (JSON or CSV).",
    )
    parser.add_argument(
        "--search-format",
        type=str,
        choices=["json", "csv"],
        default=None,
        help="Format of search results (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output figures (default: alongside mapping CSV).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    mapping_csv = os.path.abspath(args.mapping_csv)
    search_results = os.path.abspath(args.search_results)

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.dirname(mapping_csv) or "."

    _ensure_dir(out_dir)

    mapping_plot_path = os.path.join(out_dir, "mapping_id_completeness.png")
    search_plot_path = os.path.join(out_dir, "search_papers_by_gene.png")

    _plot_id_completeness(mapping_csv, mapping_plot_path)
    _plot_papers_by_gene(search_results, search_plot_path, args.search_format)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

