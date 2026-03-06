"""
Automated literature search tools for protein alignment analysis results.

This package provides a 4-module pipeline for automated literature searches:
1. Mapping: Convert UniProt IDs to Entrez Gene IDs + fallback identifiers
2. Search: Europe PMC two-phase search (UniProt + text) for query and target
3. Collect: Verify explicit gene mentions; output high-confidence paper list
4. Analysis: Download full texts and run in-depth analysis (placeholder)

Each module provides a `run(df_or_path, **kwargs)` function for pipeline integration.
"""

from .mapping import run as mapping_run
from .search import run as search_run
from .collect import run as collect_run
from .analysis import run as analysis_run

__all__ = [
    "mapping_run",
    "search_run",
    "collect_run",
    "analysis_run",
]

