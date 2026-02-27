"""
Automated literature search tools for protein alignment analysis results.

This package provides a 4-module pipeline for automated literature searches:
1. Mapping: Convert UniProt IDs to Entrez Gene IDs + fallback identifiers
2. Search: Query PubMed for query proteins via Entrez
3. [Module 3 - TBD]
4. [Module 4 - TBD]

Each module provides a `run(df, **kwargs) -> DataFrame` function for pipeline integration.
"""

from .mapping import run as mapping_run
from .search import run as search_run

__all__ = [
    'mapping_run',
    'search_run',
]

