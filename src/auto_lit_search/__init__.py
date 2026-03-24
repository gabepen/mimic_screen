"""
Automated literature search tools
"""

from __future__ import annotations
__all__ = [
    "mapping_run",
    "search_run",
    "collect_run",
    "analysis_run",
]



def __getattr__(name: str):
    if name == "mapping_run":
        from .mapping import run as mapping_run

        return mapping_run
    if name == "search_run":
        from .search import run as search_run

        return search_run
    if name == "collect_run":
        from .collect import run as collect_run

        return collect_run
    if name == "analysis_run":
        from .analysis import run as analysis_run
        return analysis_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")