"""Pytest configuration: ensure ``src/`` is importable."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from auto_lit_search.env_config import rubrics_dir  # noqa: E402

# Tests use Legionella lp-human rubrics when present; override via env for other datasets.
_rubrics = rubrics_dir()
_legionella_host = _rubrics / "host_rubric_v1.json"
_legionella_microbe = _rubrics / "legionella_rubric.json"
if _legionella_host.is_file() and _legionella_microbe.is_file():
    os.environ.setdefault("HOST_RUBRIC_PATH", str(_legionella_host))
    os.environ.setdefault("MICROBE_RUBRIC_PATH", str(_legionella_microbe))
