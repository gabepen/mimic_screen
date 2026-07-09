"""Shared environment variable helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """mimic_screen repo root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def auto_lit_pipeline_root() -> Path:
    """Runtime assets for the auto-lit pipeline (slurm, configs, launchers)."""
    return repo_root() / "pipelines" / "auto_lit"


def auto_lit_data_root() -> Path:
    """Root of auto_lit_eval_data (env, sibling checkout, or repo-local fallback)."""
    raw = os.environ.get("AUTO_LIT_DATA_ROOT", "").strip()
    if raw:
        return Path(raw)
    sibling = repo_root().parents[1] / "auto_lit_eval_data"
    if sibling.is_dir():
        return sibling
    return repo_root() / "auto_lit_eval_data"


def rubrics_dir() -> Path:
    return auto_lit_data_root() / "rubrics"


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw)


def default_host_rubric_path() -> Path:
    """Host/target rubric JSON from ``HOST_RUBRIC_PATH``."""
    path = _path_from_env("HOST_RUBRIC_PATH")
    if path is None:
        raise RuntimeError(
            "HOST_RUBRIC_PATH is not set. Export it or pass --host-rubric-path / "
            f"host_rubric in a stage2 config (files under {rubrics_dir()})."
        )
    return path


def default_microbe_rubric_path() -> Path:
    """Query/microbe rubric JSON from ``MICROBE_RUBRIC_PATH``."""
    path = _path_from_env("MICROBE_RUBRIC_PATH")
    if path is None:
        raise RuntimeError(
            "MICROBE_RUBRIC_PATH is not set. Export it or pass --microbe-rubric-path / "
            f"microbe_rubric in a stage2 config (files under {rubrics_dir()})."
        )
    return path


def resolve_rubric_paths(
    *,
    host: Path | str | None = None,
    microbe: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve rubric paths: explicit args override env vars."""
    host_path = Path(host) if host else default_host_rubric_path()
    microbe_path = Path(microbe) if microbe else default_microbe_rubric_path()
    return host_path, microbe_path


def slurm_mail_user() -> str | None:
    """Slurm notification address from ``SLURM_MAIL_USER`` or ``COLLECTOR_EMAIL``."""
    for name in ("SLURM_MAIL_USER", "COLLECTOR_EMAIL"):
        raw = os.environ.get(name, "").strip()
        if raw:
            return raw
    return None


def env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
        return v if v > 0 else default
    except ValueError:
        return default


def env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip(), 10)
        return v if v >= 1 else default
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
