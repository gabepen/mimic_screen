"""Load YAML pipeline configs for stage1/stage2 launchers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

from auto_lit_search.env_config import auto_lit_data_root, auto_lit_pipeline_root, repo_root

try:
    import yaml
except ImportError as e:  # pragma: no cover - exercised when PyYAML missing
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = e
else:
    _YAML_IMPORT_ERROR = None

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for pipeline configs. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR


def expand_env_strings(value: Any) -> Any:
    """Recursively expand ``${VAR}`` from os.environ."""
    if isinstance(value, str):

        def _repl(m: re.Match[str]) -> str:
            name = m.group(1)
            env_val = os.environ.get(name, "")
            if not env_val:
                raise ValueError(f"Environment variable {name} is not set (required by config)")
            return env_val

        return _ENV_VAR_RE.sub(_repl, value)
    if isinstance(value, list):
        return [expand_env_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env_strings(v) for k, v in value.items()}
    return value


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), val)  # type: ignore[arg-type]
        else:
            out[key] = val
    return out


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    _require_yaml()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return raw


def load_config_dict(config_path: Path | str) -> Dict[str, Any]:
    """Load config with ``extends:`` chain and env expansion."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    data = _load_yaml_file(path)
    extends = data.pop("extends", None)
    merged: Dict[str, Any] = {}
    if extends:
        base_path = (path.parent / str(extends)).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"extends config not found: {base_path}")
        merged = load_config_dict(base_path)
    merged = dict(_deep_merge(merged, data))
    return expand_env_strings(merged)


def _resolve_path(
    raw: str,
    *,
    data_root: Path,
    repo: Path,
    kind: str,
) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if kind == "data":
        return (data_root / p).resolve()
    if kind == "repo":
        return (repo / p).resolve()
    raise ValueError(f"Unknown path kind: {kind}")


@dataclass(frozen=True)
class ClusterConfig:
    repo_root: Path
    pipeline_root: Path
    mamba_bin: Path
    conda_env: str
    model_dir: Path
    grader_model_dir: Path
    gpu_image: str
    docling_image: str
    grader_image: str
    cpu_image: str
    logs_dir: Path


@dataclass(frozen=True)
class Stage1Config:
    config_path: Path
    dataset: str
    data_root: Path
    alignments_csv: Path
    search_output_dir: Path
    idmap_csv: Path
    search_json: Path
    query_taxid: int
    target_taxid: int
    query_taxids: tuple[int, ...]
    target_taxids: tuple[int, ...]
    query_col: str
    target_col: str
    no_cache: bool
    accession_text_overlap: str
    run_mapping: bool
    run_search: bool
    cluster: ClusterConfig


@dataclass(frozen=True)
class Stage2SlurmConfig:
    num_grader_nodes: int
    num_synthesis_nodes: int
    gpu_script: Path
    docling_script: Path
    grader_script: Path
    cpu_script: Path
    gpu_port: int
    docling_port: int
    grader_port: int
    no_wait: bool


@dataclass(frozen=True)
class Stage2CollectionConfig:
    org: str
    auth_scope: str
    collector_email: str
    max_workers: int
    disable_semantic_scholar: bool


@dataclass(frozen=True)
class Stage2Config:
    config_path: Path
    dataset: str
    data_root: Path
    output_root: Path
    paper_ids_json: Path
    idmap_csv: Path
    host_rubric: Path
    microbe_rubric: Path
    instructions_file: Path
    slurm: Stage2SlurmConfig
    collection: Stage2CollectionConfig
    runtime_env: Dict[str, str]
    cluster: ClusterConfig


def _parse_cluster(raw: Dict[str, Any], data_root: Path) -> ClusterConfig:
    cluster = raw.get("cluster") or {}
    if not isinstance(cluster, dict):
        raise ValueError("cluster must be a mapping")

    repo = repo_root()
    repo_raw = cluster.get("repo_root")
    if repo_raw and str(repo_raw).strip().lower() not in ("auto", ""):
        repo = Path(str(repo_raw)).resolve()

    pipeline = auto_lit_pipeline_root()
    pipeline_raw = cluster.get("pipeline_root")
    if pipeline_raw and str(pipeline_raw).strip().lower() not in ("auto", ""):
        pipeline = Path(str(pipeline_raw)).resolve()

    mamba_bin = Path(str(cluster.get("mamba_bin") or "")).expanduser()
    if not mamba_bin:
        raise ValueError("cluster.mamba_bin is required")

    model_dir = Path(str(cluster.get("model_dir") or "")).expanduser()
    grader_model_dir = Path(
        str(cluster.get("grader_model_dir") or cluster.get("model_dir") or "")
    ).expanduser()
    if not model_dir.is_dir():
        raise ValueError(f"cluster.model_dir not found: {model_dir}")

    containers = cluster.get("containers") or {}
    if not isinstance(containers, dict):
        raise ValueError("cluster.containers must be a mapping")

    gpu_image = str(containers.get("gpu") or "")
    docling_image = str(containers.get("docling") or "")
    grader_image = str(containers.get("grader") or gpu_image)
    cpu_image = str(containers.get("cpu") or "")
    for name, val in (
        ("gpu", gpu_image),
        ("docling", docling_image),
        ("cpu", cpu_image),
    ):
        if not val:
            raise ValueError(f"cluster.containers.{name} is required")

    logs_raw = cluster.get("logs_dir")
    logs_dir = (
        Path(str(logs_raw)).expanduser()
        if logs_raw
        else (data_root / "logs").resolve()
    )

    return ClusterConfig(
        repo_root=repo,
        pipeline_root=pipeline,
        mamba_bin=mamba_bin,
        conda_env=str(cluster.get("conda_env") or "evorate"),
        model_dir=model_dir,
        grader_model_dir=grader_model_dir if grader_model_dir.is_dir() else model_dir,
        gpu_image=gpu_image,
        docling_image=docling_image,
        grader_image=grader_image,
        cpu_image=cpu_image,
        logs_dir=logs_dir,
    )


def _data_root_from_raw(raw: Dict[str, Any]) -> Path:
    dr = raw.get("data_root")
    if dr:
        return Path(str(dr)).expanduser().resolve()
    return auto_lit_data_root().resolve()


def load_stage1_config(
    config_path: Path | str,
    *,
    mapping_only: bool = False,
    search_only: bool = False,
) -> Stage1Config:
    path = Path(config_path).resolve()
    raw = load_config_dict(path)
    data_root = _data_root_from_raw(raw)
    cluster = _parse_cluster(raw, data_root)

    dataset = str(raw.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("dataset is required")

    stage1 = raw.get("stage1") or {}
    if not isinstance(stage1, dict):
        raise ValueError("stage1 must be a mapping")

    search_output_dir = _resolve_path(
        str(stage1.get("search_output_dir") or "search_results"),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="data",
    )
    alignments_csv = _resolve_path(
        str(stage1.get("alignments_csv") or ""),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="data",
    )
    if not str(stage1.get("alignments_csv") or "").strip():
        raise ValueError("stage1.alignments_csv is required")

    idmap_csv = search_output_dir / f"{dataset}_idmap.csv"
    search_json = search_output_dir / f"{dataset}_search.json"
    if stage1.get("idmap_csv"):
        idmap_csv = _resolve_path(
            str(stage1["idmap_csv"]),
            data_root=data_root,
            repo=cluster.pipeline_root,
            kind="data",
        )
    if stage1.get("search_json"):
        search_json = _resolve_path(
            str(stage1["search_json"]),
            data_root=data_root,
            repo=cluster.pipeline_root,
            kind="data",
        )

    run_mapping = bool(stage1.get("run_mapping", True))
    run_search = bool(stage1.get("run_search", True))
    if mapping_only:
        run_mapping, run_search = True, False
    if search_only:
        run_mapping, run_search = False, True

    def _taxids(side: str) -> tuple[int, ...]:
        plural = stage1.get(f"{side}_taxids")
        singular = stage1.get(f"{side}_taxid")
        raw_values = plural if plural is not None else singular
        if isinstance(raw_values, (list, tuple)):
            values = raw_values
        elif isinstance(raw_values, str) and "," in raw_values:
            values = [part.strip() for part in raw_values.split(",")]
        else:
            values = [raw_values]
        parsed = tuple(
            dict.fromkeys(int(value) for value in values if value is not None and str(value).strip())
        )
        if not parsed:
            raise ValueError(
                f"stage1.{side}_taxid or stage1.{side}_taxids is required"
            )
        if any(value <= 0 for value in parsed):
            raise ValueError(f"stage1.{side}_taxids must contain positive integers")
        return parsed

    query_taxids = _taxids("query")
    target_taxids = _taxids("target")

    return Stage1Config(
        config_path=path,
        dataset=dataset,
        data_root=data_root,
        alignments_csv=alignments_csv,
        search_output_dir=search_output_dir,
        idmap_csv=idmap_csv,
        search_json=search_json,
        # Mapping APIs still use the primary (first) taxid. Search uses the full
        # ordered list in one organism-scoped Europe PMC query.
        query_taxid=query_taxids[0],
        target_taxid=target_taxids[0],
        query_taxids=query_taxids,
        target_taxids=target_taxids,
        query_col=str(stage1.get("query_col") or "query"),
        target_col=str(stage1.get("target_col") or "target"),
        no_cache=bool(stage1.get("no_cache", False)),
        accession_text_overlap=str(stage1.get("accession_text_overlap") or ""),
        run_mapping=run_mapping,
        run_search=run_search,
        cluster=cluster,
    )


def load_stage2_config(config_path: Path | str) -> Stage2Config:
    path = Path(config_path).resolve()
    raw = load_config_dict(path)
    data_root = _data_root_from_raw(raw)
    cluster = _parse_cluster(raw, data_root)

    dataset = str(raw.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("dataset is required")

    stage2 = raw.get("stage2") or {}
    if not isinstance(stage2, dict):
        raise ValueError("stage2 must be a mapping")

    output_root = _resolve_path(
        str(stage2.get("output_root") or "llm_results"),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="data",
    )

    search_dir = data_root / "search_results"
    paper_ids_json = search_dir / f"{dataset}_search.json"
    idmap_csv = search_dir / f"{dataset}_idmap.csv"
    if stage2.get("paper_ids_json"):
        paper_ids_json = _resolve_path(
            str(stage2["paper_ids_json"]),
            data_root=data_root,
            repo=cluster.pipeline_root,
            kind="data",
        )
    if stage2.get("idmap_csv"):
        idmap_csv = _resolve_path(
            str(stage2["idmap_csv"]),
            data_root=data_root,
            repo=cluster.pipeline_root,
            kind="data",
        )

    host_rubric = _resolve_path(
        str(stage2.get("host_rubric") or ""),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="data",
    )
    microbe_rubric = _resolve_path(
        str(stage2.get("microbe_rubric") or ""),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="data",
    )
    instructions_file = _resolve_path(
        str(stage2.get("instructions_file") or ""),
        data_root=data_root,
        repo=cluster.pipeline_root,
        kind="repo",
    )
    if not str(stage2.get("host_rubric") or "").strip():
        raise ValueError("stage2.host_rubric is required")
    if not str(stage2.get("microbe_rubric") or "").strip():
        raise ValueError("stage2.microbe_rubric is required")
    if not str(stage2.get("instructions_file") or "").strip():
        raise ValueError("stage2.instructions_file is required")

    slurm_raw = stage2.get("slurm") or {}
    if not isinstance(slurm_raw, dict):
        raise ValueError("stage2.slurm must be a mapping")

    def _slurm_script(key: str, default: str) -> Path:
        val = str(slurm_raw.get(key) or default)
        return _resolve_path(val, data_root=data_root, repo=cluster.pipeline_root, kind="repo")

    slurm = Stage2SlurmConfig(
        num_grader_nodes=int(slurm_raw.get("num_grader_nodes") or 1),
        num_synthesis_nodes=int(slurm_raw.get("num_synthesis_nodes") or 1),
        gpu_script=_slurm_script("gpu_script", "slurm/gpu_llm_node.slurm"),
        docling_script=_slurm_script("docling_script", "slurm/gpu_docling_node.slurm"),
        grader_script=_slurm_script("grader_script", "slurm/gpu_grader_node.slurm"),
        cpu_script=_slurm_script("cpu_script", "slurm/cpu_download_node.slurm"),
        gpu_port=int(slurm_raw.get("gpu_port") or 9000),
        docling_port=int(slurm_raw.get("docling_port") or 9100),
        grader_port=int(slurm_raw.get("grader_port") or 9200),
        no_wait=bool(slurm_raw.get("no_wait", False)),
    )

    coll_raw = stage2.get("collection") or {}
    if not isinstance(coll_raw, dict):
        raise ValueError("stage2.collection must be a mapping")

    collector_email = str(coll_raw.get("collector_email") or os.environ.get("COLLECTOR_EMAIL") or "")
    collection = Stage2CollectionConfig(
        org=str(coll_raw.get("org") or "ucsc"),
        auth_scope=str(coll_raw.get("auth_scope") or "email_only"),
        collector_email=collector_email,
        max_workers=int(coll_raw.get("max_workers") or 2),
        disable_semantic_scholar=bool(coll_raw.get("disable_semantic_scholar", False)),
    )

    runtime_env_raw = stage2.get("runtime_env") or {}
    runtime_env: Dict[str, str] = {}
    if isinstance(runtime_env_raw, dict):
        runtime_env = {str(k): str(v) for k, v in runtime_env_raw.items()}

    return Stage2Config(
        config_path=path,
        dataset=dataset,
        data_root=data_root,
        output_root=output_root,
        paper_ids_json=paper_ids_json,
        idmap_csv=idmap_csv,
        host_rubric=host_rubric,
        microbe_rubric=microbe_rubric,
        instructions_file=instructions_file,
        slurm=slurm,
        collection=collection,
        runtime_env=runtime_env,
        cluster=cluster,
    )


def apply_runtime_env(runtime_env: Mapping[str, str]) -> None:
    """Set process environment from config runtime_env (does not unset existing keys)."""
    for key, val in runtime_env.items():
        if val:
            os.environ[key] = val
