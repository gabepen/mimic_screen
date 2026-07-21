"""Tests for pipeline YAML config loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from auto_lit_search.pipeline_config import (
    expand_env_strings,
    load_config_dict,
    load_stage1_config,
    load_stage2_config,
)

pytest.importorskip("yaml")


def test_expand_env_strings():
    os.environ["TEST_PIPELINE_VAR"] = "/tmp/data"
    try:
        assert expand_env_strings("${TEST_PIPELINE_VAR}/x") == "/tmp/data/x"
    finally:
        del os.environ["TEST_PIPELINE_VAR"]


def test_expand_env_strings_missing_raises():
    with pytest.raises(ValueError, match="TEST_MISSING_VAR"):
        expand_env_strings("${TEST_MISSING_VAR}")


def test_extends_merge(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "cluster:\n  conda_env: evorate\n  containers:\n    gpu: img-gpu\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "dataset: test-ds\n"
        "stage1:\n  query_taxid: 1\n",
        encoding="utf-8",
    )
    merged = load_config_dict(child)
    assert merged["dataset"] == "test-ds"
    assert merged["cluster"]["conda_env"] == "evorate"
    assert merged["stage1"]["query_taxid"] == 1


def _write_cluster_config(
    tmp_path: Path, data_root: Path, model_dir: Path, pipeline: Path | None = None
) -> Path:
    cluster = tmp_path / "cluster.yaml"
    pipeline_line = f"  pipeline_root: {pipeline}\n" if pipeline else ""
    cluster.write_text(
        f"""
cluster:
  mamba_bin: {tmp_path / "mamba"}
  conda_env: test
  model_dir: {model_dir}
  grader_model_dir: {model_dir}
{pipeline_line}  containers:
    gpu: docker://test/gpu
    docling: docker://test/docling
    cpu: docker://test/cpu
""",
        encoding="utf-8",
    )
    (tmp_path / "mamba").write_text("", encoding="utf-8")
    return cluster


def test_load_stage1_config(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    inputs = data_root / "inputs"
    inputs.mkdir()
    align = inputs / "ds_alignments.csv"
    align.write_text("query,target\n", encoding="utf-8")
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    cluster = _write_cluster_config(tmp_path, data_root, model_dir)
    cfg_path = tmp_path / "stage1.yaml"
    cfg_path.write_text(
        f"""
extends: {cluster.name}
dataset: my-ds
data_root: {data_root}
stage1:
  alignments_csv: inputs/my-ds_alignments.csv
  search_output_dir: search_results
  query_taxids: [272624, 446]
  target_taxid: 9606
  query_organism_terms: [Legionella pneumophila, Legionella]
  target_organism_terms: [Homo sapiens, human]
""",
        encoding="utf-8",
    )
    # fix alignments path name
    align.rename(inputs / "my-ds_alignments.csv")

    cfg = load_stage1_config(cfg_path)
    assert cfg.dataset == "my-ds"
    assert cfg.idmap_csv == data_root / "search_results" / "my-ds_idmap.csv"
    assert cfg.search_json == data_root / "search_results" / "my-ds_search.json"
    assert cfg.query_taxid == 272624
    assert cfg.query_taxids == (272624, 446)
    assert cfg.target_taxid == 9606
    assert cfg.target_taxids == (9606,)
    assert cfg.query_organism_terms == ("Legionella pneumophila", "Legionella")
    assert cfg.target_organism_terms == ("Homo sapiens", "human")


def test_load_stage2_derived_paths(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "search_results").mkdir()
    rubrics = data_root / "rubrics"
    rubrics.mkdir()
    (rubrics / "host.json").write_text("{}", encoding="utf-8")
    (rubrics / "microbe.json").write_text("{}", encoding="utf-8")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    prompts = pipeline / "prompts"
    prompts.mkdir()
    (prompts / "instr.txt").write_text("test", encoding="utf-8")
    slurm = pipeline / "slurm"
    slurm.mkdir()
    for name in ("gpu_llm_node.slurm", "gpu_docling_node.slurm", "gpu_grader_node.slurm", "cpu_download_node.slurm"):
        (slurm / name).write_text("#!/bin/bash\n", encoding="utf-8")

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    cluster = _write_cluster_config(tmp_path, data_root, model_dir, pipeline=pipeline)
    cfg_path = tmp_path / "stage2.yaml"
    cfg_path.write_text(
        f"""
extends: {cluster.name}
dataset: my-ds
data_root: {data_root}
stage2:
  output_root: llm_results/out
  host_rubric: rubrics/host.json
  microbe_rubric: rubrics/microbe.json
  instructions_file: prompts/instr.txt
  collection:
    collector_email: test@example.com
""",
        encoding="utf-8",
    )

    cfg = load_stage2_config(cfg_path)
    assert cfg.paper_ids_json == data_root / "search_results" / "my-ds_search.json"
    assert cfg.idmap_csv == data_root / "search_results" / "my-ds_idmap.csv"
    assert cfg.output_root == data_root / "llm_results" / "out"
