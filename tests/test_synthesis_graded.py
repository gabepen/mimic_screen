"""Unit tests for dual-track synthesis selection and batch parsing."""

from __future__ import annotations

import json
from types import SimpleNamespace

from auto_lit_search import synthesis_graded as sg


def _gp(
    file_name: str,
    role: str,
    relevance: float,
    *,
    max_axis: float = 0.0,
    no_meaningful_mention: bool = False,
) -> SimpleNamespace:
    axis = max(max_axis, relevance)
    return SimpleNamespace(
        paper_id=file_name,
        file_name=file_name,
        paper_role=role,
        relevance_grade=relevance,
        rubric_dimension_scores={"axis_a": axis},
        rubric_axis_rationales={"axis_a": "note"},
        criterion_scores={},
        rationale="",
        rubric_tags={},
        no_meaningful_mention=no_meaningful_mention,
    )


def test_select_top_k_per_role_caps_and_sorts_within_role():
    kept = []
    for i in range(50):
        kept.append(_gp(f"host_{i}.txt", "target", float(50 - i) / 100.0))
    for i in range(30):
        kept.append(_gp(f"query_{i}.txt", "query", float(30 - i) / 100.0))

    batch_pool, excerpt_pool, host_pool, query_pool = sg._select_top_k_per_role(
        kept, top_k_host=40, top_k_query=40
    )

    assert len(host_pool) == 40
    assert len(query_pool) == 30
    assert len(batch_pool) == 70
    assert host_pool[0].file_name == "host_0.txt"
    assert host_pool[-1].file_name == "host_39.txt"
    assert query_pool[0].file_name == "query_0.txt"
    assert len(excerpt_pool) == 70


def test_parse_batch_summary_output_extracts_batch_summary():
    batch = [_gp("a.txt", "target", 0.5)]
    raw = json.dumps(
        {
            "paper_summaries": [
                {
                    "file_name": "a.txt",
                    "summary": "Host pathway evidence.",
                    "important_points": ["axis_a high"],
                    "confidence_notes": "ok",
                }
            ],
            "memory_updates": ["remember axis_a"],
            "batch_summary": "Cumulative host track: pathway overlap noted.",
        }
    )
    out = sg._parse_batch_summary_output(raw, batch)
    assert out["batch_summary"] == "Cumulative host track: pathway overlap noted."
    assert len(out["paper_summaries"]) == 1


def test_fit_synth_prompt_to_budget_trims_excerpts():
    marker = "\nTop-paper text excerpts (verify grader claims; pair-level bridge):\n"
    blocks = "".join(
        f"\n--- Excerpt: paper_{i}.txt ---\n{'x' * 5000}\n--- end excerpt ---\n"
        for i in range(20)
    )
    excerpt_sections = marker + blocks
    prefix = "instructions and summaries\n" + ("y" * 2000)
    prompt = prefix + excerpt_sections
    fitted, notes = sg._fit_synth_prompt_to_budget(
        prompt, excerpt_sections, max_input_tokens=5000
    )
    assert sg._estimate_prompt_tokens(fitted) <= 5000
    assert "trimmed" in notes or "truncated" in notes
    assert len(fitted) < len(prompt)


def test_build_excerpt_sections_empty_without_papers():
    sections, meta, stats = sg._build_excerpt_sections(
        [],
        "/tmp",
        query_id="Q1",
        target_id="T1",
        gene_context=None,
        per_paper_chars=3000,
        total_chars=70000,
        mention_cluster_gap=50,
        min_mention_chars=400,
        max_mention_chars=8000,
        max_sites=4,
        no_mention_fallback_chars=1500,
    )
    assert sections == ""
    assert meta == []
    assert stats["papers"] == 0


def test_build_excerpt_sections_mention_centered(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    text = (
        "Introduction without gene.\n\n"
        "MRPL19 was knocked down. Results show strong phenotype in cells."
    )
    (papers_dir / "host_0.txt").write_text(text, encoding="utf-8")
    gp = _gp("host_0.txt", "target", 0.9, max_axis=0.8)
    gene_context = {
        "target": {"gene_name": "MRPL19", "common_name": "mitochondrial"},
        "query": {"gene_name": "geneA"},
    }
    sections, meta, stats = sg._build_excerpt_sections(
        [gp],
        str(papers_dir),
        query_id="P11111",
        target_id="Q22222",
        gene_context=gene_context,
        per_paper_chars=3000,
        total_chars=70000,
        mention_cluster_gap=50,
        min_mention_chars=400,
        max_mention_chars=8000,
        max_sites=4,
        no_mention_fallback_chars=1500,
    )
    assert "MRPL19" in sections
    assert "genes_found_in_text:" in sections
    assert meta[0]["excerpt_mode"] == "mentions"
    assert stats["papers"] == 1


def test_run_role_batch_chain_passes_prior_summary_to_second_batch():
    papers = [_gp(f"h{i}.txt", "target", 0.9 - i * 0.01) for i in range(6)]
    prompts: list[str] = []

    def fake_llm(prompt: str, base_url: str, max_tokens: int, temperature: float) -> str:
        prompts.append(prompt)
        batch_files = [
            line.split()[1]
            for line in prompt.splitlines()
            if line.startswith("- ") and " role=" in line
        ]
        return json.dumps(
            {
                "paper_summaries": [
                    {
                        "file_name": fn,
                        "summary": f"Summary for {fn}",
                        "important_points": ["pt"],
                        "confidence_notes": "",
                    }
                    for fn in batch_files
                ],
                "memory_updates": ["mem"],
                "batch_summary": f"Rollup after {batch_files[0]}",
            }
        )

    running, memory, summaries, outputs = sg._run_role_batch_chain(
        "target",
        papers,
        alignment_id="TEST_ALIGN",
        instructions="instr",
        term_block="terms",
        host_rubric=None,
        microbe_rubric=None,
        batch_size=3,
        per_axis_cap=200,
        prior_summary_max_chars=2500,
        max_tokens=1024,
        temperature=0.0,
        call_llm=fake_llm,
        llm_base_url="http://fake",
    )

    assert len(prompts) == 2
    assert "Prior batch summary for host track:" in prompts[1]
    assert "Rollup after h0.txt" in prompts[1]
    assert running == "Rollup after h3.txt"
    assert len(summaries) == 6
    assert len(outputs) == 2


def test_run_alignment_graded_dual_track_final_prompt(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    graded = []
    for i in range(3):
        fn = f"host_{i}.txt"
        (papers_dir / fn).write_text(
            "MRPL19 pathway study with host exploitation evidence.",
            encoding="utf-8",
        )
        graded.append(_gp(fn, "target", 0.8 - i * 0.1, max_axis=0.7))
    for i in range(2):
        fn = f"query_{i}.txt"
        (papers_dir / fn).write_text(
            "geneA effector translocation into host cells.",
            encoding="utf-8",
        )
        graded.append(_gp(fn, "query", 0.6 - i * 0.1, max_axis=0.5))

    req = SimpleNamespace(
        alignment_id="PAIR_TEST",
        papers_dir=str(papers_dir),
        query="Q1",
        target_id="T1",
        instructions="test instructions",
        output_root=str(tmp_path),
        graded_papers=graded,
        grading_meta={},
        constraints=None,
        gene_context={
            "query": {"gene_name": "geneA"},
            "target": {"gene_name": "MRPL19"},
        },
    )

    def fake_write(request, payload):
        return SimpleNamespace(
            status="ok",
            alignment_id=request.alignment_id,
            results_path=str(tmp_path / f"{request.alignment_id}_results.json"),
        )

    sg.run_alignment_graded(req, call_llm=lambda *a, **k: "", write_results=fake_write)

    analysis_path = tmp_path / "PAIR_TEST_analysis.json"
    assert analysis_path.is_file()
    data = json.loads(analysis_path.read_text())
    syn = data["synthesis"]
    assert syn["host_pool_count"] == 3
    assert syn["query_pool_count"] == 2
    assert "excerpt_stats" in syn
    assert "host_running_summary" in syn
    assert "query_running_summary" in syn
    assert "batch_outputs_host" in syn
    assert "batch_outputs_query" in syn


def test_run_final_synthesis_llm_retries_and_records_attempts(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("SYNTHESIS_ENABLE_REPAIR_PASS", "0")

    def fake_llm(prompt, base_url, max_tokens, temperature):
        return "Discussion without required Quick results JSON block."

    text, notes, attempts, ok = sg._run_final_synthesis_llm(
        synth_prompt="base prompt",
        llm_base_url="http://fake",
        call_llm=fake_llm,
        final_call_max_tokens=1024,
        temperature=0.0,
        alignment_id="RETRY_TEST",
    )
    assert not ok
    assert not notes
    assert len(attempts) == 2
    assert attempts[0]["stage"] == "attempt_0"
    assert attempts[0]["diagnosis"] == "missing_quick_results_header"


def test_write_synthesis_raw_failure_debug(tmp_path):
    attempts = [
        {
            "stage": "attempt_0",
            "text_len": 12,
            "well_formed": False,
            "diagnosis": "missing_quick_results_header",
            "text": "bad output",
        }
    ]
    debug_path = sg._write_synthesis_raw_failure_debug(
        str(tmp_path),
        "ALIGN1",
        attempts,
        "synthesis missing parseable Quick results JSON after retry",
    )
    assert debug_path
    assert (tmp_path / "logs" / "ALIGN1_synthesis_raw_failed.json").is_file()
    assert (tmp_path / "logs" / "ALIGN1_synthesis_raw_failed.txt").is_file()


def test_run_alignment_graded_saves_debug_on_llm_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://fake")
    monkeypatch.setenv("SYNTHESIS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("SYNTHESIS_ENABLE_REPAIR_PASS", "0")
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    fn = "host_0.txt"
    (papers_dir / fn).write_text("MRPL19 study.", encoding="utf-8")
    graded = [_gp(fn, "target", 0.9, max_axis=0.8)]

    req = SimpleNamespace(
        alignment_id="DEBUG_TEST",
        papers_dir=str(papers_dir),
        query="Q1",
        target_id="T1",
        instructions="test instructions",
        output_root=str(tmp_path),
        graded_papers=graded,
        grading_meta={},
        constraints=None,
        gene_context={"query": {"gene_name": "geneA"}, "target": {"gene_name": "MRPL19"}},
    )

    def fake_llm(prompt, base_url, max_tokens, temperature):
        if "final pair-level synthesis" in prompt:
            return "invalid synthesis without json footer"
        return json.dumps(
            {
                "paper_summaries": [
                    {
                        "file_name": "host_0.txt",
                        "summary": "Host summary.",
                        "important_points": ["pt"],
                        "confidence_notes": "",
                    }
                ],
                "memory_updates": [],
                "batch_summary": "host rollup",
            }
        )

    def fake_write(request, payload):
        return SimpleNamespace(
            status="ok",
            alignment_id=request.alignment_id,
            results_path=str(tmp_path / f"{request.alignment_id}_results.json"),
        )

    sg.run_alignment_graded(req, call_llm=fake_llm, write_results=fake_write)

    debug_json = tmp_path / "logs" / "DEBUG_TEST_synthesis_raw_failed.json"
    assert debug_json.is_file()
    analysis = json.loads((tmp_path / "DEBUG_TEST_analysis.json").read_text())
    assert "debug" in analysis["synthesis"]
    assert analysis["synthesis"]["debug"]["raw_failed_path"]
