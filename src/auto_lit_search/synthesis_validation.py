"""Shared synthesis result validation for fix-it reruns and failure analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from auto_lit_search.synthesis_scorecard import synthesis_output_well_formed


def classify_synthesis_failure(data: dict) -> str:
    synth = data.get("synthesis") or {}
    notes = str(synth.get("notes") or "")
    notes_l = notes.lower()
    text = str(synth.get("text") or "")
    text_l = text.lower()
    conclusion = data.get("conclusion") or {}
    status = conclusion.get("synthesis_status")

    if "connection refused" in notes_l or "max retries exceeded" in notes_l:
        return "transport_error"
    if "timeout" in notes_l or "timed out" in notes_l:
        return "timeout"
    if "400 client error" in notes_l or "bad request" in notes_l:
        return "http_bad_request"
    if not str(text).strip():
        return "empty_output"
    if "rubric-derived scorecard only" in text_l or "fallback" in notes_l:
        return "fallback_grades_only"
    if "missing parseable quick results json" in notes_l:
        return "parse_error"
    if status in ("grades_only", "error"):
        return str(status)
    if status == "ok" and text.strip() and not synthesis_output_well_formed(text):
        return "parse_error_ok_status"
    return "unknown"


def needs_synthesis_fix(results_path: Path) -> bool:
    if not results_path.is_file():
        return True
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    conclusion = data.get("conclusion") or {}
    if not isinstance(conclusion, dict):
        return True
    if conclusion.get("scorecard_version") != "2":
        return True
    if conclusion.get("synthesis_status") != "ok":
        return True
    synth = data.get("synthesis") or {}
    notes = str(synth.get("notes") or "")
    notes_l = notes.lower()
    if "fallback" in notes_l or "timeout" in notes_l:
        return True
    if "connection refused" in notes_l or "max retries exceeded" in notes_l:
        return True
    text = str(synth.get("text") or "").strip()
    if not text:
        return True
    if "rubric-derived scorecard only" in text.lower():
        return True
    if not synthesis_output_well_formed(text):
        return True
    return False


def verify_synthesis_results(results_path: Path) -> Tuple[bool, str]:
    if not results_path.is_file():
        return False, "missing results file after synthesis POST"
    if needs_synthesis_fix(results_path):
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"unreadable results: {e}"
        conclusion = data.get("conclusion") or {}
        st = conclusion.get("synthesis_status")
        notes = str((data.get("synthesis") or {}).get("notes") or "")[:200]
        reason = classify_synthesis_failure(data)
        return False, f"synthesis_status={st}; reason={reason}; {notes}"
    return True, str(results_path)
