"""Parse OpenAI-compatible chat completion responses."""

from __future__ import annotations

from typing import Any, Dict


def message_text_from_chat_response(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message") or {}
    if not isinstance(msg, dict):
        return ""
    content = str(msg.get("content") or "").strip()
    if content:
        return content
    return str(msg.get("reasoning_content") or "").strip()
