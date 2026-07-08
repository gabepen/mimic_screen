"""Tests for LLM chat response parsing."""

from auto_lit_search.llm_response import message_text_from_chat_response


def test_message_text_prefers_content():
    data = {
        "choices": [
            {
                "message": {
                    "content": "final answer",
                    "reasoning_content": "thinking",
                }
            }
        ]
    }
    assert message_text_from_chat_response(data) == "final answer"


def test_message_text_falls_back_to_reasoning():
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "only reasoning",
                },
                "finish_reason": "length",
            }
        ]
    }
    assert message_text_from_chat_response(data) == "only reasoning"


def test_message_text_empty_when_missing():
    assert message_text_from_chat_response({}) == ""
