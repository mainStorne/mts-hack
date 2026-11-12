"""Streamed answers: hide markup the model writes for itself."""

import re

TOOL_CALL = re.compile(r"<tool_call>.*", re.DOTALL)


def partial_text(text: str) -> str:
    """The user-facing part of an answer being streamed: without a tool call the model wrote as text."""
    return TOOL_CALL.sub("", text).strip()
