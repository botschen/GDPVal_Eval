"""LLM client for task submission.

Sends the formatted task-submission prompt to a Gemini model and returns
the model's response text.
"""

from __future__ import annotations

import os
from typing import Optional

SUBMISSION_MODEL = "gemini-2.5-pro-preview"


def call_llm(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = SUBMISSION_MODEL,
) -> str:
    """Send a task-submission prompt to a Gemini model and return the response.

    Parameters
    ----------
    prompt:
        The fully interpolated task-submission prompt.
    api_key:
        Gemini API key.  Falls back to the ``GEMINI_API_KEY`` environment
        variable if not provided.
    model:
        Gemini model name (default: ``gemini-2.5-pro-preview``).

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    RuntimeError
        If ``google-generativeai`` is not installed.
    """
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Run: pip install google-generativeai"
        )

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    genai.configure(api_key=resolved_key)
    client = genai.GenerativeModel(model)
    response = client.generate_content(prompt)
    return response.text.strip()
