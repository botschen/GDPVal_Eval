"""Tests for gdpval.submission.client (Gemini API calls are mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gdpval.submission.client import SUBMISSION_MODEL, call_llm


class TestCallLlm:
    def _make_genai_mock(self, response_text: str = "mocked response"):
        genai = MagicMock()
        model_instance = MagicMock()
        model_instance.generate_content.return_value = MagicMock(text=f"  {response_text}  ")
        genai.GenerativeModel.return_value = model_instance
        return genai

    @patch.dict("sys.modules", {"google.generativeai": None})
    def test_raises_runtime_error_when_not_installed(self):
        with pytest.raises(RuntimeError, match="google-generativeai is not installed"):
            call_llm("some prompt")

    def test_returns_stripped_response(self):
        genai_mock = self._make_genai_mock("  hello world  ")
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            result = call_llm("some prompt")
        assert result == "hello world"

    def test_configures_api_key(self):
        genai_mock = self._make_genai_mock()
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            call_llm("some prompt", api_key="my-api-key")
        genai_mock.configure.assert_called_once_with(api_key="my-api-key")

    def test_uses_env_api_key_when_not_provided(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        genai_mock = self._make_genai_mock()
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            call_llm("some prompt")
        genai_mock.configure.assert_called_once_with(api_key="env-key")

    def test_uses_default_model(self):
        genai_mock = self._make_genai_mock()
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            call_llm("some prompt")
        genai_mock.GenerativeModel.assert_called_once_with(SUBMISSION_MODEL)

    def test_uses_custom_model(self):
        genai_mock = self._make_genai_mock()
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            call_llm("some prompt", model="gemini-1.5-pro")
        genai_mock.GenerativeModel.assert_called_once_with("gemini-1.5-pro")

    def test_prompt_passed_to_generate_content(self):
        genai_mock = self._make_genai_mock()
        model_instance = genai_mock.GenerativeModel.return_value
        with patch.dict("sys.modules", {"google.generativeai": genai_mock}):
            call_llm("my task prompt")
        model_instance.generate_content.assert_called_once_with("my task prompt")
