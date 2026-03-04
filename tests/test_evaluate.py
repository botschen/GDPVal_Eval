"""Tests for the evaluate.py entry-point (HTTP calls and LLM calls are mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evaluate import evaluate_sample, main
from gdpval.submission.client import SUBMISSION_MODEL


_MOCK_SAMPLE = {
    "task_id": "test-task-id-001",
    "sector": "Professional, Scientific, and Technical Services",
    "occupation": "Accountants and Auditors",
    "prompt": "Review the attached spreadsheet and flag anomalies.",
    "reference_files": ["data.xlsx"],
    "reference_file_urls": ["https://example.com/data.xlsx"],
    "rubric_pretty": "Award points for accuracy and completeness.",
    "rubric_json": {"max_score": 100},
}

_MOCK_LLM_RESPONSE = "I have reviewed the spreadsheet and flagged the following anomalies: ..."


class TestEvaluateSample:
    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_returns_expected_keys(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        for key in ("evaluated_at", "dataset", "offset", "model", "task_id", "sector",
                    "occupation", "prompt", "reference_files", "rubric_pretty",
                    "task_prompt", "llm_response"):
            assert key in result, f"Missing key: {key}"

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_metadata_preserved(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        assert result["task_id"] == _MOCK_SAMPLE["task_id"]
        assert result["sector"] == _MOCK_SAMPLE["sector"]
        assert result["occupation"] == _MOCK_SAMPLE["occupation"]
        assert result["dataset"] == "openai/gdpval"
        assert result["offset"] == 0

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_task_prompt_contains_task_text(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        assert _MOCK_SAMPLE["prompt"] in result["task_prompt"]

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_task_prompt_contains_reference_file(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        assert "data.xlsx" in result["task_prompt"]

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_offset_forwarded(self, mock_fetch, _mock_llm):
        evaluate_sample(offset=3)
        mock_fetch.assert_called_once_with(offset=3)

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_llm_response_stored(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        assert result["llm_response"] == _MOCK_LLM_RESPONSE

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_llm_called_with_task_prompt(self, _mock_fetch, mock_llm):
        result = evaluate_sample(offset=0)
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert result["task_prompt"] == call_args[0][0]

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_default_model_in_result(self, _mock_fetch, _mock_llm):
        result = evaluate_sample(offset=0)
        assert result["model"] == SUBMISSION_MODEL

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_custom_model_forwarded(self, _mock_fetch, mock_llm):
        result = evaluate_sample(offset=0, model="gemini-1.5-pro")
        assert result["model"] == "gemini-1.5-pro"
        mock_llm.assert_called_once()
        assert mock_llm.call_args[1]["model"] == "gemini-1.5-pro"

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_api_key_forwarded(self, _mock_fetch, mock_llm):
        evaluate_sample(offset=0, api_key="test-key")
        mock_llm.assert_called_once()
        assert mock_llm.call_args[1]["api_key"] == "test-key"


class TestMain:
    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_writes_json_file(self, _mock_fetch, _mock_llm, tmp_path):
        out = tmp_path / "out.json"
        main(["--output", str(out)])
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["task_id"] == _MOCK_SAMPLE["task_id"]
        assert data["llm_response"] == _MOCK_LLM_RESPONSE

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_creates_parent_directory(self, _mock_fetch, _mock_llm, tmp_path):
        out = tmp_path / "sub" / "result.json"
        main(["--output", str(out)])
        assert out.exists()

    @patch("evaluate.call_llm", return_value=_MOCK_LLM_RESPONSE)
    @patch("evaluate.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_custom_model_cli_arg(self, _mock_fetch, mock_llm, tmp_path):
        out = tmp_path / "out.json"
        main(["--output", str(out), "--model", "gemini-1.5-pro"])
        data = json.loads(out.read_text())
        assert data["model"] == "gemini-1.5-pro"
