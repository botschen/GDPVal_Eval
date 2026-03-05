"""Tests for the Stirrup-based GDPVal agent runner.

All Stirrup and network calls are mocked so the tests run offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gdpval.agent.runner import (
    DEFAULT_MAX_TURNS,
    GDPValAgentRunner,
    SubmissionResult,
    _TURNS_REMAINING_WARNING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_SAMPLE: Dict[str, Any] = {
    "task_id": "test-task-001",
    "sector": "Professional, Scientific, and Technical Services",
    "occupation": "Software Developers",
    "prompt": "Create a Python script that prints 'Hello, World!'.",
    "reference_files": ["/home/user/hello.py"],
    "rubric_pretty": "Award points for a working script.",
}

_MOCK_FINISH_PARAMS = {
    "reason": "Created the script successfully.",
    "paths": ["/home/user/hello.py"],
}

_MOCK_METADATA = {
    "token_usage": [MagicMock(input=100, answer=50, reasoning=0)],
    "code_exec": [MagicMock(num_uses=1)],
}


def _make_mock_client() -> MagicMock:
    """Return a minimal mock Stirrup LLMClient."""
    client = MagicMock()
    client.model = "mock-model"
    return client


def _make_mock_agent(finish_params=_MOCK_FINISH_PARAMS, metadata=_MOCK_METADATA):
    """Return a mock Stirrup Agent whose session().run() returns canned values."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=(finish_params, [], metadata))

    mock_agent = MagicMock()
    mock_agent.session = MagicMock(return_value=mock_session)
    return mock_agent


# ---------------------------------------------------------------------------
# GDPValAgentRunner unit tests
# ---------------------------------------------------------------------------


class TestGDPValAgentRunnerInit:
    def test_stores_model_name(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")
        assert runner._model_name == "gpt-4o"

    def test_default_max_turns(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")
        assert runner._max_turns == DEFAULT_MAX_TURNS

    def test_custom_max_turns(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o", max_turns=50)
        assert runner._max_turns == 50

    def test_brave_api_key_from_arg(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o", brave_api_key="key123")
        assert runner._brave_api_key == "key123"

    def test_brave_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "env-key")
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")
        assert runner._brave_api_key == "env-key"


class TestBuildAgent:
    def test_build_agent_returns_agent_instance(self):
        """_build_agent should return a Stirrup Agent with the correct tools."""
        from stirrup import Agent

        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")
        agent = runner._build_agent()
        assert isinstance(agent, Agent)

    def test_build_agent_max_turns(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o", max_turns=42)
        agent = runner._build_agent()
        assert agent._max_turns == 42

    def test_build_agent_turns_warning_threshold(self):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")
        agent = runner._build_agent()
        assert agent._turns_remaining_warning_threshold == _TURNS_REMAINING_WARNING


class TestRunTaskAsync:
    @pytest.mark.asyncio
    async def test_returns_submission_result(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert isinstance(result, SubmissionResult)

    @pytest.mark.asyncio
    async def test_task_id_preserved(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.task_id == _MOCK_SAMPLE["task_id"]

    @pytest.mark.asyncio
    async def test_model_name_preserved(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_finish_reason_captured(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.finish_reason == _MOCK_FINISH_PARAMS["reason"]

    @pytest.mark.asyncio
    async def test_submitted_paths_captured(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.submitted_paths == _MOCK_FINISH_PARAMS["paths"]

    @pytest.mark.asyncio
    async def test_output_dir_recorded(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.output_dir == str(tmp_path)

    @pytest.mark.asyncio
    async def test_output_dir_created(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        new_dir = tmp_path / "nested" / "output"
        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            await runner.run_task_async(_MOCK_SAMPLE, output_dir=new_dir)

        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_none_finish_params_handled(self, tmp_path):
        """Agent hitting the turn limit returns None finish_params; result should be empty."""
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent(finish_params=None, metadata={})
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.finish_reason == ""
        assert result.submitted_paths == []

    @pytest.mark.asyncio
    async def test_token_usage_extracted(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.token_usage["input"] == 100
        assert result.token_usage["answer"] == 50
        assert result.token_usage["reasoning"] == 0

    @pytest.mark.asyncio
    async def test_session_called_with_output_dir(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            await runner.run_task_async(_MOCK_SAMPLE, output_dir=tmp_path)

        mock_agent.session.assert_called_once_with(
            output_dir=str(tmp_path),
            input_files=None,
        )

    @pytest.mark.asyncio
    async def test_session_called_with_input_files(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        input_files = ["/tmp/ref.csv"]
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            await runner.run_task_async(
                _MOCK_SAMPLE, output_dir=tmp_path, input_files=input_files
            )

        mock_agent.session.assert_called_once_with(
            output_dir=str(tmp_path),
            input_files=input_files,
        )


class TestRunTask:
    def test_sync_wrapper_returns_submission_result(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = runner.run_task(_MOCK_SAMPLE, output_dir=tmp_path)

        assert isinstance(result, SubmissionResult)
        assert result.task_id == _MOCK_SAMPLE["task_id"]

    def test_sync_wrapper_finish_reason(self, tmp_path):
        client = _make_mock_client()
        runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

        mock_agent = _make_mock_agent()
        with patch.object(runner, "_build_agent", return_value=mock_agent):
            result = runner.run_task(_MOCK_SAMPLE, output_dir=tmp_path)

        assert result.finish_reason == _MOCK_FINISH_PARAMS["reason"]


# ---------------------------------------------------------------------------
# run_submission.py CLI tests
# ---------------------------------------------------------------------------


class TestRunSubmissionCLI:
    @patch("run_submission.GDPValAgentRunner")
    @patch("run_submission.ChatCompletionsClient")
    @patch("run_submission.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_writes_json_result(self, _mock_fetch, mock_client_cls, mock_runner_cls, tmp_path):
        mock_result = SubmissionResult(
            task_id="test-task-001",
            model="gpt-4o",
            finish_reason="Done.",
            submitted_paths=["/home/user/hello.py"],
            output_dir=str(tmp_path),
        )
        mock_runner_cls.return_value.run_task.return_value = mock_result

        from run_submission import main

        out = tmp_path / "out"
        main(["--model", "gpt-4o", "--offset", "0", "--output", str(out)])

        result_file = out / "submission_result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text())
        assert data["task_id"] == "test-task-001"
        assert data["model"] == "gpt-4o"

    @patch("run_submission.GDPValAgentRunner")
    @patch("run_submission.ChatCompletionsClient")
    @patch("run_submission.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_uses_correct_model(self, _mock_fetch, mock_client_cls, mock_runner_cls, tmp_path):
        mock_result = SubmissionResult(
            task_id="test-task-001",
            model="deepseek-chat",
            finish_reason="Done.",
            submitted_paths=[],
            output_dir=str(tmp_path),
        )
        mock_runner_cls.return_value.run_task.return_value = mock_result

        from run_submission import main

        main(["--model", "deepseek-chat", "--offset", "0", "--output", str(tmp_path / "out")])

        _call_kwargs = mock_client_cls.call_args
        assert _call_kwargs[1]["model"] == "deepseek-chat" or _call_kwargs[0][0] == "deepseek-chat"

    @patch("run_submission.GDPValAgentRunner")
    @patch("run_submission.ChatCompletionsClient")
    @patch("run_submission.fetch_sample", return_value=_MOCK_SAMPLE)
    def test_offset_forwarded_to_fetch(
        self, mock_fetch, mock_client_cls, mock_runner_cls, tmp_path
    ):
        mock_result = SubmissionResult(
            task_id="test-task-001",
            model="gpt-4o",
            finish_reason="Done.",
            submitted_paths=[],
            output_dir=str(tmp_path),
        )
        mock_runner_cls.return_value.run_task.return_value = mock_result

        from run_submission import main

        main(["--model", "gpt-4o", "--offset", "7", "--output", str(tmp_path / "out")])
        mock_fetch.assert_called_once_with(offset=7)
