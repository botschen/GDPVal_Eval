"""Stirrup-based agent runner for GDPVal task submission.

Wraps the `Stirrup <https://github.com/ArtificialAnalysis/Stirrup>`_ agent
framework to run each GDPVal task with the five tools specified in the
GDPVal-AA evaluation protocol:

1. **Run Shell** (``code_exec``) – ``LocalCodeExecToolProvider``
2. **Web Fetch** (``web_fetch``) – ``WebToolProvider``
3. **Web Search** (``web_search``) – ``WebToolProvider`` (requires ``BRAVE_API_KEY``)
4. **View Image** (``view_image``) – ``ViewImageToolProvider``
5. **Finish** (``finish``) – ``SIMPLE_FINISH_TOOL``

Example
-------
::

    import asyncio
    from gdpval.agent.runner import GDPValAgentRunner
    from gdpval.dataset.loader import fetch_sample
    from pathlib import Path
    from stirrup.clients.chat_completions_client import ChatCompletionsClient

    client = ChatCompletionsClient(model="gpt-4o")
    runner = GDPValAgentRunner(client=client, model_name="gpt-4o")

    sample = fetch_sample(offset=0)
    result = runner.run_task(sample, output_dir=Path("output/task_0"))
    print(result.finish_reason)
    print(result.submitted_paths)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from stirrup import Agent
from stirrup.core.models import LLMClient
from stirrup.tools import SIMPLE_FINISH_TOOL, LocalCodeExecToolProvider, WebToolProvider
from stirrup.tools.view_image import ViewImageToolProvider

from gdpval.submission.prompt import build_task_prompt

# Default maximum agent turns.  GDPVal-AA uses up to 100 turns per task.
DEFAULT_MAX_TURNS: int = 100

# Warn the model when this many turns remain (maps to Stirrup's
# ``turns_remaining_warning_threshold`` parameter).
_TURNS_REMAINING_WARNING: int = 20


@dataclass
class SubmissionResult:
    """Result of running a single GDPVal task through the Stirrup agent.

    Attributes
    ----------
    task_id:
        Identifier of the GDPVal task that was run.
    model:
        Human-readable name of the model used.
    finish_reason:
        Summary provided by the model when it called the finish tool, or an
        empty string if the agent stopped due to the turn limit.
    submitted_paths:
        List of absolute file paths declared by the model in the finish call.
    output_dir:
        Local directory where submitted files were copied.
    token_usage:
        Aggregated token counts (``input``, ``answer``, ``reasoning``).
    metadata:
        Raw per-tool metadata returned by Stirrup.
    """

    task_id: str
    model: str
    finish_reason: str
    submitted_paths: List[str]
    output_dir: str
    token_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GDPValAgentRunner:
    """Run GDPVal task submissions using the Stirrup agent framework.

    Parameters
    ----------
    client:
        A Stirrup ``LLMClient`` instance (e.g. ``ChatCompletionsClient`` or
        ``LiteLLMClient``).
    model_name:
        Human-readable model name stored in the result record.
    max_turns:
        Maximum number of agent turns before the run is aborted.
        Defaults to :data:`DEFAULT_MAX_TURNS` (100).
    brave_api_key:
        Brave Search API key used by the web-search tool.  If *None*,
        the value of the ``BRAVE_API_KEY`` environment variable is used.
        Web search is disabled silently when no key is available.

    Notes
    -----
    Each call to :meth:`run_task` or :meth:`run_task_async` creates a fresh
    agent instance with isolated tool providers (i.e. a new temporary
    execution directory per task).
    """

    def __init__(
        self,
        client: LLMClient,
        model_name: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        brave_api_key: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._max_turns = max_turns
        self._brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_task_async(
        self,
        sample: Dict[str, Any],
        output_dir: Path,
        input_files: Optional[List[str]] = None,
    ) -> SubmissionResult:
        """Run a single GDPVal task asynchronously via Stirrup.

        Parameters
        ----------
        sample:
            A dataset row as returned by
            :func:`gdpval.dataset.loader.fetch_sample`.
        output_dir:
            Directory where the agent's submitted output files will be saved.
            Created automatically if it does not exist.
        input_files:
            Local paths of reference files to upload into the agent's
            execution environment.  When *None*, the paths stored in
            ``sample["reference_files"]`` are used as-is (they are already
            present inside the Docker/local sandbox for official runs).

        Returns
        -------
        SubmissionResult
        """
        task_id: str = sample.get("task_id", "unknown")

        # Build the formatted task prompt.
        files_for_prompt: List[str] = input_files or sample.get("reference_files") or []
        task_prompt = build_task_prompt(
            task=sample.get("prompt", ""),
            reference_files=files_for_prompt,
        )

        agent = self._build_agent()
        output_dir.mkdir(parents=True, exist_ok=True)

        async with agent.session(
            output_dir=str(output_dir),
            input_files=input_files,
        ) as session:
            finish_params, _history, metadata = await session.run(task_prompt)

        # finish_params may be None if the agent hit the turn limit without
        # calling the finish tool.
        submitted_paths: List[str] = []
        finish_reason: str = ""
        if finish_params:
            submitted_paths = finish_params.get("paths", [])
            finish_reason = finish_params.get("reason", "")

        # Extract aggregated token usage when available.
        token_usage: Dict[str, Any] = {}
        if metadata:
            usage_list = metadata.get("token_usage", [])
            if usage_list:
                last = usage_list[-1]
                token_usage = {
                    "input": getattr(last, "input", 0),
                    "answer": getattr(last, "answer", 0),
                    "reasoning": getattr(last, "reasoning", 0),
                }

        return SubmissionResult(
            task_id=task_id,
            model=self._model_name,
            finish_reason=finish_reason,
            submitted_paths=submitted_paths,
            output_dir=str(output_dir),
            token_usage=token_usage,
            metadata=metadata or {},
        )

    def run_task(
        self,
        sample: Dict[str, Any],
        output_dir: Path,
        input_files: Optional[List[str]] = None,
    ) -> SubmissionResult:
        """Synchronous wrapper around :meth:`run_task_async`.

        Calls :func:`asyncio.run` internally.  Do **not** call this from
        inside an already-running event loop; use :meth:`run_task_async`
        directly in that case.
        """
        return asyncio.run(self.run_task_async(sample, output_dir, input_files))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_agent(self) -> Agent:
        """Construct a Stirrup :class:`~stirrup.Agent` with the five GDPVal tools."""
        code_exec = LocalCodeExecToolProvider()
        web = WebToolProvider(brave_api_key=self._brave_api_key)
        view_image = ViewImageToolProvider(exec_env=code_exec)

        return Agent(
            client=self._client,
            name="gdpval_agent",
            max_turns=self._max_turns,
            tools=[code_exec, web, view_image],
            finish_tool=SIMPLE_FINISH_TOOL,
            turns_remaining_warning_threshold=_TURNS_REMAINING_WARNING,
        )
