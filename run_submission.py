"""Run a GDPVal task through the Stirrup agent harness and save the result.

This script fetches one task row from the ``openai/gdpval`` dataset, runs it
through a Stirrup-based agent with the five GDPVal-AA tools (Run Shell, Web
Fetch, Web Search, View Image, Finish), and writes a JSON result record.

Usage
-----
    python run_submission.py --model gpt-4o --offset 0
    python run_submission.py --model gpt-4o --offset 5 --output results/task_5/
    python run_submission.py --model deepseek-chat \\
        --base-url https://api.deepseek.com --offset 0

Environment Variables
---------------------
OPENAI_API_KEY / OPENROUTER_API_KEY
    API key for OpenAI-compatible providers.
BRAVE_API_KEY
    Brave Search API key (optional; web search is disabled without it).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from stirrup.clients.chat_completions_client import ChatCompletionsClient

from gdpval.agent.runner import DEFAULT_MAX_TURNS, GDPValAgentRunner
from gdpval.dataset.loader import fetch_sample

_DEFAULT_OUTPUT_BASE = Path(__file__).parent / "results" / "submissions"


def _build_result_record(
    sample: Dict[str, Any],
    submission_result: Any,
) -> Dict[str, Any]:
    """Combine dataset metadata with the agent submission result."""
    return {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "openai/gdpval",
        "task_id": submission_result.task_id,
        "model": submission_result.model,
        "sector": sample.get("sector"),
        "occupation": sample.get("occupation"),
        "prompt": sample.get("prompt"),
        "reference_files": sample.get("reference_files") or [],
        "finish_reason": submission_result.finish_reason,
        "submitted_paths": submission_result.submitted_paths,
        "output_dir": submission_result.output_dir,
        "token_usage": submission_result.token_usage,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier passed to the API (e.g. 'gpt-4o', 'deepseek-chat').",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based row index of the GDPVal task to evaluate (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Directory to save output files and the JSON result record.  "
            "Defaults to results/submissions/<task_id>/."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Custom API base URL for OpenAI-compatible providers "
            "(e.g. 'https://api.deepseek.com' or 'https://openrouter.ai/api/v1')."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the LLM provider.  Falls back to the "
            "OPENAI_API_KEY or OPENROUTER_API_KEY environment variables."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum agent turns per task (default: {DEFAULT_MAX_TURNS}).",
    )
    args = parser.parse_args(argv)

    # Resolve API key from CLI arg → OPENAI_API_KEY → OPENROUTER_API_KEY.
    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )

    # Fetch the dataset sample.
    print(f"Fetching GDPVal sample at offset {args.offset}…")
    sample = fetch_sample(offset=args.offset)
    task_id = sample.get("task_id", f"offset_{args.offset}")

    # Resolve output directory.
    output_dir: Path = args.output or (_DEFAULT_OUTPUT_BASE / task_id)

    print(f"Task ID   : {task_id}")
    print(f"Sector    : {sample.get('sector')}")
    print(f"Occupation: {sample.get('occupation')}")
    print(f"Model     : {args.model}")
    print(f"Output dir: {output_dir}")
    print()

    # Build the Stirrup client and runner.
    client = ChatCompletionsClient(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
    )
    runner = GDPValAgentRunner(
        client=client,
        model_name=args.model,
        max_turns=args.max_turns,
    )

    # Run the task.
    submission_result = runner.run_task(sample=sample, output_dir=output_dir)

    # Save the JSON result record alongside the output files.
    record = _build_result_record(sample, submission_result)
    result_path = output_dir / "submission_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    print(f"Finish reason  : {submission_result.finish_reason}")
    print(f"Submitted files: {submission_result.submitted_paths}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
