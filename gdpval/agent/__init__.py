"""Stirrup-based agent harness for GDPVal task submission.

This package provides :class:`GDPValAgentRunner`, which wraps the Stirrup
agent framework to run GDPVal tasks with the five required tools:

- **Run Shell** (``code_exec``) – execute shell commands in an isolated temp directory
- **Web Fetch** (``web_fetch``) – fetch and parse web pages
- **Web Search** (``web_search``) – search the web (requires ``BRAVE_API_KEY``)
- **View Image** (``view_image``) – load images from the execution environment into context
- **Finish** (``finish``) – signal task completion and declare output files
"""

from gdpval.agent.runner import GDPValAgentRunner, SubmissionResult

__all__ = ["GDPValAgentRunner", "SubmissionResult"]
