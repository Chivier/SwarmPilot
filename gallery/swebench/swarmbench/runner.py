from __future__ import annotations

import logging
import os
from typing import Callable

from swarmbench.datasets.dataclaw import (
    CodingTools,
    DataclawEvaluator,
    DataclawLoader,
    TrajectoryOnlyTools,
)
from swarmbench.datasets.mcp_atlas import (
    MCPAtlasEvaluator,
    MCPAtlasLoader,
    MockMCPTools,
    RealMCPTools,
)
from swarmbench.datasets.swe_bench_pro import (
    SWEBenchProEvaluator,
    SWEBenchProLoader,
    SWEBenchProTools,
)
from swarmbench.llm import LLMClient
from swarmbench.logger import ConversationLogger
from swarmbench.loop import run_task
from swarmbench.models import ConversationLog, EvalResult, RunConfig, Task
from swarmbench.reporter import Reporter


def get_loader(dataset_name: str):
    if dataset_name == "mcp-atlas":
        return MCPAtlasLoader()
    if dataset_name == "dataclaw":
        return DataclawLoader()
    if dataset_name == "swe-bench-pro":
        return SWEBenchProLoader()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_evaluator(config: RunConfig):
    if config.dataset_name == "mcp-atlas":
        return MCPAtlasEvaluator(
            model=config.model, base_url=config.base_url, api_key=config.api_key
        )
    if config.dataset_name == "dataclaw":
        return DataclawEvaluator(
            model=config.model, base_url=config.base_url, api_key=config.api_key
        )
    if config.dataset_name == "swe-bench-pro":
        return SWEBenchProEvaluator()
    raise ValueError(f"Unknown dataset: {config.dataset_name}")


def get_tool_provider(task: Task, config: RunConfig):
    workspace = config.model_dump().get("workspace")
    if task.dataset_name == "mcp-atlas":
        if config.mode == "real":
            port = int(os.getenv("SWARMBENCH_MCP_PORT", "8000"))
            return RealMCPTools(task, port=port)
        return MockMCPTools(task)
    if task.dataset_name == "dataclaw":
        if config.mode == "live":
            if not workspace:
                raise ValueError("Dataclaw live mode requires --workspace")
            return CodingTools(str(workspace))
        return TrajectoryOnlyTools()
    if task.dataset_name == "swe-bench-pro":
        if not workspace:
            raise ValueError("SWE-bench Pro mode requires --workspace")
        return SWEBenchProTools(str(workspace), dry_run=(config.mode == "dry-run"))
    raise ValueError(f"Unknown dataset: {task.dataset_name}")


def run_dataset(config: RunConfig) -> list[EvalResult]:
    """Load tasks, run the agent loop, optionally run tests, then evaluate."""
    loader = get_loader(config.dataset_name)
    tasks = loader.load(limit=config.limit)
    llm = LLMClient(config)
    logger = ConversationLogger(config.output_dir)
    evaluator = get_evaluator(config)
    harness = _get_test_harness(config)
    results: list[EvalResult] = []

    for task in tasks:
        tools = get_tool_provider(task, config)
        conversation = run_task(task, llm, tools, config)
        conversation = _maybe_run_tests(conversation, tools, harness, task)
        logger.save(conversation)
        result = evaluator.evaluate(task, conversation)
        results.append(result)

    return results


def _get_test_harness(config: RunConfig):
    """Return a SWEBenchTestHarness for live mode, or None otherwise."""
    if config.dataset_name != "swe-bench-pro" or config.mode != "live":
        return None
    try:
        from swarmbench.datasets.swe_bench_pro.harness import SWEBenchTestHarness

        return SWEBenchTestHarness(timeout=config.docker_timeout)
    except ImportError:
        logging.getLogger(__name__).warning(
            "swebench/docker not installed — skipping Docker test execution. "
            "Install with: uv pip install -e '.[swebench]' && pip install docker"
        )
        return None


def _maybe_run_tests(
    conversation: ConversationLog,
    tools: object,
    harness: object | None,
    task: Task,
) -> ConversationLog:
    """If a harness is available and the agent submitted a patch, run Docker tests."""
    if harness is None:
        return conversation
    patch = getattr(tools, "submitted_patch", None)
    if not patch or patch == "[dry-run submit]":
        return conversation
    try:
        passed_tests = harness.run_tests(task, patch)  # type: ignore[union-attr]
    except Exception as exc:
        _log = logging.getLogger(__name__)
        # Import lazily to avoid hard dependency on the harness module
        try:
            from swarmbench.datasets.swe_bench_pro.harness import (
                DockerNotAvailableError,
                HarnessTimeoutError,
            )
        except ImportError:
            DockerNotAvailableError = type(None)  # type: ignore[assignment,misc]
            HarnessTimeoutError = type(None)  # type: ignore[assignment,misc]

        if isinstance(exc, DockerNotAvailableError):
            _log.warning(
                "Docker not available for %s — skipping test execution: %s",
                task.task_id,
                exc,
            )
        elif isinstance(exc, HarnessTimeoutError):
            _log.error(
                "Docker test execution timed out for %s: %s", task.task_id, exc
            )
        else:
            _log.error(
                "Unexpected failure in Docker test execution for %s: %s",
                task.task_id,
                exc,
            )
        passed_tests = []
    return ConversationLog(
        task_id=conversation.task_id,
        dataset_name=conversation.dataset_name,
        messages=conversation.messages,
        metadata={
            **conversation.metadata,
            "passed_tests": passed_tests,
            "submitted_patch": patch,
        },
    )


def evaluate_saved_logs(config: RunConfig) -> list[EvalResult]:
    loader = get_loader(config.dataset_name)
    tasks = loader.load(limit=config.limit)
    tasks_by_id = {task.task_id: task for task in tasks}
    logger = ConversationLogger(config.output_dir)
    evaluator = get_evaluator(config)
    logs = logger.load_all(config.dataset_name)
    results: list[EvalResult] = []

    for log in logs:
        task = tasks_by_id.get(log.task_id)
        if task is None:
            continue
        results.append(evaluator.evaluate(task, log))
    return results


def report_results(results: list[EvalResult], output_path: str | None = None) -> None:
    Reporter().report(results, output_path)
