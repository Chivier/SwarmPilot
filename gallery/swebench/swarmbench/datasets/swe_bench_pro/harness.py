"""Docker-based test execution for SWE-bench Pro evaluation.

Wraps the ``swebench`` package's harness to run tests inside Docker
containers and extract the list of passed test names.

Requires optional dependencies: ``pip install swebench docker``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from swarmbench.models import Task

logger = logging.getLogger(__name__)


class DockerNotAvailableError(RuntimeError):
    """Raised when Docker SDK is missing or the daemon is unreachable."""


class HarnessTimeoutError(RuntimeError):
    """Raised when the Docker container exceeds the configured timeout."""


class SWEBenchTestHarness:
    """Run SWE-bench Pro tests inside Docker via the ``swebench`` package.

    Args:
        run_id: Identifier written into swebench log paths.
        timeout: Per-instance timeout in seconds for Docker execution.
    """

    def __init__(self, run_id: str = "swarmbench", timeout: int = 1800):
        self.run_id = run_id
        self.timeout = timeout

    def run_tests(self, task: Task, patch: str) -> list[str]:
        """Execute tests in Docker and return the names of passed tests.

        Args:
            task: SWE-bench Pro task with ``_swebench_instance`` in metadata.
            patch: The agent-generated patch (unified diff).

        Returns:
            List of test name strings that passed.

        Raises:
            DockerNotAvailableError: Docker SDK missing or daemon unreachable.
            HarnessTimeoutError: Container exceeded timeout.
            KeyError: Task metadata missing ``_swebench_instance``.
        """
        docker_client = _ensure_docker()

        instance = task.metadata["_swebench_instance"]
        test_spec = _make_test_spec(instance)
        pred = _build_prediction(task.task_id, patch)

        try:
            result = _run_instance(
                test_spec=test_spec,
                pred=pred,
                client=docker_client,
                run_id=self.run_id,
                timeout=self.timeout,
            )
        except Exception as exc:
            if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
                raise HarnessTimeoutError(
                    f"Docker evaluation timed out after {self.timeout}s "
                    f"for {task.task_id}"
                ) from exc
            raise

        return _extract_passed_tests(result, task.task_id, self.run_id, pred)


# ---------------------------------------------------------------------------
# Internal helpers — thin wrappers so tests can mock at precise boundaries
# ---------------------------------------------------------------------------


def _ensure_docker():
    """Return a ``docker.DockerClient`` or raise ``DockerNotAvailableError``.

    Returns:
        A connected Docker client.

    Raises:
        DockerNotAvailableError: If the ``docker`` package is missing or the
            daemon cannot be reached.
    """
    try:
        import docker  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise DockerNotAvailableError(
            "Docker SDK not installed. Install with: pip install docker"
        ) from exc

    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as exc:
        raise DockerNotAvailableError(
            f"Cannot connect to Docker daemon: {exc}"
        ) from exc


def _make_test_spec(instance: dict[str, Any]):
    """Create a ``swebench`` TestSpec from a raw HuggingFace record.

    Args:
        instance: Full dataset record stored in ``task.metadata["_swebench_instance"]``.

    Returns:
        A ``swebench.harness.test_spec.test_spec.TestSpec``.
    """
    from swebench.harness.test_spec.test_spec import (  # type: ignore[reportMissingImports]
        make_test_spec,
    )

    return make_test_spec(instance)


def _build_prediction(task_id: str, patch: str) -> dict[str, str]:
    """Construct the prediction dict expected by ``swebench``.

    Args:
        task_id: Instance ID.
        patch: Agent-generated unified diff.

    Returns:
        Dict with keys ``instance_id``, ``model_name_or_path``, ``model_patch``.
    """
    return {
        "instance_id": task_id,
        "model_name_or_path": "swarmbench",
        "model_patch": patch,
    }


def _run_instance(
    test_spec,
    pred: dict[str, str],
    client,
    run_id: str,
    timeout: int,
) -> dict:
    """Thin wrapper around ``swebench.harness.run_evaluation.run_instance``.

    Args:
        test_spec: swebench TestSpec object.
        pred: Prediction dict.
        client: Docker client.
        run_id: Evaluation run identifier.
        timeout: Timeout in seconds.

    Returns:
        Result dict from swebench with keys ``completed`` and ``resolved``.
    """
    from swebench.harness.run_evaluation import (  # type: ignore[reportMissingImports]
        run_instance,
    )

    return run_instance(
        test_spec=test_spec,
        pred=pred,
        rm_image=False,
        force_rebuild=False,
        client=client,
        run_id=run_id,
        timeout=timeout,
    )


def _extract_passed_tests(
    result: dict,
    task_id: str,
    run_id: str,
    pred: dict[str, str],
) -> list[str]:
    """Parse swebench evaluation artifacts to build the passed-tests list.

    First attempts to read the detailed report written by ``run_instance``.
    Falls back to returning an empty list if the report is unreadable.

    Args:
        result: Dict returned by ``run_instance``.
        task_id: Instance ID.
        run_id: Run identifier used in the log directory path.
        pred: Prediction dict (used to derive the model path in log dir).

    Returns:
        Sorted list of test names that passed.
    """
    if not result.get("completed", False):
        logger.warning("run_instance did not complete for %s", task_id)
        return []

    # Locate the report.json written by run_instance.
    # Use swebench's own constant for the log directory to stay in sync.
    try:
        from swebench.harness.constants import (  # type: ignore[reportMissingImports]
            RUN_EVALUATION_LOG_DIR,
        )

        log_base = Path(RUN_EVALUATION_LOG_DIR)
    except ImportError:
        log_base = Path("logs/run_evaluation")

    model_dir = pred.get("model_name_or_path", "None").replace("/", "__")
    report_path = log_base / run_id / model_dir / task_id / "report.json"

    if not report_path.exists():
        logger.warning(
            "Report file not found at %s (resolved: %s)",
            report_path,
            report_path.resolve(),
        )
        return []

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read report for %s: %s", task_id, exc)
        return []

    instance_report = report.get(task_id, {})
    tests_status = instance_report.get("tests_status", {})

    passed: list[str] = []
    # FAIL_TO_PASS successes = tests that were supposed to fail but now pass (resolution)
    f2p = tests_status.get("FAIL_TO_PASS", {})
    passed.extend(f2p.get("success", []))
    # PASS_TO_PASS successes = tests that still pass (maintenance)
    p2p = tests_status.get("PASS_TO_PASS", {})
    passed.extend(p2p.get("success", []))

    return sorted(passed)
