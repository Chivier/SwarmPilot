"""Tests for the centralized SwarmPilot error hierarchy."""

from __future__ import annotations

import pytest

from swarmpilot.errors import (
    DeployError,
    ModelNotDeployed,
    SchedulerNotFound,
    SwarmPilotError,
    SwarmPilotTimeoutError,
)


class TestInheritance:
    """Every custom error must inherit from SwarmPilotError and Exception."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            SwarmPilotError,
            DeployError,
            SchedulerNotFound,
            ModelNotDeployed,
            SwarmPilotTimeoutError,
        ],
    )
    def test_inherits_from_exception(self, exc_class: type) -> None:
        """All error classes are subclasses of Exception."""
        assert issubclass(exc_class, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [
            DeployError,
            SchedulerNotFound,
            ModelNotDeployed,
            SwarmPilotTimeoutError,
        ],
    )
    def test_inherits_from_swarmpilot_error(self, exc_class: type) -> None:
        """All concrete errors are subclasses of SwarmPilotError."""
        assert issubclass(exc_class, SwarmPilotError)


class TestDeployError:
    """DeployError stores succeeded and failed lists."""

    def test_stores_succeeded_and_failed(self) -> None:
        """Succeeded and failed lists are preserved."""
        succeeded = ["inst-1", "inst-2"]
        failed = [{"replica": 3, "reason": "OOM"}]
        err = DeployError(
            "partial failure",
            succeeded=succeeded,
            failed=failed,
        )
        assert err.succeeded == succeeded
        assert err.failed == failed
        assert err.message == "partial failure"

    def test_defaults_to_empty_lists(self) -> None:
        """When omitted, succeeded and failed default to []."""
        err = DeployError("total failure")
        assert err.succeeded == []
        assert err.failed == []


class TestSchedulerNotFound:
    """SchedulerNotFound auto-generates a hint."""

    def test_auto_generates_hint(self) -> None:
        """Hint contains 'scheduler' keyword and model name."""
        err = SchedulerNotFound(model="gpt-4")
        assert "scheduler" in err.hint.lower()
        assert "gpt-4" in err.hint
        assert err.model == "gpt-4"

    def test_str_uses_hint(self) -> None:
        """String representation matches the hint."""
        err = SchedulerNotFound(model="gpt-4")
        assert str(err) == err.hint


class TestModelNotDeployed:
    """ModelNotDeployed auto-generates a hint."""

    def test_auto_generates_hint(self) -> None:
        """Hint contains 'deploy' keyword and model name."""
        err = ModelNotDeployed(model="llama-13b")
        assert "deploy" in err.hint.lower()
        assert "llama-13b" in err.hint
        assert err.model == "llama-13b"

    def test_str_uses_hint(self) -> None:
        """String representation matches the hint."""
        err = ModelNotDeployed(model="llama-13b")
        assert str(err) == err.hint


class TestSwarmPilotTimeoutError:
    """SwarmPilotTimeoutError stores timeout and name."""

    def test_stores_timeout_and_name(self) -> None:
        """Timeout seconds and instance name are preserved."""
        err = SwarmPilotTimeoutError(timeout=60, name="worker-3")
        assert err.timeout == 60
        assert err.name == "worker-3"

    def test_message_includes_details(self) -> None:
        """String representation mentions both name and timeout."""
        err = SwarmPilotTimeoutError(timeout=30, name="worker-1")
        msg = str(err)
        assert "worker-1" in msg
        assert "30" in msg

    def test_does_not_shadow_builtin(self) -> None:
        """SwarmPilotTimeoutError is distinct from builtin TimeoutError."""
        assert SwarmPilotTimeoutError is not TimeoutError
        assert not issubclass(SwarmPilotTimeoutError, TimeoutError)
