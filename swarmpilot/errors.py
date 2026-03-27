"""Centralized error hierarchy for SwarmPilot.

All SwarmPilot-specific exceptions inherit from :class:`SwarmPilotError`,
which itself inherits from :class:`Exception`.  This allows callers to
catch broad or narrow slices of the hierarchy as needed.
"""

from __future__ import annotations

from types import SimpleNamespace


class SwarmPilotError(Exception):
    """Base exception for all SwarmPilot errors.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class DeployError(SwarmPilotError):
    """Deployment failed (partially or fully).

    Args:
        message: Human-readable description of the failure.
        succeeded: Instances that deployed successfully.
        failed: Details of failed replicas.
    """

    def __init__(
        self,
        message: str,
        succeeded: list | None = None,
        failed: list | None = None,
    ) -> None:
        self.succeeded: list = succeeded if succeeded is not None else []
        self.failed: list = failed if failed is not None else []
        super().__init__(message)


class PartialDeploymentError(DeployError):
    """Raised when some replicas in a batch deployment failed."""

    @property
    def result(self) -> SimpleNamespace:
        """Return legacy ``result`` shape for backward compatibility."""
        return SimpleNamespace(
            succeeded=self.succeeded,
            failed=self.failed,
        )


class SchedulerNotFound(SwarmPilotError):  # noqa: N818
    """No scheduler registered for the given model.

    A ``hint`` is auto-generated with an actionable suggestion.

    Args:
        model: Model identifier that has no scheduler.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.hint: str = (
            f"No scheduler registered for '{model}'. "
            "Use scheduler='http://...' to specify manually, "
            "or scheduler=None to skip registration."
        )
        super().__init__(self.hint)


class ModelNotDeployed(SwarmPilotError):  # noqa: N818
    """Predictor operation attempted on a model with no scheduler.

    A ``hint`` is auto-generated with an actionable suggestion.

    Args:
        model: Model identifier that has no scheduler mapping.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.hint: str = (
            f"Model '{model}' has no scheduler mapping. "
            "Deploy the model first with "
            "swarmpilot.serve() or swarmpilot.deploy()."
        )
        super().__init__(self.hint)


# ---------------------------------------------------------------
# Predictor errors
# ---------------------------------------------------------------


class PredictorError(SwarmPilotError):
    """Base exception for predictor-related errors."""

    pass


class ModelNotFoundError(PredictorError):
    """Raised when a requested predictor model does not exist."""

    pass


class PredictorValidationError(PredictorError):
    """Raised when predictor input validation fails."""

    pass


class TrainingError(PredictorError):
    """Raised when model training fails."""

    pass


class PredictionError(PredictorError):
    """Raised when prediction fails."""

    pass


class SwarmPilotTimeoutError(SwarmPilotError):
    """Instance did not become ready within timeout.

    Named ``SwarmPilotTimeoutError`` to avoid shadowing the builtin
    :class:`TimeoutError`.

    Args:
        timeout: Number of seconds that elapsed before giving up.
        name: Instance name that failed to become ready.
    """

    def __init__(self, timeout: int, name: str) -> None:
        self.timeout = timeout
        self.name = name
        super().__init__(
            f"Instance '{name}' did not become ready within {timeout}s"
        )
