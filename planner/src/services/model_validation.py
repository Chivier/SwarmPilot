"""Model validation service for multi-scheduler architecture (PYLET-024).

Validates that requested model IDs have registered schedulers before
allowing deployment operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from ..scheduler_registry import SchedulerRegistry


@dataclass
class ValidationResult:
    """Result of model validation.

    Attributes:
        valid: Whether all requested models are valid.
        invalid_models: List of model IDs without registered schedulers.
        registered_models: List of all registered model IDs.
        message: Human-readable validation message.
    """

    valid: bool
    invalid_models: list[str] = field(default_factory=list)
    registered_models: list[str] = field(default_factory=list)
    message: str = ""


class ModelValidationService:
    """Validates model IDs against the scheduler registry.

    When schedulers are registered, this service ensures that deployment
    requests only target models with active schedulers. When no schedulers
    are registered (legacy mode), validation is skipped.

    Attributes:
        registry: The scheduler registry to validate against.
    """

    def __init__(self, registry: SchedulerRegistry) -> None:
        """Initialize with a scheduler registry.

        Args:
            registry: Scheduler registry to validate against.
        """
        self._registry = registry

    def validate_models(
        self, requested_models: list[str]
    ) -> ValidationResult:
        """Validate that all requested models have registered schedulers.

        If no schedulers are registered at all (legacy/backward-compatible
        mode), validation passes unconditionally.

        Args:
            requested_models: List of model IDs to validate.

        Returns:
            ValidationResult with validation outcome.
        """
        registered = self._registry.get_registered_models()

        # If no schedulers registered, skip validation (backward compat)
        if not registered:
            logger.debug(
                "No schedulers registered, skipping model validation"
            )
            return ValidationResult(
                valid=True,
                registered_models=[],
                message="No schedulers registered (legacy mode)",
            )

        invalid = [m for m in requested_models if m not in self._registry]

        if invalid:
            message = (
                f"Models not found in scheduler registry: {invalid}. "
                f"Registered models: {registered}"
            )
            logger.warning(f"Model validation failed: {message}")
            return ValidationResult(
                valid=False,
                invalid_models=invalid,
                registered_models=registered,
                message=message,
            )

        return ValidationResult(
            valid=True,
            registered_models=registered,
            message="All models validated",
        )
