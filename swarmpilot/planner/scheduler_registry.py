"""Scheduler registry for multi-scheduler architecture (PYLET-024).

This module provides a thread-safe registry that maps model IDs to their
dedicated scheduler URLs. Schedulers register on startup and deregister
on shutdown.

Example:
    from swarmpilot.planner.scheduler_registry import get_scheduler_registry

    registry = get_scheduler_registry()
    registry.register("llm-7b", "http://localhost:8010")
    url = registry.get_scheduler_url("llm-7b")
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from loguru import logger

from .models.scheduler import SchedulerInfo


class SchedulerRegistry:
    """Thread-safe registry mapping model IDs to scheduler URLs.

    Attributes:
        _schedulers: Internal dict mapping model_id to SchedulerInfo.
        _lock: Threading lock for thread-safe access.
    """

    def __init__(self) -> None:
        """Initialize empty scheduler registry."""
        self._schedulers: dict[str, SchedulerInfo] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_id: str,
        scheduler_url: str,
        metadata: dict[str, str] | None = None,
    ) -> bool:
        """Register a scheduler for a model.

        Args:
            model_id: Model identifier.
            scheduler_url: Base URL of the scheduler.
            metadata: Optional metadata.

        Returns:
            True if a previous registration was replaced.
        """
        with self._lock:
            replaced = model_id in self._schedulers
            self._schedulers[model_id] = SchedulerInfo(
                model_id=model_id,
                scheduler_url=scheduler_url.rstrip("/"),
                registered_at=datetime.now(UTC).isoformat(),
                is_healthy=True,
                metadata=metadata or {},
            )
            action = "replaced" if replaced else "registered"
            logger.info(
                f"Scheduler {action}: model_id={model_id} url={scheduler_url}"
            )
            return replaced

    def deregister(self, model_id: str) -> bool:
        """Deregister a scheduler for a model.

        Args:
            model_id: Model identifier to deregister.

        Returns:
            True if a scheduler was found and removed.
        """
        with self._lock:
            if model_id in self._schedulers:
                del self._schedulers[model_id]
                logger.info(f"Scheduler deregistered: model_id={model_id}")
                return True
            logger.warning(
                f"Scheduler deregister: model_id={model_id} not found"
            )
            return False

    def get_scheduler_url(self, model_id: str) -> str | None:
        """Get the scheduler URL for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Scheduler URL or None if not registered.
        """
        with self._lock:
            info = self._schedulers.get(model_id)
            return info.scheduler_url if info else None

    def get_scheduler_info(self, model_id: str) -> SchedulerInfo | None:
        """Get full scheduler info for a model.

        Args:
            model_id: Model identifier.

        Returns:
            SchedulerInfo or None if not registered.
        """
        with self._lock:
            return self._schedulers.get(model_id)

    def list_all(self) -> list[SchedulerInfo]:
        """List all registered schedulers.

        Returns:
            List of SchedulerInfo objects.
        """
        with self._lock:
            return list(self._schedulers.values())

    def get_registered_models(self) -> list[str]:
        """Get all registered model IDs.

        Returns:
            List of model ID strings.
        """
        with self._lock:
            return list(self._schedulers.keys())

    def __len__(self) -> int:
        """Return number of registered schedulers."""
        with self._lock:
            return len(self._schedulers)

    def __contains__(self, model_id: str) -> bool:
        """Check if a model has a registered scheduler."""
        with self._lock:
            return model_id in self._schedulers

    def reassign(self, old_model_id: str, new_model_id: str) -> bool:
        """Remap a scheduler from one model to another.

        Moves the SchedulerInfo entry from old_model_id to
        new_model_id, preserving the scheduler_url.

        Args:
            old_model_id: Current model identifier.
            new_model_id: Target model identifier.

        Returns:
            True if the reassignment was performed.
        """
        with self._lock:
            info = self._schedulers.pop(old_model_id, None)
            if info is None:
                return False
            self._schedulers[new_model_id] = SchedulerInfo(
                model_id=new_model_id,
                scheduler_url=info.scheduler_url,
                registered_at=datetime.now(UTC).isoformat(),
                is_healthy=info.is_healthy,
                metadata=info.metadata,
            )
            logger.info(
                f"Scheduler reassigned: {old_model_id} -> "
                f"{new_model_id} (url={info.scheduler_url})"
            )
            return True


# Global registry singleton
_scheduler_registry: SchedulerRegistry | None = None


def get_scheduler_registry() -> SchedulerRegistry:
    """Get or create the global scheduler registry.

    Returns:
        The global SchedulerRegistry instance.
    """
    global _scheduler_registry
    if _scheduler_registry is None:
        _scheduler_registry = SchedulerRegistry()
    return _scheduler_registry
