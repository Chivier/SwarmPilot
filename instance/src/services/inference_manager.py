"""Inference manager service for model lifecycle management.

Handles starting, stopping, and switching models on the inference server.
"""

from datetime import UTC, datetime
from typing import Any

from src.services.model_storage import ModelStorageService


class ModelNotFoundError(Exception):
    """Raised when a requested model does not exist."""

    def __init__(self, model_id: str):
        """Initialize with model ID.

        Args:
            model_id: The ID of the model that was not found.
        """
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class InvalidModelStateError(Exception):
    """Raised when model is in an invalid state for the operation."""

    def __init__(self, model_id: str, current_state: str, required_state: str):
        """Initialize with model ID and state info.

        Args:
            model_id: The model identifier.
            current_state: Current state of the model.
            required_state: Required state for the operation.
        """
        self.model_id = model_id
        self.current_state = current_state
        self.required_state = required_state
        super().__init__(
            f"Model {model_id} is in '{current_state}' state, "
            f"but requires '{required_state}' state"
        )


class InferenceManagerService:
    """Service for managing model inference lifecycle.

    Handles model loading, unloading, and switching on the inference server.
    This service abstracts the underlying inference server (e.g., vLLM)
    and provides a clean interface for model lifecycle management.

    Attributes:
        model_storage: ModelStorageService for model persistence.
        _active_model_id: Currently active model ID.
        _runtime_info: Runtime information for active model.
    """

    def __init__(self, model_storage: ModelStorageService):
        """Initialize the inference manager service.

        Args:
            model_storage: ModelStorageService instance for model persistence.
        """
        self.model_storage = model_storage
        self._active_model_id: str | None = None
        self._runtime_info: dict[str, Any] = {}
        self._started_at: datetime | None = None

    async def get_active_model(self) -> str | None:
        """Get the currently active model ID.

        Returns:
            Active model ID or None if no model is active.
        """
        return self._active_model_id

    async def start_model(
        self,
        model_id: str,
        gpu_ids: list[int],
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Start a model on the inference server.

        Args:
            model_id: Model to start.
            gpu_ids: GPU device IDs to use.
            config: Optional runtime configuration.

        Returns:
            Dict with model_id, type, status, and message.

        Raises:
            ModelNotFoundError: If model doesn't exist.
            InvalidModelStateError: If model is not in 'ready' state.
        """
        # Get model and verify it exists
        model = await self.model_storage.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)

        # Verify model is ready
        if model.status not in ("ready", "stopped"):
            raise InvalidModelStateError(
                model_id, model.status, "ready or stopped"
            )

        # Update model status to loading
        await self.model_storage.update_model_status(model_id, "loading")

        # Set as active model
        self._active_model_id = model_id
        self._started_at = datetime.now(UTC)
        self._runtime_info = {
            "gpu_ids": gpu_ids,
            "config": config or {},
            "loaded_at": self._started_at.isoformat().replace("+00:00", "Z"),
        }

        # In a real implementation, this would start the inference server
        # For now, we simulate immediate loading
        await self.model_storage.update_model_status(model_id, "running")

        return {
            "model_id": model_id,
            "type": model.type,
            "status": "loading",
            "message": f"Loading model {model.name}",
        }

    async def stop_model(
        self,
        model_id: str,
        force: bool,
    ) -> dict[str, Any]:
        """Stop a running model.

        Args:
            model_id: Model to stop.
            force: Whether to force immediate stop.

        Returns:
            Dict with model_id, status, and message.

        Raises:
            ModelNotFoundError: If model doesn't exist.
            InvalidModelStateError: If model is not running.
        """
        # Get model and verify it exists
        model = await self.model_storage.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)

        # Verify model is running
        if model.status not in ("running", "loading"):
            raise InvalidModelStateError(
                model_id, model.status, "running or loading"
            )

        # Update model status to stopping
        await self.model_storage.update_model_status(model_id, "stopping")

        # Clear active model
        if self._active_model_id == model_id:
            self._active_model_id = None
            self._runtime_info = {}
            self._started_at = None

        # In a real implementation, this would stop the inference server
        # For now, we simulate immediate stop
        await self.model_storage.update_model_status(model_id, "ready")

        return {
            "model_id": model_id,
            "status": "stopping",
            "message": f"Stopping model {model.name}",
        }

    async def switch_model(
        self,
        target_model_id: str,
        gpu_ids: list[int],
        graceful_timeout_seconds: int,
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Switch from current model to a new model.

        Args:
            target_model_id: Model to switch to.
            gpu_ids: GPU device IDs to use.
            graceful_timeout_seconds: Time to wait for in-flight requests.
            config: Optional runtime configuration.

        Returns:
            Dict with previous_model_id, current_model_id, status, message.

        Raises:
            ModelNotFoundError: If target model doesn't exist.
            InvalidModelStateError: If target model is not ready.
        """
        previous_model_id = self._active_model_id

        # Verify target model exists and is ready
        target_model = await self.model_storage.get_model(target_model_id)
        if target_model is None:
            raise ModelNotFoundError(target_model_id)

        if target_model.status not in ("ready", "stopped"):
            raise InvalidModelStateError(
                target_model_id, target_model.status, "ready or stopped"
            )

        # Stop current model if one is active
        if previous_model_id is not None:
            await self.stop_model(previous_model_id, force=False)

        # Start target model
        await self.start_model(target_model_id, gpu_ids, config)

        return {
            "previous_model_id": previous_model_id,
            "current_model_id": target_model_id,
            "status": "switching",
            "message": f"Switched to model {target_model.name}",
        }

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the currently serving model.

        Returns:
            Dict with serving status and model information.
        """
        if self._active_model_id is None:
            return {
                "serving": False,
                "message": "No model currently serving",
            }

        model = await self.model_storage.get_model(self._active_model_id)
        if model is None:
            return {
                "serving": False,
                "message": "Active model not found",
            }

        # Calculate uptime
        uptime_seconds = 0
        if self._started_at:
            uptime_seconds = int((datetime.now(UTC) - self._started_at).total_seconds())

        return {
            "serving": True,
            "model": {
                "model_id": model.model_id,
                "name": model.name,
                "type": model.type,
                "source": model.source.model_dump() if model.source else None,
            },
            "resources": {
                "gpu": {
                    "gpu_ids": self._runtime_info.get("gpu_ids", []),
                    "gpu_count": len(self._runtime_info.get("gpu_ids", [])),
                    "total_memory_gb": 0.0,
                    "used_memory_gb": 0.0,
                    "memory_utilization_percent": 0.0,
                    "devices": [],
                },
                "memory": {
                    "model_memory_gb": 0.0,
                    "kv_cache_memory_gb": 0.0,
                },
            },
            "parameters": {
                "runtime_config": self._runtime_info.get("config", {}),
                "model_config": None,
                "tokenizer_config": None,
            },
            "stats": {
                "loaded_at": self._runtime_info.get("loaded_at", ""),
                "uptime_seconds": uptime_seconds,
                "requests_served": 0,
                "tokens_generated": 0,
                "avg_latency_ms": 0.0,
                "avg_tokens_per_second": 0.0,
                "current_batch_size": 0,
                "pending_requests": 0,
            },
        }
