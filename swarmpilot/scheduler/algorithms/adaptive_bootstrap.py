"""Adaptive bootstrap scheduling strategy.

Auto-detects cold-start vs warm models per platform. Uses round-robin
for cold models (no trained predictor) and probabilistic scheduling
for warm models (trained quantile predictor exists).
"""

import threading
from typing import TYPE_CHECKING, Any

from loguru import logger

from swarmpilot.scheduler.algorithms.base import (
    ScheduleResult,
    SchedulingStrategy,
)
from swarmpilot.scheduler.algorithms.probabilistic import (
    ProbabilisticSchedulingStrategy,
)
from swarmpilot.scheduler.clients.models import Prediction

if TYPE_CHECKING:
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.models import Instance, InstanceQueueBase
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
    from swarmpilot.scheduler.services.worker_queue_manager import (
        WorkerQueueManager,
    )


class AdaptiveBootstrapStrategy(SchedulingStrategy):
    """Adaptive strategy that bootstraps from round-robin to probabilistic.

    On each scheduling call, checks whether all platforms have a trained
    quantile predictor model on disk. If all are warm, delegates to
    ProbabilisticSchedulingStrategy. If any are cold, falls back to
    thread-safe round-robin selection with no prediction call.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
        target_quantile: float = 0.9,
    ):
        """Initialize adaptive bootstrap strategy.

        Args:
            predictor_client: Client for getting predictions.
            instance_registry: Registry for instance and queue management.
            target_quantile: Target quantile for probabilistic selection.
        """
        super().__init__(predictor_client, instance_registry)
        self._probabilistic = ProbabilisticSchedulingStrategy(
            predictor_client,
            instance_registry,
            target_quantile=target_quantile,
        )
        self.target_quantile = target_quantile
        self.quantiles = instance_registry._quantiles
        self._rr_counter = 0
        self._rr_lock = threading.Lock()

    def _has_trained_model(
        self,
        model_id: str,
        platform_info: dict[str, str],
    ) -> bool:
        """Check if a trained quantile model exists for a platform.

        Args:
            model_id: Model identifier.
            platform_info: Platform info dict.

        Returns:
            True if a trained quantile model exists on disk.
        """
        model_key = (
            self.predictor_client._low_level._storage.generate_model_key(
                model_id=model_id,
                platform_info=platform_info,
                prediction_type="quantile",
            )
        )
        return self.predictor_client._low_level._storage.model_exists(model_key)

    async def schedule_task(
        self,
        model_id: str,
        metadata: dict[str, Any],
        available_instances: list["Instance"],
    ) -> ScheduleResult:
        """Schedule a task, auto-selecting strategy per model warmth.

        Checks all platforms for trained quantile models. If all warm,
        delegates to the probabilistic strategy. Otherwise, selects
        via thread-safe round-robin without any prediction call.

        Args:
            model_id: Model identifier.
            metadata: Task metadata for prediction.
            available_instances: List of available instances.

        Returns:
            ScheduleResult with selected instance and optional prediction.
        """
        all_warm = all(
            self._has_trained_model(model_id, inst.platform_info)
            for inst in available_instances
        )

        if all_warm:
            logger.debug(
                f"[ADAPTIVE] All platforms warm for model_id={model_id}, "
                f"delegating to probabilistic strategy"
            )
            return await self._probabilistic.schedule_task(
                model_id, metadata, available_instances
            )

        # Cold path: round-robin, no prediction call
        with self._rr_lock:
            idx = self._rr_counter % len(available_instances)
            self._rr_counter += 1

        selected = available_instances[idx]
        logger.info(
            f"[ADAPTIVE] Cold-start for model_id={model_id}, "
            f"round-robin selected instance={selected.instance_id} "
            f"(index={idx}/{len(available_instances)})"
        )

        return ScheduleResult(
            selected_instance_id=selected.instance_id,
            selected_prediction=None,
        )

    def set_worker_queue_manager(self, manager: "WorkerQueueManager") -> None:
        """Set the worker queue manager on both self and delegate.

        Args:
            manager: WorkerQueueManager instance.
        """
        super().set_worker_queue_manager(manager)
        self._probabilistic.set_worker_queue_manager(manager)

    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Not called directly (schedule_task is overridden).

        Args:
            predictions: List of predictions.
            queue_info: Queue info dict.

        Returns:
            None (delegates handle selection).
        """
        return None

    async def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """Not called directly (schedule_task is overridden).

        Args:
            instance_id: Selected instance.
            prediction: Prediction for the task.
        """
        pass

    def get_prediction_type(self) -> str:
        """Return quantile as the prediction type.

        Returns:
            "quantile" prediction type string.
        """
        return "quantile"
