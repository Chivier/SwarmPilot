"""Adaptive bootstrap scheduling strategy.

Auto-detects cold-start vs warm models per platform. Uses round-robin
for cold platforms (no trained predictor) and probabilistic scheduling
across the warm ones (trained quantile predictor exists).
"""

import json
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
from swarmpilot.scheduler.clients._predictor_lib import is_experiment_mode
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

    On each scheduling call, checks which platforms have a trained
    quantile predictor model on disk (experiment-mode requests need none).
    If all are warm, delegates to ProbabilisticSchedulingStrategy. If none
    are, uses thread-safe round-robin with no prediction call.

    In a mixed pool a round-robin turn decides the group: a turn that lands
    on a cold instance goes to it, so cold platforms keep their fair share
    of traffic and gather training samples; a turn that lands on a warm
    instance is routed by prediction among the warm instances only.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
        target_quantile: float = 0.9,
        seed: int | None = None,
    ):
        """Initialize adaptive bootstrap strategy.

        Args:
            predictor_client: Client for getting predictions.
            instance_registry: Registry for instance and queue management.
            target_quantile: Target quantile for probabilistic selection.
            seed: Seed for the random generator (experiments only)
        """
        super().__init__(predictor_client, instance_registry, seed=seed)
        self._probabilistic = ProbabilisticSchedulingStrategy(
            predictor_client,
            instance_registry,
            target_quantile=target_quantile,
            seed=seed,
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

    def _is_warm(
        self,
        model_id: str,
        metadata: dict[str, Any],
        platform_info: dict[str, str],
    ) -> bool:
        if is_experiment_mode(metadata, platform_info):
            return True
        return self._has_trained_model(model_id, platform_info)

    async def schedule_task(
        self,
        model_id: str,
        metadata: dict[str, Any],
        available_instances: list["Instance"],
        task_id: str | None = None,
    ) -> ScheduleResult:
        """Schedule a task, choosing by prediction where models exist.

        Args:
            model_id: Model identifier.
            metadata: Task metadata for prediction.
            available_instances: List of available instances.
            task_id: The task being routed.

        Returns:
            ScheduleResult with selected instance and optional prediction.
        """
        warm_by_platform: dict[str, bool] = {}
        warm: list[Instance] = []
        for inst in available_instances:
            key = json.dumps(inst.platform_info, sort_keys=True)
            if key not in warm_by_platform:
                warm_by_platform[key] = self._is_warm(
                    model_id, metadata, inst.platform_info
                )
            if warm_by_platform[key]:
                warm.append(inst)

        if len(warm) == len(available_instances):
            logger.debug(
                f"[ADAPTIVE] All platforms warm for model_id={model_id}, "
                f"delegating to probabilistic strategy"
            )
            return await self._probabilistic.schedule_task(
                model_id, metadata, available_instances, task_id=task_id
            )

        with self._rr_lock:
            idx = self._rr_counter % len(available_instances)
            self._rr_counter += 1
        selected = available_instances[idx]

        if warm and any(selected is w for w in warm):
            try:
                return await self._probabilistic.schedule_task(
                    model_id, metadata, warm, task_id=task_id
                )
            except Exception as e:
                logger.warning(
                    f"[ADAPTIVE] Prediction failed for model_id={model_id} "
                    f"({e}); using round-robin pick {selected.instance_id}"
                )

        logger.debug(
            f"[ADAPTIVE] Round-robin for model_id={model_id}: "
            f"instance={selected.instance_id} "
            f"(index={idx}/{len(available_instances)}, "
            f"{len(warm)} warm)"
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
        task_id: str | None = None,
    ) -> None:
        """Not called directly (schedule_task is overridden).

        Args:
            instance_id: Selected instance.
            prediction: Prediction for the task.
            task_id: The task (unused)
        """
        pass

    def get_prediction_type(self) -> str:
        """Return quantile as the prediction type.

        Returns:
            "quantile" prediction type string.
        """
        return "quantile"
