"""Base scheduling strategy and result classes.

This module defines the abstract base class for all scheduling strategies
and the common ScheduleResult dataclass.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from src.http_error_logger import log_http_error
from src.predictor_client import Prediction

if TYPE_CHECKING:
    from src.instance_registry import InstanceRegistry
    from src.model import Instance, InstanceQueueBase
    from src.predictor_client import PredictorClient


random.seed(42)


@dataclass
class ScheduleResult:
    """Result of scheduling operation."""

    selected_instance_id: str
    selected_prediction: Prediction | None


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies."""

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize scheduling strategy with dependencies.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
        """
        self.predictor_client = predictor_client
        self.instance_registry = instance_registry

    async def schedule_task(
        self,
        model_id: str,
        metadata: dict[str, Any],
        available_instances: list["Instance"],
    ) -> ScheduleResult:
        """Schedule a task to an instance (template method).

        This orchestrates the complete scheduling workflow:
        1. Get predictions from predictor service
        2. Collect queue information for all instances
        3. Select best instance using strategy
        4. Update queue information for selected instance

        Args:
            model_id: Model identifier
            metadata: Task metadata for prediction
            available_instances: List of available instances

        Returns:
            ScheduleResult containing selected instance and prediction

        Raises:
            ValueError: Invalid metadata or no trained model
            ConnectionError: Predictor service unavailable
            TimeoutError: Request timeout
        """
        # Step 1: Get predictions
        predictions = await self.get_predictions(
            model_id=model_id,
            metadata=metadata,
            available_instances=available_instances,
        )

        # Step 2: Collect queue information
        queue_info = await self.collect_queue_info(available_instances)

        # Step 3: Select best instance
        selected_instance_id = self.select_instance(predictions, queue_info)

        if not selected_instance_id:
            # Fallback to first instance if selection fails
            selected_instance_id = available_instances[0].instance_id
            logger.warning(
                f"Strategy {self.__class__.__name__} failed to select instance, "
                f"falling back to {selected_instance_id}"
            )

        # Step 4: Get prediction for selected instance
        selected_prediction = next(
            (p for p in predictions if p.instance_id == selected_instance_id),
            None,
        )

        # Step 5: Update queue information
        if selected_prediction:
            await self.update_queue(selected_instance_id, selected_prediction)
        else:
            logger.warning(
                f"No prediction is selected:"
                f"predictions: {predictions}"
                f"selected_instance_id: {selected_instance_id}"
            )

        return ScheduleResult(
            selected_instance_id=selected_instance_id,
            selected_prediction=selected_prediction,
        )

    async def get_predictions(
        self,
        model_id: str,
        metadata: dict[str, Any],
        available_instances: list["Instance"],
    ) -> list[Prediction]:
        """Get predictions from predictor service.

        Uses the prediction type specified by get_prediction_type().
        Converts httpx exceptions to standard Python exceptions.

        Args:
            model_id: Model identifier
            metadata: Task metadata
            available_instances: List of instances to predict for

        Returns:
            List of predictions for each instance

        Raises:
            ValueError: Invalid metadata or model not found (404, 400 status)
            ConnectionError: Predictor service unavailable
            TimeoutError: Request timeout
        """
        prediction_type = self.get_prediction_type()

        # Get custom quantiles if this is a probabilistic strategy
        quantiles = None
        if hasattr(self, "quantiles"):
            quantiles = self.quantiles

        try:
            # Build kwargs for predict call
            predict_kwargs = {
                "model_id": model_id,
                "metadata": metadata,
                "instances": available_instances,
                "prediction_type": prediction_type,
            }

            # Only add quantiles if not None
            if quantiles is not None and prediction_type == "quantile":
                predict_kwargs["quantiles"] = quantiles

            predictions = await self.predictor_client.predict(**predict_kwargs)
            logger.debug(
                f"sent prediction request model_id: {model_id}, metadata: {metadata}, instances: {available_instances}, prediction_type={prediction_type}, quantiles={quantiles}"
            )
            return predictions

        except httpx.HTTPStatusError as e:
            # Convert HTTP status errors to appropriate Python exceptions
            log_http_error(
                e,
                request_body={
                    "model_id": model_id,
                    "instances": [i.instance_id for i in available_instances],
                    "prediction_type": prediction_type,
                    "metadata": metadata,
                },
                context="scheduler prediction request",
            )
            if e.response.status_code == 404:
                raise ValueError(
                    "No trained model available for this platform. "
                    "Please train the model first or use experiment mode."
                ) from e
            elif e.response.status_code == 400:
                raise ValueError(
                    f"Invalid task metadata: {e.response.text}"
                ) from e
            else:
                raise ConnectionError(
                    f"Predictor service error: {e.response.status_code}"
                ) from e

        except httpx.TimeoutException as e:
            log_http_error(
                e,
                request_body={
                    "model_id": model_id,
                    "prediction_type": prediction_type,
                },
                context="scheduler prediction timeout",
            )
            raise TimeoutError(f"Predictor service timeout: {e!s}") from e

        except httpx.HTTPError as e:
            # Network errors
            log_http_error(
                e,
                request_body={
                    "model_id": model_id,
                    "prediction_type": prediction_type,
                },
                context="scheduler prediction connection error",
            )
            raise ConnectionError(
                f"Predictor service unavailable: {e!s}"
            ) from e

    async def collect_queue_info(
        self,
        available_instances: list["Instance"],
    ) -> dict[str, "InstanceQueueBase"]:
        """Collect queue information for all instances.

        Uses a single lock acquisition via get_all_queue_info for efficiency.

        Args:
            available_instances: List of instances to collect info for

        Returns:
            Dictionary mapping instance_id to queue info
        """
        instance_ids = [inst.instance_id for inst in available_instances]
        return await self.instance_registry.get_all_queue_info(instance_ids)

    @abstractmethod
    def select_instance(
        self,
        predictions: list[Prediction],
        queue_info: dict[str, "InstanceQueueBase"],
    ) -> str | None:
        """Select the best instance from predictions.

        Args:
            predictions: List of predictions for different instances
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID, or None if no suitable instance found
        """
        pass

    @abstractmethod
    def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """Update queue information for selected instance.

        Each strategy implements its own queue update logic:
        - MinimumExpectedTimeStrategy: error accumulation
        - ProbabilisticSchedulingStrategy: Monte Carlo sampling
        - RoundRobinStrategy: no-op

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        pass

    def get_prediction_type(self) -> str:
        """Get the prediction type required by this strategy.

        Returns:
            Prediction type: "expect_error" or "quantile"
        """
        # Default to expect_error for simplicity
        return "expect_error"
