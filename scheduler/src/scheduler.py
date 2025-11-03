"""
Scheduling strategies for task assignment to instances.

This module implements various scheduling strategies to select the best
instance for executing a task based on predictions.
"""

from typing import List, Optional, Dict, TYPE_CHECKING, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import httpx
from loguru import logger

from .predictor_client import Prediction

if TYPE_CHECKING:
    from .model import InstanceQueueBase, Instance
    from .predictor_client import PredictorClient
    from .instance_registry import InstanceRegistry


@dataclass
class ScheduleResult:
    """Result of scheduling operation."""
    selected_instance_id: str
    selected_prediction: Optional[Prediction]


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies."""

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """
        Initialize scheduling strategy with dependencies.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
        """
        self.predictor_client = predictor_client
        self.instance_registry = instance_registry

    async def schedule_task(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        available_instances: List["Instance"],
    ) -> ScheduleResult:
        """
        Schedule a task to an instance (template method).

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

        logger.info(f"Prediction result: {predictions}")

        # Step 2: Collect queue information
        queue_info = self.collect_queue_info(available_instances)

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
            None
        )

        # Step 5: Update queue information
        if selected_prediction:
            self.update_queue(selected_instance_id, selected_prediction)
        else:
            logger.warning(f"No prediction is selected:"
                           f"predictions: {predictions}"
                           f"selected_instance_id: {selected_instance_id}")

        return ScheduleResult(
            selected_instance_id=selected_instance_id,
            selected_prediction=selected_prediction,
        )

    async def get_predictions(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        available_instances: List["Instance"],
    ) -> List[Prediction]:
        """
        Get predictions from predictor service.

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

        try:
            predictions = await self.predictor_client.predict(
                model_id=model_id,
                metadata=metadata,
                instances=available_instances,
                prediction_type=prediction_type,
            )
            logger.debug(f"sent prediction request model_id: {model_id}, metadata: {metadata}, instances: {available_instances}, prediction_type={prediction_type}")
            return predictions

        except httpx.HTTPStatusError as e:
            # Convert HTTP status errors to appropriate Python exceptions
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
            raise TimeoutError(
                f"Predictor service timeout: {str(e)}"
            ) from e

        except httpx.HTTPError as e:
            # Network errors
            raise ConnectionError(
                f"Predictor service unavailable: {str(e)}"
            ) from e

    def collect_queue_info(
        self,
        available_instances: List["Instance"],
    ) -> Dict[str, "InstanceQueueBase"]:
        """
        Collect queue information for all instances.

        Args:
            available_instances: List of instances to collect info for

        Returns:
            Dictionary mapping instance_id to queue info
        """
        queue_info = {}
        for instance in available_instances:
            queue = self.instance_registry.get_queue_info(instance.instance_id)
            if queue:
                queue_info[instance.instance_id] = queue
        return queue_info

    @abstractmethod
    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select the best instance from predictions.

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
        """
        Update queue information for selected instance.

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
        """
        Get the prediction type required by this strategy.

        Returns:
            Prediction type: "expect_error" or "quantile"
        """
        # Default to expect_error for simplicity
        return "expect_error"


class MinimumExpectedTimeStrategy(SchedulingStrategy):
    """
    Strategy that selects the instance with minimum expected queue completion time.

    This strategy considers both the current queue state (expected time and error margin)
    and the predicted time for the new task. It selects the instance that minimizes
    the total expected completion time including uncertainty.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize MinimumExpectedTimeStrategy."""
        super().__init__(predictor_client, instance_registry)

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select instance with minimum total expected time (queue + new task).

        For each instance, calculates: queue_expected + queue_error + task_expected
        and selects the instance with the minimum value.
        """
        from .model import InstanceQueueExpectError

        if not predictions:
            return None

        best_instance_id = None
        best_total_time = float("inf")

        for pred in predictions:
            # Get queue information for this instance
            queue = queue_info.get(pred.instance_id)

            if queue and isinstance(queue, InstanceQueueExpectError):
                # Calculate total time: queue expected + queue error + new task expected
                total_time = queue.expected_time_ms + queue.error_margin_ms + pred.predicted_time_ms
            else:
                # Fallback: no queue info, just use prediction
                total_time = pred.predicted_time_ms

            if total_time < best_total_time:
                best_total_time = total_time
                best_instance_id = pred.instance_id

        return best_instance_id

    def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """
        Update queue using error accumulation formula.

        Formula:
        - new_expected = current_expected + task_expected
        - new_error = sqrt(current_error^2 + task_error^2)

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        from .model import InstanceQueueExpectError
        import math

        current_queue = self.instance_registry.get_queue_info(instance_id)

        if not current_queue or not isinstance(current_queue, InstanceQueueExpectError):
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueExpectError, got {type(current_queue).__name__}"
            )
            return

        task_expected = prediction.predicted_time_ms
        task_error = prediction.error_margin_ms or 0.0

        # Calculate new queue expected time (simple addition)
        new_expected = current_queue.expected_time_ms + task_expected

        # Calculate new queue error margin (error accumulation)
        new_error = math.sqrt(
            current_queue.error_margin_ms ** 2 + task_error ** 2
        )

        updated_queue = InstanceQueueExpectError(
            instance_id=instance_id,
            expected_time_ms=new_expected,
            error_margin_ms=new_error,
        )

        self.instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Updated queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )


class ProbabilisticSchedulingStrategy(SchedulingStrategy):
    """
    Probabilistic scheduling strategy based on quantile sampling.

    This strategy considers both the queue distribution and the new task distribution.
    For each instance, it samples from the queue's quantile distribution to estimate
    completion time, then selects the instance with the minimum sampled value.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """
        Initialize probabilistic strategy.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
            target_quantile: Quantile to optimize for (default 0.9 = 90th percentile)
        """
        super().__init__(predictor_client, instance_registry)

    def get_prediction_type(self) -> str:
        """Probabilistic strategy requires quantile predictions."""
        return "quantile"

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select instance based on sampling from queue quantile distribution.

        For each instance, samples once from its queue distribution by interpolating
        between quantile values, then selects the instance with minimum sampled time.

        Args:
            predictions: List of predictions with quantile information
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID
        """
        import numpy as np
        from .model import InstanceQueueProbabilistic

        if not predictions:
            return None

        best_instance_id = None
        best_sampled_time = float("inf")

        for pred in predictions:
            # Get queue information for this instance
            queue = queue_info.get(pred.instance_id)

            if queue and isinstance(queue, InstanceQueueProbabilistic):
                # Sample from queue distribution
                # Use a random percentile between 0 and 1
                random_percentile = np.random.random()

                # Interpolate queue time from quantiles using numpy
                queue_time = np.interp(
                    random_percentile,
                    queue.quantiles,
                    queue.values
                )

                # Sample from task distribution
                if pred.quantiles:
                    task_quantiles = sorted(pred.quantiles.keys())
                    task_values = [pred.quantiles[q] for q in task_quantiles]
                    task_time = np.interp(random_percentile, task_quantiles, task_values)
                else:
                    task_time = pred.predicted_time_ms

                total_time = queue_time + task_time
            else:
                # Fallback: no queue info or wrong type
                total_time = pred.predicted_time_ms

            if total_time < best_sampled_time:
                best_sampled_time = total_time
                best_instance_id = pred.instance_id

        return best_instance_id

    def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """
        Update queue using Monte Carlo sampling method.

        Uses 1000 samples to compute new quantile distribution by combining
        queue distribution and task distribution.

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        from .model import InstanceQueueProbabilistic
        import numpy as np

        current_queue = self.instance_registry.get_queue_info(instance_id)

        if not current_queue or not isinstance(current_queue, InstanceQueueProbabilistic):
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueProbabilistic, got {type(current_queue).__name__}"
            )
            return

        if not prediction.quantiles:
            # Fallback: use predicted_time_ms for all quantiles
            updated_values = [
                current_queue.values[i] + prediction.predicted_time_ms
                for i in range(len(current_queue.quantiles))
            ]
            updated_queue = InstanceQueueProbabilistic(
                instance_id=instance_id,
                quantiles=current_queue.quantiles,
                values=updated_values,
            )
            self.instance_registry.update_queue_info(instance_id, updated_queue)
            logger.debug(
                f"Updated queue (probabilistic, fallback) for {instance_id}: "
                f"quantiles={current_queue.quantiles}, "
                f"values={[f'{v:.2f}' for v in updated_values]}"
            )
            return

        # Monte Carlo method
        num_samples = 1000
        random_percentiles = np.random.random(num_samples)

        # Sample from queue distribution using vectorized numpy interpolation
        queue_samples = np.interp(
            random_percentiles,
            current_queue.quantiles,
            current_queue.values
        )

        # Sample from task distribution
        task_quantiles = sorted(prediction.quantiles.keys())
        task_values = [prediction.quantiles[q] for q in task_quantiles]
        task_samples = np.interp(
            random_percentiles,
            task_quantiles,
            task_values
        )

        # Compute total time samples
        total_samples = queue_samples + task_samples

        # Compute new quantiles from samples using numpy.percentile
        updated_values = [
            float(np.percentile(total_samples, q * 100))
            for q in current_queue.quantiles
        ]

        updated_queue = InstanceQueueProbabilistic(
            instance_id=instance_id,
            quantiles=current_queue.quantiles,
            values=updated_values,
        )

        self.instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Updated queue (probabilistic, Monte Carlo) for {instance_id}: "
            f"quantiles={current_queue.quantiles}, "
            f"values={[f'{v:.2f}' for v in updated_values]}"
        )


class RoundRobinStrategy(SchedulingStrategy):
    """
    Round-robin scheduling strategy.

    Simple strategy that cycles through instances in order.
    Useful for load balancing when predictions are not available.
    """

    def __init__(
        self,
        predictor_client: "PredictorClient",
        instance_registry: "InstanceRegistry",
    ):
        """Initialize RoundRobinStrategy."""
        super().__init__(predictor_client, instance_registry)
        self._counter = 0

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """Select next instance in round-robin order."""
        if not predictions:
            return None

        selected = predictions[self._counter % len(predictions)]
        self._counter += 1
        return selected.instance_id

    def update_queue(
        self,
        instance_id: str,
        prediction: Prediction,
    ) -> None:
        """
        No-op for RoundRobinStrategy.

        RoundRobin doesn't use queue predictions for scheduling decisions,
        so no queue update is necessary.

        Args:
            instance_id: Selected instance
            prediction: Prediction for the task
        """
        # No-op: RoundRobin doesn't maintain queue state
        pass




# Factory function to get strategy by name
def get_strategy(
    strategy_name: str,
    predictor_client: "PredictorClient",
    instance_registry: "InstanceRegistry",
    **kwargs
) -> SchedulingStrategy:
    """
    Get scheduling strategy by name.

    Args:
        strategy_name: Name of strategy
                      ("min_time", "probabilistic", "round_robin")
        predictor_client: Predictor client instance
        instance_registry: Instance registry instance
        **kwargs: Additional keyword arguments passed to strategy constructor
                 For probabilistic: target_quantile (default 0.9)

    Returns:
        Configured scheduling strategy instance
    """
    if strategy_name == "min_time":
        return MinimumExpectedTimeStrategy(predictor_client, instance_registry)
    elif strategy_name == "probabilistic":
        target_quantile = kwargs.get("target_quantile", 0.9)
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry
        )
    elif strategy_name == "round_robin":
        return RoundRobinStrategy(predictor_client, instance_registry)
    else:
        # Default to probabilistic
        target_quantile = kwargs.get("target_quantile", 0.9)
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry
        )
