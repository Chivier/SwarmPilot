"""
Scheduling strategies for task assignment to instances.

This module implements various scheduling strategies to select the best
instance for executing a task based on predictions.
"""

from typing import List, Optional, Dict, TYPE_CHECKING, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
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

        # logger.info(f"Prediction result: {predictions}")

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
            None
        )

        # Step 5: Update queue information
        if selected_prediction:
            await self.update_queue(selected_instance_id, selected_prediction)
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

        # Get custom quantiles if this is a probabilistic strategy
        quantiles = None
        if hasattr(self, 'quantiles'):
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
            logger.debug(f"sent prediction request model_id: {model_id}, metadata: {metadata}, instances: {available_instances}, prediction_type={prediction_type}, quantiles={quantiles}")
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

    async def collect_queue_info(
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
            queue = await self.instance_registry.get_queue_info(instance.instance_id)
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

    async def update_queue(
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

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueExpectError(
                instance_id=instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        elif not isinstance(current_queue, InstanceQueueExpectError):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueExpectError, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
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

        await self.instance_registry.update_queue_info(instance_id, updated_queue)
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
        target_quantile: float = 0.9,
    ):
        """
        Initialize probabilistic strategy.

        Args:
            predictor_client: Client for getting predictions
            instance_registry: Registry for instance and queue management
            target_quantile: Target quantile for probabilistic selection (default: 0.9)
        """
        super().__init__(predictor_client, instance_registry)
        self.quantiles = instance_registry._quantiles
        self.target_quantile = target_quantile

    def get_prediction_type(self) -> str:
        """Probabilistic strategy requires quantile predictions."""
        return "quantile"

    def select_instance(
        self, predictions: List[Prediction], queue_info: Dict[str, "InstanceQueueBase"]
    ) -> Optional[str]:
        """
        Select instance using Monte Carlo simulation on combined prediction + queue time.

        Performs vectorized Monte Carlo sampling (default: 10 samples) by:
        1. For each instance, interpolate both prediction quantiles and queue quantiles
        2. Sample from the same random percentiles to get prediction_time + queue_time
        3. Find the instance with minimum total time for each sample
        4. Return the instance with the most wins across all samples

        Args:
            predictions: List of predictions with quantile information
            queue_info: Dictionary mapping instance_id to queue information

        Returns:
            Selected instance ID (the one with most wins across Monte Carlo samples)
        """
        import numpy as np
        from .model import InstanceQueueProbabilistic

        if not predictions:
            return None

        num_samples = 10

        # Build prediction dictionary for quick lookup
        pred_dict = {p.instance_id: p for p in predictions}

        # Use all prediction instance IDs (queue_info is optional)
        instance_ids = list(pred_dict.keys())
        num_instances = len(instance_ids)

        # Generate all random percentiles at once (shared across all instances)
        random_percentiles = np.random.random(num_samples)

        # Build combined time matrix: shape (num_instances, num_samples)
        # Each entry = prediction_time + queue_time
        total_times_matrix = np.zeros((num_instances, num_samples))

        # Vectorized sampling for all instances
        for i, instance_id in enumerate(instance_ids):
            pred = pred_dict[instance_id]

            # Sample prediction times
            if pred.quantiles and len(pred.quantiles) > 0:
                # Convert dict to sorted arrays for interpolation
                pred_quantiles = np.array(sorted(pred.quantiles.keys()))
                pred_values = np.array([pred.quantiles[q] for q in pred_quantiles])

                # Vectorized prediction time sampling
                prediction_times = np.interp(
                    random_percentiles,
                    pred_quantiles,
                    pred_values
                )
            else:
                # Fallback: use predicted_time_ms as constant
                prediction_times = np.full(num_samples, pred.predicted_time_ms)

            # Sample queue times (optional)
            if instance_id in queue_info:
                queue = queue_info[instance_id]
                if isinstance(queue, InstanceQueueProbabilistic):
                    # Vectorized queue time sampling
                    queue_times = np.interp(
                        random_percentiles,
                        queue.quantiles,
                        queue.values
                    )
                else:
                    # Fallback: no probabilistic queue info, assume zero queue
                    queue_times = np.zeros(num_samples)
            else:
                # No queue info available for this instance
                queue_times = np.zeros(num_samples)

            # Combine prediction + queue time
            total_times_matrix[i, :] = prediction_times + queue_times

        # Find winner for each sample (argmin along axis 0)
        winners = np.argmin(total_times_matrix, axis=0)

        # Count wins for each instance
        win_counts = np.bincount(winners, minlength=num_instances)

        # Select instance with most wins
        best_instance_idx = np.argmax(win_counts)

        return instance_ids[best_instance_idx]

    async def update_queue(
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

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueProbabilistic(
                instance_id=instance_id,
                quantiles=[0.5, 0.9, 0.95, 0.99],
                values=[0.0, 0.0, 0.0, 0.0],
            )
        elif not isinstance(current_queue, InstanceQueueProbabilistic):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueProbabilistic, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
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
            await self.instance_registry.update_queue_info(instance_id, updated_queue)
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

        await self.instance_registry.update_queue_info(instance_id, updated_queue)
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

    async def update_queue(
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



class RandomStrategy(SchedulingStrategy):
    """
    Random scheduling strategy.

    Worst case of probabilistic
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
        
        return random.choice(predictions).instance_id


    async def update_queue(
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


class PowerOfTwoStrategy(SchedulingStrategy):
    """
    Random scheduling strategy.

    Worst case of probabilistic
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
        
        for idx in len(predictions):
            predictions[idx].predicted_time_ms = 1.0
        
        # Select Queue
        pred_1 = random.choice(predictions)
        pred_2 = random.choice(predictions)
        instance_1 = pred_1.instance_id
        instance_2 = pred_2.instance_id

        from .model import InstanceQueueExpectError

        if not predictions:
            return None
        
        # Retrive queue info
        queue_1 = queue_info.get(instance_1)
        queue_2 = queue_info.get(instance_2)

        # Compute metrics for queue 1
        if queue_1 and isinstance(queue_1, InstanceQueueExpectError):
            # Calculate total time: queue expected + queue error + new task expected
            total_time_1 = queue_1.expected_time_ms + pred_1.predicted_time_ms
        else:
            # Fallback: no queue info, just use prediction
            total_time_1 = pred_1.predicted_time_ms

        # Compute metrics for queue 2
        if queue_2 and isinstance(queue_1, InstanceQueueExpectError):
            # Calculate total time: queue expected + queue error + new task expected
            total_time_2 = queue_2.expected_time_ms + pred_2.predicted_time_ms
        else:
            # Fallback: no queue info, just use prediction
            total_time_2 = pred_2.predicted_time_ms    

        
        if total_time_1 < total_time_2:
            return instance_1
        else:
            return instance_2


    async def update_queue(
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

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueExpectError(
                instance_id=instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        elif not isinstance(current_queue, InstanceQueueExpectError):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueExpectError, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
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

        await self.instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Updated queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )


class MinimumExpectedTimeServerlessStrategy(SchedulingStrategy):
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

    async def update_queue(
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

        current_queue = await self.instance_registry.get_queue_info(instance_id)

        if not current_queue:
            # If no queue exists, initialize with correct type
            current_queue = InstanceQueueExpectError(
                instance_id=instance_id,
                expected_time_ms=0.0,
                error_margin_ms=0.0,
            )
        elif not isinstance(current_queue, InstanceQueueExpectError):
            # Type mismatch - this shouldn't happen if strategy switch was done properly
            logger.warning(
                f"Queue info type mismatch for {instance_id}: "
                f"expected InstanceQueueExpectError, got {type(current_queue).__name__}. "
                f"This indicates the strategy switch didn't properly reinitialize queues. Skipping update."
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

        await self.instance_registry.update_queue_info(instance_id, updated_queue)
        logger.debug(
            f"Updated queue (expect_error) for {instance_id}: "
            f"expected_time_ms={new_expected:.2f}, error_margin_ms={new_error:.2f}"
        )
        
# Factory function to get strategy by name
def get_strategy(
    strategy_name: str,
    predictor_client: "PredictorClient",
    instance_registry: "InstanceRegistry",
    target_quantile: float = 0.9,
    **kwargs
) -> SchedulingStrategy:
    """
    Get scheduling strategy by name.

    Args:
        strategy_name: Name of strategy
                      ("min_time", "probabilistic", "round_robin")
        predictor_client: Predictor client instance
        instance_registry: Instance registry instance
        target_quantile: Target quantile for probabilistic strategy (default: 0.9)
        **kwargs: Additional keyword arguments (reserved for future use)

    Returns:
        Configured scheduling strategy instance
    """
    if strategy_name == "min_time":
        return MinimumExpectedTimeStrategy(predictor_client, instance_registry)
    elif strategy_name == "probabilistic":
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )
    elif strategy_name == "round_robin":
        return RoundRobinStrategy(predictor_client, instance_registry)
    elif strategy_name == "random":
        return RandomStrategy(predictor_client, instance_registry)
    elif strategy_name == "po2":
        return PowerOfTwoStrategy(predictor_client, instance_registry)
    elif strategy_name == "severless":
        return MinimumExpectedTimeServerlessStrategy(predictor_client, instance_registry)
    else:
        # Default to probabilistic
        return ProbabilisticSchedulingStrategy(
            predictor_client, instance_registry, target_quantile=target_quantile
        )

