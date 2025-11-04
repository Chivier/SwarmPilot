"""
Training client for sending runtime data to the predictor service.

This module collects actual task execution times and sends them to the
predictor service for model training.
"""

from typing import Dict, Any, List
import httpx
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from loguru import logger


@dataclass
class TrainingSample:
    """A single training sample for the predictor."""

    model_id: str
    platform_info: Dict[str, str]
    features: Dict[str, Any]
    actual_runtime_ms: float
    timestamp: str


class TrainingClient:
    """Client for sending training data to the predictor service."""

    def __init__(
        self,
        predictor_url: str,
        timeout: float = 10.0,
        batch_size: int = 100,
        min_samples: int = 10,
    ):
        """
        Initialize training client.

        Args:
            predictor_url: Base URL of the predictor service
            timeout: Request timeout in seconds
            batch_size: Number of samples to batch before sending
            min_samples: Minimum samples required before training
        """
        self.predictor_url = predictor_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.min_samples = min_samples

        # Buffer for collecting training samples
        self._samples_buffer: List[TrainingSample] = []

        # Reusable HTTP client with SSL verification disabled for internal network
        self._http_client = httpx.AsyncClient(
            timeout=timeout,
            verify=False,  # Disable SSL verification for internal network usage
        )

    def add_sample(
        self,
        model_id: str,
        platform_info: Dict[str, str],
        features: Dict[str, Any],
        actual_runtime_ms: float,
    ) -> None:
        """
        Add a training sample to the buffer.

        Args:
            model_id: Model/tool identifier
            platform_info: Platform information
            features: Task features
            actual_runtime_ms: Actual execution time in milliseconds
        """
        sample = TrainingSample(
            model_id=model_id,
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=actual_runtime_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._samples_buffer.append(sample)

        logger.debug(
            f"Added training sample for {model_id} on "
            f"{platform_info['hardware_name']}: {actual_runtime_ms:.2f}ms "
            f"(buffer size: {len(self._samples_buffer)})"
        )

    async def flush_if_ready(self) -> bool:
        """
        Flush buffer to predictor if batch size reached.

        Returns:
            True if training was triggered, False otherwise
        """
        if len(self._samples_buffer) >= self.batch_size:
            return await self.flush()
        return False

    async def flush(self, force: bool = False) -> bool:
        """
        Send all buffered samples to predictor for training.

        Args:
            force: If True, send even if below min_samples threshold

        Returns:
            True if training succeeded, False otherwise
        """
        if not self._samples_buffer:
            logger.debug("No training samples to flush")
            return False

        if not force and len(self._samples_buffer) < self.min_samples:
            logger.debug(
                f"Buffer has {len(self._samples_buffer)} samples, "
                f"below minimum of {self.min_samples}. Skipping training."
            )
            return False

        # Group samples by (model_id, platform_info) for separate training requests
        from collections import defaultdict
        import json

        grouped_samples = defaultdict(list)
        for sample in self._samples_buffer:
            # Use JSON serialization for consistent grouping
            platform_key = json.dumps(sample.platform_info, sort_keys=True)
            key = (sample.model_id, platform_key)
            grouped_samples[key].append(sample)

        logger.info(
            f"Flushing {len(self._samples_buffer)} training samples "
            f"across {len(grouped_samples)} model-platform combinations"
        )

        success_count = 0
        failure_count = 0

        for (model_id, platform_key), samples in grouped_samples.items():
            try:
                platform_info = samples[0].platform_info

                # Prepare training data
                training_data = {
                    "model_id": model_id,
                    "platform_info": platform_info,
                    "samples": [
                        {
                            "features": s.features,
                            "actual_runtime_ms": s.actual_runtime_ms,
                        }
                        for s in samples
                    ],
                }

                logger.debug(
                    f"Training {model_id} on {platform_info['hardware_name']} "
                    f"with {len(samples)} samples"
                )

                # Send training request
                response = await self._http_client.post(
                    f"{self.predictor_url}/train",
                    json=training_data,
                )
                response.raise_for_status()

                logger.info(
                    f"Successfully trained {model_id} on "
                    f"{platform_info['hardware_name']} "
                    f"with {len(samples)} samples"
                )
                success_count += 1

            except httpx.HTTPError as e:
                logger.error(
                    f"Failed to train {model_id} on {platform_info['hardware_name']}: {e}"
                )
                failure_count += 1

        # Clear buffer after training attempt
        self._samples_buffer.clear()

        logger.info(
            f"Training complete: {success_count} succeeded, {failure_count} failed"
        )

        return failure_count == 0

    def get_buffer_size(self) -> int:
        """Get current number of buffered training samples."""
        return len(self._samples_buffer)

    def clear_buffer(self) -> None:
        """Clear all buffered training samples without sending."""
        count = len(self._samples_buffer)
        self._samples_buffer.clear()
        logger.info(f"Cleared {count} training samples from buffer")

    async def close(self) -> None:
        """
        Close the HTTP client and cleanup resources.

        Should be called when shutting down the training client.
        """
        await self._http_client.aclose()
