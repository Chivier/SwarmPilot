"""
Predictor client for getting task execution time predictions.

This module provides an interface to communicate with the predictor service
to get predictions for task execution times on different instances.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
import httpx
from dataclasses import dataclass
import logging
import asyncio
import time
import hashlib
import json

if TYPE_CHECKING:
    from .model import Instance

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction result for a task on a specific instance."""

    instance_id: str
    predicted_time_ms: float
    confidence: Optional[float] = None
    quantiles: Optional[Dict[float, float]] = None  # e.g., {0.5: 120.5, 0.9: 250.3}
    error_margin_ms: Optional[float] = None  # For expect_error prediction type


class PredictorClient:
    """Client for communicating with the predictor service."""

    def __init__(
        self,
        predictor_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_ttl: int = 300,
        enable_cache: bool = True,
    ):
        """
        Initialize predictor client.

        Args:
            predictor_url: Base URL of the predictor service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Initial delay between retries in seconds (uses exponential backoff)
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable prediction caching
        """
        self.predictor_url = predictor_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache

        # Cache: {cache_key: (prediction_data, expiry_timestamp)}
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _make_cache_key(
        self, model_id: str, platform_info: Dict[str, str], features: Dict[str, Any], prediction_type: str
    ) -> str:
        """
        Generate cache key for prediction request.

        Args:
            model_id: Model identifier
            platform_info: Platform information
            features: Task features
            prediction_type: Type of prediction

        Returns:
            Cache key string
        """
        # Create deterministic JSON representation
        cache_data = {
            "model_id": model_id,
            "platform_info": platform_info,
            "features": features,
            "prediction_type": prediction_type,
        }
        cache_json = json.dumps(cache_data, sort_keys=True)
        # Use hash for shorter keys
        return hashlib.md5(cache_json.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction from cache if not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached prediction data or None if not found/expired
        """
        if not self.enable_cache:
            return None

        if cache_key in self._cache:
            prediction_data, expiry_time = self._cache[cache_key]
            if time.time() < expiry_time:
                self._cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key}")
                return prediction_data
            else:
                # Expired, remove from cache
                del self._cache[cache_key]
                logger.debug(f"Cache entry expired for key {cache_key}")

        self._cache_misses += 1
        return None

    def _put_in_cache(self, cache_key: str, prediction_data: Dict[str, Any]) -> None:
        """
        Store prediction in cache with TTL.

        Args:
            cache_key: Cache key
            prediction_data: Prediction data to cache
        """
        if not self.enable_cache:
            return

        expiry_time = time.time() + self.cache_ttl
        self._cache[cache_key] = (prediction_data, expiry_time)
        logger.debug(f"Cached prediction for key {cache_key} (TTL: {self.cache_ttl}s)")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and current size
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear all cached predictions."""
        self._cache.clear()
        logger.info("Prediction cache cleared")

    async def _make_request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        json_data: Dict[str, Any],
    ) -> httpx.Response:
        """
        Make HTTP POST request with retry logic for transient failures.

        Args:
            client: HTTP client
            url: URL to request
            json_data: JSON data to send

        Returns:
            HTTP response

        Raises:
            httpx.HTTPStatusError: For non-retryable errors (4xx, 5xx that shouldn't retry)
            httpx.HTTPError: After exhausting all retries
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, json=json_data)
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                # Don't retry for client errors (4xx) as they won't succeed on retry
                if 400 <= e.response.status_code < 500:
                    logger.debug(f"Non-retryable error (HTTP {e.response.status_code}), not retrying")
                    raise
                # Retry for server errors (5xx)
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Predictor request failed (HTTP {e.response.status_code}), "
                        f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Predictor request failed after {self.max_retries} attempts"
                    )
                    raise

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                # Retry for transient network errors
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Predictor request failed ({type(e).__name__}), "
                        f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Predictor request failed after {self.max_retries} attempts"
                    )
                    raise httpx.HTTPError(
                        f"Predictor service unavailable after {self.max_retries} retries: {str(e)}"
                    )

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    async def predict(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        instances: List["Instance"],
        prediction_type: str = "quantile",
    ) -> List[Prediction]:
        """
        Get predictions for task execution time on multiple instances.

        This method implements platform-based batching: if multiple instances share
        the same platform_info, only one API call is made and the result is replicated.

        Args:
            model_id: Model/tool ID
            metadata: Task metadata for prediction (e.g., image dimensions)
            instances: List of Instance objects to get predictions for
            prediction_type: Type of prediction - "expect_error" or "quantile"

        Returns:
            List of predictions for each instance

        Raises:
            httpx.HTTPStatusError: If predictor returns error (404 for no model, 400 for bad features)
            httpx.HTTPError: If prediction request fails due to network issues
        """
        # Group instances by platform_info to minimize API calls
        from collections import defaultdict
        import json

        platform_to_instances = defaultdict(list)
        for instance in instances:
            # Use JSON serialization for consistent hashing of platform_info
            platform_key = json.dumps(instance.platform_info, sort_keys=True)
            platform_to_instances[platform_key].append(instance)

        logger.debug(
            f"Batching predictions: {len(instances)} instances across "
            f"{len(platform_to_instances)} unique platforms"
        )

        predictions = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Make one prediction request per unique platform
            for platform_key, platform_instances in platform_to_instances.items():
                # Use the first instance as representative for this platform
                representative_instance = platform_instances[0]
                try:
                    # Check cache first
                    cache_key = self._make_cache_key(
                        model_id=model_id,
                        platform_info=representative_instance.platform_info,
                        features=metadata,
                        prediction_type=prediction_type,
                    )
                    cached_data = self._get_from_cache(cache_key)

                    if cached_data is not None:
                        # Use cached prediction
                        data = cached_data
                        logger.debug(
                            f"Using cached prediction for platform {representative_instance.platform_info['hardware_name']}"
                        )
                    else:
                        # Construct predictor request using representative instance
                        request_data = {
                            "model_id": model_id,
                            "platform_info": representative_instance.platform_info,
                            "prediction_type": prediction_type,
                            "features": metadata,
                        }

                        logger.debug(
                            f"Requesting prediction for platform {representative_instance.platform_info} "
                            f"(covers {len(platform_instances)} instances)"
                        )

                        # Call predictor API with retry logic
                        response = await self._make_request_with_retry(
                            client=client,
                            url=f"{self.predictor_url}/predict",
                            json_data=request_data,
                        )
                        data = response.json()

                        # Cache the response
                        self._put_in_cache(cache_key, data)

                    # Parse response based on prediction type
                    if prediction_type == "expect_error":
                        predicted_time = data["result"]["expected_runtime_ms"]
                        error_margin = data["result"]["error_margin_ms"]

                        logger.info(
                            f"Prediction for platform {representative_instance.platform_info['hardware_name']}: "
                            f"{predicted_time:.2f}ms ± {error_margin:.2f}ms "
                            f"(applied to {len(platform_instances)} instances)"
                        )

                        # Replicate prediction to all instances with this platform
                        for instance in platform_instances:
                            prediction = Prediction(
                                instance_id=instance.instance_id,
                                predicted_time_ms=predicted_time,
                                confidence=None,
                                quantiles=None,
                                error_margin_ms=error_margin,
                            )
                            predictions.append(prediction)

                    elif prediction_type == "quantile":
                        quantiles_dict = data["result"]["quantiles"]
                        # Convert string keys to float
                        quantiles = {float(k): v for k, v in quantiles_dict.items()}
                        median = quantiles.get(0.5, list(quantiles.values())[0])

                        logger.info(
                            f"Prediction for platform {representative_instance.platform_info['hardware_name']}: "
                            f"P50={quantiles.get(0.5, 'N/A')}ms, "
                            f"P90={quantiles.get(0.9, 'N/A')}ms, "
                            f"P99={quantiles.get(0.99, 'N/A')}ms "
                            f"(applied to {len(platform_instances)} instances)"
                        )

                        # Replicate prediction to all instances with this platform
                        for instance in platform_instances:
                            prediction = Prediction(
                                instance_id=instance.instance_id,
                                predicted_time_ms=median,
                                confidence=None,
                                quantiles=quantiles,
                            )
                            predictions.append(prediction)

                    else:
                        raise ValueError(f"Unknown prediction type: {prediction_type}")

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # Model not trained for this platform - strict mode, reject task
                        platform_info = representative_instance.platform_info
                        logger.error(
                            f"No trained model for {model_id} on platform "
                            f"{platform_info}. Task submission rejected."
                        )
                        raise httpx.HTTPStatusError(
                            message=f"No trained model for {model_id} on platform "
                            f"{platform_info['software_name']}/"
                            f"{platform_info['hardware_name']}. "
                            f"Please train the model first or use experiment mode.",
                            request=e.request,
                            response=e.response,
                        )
                    elif e.response.status_code == 400:
                        # Invalid features
                        error_detail = e.response.json() if e.response.text else {}
                        logger.error(
                            f"Invalid features for prediction: {error_detail.get('detail', 'Unknown error')}"
                        )
                        raise httpx.HTTPStatusError(
                            message=f"Invalid task metadata: {error_detail.get('detail', 'Bad request')}",
                            request=e.request,
                            response=e.response,
                        )
                    else:
                        # Other HTTP errors
                        logger.error(
                            f"Predictor service error (HTTP {e.response.status_code}): {e}"
                        )
                        raise

                except httpx.HTTPError as e:
                    # Network errors, timeouts, etc.
                    platform_info = representative_instance.platform_info
                    logger.error(
                        f"Failed to get prediction for platform {platform_info}: {e}"
                    )
                    raise httpx.HTTPError(
                        f"Predictor service unavailable: {str(e)}"
                    )

        return predictions

    async def health_check(self) -> bool:
        """
        Check if predictor service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.predictor_url}/health")
                return response.status_code == 200
        except Exception:
            return False
