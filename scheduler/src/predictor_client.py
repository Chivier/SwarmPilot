"""
Predictor client for getting task execution time predictions.

This module provides an interface to communicate with the predictor service
to get predictions for task execution times on different instances.
Uses WebSocket for real-time bidirectional communication.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING, Tuple
import websockets
from dataclasses import dataclass
import asyncio
import time
import hashlib
import json
from loguru import logger

if TYPE_CHECKING:
    from .model import Instance


@dataclass
class Prediction:
    """Prediction result for a task on a specific instance."""

    instance_id: str
    predicted_time_ms: float
    confidence: Optional[float] = None
    quantiles: Optional[Dict[float, float]] = None  # e.g., {0.5: 120.5, 0.9: 250.3}
    error_margin_ms: Optional[float] = None  # For expect_error prediction type


class PredictorClient:
    """Client for communicating with the predictor service via WebSocket."""

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
            predictor_url: Base URL of the predictor service (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Initial delay between retries in seconds (uses exponential backoff)
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable prediction caching
        """
        # Convert HTTP URL to WebSocket URL
        self.predictor_url = predictor_url.rstrip("/")
        if self.predictor_url.startswith("http://"):
            self.ws_url = self.predictor_url.replace("http://", "ws://", 1)
        elif self.predictor_url.startswith("https://"):
            self.ws_url = self.predictor_url.replace("https://", "wss://", 1)
        else:
            # Assume it's already a WebSocket URL
            self.ws_url = self.predictor_url

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_ttl = cache_ttl
        self.enable_cache = enable_cache

        # Cache: {cache_key: (prediction_data, expiry_timestamp)}
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # WebSocket connection reuse
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_lock = asyncio.Lock()  # Ensure thread-safe access to websocket
        self._ws_endpoint = f"{self.ws_url}/ws/predict"

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

    async def _ensure_connection(self) -> websockets.WebSocketClientProtocol:
        """
        Ensure WebSocket connection is established and healthy.

        Returns:
            Active WebSocket connection

        Raises:
            ConnectionError: If connection cannot be established
        """
        # Check if connection exists and is not closed
        if self._websocket is not None:
            # Check if connection is still open by examining the state
            # websockets uses close_code to determine if connection is closed
            if self._websocket.close_code is None:
                # Connection is still open
                return self._websocket
            else:
                # Connection was closed, clear it
                logger.debug("Existing WebSocket connection is closed, reconnecting")
                self._websocket = None

        # Connection doesn't exist or is closed, establish new one
        try:
            logger.debug(f"Establishing WebSocket connection to {self._ws_endpoint}")
            self._websocket = await asyncio.wait_for(
                websockets.connect(self._ws_endpoint),
                timeout=self.timeout
            )
            logger.info(f"WebSocket connection established to {self._ws_endpoint}")
            return self._websocket
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            self._websocket = None
            raise ConnectionError(f"Failed to connect to predictor service: {e}")

    async def _close_connection(self) -> None:
        """Close the WebSocket connection if it exists."""
        if self._websocket is not None:
            try:
                await self._websocket.close()
                logger.debug("WebSocket connection closed")
            except Exception as e:
                logger.debug(f"Error closing WebSocket connection: {e}")
            finally:
                self._websocket = None

    async def close(self) -> None:
        """Close the predictor client and cleanup resources."""
        async with self._ws_lock:
            await self._close_connection()
        logger.info("PredictorClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _make_request_with_retry(
        self,
        ws_url: str,
        json_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make WebSocket request with retry logic for transient failures.

        This method reuses the WebSocket connection when possible, only reconnecting
        when the connection is closed or encounters errors.

        Args:
            ws_url: WebSocket URL to connect to (used for validation, actual endpoint from __init__)
            json_data: JSON data to send

        Returns:
            Response data as dictionary

        Raises:
            ConnectionError: For non-retryable connection errors
            TimeoutError: After exhausting all retries
            ValueError: For invalid response data
        """
        last_exception = None

        # Use lock to ensure thread-safe access to the shared WebSocket connection
        async with self._ws_lock:
            for attempt in range(self.max_retries):
                try:
                    # Ensure we have an active connection
                    websocket = await self._ensure_connection()

                    # Send request
                    await asyncio.wait_for(
                        websocket.send(json.dumps(json_data)),
                        timeout=self.timeout
                    )

                    # Receive response
                    response_text = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self.timeout
                    )

                    response_data = json.loads(response_text)

                    # Check for error responses
                    if "error" in response_data:
                        error_msg = response_data.get("message", "Unknown error")
                        error_type = response_data.get("error", "Unknown")

                        # Don't retry for client errors (invalid requests)
                        if error_type in ["Invalid request", "Invalid features", "Model not found", "Prediction type mismatch"]:
                            logger.debug(f"Non-retryable error ({error_type}), not retrying")
                            raise ValueError(f"{error_type}: {error_msg}")

                        # Retry for server errors (close connection first)
                        await self._close_connection()
                        raise RuntimeError(f"{error_type}: {error_msg}")

                    # Request successful
                    return response_data

                except (websockets.exceptions.WebSocketException, ConnectionError, OSError) as e:
                    # Connection error - close and retry
                    await self._close_connection()
                    last_exception = e

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Predictor WebSocket connection failed ({type(e).__name__}), "
                            f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Predictor request failed after {self.max_retries} attempts"
                        )
                        raise ConnectionError(
                            f"Predictor service unavailable after {self.max_retries} retries: {str(e)}"
                        )

                except asyncio.TimeoutError as e:
                    # Timeout - close connection and retry
                    await self._close_connection()
                    last_exception = e

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Predictor request timeout, "
                            f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Predictor request failed after {self.max_retries} attempts"
                        )
                        raise TimeoutError(
                            f"Predictor service timeout after {self.max_retries} retries"
                        )

                except (json.JSONDecodeError, ValueError) as e:
                    # Don't retry for parsing/validation errors
                    logger.error(f"Invalid response from predictor: {e}")
                    raise

                except RuntimeError as e:
                    # Server error, retry
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Predictor request failed (server error), "
                            f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Predictor request failed after {self.max_retries} attempts"
                        )
                        raise

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
            ValueError: If predictor returns error (Model not found, invalid features)
            ConnectionError: If prediction request fails due to network issues
            TimeoutError: If request times out
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

                    # Call predictor via WebSocket with retry logic
                    data = await self._make_request_with_retry(
                        ws_url=f"{self.ws_url}/ws/predict",
                        json_data=request_data,
                    )

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

            except ValueError as e:
                # Model not found, invalid features, or other validation errors
                platform_info = representative_instance.platform_info
                error_msg = str(e)

                if "Model not found" in error_msg:
                    logger.error(
                        f"No trained model for {model_id} on platform "
                        f"{platform_info}. Task submission rejected."
                    )
                    raise ValueError(
                        f"No trained model for {model_id} on platform "
                        f"{platform_info['software_name']}/"
                        f"{platform_info['hardware_name']}. "
                        f"Please train the model first or use experiment mode."
                    )
                elif "Invalid features" in error_msg or "Invalid request" in error_msg:
                    logger.error(
                        f"Invalid features for prediction: {error_msg}"
                    )
                    raise ValueError(f"Invalid task metadata: {error_msg}")
                else:
                    # Other validation errors
                    logger.error(f"Prediction validation error: {error_msg}")
                    raise

            except (ConnectionError, TimeoutError) as e:
                # Network errors, timeouts, etc.
                platform_info = representative_instance.platform_info
                logger.error(
                    f"Failed to get prediction for platform {platform_info}: {e}"
                )
                raise

        return predictions

    async def health_check(self) -> bool:
        """
        Check if predictor service is healthy by ensuring WebSocket connection.

        Returns:
            True if service is healthy and WebSocket is accessible, False otherwise
        """
        async with self._ws_lock:
            try:
                # Try to ensure connection exists
                await self._ensure_connection()
                return True
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
                await self._close_connection()
                return False
