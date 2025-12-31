"""Predictor client for getting task execution time predictions.

This module provides an interface to communicate with the predictor service
to get predictions for task execution times on different instances.
Uses HTTP API for communication.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from src.http_error_logger import log_http_error

if TYPE_CHECKING:
    from .model import Instance


@dataclass
class Prediction:
    """Prediction result for a task on a specific instance."""

    instance_id: str
    predicted_time_ms: float
    confidence: float | None = None
    quantiles: dict[float, float] | None = (
        None  # e.g., {0.5: 120.5, 0.9: 250.3}
    )
    error_margin_ms: float | None = None  # For expect_error prediction type


class PredictorClient:
    """Client for communicating with the predictor service via HTTP API."""

    def __init__(
        self,
        predictor_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize predictor client.

        Args:
            predictor_url: Base URL of the predictor service (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for transient failures
            retry_delay: Initial delay between retries in seconds (uses exponential backoff)
        """
        self.predictor_url = predictor_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # HTTP client for making requests
        self._client: httpx.AsyncClient | None = None
        self._client_lock = (
            asyncio.Lock()
        )  # Ensure thread-safe access to HTTP client
        self._predict_endpoint = f"{self.predictor_url}/predict"
        self._health_endpoint = f"{self.predictor_url}/health"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.

        Returns:
            Active HTTP client

        Raises:
            ConnectionError: If client cannot be initialized
        """
        # Check if client exists and is not closed
        if self._client is not None and not self._client.is_closed:
            return self._client

        # Client doesn't exist or is closed, create new one
        try:
            logger.debug(f"Initializing HTTP client for {self.predictor_url}")
            self._client = httpx.AsyncClient(
                timeout=self.timeout, base_url=self.predictor_url
            )
            logger.debug(f"HTTP client initialized for {self.predictor_url}")
            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize HTTP client: {e}")
            self._client = None
            raise ConnectionError(f"Failed to initialize HTTP client: {e}") from e

    async def _close_client(self) -> None:
        """Close the HTTP client if it exists."""
        if self._client is not None:
            try:
                await self._client.aclose()
                logger.debug("HTTP client closed")
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")
            finally:
                self._client = None

    async def close(self) -> None:
        """Close the predictor client and cleanup resources."""
        async with self._client_lock:
            await self._close_client()
        logger.info("PredictorClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _make_request_with_retry(
        self,
        json_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make HTTP POST request with retry logic for transient failures.

        This method reuses the HTTP client when possible, only recreating
        when the client is closed or encounters errors.

        Args:
            json_data: JSON data to send in request body

        Returns:
            Response data as dictionary

        Raises:
            ConnectionError: For non-retryable connection errors
            TimeoutError: After exhausting all retries
            ValueError: For invalid response data
        """
        last_exception = None

        # Use lock to ensure thread-safe access to the shared HTTP client
        async with self._client_lock:
            for attempt in range(self.max_retries):
                try:
                    # Ensure we have an active client
                    client = await self._ensure_client()

                    # Make HTTP POST request
                    response = await client.post(
                        "/predict", json=json_data, timeout=self.timeout
                    )

                    # Check HTTP status code
                    if response.status_code == 200:
                        # Request successful
                        response_data = response.json()
                        return response_data
                    elif response.status_code == 400:
                        # Bad request - don't retry
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("detail", {})
                            if isinstance(error_msg, dict):
                                error_type = error_msg.get(
                                    "error", "Invalid request"
                                )
                                error_message = error_msg.get(
                                    "message", "Unknown error"
                                )
                            else:
                                error_type = "Invalid request"
                                error_message = str(error_msg)
                        except Exception:
                            error_type = "Invalid request"
                            error_message = response.text

                        log_http_error(
                            ValueError(f"{error_type}: {error_message}"),
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            response=response,
                            context="predictor bad request (400)",
                        )
                        logger.debug(
                            f"Non-retryable error ({error_type}), not retrying"
                        )
                        raise ValueError(f"{error_type}: {error_message}")
                    elif response.status_code == 404:
                        # Model not found - don't retry
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("detail", {})
                            if isinstance(error_msg, dict):
                                error_type = error_msg.get(
                                    "error", "Model not found"
                                )
                                error_message = error_msg.get(
                                    "message", "Unknown error"
                                )
                            else:
                                error_type = "Model not found"
                                error_message = str(error_msg)
                        except Exception:
                            error_type = "Model not found"
                            error_message = response.text

                        log_http_error(
                            ValueError(f"{error_type}: {error_message}"),
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            response=response,
                            context="predictor model not found (404)",
                        )
                        logger.debug(
                            f"Non-retryable error ({error_type}), not retrying"
                        )
                        raise ValueError(f"{error_type}: {error_message}")
                    elif response.status_code >= 500:
                        # Server error - retry
                        await self._close_client()
                        error_msg = response.text
                        log_http_error(
                            RuntimeError(
                                f"Server error (HTTP {response.status_code}): {error_msg}"
                            ),
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            response=response,
                            context="predictor server error (5xx)",
                        )
                        raise RuntimeError(
                            f"Server error (HTTP {response.status_code}): {error_msg}"
                        )
                    else:
                        # Other HTTP errors
                        await self._close_client()
                        error_msg = response.text
                        log_http_error(
                            ValueError(
                                f"HTTP {response.status_code}: {error_msg}"
                            ),
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            response=response,
                            context="predictor HTTP error",
                        )
                        raise ValueError(
                            f"HTTP {response.status_code}: {error_msg}"
                        )

                except (
                    httpx.NetworkError,
                    httpx.ConnectError,
                    httpx.PoolTimeout,
                    ConnectionError,
                    OSError,
                ) as e:
                    # Connection error - close client and retry
                    await self._close_client()
                    last_exception = e

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        logger.warning(
                            f"Predictor HTTP connection failed ({type(e).__name__}), "
                            f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        log_http_error(
                            e,
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            context="predictor connection error",
                            extra={"attempts": self.max_retries},
                        )
                        logger.error(
                            f"Predictor request failed after {self.max_retries} attempts"
                        )
                        raise ConnectionError(
                            f"Predictor service unavailable after {self.max_retries} retries: {e!s}"
                        ) from e

                except (TimeoutError, httpx.TimeoutException) as e:
                    # Timeout - close client and retry
                    await self._close_client()
                    last_exception = e

                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        logger.warning(
                            f"Predictor request timeout, "
                            f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(delay)
                    else:
                        log_http_error(
                            e,
                            request_url=self._predict_endpoint,
                            request_method="POST",
                            request_body=json_data,
                            context="predictor timeout",
                            extra={
                                "attempts": self.max_retries,
                                "timeout_seconds": self.timeout,
                            },
                        )
                        logger.error(
                            f"Predictor request failed after {self.max_retries} attempts"
                        )
                        raise TimeoutError(
                            f"Predictor service timeout after {self.max_retries} retries"
                        ) from e

                except (json.JSONDecodeError, ValueError) as e:
                    # Don't retry for parsing/validation errors
                    logger.error(f"Invalid response from predictor: {e}")
                    raise

                except RuntimeError as e:
                    # Server error, retry
                    last_exception = e
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (
                            2**attempt
                        )  # Exponential backoff
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
        metadata: dict[str, Any],
        instances: list["Instance"],
        prediction_type: str = "quantile",
        quantiles: list[float] | None = None,
    ) -> list[Prediction]:
        """Get predictions for task execution time on multiple instances.

        This method implements platform-based batching: if multiple instances share
        the same platform_info, only one API call is made and the result is replicated.

        Args:
            model_id: Model/tool ID
            metadata: Task metadata for prediction (e.g., image dimensions)
            instances: List of Instance objects to get predictions for
            prediction_type: Type of prediction - "expect_error" or "quantile"
            quantiles: Optional list of quantile levels for prediction (only used in experiment mode)

        Returns:
            List of predictions for each instance

        Raises:
            ValueError: If predictor returns error (Model not found, invalid features)
            ConnectionError: If prediction request fails due to network issues
            TimeoutError: If request times out
        """
        # Group instances by platform_info to minimize API calls
        import json
        from collections import defaultdict

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
        for _platform_key, platform_instances in platform_to_instances.items():
            # Use the first instance as representative for this platform
            representative_instance = platform_instances[0]
            try:
                # Construct predictor request using representative instance
                request_data = {
                    "model_id": model_id,
                    "platform_info": representative_instance.platform_info,
                    "prediction_type": prediction_type,
                    "features": metadata,
                }

                if "llm_service" in model_id and "model" in model_id:
                    # Enable semantic preprocessor for LLM service model
                    request_data["enable_preprocessors"] = ["semantic"]
                    request_data["preprocessor_mappings"] = {
                        "semantic": ["sentence"]
                    }

                # Add custom quantiles if provided (only used in experiment mode)
                if quantiles is not None:
                    request_data["quantiles"] = quantiles

                logger.debug(
                    f"Requesting prediction for platform {representative_instance.platform_info} "
                    f"(covers {len(platform_instances)} instances)"
                )

                # Call predictor via HTTP API with retry logic
                data = await self._make_request_with_retry(
                    json_data=request_data,
                )

                # Parse response based on prediction type
                if prediction_type == "expect_error":
                    predicted_time = data["result"]["expected_runtime_ms"]
                    error_margin = data["result"]["error_margin_ms"]

                    logger.info(
                        f"Prediction (expect_error) for {representative_instance.platform_info['hardware_name']}: "
                        f"expected_runtime={predicted_time:.2f}ms, error_margin={error_margin:.2f}ms "
                        f"({len(platform_instances)} instances)"
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
                    median = quantiles.get(0.5, next(iter(quantiles.values())))

                    # Format quantiles as "q=value" pairs
                    quantile_str = ", ".join(
                        [f"{k}={v:.2f}ms" for k, v in sorted(quantiles.items())]
                    )
                    logger.info(
                        f"Prediction (quantile) for {representative_instance.platform_info['hardware_name']}: "
                        f"quantiles={{{quantile_str}}} "
                        f"({len(platform_instances)} instances)"
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
                    raise ValueError(
                        f"Unknown prediction type: {prediction_type}"
                    )

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
                    ) from e
                elif (
                    "Invalid features" in error_msg
                    or "Invalid request" in error_msg
                ):
                    logger.error(
                        f"Invalid features for prediction: {error_msg}"
                    )
                    raise ValueError(f"Invalid task metadata: {error_msg}") from e
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
        """Check if predictor service is healthy by making HTTP GET request to /health endpoint.

        Returns:
            True if service is healthy and HTTP endpoint is accessible, False otherwise
        """
        async with self._client_lock:
            try:
                # Ensure we have an active client
                client = await self._ensure_client()

                # Make HTTP GET request to health endpoint
                response = await client.get("/health", timeout=self.timeout)

                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get("status", "unhealthy")
                    return status == "healthy"
                else:
                    logger.debug(
                        f"Health check returned HTTP {response.status_code}"
                    )
                    return False
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
                await self._close_client()
                return False
