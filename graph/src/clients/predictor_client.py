"""
Asynchronous client for the Predictor service.

Provides low-level API methods for interacting with the predictor, including:
- Runtime predictions (quantile and expect-error types)
- Model training with execution samples
- Model listing and management
- Health checking

Features:
- Connection pooling and keep-alive
- Automatic retries with exponential backoff
- Configurable timeouts
- Full type hints
- Dataclass-based response models

Example usage:
    ```python
    from graph.src.clients.predictor_client import PredictorClient, PlatformInfo

    async with PredictorClient("http://localhost:8001") as client:
        # Make a prediction
        platform = PlatformInfo(
            software_name="vllm",
            software_version="0.2.5",
            hardware_name="nvidia-a100"
        )

        result = await client.predict(
            model_id="gpt-4",
            platform_info=platform,
            features={"prompt_length": 100},
            prediction_type="quantile"
        )
        print(f"Expected runtime: {result.expected_runtime_ms}ms")

        # Train a model with new samples
        training_samples = [
            {"prompt_length": 50, "runtime_ms": 120.5},
            {"prompt_length": 100, "runtime_ms": 245.8},
        ]

        response = await client.train(
            model_id="gpt-4",
            platform_info=platform,
            prediction_type="quantile",
            features_list=training_samples
        )
        print(f"Trained with {response.samples_trained} samples")
    ```
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


# ==================== Data Models ====================


@dataclass
class PlatformInfo:
    """Platform environment specification for predictions.

    Describes the hardware and software environment where models execute.
    """

    software_name: str
    software_version: str
    hardware_name: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of platform info
        """
        return {
            "software_name": self.software_name,
            "software_version": self.software_version,
            "hardware_name": self.hardware_name,
        }


@dataclass
class PredictionResult:
    """Prediction result from predictor service.

    Contains predicted runtime and optional error margins or quantile distributions.
    """

    model_id: str
    platform_info: PlatformInfo
    prediction_type: str
    expected_runtime_ms: float
    error_margin_ms: Optional[float] = None
    quantiles: Optional[Dict[float, float]] = None


@dataclass
class TrainingResponse:
    """Response from model training operation.

    Provides status and metadata about the training process.
    """

    status: str
    message: str
    model_key: str
    samples_trained: int


@dataclass
class ModelInfo:
    """Information about a trained prediction model.

    Contains metadata about model configuration and training history.
    """

    model_id: str
    platform_info: PlatformInfo
    prediction_type: str
    samples_count: int
    last_trained: str


# ==================== Predictor Client ====================


class PredictorClient:
    """Asynchronous client for the Predictor service.

    Provides methods for runtime prediction and model training with automatic
    retry, connection pooling, and comprehensive error handling.
    """

    def __init__(
        self,
        predictor_url: str = "http://localhost:8001",
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize the predictor client.

        Args:
            predictor_url: Base URL of the predictor service (default: http://localhost:8001)
            timeout: Request timeout in seconds (default: 10.0)
            max_retries: Maximum retry attempts for transient failures (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 2.0)
        """
        self.predictor_url = predictor_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # HTTP client will be initialized in __aenter__
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "PredictorClient":
        """Enter the async context manager - initialize HTTP client."""
        self._http_client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager - cleanup HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.

        Returns:
            The HTTP client instance

        Raises:
            RuntimeError: If client is not initialized (use async with PredictorClient(...))
        """
        if self._http_client is None:
            raise RuntimeError("PredictorClient must be used as an async context manager (async with)")
        return self._http_client

    # ==================== Prediction API ====================

    async def predict(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        features: Dict[str, Any],
        prediction_type: str = "quantile",
    ) -> PredictionResult:
        """Get prediction for task execution time.

        Queries the predictor for estimated runtime based on task features and platform.
        Supports two prediction types:
        - "quantile": Returns full quantile distribution (percentiles)
        - "expect_error": Returns expected value with error margin

        Args:
            model_id: Unique identifier of the model/tool (e.g., "gpt-4", "llama-2-70b")
            platform_info: Platform specification (software/hardware environment)
            features: Feature dictionary for prediction (model-specific, e.g., {"prompt_length": 100})
            prediction_type: Type of prediction - "quantile" or "expect_error" (default: "quantile")

        Returns:
            PredictionResult with:
                - expected_runtime_ms: Predicted runtime in milliseconds
                - error_margin_ms: Error margin (for "expect_error" type)
                - quantiles: Quantile distribution (for "quantile" type)

        Raises:
            ValueError: If prediction_type is invalid or response is malformed
            ConnectionError: If prediction fails after all retries

        Example:
            ```python
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")
            result = await client.predict(
                model_id="gpt-4",
                platform_info=platform,
                features={"prompt_length": 150},
                prediction_type="quantile"
            )
            print(f"P50: {result.quantiles[0.5]}ms")
            print(f"P90: {result.quantiles[0.9]}ms")
            ```
        """
        if prediction_type not in ("expect_error", "quantile"):
            raise ValueError(f"Invalid prediction_type: {prediction_type}. Must be 'expect_error' or 'quantile'")

        request_data = {
            "model_id": model_id,
            "platform_info": platform_info.to_dict(),
            "features": features,
            "prediction_type": prediction_type,
        }

        client = self._ensure_client()

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.predictor_url}/predict",
                    json=request_data,
                )
                response.raise_for_status()
                data = response.json()

                # Parse response based on prediction_type
                if prediction_type == "expect_error":
                    return PredictionResult(
                        model_id=data["model_id"],
                        platform_info=PlatformInfo(**data["platform_info"]),
                        prediction_type=data["prediction_type"],
                        expected_runtime_ms=data["result"]["expected_runtime_ms"],
                        error_margin_ms=data["result"]["error_margin_ms"],
                    )
                elif prediction_type == "quantile":
                    # Convert quantile keys from string to float
                    quantiles_dict = {float(k): v for k, v in data["result"]["quantiles"].items()}
                    # Use median (0.5 quantile) as expected runtime, or first value if no 0.5
                    median = quantiles_dict.get(0.5, list(quantiles_dict.values())[0])
                    return PredictionResult(
                        model_id=data["model_id"],
                        platform_info=PlatformInfo(**data["platform_info"]),
                        prediction_type=data["prediction_type"],
                        expected_runtime_ms=median,
                        quantiles=quantiles_dict,
                    )

            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client errors like validation failures)
                if 400 <= e.response.status_code < 500:
                    raise ValueError(f"Prediction request failed: {e.response.text}") from e
                last_exception = e

            except httpx.HTTPError as e:
                last_exception = e

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                await asyncio.sleep(delay)
            else:
                raise ConnectionError(
                    f"Prediction failed after {self.max_retries} retries: {last_exception}"
                ) from last_exception

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    # ==================== Training API ====================

    async def train(
        self,
        model_id: str,
        platform_info: PlatformInfo,
        prediction_type: str,
        features_list: List[Dict[str, Any]],
        training_config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResponse:
        """Train or update a prediction model with new execution samples.

        Trains the predictor on historical execution data to improve future predictions.
        Can be called incrementally to update models with new samples.

        Args:
            model_id: Unique identifier for the model (e.g., "gpt-4")
            platform_info: Platform specification (software/hardware environment)
            prediction_type: "expect_error" or "quantile"
            features_list: List of training samples, each must include:
                - Feature fields (model-specific, e.g., "prompt_length")
                - "runtime_ms": Actual execution time in milliseconds
            training_config: Optional training configuration (e.g., {"quantiles": [0.5, 0.9, 0.99]})

        Returns:
            TrainingResponse with:
                - status: Training status ("success" or "failed")
                - message: Detailed message
                - model_key: Unique key identifying the trained model
                - samples_trained: Number of samples processed

        Raises:
            ValueError: If training data is invalid or insufficient
            ConnectionError: If training fails after all retries

        Example:
            ```python
            platform = PlatformInfo("vllm", "0.2.5", "nvidia-a100")
            samples = [
                {"prompt_length": 50, "runtime_ms": 120.5},
                {"prompt_length": 100, "runtime_ms": 245.8},
                {"prompt_length": 200, "runtime_ms": 512.3},
            ]

            response = await client.train(
                model_id="gpt-4",
                platform_info=platform,
                prediction_type="quantile",
                features_list=samples,
                training_config={"quantiles": [0.5, 0.9, 0.95, 0.99]}
            )
            print(f"Training status: {response.status}")
            print(f"Samples trained: {response.samples_trained}")
            ```
        """
        request_data = {
            "model_id": model_id,
            "platform_info": platform_info.to_dict(),
            "prediction_type": prediction_type,
            "features_list": features_list,
        }

        if training_config is not None:
            request_data["training_config"] = training_config

        client = self._ensure_client()

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.predictor_url}/train",
                    json=request_data,
                )
                response.raise_for_status()
                data = response.json()

                return TrainingResponse(
                    status=data["status"],
                    message=data["message"],
                    model_key=data["model_key"],
                    samples_trained=data["samples_trained"],
                )

            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors
                if 400 <= e.response.status_code < 500:
                    raise ValueError(f"Training request failed: {e.response.text}") from e
                last_exception = e

            except httpx.HTTPError as e:
                last_exception = e

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                await asyncio.sleep(delay)
            else:
                raise ConnectionError(
                    f"Training failed after {self.max_retries} retries: {last_exception}"
                ) from last_exception

        if last_exception:
            raise last_exception

    # ==================== Model Management API ====================

    async def list_models(self) -> List[ModelInfo]:
        """List all available trained prediction models.

        Returns metadata for all models that have been trained, including sample counts
        and last training timestamp.

        Returns:
            List of ModelInfo objects containing:
                - model_id: Model identifier
                - platform_info: Platform configuration
                - prediction_type: Type of predictions ("quantile" or "expect_error")
                - samples_count: Number of training samples
                - last_trained: ISO 8601 timestamp of last training

        Raises:
            ConnectionError: If request fails after all retries

        Example:
            ```python
            models = await client.list_models()
            for model in models:
                print(f"{model.model_id} on {model.platform_info.hardware_name}")
                print(f"  Samples: {model.samples_count}")
                print(f"  Last trained: {model.last_trained}")
            ```
        """
        client = self._ensure_client()

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await client.get(f"{self.predictor_url}/list")
                response.raise_for_status()
                data = response.json()

                models = []
                for model_data in data["models"]:
                    models.append(
                        ModelInfo(
                            model_id=model_data["model_id"],
                            platform_info=PlatformInfo(**model_data["platform_info"]),
                            prediction_type=model_data["prediction_type"],
                            samples_count=model_data["samples_count"],
                            last_trained=model_data["last_trained"],
                        )
                    )
                return models

            except httpx.HTTPError as e:
                last_exception = e

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                await asyncio.sleep(delay)
            else:
                raise ConnectionError(
                    f"List models failed after {self.max_retries} retries: {last_exception}"
                ) from last_exception

        if last_exception:
            raise last_exception

    # ==================== Health Check API ====================

    async def health_check(self) -> bool:
        """Check if predictor service is healthy.

        Performs a lightweight health check to verify service availability.
        Uses a shorter timeout (5s) than regular requests.

        Returns:
            True if service is healthy and responding, False otherwise

        Example:
            ```python
            if await client.health_check():
                print("Predictor service is healthy")
            else:
                print("Predictor service is unavailable")
            ```
        """
        client = self._ensure_client()

        try:
            response = await client.get(
                f"{self.predictor_url}/health",
                timeout=5.0,  # Use shorter timeout for health checks
            )
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy"
        except Exception:
            return False

    # ==================== Legacy Methods ====================

    async def close(self) -> None:
        """Close HTTP client and cleanup resources.

        Note: When using async context manager, this is called automatically.
        Only use this method if you're not using the context manager pattern.
        """
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
