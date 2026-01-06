"""Pydantic data models for the predictor service.

Defines all request/response models for API endpoints and library API.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


# =============================================================================
# Enums
# =============================================================================


class PredictionType(str, Enum):
    """Supported prediction types."""

    EXPECT_ERROR = "expect_error"
    QUANTILE = "quantile"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"


class PlatformInfo(BaseModel):
    """Platform information for model identification.

    Attributes:
        software_name: Name of the software platform.
        software_version: Version of the software.
        hardware_name: Name of the hardware platform.
    """

    software_name: str = Field(
        ...,
        description="Name of the software platform",
    )
    software_version: str = Field(
        ...,
        description="Version of the software",
    )
    hardware_name: str = Field(
        ...,
        description="Name of the hardware platform",
    )

    def extract_gpu_specs(self) -> dict[str, Any] | None:
        """Extract GPU specs from hardware_name.

        This method attempts to identify a Tesla series GPU name within the
        hardware_name field and returns the detailed specifications from the
        hardware performance database.

        Returns:
            Dictionary containing GPU specifications if a match is found,
            None otherwise. The dictionary includes: cuda_cores, tensor_cores,
            fp32_tflops, fp16_tflops, tensor_tflops, memory_gb, and
            memory_bandwidth_gb_s.

        Examples:
            >>> platform = PlatformInfo(
            ...     software_name="PyTorch",
            ...     software_version="2.0",
            ...     hardware_name="NVIDIA Tesla V100-PCIE-16GB"
            ... )
            >>> specs = platform.extract_gpu_specs()
            >>> specs['cuda_cores']
            5120
        """
        from src.utils.hardware_perf_info import NVIDIA_TESLA_SPECS

        # Normalize hardware_name for matching
        hardware_name_upper = self.hardware_name.upper()

        # Define GPU model patterns in priority order (more specific first)
        # This ensures we match longer model names before shorter ones
        gpu_patterns = [
            # H20/H200 variants (check specific variants first)
            (r'H20', 'H20'),
            # H100 variants (check specific variants first)
            (r'H100[- ]?PCIE', 'H100-PCIe'),
            (r'H100[- ]?94GB', 'H100-94GB'),
            (r'H100', 'H100'),
            # A100 variants
            (r'A100[- ]?80GB', 'A100-80GB'),
            (r'A100', 'A100'),
            # V100 variants
            (r'V100[- ]?32GB', 'V100-32GB'),
            (r'V100', 'V100'),
            # Other A-series
            (r'A40', 'A40'),
            (r'A30', 'A30'),
            (r'A10', 'A10'),
            # T-series
            (r'T4', 'T4'),
        ]

        # Try to match each pattern
        for pattern, gpu_key in gpu_patterns:
            if re.search(pattern, hardware_name_upper):
                if gpu_key in NVIDIA_TESLA_SPECS:
                    return NVIDIA_TESLA_SPECS[gpu_key].copy()

        # No match found
        return None


class TrainingRequest(BaseModel):
    """Request model for training endpoint.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features_list: List of training samples with features.
        training_config: Optional training configuration.
        preprocess_config: Per-feature preprocessor chains (recommended).
        enable_preprocessors: [DEPRECATED] List of preprocessors to enable.
        preprocessor_mappings: [DEPRECATED] Feature to preprocessor mappings.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error' or 'quantile'",
    )
    features_list: list[dict[str, Any]] = Field(
        ...,
        description="List of training samples with features",
    )
    training_config: dict[str, Any] | None = Field(
        None,
        description=(
            "Optional training configuration. Supported options: "
            "epochs, learning_rate, hidden_layers, quantiles, "
            "data_augmentation, log_transform, residual_calibration"
        ),
    )
    preprocess_config: dict[str, list[str]] | None = Field(
        None,
        description=(
            "Per-feature preprocessor chains. Format: "
            '{"feature_name": ["preprocessor_0", "preprocessor_1", ...]}. '
            "Each feature is processed by its chain in order."
        ),
    )
    # DEPRECATED: Use preprocess_config instead
    enable_preprocessors: list[str] | None = Field(
        None,
        description="[DEPRECATED] Use preprocess_config instead",
        json_schema_extra={"deprecated": True},
    )
    preprocessor_mappings: dict[str, list[str]] | None = Field(
        None,
        description="[DEPRECATED] Use preprocess_config instead",
        json_schema_extra={"deprecated": True},
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v

    @field_validator('features_list')
    @classmethod
    def validate_features_list(
        cls,
        v: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate that all samples have runtime_ms field."""
        if not v:
            raise ValueError("features_list cannot be empty")

        for idx, sample in enumerate(v):
            if 'runtime_ms' not in sample:
                raise ValueError(
                    f"Sample at index {idx} missing required field 'runtime_ms'"
                )

            if not isinstance(sample['runtime_ms'], (int, float)):
                raise ValueError(
                    f"Sample at index {idx}: 'runtime_ms' must be numeric"
                )

        return v


class TrainingResponse(BaseModel):
    """Response model for training endpoint.

    Attributes:
        status: Status of training ('success' or 'error').
        message: Detailed message about the training result.
        model_key: Unique key for the trained model.
        samples_trained: Number of samples used for training.
        version: Unix timestamp version of the saved model.
        version_iso: ISO 8601 formatted version timestamp.
    """

    status: str = Field(
        ...,
        description="Status of training: 'success' or 'error'",
    )
    message: str = Field(
        ...,
        description="Detailed message about the training result",
    )
    model_key: str = Field(
        ...,
        description="Unique key for the trained model",
    )
    samples_trained: int = Field(
        ...,
        description="Number of samples used for training",
    )
    version: int | None = Field(
        None,
        description="Unix timestamp version of the saved model",
    )
    version_iso: str | None = Field(
        None,
        description="ISO 8601 formatted version timestamp",
    )


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features: Feature values for prediction.
        quantiles: Custom quantiles for prediction (experiment mode only).
        preprocess_config: Per-feature preprocessor chains (recommended).
        enable_preprocessors: [DEPRECATED] List of preprocessors to enable.
        preprocessor_mappings: [DEPRECATED] Feature to preprocessor mappings.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error' or 'quantile'",
    )
    features: dict[str, Any] = Field(
        ...,
        description="Feature values for prediction",
    )
    quantiles: list[float] | None = Field(
        None,
        description="Custom quantiles for prediction (experiment mode only)",
    )
    preprocess_config: dict[str, list[str]] | None = Field(
        None,
        description=(
            "Per-feature preprocessor chains. Format: "
            '{"feature_name": ["preprocessor_0", "preprocessor_1", ...]}. '
            "Each feature is processed by its chain in order."
        ),
    )
    # DEPRECATED: Use preprocess_config instead
    enable_preprocessors: list[str] | None = Field(
        None,
        description="[DEPRECATED] Use preprocess_config instead",
        json_schema_extra={"deprecated": True},
    )
    preprocessor_mappings: dict[str, list[str]] | None = Field(
        None,
        description="[DEPRECATED] Use preprocess_config instead",
        json_schema_extra={"deprecated": True},
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v

    @field_validator('quantiles')
    @classmethod
    def validate_quantiles(
        cls,
        v: list[float] | None,
    ) -> list[float] | None:
        """Validate that quantiles are between 0 and 1."""
        if v is not None:
            for q in v:
                if not isinstance(q, (int, float)) or not (0 < q < 1):
                    raise ValueError(
                        f"All quantiles must be between 0 and 1, got {q}"
                    )
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint.

    Attributes:
        model_id: Model identifier used for prediction.
        platform_info: Platform information.
        prediction_type: Type of prediction used.
        result: Prediction result (format varies by prediction_type).
    """

    model_id: str = Field(
        ...,
        description="Model identifier used for prediction",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction used",
    )
    result: dict[str, Any] = Field(
        ...,
        description="Prediction result (format varies by prediction_type)",
    )


class ModelMetadata(BaseModel):
    """Metadata for a trained model.

    Attributes:
        model_id: Model identifier.
        platform_info: Platform information.
        prediction_type: Type of prediction.
        samples_count: Number of training samples.
        last_trained: ISO 8601 timestamp of last training.
    """

    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction",
    )
    samples_count: int = Field(
        ...,
        description="Number of training samples",
    )
    last_trained: str = Field(
        ...,
        description="ISO 8601 timestamp of last training",
    )


class ModelListResponse(BaseModel):
    """Response model for list endpoint.

    Attributes:
        models: List of all trained models.
    """

    models: list[ModelMetadata] = Field(
        ...,
        description="List of all trained models",
    )


class ErrorResponse(BaseModel):
    """Standard error response model.

    Attributes:
        error: Error category.
        message: Detailed error message.
        details: Additional error context.
    """

    error: str = Field(
        ...,
        description="Error category",
    )
    message: str = Field(
        ...,
        description="Detailed error message",
    )
    details: dict[str, Any] | None = Field(
        None,
        description="Additional error context",
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Health status ('healthy' or 'unhealthy').
        reason: Reason if unhealthy.
    """

    status: str = Field(
        ...,
        description="Health status: 'healthy' or 'unhealthy'",
    )
    reason: str | None = Field(
        None,
        description="Reason if unhealthy",
    )


# =============================================================================
# Library API Models
# =============================================================================


class TrainingResult(BaseModel):
    """Result from a training operation in the library API.

    Attributes:
        success: Whether training completed successfully.
        model_id: Model identifier.
        platform_info: Platform information.
        prediction_type: Type of prediction model.
        samples_trained: Number of samples used for training.
        training_metadata: Additional training metadata.
        message: Human-readable result message.
    """

    success: bool = Field(
        ...,
        description="Whether training completed successfully",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction model",
    )
    samples_trained: int = Field(
        ...,
        description="Number of samples used for training",
    )
    training_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional training metadata",
    )
    message: str = Field(
        ...,
        description="Human-readable result message",
    )


class PredictionResult(BaseModel):
    """Result from a prediction operation in the library API.

    Attributes:
        model_id: Model identifier used for prediction.
        platform_info: Platform information.
        prediction_type: Type of prediction used.
        result: Prediction result (format varies by prediction_type).
    """

    model_id: str = Field(
        ...,
        description="Model identifier used for prediction",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction used",
    )
    result: dict[str, Any] = Field(
        ...,
        description="Prediction result (format varies by prediction_type)",
    )


class ModelInfo(BaseModel):
    """Detailed information about a stored model.

    Attributes:
        model_id: Model identifier.
        platform_info: Platform information.
        prediction_type: Type of prediction.
        samples_count: Number of training samples.
        last_trained: ISO 8601 timestamp of last training.
        feature_names: List of feature names the model expects.
    """

    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction",
    )
    samples_count: int = Field(
        ...,
        description="Number of training samples",
    )
    last_trained: str = Field(
        ...,
        description="ISO 8601 timestamp of last training",
    )
    feature_names: list[str] | None = Field(
        None,
        description="List of feature names the model expects",
    )


class CollectedSample(BaseModel):
    """A single sample collected for training via the accumulator pattern.

    Attributes:
        features: Feature dictionary.
        runtime_ms: Measured runtime in milliseconds.
        collected_at: ISO 8601 timestamp when sample was collected.
    """

    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary",
    )
    runtime_ms: float = Field(
        ...,
        description="Measured runtime in milliseconds",
    )
    collected_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp when sample was collected",
    )


# =============================================================================
# HTTP API Request/Response Models (New Endpoints)
# =============================================================================


class CollectRequest(BaseModel):
    """Request model for /collect endpoint.

    Use this to collect individual training samples for later batch training.
    Samples are accumulated until /train is called.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features: Feature dictionary for this sample.
        runtime_ms: Measured runtime in milliseconds.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error' or 'quantile'",
    )
    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary for this sample",
    )
    runtime_ms: float = Field(
        ...,
        description="Measured runtime in milliseconds",
        gt=0,
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v


class CollectResponse(BaseModel):
    """Response model for /collect endpoint.

    Attributes:
        status: Status of the operation ('success' or 'error').
        samples_collected: Total number of samples collected for this model.
        message: Optional message with additional details.
    """

    status: str = Field(
        ...,
        description="Status of the operation: 'success' or 'error'",
    )
    samples_collected: int = Field(
        ...,
        description="Total number of samples collected for this model",
    )
    message: str | None = Field(
        None,
        description="Optional message with additional details",
    )


# =============================================================================
# V2 API Models (Preprocessing Chain Support)
# =============================================================================


class PreprocessorStepConfigV2(BaseModel):
    """Configuration for a single preprocessor in a V2 chain.

    Attributes:
        name: Name of the preprocessor in the registry (e.g., "multiply", "remove").
        params: Parameters to pass to the preprocessor factory.
    """

    name: str = Field(
        ...,
        description="Registered preprocessor name: 'multiply', 'remove', 'token_length'",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Preprocessor-specific parameters",
    )


class ChainConfigV2(BaseModel):
    """Configuration for a V2 preprocessing chain.

    The chain is executed in order: steps[0] runs first, then steps[1], etc.

    Attributes:
        steps: Ordered list of preprocessor configurations.
    """

    steps: list[PreprocessorStepConfigV2] = Field(
        default_factory=list,
        description="Ordered list of preprocessor steps",
    )


class CollectRequestV2(BaseModel):
    """Request model for /v2/collect endpoint.

    Collects a training sample with optional preprocessing.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features: Feature dictionary for this sample.
        runtime_ms: Measured runtime in milliseconds.
        preprocess_chain: Optional chain to preprocess before storing.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error', 'quantile', etc.",
    )
    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary for this sample",
    )
    runtime_ms: float = Field(
        ...,
        description="Measured runtime in milliseconds",
        gt=0,
    )
    preprocess_chain: ChainConfigV2 | None = Field(
        None,
        description="Optional chain to preprocess features before storing",
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v


class CollectResponseV2(BaseModel):
    """Response model for /v2/collect endpoint.

    Attributes:
        status: Status of the operation ('success' or 'error').
        samples_collected: Total number of samples collected for this model.
        message: Optional message with additional details.
    """

    status: str = Field(
        ...,
        description="Status of the operation: 'success' or 'error'",
    )
    samples_collected: int = Field(
        ...,
        description="Total number of samples collected for this model",
    )
    message: str | None = Field(
        None,
        description="Optional message with additional details",
    )


class TrainingRequestV2(BaseModel):
    """Request model for /v2/train endpoint.

    Train a model with an optional preprocessing chain. The chain is stored
    with the model and automatically used during prediction.

    Chain Resolution:
    - Chain provided → use provided chain
    - No chain + existing model → use model's stored chain
    - No chain + new model → no preprocessing

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features_list: Optional list of training samples (combined with collected).
        training_config: Optional training configuration.
        preprocess_chain: Optional preprocessing chain to use and store.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error', 'quantile', etc.",
    )
    features_list: list[dict[str, Any]] | None = Field(
        None,
        description="Optional list of training samples to combine with collected data",
    )
    training_config: dict[str, Any] | None = Field(
        None,
        description="Optional training configuration",
    )
    preprocess_chain: ChainConfigV2 | None = Field(
        None,
        description="Optional preprocessing chain to use and store with model",
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v

    @field_validator('features_list')
    @classmethod
    def validate_features_list(
        cls,
        v: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Validate that all samples have runtime_ms field if provided."""
        if v is None:
            return v

        for idx, sample in enumerate(v):
            if 'runtime_ms' not in sample:
                raise ValueError(
                    f"Sample at index {idx} missing required field 'runtime_ms'"
                )

            if not isinstance(sample['runtime_ms'], (int, float)):
                raise ValueError(
                    f"Sample at index {idx}: 'runtime_ms' must be numeric"
                )

        return v


class TrainingResponseV2(BaseModel):
    """Response model for /v2/train endpoint.

    Attributes:
        status: Status of training ('success' or 'error').
        message: Detailed message about the training result.
        model_key: Unique key for the trained model.
        samples_trained: Number of samples used for training.
        chain_stored: Whether a preprocessing chain was stored with the model.
    """

    status: str = Field(
        ...,
        description="Status of training: 'success' or 'error'",
    )
    message: str = Field(
        ...,
        description="Detailed message about the training result",
    )
    model_key: str = Field(
        ...,
        description="Unique key for the trained model",
    )
    samples_trained: int = Field(
        ...,
        description="Number of samples used for training",
    )
    chain_stored: bool = Field(
        False,
        description="Whether a preprocessing chain was stored with the model",
    )


class PredictionRequestV2(BaseModel):
    """Request model for /v2/predict endpoint.

    NOTE: This request does NOT accept a preprocess_chain. The V2 API
    automatically uses the chain stored with the model during training.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction to use.
        features: Feature values for prediction.
        quantiles: Custom quantiles for prediction (quantile predictor only).
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error', 'quantile', etc.",
    )
    features: dict[str, Any] = Field(
        ...,
        description="Feature values for prediction",
    )
    quantiles: list[float] | None = Field(
        None,
        description="Custom quantiles for prediction (quantile predictor only)",
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v

    @field_validator('quantiles')
    @classmethod
    def validate_quantiles(
        cls,
        v: list[float] | None,
    ) -> list[float] | None:
        """Validate that quantiles are between 0 and 1."""
        if v is not None:
            for q in v:
                if not isinstance(q, (int, float)) or not (0 < q < 1):
                    raise ValueError(
                        f"All quantiles must be between 0 and 1, got {q}"
                    )
        return v


class PredictionResponseV2(BaseModel):
    """Response model for /v2/predict endpoint.

    Attributes:
        model_id: Model identifier used for prediction.
        platform_info: Platform information.
        prediction_type: Type of prediction used.
        result: Prediction result (format varies by prediction_type).
        chain_applied: Whether preprocessing chain was applied.
    """

    model_id: str = Field(
        ...,
        description="Model identifier used for prediction",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction used",
    )
    result: dict[str, Any] = Field(
        ...,
        description="Prediction result (format varies by prediction_type)",
    )
    chain_applied: bool = Field(
        False,
        description="Whether preprocessing chain was applied",
    )


class ChainValidationErrorV2(BaseModel):
    """Error details when chain validation fails.

    Attributes:
        step_index: Index of the step that failed.
        preprocessor_name: Name of the preprocessor that failed.
        error: Detailed error message.
    """

    step_index: int = Field(
        ...,
        description="Index of the step that failed (0-based)",
    )
    preprocessor_name: str = Field(
        ...,
        description="Name of the preprocessor that failed",
    )
    error: str = Field(
        ...,
        description="Detailed error message",
    )


# =============================================================================
# Version Management Models
# =============================================================================


class VersionCheckRequest(BaseModel):
    """Request model for /version/check endpoint.

    Use this to check model version information without loading the model
    or making predictions.

    Attributes:
        model_id: Unique identifier for the model.
        platform_info: Platform information.
        prediction_type: Type of prediction.
    """

    model_id: str = Field(
        ...,
        description="Unique identifier for the model",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'expect_error', 'quantile', etc.",
    )

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {
            'expect_error',
            'quantile',
            'linear_regression',
            'decision_tree',
        }
        if v not in allowed_types:
            raise ValueError(
                f"prediction_type must be one of {allowed_types}, got '{v}'"
            )
        return v


class VersionCheckResponse(BaseModel):
    """Response model for /version/check endpoint.

    Provides comprehensive version information for a model configuration.

    Attributes:
        model_id: Model identifier.
        platform_info: Platform information.
        prediction_type: Type of prediction.
        exists: Whether any version of this model exists.
        latest_version: Unix timestamp of latest version (None if no versions).
        latest_version_iso: ISO 8601 timestamp of latest version.
        available_versions: All available version timestamps (descending order).
        version_count: Total number of available versions.
    """

    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    platform_info: PlatformInfo = Field(
        ...,
        description="Platform information",
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction",
    )
    exists: bool = Field(
        ...,
        description="Whether any version of this model exists",
    )
    latest_version: int | None = Field(
        None,
        description="Unix timestamp of latest version (None if no versions)",
    )
    latest_version_iso: str | None = Field(
        None,
        description="ISO 8601 timestamp of latest version",
    )
    available_versions: list[int] = Field(
        default_factory=list,
        description="All available version timestamps (descending order)",
    )
    version_count: int = Field(
        0,
        description="Total number of available versions",
    )
