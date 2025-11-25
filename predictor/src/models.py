"""
Pydantic data models for the predictor service.

Defines all request/response models for API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class PlatformInfo(BaseModel):
    """Platform information for model identification."""
    software_name: str = Field(..., description="Name of the software platform")
    software_version: str = Field(..., description="Version of the software")
    hardware_name: str = Field(..., description="Name of the hardware platform")

    def extract_gpu_specs(self) -> Optional[Dict[str, Any]]:
        """
        Extract GPU name from hardware_name and return corresponding specifications.

        This method attempts to identify a Tesla series GPU name within the hardware_name
        field and returns the detailed specifications from the hardware performance database.

        Returns:
            Dictionary containing GPU specifications if a match is found, None otherwise.
            The dictionary includes: cuda_cores, tensor_cores, fp32_tflops, fp16_tflops,
            tensor_tflops, memory_gb, and memory_bandwidth_gb_s.

        Examples:
            >>> platform = PlatformInfo(
            ...     software_name="PyTorch",
            ...     software_version="2.0",
            ...     hardware_name="NVIDIA Tesla V100-PCIE-16GB"
            ... )
            >>> specs = platform.extract_gpu_specs()
            >>> specs['cuda_cores']
            5120

            >>> platform = PlatformInfo(
            ...     software_name="PyTorch",
            ...     software_version="2.0",
            ...     hardware_name="NVIDIA H100 80GB HBM3"
            ... )
            >>> specs = platform.extract_gpu_specs()
            >>> specs['memory_gb']
            80
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
    """Request model for training endpoint."""
    model_id: str = Field(..., description="Unique identifier for the model")
    platform_info: PlatformInfo = Field(..., description="Platform information")
    prediction_type: str = Field(..., description="Type of prediction: 'expect_error' or 'quantile'")
    features_list: List[Dict[str, Any]] = Field(..., description="List of training samples with features")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Optional training configuration")
    enable_preprocessors: Optional[List[str]] = Field(None, description="List of preprocessors to enable")
    preprocessor_mappings: Optional[Dict[str, List[str]]] = Field(None, description="Specific which feature need to be preprocessed by which preprocessor")


    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {'expect_error', 'quantile', 'linear_regression', 'decision_tree'}
        if v not in allowed_types:
            raise ValueError(f"prediction_type must be one of {allowed_types}, got '{v}'")
        return v

    @field_validator('features_list')
    @classmethod
    def validate_features_list(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that all samples have runtime_ms field."""
        if not v:
            raise ValueError("features_list cannot be empty")

        for idx, sample in enumerate(v):
            if 'runtime_ms' not in sample:
                raise ValueError(f"Sample at index {idx} missing required field 'runtime_ms'")

            if not isinstance(sample['runtime_ms'], (int, float)):
                raise ValueError(f"Sample at index {idx}: 'runtime_ms' must be numeric")

        return v


class TrainingResponse(BaseModel):
    """Response model for training endpoint."""
    status: str = Field(..., description="Status of training: 'success' or 'error'")
    message: str = Field(..., description="Detailed message about the training result")
    model_key: str = Field(..., description="Unique key for the trained model")
    samples_trained: int = Field(..., description="Number of samples used for training")


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    model_id: str = Field(..., description="Unique identifier for the model")
    platform_info: PlatformInfo = Field(..., description="Platform information")
    prediction_type: str = Field(..., description="Type of prediction: 'expect_error' or 'quantile'")
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    quantiles: Optional[List[float]] = Field(None, description="Custom quantiles for prediction (only used in experiment mode)")
    enable_preprocessors: Optional[List[str]] = Field(None, description="List of preprocessors to enable")
    preprocessor_mappings: Optional[Dict[str, List[str]]] = Field(None, description="Specific which feature need to be preprocessed by which preprocessor")


    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        """Validate that prediction_type is one of the allowed values."""
        allowed_types = {'expect_error', 'quantile', 'linear_regression', 'decision_tree'}
        if v not in allowed_types:
            raise ValueError(f"prediction_type must be one of {allowed_types}, got '{v}'")
        return v

    @field_validator('quantiles')
    @classmethod
    def validate_quantiles(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate that quantiles are between 0 and 1."""
        if v is not None:
            for q in v:
                if not isinstance(q, (int, float)) or not (0 < q < 1):
                    raise ValueError(f"All quantiles must be between 0 and 1, got {q}")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    model_id: str = Field(..., description="Model identifier used for prediction")
    platform_info: PlatformInfo = Field(..., description="Platform information")
    prediction_type: str = Field(..., description="Type of prediction used")
    result: Dict[str, Any] = Field(..., description="Prediction result (format varies by prediction_type)")


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    model_id: str = Field(..., description="Model identifier")
    platform_info: PlatformInfo = Field(..., description="Platform information")
    prediction_type: str = Field(..., description="Type of prediction")
    samples_count: int = Field(..., description="Number of training samples")
    last_trained: str = Field(..., description="ISO 8601 timestamp of last training")


class ModelListResponse(BaseModel):
    """Response model for list endpoint."""
    models: List[ModelMetadata] = Field(..., description="List of all trained models")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error category")
    message: str = Field(..., description="Detailed error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    reason: Optional[str] = Field(None, description="Reason if unhealthy")
