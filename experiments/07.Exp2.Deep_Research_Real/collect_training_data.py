#!/usr/bin/env python3
"""
Training Data Collection Script for Predictor Model

This script:
1. Reads LLM execution tasks from dataset.jsonl
2. Executes each task on the LLM service and collects execution times
3. Transforms metadata according to scheduler's transformation mechanism
4. Submits training data to the predictor service

Usage:
    # Train both quantile and expect_error models (default)
    python collect_training_data.py \
        --dataset data/dataset.jsonl \
        --instance-url http://localhost:8001 \
        --predictor-url http://localhost:8002 \
        --model-id llama-7b
    
    # Train only quantile model
    python collect_training_data.py \
        --dataset data/dataset.jsonl \
        --instance-url http://localhost:8001 \
        --predictor-url http://localhost:8002 \
        --model-id llama-7b \
        --prediction-types quantile
    
    # Train both model types explicitly
    python collect_training_data.py \
        --dataset data/dataset.jsonl \
        --instance-url http://localhost:8001 \
        --predictor-url http://localhost:8002 \
        --model-id llama-7b \
        --prediction-types expect_error quantile
"""

import argparse
import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import httpx
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Hardware Performance Information (copied from predictor/src/utils/hardware_perf_info.py)
# ============================================================================

NVIDIA_TESLA_SPECS = {
    "V100": {
        "cuda_cores": 5120,
        "tensor_cores": 640,
        "fp32_tflops": 15.7,
        "fp16_tflops": 31.4,
        "tensor_tflops": 125.0,
        "memory_gb": 16,
        "memory_bandwidth_gb_s": 900,
    },
    "V100-32GB": {
        "cuda_cores": 5120,
        "tensor_cores": 640,
        "fp32_tflops": 15.7,
        "fp16_tflops": 31.4,
        "tensor_tflops": 125.0,
        "memory_gb": 32,
        "memory_bandwidth_gb_s": 900,
    },
    "T4": {
        "cuda_cores": 2560,
        "tensor_cores": 320,
        "fp32_tflops": 8.1,
        "fp16_tflops": 65.0,
        "tensor_tflops": 130.0,
        "memory_gb": 16,
        "memory_bandwidth_gb_s": 300,
    },
    "A100": {
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "fp32_tflops": 19.5,
        "fp16_tflops": 78.0,
        "tensor_tflops": 312.0,
        "memory_gb": 40,
        "memory_bandwidth_gb_s": 1555,
    },
    "A100-80GB": {
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "fp32_tflops": 19.5,
        "fp16_tflops": 78.0,
        "tensor_tflops": 312.0,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 2039,
    },
    "A10": {
        "cuda_cores": 9216,
        "tensor_cores": 288,
        "fp32_tflops": 31.2,
        "fp16_tflops": 125.0,
        "tensor_tflops": 250.0,
        "memory_gb": 24,
        "memory_bandwidth_gb_s": 600,
    },
    "A30": {
        "cuda_cores": 3584,
        "tensor_cores": 224,
        "fp32_tflops": 10.3,
        "fp16_tflops": 82.0,
        "tensor_tflops": 165.0,
        "memory_gb": 24,
        "memory_bandwidth_gb_s": 933,
    },
    "A40": {
        "cuda_cores": 10752,
        "tensor_cores": 336,
        "fp32_tflops": 37.4,
        "fp16_tflops": 74.8,
        "tensor_tflops": 150.0,
        "memory_gb": 48,
        "memory_bandwidth_gb_s": 696,
    },
    "H100": {
        "cuda_cores": 14592,
        "tensor_cores": 456,
        "fp32_tflops": 51.0,
        "fp16_tflops": 204.0,
        "tensor_tflops": 989.0,
        "fp8_tensor_tflops": 1979.0,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 3350,
    },
    "H100-94GB": {
        "cuda_cores": 14592,
        "tensor_cores": 456,
        "fp32_tflops": 51.0,
        "fp16_tflops": 204.0,
        "tensor_tflops": 989.0,
        "fp8_tensor_tflops": 1979.0,
        "memory_gb": 94,
        "memory_bandwidth_gb_s": 3350,
    },
    "H100-PCIe": {
        "cuda_cores": 14592,
        "tensor_cores": 456,
        "fp32_tflops": 48.0,
        "fp16_tflops": 192.0,
        "tensor_tflops": 756.0,
        "fp8_tensor_tflops": 1513.0,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 2000,
    },
    "H20": {
        "cuda_cores": 17920,
        "tensor_cores": 560,
        "fp32_tflops": 63.0,
        "fp16_tflops": 252.0,
        "tensor_tflops": 1230.0,
        "fp8_tensor_tflops": 2460.0,
        "memory_gb": 96,
        "memory_bandwidth_gb_s": 4000,
    },
}


def extract_gpu_specs(hardware_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract GPU specifications from hardware name.

    Copied from predictor/src/models.py::PlatformInfo.extract_gpu_specs()

    Args:
        hardware_name: Name of the hardware platform (e.g., "NVIDIA H20", "Tesla V100")

    Returns:
        Dictionary containing GPU specifications if a match is found, None otherwise.
    """
    # Normalize hardware_name for matching
    hardware_name_upper = hardware_name.upper()

    # Define GPU model patterns in priority order (more specific first)
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


def estimate_token_length(text: Optional[str]) -> int:
    """
    Estimate token length from text.
    
    Uses a simple heuristic: approximately 4 characters per token for English text.
    This is a rough approximation commonly used for LLM token estimation.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple heuristic: ~4 characters per token
    return max(1, len(text) // 4)


async def get_platform_info_from_instance(instance_url: str) -> Optional[Dict[str, str]]:
    """
    Try to get platform information from instance service.
    
    Args:
        instance_url: Base URL of the instance service
        
    Returns:
        Dictionary with platform info (software_name, software_version, hardware_name)
        if found, None otherwise
    """
    try:
        client = httpx.AsyncClient(timeout=5.0)
        response = await client.get(f"{instance_url.rstrip('/')}/info")
        await client.aclose()
        
        if response.status_code == 200:
            data = response.json()
            # Get platform info from instance info
            instance_data = data.get("instance", {})
            
            hardware_name = instance_data.get("hardware_name")
            software_name = instance_data.get("software_name")
            software_version = instance_data.get("software_version")
            
            # Return if we have at least hardware_name
            if hardware_name:
                platform_info = {
                    "hardware_name": hardware_name
                }
                if software_name:
                    platform_info["software_name"] = software_name
                if software_version:
                    platform_info["software_version"] = software_version
                
                logger.info(f"Retrieved platform info from instance service: {platform_info}")
                return platform_info
    except Exception as e:
        logger.debug(f"Failed to get platform info from instance: {e}")
    
    return None


def get_gpu_name_from_system() -> str:
    """
    Get GPU name from system using nvidia-smi or pynvml.
    
    Tries multiple methods:
    1. pynvml (NVIDIA Management Library Python bindings)
    2. nvidia-smi command
    3. Falls back to 'CPU' if no GPU is found
    
    Returns:
        GPU name string (e.g., "NVIDIA H20", "NVIDIA GeForce RTX 3090") or "CPU"
    """
    # Method 1: Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        # Decode bytes to string if needed
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        logger.info(f"Detected GPU via pynvml: {name}")
        return name
    except (ImportError, Exception) as e:
        logger.debug(f"pynvml method failed: {e}")
    
    # Method 2: Try nvidia-smi command
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '-i', '0'],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        if result.stdout:
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first line (GPU0)
            if gpu_name:
                logger.info(f"Detected GPU via nvidia-smi: {gpu_name}")
                return gpu_name
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        logger.debug(f"nvidia-smi method failed: {e}")
    
    # No GPU found
    logger.warning("No GPU detected, falling back to 'CPU'")
    return "CPU"


async def detect_platform_info(instance_url: Optional[str] = None) -> Dict[str, str]:
    """
    Detect platform information, trying instance service first, then system detection.
    
    Args:
        instance_url: Optional instance service URL to query first
        
    Returns:
        Dictionary with platform info (software_name, software_version, hardware_name)
    """
    # Try instance service first if URL is provided
    if instance_url:
        platform_info = await get_platform_info_from_instance(instance_url)
        if platform_info and platform_info.get("hardware_name"):
            logger.info(f"Detected platform info from instance service: {platform_info}")
            # Fill in defaults if missing
            if "software_name" not in platform_info:
                platform_info["software_name"] = "sglang"  # Default
            if "software_version" not in platform_info:
                platform_info["software_version"] = "1.0.0"  # Default
            return platform_info
    
    # Fall back to system detection
    import platform
    return {
        "hardware_name": get_gpu_name_from_system(),
        "software_name": platform.system(),
        "software_version": platform.release()
    }


class LLMServiceClient:
    """Client for interacting with LLM service instance."""

    def __init__(self, instance_url: str, timeout: float = 300.0, instance_id: str = ""):
        """
        Initialize LLM service client.

        Args:
            instance_url: Base URL of the instance service
            timeout: Request timeout in seconds
            instance_id: Identifier for this instance (for logging)
        """
        self.instance_url = instance_url.rstrip('/')
        self.timeout = timeout
        self.instance_id = instance_id or instance_url
        self.client = httpx.AsyncClient(timeout=timeout)

    async def execute_task(
        self,
        sentence: str,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Execute a task on the LLM service and return execution time.

        Args:
            sentence: Input sentence for the LLM
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing:
            - execution_time_ms: Execution time in milliseconds
            - success: Whether the task succeeded
            - result: Task result (if successful)
            - error: Error message (if failed)
            - instance_id: Instance that handled this request
        """
        try:
            start_time = time.time()

            # Call LLM service inference endpoint
            response = await self.client.post(
                f"{self.instance_url}/inference",
                json={
                    "sentence": sentence,
                    "max_tokens": max_tokens
                }
            )

            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                result_data = response.json()
                return {
                    "execution_time_ms": execution_time_ms,
                    "success": True,
                    "result": result_data.get("result"),
                    "error": None,
                    "instance_id": self.instance_id
                }
            else:
                return {
                    "execution_time_ms": execution_time_ms,
                    "success": False,
                    "result": None,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "instance_id": self.instance_id
                }
        except Exception as e:
            logger.error(f"Error executing task on {self.instance_id}: {e}")
            return {
                "execution_time_ms": 0.0,
                "success": False,
                "result": None,
                "error": str(e),
                "instance_id": self.instance_id
            }

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MultiInstanceLLMClient:
    """
    Client for load-balanced parallel execution across multiple LLM instances.

    Uses round-robin load balancing to distribute tasks across instances.
    """

    def __init__(self, instance_configs: List[Dict[str, str]], timeout: float = 300.0):
        """
        Initialize multi-instance LLM client.

        Args:
            instance_configs: List of instance configurations, each with 'url' key
            timeout: Request timeout in seconds
        """
        self.clients = []
        for idx, config in enumerate(instance_configs):
            instance_id = f"instance-{idx}@{config['url']}"
            client = LLMServiceClient(
                instance_url=config['url'],
                timeout=timeout,
                instance_id=instance_id
            )
            self.clients.append({
                'client': client,
                'config': config,
                'instance_id': instance_id
            })

        self.current_idx = 0
        logger.info(f"Initialized {len(self.clients)} instance(s) for parallel execution")
        for client_info in self.clients:
            logger.info(f"  - {client_info['instance_id']}")

    def get_next_client(self) -> LLMServiceClient:
        """
        Get next client using round-robin load balancing.

        Returns:
            LLM service client
        """
        client_info = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        return client_info['client']

    def get_instance_config(self, instance_id: str) -> Dict[str, str]:
        """
        Get instance configuration by instance_id.

        Args:
            instance_id: Instance identifier

        Returns:
            Instance configuration dict
        """
        for client_info in self.clients:
            if client_info['instance_id'] == instance_id:
                return client_info['config']
        return {}

    async def close_all(self):
        """Close all HTTP clients."""
        for client_info in self.clients:
            await client_info['client'].close()


class PredictorClient:
    """Client for interacting with predictor service (training and prediction)."""

    def __init__(self, predictor_url: str, timeout: float = 600.0):
        """
        Initialize predictor client.

        Args:
            predictor_url: Base URL of the predictor service
            timeout: Request timeout in seconds
        """
        self.predictor_url = predictor_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def submit_training_data(
        self,
        model_id: str,
        platform_info: Dict[str, str],
        prediction_type: str,
        features_list: List[Dict[str, Any]],
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submit training data to predictor service.
        
        Request format matches TrainingRequest model from predictor/src/models.py:
        {
            "model_id": str,                    # Required: Unique identifier for the model
            "platform_info": {                  # Required: PlatformInfo object
                "software_name": str,           # Required: Name of software platform (e.g., "sglang")
                "software_version": str,        # Required: Version of software (e.g., "1.0.0")
                "hardware_name": str            # Required: Name of hardware platform (e.g., "NVIDIA H20")
            },
            "prediction_type": str,             # Required: "expect_error" or "quantile"
            "features_list": [                  # Required: List of training samples (min 10)
                {
                    # User metadata features
                    "token_length": float,      # Required: Estimated input token length
                    "max_tokens": float,        # Required: Maximum output tokens
                    
                    # Hardware specs (automatically added by extract_gpu_specs)
                    "cuda_cores": float,        # Optional: CUDA cores count
                    "tensor_cores": float,      # Optional: Tensor cores count
                    "fp32_tflops": float,       # Optional: FP32 TFLOPS
                    "fp16_tflops": float,       # Optional: FP16 TFLOPS
                    "tensor_tflops": float,     # Optional: Tensor core TFLOPS (FP16)
                    "fp8_tensor_tflops": float, # Optional: FP8 tensor core TFLOPS
                    "memory_gb": float,         # Optional: GPU memory in GB
                    "memory_bandwidth_gb_s": float,  # Optional: Memory bandwidth in GB/s
                    
                    # Target variable
                    "runtime_ms": float         # Required: Actual execution time in milliseconds
                },
                ...
            ],
            "training_config": {                # Optional: Training configuration
                ...
            }
        }
        
        Response format matches TrainingResponse model:
        {
            "status": str,                      # "success" or "error"
            "message": str,                     # Detailed message about training result
            "model_key": str,                   # Unique key for the trained model
            "samples_trained": int              # Number of samples used for training
        }
        
        Args:
            model_id: Model identifier
            platform_info: Platform information dict with keys:
                          - software_name (str): Software platform name
                          - software_version (str): Software platform version
                          - hardware_name (str): Hardware platform name (GPU name)
            prediction_type: Type of prediction ('expect_error' or 'quantile')
            features_list: List of training samples, each containing:
                          - All feature values (token_length, max_tokens, hardware specs, etc.)
                          - runtime_ms (float): Actual execution time in milliseconds
            training_config: Optional training configuration dict
            
        Returns:
            Response from predictor service (TrainingResponse format)
            
        Raises:
            Exception: If training submission fails (HTTP error or validation error)
        """
        # Build request data according to TrainingRequest model
        # This format must match exactly with predictor/src/models.py::TrainingRequest
        request_data = {
            "model_id": model_id,
            "platform_info": {
                "software_name": platform_info["software_name"],
                "software_version": platform_info["software_version"],
                "hardware_name": platform_info["hardware_name"]
            },
            "prediction_type": prediction_type,
            "features_list": features_list
        }
        
        # Add optional training_config if provided
        if training_config:
            request_data["training_config"] = training_config
        
        # Submit to predictor service /train endpoint
        response = await self.client.post(
            f"{self.predictor_url}/train",
            json=request_data
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Training request submitted successfully: {result}")
            return result
        else:
            # Parse error response
            try:
                error_detail = response.json()
            except Exception:
                error_detail = {"error": response.text}
            
            error_msg = f"Training submission failed: HTTP {response.status_code}"
            if isinstance(error_detail, dict):
                error_msg += f" - {error_detail.get('error', 'Unknown error')}"
                if "message" in error_detail:
                    error_msg += f": {error_detail['message']}"
            else:
                error_msg += f" - {error_detail}"
            
            raise Exception(error_msg)

    async def predict(
        self,
        model_id: str,
        platform_info: Dict[str, str],
        prediction_type: str,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get runtime prediction from predictor service.

        Args:
            model_id: Model identifier
            platform_info: Platform information dict
            prediction_type: Type of prediction ('expect_error' or 'quantile')
            features: Feature dictionary (without runtime_ms)

        Returns:
            Prediction response from predictor service

        Raises:
            Exception: If prediction fails
        """
        request_data = {
            "model_id": model_id,
            "platform_info": {
                "software_name": platform_info["software_name"],
                "software_version": platform_info["software_version"],
                "hardware_name": platform_info["hardware_name"]
            },
            "prediction_type": prediction_type,
            "features": features,
            "enable_preprocessors": ["semantic"],
            "preprocessor_mappings": {
                "semantic": ["sentence"]
            }
        }

        response = await self.client.post(
            f"{self.predictor_url}/predict",
            json=request_data
        )

        if response.status_code == 200:
            return response.json()
        else:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = {"error": response.text}

            error_msg = f"Prediction failed: HTTP {response.status_code}"
            if isinstance(error_detail, dict):
                error_msg += f" - {error_detail.get('error', 'Unknown error')}"
                if "message" in error_detail:
                    error_msg += f": {error_detail['message']}"
            else:
                error_msg += f" - {error_detail}"

            raise Exception(error_msg)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def calculate_pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """
    Calculate pinball loss for quantile regression.

    Pinball loss formula:
    L(y, q_hat) = sum((y - q_hat) * quantile) if y >= q_hat
                  sum((q_hat - y) * (1 - quantile)) if y < q_hat

    Args:
        actual: Actual runtime values (array)
        predicted: Predicted runtime values (array)
        quantile: Quantile level (0 < quantile < 1)

    Returns:
        Average pinball loss
    """
    errors = actual - predicted
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    return float(np.mean(loss))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE formula:
    MAPE = (1/n) * sum(|actual - predicted| / actual) * 100%

    Args:
        actual: Actual runtime values (array)
        predicted: Predicted runtime values (array)
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE percentage
    """
    # Avoid division by zero
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
    mape = np.mean(np.abs((actual - predicted) / actual_safe)) * 100
    return float(mape)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL file.
    
    Args:
        dataset_path: Path to the dataset.jsonl file
        
    Returns:
        List of dataset entries
    """
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    logger.info(f"Loaded {len(dataset)} entries from {dataset_path}")
    return dataset


def extract_tasks_from_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract individual LLM tasks from dataset entries.

    Each dataset entry contains a boot (query generation) and summary task.
    We extract all tasks that have sentences for LLM execution.

    Handles both expected format (dict with 'input' and 'max_tokens') and
    actual format (plain string for boot/summary).

    Args:
        dataset: List of dataset entries

    Returns:
        List of tasks, each with sentence and max_tokens
    """
    tasks = []
    boot_count = 0
    summary_count = 0
    query_count = 0

    for idx, entry in enumerate(dataset):
        # Generate entry_id if missing
        entry_id = entry.get("id", f"entry-{idx:03d}")

        # Extract boot task (query generation)
        # Handle both dict format and string format
        if "boot" in entry and entry["boot"]:
            boot_data = entry["boot"]

            if isinstance(boot_data, dict) and "input" in boot_data:
                # Expected format: {"input": "...", "max_tokens": 512}
                tasks.append({
                    "sentence": boot_data["input"],
                    "max_tokens": boot_data.get("max_tokens", 512),
                    "task_type": "boot",
                    "entry_id": entry_id
                })
                boot_count += 1
            elif isinstance(boot_data, str):
                # Actual format: plain string
                # Estimate max_tokens from content length
                estimated_tokens = estimate_token_length(boot_data)
                max_tokens = 512 if estimated_tokens < 500 else 1024

                tasks.append({
                    "sentence": boot_data,
                    "max_tokens": max_tokens,
                    "task_type": "boot",
                    "entry_id": entry_id
                })
                boot_count += 1

        # Extract summary task
        # Handle both dict format and string format
        if "summary" in entry and entry["summary"]:
            summary_data = entry["summary"]

            if isinstance(summary_data, dict) and "input" in summary_data:
                # Expected format: {"input": "...", "max_tokens": 512}
                tasks.append({
                    "sentence": summary_data["input"],
                    "max_tokens": summary_data.get("max_tokens", 512),
                    "task_type": "summary",
                    "entry_id": entry_id
                })
                summary_count += 1
            elif isinstance(summary_data, str):
                # Actual format: plain string
                # Estimate max_tokens from content length
                estimated_tokens = estimate_token_length(summary_data)
                max_tokens = 512 if estimated_tokens < 500 else 1024

                tasks.append({
                    "sentence": summary_data,
                    "max_tokens": max_tokens,
                    "task_type": "summary",
                    "entry_id": entry_id
                })
                summary_count += 1

        # Extract query tasks from queries list
        if "queries" in entry and isinstance(entry["queries"], list):
            for query_idx, query_data in enumerate(entry["queries"]):
                if "input" in query_data:
                    # Try to get max_tokens from multiple sources
                    max_tokens = query_data.get("max_tokens")

                    if max_tokens is None:
                        # Try to infer from output_len field (actual dataset has this)
                        output_len = query_data.get("output_len")
                        if output_len:
                            # Use output_len as a proxy for max_tokens
                            # Round up to nearest power of 2 for better model training
                            max_tokens = 128 if output_len < 128 else \
                                        256 if output_len < 256 else \
                                        512 if output_len < 512 else \
                                        1024 if output_len < 1024 else 2048
                        else:
                            # Fall back to estimating from input length
                            estimated_tokens = estimate_token_length(query_data["input"])
                            max_tokens = 256 if estimated_tokens < 200 else \
                                        512 if estimated_tokens < 500 else 1024

                    tasks.append({
                        "sentence": query_data["input"],
                        "max_tokens": max_tokens,
                        "task_type": "query",
                        "entry_id": f"{entry_id}-q{query_idx}"
                    })
                    query_count += 1

    # Log extraction statistics
    logger.info(f"Extracted {len(tasks)} LLM tasks from {len(dataset)} dataset entries")
    logger.info(f"  - Boot tasks: {boot_count}")
    logger.info(f"  - Summary tasks: {summary_count}")
    logger.info(f"  - Query tasks: {query_count}")

    # Log max_tokens distribution for data quality validation
    max_tokens_dist = {}
    for task in tasks:
        mt = task["max_tokens"]
        max_tokens_dist[mt] = max_tokens_dist.get(mt, 0) + 1

    logger.info(f"  - max_tokens distribution: {dict(sorted(max_tokens_dist.items()))}")

    return tasks


async def collect_training_samples(
    multi_client: MultiInstanceLLMClient,
    tasks: List[Dict[str, Any]],
    max_concurrent: int = 10,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Collect training samples by executing tasks in parallel across multiple instances.

    Args:
        multi_client: Multi-instance LLM client with load balancing
        tasks: List of tasks to execute
        max_concurrent: Maximum number of concurrent requests
        max_samples: Maximum number of samples to collect (None for all)

    Returns:
        List of training samples, each with features and runtime_ms
    """
    samples = []
    num_tasks = min(len(tasks), max_samples) if max_samples else len(tasks)

    logger.info(f"Collecting training samples from {num_tasks} tasks using {len(multi_client.clients)} instance(s)...")
    logger.info(f"Max concurrent requests: {max_concurrent}")

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_task_with_semaphore(task: Dict[str, Any], task_idx: int) -> Optional[Dict[str, Any]]:
        """Execute a single task with semaphore control."""
        async with semaphore:
            sentence = task["sentence"]
            max_tokens = task.get("max_tokens", 512)
            token_length = estimate_token_length(sentence)

            # Get next client using round-robin
            client = multi_client.get_next_client()

            # Execute task
            result = await client.execute_task(sentence, max_tokens)

            if result["success"]:
                # Get instance config to extract hardware specs
                instance_config = multi_client.get_instance_config(result["instance_id"])
                hardware_name = instance_config.get("hardware_name", "Unknown")

                # Extract hardware specs for this instance
                hardware_specs = extract_gpu_specs(hardware_name) or {}

                # Build features: user metadata + hardware specs
                features = {
                    "token_length": float(token_length),
                    "max_tokens": float(max_tokens),
                }

                # Add hardware specs
                for k, v in hardware_specs.items():
                    if isinstance(v, (int, float)):
                        features[k] = float(v)

                # Add runtime_ms
                sample = features.copy()
                sample["runtime_ms"] = float(result["execution_time_ms"])

                return sample
            else:
                logger.warning(f"Task {task_idx} failed on {result.get('instance_id')}: {result.get('error')}")
                return None

    # Execute all tasks in parallel with concurrency limit
    tasks_to_execute = tasks[:num_tasks]

    # Use tqdm with asyncio.gather for progress tracking
    with tqdm(total=len(tasks_to_execute), desc="Executing tasks") as pbar:
        async def execute_and_update(task: Dict[str, Any], idx: int):
            result = await execute_task_with_semaphore(task, idx)
            pbar.update(1)
            return result

        # Execute all tasks concurrently
        results = await asyncio.gather(*[
            execute_and_update(task, idx)
            for idx, task in enumerate(tasks_to_execute)
        ])

    # Filter out failed tasks (None results)
    samples = [r for r in results if r is not None]

    logger.info(f"Collected {len(samples)} successful training samples ({len(results) - len(samples)} failed)")
    return samples


async def validate_model(
    predictor_client: PredictorClient,
    model_id: str,
    platform_info: Dict[str, str],
    prediction_type: str,
    validation_samples: List[Dict[str, Any]],
    quantiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Validate trained model by making predictions on validation samples and calculating metrics.

    Args:
        predictor_client: Predictor service client
        model_id: Model identifier
        platform_info: Platform information dict
        prediction_type: Type of prediction ('expect_error' or 'quantile')
        validation_samples: List of samples with features and runtime_ms
        quantiles: List of quantile levels (for quantile prediction type)

    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating {prediction_type.upper()} model with {len(validation_samples)} samples...")
    logger.info(f"{'='*60}")

    # Extract actual runtimes
    actual_runtimes = np.array([sample["runtime_ms"] for sample in validation_samples])

    if prediction_type == "expect_error":
        # Make predictions for all samples
        predictions = []
        errors = []

        logger.info("Making predictions on validation samples...")
        for idx, sample in enumerate(tqdm(validation_samples, desc="Predicting")):
            # Extract features (remove runtime_ms)
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            try:
                pred_response = await predictor_client.predict(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    features=features
                )

                predicted_runtime = pred_response["result"]["expected_runtime_ms"]
                error_margin = pred_response["result"]["error_margin_ms"]

                predictions.append(predicted_runtime)
                errors.append(error_margin)

            except Exception as e:
                logger.warning(f"Prediction failed for sample {idx}: {e}")
                predictions.append(np.nan)
                errors.append(np.nan)

        predictions = np.array(predictions)
        errors = np.array(errors)

        # Filter out failed predictions
        valid_mask = ~np.isnan(predictions)
        if not np.any(valid_mask):
            logger.error("All predictions failed - cannot calculate metrics")
            return {"error": "All predictions failed"}

        valid_actual = actual_runtimes[valid_mask]
        valid_predictions = predictions[valid_mask]
        valid_errors = errors[valid_mask]

        # Calculate MAPE
        mape = calculate_mape(valid_actual, valid_predictions)

        # Calculate Mean Absolute Error
        mae = float(np.mean(np.abs(valid_actual - valid_predictions)))

        # Calculate Mean Error Margin
        mean_error_margin = float(np.mean(valid_errors))

        # Calculate coverage (percentage of actual values within error margin)
        within_margin = np.abs(valid_actual - valid_predictions) <= valid_errors
        coverage = float(np.mean(within_margin)) * 100

        results = {
            "prediction_type": "expect_error",
            "samples_validated": int(np.sum(valid_mask)),
            "samples_failed": int(len(validation_samples) - np.sum(valid_mask)),
            "metrics": {
                "mape_percent": round(mape, 2),
                "mae_ms": round(mae, 2),
                "mean_error_margin_ms": round(mean_error_margin, 2),
                "coverage_percent": round(coverage, 2)
            }
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPECT_ERROR Model Validation Results:")
        logger.info(f"{'='*60}")
        logger.info(f"  Samples validated: {results['samples_validated']}/{len(validation_samples)}")
        logger.info(f"  MAPE: {results['metrics']['mape_percent']:.2f}%")
        logger.info(f"  MAE: {results['metrics']['mae_ms']:.2f} ms")
        logger.info(f"  Mean Error Margin: {results['metrics']['mean_error_margin_ms']:.2f} ms")
        logger.info(f"  Coverage (within error margin): {results['metrics']['coverage_percent']:.2f}%")
        logger.info(f"{'='*60}\n")

        return results

    elif prediction_type == "quantile":
        if not quantiles:
            logger.error("Quantiles must be provided for quantile prediction type")
            return {"error": "Missing quantiles"}

        # Make predictions for all samples
        all_quantile_predictions = {q: [] for q in quantiles}

        logger.info("Making predictions on validation samples...")
        for idx, sample in enumerate(tqdm(validation_samples, desc="Predicting")):
            # Extract features (remove runtime_ms)
            features = {k: v for k, v in sample.items() if k != "runtime_ms"}

            try:
                pred_response = await predictor_client.predict(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    features=features
                )

                # Extract quantile predictions
                quantile_results = pred_response["result"]["quantiles"]

                for q in quantiles:
                    q_str = str(q)
                    if q_str in quantile_results:
                        all_quantile_predictions[q].append(quantile_results[q_str])
                    else:
                        all_quantile_predictions[q].append(np.nan)

            except Exception as e:
                logger.warning(f"Prediction failed for sample {idx}: {e}")
                for q in quantiles:
                    all_quantile_predictions[q].append(np.nan)

        # Calculate metrics for each quantile (Pinball Loss only)
        quantile_metrics = {}

        for q in quantiles:
            predictions = np.array(all_quantile_predictions[q])

            # Filter out failed predictions
            valid_mask = ~np.isnan(predictions)
            if not np.any(valid_mask):
                logger.warning(f"All predictions failed for quantile {q}")
                continue

            valid_actual = actual_runtimes[valid_mask]
            valid_predictions = predictions[valid_mask]

            # Calculate Pinball Loss
            pinball = calculate_pinball_loss(valid_actual, valid_predictions, q)

            quantile_metrics[q] = {
                "pinball_loss": round(pinball, 2),
                "samples_valid": int(np.sum(valid_mask))
            }

        # Calculate distribution statistics (mean and median) for MAPE
        # For each sample, compute mean and median across all quantiles
        distribution_means = []
        distribution_medians = []

        num_samples = len(validation_samples)
        for sample_idx in range(num_samples):
            sample_quantile_values = []
            for q in quantiles:
                if sample_idx < len(all_quantile_predictions[q]):
                    val = all_quantile_predictions[q][sample_idx]
                    if not np.isnan(val):
                        sample_quantile_values.append(val)

            if sample_quantile_values:
                distribution_means.append(np.mean(sample_quantile_values))
                distribution_medians.append(np.median(sample_quantile_values))
            else:
                distribution_means.append(np.nan)
                distribution_medians.append(np.nan)

        distribution_means = np.array(distribution_means)
        distribution_medians = np.array(distribution_medians)

        # Calculate MAPE for distribution mean and median
        # For each sample, calculate the absolute percentage error, then average
        mape_mean = np.nan
        mape_median = np.nan

        # Filter out failed predictions
        valid_mask = ~(np.isnan(distribution_means) | np.isnan(distribution_medians))

        if np.any(valid_mask):
            valid_actual = actual_runtimes[valid_mask]
            valid_means = distribution_means[valid_mask]
            valid_medians = distribution_medians[valid_mask]

            # Calculate MAPE for distribution mean (per-sample error → aggregate MAPE)
            mape_mean = calculate_mape(valid_actual, valid_means)

            # Calculate MAPE for distribution median (per-sample error → aggregate MAPE)
            mape_median = calculate_mape(valid_actual, valid_medians)

        # Calculate average pinball loss across all quantiles
        avg_pinball = np.mean([m["pinball_loss"] for m in quantile_metrics.values()])

        results = {
            "prediction_type": "quantile",
            "samples_validated": len(validation_samples),
            "quantile_metrics": quantile_metrics,
            "distribution_metrics": {
                "avg_pinball_loss": round(float(avg_pinball), 2),
                "mape_mean_percent": round(float(mape_mean), 2) if not np.isnan(mape_mean) else None,
                "mape_median_percent": round(float(mape_median), 2) if not np.isnan(mape_median) else None
            }
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"QUANTILE Model Validation Results:")
        logger.info(f"{'='*60}")
        logger.info(f"  Samples validated: {results['samples_validated']}")
        logger.info(f"\nPer-Quantile Pinball Loss:")
        for q, metrics in quantile_metrics.items():
            logger.info(f"  Quantile {q}: {metrics['pinball_loss']:.2f} (samples: {metrics['samples_valid']})")

        logger.info(f"\nDistribution Metrics:")
        logger.info(f"  Avg Pinball Loss: {results['distribution_metrics']['avg_pinball_loss']:.2f}")
        if results['distribution_metrics']['mape_mean_percent'] is not None:
            logger.info(f"  MAPE (distribution mean): {results['distribution_metrics']['mape_mean_percent']:.2f}%")
        if results['distribution_metrics']['mape_median_percent'] is not None:
            logger.info(f"  MAPE (distribution median): {results['distribution_metrics']['mape_median_percent']:.2f}%")
        logger.info(f"{'='*60}\n")

        return results

    else:
        logger.error(f"Unknown prediction type: {prediction_type}")
        return {"error": f"Unknown prediction type: {prediction_type}"}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ['dataset', 'model_id', 'instances', 'predictor']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    # Set defaults
    config.setdefault('prediction_types', ['expect_error', 'quantile'])
    config.setdefault('max_samples', None)
    config.setdefault('training_config', {
        'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    })
    config.setdefault('execution', {})
    config['execution'].setdefault('timeout', 300.0)
    config['execution'].setdefault('max_concurrent_requests', 10)

    # Ensure training_config has quantiles
    if 'quantiles' not in config['training_config']:
        config['training_config']['quantiles'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    return config


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Collect training data for predictor model using configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config.json:
{
  "dataset": "data/dataset.jsonl",
  "model_id": "llama-7b",
  "prediction_types": ["expect_error", "quantile"],
  "instances": [
    {
      "url": "http://localhost:8001",
      "hardware_name": "NVIDIA H20",
      "software_name": "sglang",
      "software_version": "1.0.0"
    }
  ],
  "predictor": {
    "url": "http://localhost:9000"
  },
  "training_config": {
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
  },
  "execution": {
    "timeout": 300.0,
    "max_concurrent_requests": 10
  }
}
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file (see --help epilog for example)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="training_data.json",
        help="Path to output file (default: training_data.json)"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Load dataset
    logger.info(f"Loading dataset from {config['dataset']}")
    dataset = load_dataset(config['dataset'])

    # Extract tasks
    tasks = extract_tasks_from_dataset(dataset)

    if not tasks:
        logger.error("No tasks found in dataset")
        return

    # Initialize multi-instance LLM client
    multi_client = MultiInstanceLLMClient(
        instance_configs=config['instances'],
        timeout=config['execution']['timeout']
    )

    # Initialize predictor client
    predictor_client = PredictorClient(config['predictor']['url'])

    

    try:
        if os.path.exists(args.output_file):
            logger.error(f"Output file {args.output_file} already exists, train with it instead of collecting new data")
            logger.info(f"Loading training data from {args.output_file}")
            with open(args.output_file, 'r') as f:
                training_data = json.load(f)
                all_samples = training_data['samples']
                model_id = training_data['model_id']
                platform_info = training_data['platform_info']
                logger.info(f"Model ID: {model_id}")
                logger.info(f"Platform info: {platform_info}")
                logger.info(f"Samples: {all_samples}")
                logger.info(f"Number of samples: {len(all_samples)}")
                logger.info(f"Prediction types: {config['prediction_types']}")
                logger.info(f"Training config: {config['training_config']}")
                logger.info(f"Training data loaded from {args.output_file}")
            logger.info(f"Training with {len(all_samples)} samples from {args.output_file}")
        else:
            # Collect all training samples using parallel execution
            logger.info(f"Collecting training samples from all {len(tasks)} tasks...")

            # Apply max_samples limit if specified
            max_tasks = min(len(tasks), config['max_samples']) if config['max_samples'] else len(tasks)

            # Collect all samples with parallel execution across instances
            all_samples = await collect_training_samples(
                multi_client=multi_client,
                tasks=tasks,
                max_concurrent=config['execution']['max_concurrent_requests'],
                max_samples=max_tasks
            )

            # Check if we have enough samples (minimum 10 required by predictor)
            if len(all_samples) < 10:
                logger.error(f"Insufficient samples: {len(all_samples)} collected, but at least 10 required for training")
                logger.error("Check if tasks are failing or increase dataset size")
                return

            logger.info(f"Successfully collected {len(all_samples)} training samples")

            # Use the first instance's platform info (all instances should have same platform)
            first_instance = config['instances'][0]
            platform_info = {
                "software_name": first_instance.get("software_name", "sglang"),
                "software_version": first_instance.get("software_version", "1.0.0"),
                "hardware_name": first_instance.get("hardware_name", "Unknown")
            }

            # Save collected training data to file
            output_file = args.output_file

            logger.info(f"Saving training data to {output_file}")
            with open(output_file, 'w') as f:
                json.dump({
                    'model_id': config['model_id'],
                    'platform_info': platform_info,
                    'samples': all_samples,
                    'num_samples': len(all_samples),
                    'config': {
                        'prediction_types': config['prediction_types'],
                        'training_config': config['training_config']
                    }
                }, f, indent=2)
            logger.info(f"✓ Training data saved to {output_file}")

        # Submit training data once for each prediction type

        logger.info(f"Submitting training data to predictor for {len(config['prediction_types'])} model type(s)...")
        logger.info(f"Platform info: {platform_info}")

        for prediction_type in config['prediction_types']:
            try:
                logger.info(f"Training {prediction_type} model with {len(all_samples)} samples...")
                response = await predictor_client.submit_training_data(
                    model_id=config['model_id'],
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    features_list=all_samples.copy(),  # Use copy to avoid modification
                    training_config=config['training_config']
                )

                logger.info(f"✓ {prediction_type.upper()} model training completed successfully:")
                logger.info(f"  Model key: {response.get('model_key')}")
                logger.info(f"  Samples trained: {response.get('samples_trained')}")
                logger.info(f"  Status: {response.get('status')}")
                logger.info(f"  Message: {response.get('message')}")

                # Validate the trained model
                logger.info(f"\nValidating {prediction_type} model...")
                try:
                    validation_results = await validate_model(
                        predictor_client=predictor_client,
                        model_id=config['model_id'],
                        platform_info=platform_info,
                        prediction_type=prediction_type,
                        validation_samples=all_samples,
                        quantiles=config['training_config'].get('quantiles') if prediction_type == 'quantile' else None
                    )

                    # Store validation results
                    if "error" not in validation_results:
                        logger.info(f"✓ {prediction_type.upper()} model validation completed successfully")
                    else:
                        logger.error(f"✗ {prediction_type.upper()} model validation failed: {validation_results['error']}")

                except Exception as e:
                    logger.error(f"✗ Failed to validate {prediction_type} model: {e}")

            except Exception as e:
                logger.error(f"✗ Failed to train {prediction_type} model: {e}")
                # Continue with other prediction types even if one fails

        logger.info(f"\n{'='*60}")
        logger.info(f"Training data collection and model training completed!")
        logger.info(f"  Total samples collected: {len(all_samples)}")
        logger.info(f"  Instances used: {len(config['instances'])}")
        logger.info(f"  Prediction types trained: {', '.join(config['prediction_types'])}")
        logger.info(f"{'='*60}")

    finally:
        await multi_client.close_all()
        await predictor_client.close()


if __name__ == "__main__":
    asyncio.run(main())

