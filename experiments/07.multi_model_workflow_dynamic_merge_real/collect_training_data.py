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

import httpx
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
    
    def __init__(self, instance_url: str, timeout: float = 300.0):
        """
        Initialize LLM service client.
        
        Args:
            instance_url: Base URL of the instance service
            timeout: Request timeout in seconds
        """
        self.instance_url = instance_url.rstrip('/')
        self.timeout = timeout
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
                    "error": None
                }
            else:
                return {
                    "execution_time_ms": execution_time_ms,
                    "success": False,
                    "result": None,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                "execution_time_ms": 0.0,
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class PredictorTrainingClient:
    """Client for submitting training data to predictor service."""
    
    def __init__(self, predictor_url: str, timeout: float = 600.0):
        """
        Initialize predictor training client.
        
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
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


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
    llm_client: LLMServiceClient,
    tasks: List[Dict[str, Any]],
    hardware_specs: Dict[str, Any],
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Collect training samples by executing tasks and measuring execution times.
    
    Args:
        llm_client: LLM service client
        tasks: List of tasks to execute
        hardware_specs: Hardware specifications extracted from platform_info
        max_samples: Maximum number of samples to collect (None for all)
        
    Returns:
        List of training samples, each with features and runtime_ms
    """
    samples = []
    num_tasks = min(len(tasks), max_samples) if max_samples else len(tasks)
    
    logger.info(f"Collecting training samples from {num_tasks} tasks...")
    
    for i, task in enumerate(tqdm(tasks[:num_tasks], desc="Executing tasks")):
        sentence = task["sentence"]
        max_tokens = task.get("max_tokens", 512)
        
        # Calculate metadata (same as scheduler submission)
        token_length = estimate_token_length(sentence)
        
        # Execute task and measure execution time
        result = await llm_client.execute_task(sentence, max_tokens)
        
        if result["success"]:
            # Build features: user metadata + hardware specs
            # This matches the transformation done by predictor's /predict endpoint
            # The predictor directly adds hardware spec fields (cuda_cores, tensor_cores, etc.)
            # to the features dict without any prefix
            features = {
                # User metadata (as submitted to scheduler)
                "token_length": float(token_length),
                "max_tokens": float(max_tokens),
            }
            
            # Add hardware specs directly (same keys as in hardware_specs dict)
            # This matches how predictor's /predict endpoint does it:
            # all_features[key] = value  (no prefix added)
            for k, v in hardware_specs.items():
                if isinstance(v, (int, float)):
                    features[k] = float(v)
            
            # Add runtime_ms (required for training)
            sample = features.copy()
            sample["runtime_ms"] = float(result["execution_time_ms"])
            
            samples.append(sample)
            
            # Small delay to avoid overwhelming the service
            await asyncio.sleep(0.1)
        else:
            logger.warning(f"Task {i} failed: {result.get('error')}")
            # Skip failed tasks
    
    logger.info(f"Collected {len(samples)} successful training samples")
    return samples


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect training data for predictor model")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset.jsonl file"
    )
    parser.add_argument(
        "--instance-url",
        type=str,
        default="http://localhost:8001",
        help="URL of the instance service (LLM service)"
    )
    parser.add_argument(
        "--predictor-url",
        type=str,
        default="http://localhost:8002",
        help="URL of the predictor service"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier for training"
    )
    parser.add_argument(
        "--prediction-types",
        type=str,
        nargs="+",
        choices=["expect_error", "quantile"],
        default=["expect_error", "quantile"],
        help="Types of prediction models to train (can specify both or one). Default: both"
    )
    parser.add_argument(
        "--hardware-name",
        type=str,
        default=None,
        help="Hardware name (e.g., 'NVIDIA H20'). If not provided, will be auto-detected from system or instance"
    )
    parser.add_argument(
        "--software-name",
        type=str,
        default="sglang",
        help="Software name (e.g., 'sglang')"
    )
    parser.add_argument(
        "--software-version",
        type=str,
        default="1.0.0",
        help="Software version"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of tasks to process for testing (None for all tasks). Use small value (e.g., 10-20) for quick testing"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Path to training configuration JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    # Detect platform info if not provided
    platform_info_dict = None
    if args.hardware_name is None or args.software_name is None or args.software_version is None:
        logger.info("Auto-detecting platform information...")
        platform_info_dict = await detect_platform_info(args.instance_url)
        logger.info(f"Using detected platform info: {platform_info_dict}")
        
        # Update args with detected values if not provided
        if args.hardware_name is None:
            args.hardware_name = platform_info_dict.get("hardware_name", "CPU")
        if args.software_name is None:
            args.software_name = platform_info_dict.get("software_name", "sglang")
        if args.software_version is None:
            args.software_version = platform_info_dict.get("software_version", "1.0.0")
    else:
        logger.info(f"Using provided platform info: hardware={args.hardware_name}, software={args.software_name} {args.software_version}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    # Extract tasks
    tasks = extract_tasks_from_dataset(dataset)
    
    if not tasks:
        logger.error("No tasks found in dataset")
        return
    
    # Get hardware specs using local extract_gpu_specs function
    # This replicates predictor's PlatformInfo.extract_gpu_specs() behavior
    hardware_specs = extract_gpu_specs(args.hardware_name) or {}
    
    if not hardware_specs:
        logger.warning(f"Could not extract hardware specs for {args.hardware_name}")
        logger.warning("Proceeding without hardware features (may cause training errors)")
    
    # Load training config if provided, or use defaults
    training_config = None
    if args.training_config:
        with open(args.training_config, 'r') as f:
            training_config = json.load(f)
    else:
        # Use default training config with custom quantiles
        # Default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        # This provides fine-grained percentile predictions across the distribution
        training_config = {
            'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        }
    
    # Initialize clients
    llm_client = LLMServiceClient(args.instance_url)
    predictor_client = PredictorTrainingClient(args.predictor_url)
    
    try:
        # Collect all training samples at once
        logger.info(f"Collecting training samples from all {len(tasks)} tasks...")

        # Apply max_samples limit if specified
        max_tasks = min(len(tasks), args.max_samples) if args.max_samples else len(tasks)
        tasks_to_process = tasks[:max_tasks]

        # Collect all samples in one go
        all_samples = await collect_training_samples(
            llm_client,
            tasks_to_process,
            hardware_specs,
            max_samples=None  # Process all selected tasks
        )

        # Check if we have enough samples (minimum 10 required by predictor)
        if len(all_samples) < 10:
            logger.error(f"Insufficient samples: {len(all_samples)} collected, but at least 10 required for training")
            logger.error("Try removing --max-samples limit or check if tasks are failing")
            return

        logger.info(f"Successfully collected {len(all_samples)} training samples")

        # Submit training data once for each prediction type
        logger.info(f"Submitting training data to predictor for {len(args.prediction_types)} model type(s)...")

        for prediction_type in args.prediction_types:
            try:
                logger.info(f"Training {prediction_type} model with {len(all_samples)} samples...")
                response = await predictor_client.submit_training_data(
                    model_id=args.model_id,
                    platform_info={
                        "software_name": args.software_name,
                        "software_version": args.software_version,
                        "hardware_name": args.hardware_name
                    },
                    prediction_type=prediction_type,
                    features_list=all_samples.copy(),  # Use copy to avoid modification
                    training_config=training_config
                )

                logger.info(f"✓ {prediction_type.upper()} model training completed successfully:")
                logger.info(f"  Model key: {response.get('model_key')}")
                logger.info(f"  Samples trained: {response.get('samples_trained')}")
                logger.info(f"  Status: {response.get('status')}")
                logger.info(f"  Message: {response.get('message')}")

            except Exception as e:
                logger.error(f"✗ Failed to train {prediction_type} model: {e}")
                # Continue with other prediction types even if one fails

        logger.info(f"\n{'='*60}")
        logger.info(f"Training data collection and model training completed!")
        logger.info(f"  Total samples collected: {len(all_samples)}")
        logger.info(f"  Prediction types trained: {', '.join(args.prediction_types)}")
        logger.info(f"{'='*60}")
        
    finally:
        await llm_client.close()
        await predictor_client.close()


if __name__ == "__main__":
    asyncio.run(main())

