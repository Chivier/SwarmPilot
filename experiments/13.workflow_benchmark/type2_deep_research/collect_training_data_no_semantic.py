#!/usr/bin/env python3
"""
Training Data Collection Script for Type2 Deep Research Workflow Predictor Model.

This script:
1. Reads LLM execution tasks from dataset.jsonl (aligned with Exp07 format)
2. Executes each task on the LLM service and collects execution times
3. Transforms metadata according to scheduler's transformation mechanism
4. Submits training data to the predictor service

Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
  - Model A (llm_service_large_model): for A and Merge tasks (boot/summary)
  - Model B (llm_service_small_model): for B1/B2 tasks (query)

Supports multi-model configuration to train both Model A and Model B predictors
in a single execution.

Usage:
    # Using configuration file (trains both models)
    python collect_training_data.py --config training_configs/config_regular.json

    # With custom output directory
    python collect_training_data.py --config training_configs/config_batchgen.json --output_dir training_data_batchgen/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Extract GPU specifications from hardware name."""
    hardware_name_upper = hardware_name.upper()

    gpu_patterns = [
        (r'H20', 'H20'),
        (r'H100[- ]?PCIE', 'H100-PCIe'),
        (r'H100[- ]?94GB', 'H100-94GB'),
        (r'H100', 'H100'),
        (r'A100[- ]?80GB', 'A100-80GB'),
        (r'A100', 'A100'),
        (r'V100[- ]?32GB', 'V100-32GB'),
        (r'V100', 'V100'),
        (r'A40', 'A40'),
        (r'A30', 'A30'),
        (r'A10', 'A10'),
        (r'T4', 'T4'),
    ]

    for pattern, gpu_key in gpu_patterns:
        if re.search(pattern, hardware_name_upper):
            if gpu_key in NVIDIA_TESLA_SPECS:
                return NVIDIA_TESLA_SPECS[gpu_key].copy()

    return None


def estimate_token_length(text: Optional[str]) -> int:
    """Estimate token length from text (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class LLMServiceClient:
    """Client for interacting with LLM service instance."""

    def __init__(self, instance_url: str, timeout: float = 300.0, instance_id: str = ""):
        self.instance_url = instance_url.rstrip('/')
        self.timeout = timeout
        self.instance_id = instance_id or instance_url
        self.client = httpx.AsyncClient(timeout=timeout)

    async def execute_task(
        self,
        sentence: str,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """Execute a task on the LLM service and return execution time."""
        try:
            start_time = time.time()

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
    """Client for load-balanced parallel execution across multiple LLM instances."""

    def __init__(self, instance_configs: List[Dict[str, str]], timeout: float = 300.0):
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
        """Get next client using round-robin load balancing."""
        client_info = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        return client_info['client']

    def get_instance_config(self, instance_id: str) -> Dict[str, str]:
        """Get instance configuration by instance_id."""
        for client_info in self.clients:
            if client_info['instance_id'] == instance_id:
                return client_info['config']
        return {}

    async def close_all(self):
        """Close all HTTP clients."""
        for client_info in self.clients:
            await client_info['client'].close()


class PredictorClient:
    """Client for interacting with predictor service."""

    def __init__(self, predictor_url: str, timeout: float = 600.0):
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
        """Submit training data to predictor service."""
        features_list = [{k: v for k, v in s.items() if k not in ['sentence']} for s in features_list]
        request_data = {
            "model_id": model_id,
            "platform_info": {
                "software_name": platform_info["software_name"],
                "software_version": platform_info["software_version"],
                "hardware_name": platform_info["hardware_name"]
            },
            "prediction_type": prediction_type,
            "features_list": features_list,
        }

        if training_config:
            request_data["training_config"] = training_config

        response = await self.client.post(
            f"{self.predictor_url}/train",
            json=request_data
        )

        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Training request submitted successfully: {result}")
            return result
        else:
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
        """Get runtime prediction from predictor service."""
        del features["sentence"]
        request_data = {
            "model_id": model_id,
            "platform_info": {
                "software_name": platform_info["software_name"],
                "software_version": platform_info["software_version"],
                "hardware_name": platform_info["hardware_name"]
            },
            "prediction_type": prediction_type,
            "features": features,
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
            else:
                error_msg += f" - {error_detail}"

            raise Exception(error_msg)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def calculate_pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Calculate pinball loss for quantile regression."""
    errors = actual - predicted
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    return float(np.mean(loss))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
    mape = np.mean(np.abs((actual - predicted) / actual_safe)) * 100
    return float(mape)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    logger.info(f"Loaded {len(dataset)} entries from {dataset_path}")
    return dataset


def extract_tasks_from_dataset(
    dataset: List[Dict[str, Any]],
    task_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract individual LLM tasks from dataset entries.

    For Type2 Deep Research workflow:
    - boot: A task (llm_service_small_model, max_tokens=4096)
    - query: B1 tasks (llm_service_large_model, max_tokens=300)
    - criteria: B2 tasks (llm_service_large_model, max_tokens=1, receives B1 output)
    - summary: Merge task (llm_service_small_model, max_tokens=4096)

    Args:
        dataset: List of dataset entries
        task_types: Optional list of task types to extract ('boot', 'query', 'criteria', 'summary')

    Returns:
        List of tasks with sentence, max_tokens, and task_type
    """
    if task_types is None:
        task_types = ['boot', 'query', 'summary']

    tasks = []
    boot_count = 0
    summary_count = 0
    query_count = 0
    criteria_count = 0

    for idx, entry in enumerate(dataset):
        entry_id = entry.get("id", f"entry-{idx:03d}")

        # Extract boot task (A task)
        if 'boot' in task_types and "boot" in entry and entry["boot"]:
            boot_data = entry["boot"]

            if isinstance(boot_data, dict) and "input" in boot_data:
                sentence = boot_data["input"]
                max_tokens = boot_data.get("max_tokens", 4096)
            elif isinstance(boot_data, str):
                sentence = boot_data
                max_tokens = 4096  # A task uses 4096 tokens

            tasks.append({
                "sentence": sentence,
                "max_tokens": max_tokens,
                "task_type": "boot",
                "model_group": "A",  # Uses llm_service_small_model
                "entry_id": entry_id
            })
            boot_count += 1

        # Extract summary task (Merge task)
        if 'summary' in task_types and "summary" in entry and entry["summary"]:
            summary_data = entry["summary"]

            if isinstance(summary_data, dict) and "input" in summary_data:
                sentence = summary_data["input"]
                max_tokens = summary_data.get("max_tokens", 4096)
            elif isinstance(summary_data, str):
                sentence = summary_data
                max_tokens = 4096  # Merge task uses 4096 tokens

            tasks.append({
                "sentence": sentence,
                "max_tokens": max_tokens,
                "task_type": "summary",
                "model_group": "A",  # Uses llm_service_small_model
                "entry_id": entry_id
            })
            summary_count += 1

        # Extract query tasks (B1 tasks)
        if 'query' in task_types and "queries" in entry and isinstance(entry["queries"], list):
            for query_idx, query_data in enumerate(entry["queries"]):
                if "input" in query_data:
                    sentence = query_data["input"]
                    # B1 tasks use max_tokens=300 (full inference)
                    max_tokens = query_data.get("max_tokens", 300)

                    tasks.append({
                        "sentence": sentence,
                        "max_tokens": max_tokens,
                        "task_type": "query",
                        "model_group": "B",  # Uses llm_service_large_model
                        "entry_id": f"{entry_id}-q{query_idx}"
                    })
                    query_count += 1

        # Extract criteria tasks (B2 tasks)
        # B2 receives B1's output as input in real workflow, but for training data
        # collection we use the query inputs with max_tokens=1 to capture the
        # runtime characteristics of criteria tasks
        if 'criteria' in task_types and "queries" in entry and isinstance(entry["queries"], list):
            for query_idx, query_data in enumerate(entry["queries"]):
                if "input" in query_data:
                    sentence = query_data["input"]
                    # B2 tasks use max_tokens=1 (criteria/classification)
                    max_tokens = 1

                    tasks.append({
                        "sentence": sentence,
                        "max_tokens": max_tokens,
                        "task_type": "criteria",
                        "model_group": "B",  # Uses llm_service_large_model
                        "entry_id": f"{entry_id}-c{query_idx}"
                    })
                    criteria_count += 1

    logger.info(f"Extracted {len(tasks)} LLM tasks from {len(dataset)} dataset entries")
    logger.info(f"  - Boot tasks (Model A): {boot_count}")
    logger.info(f"  - Summary tasks (Model A): {summary_count}")
    logger.info(f"  - Query tasks (Model B, B1): {query_count}")
    logger.info(f"  - Criteria tasks (Model B, B2): {criteria_count}")

    # Log max_tokens distribution
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
    """Collect training samples by executing tasks in parallel across multiple instances."""
    samples = []
    num_tasks = min(len(tasks), max_samples) if max_samples else len(tasks)

    logger.info(f"Collecting training samples from {num_tasks} tasks using {len(multi_client.clients)} instance(s)...")
    logger.info(f"Max concurrent requests: {max_concurrent}")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_task_with_semaphore(task: Dict[str, Any], task_idx: int) -> Optional[Dict[str, Any]]:
        """Execute a single task with semaphore control."""
        async with semaphore:
            sentence = task["sentence"]
            max_tokens = task.get("max_tokens", 512)
            token_length = estimate_token_length(sentence)

            client = multi_client.get_next_client()
            result = await client.execute_task(sentence, max_tokens)

            if result["success"]:
                instance_config = multi_client.get_instance_config(result["instance_id"])
                hardware_name = instance_config.get("hardware_name", "Unknown")

                hardware_specs = extract_gpu_specs(hardware_name) or {}

                features = {
                    "sentence": sentence,
                    "token_length": float(token_length),
                    "max_tokens": float(max_tokens),
                }

                for k, v in hardware_specs.items():
                    if isinstance(v, (int, float)):
                        features[k] = float(v)

                sample = features.copy()
                sample["runtime_ms"] = float(result["execution_time_ms"])
                sample["task_type"] = task.get("task_type", "unknown")
                sample["model_group"] = task.get("model_group", "unknown")

                return sample
            else:
                logger.warning(f"Task {task_idx} failed on {result.get('instance_id')}: {result.get('error')}")
                return None

    tasks_to_execute = tasks[:num_tasks]

    with tqdm(total=len(tasks_to_execute), desc="Executing tasks") as pbar:
        async def execute_and_update(task: Dict[str, Any], idx: int):
            result = await execute_task_with_semaphore(task, idx)
            pbar.update(1)
            return result

        results = await asyncio.gather(*[
            execute_and_update(task, idx)
            for idx, task in enumerate(tasks_to_execute)
        ])

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
    """Validate trained model by making predictions on validation samples."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating {prediction_type.upper()} model with {len(validation_samples)} samples...")
    logger.info(f"{'='*60}")

    actual_runtimes = np.array([sample["runtime_ms"] for sample in validation_samples])

    if prediction_type == "expect_error":
        predictions = []
        errors = []

        logger.info("Making predictions on validation samples...")
        for idx, sample in enumerate(tqdm(validation_samples, desc="Predicting")):
            features = {k: v for k, v in sample.items() if k not in ["runtime_ms", "task_type", "model_group"]}

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

        valid_mask = ~np.isnan(predictions)
        if not np.any(valid_mask):
            logger.error("All predictions failed - cannot calculate metrics")
            return {"error": "All predictions failed"}

        valid_actual = actual_runtimes[valid_mask]
        valid_predictions = predictions[valid_mask]
        valid_errors = errors[valid_mask]

        mape = calculate_mape(valid_actual, valid_predictions)
        mae = float(np.mean(np.abs(valid_actual - valid_predictions)))
        mean_error_margin = float(np.mean(valid_errors))

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

        all_quantile_predictions = {q: [] for q in quantiles}

        logger.info("Making predictions on validation samples...")
        for idx, sample in enumerate(tqdm(validation_samples, desc="Predicting")):
            features = {k: v for k, v in sample.items() if k not in ["runtime_ms", "task_type", "model_group"]}

            try:
                pred_response = await predictor_client.predict(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    features=features
                )

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

        quantile_metrics = {}

        for q in quantiles:
            predictions = np.array(all_quantile_predictions[q])
            valid_mask = ~np.isnan(predictions)
            if not np.any(valid_mask):
                logger.warning(f"All predictions failed for quantile {q}")
                continue

            valid_actual = actual_runtimes[valid_mask]
            valid_predictions = predictions[valid_mask]

            pinball = calculate_pinball_loss(valid_actual, valid_predictions, q)

            quantile_metrics[q] = {
                "pinball_loss": round(pinball, 2),
                "samples_valid": int(np.sum(valid_mask))
            }

        avg_pinball = np.mean([m["pinball_loss"] for m in quantile_metrics.values()])

        results = {
            "prediction_type": "quantile",
            "samples_validated": len(validation_samples),
            "quantile_metrics": quantile_metrics,
            "distribution_metrics": {
                "avg_pinball_loss": round(float(avg_pinball), 2),
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
        logger.info(f"{'='*60}\n")

        return results

    else:
        logger.error(f"Unknown prediction type: {prediction_type}")
        return {"error": f"Unknown prediction type: {prediction_type}"}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Supports both single-model and multi-model configurations:

    Single-model (legacy):
    {
        "dataset": "...",
        "model_id": "...",
        "instances": [...],
        ...
    }

    Multi-model (new):
    {
        "dataset": "...",
        "models": [
            {"model_id": "...", "task_types": [...], "instances": [...]},
            {"model_id": "...", "task_types": [...], "instances": [...]}
        ],
        ...
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if it's a multi-model config
    if 'models' in config:
        # Multi-model config
        required_fields = ['dataset', 'models', 'predictor']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        # Validate each model config
        for idx, model_config in enumerate(config['models']):
            if 'model_id' not in model_config:
                raise ValueError(f"Missing 'model_id' in models[{idx}]")
            if 'instances' not in model_config:
                raise ValueError(f"Missing 'instances' in models[{idx}]")
            model_config.setdefault('task_types', ['boot', 'query', 'summary'])
    else:
        # Single-model config (legacy) - convert to multi-model format
        required_fields = ['dataset', 'model_id', 'instances', 'predictor']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        # Convert to multi-model format
        config['models'] = [{
            'model_id': config.pop('model_id'),
            'instances': config.pop('instances'),
            'task_types': config.pop('task_types', ['boot', 'query', 'summary'])
        }]

    # Set global defaults
    config.setdefault('prediction_types', ['expect_error', 'quantile'])
    config.setdefault('max_samples', None)
    config.setdefault('training_config', {
        'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    })
    config.setdefault('execution', {})
    config['execution'].setdefault('timeout', 300.0)
    config['execution'].setdefault('max_concurrent_requests', 10)

    if 'quantiles' not in config['training_config']:
        config['training_config']['quantiles'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    return config


async def process_model_config(
    model_config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    predictor_client: PredictorClient,
    global_config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Process a single model configuration: collect data, train, and validate.

    Args:
        model_config: Model-specific configuration (model_id, instances, task_types)
        dataset: Loaded dataset entries
        predictor_client: Predictor service client
        global_config: Global configuration (prediction_types, training_config, etc.)
        output_dir: Directory to save training data

    Returns:
        Summary of results for this model
    """
    model_id = model_config['model_id']
    task_types = model_config.get('task_types', ['boot', 'query', 'summary'])
    instances = model_config['instances']

    logger.info(f"\n{'#'*60}")
    logger.info(f"Processing model: {model_id}")
    logger.info(f"Task types: {task_types}")
    logger.info(f"Instances: {len(instances)}")
    logger.info(f"{'#'*60}\n")

    # Extract tasks for this model
    tasks = extract_tasks_from_dataset(dataset, task_types=task_types)

    if not tasks:
        logger.error(f"No tasks found for model {model_id} with task_types {task_types}")
        return {"model_id": model_id, "status": "error", "message": "No tasks found"}

    # Initialize multi-instance client for this model
    multi_client = MultiInstanceLLMClient(
        instance_configs=instances,
        timeout=global_config['execution']['timeout']
    )

    # Output file for this model
    output_file = os.path.join(output_dir, f"training_data_{model_id}.json")

    try:
        if os.path.exists(output_file):
            logger.info(f"Loading existing training data from {output_file}")
            with open(output_file, 'r') as f:
                training_data = json.load(f)
                all_samples = training_data['samples']
                platform_info = training_data['platform_info']
                logger.info(f"Loaded {len(all_samples)} samples")
        else:
            logger.info(f"Collecting training samples for {model_id}...")

            max_tasks = min(len(tasks), global_config['max_samples']) if global_config['max_samples'] else len(tasks)

            all_samples = await collect_training_samples(
                multi_client=multi_client,
                tasks=tasks,
                max_concurrent=global_config['execution']['max_concurrent_requests'],
                max_samples=max_tasks
            )

            if len(all_samples) < 10:
                logger.error(f"Insufficient samples for {model_id}: {len(all_samples)} collected, but at least 10 required")
                return {"model_id": model_id, "status": "error", "message": f"Insufficient samples: {len(all_samples)}"}

            logger.info(f"Successfully collected {len(all_samples)} training samples for {model_id}")

            first_instance = instances[0]
            platform_info = {
                "software_name": first_instance.get("software_name", "sglang"),
                "software_version": first_instance.get("software_version", "0.5.5.post2"),
                "hardware_name": first_instance.get("hardware_name", "Unknown")
            }

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Saving training data to {output_file}")
            with open(output_file, 'w') as f:
                json.dump({
                    'model_id': model_id,
                    'platform_info': platform_info,
                    'samples': all_samples,
                    'num_samples': len(all_samples),
                    'config': {
                        'task_types': task_types,
                        'prediction_types': global_config['prediction_types'],
                        'training_config': global_config['training_config']
                    }
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Training data saved to {output_file}")

        # Train and validate for each prediction type
        training_results = {}

        for prediction_type in global_config['prediction_types']:
            try:
                logger.info(f"Training {prediction_type} model for {model_id} with {len(all_samples)} samples...")
                response = await predictor_client.submit_training_data(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=prediction_type,
                    features_list=[{k: v for k, v in s.items() if k not in ['task_type', 'model_group']} for s in all_samples],
                    training_config=global_config['training_config']
                )

                logger.info(f"✓ {prediction_type.upper()} model training completed for {model_id}:")
                logger.info(f"  Model key: {response.get('model_key')}")
                logger.info(f"  Samples trained: {response.get('samples_trained')}")
                logger.info(f"  Status: {response.get('status')}")

                # Validate
                logger.info(f"\nValidating {prediction_type} model for {model_id}...")
                try:
                    validation_results = await validate_model(
                        predictor_client=predictor_client,
                        model_id=model_id,
                        platform_info=platform_info,
                        prediction_type=prediction_type,
                        validation_samples=all_samples,
                        quantiles=global_config['training_config'].get('quantiles') if prediction_type == 'quantile' else None
                    )

                    training_results[prediction_type] = {
                        "status": "success" if "error" not in validation_results else "validation_failed",
                        "training_response": response,
                        "validation": validation_results
                    }

                except Exception as e:
                    logger.error(f"✗ Failed to validate {prediction_type} model for {model_id}: {e}")
                    training_results[prediction_type] = {
                        "status": "validation_error",
                        "training_response": response,
                        "error": str(e)
                    }

            except Exception as e:
                logger.error(f"✗ Failed to train {prediction_type} model for {model_id}: {e}")
                training_results[prediction_type] = {
                    "status": "training_error",
                    "error": str(e)
                }

        return {
            "model_id": model_id,
            "status": "success",
            "samples_collected": len(all_samples),
            "task_types": task_types,
            "training_results": training_results
        }

    finally:
        await multi_client.close_all()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Collect training data for Type2 Deep Research workflow predictor models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example multi-model config.json (trains both Model A and Model B):
{
  "dataset": "data/dataset.jsonl",
  "models": [
    {
      "model_id": "llm_service_small_model",
      "task_types": ["boot", "summary"],
      "instances": [
        {"url": "http://29.209.106.237:8200", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "0.5.5.post2"},
        ...
      ]
    },
    {
      "model_id": "llm_service_large_model",
      "task_types": ["query", "criteria"],
      "instances": [
        {"url": "http://29.209.113.166:8200", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "0.5.5.post2"},
        ...
      ]
    }
  ],
  "predictor": {
    "url": "http://localhost:9000"
  },
  "prediction_types": ["expect_error", "quantile"],
  "training_config": {
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
  },
  "execution": {
    "timeout": 300.0,
    "max_concurrent_requests": 24
  }
}
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_data",
        help="Directory to save training data files (default: training_data)"
    )

    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    logger.info(f"Loading dataset from {config['dataset']}")
    dataset = load_dataset(config['dataset'])

    if not dataset:
        logger.error("No entries found in dataset")
        return

    # Initialize predictor client
    predictor_client = PredictorClient(config['predictor']['url'])

    try:
        # Process each model configuration
        all_results = []

        for model_config in config['models']:
            result = await process_model_config(
                model_config=model_config,
                dataset=dataset,
                predictor_client=predictor_client,
                global_config=config,
                output_dir=args.output_dir
            )
            all_results.append(result)

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Data Collection Complete!")
        logger.info(f"{'='*60}")

        for result in all_results:
            model_id = result['model_id']
            status = result['status']

            if status == 'success':
                logger.info(f"\n✓ {model_id}:")
                logger.info(f"    Samples: {result['samples_collected']}")
                logger.info(f"    Task types: {', '.join(result['task_types'])}")
                for pred_type, pred_result in result.get('training_results', {}).items():
                    logger.info(f"    {pred_type}: {pred_result.get('status', 'unknown')}")
            else:
                logger.info(f"\n✗ {model_id}: {status}")
                if 'message' in result:
                    logger.info(f"    {result['message']}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Models processed: {len(all_results)}")
        logger.info(f"{'='*60}")

    finally:
        await predictor_client.close()


if __name__ == "__main__":
    asyncio.run(main())
