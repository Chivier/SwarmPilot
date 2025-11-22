#!/usr/bin/env python3
"""
Training Data Collection Script for Predictor Model (Text2Video Experiment)

This script:
1. Reads tasks from dataset (local jsonl or HuggingFace)
2. Executes pipeline:
   - Stage 1: LLM (A1/A2) to generate prompts
   - Stage 2: T2Vid (B) using generated prompts with varying frame counts
3. Submits training data to the predictor service

Usage:
    python collect_training_data.py \
        --config config_pipeline.json \
        --llm_limit 1000 \
        --t2vid_limit 100
"""

import argparse
import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import random

import httpx
import numpy as np
from tqdm import tqdm

# Try importing datasets
try:
    from datasets import load_dataset as hf_load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Hardware Performance Information
# ============================================================================

NVIDIA_TESLA_SPECS = {
    "V100": {"cuda_cores": 5120, "tensor_cores": 640, "fp32_tflops": 15.7, "fp16_tflops": 31.4, "tensor_tflops": 125.0, "memory_gb": 16, "memory_bandwidth_gb_s": 900},
    "V100-32GB": {"cuda_cores": 5120, "tensor_cores": 640, "fp32_tflops": 15.7, "fp16_tflops": 31.4, "tensor_tflops": 125.0, "memory_gb": 32, "memory_bandwidth_gb_s": 900},
    "T4": {"cuda_cores": 2560, "tensor_cores": 320, "fp32_tflops": 8.1, "fp16_tflops": 65.0, "tensor_tflops": 130.0, "memory_gb": 16, "memory_bandwidth_gb_s": 300},
    "A100": {"cuda_cores": 6912, "tensor_cores": 432, "fp32_tflops": 19.5, "fp16_tflops": 78.0, "tensor_tflops": 312.0, "memory_gb": 40, "memory_bandwidth_gb_s": 1555},
    "A100-80GB": {"cuda_cores": 6912, "tensor_cores": 432, "fp32_tflops": 19.5, "fp16_tflops": 78.0, "tensor_tflops": 312.0, "memory_gb": 80, "memory_bandwidth_gb_s": 2039},
    "A10": {"cuda_cores": 9216, "tensor_cores": 288, "fp32_tflops": 31.2, "fp16_tflops": 125.0, "tensor_tflops": 250.0, "memory_gb": 24, "memory_bandwidth_gb_s": 600},
    "A30": {"cuda_cores": 3584, "tensor_cores": 224, "fp32_tflops": 10.3, "fp16_tflops": 82.0, "tensor_tflops": 165.0, "memory_gb": 24, "memory_bandwidth_gb_s": 933},
    "A40": {"cuda_cores": 10752, "tensor_cores": 336, "fp32_tflops": 37.4, "fp16_tflops": 74.8, "tensor_tflops": 150.0, "memory_gb": 48, "memory_bandwidth_gb_s": 696},
    "H100": {"cuda_cores": 14592, "tensor_cores": 456, "fp32_tflops": 51.0, "fp16_tflops": 204.0, "tensor_tflops": 989.0, "fp8_tensor_tflops": 1979.0, "memory_gb": 80, "memory_bandwidth_gb_s": 3350},
    "H100-94GB": {"cuda_cores": 14592, "tensor_cores": 456, "fp32_tflops": 51.0, "fp16_tflops": 204.0, "tensor_tflops": 989.0, "fp8_tensor_tflops": 1979.0, "memory_gb": 94, "memory_bandwidth_gb_s": 3350},
    "H100-PCIe": {"cuda_cores": 14592, "tensor_cores": 456, "fp32_tflops": 48.0, "fp16_tflops": 192.0, "tensor_tflops": 756.0, "fp8_tensor_tflops": 1513.0, "memory_gb": 80, "memory_bandwidth_gb_s": 2000},
    "H20": {"cuda_cores": 17920, "tensor_cores": 560, "fp32_tflops": 63.0, "fp16_tflops": 252.0, "tensor_tflops": 1230.0, "fp8_tensor_tflops": 2460.0, "memory_gb": 96, "memory_bandwidth_gb_s": 4000},
}


def extract_gpu_specs(hardware_name: str) -> Optional[Dict[str, Any]]:
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
    if not text:
        return 0
    return max(1, len(text) // 4)


class ServiceClient:
    """Client for interacting with Model Service instance."""

    def __init__(self, instance_url: str, model_id: str, timeout: float = 300.0, instance_id: str = ""):
        self.instance_url = instance_url.rstrip('/')
        self.model_id = model_id
        self.timeout = timeout
        self.instance_id = instance_id or instance_url
        self.client = httpx.AsyncClient(timeout=timeout)

    async def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            # Construct payload based on model type
            payload = {}
            if self.model_id == "llm_service_small_model":
                payload = {
                    "sentence": task_input.get("sentence", ""),
                    "max_tokens": task_input.get("max_tokens", 512)
                }
            elif self.model_id == "t2vid":
                payload = {
                    "prompt": task_input.get("prompt", ""),
                    "negative_prompt": task_input.get("negative_prompt", ""),
                    "frames": task_input.get("frames", 16)
                }
            else:
                # Default pass-through
                payload = task_input

            # Mock execution for testing
            # if self.instance_url == "mock":
            #     return {
            #         "execution_time_ms": 100.0,
            #         "success": True,
            #         "result": {"output": "mock output"},
            #         "error": None,
            #         "instance_id": self.instance_id
            #     }

            response = await self.client.post(
                f"{self.instance_url}/inference",
                json=payload
            )

            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                result_data = response.json()
                return {
                    "execution_time_ms": execution_time_ms,
                    "success": True,
                    "result": result_data,
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
        await self.client.aclose()


class MultiInstanceClient:
    def __init__(self, instance_configs: List[Dict[str, str]], model_id: str, timeout: float = 300.0):
        self.clients = []
        for idx, config in enumerate(instance_configs):
            instance_id = f"instance-{idx}@{config['url']}"
            client = ServiceClient(
                instance_url=config['url'],
                model_id=model_id,
                timeout=timeout,
                instance_id=instance_id
            )
            self.clients.append({
                'client': client,
                'config': config,
                'instance_id': instance_id
            })
        self.current_idx = 0
        logger.info(f"Initialized {len(self.clients)} instance(s) for {model_id}")

    def get_next_client(self) -> ServiceClient:
        if not self.clients:
            raise Exception("No clients initialized")
        client_info = self.clients[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.clients)
        return client_info['client']

    def get_instance_config(self, instance_id: str) -> Dict[str, str]:
        for client_info in self.clients:
            if client_info['instance_id'] == instance_id:
                return client_info['config']
        return {}

    async def close_all(self):
        for client_info in self.clients:
            await client_info['client'].close()


class PredictorClient:
    def __init__(self, predictor_url: str, timeout: float = 600.0):
        self.predictor_url = predictor_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def submit_training_data(self, model_id: str, platform_info: Dict[str, str], prediction_type: str, features_list: List[Dict[str, Any]], training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        if model_id == "llm_service_small_model":
            request_data = {
                "model_id": model_id,
                "platform_info": {
                    "software_name": platform_info["software_name"],
                    "software_version": platform_info["software_version"],
                    "hardware_name": platform_info["hardware_name"]
                },
                "prediction_type": prediction_type,
                "features_list": features_list,
                "enable_preprocessors": ["semantic"],
                "preprocessor_mappings": {
                    "semantic": ["sentence"]
                }
            }
        else:
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
        
        response = await self.client.post(f"{self.predictor_url}/train", json=request_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Training submission failed: HTTP {response.status_code} - {response.text}")

    async def predict(self, model_id: str, platform_info: Dict[str, str], prediction_type: str, features: Dict[str, Any]) -> Dict[str, Any]:
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
                "semantic": ["sentence", "prompt", "negative_prompt"]
            }
        }
        response = await self.client.post(f"{self.predictor_url}/predict", json=request_data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Prediction failed: HTTP {response.status_code} - {response.text}")

    async def close(self):
        await self.client.aclose()


def calculate_pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    errors = actual - predicted
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-10) -> float:
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
    mape = np.mean(np.abs((actual - predicted) / actual_safe)) * 100
    return float(mape)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset_entries(dataset_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset from local JSON file (list of strings).
    Expected format: ["caption1", "caption2", ...]
    """
    dataset = []
    
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error(f"Dataset must be a JSON list, got {type(data)}")
            return []

        count = 0
        for item in data:
            if limit and count >= limit:
                break
            
            if isinstance(item, str):
                dataset.append({"caption": item, "id": f"local-{count}"})
                count += 1
            elif isinstance(item, dict):
                # Handle list of dicts if present, though primarily targeting list of strings
                if "caption" in item:
                    dataset.append({"caption": item["caption"], "id": f"local-{count}"})
                    count += 1
                elif "text" in item:
                    dataset.append({"caption": item["text"], "id": f"local-{count}"})
                    count += 1
        
        logger.info(f"Loaded {len(dataset)} entries from {dataset_path}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []


# ============================================================================
# Prompt Templates
# ============================================================================

A1_TEMPLATE = "Generate a detailed image generation prompt based on this caption: {caption}"
A2_TEMPLATE = "Generate a negative prompt for image generation to avoid artifacts, based on this positive prompt: {positive_prompt}"


# ============================================================================
# Pipeline Execution
# ============================================================================

async def execute_tasks(
    multi_client: MultiInstanceClient,
    tasks: List[Dict[str, Any]],
    max_concurrent: int = 10,
    desc: str = "Executing tasks"
) -> List[Dict[str, Any]]:
    
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_task_with_semaphore(task: Dict[str, Any], task_idx: int) -> Optional[Dict[str, Any]]:
        async with semaphore:
            client = multi_client.get_next_client()
            result = await client.execute_task(task)

            if result["success"]:
                instance_config = multi_client.get_instance_config(result["instance_id"])
                hardware_name = instance_config.get("hardware_name", "Unknown")
                hardware_specs = extract_gpu_specs(hardware_name) or {}

                features = {}

                # Add hardware specs
                for k, v in hardware_specs.items():
                    if isinstance(v, (int, float)):
                        features[k] = float(v)

                # LLM Features (match experiments/07.Exp2.Deep_Research_Real/collect_training_data.py)
                if "sentence" in task:
                    features["sentence"] = task["sentence"]
                    features["token_length"] = float(estimate_token_length(task["sentence"]))
                    if "max_tokens" in task:
                        features["max_tokens"] = float(task["max_tokens"])
                
                # T2Vid Features (frames and input token count)
                if "prompt" in task:
                    features["positive_prompt_length"] = float(estimate_token_length(task["prompt"]))
                    if "negative_prompt" in task:
                        features["negative_prompt_length"] = float(estimate_token_length(task["negative_prompt"]))
                    if "frames" in task:
                        features["frames"] = float(task["frames"])3

                sample = features.copy()
                sample["runtime_ms"] = float(result["execution_time_ms"])
                
                # Store raw output for pipeline chaining
                sample["_raw_output"] = result["result"].get("output", "")
                sample["_entry_id"] = task.get("entry_id")
                
                return sample
            else:
                logger.warning(f"Task {task_idx} failed: {result.get('error')}")
                return None

    with tqdm(total=len(tasks), desc=desc) as pbar:
        async def execute_and_update(task, idx):
            res = await execute_task_with_semaphore(task, idx)
            pbar.update(1)
            return res

        results = await asyncio.gather(*[execute_and_update(t, i) for i, t in enumerate(tasks)])

    samples = [r for r in results if r is not None]
    return samples


async def collect_pipeline_samples(
    config: Dict[str, Any],
    dataset: List[Dict[str, Any]],
    llm_limit: int,
    t2vid_limit: int
) -> Dict[str, List[Dict[str, Any]]]:
    
    # 1. Setup Clients
    llm_config = config.get("llm_service", {})
    t2vid_config = config.get("t2vid_service", {})
    
    if not llm_config or not t2vid_config:
        logger.error("Config must contain 'llm_service' and 't2vid_service' sections")
        return {}, {}

    llm_client = MultiInstanceClient(llm_config['instances'], "llm_service_small_model")
    t2vid_client = MultiInstanceClient(t2vid_config['instances'], "t2vid")

    all_llm_samples = []
    all_t2vid_samples = []

    try:
        # 2. Generate LLM Tasks (A1 & A2)
        # We need to run A1 first to get positive prompt, then A2 to get negative prompt
        # But for parallel efficiency in data collection, we can treat them somewhat independently if we just want training data.
        # HOWEVER, the user requirement implies a pipeline: "combine prompt... pass to t2vid".
        # So we must chain: Caption -> A1 -> Positive Prompt -> A2 -> Negative Prompt -> T2Vid
        
        # Limit initial captions
        captions = dataset[:llm_limit]
        logger.info(f"Starting pipeline with {len(captions)} captions")

        # --- Stage 1: A1 (Caption -> Positive Prompt) ---
        a1_tasks = []
        for idx, entry in enumerate(captions):
            entry_id = entry.get("id", f"entry-{idx}")
            if "caption" in entry:
                a1_tasks.append({
                    "sentence": A1_TEMPLATE.format(caption=entry["caption"]),
                    "max_tokens": 512,
                    "task_type": "A1",
                    "entry_id": entry_id,
                    "caption_raw": entry["caption"] # Keep for reference
                })
        
        logger.info(f"Executing {len(a1_tasks)} A1 tasks...")
        a1_samples = await execute_tasks(llm_client, a1_tasks, desc="LLM A1 Tasks")
        all_llm_samples.extend(a1_samples)
        
        # --- Stage 2: A2 (Positive Prompt -> Negative Prompt) ---
        # Only proceed with successful A1 samples
        a2_tasks = []
        valid_chains = {} # entry_id -> {positive_prompt: ...}
        
        for sample in a1_samples:
            entry_id = sample["_entry_id"]
            positive_prompt = sample["_raw_output"]
            if positive_prompt:
                valid_chains[entry_id] = {"positive_prompt": positive_prompt}
                a2_tasks.append({
                    "sentence": A2_TEMPLATE.format(positive_prompt=positive_prompt),
                    "max_tokens": 512,
                    "task_type": "A2",
                    "entry_id": entry_id
                })
        
        logger.info(f"Executing {len(a2_tasks)} A2 tasks...")
        a2_samples = await execute_tasks(llm_client, a2_tasks, desc="LLM A2 Tasks")
        all_llm_samples.extend(a2_samples)

        # Update chains with negative prompts
        completed_chains = []
        for sample in a2_samples:
            entry_id = sample["_entry_id"]
            negative_prompt = sample["_raw_output"]
            if entry_id in valid_chains and negative_prompt:
                chain = valid_chains[entry_id]
                chain["negative_prompt"] = negative_prompt
                completed_chains.append(chain)

        logger.info(f"Completed {len(completed_chains)} full LLM chains (Prompt Generation)")

        # --- Stage 3: T2Vid (Prompts -> Video) ---
        # Select subset for T2Vid
        t2vid_chains = completed_chains[:t2vid_limit]
        
        t2vid_tasks = []
        frame_counts = [30, 60, 90, 120]
        
        for chain in t2vid_chains:
            for frames in frame_counts:
                t2vid_tasks.append({
                    "prompt": chain["positive_prompt"],
                    "negative_prompt": chain["negative_prompt"],
                    "frames": frames,
                    "task_type": "B",
                    "entry_id": "t2vid-task" # ID doesn't matter as much here
                })
        
        logger.info(f"Executing {len(t2vid_tasks)} T2Vid tasks ({len(t2vid_chains)} prompts * {len(frame_counts)} frame settings)...")
        t2vid_samples = await execute_tasks(t2vid_client, t2vid_tasks, desc="T2Vid Tasks", max_concurrent=2) # Lower concurrency for video
        all_t2vid_samples.extend(t2vid_samples)

    finally:
        await llm_client.close_all()
        await t2vid_client.close_all()

    return {
        "llm_samples": all_llm_samples,
        "t2vid_samples": all_t2vid_samples
    }


async def validate_model(predictor_client, model_id, platform_info, prediction_type, validation_samples, quantiles=None):
    if not validation_samples:
        return
        
    logger.info(f"Validating {prediction_type} model for {model_id} with {len(validation_samples)} samples...")
    actual_runtimes = np.array([s["runtime_ms"] for s in validation_samples])
    
    predictions = []
    
    for sample in tqdm(validation_samples, desc=f"Predicting {model_id}"):
        # Filter features to only what predictor expects
        features = {k: v for k, v in sample.items() if k not in ["runtime_ms", "_raw_output", "_entry_id"]}
        try:
            resp = await predictor_client.predict(model_id, platform_info, prediction_type, features)
            
            if prediction_type == "expect_error":
                predictions.append(resp["result"]["expected_runtime_ms"])
            elif prediction_type == "quantile":
                predictions.append(resp["result"]["quantiles"])
        except Exception as e:
            # logger.warning(f"Prediction failed: {e}")
            predictions.append(None)

    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    if not valid_indices:
        logger.warning("All predictions failed")
        return
        
    if prediction_type == "expect_error":
        valid_preds = np.array([predictions[i] for i in valid_indices])
        valid_actual = actual_runtimes[valid_indices]
        mape = calculate_mape(valid_actual, valid_preds)
        logger.info(f"MAPE ({model_id}): {mape:.2f}%")
    
    elif prediction_type == "quantile":
        results = {}
        for q in quantiles:
            q_str = str(q)
            q_preds = []
            q_actuals = []
            for i in valid_indices:
                if q_str in predictions[i]:
                    q_preds.append(predictions[i][q_str])
                    q_actuals.append(actual_runtimes[i])
            
            if q_preds:
                loss = calculate_pinball_loss(np.array(q_actuals), np.array(q_preds), q)
                results[q] = loss
                logger.info(f"Quantile {q} Pinball Loss ({model_id}): {loss:.2f}")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


async def main():
    parser = argparse.ArgumentParser(description="Collect training data for Text2Video experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--output_file", type=str, default="training_data.json", help="Path to output file")
    parser.add_argument("--llm_limit", type=int, default=100, help="Max LLM samples to collect")
    parser.add_argument("--t2vid_limit", type=int, default=20, help="Max T2Vid samples to collect")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    dataset_path = config.get('dataset', 'nkp37/OpenVid-1M')
    logger.info(f"Loading dataset from {dataset_path}")
    
    # Load enough dataset entries to satisfy LLM limit
    dataset = load_dataset_entries(dataset_path, limit=args.llm_limit * 2) # Load a bit more to be safe
    
    if not dataset:
        logger.error("No dataset entries found")
        return

    # Collect samples
    samples_dict = await collect_pipeline_samples(
        config, dataset, args.llm_limit, args.t2vid_limit
    )
    
    llm_samples = samples_dict.get("llm_samples", [])
    t2vid_samples = samples_dict.get("t2vid_samples", [])
    
    logger.info(f"Collected {len(llm_samples)} LLM samples and {len(t2vid_samples)} T2Vid samples")

    # Save Data
    output_data = {
        'config': config,
        'llm_samples': llm_samples,
        't2vid_samples': t2vid_samples,
        'timestamp': time.time()
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved training data to {args.output_file}")

    # Train/Validate if predictor is configured
    if 'predictor' in config:
        predictor_client = PredictorClient(config['predictor']['url'])
        
        # Helper to get platform info from first instance of a service
        def get_platform_info(service_config):
            if not service_config or not service_config.get('instances'):
                return {"hardware_name": "Unknown", "software_name": "Unknown", "software_version": "0.0"}
            inst = service_config['instances'][0]
            return {
                "hardware_name": inst.get("hardware_name", "Unknown"),
                "software_name": inst.get("software_name", "Unknown"),
                "software_version": inst.get("software_version", "0.0")
            }

        try:
            # Train LLM Model
            if llm_samples:
                llm_platform = get_platform_info(config.get('llm_service'))
                for p_type in config.get('prediction_types', ['expect_error', 'quantile']):
                    await predictor_client.submit_training_data(
                        "llm_service_small_model", llm_platform, p_type, llm_samples, config.get('training_config')
                    )
                    await validate_model(
                        predictor_client, "llm_service_small_model", llm_platform, p_type, llm_samples,
                        quantiles=config.get('training_config', {}).get('quantiles')
                    )

            # Train T2Vid Model
            if t2vid_samples:
                t2vid_platform = get_platform_info(config.get('t2vid_service'))
                for p_type in config.get('prediction_types', ['expect_error', 'quantile']):
                    await predictor_client.submit_training_data(
                        "t2vid", t2vid_platform, p_type, t2vid_samples, config.get('training_config')
                    )
                    await validate_model(
                        predictor_client, "t2vid", t2vid_platform, p_type, t2vid_samples,
                        quantiles=config.get('training_config', {}).get('quantiles')
                    )

        finally:
            await predictor_client.close()

if __name__ == "__main__":
    asyncio.run(main())
