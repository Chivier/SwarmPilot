#!/usr/bin/env python3
"""Collect runtime data for Qwen3-VL-8B-Instruct and train a quantile predictor.

Before deploying, checks for existing instances on the cluster.
If instances exist, terminates them and deploys the same number of
Qwen3-VL-8B-Instruct instances. Otherwise, deploys the default count.

Phases:
    A. Check existing deployments; terminate if present.
    B. Deploy Qwen3-VL-8B-Instruct instances via SwarmPilot SDK.
    C. Wait for scheduler backend to be ready.
    D. Send chat-completion requests through the Scheduler proxy.
    E. Save collected runtime data to a JSON file.
    F. Train a QuantilePredictor MLP and save the model.

Usage:
    # Default: 4 instances, collect 200 samples and train
    uv run python predictor_training_playground/collect_and_train_qwen3vl.py --train

    # Collect only
    uv run python predictor_training_playground/collect_and_train_qwen3vl.py

    # Train from existing data
    uv run python predictor_training_playground/collect_and_train_qwen3vl.py \
        --load-json qwen3vl_runtime_data.json --train
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from loguru import logger

# ── Defaults ─────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_PLANNER_URL = "http://localhost:8002"
DEFAULT_SCHEDULER_URL = "http://localhost:8000"
DEFAULT_CAPTIONS_FILE = "captions_10k.jsonl"
DEFAULT_OUTPUT_JSON = "qwen3vl_runtime_data.json"
DEFAULT_STORAGE_DIR = "models"
DEFAULT_NUM_REQUESTS = 200
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_GPU = 1
DEFAULT_REPLICAS = 4
DEFAULT_TIMEOUT = 600.0
DEFAULT_HEALTH_TIMEOUT = 300.0
DEFAULT_RETRY_DELAY = 5.0
MAX_RETRIES_PER_REQUEST = 60

PLATFORM_INFO = {
    "software_name": "vllm",
    "software_version": "0.11.0",
    "hardware_name": "NVIDIA RTX A6000",
}

MAX_TOKENS_CHOICES = [64, 128, 256, 512]


# ── Helpers ──────────────────────────────────────────────────────


def estimate_prompt_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token).

    Args:
        text: Input text string.

    Returns:
        Estimated token count (at least 1).
    """
    return max(1, len(text) // 4)


def load_captions(path: str, limit: int) -> list[str]:
    """Load captions from a JSONL file.

    Args:
        path: Path to the JSONL file (each line has a "caption" field).
        limit: Maximum number of captions to load.

    Returns:
        List of caption strings.
    """
    captions: list[str] = []
    with open(path) as f:
        for line in f:
            if len(captions) >= limit:
                break
            data = json.loads(line.strip())
            caption = data.get("caption", "")
            if caption:
                captions.append(caption)
    logger.info(f"Loaded {len(captions)} captions from {path}")
    return captions


# ── Health check ─────────────────────────────────────────────────


async def wait_scheduler_ready(
    scheduler_url: str,
    model_id: str,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> None:
    """Wait until the scheduler proxy can reach the vLLM backend.

    Sends a minimal chat-completion probe through the scheduler proxy
    and waits until it returns HTTP 200.

    Args:
        scheduler_url: Scheduler service URL.
        model_id: Model identifier for the probe request.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If the backend is not ready within timeout.
    """
    probe_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    start = time.time()
    attempt = 0
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
    ) as client:
        while time.time() - start < timeout:
            attempt += 1
            try:
                resp = await client.post(
                    f"{scheduler_url}/v1/chat/completions",
                    json=probe_payload,
                )
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    logger.success(
                        f"Scheduler backend ready after "
                        f"{elapsed:.1f}s ({attempt} probes)"
                    )
                    return
                logger.info(
                    f"Probe {attempt}: HTTP {resp.status_code}, "
                    f"retrying in {DEFAULT_RETRY_DELAY}s..."
                )
            except Exception as exc:
                logger.info(
                    f"Probe {attempt}: {exc}, "
                    f"retrying in {DEFAULT_RETRY_DELAY}s..."
                )
            await asyncio.sleep(DEFAULT_RETRY_DELAY)
    raise TimeoutError(
        f"Scheduler backend not ready after {timeout}s"
    )


# ── Phase A: Check & teardown existing deployments ───────────────


async def check_and_teardown(
    planner_url: str,
    default_replicas: int,
) -> int:
    """Check for existing instances and terminate them if present.

    Queries the planner for all running instances. If any exist,
    terminates them all and returns the count so the new deployment
    can match it.

    Args:
        planner_url: Planner service URL.
        default_replicas: Replica count to use if no existing
            instances are found.

    Returns:
        Number of replicas to deploy (existing count or default).
    """
    from swarmpilot.sdk import SwarmPilotClient

    async with SwarmPilotClient(planner_url=planner_url) as sp:
        state = await sp.instances()
        existing = state.instances
        if not existing:
            logger.info(
                "No existing instances found, will deploy "
                f"{default_replicas} replicas"
            )
            return default_replicas

        count = len(existing)
        models = {inst.model or "unknown" for inst in existing}
        logger.info(
            f"Found {count} existing instance(s) "
            f"(models: {models}), terminating..."
        )
        await sp.terminate(all=True)
        logger.success("All existing instances terminated")

        # Wait a moment for cluster resources to release
        await asyncio.sleep(3)
        return count


# ── Phase B: Deploy ──────────────────────────────────────────────


async def deploy_instances(
    planner_url: str,
    scheduler_url: str,
    model_id: str,
    gpu: int,
    replicas: int,
) -> None:
    """Deploy vLLM instances via SwarmPilot SDK.

    Args:
        planner_url: Planner service URL.
        scheduler_url: Scheduler service URL.
        model_id: HuggingFace model identifier.
        gpu: GPUs per instance.
        replicas: Number of instances to deploy.
    """
    from swarmpilot.sdk import SwarmPilotClient

    logger.info(
        f"Deploying {replicas} instance(s) of {model_id} "
        f"(gpu={gpu})"
    )

    async with SwarmPilotClient(
        planner_url=planner_url,
        scheduler_url=scheduler_url,
    ) as sp:
        group = await sp.serve(
            model_id,
            gpu=gpu,
            replicas=replicas,
        )
        logger.info("Deployment submitted, waiting for instances...")
        await group.wait_ready(timeout=DEFAULT_TIMEOUT)
        logger.success(
            f"{replicas} instance(s) ready: {group.endpoints}"
        )


# ── Phase D: Collect ─────────────────────────────────────────────


async def collect_single(
    client: httpx.AsyncClient,
    scheduler_url: str,
    model_id: str,
    caption: str,
    max_tokens: int,
) -> dict | None:
    """Send one chat-completion request and record timing.

    Args:
        client: Reusable async HTTP client.
        scheduler_url: Scheduler proxy URL.
        model_id: Model identifier for the request.
        caption: User prompt text.
        max_tokens: Maximum tokens to generate.

    Returns:
        Sample dict with features + runtime, or None on failure.
    """
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": caption},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start = time.time()
    try:
        response = await client.post(
            f"{scheduler_url}/v1/chat/completions",
            json=payload,
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            logger.warning(
                f"Request failed ({response.status_code}): "
                f"{response.text[:200]}"
            )
            return None

        result = response.json()
        usage = result.get("usage", {})

        prompt_tokens = usage.get(
            "prompt_tokens",
            estimate_prompt_tokens(caption),
        )

        return {
            "prompt_tokens": float(prompt_tokens),
            "max_tokens": float(max_tokens),
            "runtime_ms": elapsed_ms,
            "_completion_tokens": usage.get("completion_tokens", 0),
            "_total_tokens": usage.get("total_tokens", 0),
        }

    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        logger.warning(
            f"Request error after {elapsed_ms:.0f}ms: {exc}"
        )
        return None


async def collect_samples(
    scheduler_url: str,
    model_id: str,
    captions: list[str],
    max_concurrent: int,
) -> list[dict]:
    """Send requests through the scheduler proxy and collect timing.

    Args:
        scheduler_url: Scheduler proxy URL.
        model_id: Model identifier.
        captions: List of prompt texts.
        max_concurrent: Concurrency limit.

    Returns:
        List of sample dicts.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    samples: list[dict] = []
    completed = 0
    total = len(captions)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    ) as client:

        async def worker(caption: str) -> None:
            nonlocal completed
            max_tokens = random.choice(MAX_TOKENS_CHOICES)
            for retry in range(MAX_RETRIES_PER_REQUEST):
                async with semaphore:
                    sample = await collect_single(
                        client, scheduler_url, model_id,
                        caption, max_tokens,
                    )
                if sample is not None:
                    break
                logger.debug(
                    f"Retry {retry + 1}/{MAX_RETRIES_PER_REQUEST}"
                )
                await asyncio.sleep(DEFAULT_RETRY_DELAY)
            completed += 1
            if sample is not None:
                samples.append(sample)
            if completed % 10 == 0 or completed == total:
                logger.info(
                    f"Progress: {completed}/{total} "
                    f"({len(samples)} successful)"
                )

        tasks = [worker(cap) for cap in captions]
        await asyncio.gather(*tasks)

    logger.info(
        f"Collection complete: {len(samples)}/{total} successful"
    )
    return samples


# ── Phase E: Save ────────────────────────────────────────────────


def save_to_json(
    samples: list[dict],
    model_id: str,
    output_path: str,
) -> None:
    """Save collected samples to a JSON file.

    Args:
        samples: List of feature dicts.
        model_id: Model identifier.
        output_path: Output file path.
    """
    data = {
        "metadata": {
            "model_id": model_id,
            "platform_info": PLATFORM_INFO,
            "num_samples": len(samples),
            "strategy": "round_robin",
            "collected_at": datetime.now(UTC).isoformat(),
        },
        "features_list": samples,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.success(f"Saved {len(samples)} samples to {output_path}")


def load_from_json(path: str) -> tuple[dict, list[dict]]:
    """Load previously collected data.

    Args:
        path: JSON file path.

    Returns:
        Tuple of (metadata dict, features_list).
    """
    with open(path) as f:
        data = json.load(f)
    metadata = data["metadata"]
    features_list = data["features_list"]
    logger.info(
        f"Loaded {len(features_list)} samples from {path}"
    )
    return metadata, features_list


# ── Phase F: Train ───────────────────────────────────────────────


def train_quantile_predictor(
    features_list: list[dict],
    model_id: str,
    storage_dir: str,
) -> None:
    """Train a QuantilePredictor and save to storage.

    Args:
        features_list: Training data (each dict has features + runtime_ms).
        model_id: Model identifier for storage key.
        storage_dir: Directory for model persistence.
    """
    from swarmpilot.predictor.predictor.quantile import (
        QuantilePredictor,
    )
    from swarmpilot.predictor.storage.model_storage import (
        ModelStorage,
    )

    clean = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in features_list
    ]

    logger.info(
        f"Training quantile predictor with {len(clean)} samples "
        f"(features: {[k for k in clean[0] if k != 'runtime_ms']})"
    )

    training_config = {
        "epochs": 200,
        "learning_rate": 0.001,
        "hidden_layers": [64, 32],
        "quantiles": [0.5, 0.9, 0.95, 0.99],
        "data_augmentation": {
            "enabled": True,
            "samples_per_point": 5,
            "distribution": "lognormal",
        },
        "residual_calibration": {
            "enabled": True,
            "min_sigma": 0.1,
        },
    }

    predictor = QuantilePredictor()
    predictor.train(features_list=clean, config=training_config)
    logger.success("Training complete")

    storage = ModelStorage(storage_dir=storage_dir)
    model_key = storage.generate_model_key(
        model_id=model_id,
        platform_info=PLATFORM_INFO,
        prediction_type="quantile",
    )
    predictor_state = predictor.get_model_state()
    metadata = {
        "model_id": model_id,
        "platform_info": PLATFORM_INFO,
        "prediction_type": "quantile",
        "samples_count": len(clean),
        "training_config": training_config,
    }
    storage.save_model(model_key, predictor_state, metadata)
    logger.success(f"Model saved: {model_key}")

    test_features = {
        "prompt_tokens": 200.0,
        "max_tokens": 256.0,
    }
    result = predictor.predict(test_features)
    quantiles = result.get("quantiles", {})
    logger.info(f"Test prediction (prompt=200, max=256): {quantiles}")


# ── Main ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Collect Qwen3-VL-8B-Instruct runtime data and train "
            "a quantile predictor."
        ),
    )
    parser.add_argument(
        "--planner-url",
        default=DEFAULT_PLANNER_URL,
        help="Planner service URL",
    )
    parser.add_argument(
        "--scheduler-url",
        default=DEFAULT_SCHEDULER_URL,
        help="Scheduler service URL",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help="Number of inference requests to send",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON file for collected data",
    )
    parser.add_argument(
        "--captions-file",
        default=DEFAULT_CAPTIONS_FILE,
        help="JSONL file with caption prompts",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train quantile predictor after collection",
    )
    parser.add_argument(
        "--storage-dir",
        default=DEFAULT_STORAGE_DIR,
        help="Model storage directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU,
        help="GPUs per instance",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=DEFAULT_REPLICAS,
        help="Default number of instances (overridden by existing count)",
    )
    parser.add_argument(
        "--load-json",
        default="",
        help="Load existing data (skip collection)",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip instance deployment (already deployed)",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Terminate instances after collection",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the full collection and training pipeline."""
    args = parse_args()

    replicas = args.replicas

    # Phase A: Check existing deployments and teardown
    if not args.load_json and not args.skip_deploy:
        replicas = await check_and_teardown(
            planner_url=args.planner_url,
            default_replicas=args.replicas,
        )
        logger.info(f"Will deploy {replicas} instance(s)")

    # Phase B: Deploy
    if not args.load_json and not args.skip_deploy:
        await deploy_instances(
            planner_url=args.planner_url,
            scheduler_url=args.scheduler_url,
            model_id=args.model_id,
            gpu=args.gpu,
            replicas=replicas,
        )

    # Phase C: Wait for backend
    if not args.load_json and not args.skip_deploy:
        logger.info("Waiting for scheduler backend to be ready...")
        await wait_scheduler_ready(
            scheduler_url=args.scheduler_url,
            model_id=args.model_id,
        )

    if args.load_json:
        _metadata, features_list = load_from_json(args.load_json)
    else:
        # Phase D: Collect
        captions_path = Path(args.captions_file)
        if not captions_path.is_absolute():
            captions_path = Path(__file__).parent / captions_path
        captions = load_captions(
            str(captions_path), args.num_requests
        )
        features_list = await collect_samples(
            scheduler_url=args.scheduler_url,
            model_id=args.model_id,
            captions=captions,
            max_concurrent=args.max_concurrent,
        )

        # Phase E: Save
        save_to_json(
            features_list, args.model_id, args.output_json
        )

    # Phase F: Train
    if args.train:
        if len(features_list) < 10:
            logger.error(
                f"Need at least 10 samples for training, "
                f"got {len(features_list)}"
            )
        else:
            train_quantile_predictor(
                features_list=features_list,
                model_id=args.model_id,
                storage_dir=args.storage_dir,
            )

    # Cleanup
    if args.terminate:
        from swarmpilot.sdk import SwarmPilotClient

        async with SwarmPilotClient(
            planner_url=args.planner_url,
            scheduler_url=args.scheduler_url,
        ) as sp:
            await sp.terminate(model=args.model_id)
            logger.info("Instances terminated")


if __name__ == "__main__":
    asyncio.run(main())
