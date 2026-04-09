#!/usr/bin/env python3
r"""Profile Qwen-8B and Qwen-Next-80B using extracted prompts.

Sends prompts from the JSONL file produced by extract_prompts.py to
both model endpoints, records end-to-end inference time, and saves
runtime data in the SwarmPilot predictor training format.

Output format per model:
    {
        "metadata": { "model_id", "platform_info", "num_samples", ... },
        "features_list": [
            { "prompt_tokens": float, "max_tokens": float, "runtime_ms": float, ... }
        ]
    }

Usage:
    # Profile both models with 600 prompts each
    uv run python profile_models.py \
        --prompts profiling_prompts.jsonl \
        --num-requests 600

    # Profile only the large model
    uv run python profile_models.py \
        --prompts profiling_prompts.jsonl \
        --models large \
        --large-endpoint http://localhost:8010

    # Use existing SwarmPilot cluster endpoints
    uv run python profile_models.py \
        --prompts profiling_prompts.jsonl \
        --large-endpoint http://localhost:8010 \
        --small-endpoint http://localhost:8020
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
LARGE_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
SMALL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LARGE_ENDPOINT = "http://localhost:8010"
DEFAULT_SMALL_ENDPOINT = "http://localhost:8020"
DEFAULT_NUM_REQUESTS = 600
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_TIMEOUT = 600.0
DEFAULT_HEALTH_TIMEOUT = 300.0

PLATFORM_INFO_LARGE = {
    "software_name": "vllm",
    "software_version": "0.11.0",
    "hardware_name": "NVIDIA RTX A6000",
}

PLATFORM_INFO_SMALL = {
    "software_name": "vllm",
    "software_version": "0.11.0",
    "hardware_name": "NVIDIA RTX A6000",
}

MAX_TOKENS_CHOICES = [64, 128, 256, 512]


# ── Helpers ──────────────────────────────────────────────────────


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate prompt token count from message list.

    Uses ~4 chars per token heuristic.

    Args:
        messages: List of message dicts with "content" field.

    Returns:
        Estimated token count (at least 1).
    """
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, total_chars // 4)


def load_prompts(path: str, limit: int | None = None) -> list[dict]:
    """Load prompts from JSONL file.

    Args:
        path: Path to the JSONL file produced by extract_prompts.py.
        limit: Maximum number of prompts to load.

    Returns:
        List of prompt dicts.
    """
    prompts: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
            if limit is not None and len(prompts) >= limit:
                break
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


# ── Health check ─────────────────────────────────────────────────


async def wait_endpoint_ready(
    endpoint: str,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> bool:
    """Poll the vLLM endpoint until it responds to /v1/models.

    Args:
        endpoint: vLLM base URL.
        timeout: Maximum seconds to wait.

    Returns:
        True if endpoint is ready, False on timeout.
    """
    url = f"{endpoint}/v1/models"
    start = time.time()
    attempt = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        while time.time() - start < timeout:
            attempt += 1
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    logger.success(
                        f"Endpoint {endpoint} ready after "
                        f"{elapsed:.1f}s ({attempt} attempts)"
                    )
                    return True
            except httpx.ConnectError:
                pass
            except Exception as exc:
                logger.debug(f"Health check attempt {attempt}: {exc}")
            if attempt % 10 == 0:
                elapsed = time.time() - start
                logger.info(
                    f"Waiting for {endpoint}... " f"({elapsed:.0f}s / {timeout:.0f}s)"
                )
            await asyncio.sleep(2)
    logger.error(f"Endpoint {endpoint} not ready after {timeout}s")
    return False


# ── Profiling ────────────────────────────────────────────────────


async def profile_single(
    client: httpx.AsyncClient,
    endpoint: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
) -> dict | None:
    """Send one inference request and record timing.

    Args:
        client: Reusable async HTTP client.
        endpoint: vLLM base URL.
        model_id: Model identifier for the request.
        messages: Full message history (system + conversation).
        max_tokens: Maximum tokens to generate.

    Returns:
        Sample dict with features + runtime, or None on failure.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    start = time.time()
    try:
        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            logger.warning(
                f"Request failed ({response.status_code}): " f"{response.text[:200]}"
            )
            return None

        result = response.json()
        usage = result.get("usage", {})

        prompt_tokens = usage.get(
            "prompt_tokens",
            estimate_prompt_tokens(messages),
        )

        return {
            "prompt_tokens": float(prompt_tokens),
            "max_tokens": float(max_tokens),
            "runtime_ms": elapsed_ms,
            "_completion_tokens": usage.get("completion_tokens", 0),
            "_total_tokens": usage.get("total_tokens", 0),
            "_dataset": messages[0].get("_dataset", ""),
            "_group_id": messages[0].get("_group_id", ""),
        }

    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        logger.warning(f"Request error after {elapsed_ms:.0f}ms: {exc}")
        return None


async def profile_model(
    endpoint: str,
    model_id: str,
    prompts: list[dict],
    max_concurrent: int,
    num_requests: int,
) -> list[dict]:
    """Profile a model by sending prompts and collecting runtime.

    Args:
        endpoint: vLLM base URL for the model.
        model_id: Model identifier.
        prompts: List of prompt dicts from extract_prompts.py.
        max_concurrent: Concurrency limit.
        num_requests: Number of profiling requests to send.

    Returns:
        List of sample dicts with features + runtime_ms.
    """
    # Select prompts: cycle if we need more than available.
    selected = []
    for i in range(num_requests):
        selected.append(prompts[i % len(prompts)])

    semaphore = asyncio.Semaphore(max_concurrent)
    samples: list[dict] = []
    completed = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
    ) as client:

        async def worker(prompt_data: dict) -> None:
            nonlocal completed
            max_tokens = random.choice(MAX_TOKENS_CHOICES)
            messages = prompt_data["messages"]

            async with semaphore:
                sample = await profile_single(
                    client,
                    endpoint,
                    model_id,
                    messages,
                    max_tokens,
                )

            completed += 1
            if sample is not None:
                # Tag with source metadata.
                sample["_dataset"] = prompt_data.get("dataset", "")
                sample["_group_id"] = prompt_data.get("group_id", "")
                samples.append(sample)

            if completed % 20 == 0 or completed == num_requests:
                logger.info(
                    f"[{model_id}] Progress: {completed}/{num_requests} "
                    f"({len(samples)} successful)"
                )

        tasks = [worker(p) for p in selected]
        await asyncio.gather(*tasks)

    logger.info(f"[{model_id}] Complete: {len(samples)}/{num_requests} successful")
    return samples


# ── Save ─────────────────────────────────────────────────────────


def save_runtime_data(
    samples: list[dict],
    model_id: str,
    platform_info: dict,
    output_path: str,
) -> None:
    """Save profiling results in SwarmPilot predictor training format.

    Args:
        samples: List of feature dicts.
        model_id: Model identifier.
        platform_info: Platform metadata.
        output_path: Output JSON file path.
    """
    data = {
        "metadata": {
            "model_id": model_id,
            "platform_info": platform_info,
            "num_samples": len(samples),
            "collected_at": datetime.now(UTC).isoformat(),
            "source": "profile_models.py",
        },
        "features_list": samples,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.success(f"Saved {len(samples)} samples to {output_path}")


# ── Main ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Profile models using extracted prompts.",
    )
    parser.add_argument(
        "--prompts",
        default="profiling_prompts.jsonl",
        help="Input JSONL file from extract_prompts.py",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["large", "small"],
        choices=["large", "small"],
        help="Which models to profile (default: both)",
    )
    parser.add_argument(
        "--large-endpoint",
        default=DEFAULT_LARGE_ENDPOINT,
        help=f"Endpoint for Qwen-Next-80B (default: {DEFAULT_LARGE_ENDPOINT})",
    )
    parser.add_argument(
        "--small-endpoint",
        default=DEFAULT_SMALL_ENDPOINT,
        help=f"Endpoint for Qwen-8B (default: {DEFAULT_SMALL_ENDPOINT})",
    )
    parser.add_argument(
        "--large-model-id",
        default=LARGE_MODEL_ID,
        help="Large model HuggingFace ID",
    )
    parser.add_argument(
        "--small-model-id",
        default=SMALL_MODEL_ID,
        help="Small model HuggingFace ID",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
        help="Profiling requests per model (default: 600)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Max concurrent requests (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip initial endpoint health check",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the model profiling pipeline."""
    args = parse_args()
    random.seed(args.seed)

    # Load prompts.
    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = Path(__file__).parent / prompts_path
    all_prompts = load_prompts(str(prompts_path))

    if not all_prompts:
        logger.error("No prompts loaded. Run extract_prompts.py first.")
        return

    # Split prompts by model_size for targeted profiling.
    large_prompts = [p for p in all_prompts if p.get("model_size") == "large"]
    small_prompts = [p for p in all_prompts if p.get("model_size") == "small"]

    # If not enough prompts for a model size, use all prompts.
    if len(large_prompts) < 10:
        logger.warning(
            f"Only {len(large_prompts)} large prompts, "
            f"using all {len(all_prompts)} prompts"
        )
        large_prompts = all_prompts
    if len(small_prompts) < 10:
        logger.warning(
            f"Only {len(small_prompts)} small prompts, "
            f"using all {len(all_prompts)} prompts"
        )
        small_prompts = all_prompts

    output_dir = Path(args.output_dir)

    # Model configs: (size_key, model_id, endpoint, prompts, platform, output).
    model_configs = []
    if "large" in args.models:
        model_configs.append(
            (
                "large",
                args.large_model_id,
                args.large_endpoint,
                large_prompts,
                PLATFORM_INFO_LARGE,
                "runtime_qwen80b.json",
            )
        )
    if "small" in args.models:
        model_configs.append(
            (
                "small",
                args.small_model_id,
                args.small_endpoint,
                small_prompts,
                PLATFORM_INFO_SMALL,
                "runtime_qwen8b.json",
            )
        )

    for (
        size_key,
        model_id,
        endpoint,
        prompts,
        platform_info,
        output_file,
    ) in model_configs:
        logger.info(f"═══ Profiling {model_id} ═══")

        if not args.skip_health_check:
            ready = await wait_endpoint_ready(endpoint)
            if not ready:
                logger.error(
                    f"{size_key.title()} model endpoint " f"not available, skipping."
                )
                continue

        samples = await profile_model(
            endpoint=endpoint,
            model_id=model_id,
            prompts=prompts,
            max_concurrent=args.max_concurrent,
            num_requests=args.num_requests,
        )
        save_runtime_data(
            samples,
            model_id,
            platform_info,
            str(output_dir / output_file),
        )

    logger.success("Profiling complete!")


if __name__ == "__main__":
    asyncio.run(main())
