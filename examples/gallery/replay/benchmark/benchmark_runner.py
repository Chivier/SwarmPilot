r"""SwarmPilot Real-Model Benchmark Runner.

Replays conversation datasets through a SwarmPilot cluster using the
scheduler's transparent proxy (``/v1/chat/completions``).  Each replay
group simulates one independent user.  Groups run fully concurrently;
within a group, steps are sequential: build cumulative history -> send
request -> wait for response -> inter-step delay -> next step.

Unlike the mock runner, this version sends REAL cumulative conversation
histories to actual LLM endpoints instead of sleep-time placeholders.

Usage::

    python benchmark_runner.py \
        --data path/to/replay_groups.jsonl \
        --config path/to/config.yaml \
        --output ./results [--limit 10]
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# Thread pool for sending HTTP requests — each request gets its own thread
# so that long-running proxy calls never block other groups.
_REQUEST_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=256)


# ── Data Loading ──────────────────────────────────────────────────


def load_config(path: str) -> dict[str, Any]:
    """Load benchmark configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed configuration dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_replay_groups(path: str, limit: int | None = None) -> list[dict]:
    """Load ReplayGroup dicts from a JSONL file.

    Each line must be a JSON object with the keys: ``group_id``,
    ``dataset_name``, ``initial_messages``, and ``steps``.  Each step
    has ``step_index``, ``model_size``, ``sender_role``, and
    ``history_message``.

    Args:
        path: Filesystem path to the JSONL file.
        limit: If set, stop loading after this many groups.

    Returns:
        List of raw group dicts (not validated against Pydantic models so
        that this module has no runtime dependency on the replay package).
    """
    groups: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
            if limit is not None and len(groups) >= limit:
                break
    return groups


# ── Runtime Estimation ────────────────────────────────────────────


def _estimate_runtime_ms(
    input_tokens: float, model_size: str, config: dict
) -> float:
    """Estimate request runtime in milliseconds from cumulative token count.

    Formula: ``(overhead_s + input_tokens / prefill_tps + decode_s) * 1000``

    Config keys used (prefix is ``large_`` or ``small_`` depending on
    ``model_size``):
    - ``{size}_overhead_s``: Fixed startup overhead in seconds.
    - ``{size}_prefill_tps``: Prefill throughput in tokens per second.
    - ``{size}_decode_s``: Fixed decode/generation time in seconds.

    Args:
        input_tokens: Estimated number of input (prompt) tokens.
        model_size: Either ``"large"`` or ``"small"``.
        config: Benchmark configuration dict.

    Returns:
        Estimated runtime in milliseconds.
    """
    prefix = "large" if model_size == "large" else "small"
    overhead_s: float = config.get(f"{prefix}_overhead_s", 0.5)
    prefill_tps: float = config.get(f"{prefix}_prefill_tps", 1000.0)
    decode_s: float = config.get(f"{prefix}_decode_s", 1.0)
    runtime_s = overhead_s + input_tokens / prefill_tps + decode_s
    return runtime_s * 1000.0


def compute_dataset_stats(groups: list[dict], config: dict) -> dict[str, Any]:
    """Compute P90 and mean estimated runtimes from replay group data.

    Estimates are based on cumulative character counts converted to tokens
    via ``chars_per_token`` (default 4) and the estimation formula from
    ``_estimate_runtime_ms``.

    Args:
        groups: List of raw group dicts loaded from JSONL.
        config: Benchmark configuration dict containing estimation params.

    Returns:
        Dict with keys: ``large_count``, ``small_count``,
        ``large_p90_ms``, ``small_p90_ms``,
        ``large_mean_ms``, ``small_mean_ms``.
    """
    chars_per_token: float = config.get("chars_per_token", 4.0)
    large_times: list[float] = []
    small_times: list[float] = []

    for g in groups:
        cumulative_chars = sum(
            len(json.dumps(m)) for m in g.get("initial_messages", [])
        )
        for step in g.get("steps", []):
            msg = step.get("history_message", {})
            cumulative_chars += len(json.dumps(msg))
            tokens = cumulative_chars / chars_per_token
            model_size = step.get("model_size", "small")
            est_ms = _estimate_runtime_ms(tokens, model_size, config)
            if model_size == "large":
                large_times.append(est_ms)
            else:
                small_times.append(est_ms)

    def _p90(vals: list[float]) -> float:
        """Return the 90th-percentile value of a sorted list."""
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * 0.9), len(s) - 1)]

    def _mean(vals: list[float]) -> float:
        """Return the arithmetic mean, or 0.0 for an empty list."""
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "large_count": len(large_times),
        "small_count": len(small_times),
        "large_p90_ms": _p90(large_times),
        "small_p90_ms": _p90(small_times),
        "large_mean_ms": _mean(large_times),
        "small_mean_ms": _mean(small_times),
    }


# ── Token Bucket (Global QPS) ────────────────────────────────────


class TokenBucket:
    """Async token bucket rate limiter with waiter tracking.

    Refills at ``rate`` tokens per second (capacity == rate).  Callers
    ``await acquire()`` to consume one token, blocking until available.

    Args:
        rate: Replenishment rate and maximum capacity in tokens/second.
    """

    def __init__(self, rate: float) -> None:
        """Initialise the bucket with ``rate`` tokens already available."""
        self._rate = rate
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        # Number of coroutines currently blocked waiting for a token.
        self.waiters = 0

    async def acquire(self) -> None:
        """Consume one token, waiting if none are currently available."""
        # Optimistic fast-path — try without declaring ourselves a waiter.
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._rate,
                self._tokens + (now - self._last) * self._rate,
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

        # Slow path — no token available; enter waiting state.
        self.waiters += 1
        try:
            while True:
                await asyncio.sleep(1.0 / self._rate)
                async with self._lock:
                    now = time.monotonic()
                    self._tokens = min(
                        self._rate,
                        self._tokens + (now - self._last) * self._rate,
                    )
                    self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
        finally:
            self.waiters -= 1


# ── Request Helpers ───────────────────────────────────────────────


def get_scheduler_url(model_size: str, config: dict) -> str:
    """Route model size to the correct scheduler base URL.

    Args:
        model_size: Either ``"large"`` or ``"small"``.
        config: Benchmark config containing ``scheduler_large_port`` and
            ``scheduler_small_port``.

    Returns:
        Scheduler base URL string (e.g. ``"http://localhost:8010"``).
    """
    if model_size == "large":
        port = config.get("scheduler_large_port", 8010)
    else:
        port = config.get("scheduler_small_port", 8020)
    return f"http://localhost:{port}"


def build_chat_request(
    messages: list[dict],
    model_id: str,
    max_tokens: int = 1,
    exp_runtime_ms: float | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible ``/v1/chat/completions`` request body.

    Sends real cumulative conversation history to the model.  The optional
    ``exp_runtime`` field is forwarded by the scheduler proxy into scheduling
    metadata for experiment-mode predictions.

    Args:
        messages: Cumulative list of message dicts (each with ``role`` and
            ``content`` keys) to send as the conversation history.
        model_id: Model identifier string forwarded in the request body.
        max_tokens: Maximum tokens the model should generate.
        exp_runtime_ms: Optional expected runtime hint in milliseconds passed
            to the scheduler for admission-control decisions.

    Returns:
        Request body dict ready to be JSON-serialised.
    """
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if exp_runtime_ms is not None:
        body["exp_runtime"] = exp_runtime_ms
    return body


def _sync_send_chat(
    scheduler_url: str,
    body: dict,
    timeout: float,
) -> tuple[str, dict | None]:
    """Blocking HTTP POST to the scheduler — executed in a dedicated thread.

    Args:
        scheduler_url: Base URL of the scheduler (without path).
        body: JSON-serialisable request body.
        timeout: HTTP timeout in seconds.

    Returns:
        Tuple of (status, response_data) where status is ``"completed"``
        or ``"failed"`` and response_data is the parsed JSON body or an
        error dict.
    """
    try:
        with httpx.Client() as c:
            resp = c.post(
                f"{scheduler_url}/v1/chat/completions",
                json=body,
                timeout=timeout,
            )
        if resp.status_code == 200:
            return "completed", resp.json()
        else:
            return "failed", {
                "error": resp.text,
                "status_code": resp.status_code,
            }
    except Exception as exc:
        return "failed", {"error": str(exc)}


async def send_chat_request(
    scheduler_url: str,
    body: dict,
    timeout: float = 300.0,
) -> tuple[str, dict | None]:
    """Send a chat completion request in a dedicated thread.

    Each call runs in its own OS thread via the global thread pool, so a
    long-running proxy response never blocks the event loop or other groups.

    Args:
        scheduler_url: Base URL of the scheduler.
        body: JSON-serialisable request body.
        timeout: HTTP timeout in seconds (default 300 s).

    Returns:
        Tuple of (status, response_data) from ``_sync_send_chat``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _REQUEST_POOL,
        _sync_send_chat,
        scheduler_url,
        body,
        timeout,
    )


# ── Per-Group Runner ──────────────────────────────────────────────


def _segment_steps(steps: list[dict]) -> list[list[dict]]:
    """Split steps into segments by model size for burst-mode execution.

    Consecutive small-model steps form a single burst segment; each
    large-model step forms its own single-element segment.

    Example::

        [S, S, S, L, S, S, L, L, S]
        → [[S, S, S], [L], [S, S], [L], [L], [S]]

    Args:
        steps: Ordered list of step dicts, each with a ``model_size`` key.

    Returns:
        List of segments, each a non-empty list of step dicts.
    """
    segments: list[list[dict]] = []
    i = 0
    while i < len(steps):
        if steps[i].get("model_size") == "large":
            segments.append([steps[i]])
            i += 1
        else:
            j = i
            while j < len(steps) and steps[j].get("model_size") != "large":
                j += 1
            segments.append(steps[i:j])
            i = j
    return segments


async def run_group(
    group: dict,
    config: dict,
    bucket_large: TokenBucket,
    bucket_small: TokenBucket,
    progress: dict,
    progress_lock: asyncio.Lock,
    stats: dict[str, Any] | None = None,
) -> dict:
    """Execute one replay group, building cumulative history across steps.

    Starts from ``initial_messages`` and appends each step's
    ``history_message`` to build the cumulative context.  Consecutive
    small-model steps are fired as a concurrent burst (max 6 in-flight);
    large-model steps are always sequential.

    For burst segments, all burst messages are appended to the cumulative
    history BEFORE any request is sent — this models the scenario where
    multiple agent tool calls share the same conversation context snapshot.

    Args:
        group: Raw group dict with ``group_id``, ``initial_messages``,
            ``steps``, and ``dataset_name``.
        config: Benchmark configuration dict.
        bucket_large: Global QPS rate limiter for large-model requests.
        bucket_small: Global QPS rate limiter for small-model requests.
        progress: Shared progress tracking dict (mutated under lock).
        progress_lock: Async lock protecting ``progress``.
        stats: Optional pre-computed dataset statistics for runtime hints.

    Returns:
        Group result dict containing step-level records and aggregate
        metrics.
    """
    group_id = group["group_id"]
    steps = group["steps"]
    strategy = config.get("scheduling_algorithm", "probabilistic")
    chars_per_token: float = config.get("chars_per_token", 4.0)
    max_tokens: int = config.get("max_tokens", 1)
    timeout: float = config.get("timeout_s", 300.0)

    # Cumulative conversation history — grows as steps are appended.
    cumulative_messages: list[dict] = list(group.get("initial_messages", []))

    # Pre-compute per-step estimated runtimes from cumulative char counts.
    step_est_ms: list[float] = []
    preview_chars = sum(len(json.dumps(m)) for m in cumulative_messages)
    for step in steps:
        msg = step.get("history_message", {})
        preview_chars += len(json.dumps(msg))
        tokens = preview_chars / chars_per_token
        step_est_ms.append(
            _estimate_runtime_ms(
                tokens, step.get("model_size", "small"), config
            )
        )

    step_results: list[dict] = []
    group_start = time.monotonic()
    step_index_map = {s["step_index"]: i for i, s in enumerate(steps)}

    async def _exec_step(
        step: dict,
        messages_snapshot: list[dict],
        skip_qps: bool = False,
    ) -> dict:
        """Acquire QPS token, send one request, and return a result record.

        Args:
            step: The step dict being executed.
            messages_snapshot: Cumulative history at the time of sending.
            skip_qps: If True, bypass the rate limiter for this request
                (used for the very first request in a group).

        Returns:
            Step result dict with latency, status, and token counts.
        """
        model_size = step.get("model_size", "small")
        if not skip_qps:
            bucket = bucket_large if model_size == "large" else bucket_small
            await bucket.acquire()

        local_idx = step_index_map.get(step["step_index"], 0)
        est_ms = step_est_ms[local_idx]

        if strategy == "probabilistic":
            # Per-request exact estimate for the probabilistic scheduler.
            exp_runtime_ms = est_ms
        else:
            # Use the dataset-wide mean for deterministic strategies.
            if model_size == "large":
                exp_runtime_ms = stats["large_mean_ms"] if stats else est_ms
            else:
                exp_runtime_ms = stats["small_mean_ms"] if stats else est_ms

        model_id = (
            config.get("large_model_id", "large")
            if model_size == "large"
            else config.get("small_model_id", "small")
        )
        scheduler_url = get_scheduler_url(model_size, config)
        body = build_chat_request(
            messages_snapshot,
            model_id,
            max_tokens=max_tokens,
            exp_runtime_ms=exp_runtime_ms,
        )

        send_time = time.monotonic()
        status, resp_data = await send_chat_request(
            scheduler_url, body, timeout=timeout
        )
        done_time = time.monotonic()

        # Extract token usage from the OpenAI-compatible response.
        usage: dict = {}
        if isinstance(resp_data, dict):
            usage = resp_data.get("usage", {}) or {}

        return {
            "group_id": group_id,
            "step_index": step["step_index"],
            "model_size": model_size,
            "status": status,
            "send_time": send_time,
            "done_time": done_time,
            "e2e_latency_ms": (done_time - send_time) * 1000,
            "est_runtime_ms": est_ms,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }

    segments = _segment_steps(steps)
    # Tracks position in the flat steps list as segments are consumed.
    flat_step_cursor = 0
    is_first_request = True  # First request in the group skips global QPS.

    for seg_idx, segment in enumerate(segments):
        # Inter-segment delay based on the first step's sender role.
        if seg_idx > 0:
            first = segment[0]
            if first.get("sender_role") == "user":
                await asyncio.sleep(config.get("user_delay_ms", 5000) / 1000)
            else:
                await asyncio.sleep(config.get("agent_delay_ms", 100) / 1000)

        if len(segment) == 1:
            # Single step (large model or isolated small) — strictly sequential.
            step = segment[0]
            cumulative_messages.append(step.get("history_message", {}))
            snapshot = list(cumulative_messages)
            flat_step_cursor += 1

            record = await _exec_step(step, snapshot, skip_qps=is_first_request)
            is_first_request = False
            step_results.append(record)
            async with progress_lock:
                progress["completed"] += 1
                _stream_step(record)
                _print_progress(progress, bucket_large, bucket_small)
        else:
            # Burst segment — append ALL messages to cumulative history
            # BEFORE sending so every request in the burst sees the same
            # snapshot (they model concurrent tool calls in one conversation).
            for step in segment:
                cumulative_messages.append(step.get("history_message", {}))
            snapshot = list(cumulative_messages)
            flat_step_cursor += len(segment)

            burst_sem = asyncio.Semaphore(6)
            first_in_group = is_first_request
            is_first_request = False

            async def _exec_burst_step(
                s: dict,
                snap: list[dict],
                skip: bool,
                # Bind burst_sem at definition time to avoid B023 late-binding.
                _sem: asyncio.Semaphore = burst_sem,
            ) -> dict:
                """Wrap ``_exec_step`` with the burst concurrency semaphore.

                Args:
                    s: Step dict.
                    snap: Cumulative message snapshot for this burst.
                    skip: Whether to skip QPS acquisition.
                    _sem: Semaphore bound at closure creation time.

                Returns:
                    Step result dict from ``_exec_step``.
                """
                async with _sem:
                    return await _exec_step(s, snap, skip_qps=skip)

            burst_tasks = []
            for i, s in enumerate(segment):
                # Only the very first request in the group skips QPS.
                skip = first_in_group and i == 0
                burst_tasks.append(
                    asyncio.create_task(_exec_burst_step(s, snapshot, skip))
                )
            burst_results = await asyncio.gather(
                *burst_tasks, return_exceptions=True
            )
            async with progress_lock:
                for r in burst_results:
                    if isinstance(r, dict):
                        step_results.append(r)
                        progress["completed"] += 1
                        _stream_step(r)
                    elif isinstance(r, Exception):
                        progress["completed"] += 1
                _print_progress(progress, bucket_large, bucket_small)

    group_end = time.monotonic()

    return {
        "group_id": group_id,
        "dataset_name": group.get("dataset_name", ""),
        "total_steps": len(steps),
        "completed_steps": sum(
            1 for r in step_results if r["status"] == "completed"
        ),
        "failed_steps": len(steps)
        - sum(1 for r in step_results if r["status"] == "completed"),
        "group_e2e_latency_ms": (group_end - group_start) * 1000,
        "step_results": step_results,
    }


# ── Streaming Output ──────────────────────────────────────────────

_OUTPUT_LARGE_FP: Any = None  # file handle for large-model step JSONL
_OUTPUT_SMALL_FP: Any = None  # file handle for small-model step JSONL
_OUTPUT_E2E_FP: Any = None  # file handle for group E2E JSONL


def _stream_step(record: dict) -> None:
    """Append a completed step record to the appropriate model JSONL file.

    Routes to the large or small file based on ``record["model_size"]``.

    Args:
        record: Step result dict produced by ``_exec_step``.
    """
    fp = (
        _OUTPUT_LARGE_FP
        if record.get("model_size") == "large"
        else _OUTPUT_SMALL_FP
    )
    if fp:
        fp.write(json.dumps(record, default=str) + "\n")
        fp.flush()


def _stream_group(result: dict, phase: str = "measure") -> None:
    """Append a group-level E2E record to the E2E JSONL file.

    Args:
        result: Group result dict produced by ``run_group``.
        phase: Phase label (``"warmup"``, ``"measure"``, or
            ``"background"``).
    """
    if _OUTPUT_E2E_FP:
        record = {
            "group_id": result["group_id"],
            "dataset_name": result.get("dataset_name", ""),
            "total_steps": result["total_steps"],
            "completed_steps": result["completed_steps"],
            "failed_steps": result["failed_steps"],
            "group_e2e_latency_ms": result["group_e2e_latency_ms"],
            "phase": phase,
        }
        _OUTPUT_E2E_FP.write(json.dumps(record, default=str) + "\n")
        _OUTPUT_E2E_FP.flush()


# ── Progress Display ──────────────────────────────────────────────

_PROGRESS_FILE: str | None = None


def _print_progress(
    progress: dict,
    bucket_large: TokenBucket | None = None,
    bucket_small: TokenBucket | None = None,
) -> None:
    """Write a one-line progress indicator to stdout and optionally a file.

    Shows warmup vs. measurement phase, groups done/target, steps
    completed, and rate-limiter backpressure indicators.

    Args:
        progress: Shared progress dict with counters and latency lists.
        bucket_large: Large-model token bucket (to detect backpressure).
        bucket_small: Small-model token bucket (to detect backpressure).
    """
    warmup_done = progress.get("warmup_done", 0)
    warmup_target = progress.get("warmup_target", 0)
    groups_done = progress.get("groups_done", 0)
    groups_target = progress.get("groups_target", 0)
    steps_done = progress["completed"]

    rl_parts = []
    if bucket_large and bucket_large.waiters > 0:
        rl_parts.append("L")
    if bucket_small and bucket_small.waiters > 0:
        rl_parts.append("S")
    rl = f" \033[1;31mRL:{'+'.join(rl_parts)}\033[0m" if rl_parts else ""

    if warmup_target > 0 and warmup_done < warmup_target:
        line = (
            f"\r[\033[0;33mWARMUP {warmup_done}/{warmup_target}\033[0m"
            f" | 0/{groups_target} groups, {steps_done} steps]{rl}   "
        )
    else:
        pct = groups_done / groups_target * 100 if groups_target else 0
        warmup_note = (
            f" \033[0;90m(warmup:{warmup_target})\033[0m"
            if warmup_target > 0
            else ""
        )
        line = (
            f"\r[{groups_done}/{groups_target} groups, {steps_done} steps]"
            f" ({pct:.1f}%){warmup_note}{rl}   "
        )
    sys.stdout.write(line)
    sys.stdout.flush()

    # Write machine-readable progress for external dashboards.
    if _PROGRESS_FILE:
        try:
            data: dict[str, Any] = {
                "warmup_done": warmup_done,
                "warmup_target": warmup_target,
                "groups_done": groups_done,
                "groups_target": groups_target,
                "steps_done": steps_done,
            }
            gl = progress.get("group_latencies")
            if gl:
                s = sorted(gl)
                n = len(s)
                data["group_e2e_p90_ms"] = round(s[min(int(n * 0.9), n - 1)], 1)
            tmp = _PROGRESS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, _PROGRESS_FILE)
        except OSError:
            pass


# ── Orchestration ─────────────────────────────────────────────────

_STRATEGY_ALIASES: dict[str, str] = {
    "probabilistic": "probabilistic",
    "min_time": "min_time",
    "round_robin": "round_robin",
    "random": "random",
    "power_of_two": "po2",
    "po2": "po2",
    "serverless": "serverless",
    "adaptive_bootstrap": "adaptive_bootstrap",
}


async def configure_strategy(
    client: httpx.AsyncClient,
    config: dict,
    strategy: str,
) -> None:
    """Set the scheduling strategy on both large and small schedulers.

    Args:
        client: Shared async HTTP client for strategy configuration calls.
        config: Benchmark config with ``scheduler_large_port`` and
            ``scheduler_small_port``.
        strategy: Strategy name (will be looked up in ``_STRATEGY_ALIASES``).
    """
    strategy_name = _STRATEGY_ALIASES.get(strategy, strategy)
    quantiles = (
        [0.2, 0.4, 0.6, 0.8, 1.0] if strategy_name == "probabilistic" else None
    )
    body: dict[str, Any] = {"strategy_name": strategy_name}
    if quantiles:
        body["quantiles"] = quantiles

    for port_key in ("scheduler_large_port", "scheduler_small_port"):
        port = config.get(port_key)
        url = f"http://localhost:{port}/v1/strategy/set"
        try:
            resp = await client.post(url, json=body, timeout=10.0)
            if resp.status_code == 200:
                print(f"  Strategy set on :{port} -> {strategy_name}")
            else:
                print(
                    f"  [WARN] strategy set on :{port} returned"
                    f" {resp.status_code}: {resp.text}"
                )
        except Exception as exc:
            print(f"  [WARN] strategy set on :{port} failed: {exc}")


async def run_benchmark(
    data_path: str,
    config_path: str,
    output_path: str,
    limit: int | None = None,
    warmup: int = 0,
    progress_file: str | None = None,
) -> None:
    """Run the full real-model benchmark with optional warmup phase.

    Loads replay groups, configures the schedulers, launches all groups
    concurrently with Poisson arrival offsets, streams results to JSONL
    files, and prints a final summary when the measurement target is
    reached.

    Args:
        data_path: Path to the replay JSONL data file.
        config_path: Path to the benchmark YAML configuration file.
        output_path: Base path for output files; creates
            ``{output_path}-large.jsonl``, ``-small.jsonl``, ``-e2e.jsonl``.
        limit: Stop measuring after this many groups complete (remaining
            groups are cancelled).
        warmup: Number of warmup groups to run before measurement starts.
        progress_file: If set, write machine-readable JSON progress here
            for external monitoring.
    """
    global _PROGRESS_FILE
    if progress_file:
        _PROGRESS_FILE = progress_file
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    all_groups = load_replay_groups(data_path)
    if not all_groups:
        print("No replay groups found.")
        return

    strategy = config.get("scheduling_algorithm", "probabilistic")
    measure_target = limit if limit is not None else len(all_groups) - warmup
    submit_count = min(len(all_groups), warmup + measure_target)
    groups = all_groups[:submit_count]
    stats = compute_dataset_stats(all_groups, config)
    total_steps = sum(len(g["steps"]) for g in groups)

    print(
        f"Loaded {len(all_groups)} groups, submitting {submit_count}"
        f" (warmup={warmup} + measure={measure_target})"
    )
    print(f"Strategy: {strategy}")
    print(
        f"Large P90: {stats['large_p90_ms']:.0f}ms"
        f" ({stats['large_count']} requests)"
    )
    print(
        f"Small P90: {stats['small_p90_ms']:.0f}ms"
        f" ({stats['small_count']} requests)"
    )
    print(
        f"Large Mean: {stats['large_mean_ms']:.0f}ms,"
        f" Small Mean: {stats['small_mean_ms']:.0f}ms"
    )
    global_qps = config.get("global_qps", 5.0)
    large_qps = config.get("global_qps_large", global_qps)
    small_qps = config.get("global_qps_small", global_qps)
    print(
        f"Poisson QPS: {config.get('poisson_qps', 0.1)},"
        f" Global QPS: large={large_qps}, small={small_qps}"
    )
    print()

    bucket_large = TokenBucket(large_qps)
    bucket_small = TokenBucket(small_qps)
    progress: dict[str, Any] = {
        "completed": 0,
        "total": total_steps,
        "warmup_done": 0,
        "warmup_target": warmup,
        "groups_done": 0,
        "groups_target": measure_target,
        "warmup_latencies": [],
        "group_latencies": [],
    }

    async with httpx.AsyncClient() as client:
        print("Configuring scheduling strategy...")
        await configure_strategy(client, config, strategy)
        print()

    # Generate Poisson arrival offsets for all groups.
    poisson_qps = config.get("poisson_qps", 0.1)
    offsets = [0.0]
    cumulative = 0.0
    for _ in range(len(groups) - 1):
        cumulative += random.expovariate(poisson_qps)
        offsets.append(cumulative)

    # Open three streaming output files (records all phases).
    # Files are intentionally kept open across the full benchmark run so
    # that results stream continuously; they are closed after all tasks
    # complete.  Context-manager use is intentionally skipped here.
    global _OUTPUT_LARGE_FP, _OUTPUT_SMALL_FP, _OUTPUT_E2E_FP
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_LARGE_FP = open(f"{output_path}-large.jsonl", "w")  # noqa: SIM115
    _OUTPUT_SMALL_FP = open(f"{output_path}-small.jsonl", "w")  # noqa: SIM115
    _OUTPUT_E2E_FP = open(f"{output_path}-e2e.jsonl", "w")  # noqa: SIM115

    # Pre-assign phases by dataset index so all strategies use the same groups.
    #   groups[0..warmup-1]              → warmup
    #   groups[warmup..warmup+measure-1] → measure
    #   groups[warmup+measure..]         → background (runs but not counted)
    group_phase: dict[str, str] = {}
    measure_gids: set[str] = set()
    for i, g in enumerate(groups):
        gid = g["group_id"]
        if i < warmup:
            group_phase[gid] = "warmup"
        elif i < warmup + measure_target:
            group_phase[gid] = "measure"
            measure_gids.add(gid)
        else:
            group_phase[gid] = "background"

    print(
        f"Phase assignment: warmup={warmup} groups [0..{warmup - 1}],"
        f" measure={measure_target} groups"
        f" [{warmup}..{warmup + measure_target - 1}],"
        f" background={len(groups) - warmup - measure_target}"
    )

    progress_lock = asyncio.Lock()
    done_event = asyncio.Event()

    async def _run_with_offset(group: dict, offset: float) -> dict | None:
        """Delay group start by ``offset`` seconds then run the group.

        Args:
            group: Raw group dict.
            offset: Poisson-sampled delay before this group starts.

        Returns:
            Group result dict, or ``None`` if the benchmark already
            finished before this group started.
        """
        await asyncio.sleep(offset)
        if done_event.is_set():
            return None
        result = await run_group(
            group,
            config,
            bucket_large,
            bucket_small,
            progress,
            progress_lock,
            stats=stats,
        )
        gid = group["group_id"]
        phase = group_phase[gid]
        result["_phase"] = phase
        async with progress_lock:
            if phase == "warmup":
                progress["warmup_done"] += 1
                progress["warmup_latencies"].append(
                    result["group_e2e_latency_ms"]
                )
                _stream_group(result, phase="warmup")
                if progress["warmup_done"] == warmup:
                    print(
                        f"\n\033[0;33mWarmup complete ({warmup} groups)."
                        f" Starting measurement...\033[0m"
                    )
            elif phase == "measure":
                progress["groups_done"] += 1
                progress["group_latencies"].append(
                    result["group_e2e_latency_ms"]
                )
                _stream_group(result, phase="measure")
                if progress["groups_done"] >= measure_target:
                    done_event.set()
            else:
                # Background groups: stream but do not count toward target.
                _stream_group(result, phase="background")
            _print_progress(progress, bucket_large, bucket_small)
        return result

    tasks = [
        asyncio.create_task(_run_with_offset(g, o))
        for g, o in zip(groups, offsets)
    ]

    await done_event.wait()

    for t in tasks:
        if not t.done():
            t.cancel()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate warmup and measurement results.
    warmup_metrics: list[dict] = []
    measure_metrics: list[dict] = []
    for r in raw_results:
        if isinstance(r, dict) and r is not None:
            if r.get("_phase") == "warmup":
                warmup_metrics.append(r)
            elif r.get("_phase") == "measure":
                measure_metrics.append(r)

    # Write per-file summaries as the final JSONL line (measurement only).
    large_lats = [
        r["e2e_latency_ms"]
        for gm in measure_metrics
        for r in gm["step_results"]
        if r["status"] == "completed" and r.get("model_size") == "large"
    ]
    small_lats = [
        r["e2e_latency_ms"]
        for gm in measure_metrics
        for r in gm["step_results"]
        if r["status"] == "completed" and r.get("model_size") != "large"
    ]
    group_lats = [gm["group_e2e_latency_ms"] for gm in measure_metrics]

    for fp, lats, label in [
        (_OUTPUT_LARGE_FP, large_lats, "large"),
        (_OUTPUT_SMALL_FP, small_lats, "small"),
        (_OUTPUT_E2E_FP, group_lats, "e2e"),
    ]:
        summary = {
            "_type": "summary",
            "label": label,
            "phase": "measure",
            "count": len(lats),
            "latency_ms": _percentile_stats(lats),
        }
        fp.write(json.dumps(summary, default=str) + "\n")
        fp.close()
    _OUTPUT_LARGE_FP = _OUTPUT_SMALL_FP = _OUTPUT_E2E_FP = None

    # Print human-readable summaries.
    print()
    if warmup_metrics:
        print("=" * 60)
        print(f"WARMUP SUMMARY ({len(warmup_metrics)} groups)")
        print("=" * 60)
        _print_summary(warmup_metrics)
        print()
    print("=" * 60)
    print(f"MEASUREMENT SUMMARY ({len(measure_metrics)} groups)")
    print("=" * 60)
    _print_summary(measure_metrics)
    cancelled = sum(1 for t in tasks if t.cancelled())
    print(
        f"\n{len(warmup_metrics)} warmup + {len(measure_metrics)} measured"
        f" + {cancelled} cancelled"
    )
    print(f"Results streamed to {output_path}-{{large,small,e2e}}.jsonl")


# ── Reporting ─────────────────────────────────────────────────────


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute percentile and summary statistics for a list of values.

    Args:
        values: Numeric values to summarise (e.g. latencies in ms).

    Returns:
        Dict with keys ``p50``, ``p90``, ``p99``, ``mean``, ``min``,
        ``max`` (all floats, zero for an empty list).
    """
    if not values:
        return {
            "p50": 0,
            "p90": 0,
            "p99": 0,
            "mean": 0,
            "min": 0,
            "max": 0,
        }
    s = sorted(values)
    n = len(s)
    return {
        "p50": s[int(n * 0.5)],
        "p90": s[min(int(n * 0.9), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "mean": round(sum(s) / n, 2),
        "min": s[0],
        "max": s[-1],
    }


def _print_summary(group_metrics: list[dict]) -> None:
    """Print a formatted latency summary table for a set of group results.

    Breaks down latency by all requests, large-model requests, small-model
    requests, and group end-to-end time.

    Args:
        group_metrics: List of group result dicts from ``run_group``.
    """
    all_latencies = [
        r["e2e_latency_ms"]
        for gm in group_metrics
        for r in gm["step_results"]
        if r["status"] == "completed"
    ]
    large_latencies = [
        r["e2e_latency_ms"]
        for gm in group_metrics
        for r in gm["step_results"]
        if r["status"] == "completed" and r.get("model_size") == "large"
    ]
    small_latencies = [
        r["e2e_latency_ms"]
        for gm in group_metrics
        for r in gm["step_results"]
        if r["status"] == "completed" and r.get("model_size") != "large"
    ]
    group_latencies = [gm["group_e2e_latency_ms"] for gm in group_metrics]

    total_reqs = sum(gm["total_steps"] for gm in group_metrics)
    ok_reqs = sum(gm["completed_steps"] for gm in group_metrics)
    fail_reqs = sum(gm["failed_steps"] for gm in group_metrics)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Groups: {len(group_metrics)}")
    print(
        f"Requests: {total_reqs} total, {ok_reqs} success, {fail_reqs} failed"
    )

    def _row(label: str, vals: list[float]) -> None:
        """Print one formatted latency row.

        Args:
            label: Human-readable row label.
            vals: Latency values in milliseconds.
        """
        st = _percentile_stats(vals)
        if not vals:
            print(f"  {label}: (no data)")
            return
        print(
            f"  {label}: "
            f"p50={st['p50']:.0f}ms p90={st['p90']:.0f}ms"
            f" p99={st['p99']:.0f}ms mean={st['mean']:.0f}ms"
            f" min={st['min']:.0f}ms max={st['max']:.0f}ms"
        )

    print()
    _row("All requests", all_latencies)
    _row("Large model", large_latencies)
    _row("Small model", small_latencies)
    _row("Group e2e", group_latencies)


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the real-model benchmark runner CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "SwarmPilot real-model benchmark runner — replay conversation"
            " datasets through a live scheduler cluster"
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to replay JSONL file",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--output",
        default="./results",
        help=(
            "Output base path (creates {base}-large.jsonl,"
            " -small.jsonl, -e2e.jsonl)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many groups complete",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup groups before measurement starts",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Write JSON progress for external monitoring",
    )
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            args.data,
            args.config,
            args.output,
            args.limit,
            args.warmup,
            args.progress_file,
        )
    )


if __name__ == "__main__":
    main()
