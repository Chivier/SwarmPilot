#!/usr/bin/env python3
"""
Workload helpers for Experiment 03 (Text2Video A-A-B workflow).

Responsibilities:
1) Sampling captions from nkp37/OpenVid-1M (streaming) or a local cache file
2) Generating four-peak frame count distribution for video generation
3) Lightweight summary helpers for logging/debugging
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

DataPath = Path | str

# Defaults for caption sampling
DEFAULT_DATASET = "nkp37/OpenVid-1M"
DEFAULT_SPLIT = "train"
DEFAULT_STREAM_LIMIT = 20000  # Limit streaming rows to keep local runs reasonable
DEFAULT_CAPTIONS_FILE = Path(__file__).parent / "captions_10k.json"


# ============================================================================
# Caption Sampling
# ============================================================================

def _extract_caption(entry: dict) -> str | None:
    """Best-effort extraction for common caption fields."""
    for key in ("caption", "text", "description", "prompt"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Some datasets expose a list of captions
    if "captions" in entry and isinstance(entry["captions"], list):
        for item in entry["captions"]:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _read_local_captions(path: Path) -> List[str]:
    """
    Load captions from a JSONL/JSON/TXT file.

    - JSONL: each line is {"caption": "..."} or {"text": "..."}
    - JSON:  list of strings or list of {"caption": "..."}
    - TXT:   one caption per line
    """
    captions: List[str] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    caption = _extract_caption(obj) if isinstance(obj, dict) else None
                    if caption:
                        captions.append(caption)
                except json.JSONDecodeError:
                    captions.append(line)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        captions.append(item)
                    elif isinstance(item, dict):
                        cap = _extract_caption(item)
                        if cap:
                            captions.append(cap)
    else:
        with path.open("r", encoding="utf-8") as f:
            captions.extend([line.strip() for line in f if line.strip()])

    if not captions:
        raise ValueError(f"No captions found in {path}")
    return captions


def save_captions(captions: Sequence[str], path: DataPath) -> None:
    """Persist sampled captions to a UTF-8 JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for caption in captions:
            f.write(json.dumps({"caption": caption}, ensure_ascii=False) + "\n")


def sample_captions(
    num_captions: int,
    seed: int = 42,
    *,
    cache_path: DataPath | None = None,
    dataset_name: str = DEFAULT_DATASET,
    dataset_split: str = DEFAULT_SPLIT,
    stream_limit: int = DEFAULT_STREAM_LIMIT,
) -> Tuple[List[str], dict]:
    """
    Sample captions for all workflows.
    
    Reads from local JSON file (default: captions_10k.json) or provided cache_path.
    HuggingFace dataset loading has been replaced by local file loading.

    Returns:
        captions: List[str] length == num_captions
        meta: {source, path, total_available}
    """
    rng = random.Random(seed)

    # Determine source path
    if cache_path:
        source_path = Path(cache_path)
    else:
        source_path = DEFAULT_CAPTIONS_FILE

    if not source_path.exists():
        raise FileNotFoundError(
            f"Captions file not found at {source_path}. "
            "Please ensure experiments/03.Exp4.Text2Video/captions_10k.json exists."
        )

    # Read all captions
    all_captions = _read_local_captions(source_path)
    
    # Shuffle and sample
    rng.shuffle(all_captions)
    
    if len(all_captions) >= num_captions:
        selected_captions = all_captions[:num_captions]
    else:
        # Sample with replacement if we need more than available
        selected_captions = [rng.choice(all_captions) for _ in range(num_captions)]

    return selected_captions, {
        "source": "local_file",
        "path": str(source_path),
        "total_available": len(all_captions),
        "dataset": dataset_name, # Preserved for interface compatibility
        "split": dataset_split,
    }


# ============================================================================
# Frame Count Generation (Four-Peak)
# ============================================================================

@dataclass
class FrameDistribution:
    peaks: Tuple[int, int, int, int]
    std: float
    weights: Tuple[float, float, float, float]
    min_frames: int
    max_frames: int


DEFAULT_FRAMES = FrameDistribution(
    peaks=(16, 24, 32, 48),
    std=2.5,
    weights=(5/6, 1/18, 1/18, 1/18),  # 5:1 ratio - first peak dominates (83.3% vs 16.7% total)
    min_frames=8,
    max_frames=72,
)


def generate_four_peak_frames(
    num_items: int,
    seed: int = 42,
    dist: FrameDistribution = DEFAULT_FRAMES,
) -> List[int]:
    """Generate frame counts from a four-peak Gaussian mixture."""
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(dist.peaks), size=num_items, p=list(dist.weights))

    frames: List[int] = []
    for idx in choices:
        value = rng.normal(loc=dist.peaks[idx], scale=dist.std)
        value = int(round(np.clip(value, dist.min_frames, dist.max_frames)))
        frames.append(value)
    return frames


# ============================================================================
# Summaries
# ============================================================================

def summarize_captions(captions: Sequence[str]) -> dict:
    lengths = [len(c) for c in captions]
    return {
        "count": len(captions),
        "min_len": int(np.min(lengths)) if lengths else 0,
        "max_len": int(np.max(lengths)) if lengths else 0,
        "mean_len": float(np.mean(lengths)) if lengths else 0.0,
        "median_len": float(np.median(lengths)) if lengths else 0.0,
    }


def summarize_frames(frames: Sequence[int]) -> dict:
    arr = np.array(frames)
    return {
        "count": len(frames),
        "min": int(arr.min()) if len(arr) else 0,
        "max": int(arr.max()) if len(arr) else 0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
    }


def summarize_durations(durations: Sequence[float]) -> dict:
    arr = np.array(durations)
    return {
        "count": len(arr),
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p95": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
    }


# ============================================================================
# Simulation duration generators
# ============================================================================

def generate_long_tail_durations(
    num_items: int,
    mean: float = 5.0,
    alpha: float = 1.5,
    seed: int = 123,
    min_time: float = 0.5,
    max_time: float = 60.0,
) -> List[float]:
    """
    Generate long-tail (Pareto-like) durations with target mean.
    """
    rng = np.random.default_rng(seed)
    raw = rng.pareto(alpha, num_items) + 1.0  # mean = alpha/(alpha-1)
    # scale to target mean
    scaled = raw * (mean / raw.mean())
    clipped = np.clip(scaled, min_time, max_time)
    return clipped.tolist()


def generate_four_peak_durations(
    num_items: int,
    peaks: Tuple[float, float, float, float] = (15.0, 30.0, 60.0, 120.0),
    stds: Tuple[float, float, float, float] = (3.0, 4.0, 8.0, 15.0),
    weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
    seed: int = 124,
    min_time: float = 5.0,
    max_time: float = 200.0,
) -> List[float]:
    """Four-peak Gaussian mixture for B stage (simulation)."""
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(peaks), size=num_items, p=list(weights))
    durations = []
    for idx in choices:
        val = rng.normal(peaks[idx], stds[idx])
        durations.append(float(np.clip(val, min_time, max_time)))
    return durations


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Test caption sampling and frame generation")
    parser.add_argument("--num-captions", type=int, default=8, help="Number of captions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stream-limit", type=int, default=2000, help="Max rows to stream from dataset")
    parser.add_argument("--cache-path", type=Path, default=None, help="Optional JSONL cache for captions")
    args = parser.parse_args()

    frames = generate_four_peak_frames(args.num_captions, seed=args.seed)
    print("Frames:", frames)
    print("Frame stats:", summarize_frames(frames))

    try:
        caps, meta = sample_captions(
            args.num_captions,
            seed=args.seed,
            cache_path=args.cache_path,
            stream_limit=args.stream_limit,
        )
        print("Sampled captions meta:", meta)
        print("Caption stats:", summarize_captions(caps))
    except Exception as exc:
        print(f"Failed to sample captions: {exc}")

# ============================================================================
# Workload Data Structures (matching exp07 pattern)
# ============================================================================

@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    num_workflows: int
    seed: int
    caption_source: str
    frame_distribution: str
    description: str


@dataclass
class Text2VideoWorkload:
    """
    Complete workload for Text2Video experiment.
    
    This matches the WorkflowWorkload pattern from exp07.
    """
    name: str
    captions: List[str]
    frame_counts: List[int]
    description: str


def generate_text2video_workload(
    num_workflows: int,
    seed: int = 42,
    cache_path: DataPath | None = None,
    dataset_name: str = DEFAULT_DATASET,
    dataset_split: str = DEFAULT_SPLIT,
    stream_limit: int = DEFAULT_STREAM_LIMIT
) -> Tuple[Text2VideoWorkload, WorkloadConfig]:
    """
    Generate complete Text2Video workload matching exp07 pattern.
    
    Args:
        num_workflows: Number of workflows to generate
        seed: Random seed for reproducibility
        cache_path: Optional path to cached captions
        dataset_name: HuggingFace dataset name
        dataset_split: Dataset split to use
        stream_limit: Maximum rows to stream from dataset
    
    Returns:
        Tuple of (Text2VideoWorkload, WorkloadConfig)
    """
    # Sample captions
    captions, caption_meta = sample_captions(
        num_captions=num_workflows,
        seed=seed,
        cache_path=cache_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        stream_limit=stream_limit
    )
    
    # Generate frame counts
    frame_counts = generate_four_peak_frames(
        num_items=num_workflows,
        seed=seed
    )
    
    # Create workload
    workload = Text2VideoWorkload(
        name="text2video_workload",
        captions=captions,
        frame_counts=frame_counts,
        description=f"Text2Video workload with {num_workflows} workflows"
    )
    
    # Create config
    config = WorkloadConfig(
        num_workflows=num_workflows,
        seed=seed,
        caption_source=caption_meta.get("source", "unknown"),
        frame_distribution="four_peak",
        description=f"Text2Video workload generated with seed {seed}"
    )
    
    return workload, config


def print_text2video_stats(workload: Text2VideoWorkload):
    """
    Print statistics for Text2Video workload matching exp07 pattern.
    
    Args:
        workload: Text2VideoWorkload to analyze
    """
    print("\n" + "=" * 80)
    print("Text2Video Workload Statistics")
    print("=" * 80)
    
    # Caption statistics
    caption_stats = summarize_captions(workload.captions)
    print(f"\nCaptions ({caption_stats['count']} total):")
    print(f"  Length: min={caption_stats['min_len']}, max={caption_stats['max_len']}, "
          f"mean={caption_stats['mean_len']:.1f}, median={caption_stats['median_len']:.1f}")
    
    # Frame count statistics
    frame_stats = summarize_frames(workload.frame_counts)
    print(f"\nFrame Counts ({frame_stats['count']} total):")
    print(f"  Range: min={frame_stats['min']}, max={frame_stats['max']}, "
          f"mean={frame_stats['mean']:.1f}")
    print(f"  Percentiles: p50={frame_stats['p50']:.1f}, p90={frame_stats['p90']:.1f}, "
          f"p99={frame_stats['p99']:.1f}")
    
    # Frame count distribution
    frame_array = np.array(workload.frame_counts)
    print(f"\nFrame Count Distribution (5:1 ratio - first peak dominates):")
    for i, peak in enumerate([30, 50, 100, 200]):
        count = np.sum((frame_array >= peak - 4) & (frame_array <= peak + 4))
        percentage = 100.0 * count / len(frame_array)
        peak_label = f"Peak {i+1} ({peak} frames)"
        if i == 0:
            peak_label += " [DOMINANT]"
        print(f"  {peak_label}: {count} ({percentage:.1f}%)")

    print("=" * 80)


# ============================================================================
# Sleep Model Time Generation (for simulation)
# ============================================================================

def generate_long_tail_with_mean(
    num_tasks: int,
    target_mean: float = 8.0,
    seed: int = 42
) -> List[float]:
    """
    Generate task execution times from a log-normal (long-tail) distribution
    with a specified target mean.

    The log-normal distribution creates a long-tail effect where:
    - Most tasks complete around the mean
    - A small percentage takes significantly longer (long tail)
    - The distribution is right-skewed

    Args:
        num_tasks: Number of tasks to generate
        target_mean: Target mean execution time in seconds
        seed: Random seed for reproducibility

    Returns:
        List of task execution times in seconds
    """
    rng = np.random.default_rng(seed)

    # For log-normal distribution: mean = exp(mu + sigma^2/2)
    # We want to achieve target_mean, so we solve for mu given sigma
    # Using sigma = 0.6 gives a reasonable long-tail shape
    sigma = 0.6
    mu = np.log(target_mean) - (sigma**2) / 2

    # Generate log-normal samples
    times = rng.lognormal(mu, sigma, num_tasks)

    # Scale to achieve exact target mean
    actual_mean = np.mean(times)
    times = times * (target_mean / actual_mean)

    # Shuffle to randomize order
    rng.shuffle(times)

    return times.tolist()


def generate_random_peak_weights(
    seed: int,
    first_peak_ratio: float = 5/6,
    num_peaks: int = 4
) -> Tuple[float, ...]:
    """
    Generate random weights for four-peak distribution with first peak dominating.

    Args:
        seed: Random seed for reproducibility
        first_peak_ratio: Ratio for the first peak (default: 5/6 ≈ 0.833)
        num_peaks: Total number of peaks (default: 4)

    Returns:
        Tuple of weights that sum to 1.0
    """
    rng = np.random.default_rng(seed)

    # Remaining ratio to distribute among other peaks
    remaining_ratio = 1.0 - first_peak_ratio

    # Generate random weights for remaining peaks (num_peaks - 1)
    # Use Dirichlet distribution to ensure they sum to remaining_ratio
    num_remaining = num_peaks - 1
    random_weights = rng.dirichlet(np.ones(num_remaining)) * remaining_ratio

    # Combine first peak ratio with random weights
    weights = [first_peak_ratio] + random_weights.tolist()

    return tuple(weights)


def generate_four_peak_distribution(
    num_tasks: int,
    peaks: Tuple[float, float, float, float] = (15.0, 30.0, 60.0, 120.0),
    peak_ratios: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    std_factor: float = 0.1,
    seed: int = 42
) -> List[float]:
    """
    Generate task execution times from a four-peak normal distribution.

    Each peak is a normal distribution centered at the specified value.
    Tasks are randomly assigned to one of the four peaks based on peak_ratios.

    Args:
        num_tasks: Number of tasks to generate
        peaks: Tuple of 4 peak centers (mean values for each Gaussian)
        peak_ratios: Tuple of 4 ratios (must sum to 1.0) for each peak
        std_factor: Standard deviation factor (std = peak_center * std_factor)
        seed: Random seed for reproducibility

    Returns:
        List of task execution times in seconds
    """
    rng = np.random.default_rng(seed)

    if len(peaks) != 4 or len(peak_ratios) != 4:
        raise ValueError("Must provide exactly 4 peaks and 4 peak_ratios")

    if not np.isclose(sum(peak_ratios), 1.0):
        raise ValueError(f"peak_ratios must sum to 1.0, got {sum(peak_ratios)}")

    times = []

    # Generate samples for each peak
    for peak_center, ratio in zip(peaks, peak_ratios):
        n_samples = int(num_tasks * ratio)
        std_dev = peak_center * std_factor

        # Generate normal distribution around this peak
        peak_times = rng.normal(peak_center, std_dev, n_samples)

        # Clip to ensure positive values
        peak_times = np.clip(peak_times, 0.1, None)

        times.extend(peak_times)

    # Handle rounding errors (if total < num_tasks)
    while len(times) < num_tasks:
        # Add one more sample from the last peak
        std_dev = peaks[-1] * std_factor
        extra = rng.normal(peaks[-1], std_dev, 1)[0]
        times.append(max(0.1, extra))

    # Convert to array and shuffle
    times_array = np.array(times[:num_tasks])
    rng.shuffle(times_array)

    return times_array.tolist()


@dataclass
class SleepModelWorkload:
    """
    Complete workload for sleep model simulation.

    A1 → A2 → B workflow:
    - A1: Generate positive prompt (long-tail, mean=8s)
    - A2: Generate negative prompt (long-tail, mean=8s)
    - B: Generate video (four-peak: 15/30/60/120s)
    """
    name: str
    a1_times: List[float]  # A1 task execution times
    a2_times: List[float]  # A2 task execution times
    b_times: List[float]   # B task execution times
    frame_counts: List[int]  # Frame counts for B tasks
    description: str


def generate_sleep_model_workload(
    num_workflows: int,
    seed: int = 42
) -> Tuple[SleepModelWorkload, WorkloadConfig]:
    """
    Generate complete sleep model workload for Text2Video simulation.

    Process:
    1. Generate A1 task times: Long-tail distribution with mean=8s
    2. Generate A2 task times: Long-tail distribution with mean=8s
    3. Generate B task times: Four-peak distribution (15s, 30s, 60s, 120s)
    4. Generate frame counts: Four-peak frame distribution (for metadata)

    Args:
        num_workflows: Number of workflows to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (SleepModelWorkload, WorkloadConfig)
    """
    # Generate execution times
    a1_times = generate_long_tail_with_mean(num_workflows, target_mean=8.0, seed=seed)
    a2_times = generate_long_tail_with_mean(num_workflows, target_mean=8.0, seed=seed + 1)

    # Generate random peak weights with first peak dominating (5:1 ratio)
    peak_weights = generate_random_peak_weights(seed=seed + 2, first_peak_ratio=5/6)

    b_times = generate_four_peak_distribution(
        num_workflows,
        peaks=(15.0, 30.0, 60.0, 120.0),
        peak_ratios=peak_weights,
        seed=seed + 2
    )

    # Generate frame counts for metadata
    frame_counts = generate_four_peak_frames(num_workflows, seed=seed)

    # Create workload
    workload = SleepModelWorkload(
        name="sleep_model_text2video",
        a1_times=a1_times,
        a2_times=a2_times,
        b_times=b_times,
        frame_counts=frame_counts,
        description=f"Sleep model Text2Video: {num_workflows} workflows, "
                    f"A1 long-tail (mean=8s), A2 long-tail (mean=8s), "
                    f"B four-peak (15/30/60/120s, 5:1 ratio, weights={[f'{w:.4f}' for w in peak_weights]})"
    )

    # Create config
    config = WorkloadConfig(
        num_workflows=num_workflows,
        seed=seed,
        caption_source="sleep_model",
        frame_distribution="four_peak",
        description=f"Sleep model simulation with seed {seed}"
    )

    return workload, config


def print_sleep_model_stats(workload: SleepModelWorkload):
    """
    Print statistics for sleep model workload.

    Args:
        workload: SleepModelWorkload to analyze
    """
    print("\n" + "=" * 80)
    print("Sleep Model Workload Statistics")
    print("=" * 80)

    # A1 task statistics
    a1_stats = summarize_durations(workload.a1_times)
    print(f"\nA1 Tasks (Generate Positive Prompt) - {a1_stats['count']} total:")
    print(f"  Mean: {a1_stats['mean']:.2f}s (target: 8.00s)")
    print(f"  Range: [{a1_stats['min']:.2f}s, {a1_stats['max']:.2f}s]")
    print(f"  Percentiles: p50={a1_stats['p50']:.2f}s, p90={a1_stats['p90']:.2f}s, "
          f"p95={a1_stats['p95']:.2f}s, p99={a1_stats['p99']:.2f}s")

    # A2 task statistics
    a2_stats = summarize_durations(workload.a2_times)
    print(f"\nA2 Tasks (Generate Negative Prompt) - {a2_stats['count']} total:")
    print(f"  Mean: {a2_stats['mean']:.2f}s (target: 8.00s)")
    print(f"  Range: [{a2_stats['min']:.2f}s, {a2_stats['max']:.2f}s]")
    print(f"  Percentiles: p50={a2_stats['p50']:.2f}s, p90={a2_stats['p90']:.2f}s, "
          f"p95={a2_stats['p95']:.2f}s, p99={a2_stats['p99']:.2f}s")

    # B task statistics
    b_stats = summarize_durations(workload.b_times)
    print(f"\nB Tasks (Generate Video) - {b_stats['count']} total:")
    print(f"  Mean: {b_stats['mean']:.2f}s")
    print(f"  Range: [{b_stats['min']:.2f}s, {b_stats['max']:.2f}s]")
    print(f"  Percentiles: p50={b_stats['p50']:.2f}s, p90={b_stats['p90']:.2f}s, "
          f"p95={b_stats['p95']:.2f}s, p99={b_stats['p99']:.2f}s")

    # Peak distribution
    b_array = np.array(workload.b_times)
    print(f"\nB Task Time Distribution (5:1 ratio - first peak dominates):")
    for i, peak in enumerate([15.0, 30.0, 60.0, 120.0]):
        window = peak * 0.2  # 20% window around peak
        count = np.sum((b_array >= peak - window) & (b_array <= peak + window))
        percentage = 100.0 * count / len(b_array)
        peak_label = f"Peak {i+1} (~{peak:.0f}s)"
        if i == 0:
            peak_label += " [DOMINANT]"
        print(f"  {peak_label}: {count} ({percentage:.1f}%)")

    # Frame count statistics
    frame_stats = summarize_frames(workload.frame_counts)
    print(f"\nFrame Counts (metadata) - {frame_stats['count']} total:")
    print(f"  Range: min={frame_stats['min']}, max={frame_stats['max']}, "
          f"mean={frame_stats['mean']:.1f}")
    print(f"  Percentiles: p50={frame_stats['p50']:.1f}, p90={frame_stats['p90']:.1f}, "
          f"p99={frame_stats['p99']:.1f}")

    print("=" * 80)
