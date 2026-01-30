"""Shared utilities for Library API examples.

Provides common functions for data generation, storage management,
and result formatting used across all examples.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from src.models import PlatformInfo


# =============================================================================
# Data Generation
# =============================================================================


def generate_training_data(
    n_samples: int = 25,
    base_runtime: float = 100.0,
    noise_factor: float = 0.1,
) -> list[dict[str, Any]]:
    """Generate ML inference training data with realistic correlation.

    Creates training samples simulating ML inference workloads where runtime
    depends on batch_size, sequence_length, and hidden_size.

    Runtime formula:
        runtime = base + (batch * 0.5 + seq * 0.1 + hidden * 0.05) + noise

    Args:
        n_samples: Number of samples to generate.
        base_runtime: Base runtime in milliseconds.
        noise_factor: Noise factor as fraction of computed runtime.

    Returns:
        List of feature dictionaries with runtime_ms field.

    Example:
        >>> data = generate_training_data(25)
        >>> data[0]  # doctest: +SKIP
        {'batch_size': 16, ..., 'runtime_ms': 142.5}
    """
    random.seed(42)  # Reproducible results
    data = []

    batch_sizes = [8, 16, 32, 64, 128]
    sequence_lengths = [128, 256, 512, 1024, 2048]
    hidden_sizes = [256, 512, 768, 1024]

    for _ in range(n_samples):
        batch = random.choice(batch_sizes)
        seq = random.choice(sequence_lengths)
        hidden = random.choice(hidden_sizes)

        # Compute runtime with realistic correlation
        computed = base_runtime + (batch * 0.5 + seq * 0.1 + hidden * 0.05)
        noise = random.uniform(-noise_factor, noise_factor) * computed
        runtime = computed + noise

        data.append(
            {
                "batch_size": batch,
                "sequence_length": seq,
                "hidden_size": hidden,
                "runtime_ms": round(runtime, 2),
            }
        )

    return data


def get_platform_info() -> PlatformInfo:
    """Get standard platform info for examples.

    Returns:
        PlatformInfo for PyTorch 2.0 on NVIDIA A100.
    """
    return PlatformInfo(
        software_name="PyTorch",
        software_version="2.0",
        hardware_name="NVIDIA A100",
    )


# =============================================================================
# Storage Management
# =============================================================================


def get_examples_storage_dir() -> Path:
    """Get the base storage directory for examples.

    Returns:
        Path to examples storage directory.
    """
    return Path("/tmp/predictor_examples")


def setup_storage(example_name: str) -> Path:
    """Create isolated storage directory for an example.

    Creates a fresh directory for the example, removing any existing data.

    Args:
        example_name: Name of the example (used as subdirectory).

    Returns:
        Path to the storage directory.
    """
    storage_dir = get_examples_storage_dir() / example_name
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


def cleanup_storage(storage_path: Path) -> None:
    """Remove example storage directory.

    Args:
        storage_path: Path to storage directory to remove.
    """
    if storage_path.exists():
        shutil.rmtree(storage_path)


def cleanup_all_examples() -> None:
    """Remove all example storage directories."""
    base_dir = get_examples_storage_dir()
    if base_dir.exists():
        shutil.rmtree(base_dir)


# =============================================================================
# Result Formatting
# =============================================================================


def print_section(title: str) -> None:
    """Print a section header.

    Args:
        title: Section title.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_subsection(title: str) -> None:
    """Print a subsection header.

    Args:
        title: Subsection title.
    """
    print(f"\n{title}")
    print("-" * 40)


def print_result(title: str, data: Any, indent: int = 2) -> None:
    """Pretty-print a result with formatting.

    Args:
        title: Result title.
        data: Data to print (dict, list, or other).
        indent: Indentation level.
    """
    print(f"\n{title}:")
    prefix = " " * indent

    if hasattr(data, "__dict__"):
        # Object with attributes
        for key, value in vars(data).items():
            if not key.startswith("_"):
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print(f"{prefix}  {json.dumps(value, indent=2)}")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                print(f"{prefix}[{i}] {json.dumps(item)}")
            else:
                print(f"{prefix}[{i}] {item}")
    else:
        print(f"{prefix}{data}")


def print_features_summary(features_list: list[dict[str, Any]]) -> None:
    """Print summary of training features.

    Args:
        features_list: List of feature dictionaries.
    """
    if not features_list:
        print("  No features")
        return

    sample = features_list[0]
    feature_names = [k for k in sample.keys() if k != "runtime_ms"]
    print(f"  Samples: {len(features_list)}")
    print(f"  Features: {', '.join(feature_names)}")
    if "runtime_ms" in sample:
        runtimes = [f["runtime_ms"] for f in features_list]
        print(f"  Runtime range: {min(runtimes):.1f} - {max(runtimes):.1f} ms")


def print_chain_info(chain) -> None:
    """Print information about a preprocessor chain.

    Args:
        chain: PreprocessorChainV2 instance.
    """
    if chain is None:
        print("  Chain: None (no preprocessing)")
        return

    print(f"  Chain: {chain.name}")
    print(f"  Steps: {len(chain.preprocessors)}")
    for i, prep in enumerate(chain.preprocessors):
        print(f"    [{i + 1}] {prep.name}")
        print(f"        Inputs: {prep.input_features}")
        print(f"        Outputs: {prep.output_features}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_core_with_storage(storage_dir: Path):
    """Create PredictorCore with specified storage directory.

    Args:
        storage_dir: Path to storage directory.

    Returns:
        PredictorCore instance configured for the storage.
    """
    from src.api.core import PredictorCore

    # Set environment variable for storage
    os.environ["PREDICTOR_STORAGE_DIR"] = str(storage_dir)

    return PredictorCore(storage_dir=str(storage_dir))


def create_low_level_with_storage(storage_dir: Path):
    """Create PredictorLowLevel with specified storage directory.

    Args:
        storage_dir: Path to storage directory.

    Returns:
        PredictorLowLevel instance configured for the storage.
    """
    from src.api.core import PredictorLowLevel

    return PredictorLowLevel(storage_dir=str(storage_dir))
