#!/usr/bin/env python3
r"""Systematic MLP training sweep over architectures and dataset sizes.

Loads runtime data collected by profile_models.py, then trains
QuantilePredictor MLPs with varying hidden_layers configurations
and training set sizes. Evaluates on a fixed held-out test set
and records all metrics for analysis.

Sweep dimensions:
    - hidden_layers: [32], [64,32], [128,64], [256,128],
                     [128,64,32], [256,128,64]
    - training_size: 100, 200, 300, 400, 500
    - test_size: 100 (fixed hold-out)

Output:
    sweep_results.json -- structured results with all metrics
    per-config model files saved to storage_dir

Usage:
    # Run full sweep on large model data
    uv run python training_sweep.py \
        --data runtime_qwen80b.json \
        --output sweep_results_80b.json

    # Run sweep on small model data
    uv run python training_sweep.py \
        --data runtime_qwen8b.json \
        --output sweep_results_8b.json

    # Custom architectures and sizes
    uv run python training_sweep.py \
        --data runtime_qwen80b.json \
        --hidden-layers "64,32" "128,64" "256,128,64" \
        --train-sizes 100 200 300 \
        --test-size 100
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_HIDDEN_LAYERS = [
    # ── Width sweep (single layer) ──
    [16],
    [32],
    [64],
    [128],
    # ── Two-layer sweep ──
    [32, 16],
    [64, 32],
    [128, 64],
    [256, 128],
    [512, 256],
    # ── Three-layer sweep ──
    [64, 32, 16],
    [128, 64, 32],
    [256, 128, 64],
    [512, 256, 128],
    # ── Four-layer (deep) ──
    [128, 64, 32, 16],
    [256, 128, 64, 32],
]

DEFAULT_TRAIN_SIZES = [100, 200, 300, 400, 500]
DEFAULT_TEST_SIZE = 100
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 42


# ── Data Loading ─────────────────────────────────────────────────


def load_runtime_data(path: str) -> tuple[dict, list[dict]]:
    """Load runtime data from JSON file.

    Args:
        path: Path to the JSON file from profile_models.py.

    Returns:
        Tuple of (metadata, features_list).
    """
    with open(path) as f:
        data = json.load(f)
    metadata = data["metadata"]
    features_list = data["features_list"]

    # Strip metadata fields (prefixed with _).
    clean = [
        {k: v for k, v in s.items() if not k.startswith("_")} for s in features_list
    ]
    logger.info(
        f"Loaded {len(clean)} samples from {path} " f"(model: {metadata['model_id']})"
    )
    return metadata, clean


def split_data(
    features_list: list[dict],
    test_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split data into train pool and fixed test set.

    Args:
        features_list: Full dataset.
        test_size: Number of test samples to hold out.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_pool, test_set).
    """
    rng = random.Random(seed)
    indices = list(range(len(features_list)))
    rng.shuffle(indices)

    test_indices = set(indices[:test_size])
    train_pool = [features_list[i] for i in indices if i not in test_indices]
    test_set = [features_list[i] for i in indices[:test_size]]

    logger.info(
        f"Split: {len(train_pool)} train pool, " f"{len(test_set)} test samples"
    )
    return train_pool, test_set


# ── Evaluation ───────────────────────────────────────────────────


def evaluate_quantile(
    predictor: object,
    test_set: list[dict],
    quantiles: list[float],
) -> dict:
    """Evaluate a QuantilePredictor on the test set.

    Computes pinball loss, mean absolute error, and coverage
    for each quantile level.

    Args:
        predictor: Trained QuantilePredictor instance.
        test_set: List of feature dicts with runtime_ms.
        quantiles: Quantile levels to evaluate.

    Returns:
        Dict of evaluation metrics.
    """
    errors: list[float] = []
    pinball_losses: dict[str, list[float]] = {str(q): [] for q in quantiles}
    coverages: dict[str, int] = {str(q): 0 for q in quantiles}

    for sample in test_set:
        features = {k: v for k, v in sample.items() if k != "runtime_ms"}
        actual = sample["runtime_ms"]

        result = predictor.predict(features)
        q_preds = result.get("quantiles", {})

        # Use median (q0.5) as point estimate for MAE.
        median_pred = q_preds.get("0.5", 0.0)
        errors.append(abs(actual - median_pred))

        for q in quantiles:
            q_str = str(q)
            pred = q_preds.get(q_str, 0.0)

            # Pinball loss: max(q * (y - ŷ), (q-1) * (y - ŷ))
            diff = actual - pred
            loss = max(q * diff, (q - 1) * diff)
            pinball_losses[q_str].append(loss)

            # Coverage: fraction of actuals below quantile.
            if actual <= pred:
                coverages[q_str] += 1

    n = len(test_set)
    mae = float(np.mean(errors))
    mape = (
        float(np.mean([e / max(1, s["runtime_ms"]) for e, s in zip(errors, test_set)]))
        * 100
    )

    metrics = {
        "mae_ms": round(mae, 2),
        "mape_pct": round(mape, 2),
        "median_abs_error_ms": round(float(np.median(errors)), 2),
    }

    for q in quantiles:
        q_str = str(q)
        avg_pinball = float(np.mean(pinball_losses[q_str]))
        coverage_pct = coverages[q_str] / n * 100
        metrics[f"pinball_q{q_str}"] = round(avg_pinball, 4)
        metrics[f"coverage_q{q_str}"] = round(coverage_pct, 1)

    return metrics


# ── Training ─────────────────────────────────────────────────────


def train_single_config(
    train_data: list[dict],
    test_data: list[dict],
    hidden_layers: list[int],
    epochs: int,
    learning_rate: float,
    quantiles: list[float],
) -> dict:
    """Train one MLP configuration and evaluate it.

    Args:
        train_data: Training samples.
        test_data: Test samples.
        hidden_layers: MLP architecture spec.
        epochs: Training epochs.
        learning_rate: Learning rate for Adam optimizer.
        quantiles: Quantile levels for the predictor.

    Returns:
        Dict with config, training metrics, and evaluation metrics.
    """
    from swarmpilot.predictor.predictor.quantile import (
        QuantilePredictor,
    )

    training_config = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_layers": hidden_layers,
        "quantiles": quantiles,
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
    start_time = time.time()
    train_metrics = predictor.train(
        features_list=train_data,
        config=training_config,
    )
    train_time = time.time() - start_time

    # Evaluate on test set.
    eval_metrics = evaluate_quantile(predictor, test_data, quantiles)

    return {
        "training_config": {
            "hidden_layers": hidden_layers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "quantiles": quantiles,
        },
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_time_s": round(train_time, 2),
        "train_metrics": {
            "final_loss": train_metrics.get("final_loss", None),
            "samples_count": train_metrics.get("samples_count", 0),
            "augmented_count": train_metrics.get(
                "augmented_samples_count",
                0,
            ),
        },
        "eval_metrics": eval_metrics,
    }


# ── Sweep ────────────────────────────────────────────────────────


def run_sweep(
    train_pool: list[dict],
    test_set: list[dict],
    hidden_layers_list: list[list[int]],
    train_sizes: list[int],
    epochs: int,
    learning_rate: float,
    quantiles: list[float],
    seed: int,
) -> list[dict]:
    """Run the full training sweep.

    Args:
        train_pool: Full training data pool.
        test_set: Fixed test set.
        hidden_layers_list: List of hidden_layers configs to test.
        train_sizes: List of training set sizes to test.
        epochs: Training epochs per run.
        learning_rate: Learning rate.
        quantiles: Quantile levels.
        seed: Random seed for subset selection.

    Returns:
        List of result dicts, one per (hidden_layers, train_size).
    """
    results: list[dict] = []
    total_runs = len(hidden_layers_list) * len(train_sizes)
    run_idx = 0

    for hidden_layers in hidden_layers_list:
        for train_size in train_sizes:
            run_idx += 1
            layer_str = "x".join(str(x) for x in hidden_layers)

            if train_size > len(train_pool):
                logger.warning(
                    f"Skipping [{layer_str}] n={train_size}: "
                    f"only {len(train_pool)} samples available"
                )
                continue

            # Select a reproducible subset.
            rng = random.Random(seed)
            subset_indices = rng.sample(
                range(len(train_pool)),
                train_size,
            )
            train_subset = [train_pool[i] for i in subset_indices]

            logger.info(
                f"[{run_idx}/{total_runs}] Training MLP "
                f"[{layer_str}] with {train_size} samples..."
            )

            try:
                result = train_single_config(
                    train_data=train_subset,
                    test_data=test_set,
                    hidden_layers=hidden_layers,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    quantiles=quantiles,
                )
                result["run_id"] = f"{layer_str}_n{train_size}"
                results.append(result)

                mae = result["eval_metrics"]["mae_ms"]
                mape = result["eval_metrics"]["mape_pct"]
                logger.info(
                    f"  → MAE={mae:.1f}ms, "
                    f"MAPE={mape:.1f}%, "
                    f"time={result['train_time_s']:.1f}s"
                )

            except Exception as exc:
                logger.error(f"  → Failed: {exc}")
                results.append(
                    {
                        "run_id": f"{layer_str}_n{train_size}",
                        "training_config": {
                            "hidden_layers": hidden_layers,
                            "epochs": epochs,
                        },
                        "train_samples": train_size,
                        "test_samples": len(test_set),
                        "error": str(exc),
                    }
                )

    return results


def find_best_config(results: list[dict]) -> dict | None:
    """Find the best configuration by lowest MAE.

    Args:
        results: List of sweep result dicts.

    Returns:
        Best result dict, or None if no valid results.
    """
    valid = [
        r for r in results if "eval_metrics" in r and "mae_ms" in r["eval_metrics"]
    ]
    if not valid:
        return None
    return min(valid, key=lambda r: r["eval_metrics"]["mae_ms"])


def analyze_data_efficiency(results: list[dict]) -> dict:
    """Analyze how prediction quality improves with more data.

    For each architecture, find the minimum training size that
    achieves within 10% of the best MAE for that architecture.

    Args:
        results: List of sweep result dicts.

    Returns:
        Analysis dict with per-architecture efficiency metrics.
    """
    # Group by architecture.
    by_arch: dict[str, list[dict]] = {}
    for r in results:
        if "eval_metrics" not in r:
            continue
        layers = r["training_config"]["hidden_layers"]
        key = "x".join(str(x) for x in layers)
        by_arch.setdefault(key, []).append(r)

    analysis: dict[str, dict] = {}
    for arch, runs in by_arch.items():
        runs_sorted = sorted(runs, key=lambda r: r["train_samples"])
        best_mae = min(r["eval_metrics"]["mae_ms"] for r in runs_sorted)
        threshold = best_mae * 1.10  # Within 10% of best.

        min_sufficient = None
        for r in runs_sorted:
            if r["eval_metrics"]["mae_ms"] <= threshold:
                min_sufficient = r["train_samples"]
                break

        analysis[arch] = {
            "best_mae_ms": round(best_mae, 2),
            "best_train_size": min(
                (r for r in runs_sorted if r["eval_metrics"]["mae_ms"] == best_mae),
                key=lambda r: r["train_samples"],
            )["train_samples"],
            "min_sufficient_size": min_sufficient,
            "threshold_mae_ms": round(threshold, 2),
            "curve": [
                {
                    "train_size": r["train_samples"],
                    "mae_ms": r["eval_metrics"]["mae_ms"],
                    "mape_pct": r["eval_metrics"]["mape_pct"],
                }
                for r in runs_sorted
            ],
        }

    return analysis


# ── Main ─────────────────────────────────────────────────────────


def parse_hidden_layers(spec: str) -> list[int]:
    """Parse a hidden layers specification string.

    Args:
        spec: Comma-separated integers (e.g. "128,64,32").

    Returns:
        List of integers.
    """
    return [int(x.strip()) for x in spec.split(",")]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Systematic MLP training sweep.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Runtime data JSON file from profile_models.py",
    )
    parser.add_argument(
        "--output",
        default="sweep_results.json",
        help="Output JSON for sweep results",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        default=None,
        help="Hidden layer specs (e.g. '64,32' '128,64,32')",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help="Training set sizes to test",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help="Fixed test set size (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs per run (default: 200)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.5, 0.9, 0.95, 0.99],
        help="Quantile levels (default: 0.5 0.9 0.95 0.99)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MLP training sweep."""
    args = parse_args()

    # Set seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse hidden layers.
    if args.hidden_layers:
        hidden_layers_list = [parse_hidden_layers(s) for s in args.hidden_layers]
    else:
        hidden_layers_list = DEFAULT_HIDDEN_LAYERS

    # Load data.
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path
    metadata, features_list = load_runtime_data(str(data_path))

    min_required = max(args.train_sizes) + args.test_size
    if len(features_list) < min_required:
        logger.error(
            f"Need at least {min_required} samples "
            f"({max(args.train_sizes)} train + {args.test_size} test), "
            f"got {len(features_list)}"
        )
        return

    # Split data.
    train_pool, test_set = split_data(
        features_list,
        args.test_size,
        args.seed,
    )

    # Run sweep.
    logger.info(
        f"Starting sweep: {len(hidden_layers_list)} architectures "
        f"x {len(args.train_sizes)} sizes = "
        f"{len(hidden_layers_list) * len(args.train_sizes)} runs"
    )
    sweep_start = time.time()
    results = run_sweep(
        train_pool=train_pool,
        test_set=test_set,
        hidden_layers_list=hidden_layers_list,
        train_sizes=args.train_sizes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        quantiles=args.quantiles,
        seed=args.seed,
    )
    sweep_time = time.time() - sweep_start

    # Analyze.
    best = find_best_config(results)
    efficiency = analyze_data_efficiency(results)

    # Save results.
    output = {
        "metadata": {
            "source_data": str(args.data),
            "model_id": metadata["model_id"],
            "platform_info": metadata.get("platform_info", {}),
            "sweep_time_s": round(sweep_time, 1),
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "sweep_config": {
            "hidden_layers_tested": hidden_layers_list,
            "train_sizes": args.train_sizes,
            "test_size": args.test_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "quantiles": args.quantiles,
            "seed": args.seed,
        },
        "results": results,
        "best_config": best,
        "data_efficiency": efficiency,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Sweep results saved to {output_path}")

    # Summary.
    logger.info("═══ Sweep Summary ═══")
    logger.info(f"Total time: {sweep_time:.1f}s")
    logger.info(f"Total runs: {len(results)}")
    if best:
        layers = best["training_config"]["hidden_layers"]
        logger.info(
            f"Best config: hidden_layers={layers}, "
            f"train_size={best['train_samples']}, "
            f"MAE={best['eval_metrics']['mae_ms']:.1f}ms, "
            f"MAPE={best['eval_metrics']['mape_pct']:.1f}%"
        )
    for arch, info in efficiency.items():
        logger.info(
            f"  [{arch}] min sufficient: "
            f"{info['min_sufficient_size']} samples "
            f"(best MAE: {info['best_mae_ms']:.1f}ms)"
        )


if __name__ == "__main__":
    main()
