#!/usr/bin/env python3
r"""Visualize MLP training sweep results.

Reads the JSON output from training_sweep.py and generates:
    1. Training curves: MAE vs dataset size per architecture
    2. Architecture comparison: bar chart at max training size
    3. Data efficiency: annotated minimum sufficient data point
    4. Quantile coverage: calibration check

Usage:
    # Visualize single model sweep
    uv run python visualize_training.py \
        --input sweep_results_80b.json \
        --output-dir figures/

    # Compare both models
    uv run python visualize_training.py \
        --input sweep_results_80b.json sweep_results_8b.json \
        --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# ── Style ────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

MARKERS = ["o", "s", "^", "D", "v", "P"]


# ── Data Loading ─────────────────────────────────────────────────


def load_sweep(path: str) -> dict:
    """Load sweep results JSON.

    Args:
        path: Path to sweep_results.json.

    Returns:
        Parsed sweep results dict.
    """
    with open(path) as f:
        data = json.load(f)
    logger.info(
        f"Loaded sweep: {data['metadata']['model_id']} "
        f"({len(data['results'])} runs)"
    )
    return data


def group_by_architecture(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by MLP architecture.

    Args:
        results: List of sweep result dicts.

    Returns:
        Dict mapping architecture string to sorted results.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        if "eval_metrics" not in r:
            continue
        layers = r["training_config"]["hidden_layers"]
        key = "x".join(str(x) for x in layers)
        groups.setdefault(key, []).append(r)

    # Sort each group by train_samples.
    for key in groups:
        groups[key].sort(key=lambda r: r["train_samples"])

    return groups


# ── Plot 1: Training Curves ─────────────────────────────────────


def plot_training_curves(
    sweep: dict,
    output_path: str,
    metric: str = "mae_ms",
    ylabel: str = "MAE (ms)",
) -> None:
    """Plot MAE vs dataset size for each architecture.

    Args:
        sweep: Sweep results dict.
        output_path: Output image path.
        metric: Metric key in eval_metrics (default: mae_ms).
        ylabel: Y-axis label.
    """
    groups = group_by_architecture(sweep["results"])
    model_id = sweep["metadata"]["model_id"]

    fig, ax = plt.subplots()

    for i, (arch, runs) in enumerate(sorted(groups.items())):
        sizes = [r["train_samples"] for r in runs]
        values = [r["eval_metrics"][metric] for r in runs]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        ax.plot(
            sizes,
            values,
            color=color,
            marker=marker,
            label=f"[{arch}]",
        )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel(ylabel)
    ax.set_title(f"MLP Training Curve — {model_id}")
    ax.legend(title="hidden_layers", loc="upper right")
    ax.set_xticks(sweep["sweep_config"]["train_sizes"])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.success(f"Saved training curves → {output_path}")


# ── Plot 2: Architecture Comparison ─────────────────────────────


def plot_architecture_comparison(
    sweep: dict,
    output_path: str,
) -> None:
    """Bar chart comparing architectures at max training size.

    Args:
        sweep: Sweep results dict.
        output_path: Output image path.
    """
    groups = group_by_architecture(sweep["results"])
    model_id = sweep["metadata"]["model_id"]
    max_size = max(sweep["sweep_config"]["train_sizes"])

    archs = []
    maes = []
    mapes = []

    for arch, runs in sorted(groups.items()):
        # Find run with max training size.
        max_run = [r for r in runs if r["train_samples"] == max_size]
        if not max_run:
            continue
        archs.append(f"[{arch}]")
        maes.append(max_run[0]["eval_metrics"]["mae_ms"])
        mapes.append(max_run[0]["eval_metrics"]["mape_pct"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(archs))
    width = 0.6

    bars1 = ax1.bar(x, maes, width, color=COLORS[: len(archs)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs, rotation=30, ha="right")
    ax1.set_ylabel("MAE (ms)")
    ax1.set_title(f"MAE at n={max_size} — {model_id}")
    ax1.bar_label(bars1, fmt="%.1f", padding=3)

    bars2 = ax2.bar(x, mapes, width, color=COLORS[: len(archs)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(archs, rotation=30, ha="right")
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title(f"MAPE at n={max_size} — {model_id}")
    ax2.bar_label(bars2, fmt="%.1f", padding=3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.success(f"Saved architecture comparison → {output_path}")


# ── Plot 3: Data Efficiency ─────────────────────────────────────


def plot_data_efficiency(
    sweep: dict,
    output_path: str,
) -> None:
    """Training curves with minimum sufficient data annotated.

    Args:
        sweep: Sweep results dict.
        output_path: Output image path.
    """
    groups = group_by_architecture(sweep["results"])
    efficiency = sweep.get("data_efficiency", {})
    model_id = sweep["metadata"]["model_id"]

    fig, ax = plt.subplots()

    for i, (arch, runs) in enumerate(sorted(groups.items())):
        sizes = [r["train_samples"] for r in runs]
        values = [r["eval_metrics"]["mae_ms"] for r in runs]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        arch_key = arch.replace("x", "x")

        ax.plot(
            sizes,
            values,
            color=color,
            marker=marker,
            label=f"[{arch}]",
        )

        # Annotate minimum sufficient point.
        eff = efficiency.get(arch_key)
        if eff and eff.get("min_sufficient_size"):
            ms = eff["min_sufficient_size"]
            # Find the MAE at that size.
            matching = [r for r in runs if r["train_samples"] == ms]
            if matching:
                mae_val = matching[0]["eval_metrics"]["mae_ms"]
                ax.annotate(
                    f"n≥{ms}",
                    xy=(ms, mae_val),
                    xytext=(ms + 20, mae_val + 10),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "lw": 1.5,
                    },
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("MAE (ms)")
    ax.set_title(
        f"Data Efficiency — {model_id}\n"
        "(arrows mark min sufficient data within 10% of best)"
    )
    ax.legend(title="hidden_layers", loc="upper right")
    ax.set_xticks(sweep["sweep_config"]["train_sizes"])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.success(f"Saved data efficiency plot → {output_path}")


# ── Plot 4: Quantile Coverage ───────────────────────────────────


def plot_quantile_coverage(
    sweep: dict,
    output_path: str,
) -> None:
    """Check quantile calibration for the best config.

    Args:
        sweep: Sweep results dict.
        output_path: Output image path.
    """
    best = sweep.get("best_config")
    if not best or "eval_metrics" not in best:
        logger.warning("No best config found, skipping coverage plot")
        return

    model_id = sweep["metadata"]["model_id"]
    quantiles = sweep["sweep_config"]["quantiles"]

    # Extract actual vs expected coverage for best config.
    expected = [q * 100 for q in quantiles]
    actual = [best["eval_metrics"].get(f"coverage_q{q}", 0) for q in quantiles]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(quantiles))
    width = 0.35

    ax.bar(x - width / 2, expected, width, label="Expected", alpha=0.7)
    ax.bar(x + width / 2, actual, width, label="Actual", alpha=0.7)

    # Perfect calibration line.
    ax.plot(
        [-0.5, len(quantiles) - 0.5],
        [0, 0],
        "k--",
        alpha=0.3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"q{q}" for q in quantiles])
    ax.set_ylabel("Coverage (%)")
    ax.set_title(
        f"Quantile Calibration — {model_id}\n"
        f"(best config: {best['training_config']['hidden_layers']})"
    )
    ax.legend()
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.success(f"Saved quantile coverage → {output_path}")


# ── Plot 5: Multi-model Comparison ──────────────────────────────


def plot_multi_model_comparison(
    sweeps: list[dict],
    output_path: str,
) -> None:
    """Compare training curves across models (same best architecture).

    Args:
        sweeps: List of sweep result dicts (one per model).
        output_path: Output image path.
    """
    fig, ax = plt.subplots()

    for i, sweep in enumerate(sweeps):
        model_id = sweep["metadata"]["model_id"]
        best = sweep.get("best_config")
        if not best:
            continue

        best_layers = best["training_config"]["hidden_layers"]
        arch_key = "x".join(str(x) for x in best_layers)
        groups = group_by_architecture(sweep["results"])

        runs = groups.get(arch_key, [])
        if not runs:
            continue

        sizes = [r["train_samples"] for r in runs]
        values = [r["eval_metrics"]["mae_ms"] for r in runs]
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        # Shorten model_id for legend.
        short_name = model_id.split("/")[-1]
        ax.plot(
            sizes,
            values,
            color=color,
            marker=marker,
            label=f"{short_name} [{arch_key}]",
        )

    ax.set_xlabel("Training Samples")
    ax.set_ylabel("MAE (ms)")
    ax.set_title("Model Comparison — Best Architecture per Model")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.success(f"Saved multi-model comparison → {output_path}")


# ── Main ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize MLP training sweep results.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Sweep result JSON file(s)",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for output figures (default: figures/)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all visualization plots."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sweeps = [load_sweep(p) for p in args.input]

    for sweep in sweeps:
        model_id = sweep["metadata"]["model_id"]
        safe_name = model_id.split("/")[-1].lower().replace("-", "_")

        plot_training_curves(
            sweep,
            str(output_dir / f"training_curves_{safe_name}.png"),
        )
        plot_training_curves(
            sweep,
            str(output_dir / f"training_curves_mape_{safe_name}.png"),
            metric="mape_pct",
            ylabel="MAPE (%)",
        )
        plot_architecture_comparison(
            sweep,
            str(output_dir / f"arch_comparison_{safe_name}.png"),
        )
        plot_data_efficiency(
            sweep,
            str(output_dir / f"data_efficiency_{safe_name}.png"),
        )
        plot_quantile_coverage(
            sweep,
            str(output_dir / f"quantile_coverage_{safe_name}.png"),
        )

    # Multi-model comparison if multiple sweeps.
    if len(sweeps) > 1:
        plot_multi_model_comparison(
            sweeps,
            str(output_dir / "model_comparison.png"),
        )

    logger.success(
        f"All figures saved to {output_dir}/ " f"({len(sweeps)} model(s) visualized)"
    )


if __name__ == "__main__":
    main()
