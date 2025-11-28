# Experiment Mode Distribution Configuration

## Overview

Experiment mode generates synthetic predictions without trained models. This document explains how to configure the returned distribution to match your workload characteristics.

## Quick Reference

| Distribution Type | Use Case | Configuration |
|------------------|----------|---------------|
| Unimodal Normal | Stable tasks | `exp_runtime` only |
| Unimodal Log-normal | Long-tail tasks | `exp_cv` + `exp_skewness` |
| Multimodal (GMM) | Mixed workloads | `exp_modes` |

## Configuration Parameters

All parameters are passed in the `features` dict of the prediction request.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exp_runtime` | float | required | Expected runtime in ms |
| `exp_cv` | float | 0.30 | Coefficient of variation (std/mean) |
| `exp_skewness` | float | 0.0 | Skewness (0=normal, >0=right-skewed) |
| `exp_modes` | list | None | Mode specifications for GMM |

## Distribution Types

### 1. Unimodal Normal (Default)

Symmetric distribution around the expected runtime. Good for stable, predictable tasks.

```python
features = {
    "exp_runtime": 1000,  # Expected runtime in ms
    "exp_cv": 0.30        # Optional: 30% CV (default)
}
```

**Output characteristics:**
- Median ≈ `exp_runtime`
- q99 ≈ `exp_runtime * (1 + 2.33 * exp_cv)`

### 2. Unimodal Log-normal (Long-tail)

Right-skewed distribution with heavy tail. Good for tasks with occasional long executions.

```python
features = {
    "exp_runtime": 10000,
    "exp_cv": 1.0,         # High variability
    "exp_skewness": 2.5    # Heavy right tail
}
```

**Output characteristics:**
- Median < `exp_runtime`
- q99/q50 ratio typically 5-15x for heavy tails

### 3. Multimodal (Gaussian Mixture Model)

Multiple peaks in the distribution. Good for mixed workloads like cache hit/miss scenarios.

```python
features = {
    "exp_runtime": 275,  # Reference value for error margin
    "exp_modes": [
        {"mean": 50, "weight": 0.8, "cv": 0.10},   # Cache hit (80%)
        {"mean": 500, "weight": 0.2, "cv": 0.30}   # Cache miss (20%)
    ]
}
```

**Mode specification:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `mean` | Yes | - | Mode center in ms |
| `weight` | Yes | - | Probability weight (auto-normalized) |
| `cv` | No | 0.20 | Mode's coefficient of variation |
| `skewness` | No | 0.0 | Mode's skewness (0=normal) |

## Examples

### Example 1: LLM Inference Task

Relatively stable with moderate variability:

```python
features = {
    "exp_runtime": 3000,
    "exp_cv": 0.40,
    "exp_skewness": 0.0
}
```

### Example 2: Video Generation Task

High variability with long tail:

```python
features = {
    "exp_runtime": 10000,
    "exp_cv": 1.0,
    "exp_skewness": 2.5
}
```

### Example 3: Cache-Dependent Task

Bimodal distribution (fast cache hit vs slow cache miss):

```python
features = {
    "exp_runtime": 140,
    "exp_modes": [
        {"mean": 50, "weight": 0.8, "cv": 0.10},
        {"mean": 500, "weight": 0.2, "cv": 0.30}
    ]
}
```

### Example 4: Batch Size Variation

Trimodal distribution for different batch sizes:

```python
features = {
    "exp_runtime": 5000,
    "exp_modes": [
        {"mean": 1000, "weight": 0.50, "cv": 0.15},   # Small batch
        {"mean": 5000, "weight": 0.35, "cv": 0.20},   # Medium batch
        {"mean": 15000, "weight": 0.15, "cv": 0.10}   # Large batch
    ]
}
```

## Prediction Type Behavior

### `expect_error` Prediction

Returns `expected_runtime_ms` and `error_margin_ms`.

- **Unimodal**: `error_margin = exp_runtime * exp_cv`
- **Multimodal**: Error margin accounts for both within-mode and between-mode variance

### `quantile` Prediction

Returns quantile values (default: 0.5, 0.9, 0.95, 0.99).

Generated from 2000 Monte Carlo samples of the configured distribution.

## Backward Compatibility

All new parameters are optional. Existing code using only `exp_runtime` continues to work:

```python
# This still works (uses CV=0.30, skewness=0.0)
features = {"exp_runtime": 1000}
```

## CV Guidelines

| Workload Type | Recommended CV |
|---------------|----------------|
| Stable (low variability) | 0.10 - 0.20 |
| Moderate variability | 0.30 - 0.50 |
| High variability | 0.50 - 1.50 |

## API Request Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test_model",
    "platform_info": {
      "software_name": "exp",
      "software_version": "exp",
      "hardware_name": "exp"
    },
    "features": {
      "exp_runtime": 1000,
      "exp_cv": 0.50,
      "exp_skewness": 1.5
    },
    "prediction_type": "quantile"
  }'
```

## Related Documentation

- [Prediction Types](6.PREDICTION_TYPES.md)
- [Usage Examples](3.USAGE_EXAMPLES.md#experiment-mode)
