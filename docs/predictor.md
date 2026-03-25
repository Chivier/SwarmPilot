# Predictor

MLP-based runtime prediction for ML inference workloads. The Predictor trains lightweight neural networks to estimate execution times, supporting two modes: point estimates with error margins (expect_error) and quantile regression (quantile).

## Overview

| Mode | Output | Use Case |
|------|--------|----------|
| **expect_error** | `expected_runtime_ms` + `error_margin_ms` | Simple scheduling, monitoring |
| **quantile** | Percentile values (e.g., p50, p90, p95, p99) | SLA-aware scheduling |
| **experiment** | Synthetic predictions from `exp_runtime` feature | Testing without trained models |

---

## Quick Start

### Starting the Service

```bash
spredictor start --port 8001
```

### CLI Commands

```bash
spredictor start          # Start the service
spredictor health         # Check service health
spredictor version        # Show version
spredictor list           # List stored models
spredictor config show    # Show current configuration
spredictor config init    # Generate example config file
```

### Configuration

All settings use the `PREDICTOR_` env prefix. Configuration priority: CLI arguments > Environment variables > TOML file > Defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_HOST` | `0.0.0.0` | Bind address |
| `PREDICTOR_PORT` | `8000` | Bind port |
| `PREDICTOR_RELOAD` | `false` | Enable auto-reload (development) |
| `PREDICTOR_WORKERS` | `1` | Number of worker processes |
| `PREDICTOR_STORAGE_DIR` | `models` | Trained model storage directory |
| `PREDICTOR_CACHE_MAX_SIZE` | `100` | Max models in memory cache |
| `PREDICTOR_LOG_LEVEL` | `info` | Log level |
| `PREDICTOR_LOG_DIR` | `logs` | Log file directory |

Settings can also be provided via a `predictor.toml` file (searched in current and parent directories).

### Verify

```bash
curl http://localhost:8001/health
# {"status": "healthy"}

curl http://localhost:8001/list
# {"models": []}
```

---

## Training

### Via Predictor HTTP API

`POST /train` -- requires at least 10 training samples, each with a `runtime_ms` field.

```bash
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-7b",
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.4.0",
      "hardware_name": "NVIDIA A100"
    },
    "prediction_type": "expect_error",
    "features_list": [
      {"batch_size": 1, "seq_len": 128, "runtime_ms": 45.2},
      {"batch_size": 1, "seq_len": 256, "runtime_ms": 82.1},
      {"batch_size": 1, "seq_len": 512, "runtime_ms": 155.3},
      {"batch_size": 2, "seq_len": 128, "runtime_ms": 58.7},
      {"batch_size": 2, "seq_len": 256, "runtime_ms": 105.4},
      {"batch_size": 2, "seq_len": 512, "runtime_ms": 198.6},
      {"batch_size": 4, "seq_len": 128, "runtime_ms": 75.0},
      {"batch_size": 4, "seq_len": 256, "runtime_ms": 140.2},
      {"batch_size": 4, "seq_len": 512, "runtime_ms": 270.8},
      {"batch_size": 8, "seq_len": 128, "runtime_ms": 110.5}
    ]
  }'
```

Response:

```json
{
  "status": "success",
  "message": "Model trained successfully with 10 samples",
  "model_key": "llama-7b__vllm-0.4.0__NVIDIA A100",
  "samples_trained": 10
}
```

**Training config parameters** (optional, passed in `training_config`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | `500` | Number of training epochs |
| `learning_rate` | `0.01` | Adam optimizer learning rate |
| `hidden_layers` | `[64, 32]` | MLP hidden layer sizes |

Quantile-specific:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantiles` | `[0.5, 0.9, 0.95, 0.99]` | Quantile levels to predict |
| `data_augmentation.enabled` | `true` | Enable data augmentation |
| `data_augmentation.cv` | auto | Coefficient of variation |
| `data_augmentation.samples_per_point` | `5` | Samples per original |
| `data_augmentation.distribution` | `"lognormal"` | `"lognormal"` or `"normal"` |
| `residual_calibration.enabled` | `false` | Enable residual calibration |
| `log_transform.enabled` | `false` | Apply log transform to runtime_ms |

### Via Scheduler API

The Scheduler exposes a training endpoint that flushes its sample buffer and trains:

```bash
curl -X POST http://localhost:8000/v1/predictor/train \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-7B"}'
```

### Via SDK

```python
async with SwarmPilotClient(
    "http://localhost:8002",
    scheduler_url="http://localhost:8000",
) as sp:
    result = await sp.train("Qwen/Qwen2.5-7B")
    print(result.samples_trained, result.metrics)
```

### Auto-Training Pipeline

When `TRAINING_ENABLE_AUTO=true` on the Scheduler, the training pipeline runs automatically:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_ENABLE_AUTO` | `false` | Enable auto-retraining |
| `TRAINING_BATCH_SIZE` | `100` | Batch size for data collection |
| `TRAINING_FREQUENCY` | `3600` | Training frequency (seconds) |
| `TRAINING_MIN_SAMPLES` | `10` | Minimum samples before first training |
| `TRAINING_PREDICTION_TYPES` | `quantile` | Prediction types to train |

---

## Prediction

### Via Predictor HTTP API

`POST /predict`:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-7b",
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.4.0",
      "hardware_name": "NVIDIA A100"
    },
    "prediction_type": "expect_error",
    "features": {"batch_size": 4, "seq_len": 256}
  }'
```

**expect_error response:**

```json
{
  "result": {
    "expected_runtime_ms": 138.5,
    "error_margin_ms": 12.3
  }
}
```

**quantile response:**

```json
{
  "result": {
    "quantiles": {
      "0.5": 120.0,
      "0.9": 150.0,
      "0.95": 180.0,
      "0.99": 250.0
    }
  }
}
```

### Via Scheduler API

```bash
curl -X POST http://localhost:8000/v1/predictor/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "Qwen/Qwen2.5-7B",
    "features": {"token_count": 150},
    "platform_info": {
      "software_name": "vllm",
      "software_version": "0.4.0",
      "hardware_name": "A100"
    }
  }'
```

### Via SDK

```python
pred = await sp.predict(
    "Qwen/Qwen2.5-7B",
    features={"token_count": 150},
    prediction_type="expect_error",
)
print(pred.expected_runtime_ms, pred.error_margin_ms)
```

### Experiment Mode

Test predictions without a trained model by using `exp_runtime` in features or setting all `platform_info` fields to `"exp"`:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "any-model",
    "platform_info": {
      "software_name": "exp",
      "software_version": "exp",
      "hardware_name": "exp"
    },
    "prediction_type": "expect_error",
    "features": {"exp_runtime": 100.0, "batch_size": 4}
  }'
```

---

## MLP Architecture & Algorithm Details

### Network Architecture

```
Input → [Linear → ReLU] × H → Linear → Output
```

| Component | Description |
|-----------|-------------|
| **Input layer** | Dimensionality = number of features (after preprocessing) |
| **Hidden layers** | Configurable sizes (default: `[64, 32]`) with ReLU activation |
| **Output layer** | 1 neuron (expect_error) or N neurons (N quantiles) |
| **Optimizer** | Adam with configurable learning rate (default: 0.01) |

### Feature Processing

1. **Feature extraction**: Separate `runtime_ms` (target) from other numeric keys; sort alphabetically
2. **Constant feature filtering**: Remove zero-variance features
3. **Z-score normalization**: `X_normalized = (X - mean) / (std + 1e-8)`
4. **GPU enrichment**: Auto-append hardware specs (cuda_cores, memory_gb, etc.) when a known NVIDIA GPU is detected

### Expect Error Predictor

- **Loss**: MSE (mean squared error)
- **Error margin**: `mean(|y_i - f(x_i)|)` over training residuals (fixed after training)
- **Output**: `expected_runtime_ms` + `error_margin_ms`

### Quantile Predictor

- **Architecture**: Base + delta outputs guarantee monotonicity by construction
  - `q₀ = base`, `qᵢ = qᵢ₋₁ + softplus(δᵢ)` (since softplus > 0)
- **Loss**: Pinball (quantile) loss per level
- **Target normalization**: Z-score on `runtime_ms` (denormalized after inference)
- **Output**: `{quantile_level: predicted_runtime_ms}`

### Data Augmentation (Quantile)

For single-observation feature combinations, synthetic samples are generated:

- **Lognormal** (default): `new_runtime ~ Lognormal(μ_ln, σ_ln)` centered on original runtime
- **Normal**: `new_runtime ~ max(Normal(r, r*cv), 1.0)`
- **CV estimation**: Auto-calculated from data when not specified, bounded to `[0.1, 1.5]`

### Residual Calibration

Post-training step: fits log-normal distribution to `actual/predicted` ratios for uncertainty estimation.

### Predictor Comparison

| Aspect | Expect Error | Quantile |
|--------|-------------|----------|
| Loss | MSE | Pinball |
| Output | Point + margin | Multiple percentiles |
| Architecture | MLP → 1 | MLP → base+delta → N |
| Target normalization | None | Z-score |
| Data augmentation | Not applied | Enabled by default |
| Use case | Simple scheduling | SLA-aware scheduling |

---

## API Reference

All Predictor endpoints are at the root path (no prefix).

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check (returns 503 if storage inaccessible) |
| GET | `/list` | List all trained models with metadata |
| POST | `/train` | Train or retrain a model (min 10 samples) |
| POST | `/predict` | Make a runtime prediction |
| WS | `/ws/predict` | WebSocket for streaming predictions |
| GET | `/cache/stats` | Cache hit/miss statistics |
| POST | `/cache/clear` | Clear in-memory model cache |

### Model Key Format

Models are stored with a composite key:

```
{model_id}__{software_name}-{software_version}__{hardware_name}
```

Example: `llama-7b__vllm-0.4.0__NVIDIA A100`

### Error Codes

| Status | Meaning |
|--------|---------|
| `200` | Success |
| `400` | Invalid input (insufficient samples, invalid type, feature errors) |
| `404` | Model not found |
| `422` | Request body validation error |
| `500` | Internal error (training/prediction/storage failure) |
| `503` | Service unhealthy (storage not accessible) |

---

## Model Management

### Predictor Status (via Scheduler)

```bash
curl http://localhost:8000/v1/predictor/status/Qwen%2FQwen2.5-7B
```

### List Trained Models

```bash
# Via Predictor
curl http://localhost:8001/list

# Via Scheduler
curl http://localhost:8000/v1/predictor/models
```

### Cache Management

```bash
curl http://localhost:8001/cache/stats
curl -X POST http://localhost:8001/cache/clear
```

### Interactive API Docs

When the Predictor is running: Swagger UI at `http://localhost:8001/docs`, ReDoc at `http://localhost:8001/redoc`.
