# Predictor Service - LLM Reference

> Single-file reference for the Predictor service. For detailed documentation, see [docs/](../).

## Overview

| Property | Value |
|----------|-------|
| **Service** | Runtime Predictor |
| **Version** | 0.1.0 |
| **Port** | 8001 (typical) |
| **API Prefix** | _(none)_ |
| **Framework** | FastAPI |
| **ML Framework** | PyTorch 2.0+ |
| **Entry Point** | `swarmpilot/predictor/cli.py` |
| **Main API** | `swarmpilot/predictor/api/app.py` |
| **CLI** | `spredictor` |

**Purpose:** Predicts task execution runtimes using MLP-based regression models. Supports point estimates (expect/error) and quantile predictions.

---

## File Structure

```
swarmpilot/predictor/
├── cli.py                    # CLI entry point (spredictor)
├── config.py                 # Configuration (pydantic-settings, PREDICTOR_* prefix)
├── models.py                 # Pydantic models
├── api/
│   ├── app.py               # FastAPI application factory
│   ├── dependencies.py      # Shared dependencies (storage, cache)
│   ├── routes/
│   │   ├── prediction.py    # POST /predict
│   │   ├── training.py      # POST /train
│   │   ├── models.py        # GET /list
│   │   ├── health.py        # GET /health
│   │   ├── cache.py         # GET /cache/stats, POST /cache/clear
│   │   └── websocket.py     # WS /ws/predict
│   └── ...
├── predictor/                # Prediction algorithms
│   ├── base.py              # Abstract base class
│   ├── expect_error.py      # MSE-based predictor
│   └── quantile.py          # Quantile regression
├── preprocessor/             # Feature processing
│   ├── base_preprocessor.py
│   └── preprocessors_registry.py
├── storage/
│   └── model_storage.py     # Model persistence
└── utils/
    └── logging.py
```

---

## API Endpoints

**Predictor endpoints have NO prefix (mounted at root).**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get runtime prediction |
| `/train` | POST | Train new model |
| `/list` | GET | List trained models |
| `/health` | GET | Health check |
| `/ws/predict` | WebSocket | Real-time predictions |
| `/cache/stats` | GET | Get cache statistics |
| `/cache/clear` | POST | Clear prediction cache |

---

## Key Request/Response Schemas

### Prediction Request
```json
// POST /predict
{
  "model_id": "image-classifier-v1",
  "platform_info": {
    "software_name": "pytorch",
    "software_version": "2.0.1",
    "hardware_name": "nvidia-a100"
  },
  "prediction_type": "expect_error",  // or "quantile"
  "features": {
    "batch_size": 32,
    "image_height": 224,
    "image_width": 224
  }
}
```

### Expect/Error Response
```json
{
  "model_id": "...",
  "platform_info": {...},
  "prediction_type": "expect_error",
  "result": {
    "expected_runtime_ms": 125.5,
    "error_margin_ms": 10.2
  }
}
```

### Quantile Response
```json
{
  "model_id": "...",
  "platform_info": {...},
  "prediction_type": "quantile",
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

### Training Request
```json
// POST /train
{
  "model_id": "...",
  "platform_info": {...},
  "prediction_type": "expect_error",
  "features_list": [
    {"batch_size": 16, "runtime_ms": 95.3},
    {"batch_size": 32, "runtime_ms": 125.7}
    // ... minimum 10 samples
  ],
  "training_config": {
    "epochs": 500,
    "learning_rate": 0.01,
    "hidden_layers": [64, 32]
  }
}
```

---

## Prediction Types

| Type | Description | Output |
|------|-------------|--------|
| `expect_error` | Point estimate with error margin | `expected_runtime_ms`, `error_margin_ms` |
| `quantile` | Distribution percentiles | `quantiles` dict (0.5, 0.9, 0.95, 0.99) |

---

## Model Key Format

Models are uniquely identified by:

```
{model_id}__{software_name}-{software_version}__{hardware_name}__{prediction_type}
```

Example: `image-classifier-v1__pytorch-2.0.1__nvidia-a100__expect_error`

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTOR_HOST` | `0.0.0.0` | Bind host |
| `PREDICTOR_PORT` | `8000` | Bind port |
| `PREDICTOR_RELOAD` | `false` | Auto-reload (development) |
| `PREDICTOR_WORKERS` | `1` | Worker processes |
| `PREDICTOR_STORAGE_DIR` | `models` | Model storage directory |
| `PREDICTOR_LOG_LEVEL` | `info` | Log level |
| `PREDICTOR_LOG_DIR` | `logs` | Log file directory |

Also reads `.env` file and `predictor.toml` if present. See [CONFIGURATION.md](../CONFIGURATION.md) for details.

---

## Training Requirements

- **Minimum samples:** 10
- **Required field:** `runtime_ms` in each sample
- **Consistent features:** All samples must have identical feature names
- **Numeric values:** All features must be numeric

---

## Feature Processing

- Features sorted **alphabetically by key name**
- Z-score normalization: `(value - mean) / std`
- Normalization parameters stored with model
- Special fields excluded: `runtime_ms`, `exp_runtime`
- Preprocessor pipeline configurable via `enable_preprocessors` and `preprocessor_mappings`

---

## Experiment Mode

Activated when `platform_info` fields set to `"exp"`:

```json
{
  "platform_info": {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp"
  },
  "features": {"exp_runtime": 100.0}
}
```

Returns synthetic predictions without trained models.

---

**Version:** 0.1.0 | **Updated:** 2026-01-30
