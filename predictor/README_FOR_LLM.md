# README FOR LLM: Swarmpilot Runtime Predictor Service

> **Optimized for LLM consumption**: This document provides complete, unambiguous specifications for the Runtime Predictor Service. All API schemas, configurations, and interaction patterns are fully specified.

---

## Quick Reference

| Property | Value |
|----------|-------|
| **Service Name** | Swarmpilot Runtime Predictor Service |
| **Version** | 0.1.0 |
| **Framework** | FastAPI (Python 3.11+) |
| **Default URL** | `http://localhost:8000` |
| **Default Port** | `8000` |
| **ML Framework** | PyTorch 2.0+ |
| **Storage Backend** | Local filesystem (joblib) |
| **API Documentation** | `/docs` (Swagger UI), `/redoc` (ReDoc) |

**Purpose**: Machine learning service that predicts task execution runtimes using MLP-based regression models. Supports two prediction modes: expect/error (point estimate with error margin) and quantile (distribution percentiles).

**Key Concepts**:
- **model_id**: Identifier for the task/workload type (e.g., "image-classifier-v1")
- **platform_info**: Execution environment (software_name, software_version, hardware_name)
- **Model Key**: Unique combination of `{model_id}__{software_name}-{software_version}__{hardware_name}`
- **Prediction Types**: `expect_error` (MSE-based) or `quantile` (Pinball loss-based)

---

## Environment Variables

All configuration uses the `PREDICTOR_` prefix. Configuration priority: CLI args > Environment variables > Config file > Defaults.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PREDICTOR_HOST` | `string` | `"0.0.0.0"` | Server bind host address |
| `PREDICTOR_PORT` | `integer` | `8000` | Server bind port number |
| `PREDICTOR_RELOAD` | `boolean` | `false` | Enable auto-reload for development |
| `PREDICTOR_WORKERS` | `integer` | `1` | Number of Uvicorn worker processes |
| `PREDICTOR_STORAGE_DIR` | `string` | `"models"` | Directory path for model storage (absolute or relative) |
| `PREDICTOR_LOG_LEVEL` | `string` | `"info"` | Logging level: `debug`, `info`, `warning`, `error`, `critical` |
| `PREDICTOR_APP_NAME` | `string` | `"Runtime Predictor Service"` | Application display name |
| `PREDICTOR_APP_VERSION` | `string` | `"0.1.0"` | Application version string |

### Configuration File Format

File: `predictor.toml` (TOML format)

```toml
[predictor]
host = "0.0.0.0"
port = 8000
reload = false
workers = 1
storage_dir = "models"
log_level = "info"
app_name = "Runtime Predictor Service"
app_version = "0.1.0"
```

Load custom config file with CLI: `spredictor start --config /path/to/predictor.toml`

---

## API Endpoints

### Base URL
`http://{PREDICTOR_HOST}:{PREDICTOR_PORT}` (default: `http://localhost:8000`)

---

### 1. POST `/predict` - Get Runtime Prediction

**Purpose**: Predict task execution runtime using a trained model for specific platform and features.

**HTTP Method**: `POST`

**Request Headers**:
```
Content-Type: application/json
```

**Request Body Schema** (`PredictionRequest`):
```json
{
  "model_id": "string",           // Required: Task/workload identifier
  "platform_info": {              // Required: Execution environment
    "software_name": "string",    // e.g., "pytorch", "tensorflow", "jax"
    "software_version": "string", // e.g., "2.0.1", "2.13.0"
    "hardware_name": "string"     // e.g., "nvidia-a100", "cpu", "nvidia-v100"
  },
  "prediction_type": "string",    // Required: "expect_error" or "quantile"
  "features": {                   // Required: Feature dictionary (keys must match training data)
    "feature_name_1": 123.45,     // All values must be numeric (int or float)
    "feature_name_2": 67.89,
    "...": "..."
  }
}
```

**Important Rules**:
- Feature keys are **case-sensitive** and must match training data exactly
- Feature keys are processed in **alphabetical order** (internal normalization)
- Extra features are **ignored** (no error)
- Missing features cause **400 Bad Request**
- Feature values must be **numeric** (int or float)

**Response Body Schema** (`PredictionResponse`):

**For `prediction_type = "expect_error"`**:
```json
{
  "model_id": "string",
  "platform_info": {
    "software_name": "string",
    "software_version": "string",
    "hardware_name": "string"
  },
  "prediction_type": "expect_error",
  "result": {
    "expected_runtime_ms": 125.5,    // float: Expected runtime in milliseconds
    "error_margin_ms": 10.2          // float: Mean absolute error from training
  }
}
```

**For `prediction_type = "quantile"`**:
```json
{
  "model_id": "string",
  "platform_info": {
    "software_name": "string",
    "software_version": "string",
    "hardware_name": "string"
  },
  "prediction_type": "quantile",
  "result": {
    "quantiles": {                   // Dict[str, float]: Quantile level -> runtime (ms)
      "0.5": 120.0,                  // 50th percentile (median)
      "0.9": 150.0,                  // 90th percentile
      "0.95": 180.0,                 // 95th percentile
      "0.99": 250.0                  // 99th percentile
    }
  }
}
```

**Status Codes**:
- `200 OK`: Prediction successful, response contains result
- `400 Bad Request`: Invalid request (missing features, wrong prediction_type, invalid JSON)
- `404 Not Found`: No trained model exists for this model_id + platform_info combination
- `422 Unprocessable Entity`: Request validation failed (invalid schema)
- `500 Internal Server Error`: Prediction computation failed

**Error Response Schema** (`ErrorResponse`):
```json
{
  "error": "string",      // Error category: "ValidationError", "ModelNotFound", "PredictionError"
  "message": "string",    // Human-readable error description
  "details": {}           // Optional: Additional context (field errors, stack trace, etc.)
}
```

**Example Request (expect_error)**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "image-classifier-v1",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "expect_error",
    "features": {
      "batch_size": 32,
      "image_height": 224,
      "image_width": 224,
      "num_layers": 50
    }
  }'
```

**Example Response (expect_error)**:
```json
{
  "model_id": "image-classifier-v1",
  "platform_info": {
    "software_name": "pytorch",
    "software_version": "2.0.1",
    "hardware_name": "nvidia-a100"
  },
  "prediction_type": "expect_error",
  "result": {
    "expected_runtime_ms": 125.5,
    "error_margin_ms": 10.2
  }
}
```

**Example Request (quantile)**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "image-classifier-v1",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "quantile",
    "features": {
      "batch_size": 32,
      "image_height": 224,
      "image_width": 224,
      "num_layers": 50
    }
  }'
```

**Example Response (quantile)**:
```json
{
  "model_id": "image-classifier-v1",
  "platform_info": {
    "software_name": "pytorch",
    "software_version": "2.0.1",
    "hardware_name": "nvidia-a100"
  },
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

---

### 2. POST `/train` - Train or Update Model

**Purpose**: Train a new prediction model or update an existing model with training data.

**HTTP Method**: `POST`

**Request Headers**:
```
Content-Type: application/json
```

**Request Body Schema** (`TrainingRequest`):
```json
{
  "model_id": "string",           // Required: Task/workload identifier
  "platform_info": {              // Required: Execution environment
    "software_name": "string",
    "software_version": "string",
    "hardware_name": "string"
  },
  "prediction_type": "string",    // Required: "expect_error" or "quantile"
  "features_list": [              // Required: Array of training samples (min 10 samples)
    {
      "feature_name_1": 123.45,   // Numeric feature values
      "feature_name_2": 67.89,
      "runtime_ms": 125.5         // REQUIRED: Actual measured runtime in milliseconds
    },
    {
      "feature_name_1": 234.56,
      "feature_name_2": 78.90,
      "runtime_ms": 230.2
    }
    // ... minimum 10 samples total
  ],
  "training_config": {            // Optional: Training hyperparameters
    "epochs": 500,                // Default: 500
    "learning_rate": 0.01,        // Default: 0.01
    "hidden_layers": [64, 32],    // Default: [64, 32]
    "quantiles": [0.5, 0.9, 0.95, 0.99]  // Default: [0.5, 0.9, 0.95, 0.99] (quantile type only)
  }
}
```

**Important Validation Rules**:
1. **Minimum Samples**: At least 10 samples required in `features_list`
2. **runtime_ms Required**: Every sample MUST contain `runtime_ms` field (actual measured runtime)
3. **Consistent Features**: All samples must have **identical feature names** (excluding `runtime_ms`)
4. **Numeric Values**: All feature values and `runtime_ms` must be numeric (int or float)
5. **Positive Runtime**: `runtime_ms` must be > 0

**Training Configuration Options**:
- `epochs` (integer, default: 500): Number of training epochs
- `learning_rate` (float, default: 0.01): Adam optimizer learning rate
- `hidden_layers` (array of integers, default: [64, 32]): MLP hidden layer sizes
- `quantiles` (array of floats, default: [0.5, 0.9, 0.95, 0.99]): Quantile levels for quantile prediction type (ignored for expect_error)

**Response Body Schema** (`TrainingResponse`):
```json
{
  "status": "success",                                           // "success" or "error"
  "message": "Model trained successfully with 20 samples",       // Human-readable message
  "model_key": "image-classifier-v1__pytorch-2.0.1__nvidia-a100", // Generated model key
  "samples_trained": 20                                          // Number of samples used
}
```

**Status Codes**:
- `200 OK`: Training successful, model saved
- `400 Bad Request`: Validation failed (insufficient samples, missing runtime_ms, inconsistent features)
- `422 Unprocessable Entity`: Request validation failed (invalid schema)
- `500 Internal Server Error`: Training or model save failed

**Example Request**:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "image-classifier-v1",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "expect_error",
    "features_list": [
      {"batch_size": 16, "image_height": 224, "image_width": 224, "num_layers": 50, "runtime_ms": 95.3},
      {"batch_size": 32, "image_height": 224, "image_width": 224, "num_layers": 50, "runtime_ms": 125.7},
      {"batch_size": 64, "image_height": 224, "image_width": 224, "num_layers": 50, "runtime_ms": 210.2},
      {"batch_size": 16, "image_height": 512, "image_width": 512, "num_layers": 50, "runtime_ms": 340.5},
      {"batch_size": 32, "image_height": 512, "image_width": 512, "num_layers": 50, "runtime_ms": 620.8},
      {"batch_size": 16, "image_height": 224, "image_width": 224, "num_layers": 101, "runtime_ms": 150.2},
      {"batch_size": 32, "image_height": 224, "image_width": 224, "num_layers": 101, "runtime_ms": 198.4},
      {"batch_size": 64, "image_height": 224, "image_width": 224, "num_layers": 101, "runtime_ms": 330.1},
      {"batch_size": 16, "image_height": 512, "image_width": 512, "num_layers": 101, "runtime_ms": 580.3},
      {"batch_size": 32, "image_height": 512, "image_width": 512, "num_layers": 101, "runtime_ms": 980.7}
    ],
    "training_config": {
      "epochs": 1000,
      "learning_rate": 0.005,
      "hidden_layers": [128, 64, 32]
    }
  }'
```

**Example Response**:
```json
{
  "status": "success",
  "message": "Model trained successfully with 10 samples",
  "model_key": "image-classifier-v1__pytorch-2.0.1__nvidia-a100",
  "samples_trained": 10
}
```

---

### 3. GET `/list` - List All Trained Models

**Purpose**: Retrieve metadata for all trained models in storage.

**HTTP Method**: `GET`

**Request Headers**: None required

**Query Parameters**: None

**Response Body Schema** (`ModelListResponse`):
```json
{
  "models": [                      // Array of model metadata objects
    {
      "model_id": "string",        // Task/workload identifier
      "platform_info": {           // Execution environment
        "software_name": "string",
        "software_version": "string",
        "hardware_name": "string"
      },
      "prediction_type": "string", // "expect_error" or "quantile"
      "samples_count": 100,        // Number of training samples
      "last_trained": "2025-11-02T10:30:00Z"  // ISO 8601 timestamp
    }
    // ... more models
  ]
}
```

**Status Codes**:
- `200 OK`: Successfully retrieved model list (may be empty array)
- `500 Internal Server Error`: Failed to read storage directory

**Example Request**:
```bash
curl -X GET http://localhost:8000/list
```

**Example Response**:
```json
{
  "models": [
    {
      "model_id": "image-classifier-v1",
      "platform_info": {
        "software_name": "pytorch",
        "software_version": "2.0.1",
        "hardware_name": "nvidia-a100"
      },
      "prediction_type": "expect_error",
      "samples_count": 100,
      "last_trained": "2025-11-02T10:30:00Z"
    },
    {
      "model_id": "image-classifier-v1",
      "platform_info": {
        "software_name": "pytorch",
        "software_version": "2.0.1",
        "hardware_name": "nvidia-v100"
      },
      "prediction_type": "quantile",
      "samples_count": 50,
      "last_trained": "2025-11-02T09:15:00Z"
    }
  ]
}
```

---

### 4. GET `/health` - Health Check

**Purpose**: Check if the service is operational and storage is accessible.

**HTTP Method**: `GET`

**Request Headers**: None required

**Query Parameters**: None

**Response Body Schema** (`HealthResponse`):
```json
{
  "status": "string",    // "healthy" or "unhealthy"
  "reason": "string"     // Optional: Error message if unhealthy (null if healthy)
}
```

**Status Codes**:
- `200 OK`: Service is healthy and operational
- `503 Service Unavailable`: Service is unhealthy (storage not accessible, etc.)

**Example Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Example Response (Healthy)**:
```json
{
  "status": "healthy",
  "reason": null
}
```

**Example Response (Unhealthy)**:
```json
{
  "status": "unhealthy",
  "reason": "Storage directory not accessible: /path/to/models"
}
```

---

### 5. WebSocket `/ws/predict` - Real-Time Predictions

**Purpose**: Establish persistent WebSocket connection for low-latency, real-time predictions.

**Protocol**: WebSocket (JSON messages)

**WebSocket URL**: `ws://{PREDICTOR_HOST}:{PREDICTOR_PORT}/ws/predict`

**Message Format**:

**Client → Server (Send)**:
Same JSON schema as POST `/predict` request:
```json
{
  "model_id": "string",
  "platform_info": {
    "software_name": "string",
    "software_version": "string",
    "hardware_name": "string"
  },
  "prediction_type": "string",
  "features": {
    "feature_name_1": 123.45,
    "feature_name_2": 67.89
  }
}
```

**Server → Client (Receive)**:
Same JSON schema as POST `/predict` response OR error:
```json
// Success response
{
  "model_id": "string",
  "platform_info": {...},
  "prediction_type": "string",
  "result": {...}
}

// Error response
{
  "error": "string",
  "message": "string",
  "details": {}
}
```

**Connection Behavior**:
- Connection remains open for multiple prediction requests
- Each message is processed independently
- Errors are sent as JSON messages (connection stays open)
- Client or server can close connection at any time

**Example JavaScript Client**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predict');

ws.onopen = () => {
  console.log('WebSocket connected');

  // Send prediction request
  ws.send(JSON.stringify({
    "model_id": "image-classifier-v1",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "expect_error",
    "features": {
      "batch_size": 32,
      "image_height": 224,
      "image_width": 224,
      "num_layers": 50
    }
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);

  if (response.error) {
    console.error('Prediction error:', response.message);
  } else {
    console.log('Prediction result:', response.result);
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

**Example Python Client** (using `websockets` library):
```python
import asyncio
import json
import websockets

async def predict_websocket():
    uri = "ws://localhost:8000/ws/predict"

    async with websockets.connect(uri) as websocket:
        request = {
            "model_id": "image-classifier-v1",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0.1",
                "hardware_name": "nvidia-a100"
            },
            "prediction_type": "expect_error",
            "features": {
                "batch_size": 32,
                "image_height": 224,
                "image_width": 224,
                "num_layers": 50
            }
        }

        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        result = json.loads(response)

        if "error" in result:
            print(f"Error: {result['message']}")
        else:
            print(f"Prediction: {result['result']}")

asyncio.run(predict_websocket())
```

---

## Data Models (Pydantic Schemas)

### PlatformInfo
Identifies the execution environment for model training and prediction.

```python
{
  "software_name": str,      # e.g., "pytorch", "tensorflow", "jax"
  "software_version": str,   # e.g., "2.0.1", "2.13.0"
  "hardware_name": str       # e.g., "nvidia-a100", "cpu", "nvidia-v100", "amd-mi100"
}
```

**Usage**: Models are isolated by platform_info. A task running on different platforms requires separate trained models.

---

### PredictionRequest
Request body for POST `/predict` and WebSocket `/ws/predict`.

```python
{
  "model_id": str,                # Required: Task identifier
  "platform_info": PlatformInfo,  # Required: Execution environment
  "prediction_type": str,         # Required: "expect_error" or "quantile"
  "features": Dict[str, float]    # Required: Feature dictionary (numeric values)
}
```

---

### PredictionResponse
Response body for POST `/predict` and WebSocket `/ws/predict`.

```python
{
  "model_id": str,
  "platform_info": PlatformInfo,
  "prediction_type": str,
  "result": Union[ExpectErrorResult, QuantileResult]
}

# ExpectErrorResult (when prediction_type = "expect_error")
{
  "expected_runtime_ms": float,  # Expected runtime in milliseconds
  "error_margin_ms": float       # Mean absolute error from training
}

# QuantileResult (when prediction_type = "quantile")
{
  "quantiles": Dict[str, float]  # Quantile level -> runtime (ms)
}
```

---

### TrainingRequest
Request body for POST `/train`.

```python
{
  "model_id": str,                       # Required: Task identifier
  "platform_info": PlatformInfo,         # Required: Execution environment
  "prediction_type": str,                # Required: "expect_error" or "quantile"
  "features_list": List[Dict[str, Any]], # Required: Training samples (min 10)
  "training_config": Optional[Dict]      # Optional: Hyperparameters
}

# Each item in features_list must contain:
{
  "feature_name_1": float,  # Numeric features
  "feature_name_2": float,
  "runtime_ms": float       # REQUIRED: Actual measured runtime
}

# training_config structure:
{
  "epochs": int,              # Default: 500
  "learning_rate": float,     # Default: 0.01
  "hidden_layers": List[int], # Default: [64, 32]
  "quantiles": List[float]    # Default: [0.5, 0.9, 0.95, 0.99] (quantile only)
}
```

---

### TrainingResponse
Response body for POST `/train`.

```python
{
  "status": str,          # "success" or "error"
  "message": str,         # Human-readable message
  "model_key": str,       # Generated model key
  "samples_trained": int  # Number of samples used
}
```

---

### ErrorResponse
Error response for all endpoints.

```python
{
  "error": str,              # Error category
  "message": str,            # Detailed error message
  "details": Optional[Dict]  # Additional context
}
```

**Common Error Categories**:
- `"ValidationError"`: Request validation failed
- `"ModelNotFound"`: No trained model for model_id + platform_info
- `"PredictionError"`: Prediction computation failed
- `"TrainingError"`: Model training failed
- `"StorageError"`: Model storage/retrieval failed

---

## Component Integration

### Scheduler Integration

The Predictor service is designed to integrate with a Scheduler component in the Swarmpilot system.

**Typical Workflow**:

```
┌─────────────┐                    ┌───────────────┐
│  Scheduler  │                    │  Predictor    │
└──────┬──────┘                    └───────┬───────┘
       │                                   │
       │  1. Collect runtime measurements  │
       │     (task features + actual time) │
       │                                   │
       │  2. POST /train                   │
       │─────────────────────────────────>│
       │                                   │
       │  3. TrainingResponse              │
       │<─────────────────────────────────│
       │                                   │
       │  4. POST /predict (new tasks)     │
       │─────────────────────────────────>│
       │                                   │
       │  5. PredictionResponse            │
       │<─────────────────────────────────│
       │                                   │
       │  6. Use predictions for           │
       │     scheduling decisions          │
       │                                   │
```

**Training Phase**:
1. Scheduler executes tasks and measures actual runtimes
2. Scheduler collects task features (e.g., batch size, input dimensions)
3. Scheduler sends training data to `/train` endpoint
4. Predictor trains MLP model and persists to storage
5. Scheduler receives confirmation

**Prediction Phase**:
1. Scheduler receives new task with features
2. Scheduler sends features to `/predict` endpoint
3. Predictor loads appropriate model (based on model_id + platform_info)
4. Predictor returns runtime prediction
5. Scheduler uses prediction for scheduling decisions (e.g., resource allocation, task ordering)

**Model Isolation Strategy**:
Models are uniquely identified by:
- `model_id`: Task/workload type (e.g., "image-classification", "matrix-multiplication")
- `platform_info`: Execution environment (software + hardware)

This allows:
- Same task on different platforms → different models
- Different tasks on same platform → different models
- Platform-specific optimizations captured in separate models

**Example**:
- Model A: `image-classifier-v1__pytorch-2.0.1__nvidia-a100`
- Model B: `image-classifier-v1__pytorch-2.0.1__nvidia-v100`
- Model C: `image-classifier-v1__tensorflow-2.13.0__nvidia-a100`

All three models predict the same task but for different execution environments.

---

## Prediction Types

### 1. Expect/Error Mode (`expect_error`)

**Purpose**: Provides point estimate with error margin for runtime prediction.

**Algorithm**:
- **Model**: Multi-Layer Perceptron (MLP) regression
- **Loss Function**: Mean Squared Error (MSE)
- **Training**: Predicts expected runtime, computes error margin from residuals
- **Output**:
  - `expected_runtime_ms`: Mean predicted runtime
  - `error_margin_ms`: Mean absolute error computed from training data residuals

**When to Use**:
- Need single-value runtime estimate
- Want to know typical error range
- Scheduling algorithms that use point estimates with uncertainty bounds

**Interpretation**:
- **expected_runtime_ms**: Best guess for task runtime
- **error_margin_ms**: Average prediction error (±)
- Confidence interval: `[expected_runtime_ms - error_margin_ms, expected_runtime_ms + error_margin_ms]`

**Example Output**:
```json
{
  "expected_runtime_ms": 125.5,
  "error_margin_ms": 10.2
}
```
Interpretation: Task will take approximately 125.5ms, with typical error of ±10.2ms.

---

### 2. Quantile Mode (`quantile`)

**Purpose**: Provides runtime distribution at specified percentiles.

**Algorithm**:
- **Model**: Multi-Layer Perceptron (MLP) with multi-output
- **Loss Function**: Pinball loss (quantile regression)
- **Training**: Simultaneously predicts all quantiles with quantile-specific loss
- **Output**:
  - `quantiles`: Dictionary mapping quantile levels (0.5, 0.9, 0.95, 0.99) to predicted runtimes

**When to Use**:
- Need runtime distribution information
- Want to handle tail latencies (e.g., P99 for SLA guarantees)
- Probabilistic scheduling with risk management
- Need different confidence levels for different decisions

**Interpretation**:
- **quantiles["0.5"]**: Median runtime (50% of tasks finish faster)
- **quantiles["0.9"]**: 90th percentile (90% of tasks finish faster)
- **quantiles["0.95"]**: 95th percentile (95% of tasks finish faster)
- **quantiles["0.99"]**: 99th percentile (99% of tasks finish faster)

**Example Output**:
```json
{
  "quantiles": {
    "0.5": 120.0,
    "0.9": 150.0,
    "0.95": 180.0,
    "0.99": 250.0
  }
}
```
Interpretation:
- 50% of tasks finish within 120ms
- 90% finish within 150ms
- 95% finish within 180ms
- 99% finish within 250ms (long tail)

**Custom Quantiles**:
Specify custom quantiles during training:
```json
{
  "training_config": {
    "quantiles": [0.1, 0.5, 0.9, 0.99]  // 10th, 50th, 90th, 99th percentiles
  }
}
```

---

## Setup and Usage

### Installation

**Prerequisites**:
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

**Install uv** (if not already installed):
```bash
# macOS (Homebrew)
brew install uv

# Generic installer (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# pipx
pipx install uv
```

**Install Predictor Service**:
```bash
cd /chivier-disk/yanweiye/Projects/swarmpilot-refresh/predictor

# Initialize and sync dependencies
uv sync

# Install in editable mode (makes `spredictor` command available)
uv pip install -e .
```

**Verify Installation**:
```bash
spredictor version
spredictor --help
```

---

### Starting the Service

**Default Configuration** (development):
```bash
spredictor start
```
Service runs on `http://0.0.0.0:8000`

**Development Mode** (auto-reload on code changes):
```bash
spredictor start --reload
```

**Production Mode** (multiple workers):
```bash
spredictor start --workers 4 --log-level warning
```

**Custom Host and Port**:
```bash
spredictor start --host 127.0.0.1 --port 9000
```

**With Configuration File**:
```bash
# Create config file
spredictor config init --output predictor.toml

# Edit predictor.toml as needed

# Start with config
spredictor start --config predictor.toml
```

**All CLI Options**:
```bash
spredictor start --help

Options:
  --host, -h TEXT         Host to bind (default: 0.0.0.0)
  --port, -p INTEGER      Port to bind (default: 8000)
  --reload                Enable auto-reload (development)
  --workers, -w INTEGER   Number of workers (default: 1)
  --storage-dir, -s PATH  Model storage directory
  --log-level, -l TEXT    Logging level (default: info)
  --config, -c PATH       Configuration file path
```

---

### CLI Commands

#### Health Check
Check if service is running and healthy:
```bash
spredictor health
spredictor health --host localhost --port 9000
```

#### List Models
View all trained models:
```bash
# Brief listing
spredictor list

# Verbose with details
spredictor list --verbose

# Custom storage directory
spredictor list --storage-dir /path/to/models
```

#### Configuration Management
```bash
# Show current configuration
spredictor config show

# Show configuration from file
spredictor config show --config predictor.toml

# Initialize new config file
spredictor config init

# Initialize with custom path
spredictor config init --output /etc/predictor.toml

# Force overwrite existing config
spredictor config init --force
```

#### Version Info
```bash
spredictor version
```

---

### API Documentation

Once the service is running, access interactive API documentation:

**Swagger UI** (recommended for testing):
```
http://localhost:8000/docs
```

**ReDoc** (better for reading):
```
http://localhost:8000/redoc
```

**OpenAPI JSON**:
```
http://localhost:8000/openapi.json
```

---

## Important Implementation Details

### Model Key Generation

Models are uniquely identified by combining `model_id` and `platform_info`:

```python
model_key = f"{model_id}__{software_name}-{software_version}__{hardware_name}"
```

**Examples**:
- `image-classifier-v1__pytorch-2.0.1__nvidia-a100`
- `matrix-multiply__tensorflow-2.13.0__cpu`
- `nlp-transformer__jax-0.4.13__tpu-v4`

**Filesystem Storage**:
Models stored as: `{PREDICTOR_STORAGE_DIR}/{model_key}.joblib`

---

### Feature Processing

**Feature Ordering**:
- Features are automatically sorted **alphabetically by key name** during training
- Same ordering applied during prediction
- Ensures consistent feature vector alignment

**Example**:
```python
# Input features (any order)
{"batch_size": 32, "image_height": 224, "image_width": 224}

# Internal processing order (alphabetical)
["batch_size", "image_height", "image_width"]
# → [32, 224, 224]
```

**Normalization**:
- Z-score normalization applied during training:
  ```
  normalized = (value - mean) / std_dev
  ```
- Mean and std_dev computed from training data
- **Normalization parameters stored with model**
- Same normalization applied during prediction for consistency

**Special Fields**:
- `runtime_ms`: Target variable, excluded from features
- `exp_runtime`: Experiment mode trigger, excluded from features

---

### Model Persistence

**Storage Format**: Joblib (`.joblib` files)

**Stored State**:
```python
{
  "model_state": {
    "weights": {...},           # PyTorch model state_dict
    "normalization": {
      "mean": [...],            # Feature means
      "std": [...]              # Feature standard deviations
    },
    "feature_names": [...],     # Ordered feature names
    "quantiles": [...]          # Quantile levels (quantile type only)
  },
  "metadata": {
    "model_id": "...",
    "platform_info": {...},
    "prediction_type": "...",
    "samples_count": 100,
    "last_trained": "2025-11-02T10:30:00Z"
  }
}
```

**Thread Safety**:
- Storage operations use file-level locking
- Safe for concurrent reads
- Writes are atomic (write to temp file → rename)

---

### MLP Architecture

**Default Configuration**:
- **Input dimension**: Auto-detected from number of features
- **Hidden layers**: [64, 32] (configurable)
- **Output dimension**:
  - 1 for `expect_error` (single runtime prediction)
  - N for `quantile` (N = number of quantiles)
- **Activation**: ReLU for hidden layers, Linear for output layer
- **Optimizer**: Adam
- **Learning rate**: 0.01 (default)
- **Epochs**: 500 (default)

**Architecture Diagram** (expect_error with default config):
```
Input (M features)
       ↓
   Linear(M → 64) → ReLU
       ↓
   Linear(64 → 32) → ReLU
       ↓
   Linear(32 → 1)
       ↓
Output (1 value: runtime)
```

**Architecture Diagram** (quantile with 4 quantiles):
```
Input (M features)
       ↓
   Linear(M → 64) → ReLU
       ↓
   Linear(64 → 32) → ReLU
       ↓
   Linear(32 → 4)
       ↓
Output (4 values: q0.5, q0.9, q0.95, q0.99)
```

**Custom Architecture**:
```json
{
  "training_config": {
    "hidden_layers": [128, 64, 32],  // 3 hidden layers
    "learning_rate": 0.005,
    "epochs": 1000
  }
}
```

---

### Experiment Mode

**Purpose**: Generate synthetic predictions for testing and development without trained models.

**Activation Conditions** (either):
1. Request features contain `exp_runtime` field, OR
2. All platform_info fields set to `"exp"`:
   ```json
   {
     "software_name": "exp",
     "software_version": "exp",
     "hardware_name": "exp"
   }
   ```

**Behavior for Expect/Error**:
```python
{
  "expected_runtime_ms": exp_runtime,
  "error_margin_ms": exp_runtime * 0.05  # 5% of exp_runtime
}
```

**Behavior for Quantile**:
Generates synthetic quantiles from normal distribution:
- Mean: `exp_runtime`
- Standard deviation: `exp_runtime * 0.05`
- Computes percentiles for requested quantiles

**Example Experiment Request**:
```json
{
  "model_id": "test-task",
  "platform_info": {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp"
  },
  "prediction_type": "expect_error",
  "features": {
    "exp_runtime": 100.0,
    "batch_size": 32
  }
}
```

**Example Experiment Response**:
```json
{
  "model_id": "test-task",
  "platform_info": {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp"
  },
  "prediction_type": "expect_error",
  "result": {
    "expected_runtime_ms": 100.0,
    "error_margin_ms": 5.0
  }
}
```

---

### Error Handling

**Validation Errors** (400 Bad Request):
```json
{
  "error": "ValidationError",
  "message": "Insufficient training data: 5 samples provided, minimum 10 required",
  "details": {
    "samples_provided": 5,
    "minimum_required": 10
  }
}
```

**Model Not Found** (404 Not Found):
```json
{
  "error": "ModelNotFound",
  "message": "No model found for model_id='task-1' and platform_info=PlatformInfo(...)",
  "details": {
    "model_key": "task-1__pytorch-2.0.1__nvidia-a100",
    "available_models": ["task-2__pytorch-2.0.1__nvidia-a100"]
  }
}
```

**Prediction Errors** (500 Internal Server Error):
```json
{
  "error": "PredictionError",
  "message": "Failed to compute prediction: Feature dimension mismatch",
  "details": {
    "expected_features": 4,
    "received_features": 3
  }
}
```

**Storage Errors** (503 Service Unavailable):
```json
{
  "error": "StorageError",
  "message": "Storage directory not accessible",
  "details": {
    "storage_dir": "/path/to/models",
    "reason": "Permission denied"
  }
}
```

---

## Testing

**Test Suite Location**: `/chivier-disk/yanweiye/Projects/swarmpilot-refresh/predictor/tests/`

**Run All Tests**:
```bash
uv run pytest
```

**Run with Coverage**:
```bash
uv run pytest --cov=src
```

**Run Specific Test File**:
```bash
uv run pytest tests/test_api.py
```

**Run with Verbose Output**:
```bash
uv run pytest -v
```

**Test Coverage Includes**:
- API endpoints (POST /predict, POST /train, GET /list, GET /health)
- WebSocket connections
- Both prediction types (expect_error, quantile)
- Model training and loading
- Feature validation
- Error scenarios (missing models, invalid features, insufficient data)
- Experiment mode
- Configuration management
- CLI commands

---

## Dependencies

**Core Runtime Dependencies** (from `pyproject.toml`):
- `fastapi>=0.104.0` - Web framework for API endpoints
- `uvicorn[standard]>=0.24.0` - ASGI server with WebSocket support
- `pydantic>=2.5.0` - Data validation and serialization
- `pydantic-settings>=2.0.0` - Settings management with environment variables
- `torch>=2.0.0` - PyTorch for neural network training and inference
- `numpy>=1.24.0` - Numerical computing for feature processing
- `joblib>=1.3.0` - Model serialization and persistence
- `python-dateutil>=2.8.0` - Date/time utilities for timestamps
- `websockets>=12.0` - WebSocket protocol support
- `typer>=0.9.0` - CLI framework for spredictor commands

**Development Dependencies**:
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `httpx>=0.25.0` - HTTP client for API testing
- `pytest-cov>=4.1.0` - Coverage reporting

**Python Version Requirement**: `>=3.11`

**Install All Dependencies**:
```bash
uv sync
```

**Install Only Runtime Dependencies**:
```bash
uv sync --no-dev
```

**Add New Dependency**:
```bash
uv add <package-name>         # Runtime dependency
uv add --dev <package-name>   # Development dependency
```

---

## Complete Example: End-to-End Workflow

### Scenario: Train and predict image classification runtime

**Step 1: Start the service**
```bash
spredictor start --reload
```

**Step 2: Train a model**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "resnet50-inference",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "quantile",
    "features_list": [
      {"batch_size": 1, "image_size": 224, "runtime_ms": 8.5},
      {"batch_size": 8, "image_size": 224, "runtime_ms": 45.2},
      {"batch_size": 16, "image_size": 224, "runtime_ms": 82.1},
      {"batch_size": 32, "image_size": 224, "runtime_ms": 155.3},
      {"batch_size": 1, "image_size": 512, "runtime_ms": 35.7},
      {"batch_size": 8, "image_size": 512, "runtime_ms": 210.4},
      {"batch_size": 16, "image_size": 512, "runtime_ms": 398.2},
      {"batch_size": 32, "image_size": 512, "runtime_ms": 750.6},
      {"batch_size": 64, "image_size": 224, "runtime_ms": 295.1},
      {"batch_size": 64, "image_size": 512, "runtime_ms": 1450.3}
    ],
    "training_config": {
      "epochs": 1000,
      "learning_rate": 0.01,
      "hidden_layers": [128, 64],
      "quantiles": [0.5, 0.9, 0.95, 0.99]
    }
  }'
```

**Response**:
```json
{
  "status": "success",
  "message": "Model trained successfully with 10 samples",
  "model_key": "resnet50-inference__pytorch-2.0.1__nvidia-a100",
  "samples_trained": 10
}
```

**Step 3: List trained models**
```bash
curl -X GET http://localhost:8000/list
```

**Response**:
```json
{
  "models": [
    {
      "model_id": "resnet50-inference",
      "platform_info": {
        "software_name": "pytorch",
        "software_version": "2.0.1",
        "hardware_name": "nvidia-a100"
      },
      "prediction_type": "quantile",
      "samples_count": 10,
      "last_trained": "2025-11-02T14:30:00Z"
    }
  ]
}
```

**Step 4: Make a prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "resnet50-inference",
    "platform_info": {
      "software_name": "pytorch",
      "software_version": "2.0.1",
      "hardware_name": "nvidia-a100"
    },
    "prediction_type": "quantile",
    "features": {
      "batch_size": 24,
      "image_size": 384
    }
  }'
```

**Response**:
```json
{
  "model_id": "resnet50-inference",
  "platform_info": {
    "software_name": "pytorch",
    "software_version": "2.0.1",
    "hardware_name": "nvidia-a100"
  },
  "prediction_type": "quantile",
  "result": {
    "quantiles": {
      "0.5": 285.3,
      "0.9": 320.5,
      "0.95": 345.2,
      "0.99": 398.7
    }
  }
}
```

**Interpretation**: For batch_size=24 and image_size=384 on A100:
- Median runtime: 285.3ms
- 90% of executions complete within 320.5ms
- 95% complete within 345.2ms
- 99% complete within 398.7ms

---

## Summary: Key Points for LLMs

1. **Service Purpose**: Runtime prediction for tasks using MLP regression models
2. **Two Prediction Modes**:
   - `expect_error`: Point estimate with error margin
   - `quantile`: Runtime distribution at percentiles
3. **Model Isolation**: Models unique to `{model_id}__{platform_info}`
4. **API Endpoints**: `/predict`, `/train`, `/list`, `/health`, `/ws/predict`
5. **Training Requirements**: Minimum 10 samples, must include `runtime_ms`
6. **Feature Processing**: Alphabetical ordering + Z-score normalization
7. **Storage**: Local filesystem with joblib serialization
8. **Configuration**: Environment variables (`PREDICTOR_*`), config file, or CLI args
9. **CLI**: `spredictor` command with start, health, list, config subcommands
10. **Integration**: Works with Scheduler for training data collection and prediction requests

---

**End of README_FOR_LLM.md**
