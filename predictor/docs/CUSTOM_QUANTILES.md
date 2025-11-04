# Custom Quantiles Feature

## Overview

The predictor service now supports custom quantile configurations for predictions. This feature allows users to specify arbitrary quantile levels when making predictions, providing more flexibility for different use cases.

## Feature Behavior

### Experiment Mode
In **experiment mode**, custom quantiles are fully supported and used to generate synthetic predictions:
- The provided quantiles are used to calculate the corresponding values from a normal distribution
- The mean is the `exp_runtime` value, and standard deviation is 5% of the mean
- Any valid quantile levels between 0 and 1 (exclusive) are accepted

### Normal Mode
In **normal mode** (with trained models), custom quantiles are **ignored**:
- The prediction uses the quantiles that were configured during model training
- This ensures consistency and prevents runtime errors from mismatched quantile dimensions
- The custom quantiles parameter is simply ignored without any error

## API Usage

### REST API

Add the optional `quantiles` field to your prediction request:

```json
POST /predict
{
  "model_id": "my_model",
  "platform_info": {
    "software_name": "exp",
    "software_version": "exp",
    "hardware_name": "exp"
  },
  "prediction_type": "quantile",
  "features": {
    "exp_runtime": 1000.0,
    "feature1": 10
  },
  "quantiles": [0.25, 0.5, 0.75, 0.9, 0.95]  // Custom quantiles
}
```

Response:
```json
{
  "model_id": "my_model",
  "platform_info": {...},
  "prediction_type": "quantile",
  "result": {
    "quantiles": {
      "0.25": 968.87,
      "0.5": 1002.23,
      "0.75": 1034.15,
      "0.9": 1065.42,
      "0.95": 1084.38
    }
  }
}
```

### WebSocket API

The WebSocket endpoint `/ws/predict` accepts the same request format:

```python
import websocket
import json

ws = websocket.WebSocket()
ws.connect("ws://localhost:8000/ws/predict")

request = {
    "model_id": "my_model",
    "platform_info": {
        "software_name": "exp",
        "software_version": "exp",
        "hardware_name": "exp"
    },
    "prediction_type": "quantile",
    "features": {
        "exp_runtime": 500.0
    },
    "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9]
}

ws.send(json.dumps(request))
response = json.loads(ws.recv())
```

## Validation Rules

- Quantile values must be strictly between 0 and 1 (exclusive)
- Invalid values will result in a 422 Validation Error
- Empty quantiles list `[]` is allowed and returns empty results
- If `quantiles` field is omitted, default quantiles are used: `[0.5, 0.9, 0.95, 0.99]`

## Examples

### Example 1: Custom Risk Levels
```python
# Request specific risk quantiles for SLA analysis
quantiles = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
```

### Example 2: Percentile Grid
```python
# Request deciles (10th, 20th, ... 90th percentiles)
quantiles = [i/10 for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
```

### Example 3: Focused Range
```python
# Focus on high-risk tail
quantiles = [0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
```

## Implementation Details

### Code Changes

1. **Model Update** (`src/models.py`):
   - Added optional `quantiles: Optional[List[float]]` field to `PredictionRequest`
   - Added validation to ensure quantiles are between 0 and 1

2. **API Updates** (`src/api.py`):
   - REST endpoint `/predict` passes custom quantiles to experiment mode
   - WebSocket endpoint `/ws/predict` passes custom quantiles to experiment mode
   - Normal mode predictions ignore the custom quantiles parameter

3. **Experiment Mode** (`src/utils/experiment.py`):
   - `generate_quantile_prediction()` accepts optional quantiles parameter
   - Uses provided quantiles or defaults to `[0.5, 0.9, 0.95, 0.99]`

## Testing

The feature includes comprehensive tests in `tests/test_custom_quantiles.py`:
- Experiment mode with custom quantiles
- Normal mode ignoring custom quantiles
- Invalid quantile validation
- Empty quantiles handling
- WebSocket support
- Monotonicity of quantile values

Run tests with:
```bash
uv run pytest tests/test_custom_quantiles.py -v
```

## Migration Notes

This is a **backward-compatible** feature:
- Existing API calls without the `quantiles` field continue to work
- The field is optional and defaults to standard quantiles when omitted
- Normal mode behavior is unchanged