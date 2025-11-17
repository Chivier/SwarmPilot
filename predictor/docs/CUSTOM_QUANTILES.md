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

## Monotonicity Enforcement

### Overview

Quantile predictions should satisfy the monotonicity constraint: q₀.₅ ≤ q₀.₉ ≤ q₀.₉₅ ≤ q₀.₉₉. The predictor now supports two mechanisms to ensure this ordering:

1. **Training-time penalty** (soft constraint)
2. **Prediction-time enforcement** (hard constraint)

### Training-Time Monotonicity Penalty

Add a penalty term during training to encourage monotonic predictions:

```python
from src.predictor.quantile import QuantilePredictor

predictor = QuantilePredictor()
training_data = [...]

# Train with monotonicity penalty
config = {
    'epochs': 500,
    'learning_rate': 0.01,
    'quantiles': [0.5, 0.9, 0.95, 0.99],
    'monotonicity_penalty': 1.0  # Penalty weight (0.1 to 10.0 typical range)
}

predictor.train(training_data, config=config)
```

**How it works:**
- Adds a penalty term to the loss function: `Loss = Pinball Loss + α × mean(max(0, qᵢ - qᵢ₊₁))`
- Only penalizes violations where lower quantile > higher quantile
- Higher penalty values enforce stricter monotonicity during training
- Default value is 0.0 (no penalty, backward compatible)

**Recommended values:**
- `monotonicity_penalty=0.0`: No enforcement (default)
- `monotonicity_penalty=0.1-1.0`: Soft encouragement
- `monotonicity_penalty=1.0-5.0`: Moderate enforcement
- `monotonicity_penalty=5.0-10.0`: Strong enforcement

### Prediction-Time Monotonicity Enforcement

Apply strict monotonicity as a post-processing step:

```python
# Predict with strict monotonicity guarantee
result = predictor.predict(
    features={'batch_size': 25, 'sequence_length': 128},
    enforce_monotonicity=True
)

# Result is guaranteed to be monotonic
quantiles = result['quantiles']
# q₀.₅ ≤ q₀.₉ ≤ q₀.₉₅ ≤ q₀.₉₉ (guaranteed)
```

**How it works:**
- Uses isotonic regression approach
- For any violation qᵢ > qᵢ₊₁, sets both to their average
- Minimizes changes to predictions while ensuring monotonicity
- Applied after denormalization, before returning results

**When to use:**
- When you need guaranteed monotonicity regardless of training
- When using low or zero monotonicity penalty during training
- For production systems with strict ordering requirements

### Choosing the Right Approach

| Approach | Pros | Cons | When to Use |
|----------|------|------|-------------|
| **Training Penalty** | - Model learns monotonicity naturally<br>- Better generalization<br>- No post-processing overhead | - Not a hard guarantee<br>- Requires retraining<br>- Need to tune penalty weight | - New models<br>- Flexibility in accuracy vs. monotonicity trade-off |
| **Prediction Enforcement** | - Guaranteed monotonicity<br>- Works with existing models<br>- No retraining needed | - Post-processing overhead<br>- May reduce prediction accuracy<br>- Doesn't improve underlying model | - Existing models<br>- Strict requirements<br>- Quick fix without retraining |
| **Both Combined** | - Best of both worlds<br>- Training guides the model<br>- Enforcement ensures guarantee | - Slight computational overhead | - Production systems<br>- High-stakes predictions |

### Examples

**Example 1: Train with moderate penalty**
```python
config = {
    'epochs': 500,
    'monotonicity_penalty': 1.0,
    'quantiles': [0.5, 0.9, 0.95, 0.99]
}
predictor.train(training_data, config=config)

# Predictions will likely be monotonic, but not guaranteed
result = predictor.predict(features)
```

**Example 2: Strict enforcement at prediction**
```python
# Train without penalty (default)
predictor.train(training_data)

# Enforce at prediction time
result = predictor.predict(features, enforce_monotonicity=True)
# Guaranteed monotonic
```

**Example 3: Combined approach (recommended for production)**
```python
# Train with penalty to guide learning
config = {'monotonicity_penalty': 1.0}
predictor.train(training_data, config=config)

# Enforce at prediction for guarantee
result = predictor.predict(features, enforce_monotonicity=True)
# Best accuracy with guaranteed monotonicity
```

### Testing Monotonicity

The test suite includes comprehensive monotonicity tests:

```bash
# Run monotonicity-specific tests
uv run pytest tests/test_predictors.py::TestQuantilePredictor::test_monotonicity_penalty_improves_ordering -v
uv run pytest tests/test_predictors.py::TestQuantilePredictor::test_enforce_monotonicity_parameter -v

# Run all quantile tests
uv run pytest tests/test_predictors.py::TestQuantilePredictor -v
```

## Migration Notes

This is a **backward-compatible** feature:
- Existing API calls without the `quantiles` field continue to work
- The field is optional and defaults to standard quantiles when omitted
- Normal mode behavior is unchanged
- Default `monotonicity_penalty=0.0` maintains existing behavior
- Default `enforce_monotonicity=False` maintains existing behavior