# Predictor V2 Preprocessing API

This document describes the V2 preprocessing system for the predictor module.
The V2 API provides flexible, chainable preprocessing with support for feature
modification, addition, and removal.

## Overview

### V1 vs V2 Comparison

| Feature | V1 API | V2 API |
|---------|--------|--------|
| Configuration | `preprocess_config: dict[str, list[str]]` | `preprocess_chain: ChainConfigV2` |
| Feature Handling | Process single feature at a time | Modify/add/remove any features |
| Chaining | Per-feature chains | Global preprocessing chain |
| Chain Storage | Not stored with model | Stored with model, used automatically |
| HTTP Endpoints | `/train`, `/predict` | `/v2/train`, `/v2/predict` |

### Key Design Decisions

1. **Chain set at training time only**: The preprocessing chain is provided during
   training and stored with the model metadata.

2. **Prediction uses stored chain**: The `/v2/predict` endpoint automatically applies
   the chain stored with the model - no chain in the prediction request.

3. **Chain validation**: Before training begins, the chain is validated on the first
   sample. If validation fails, training is rejected with a detailed error.

4. **Fail-fast error handling**: Errors are raised immediately with detailed
   information about which preprocessor failed and why.

---

## Library API

### PredictorLowLevel V2 Methods

```python
from src.api.core import PredictorLowLevel
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.preprocessors_v2 import (
    MultiplyPreprocessor,
    RemoveFeaturePreprocessor,
    TokenLengthPreprocessor,
)

low = PredictorLowLevel()

# Create a preprocessing chain
chain = (
    PreprocessorChainV2(name="image_pipeline")
    .add(MultiplyPreprocessor("width", "height", "pixel_num"))
    .add(RemoveFeaturePreprocessor(["width", "height"]))
    .add(TokenLengthPreprocessor("prompt", output_feature="input_length"))
    .add(RemoveFeaturePreprocessor(["prompt"]))
)

# Apply chain to features
result = low.apply_preprocess_pipeline_v2(
    features={"width": 100, "height": 200, "prompt": "Hello world"},
    chain=chain,
)
# result: {"pixel_num": 20000, "input_length": 2}

# Train with preprocessing
predictor = low.train_predictor_with_pipeline_v2(
    features_list=training_data,  # List of {features..., runtime_ms: float}
    prediction_type="quantile",
    config=None,
    chain=chain,
)

# Predict with preprocessing
result = low.predict_with_pipeline_v2(
    predictor=predictor,
    features={"width": 50, "height": 100, "prompt": "Test"},
    chain=chain,
)
```

### PredictorCore V2 Method

```python
from src.api.core import PredictorCore
from src.models import PlatformInfo

core = PredictorCore()
platform = PlatformInfo(
    software_name="PyTorch",
    software_version="2.0",
    hardware_name="NVIDIA A100"
)

# Complete inference pipeline with V2 preprocessing
result = core.inference_pipeline_v2(
    model_id="my_model",
    platform_info=platform,
    prediction_type="quantile",
    features={"width": 100, "height": 200, "prompt": "Hello"},
    chain=chain,  # Optional - if None, no preprocessing
)
```

---

## Preprocessing Components

### FeatureContext

The mutable context passed through preprocessors. Tracks feature state changes.

```python
from src.preprocessor.base_preprocessor_v2 import FeatureContext

context = FeatureContext(features={"x": 1, "y": 2})

# Get features
value = context.get("x")          # Returns 1
value = context.get("z", 0)       # Returns 0 (default)

# Set features (tracks add/modify)
context.set("z", 3)               # Adds new feature
context.set("x", 10)              # Modifies existing feature

# Remove features (tracks removal)
removed = context.remove("y")     # Returns 2, removes "y"

# Check existence
exists = context.has("x")         # Returns True

# Access final state
final = context.features          # {"x": 10, "z": 3}

# Tracking attributes
context.added_features            # {"z"}
context.modified_features         # {"x"}
context.removed_features          # {"y"}
context.original_features         # {"x": 1, "y": 2} (immutable copy)
```

### BasePreprocessorV2

Abstract base class for all V2 preprocessors.

```python
from abc import ABC, abstractmethod
from src.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
    OperationType,
)

class MyPreprocessor(BasePreprocessorV2):
    def __init__(self, input_name: str, output_name: str) -> None:
        super().__init__(name="my_preprocessor", operation_type=OperationType.TRANSFORM)
        self._input = input_name
        self._output = output_name

    @property
    def input_features(self) -> list[str]:
        return [self._input]

    @property
    def output_features(self) -> list[str]:
        return [self._output]

    def transform(self, context: FeatureContext) -> None:
        value = context.get(self._input)
        context.set(self._output, value * 2)
```

**OperationType enum:**
- `MODIFY` - Modifies existing features in place
- `ADD` - Adds new computed features
- `REMOVE` - Removes specified features
- `TRANSFORM` - General transformation (default)

### Built-in Preprocessors

#### MultiplyPreprocessor

Multiplies two features and stores the result.

```python
from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

prep = MultiplyPreprocessor(
    feature_a="width",
    feature_b="height",
    output_feature="pixels",
    remove_inputs=False,  # Default: keep inputs
)

# Input:  {"width": 100, "height": 200, "other": 1}
# Output: {"width": 100, "height": 200, "other": 1, "pixels": 20000}

# With remove_inputs=True
prep = MultiplyPreprocessor("width", "height", "pixels", remove_inputs=True)
# Input:  {"width": 100, "height": 200, "other": 1}
# Output: {"other": 1, "pixels": 20000}
```

#### RemoveFeaturePreprocessor

Removes specified features from the context. Does not raise error if feature is missing.

```python
from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

prep = RemoveFeaturePreprocessor(features_to_remove=["temp1", "temp2"])

# Input:  {"x": 1, "temp1": "a", "temp2": "b"}
# Output: {"x": 1}

# Missing features are silently ignored
# Input:  {"x": 1, "temp1": "a"}  # temp2 missing
# Output: {"x": 1}  # No error
```

#### TokenLengthPreprocessor

Computes token count from a text feature.

```python
from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

# Default tokenizer splits on whitespace
prep = TokenLengthPreprocessor(
    input_feature="prompt",
    output_feature="input_length",  # Default
    remove_input=False,  # Default: keep input
)

# Input:  {"prompt": "Hello world", "x": 1}
# Output: {"prompt": "Hello world", "x": 1, "input_length": 2}

# Custom tokenizer
prep = TokenLengthPreprocessor(
    input_feature="text",
    tokenizer=lambda s: list(s),  # Character-level
)

# With remove_input=True
prep = TokenLengthPreprocessor("prompt", "input_length", remove_input=True)
# Input:  {"prompt": "Hello world", "x": 1}
# Output: {"x": 1, "input_length": 2}
```

### PreprocessorChainV2

Orchestrates sequential execution of preprocessors.

```python
from src.preprocessor.chain_v2 import PreprocessorChainV2

chain = PreprocessorChainV2(name="my_pipeline")

# Fluent interface for building chains
chain = (
    PreprocessorChainV2(name="image_pipeline")
    .add(MultiplyPreprocessor("w", "h", "pixels"))
    .add(RemoveFeaturePreprocessor(["w", "h"]))
)

# Insert at specific position
chain.insert(0, TokenLengthPreprocessor("prompt"))

# Validate chain against initial features
errors = chain.validate(initial_features={"w", "h", "prompt"})
if errors:
    print("Validation errors:", errors)

# Get required inputs (features needed from outside)
required = chain.get_required_inputs()

# Get final outputs (features produced by chain)
outputs = chain.get_final_outputs()

# Transform features
result = chain.transform({"w": 10, "h": 20, "prompt": "Hi"})

# Callable interface
result = chain({"w": 10, "h": 20, "prompt": "Hi"})

# Access chain properties
chain.name                # "image_pipeline"
chain.preprocessors       # List of preprocessors in order
len(chain.preprocessors)  # Number of preprocessors
```

### PreprocessorsRegistryV2

Registry for managing preprocessors and creating chains from config.

```python
from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

registry = PreprocessorsRegistryV2()

# Built-in preprocessors are auto-registered: multiply, remove, token_length

# Get preprocessor by name with parameters
prep = registry.get("multiply", feature_a="w", feature_b="h", output_feature="p")

# Register custom preprocessor instance
registry.register("custom", my_preprocessor_instance)

# Register factory function
registry.register_factory("custom_factory", lambda **kwargs: MyPreprocessor(**kwargs))

# Create chain from configuration
config = [
    {"name": "multiply", "params": {"feature_a": "w", "feature_b": "h", "output_feature": "p"}},
    {"name": "remove", "params": {"features_to_remove": ["w", "h"]}},
    {"name": "token_length", "params": {"input_feature": "prompt"}},
]
chain = registry.create_chain_from_config(config, chain_name="my_pipeline")

# List available preprocessors
available = registry.list_available()  # ["multiply", "remove", "token_length", ...]

# Register and retrieve chains
registry.register_chain("image_pipeline", chain)
retrieved = registry.get_chain("image_pipeline")
```

---

## V1 to V2 Adapter

Use `V1PreprocessorAdapter` to wrap existing V1 preprocessors for use in V2 chains.

```python
from src.preprocessor.adapters import V1PreprocessorAdapter
from src.preprocessor.semantic_predictor import SemanticPredictor

# Wrap V1 preprocessor
v1_preprocessor = SemanticPredictor(model_path="...", config_path="...")
adapter = V1PreprocessorAdapter(
    v1_preprocessor=v1_preprocessor,
    input_feature="prompt",
    name="semantic",  # Optional custom name
)

# Use in V2 chain
chain = (
    PreprocessorChainV2(name="with_semantic")
    .add(adapter)
    .add(RemoveFeaturePreprocessor(["prompt"]))
)
```

**Adapter behavior:**
- Calls `v1_preprocessor([input_value])` and merges returned dict into context
- Respects V1 preprocessor's `remove_origin` flag
- Tracks added/removed features in context

---

## HTTP API V2

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/collect` | POST | Collect training samples (optional preprocessing) |
| `/v2/train` | POST | Train model with preprocessing chain |
| `/v2/predict` | POST | Predict using stored chain (no chain in request) |

### Chain Configuration Format

The HTTP API uses JSON configuration for preprocessing chains:

```json
{
  "steps": [
    {
      "name": "multiply",
      "params": {
        "feature_a": "width",
        "feature_b": "height",
        "output_feature": "pixel_num"
      }
    },
    {
      "name": "remove",
      "params": {
        "features_to_remove": ["width", "height"]
      }
    },
    {
      "name": "token_length",
      "params": {
        "input_feature": "prompt",
        "output_feature": "input_length"
      }
    }
  ]
}
```

### POST /v2/collect

Collect a training sample with optional preprocessing.

**Request:**
```json
{
  "model_id": "my_model",
  "platform_info": {
    "software_name": "PyTorch",
    "software_version": "2.0",
    "hardware_name": "NVIDIA A100"
  },
  "prediction_type": "quantile",
  "features": {
    "width": 100,
    "height": 200,
    "prompt": "Hello world"
  },
  "runtime_ms": 42.5,
  "preprocess_chain": {
    "steps": [...]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "samples_collected": 15,
  "message": "Sample collected. Total: 15 samples."
}
```

### POST /v2/train

Train a model with a preprocessing chain. The chain is validated and stored with the model.

**Request:**
```json
{
  "model_id": "my_model",
  "platform_info": {
    "software_name": "PyTorch",
    "software_version": "2.0",
    "hardware_name": "NVIDIA A100"
  },
  "prediction_type": "quantile",
  "features_list": [
    {"width": 100, "height": 200, "prompt": "Hello", "runtime_ms": 42.5},
    {"width": 50, "height": 100, "prompt": "World", "runtime_ms": 21.0}
  ],
  "training_config": null,
  "preprocess_chain": {
    "steps": [
      {"name": "multiply", "params": {"feature_a": "width", "feature_b": "height", "output_feature": "pixels"}},
      {"name": "remove", "params": {"features_to_remove": ["width", "height"]}}
    ]
  }
}
```

**Response (success):**
```json
{
  "status": "success",
  "message": "Model trained with 20 samples (10 collected + 10 from request)",
  "model_key": "my_model__PyTorch-2.0__NVIDIA_A100__quantile",
  "samples_trained": 20,
  "chain_stored": true
}
```

**Response (chain validation error - HTTP 400):**
```json
{
  "error": "Chain validation failed",
  "message": "Preprocessor 'multiply' missing required features: ['width']. Available features: ['height', 'prompt']"
}
```

### POST /v2/predict

Make a prediction. Uses the stored chain automatically.

**Request:**
```json
{
  "model_id": "my_model",
  "platform_info": {
    "software_name": "PyTorch",
    "software_version": "2.0",
    "hardware_name": "NVIDIA A100"
  },
  "prediction_type": "quantile",
  "features": {
    "width": 100,
    "height": 200,
    "prompt": "Hello world"
  },
  "quantiles": [0.5, 0.9, 0.99]
}
```

**Note:** The `preprocess_chain` field is NOT accepted in prediction requests.
The chain stored with the model during training is applied automatically.

**Response:**
```json
{
  "model_id": "my_model",
  "platform_info": {
    "software_name": "PyTorch",
    "software_version": "2.0",
    "hardware_name": "NVIDIA A100"
  },
  "prediction_type": "quantile",
  "result": {
    "quantile_0.5": 45.2,
    "quantile_0.9": 78.3,
    "quantile_0.99": 102.1
  },
  "chain_applied": true
}
```

---

## Pydantic Models Reference

### Request Models

| Model | Endpoint | Description |
|-------|----------|-------------|
| `CollectRequestV2` | `/v2/collect` | Collect sample with optional chain |
| `TrainingRequestV2` | `/v2/train` | Train with optional chain |
| `PredictionRequestV2` | `/v2/predict` | Predict (no chain accepted) |

### Response Models

| Model | Endpoint | Key Fields |
|-------|----------|------------|
| `CollectResponseV2` | `/v2/collect` | `status`, `samples_collected`, `message` |
| `TrainingResponseV2` | `/v2/train` | `status`, `message`, `model_key`, `samples_trained`, `chain_stored` |
| `PredictionResponseV2` | `/v2/predict` | `model_id`, `platform_info`, `prediction_type`, `result`, `chain_applied` |

### Configuration Models

| Model | Description |
|-------|-------------|
| `PreprocessorStepConfigV2` | Single step: `name`, `params` |
| `ChainConfigV2` | Chain config: `steps` (list of steps) |
| `ChainValidationErrorV2` | Error details: `step_index`, `preprocessor_name`, `error` |

---

## Workflow Examples

### Example 1: Image Processing Pipeline

```python
# Training workflow
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.preprocessors_v2 import (
    MultiplyPreprocessor,
    RemoveFeaturePreprocessor,
    TokenLengthPreprocessor,
)
from src.api.core import PredictorLowLevel

# Create preprocessing chain
chain = (
    PreprocessorChainV2(name="image_pipeline")
    .add(MultiplyPreprocessor("width", "height", "pixel_num"))
    .add(RemoveFeaturePreprocessor(["width", "height"]))
    .add(TokenLengthPreprocessor("prompt", output_feature="input_length"))
    .add(RemoveFeaturePreprocessor(["prompt"]))
)

# Training data
training_data = [
    {"width": 100, "height": 200, "prompt": "cat", "channels": 3, "runtime_ms": 42.5},
    {"width": 50, "height": 100, "prompt": "dog sitting", "channels": 3, "runtime_ms": 21.0},
    # ... more samples
]

# Train
low = PredictorLowLevel()
predictor = low.train_predictor_with_pipeline_v2(
    features_list=training_data,
    prediction_type="quantile",
    chain=chain,
)

# Prediction
result = low.predict_with_pipeline_v2(
    predictor=predictor,
    features={"width": 80, "height": 160, "prompt": "bird flying", "channels": 3},
    chain=chain,
)
```

### Example 2: Using Registry for Configuration-Driven Chains

```python
from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

registry = PreprocessorsRegistryV2()

# Configuration from external source (e.g., YAML, JSON, database)
config = [
    {"name": "multiply", "params": {"feature_a": "width", "feature_b": "height", "output_feature": "pixels"}},
    {"name": "remove", "params": {"features_to_remove": ["width", "height"]}},
]

# Create chain from config
chain = registry.create_chain_from_config(config, chain_name="image_pipeline")

# Validate before use
errors = chain.validate(initial_features={"width", "height", "channels"})
if errors:
    raise ValueError(f"Invalid chain configuration: {errors}")

# Use chain...
```

### Example 3: HTTP API Workflow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Collect training samples
for sample in training_data:
    response = requests.post(f"{BASE_URL}/v2/collect", json={
        "model_id": "my_model",
        "platform_info": {"software_name": "PyTorch", "software_version": "2.0", "hardware_name": "GPU"},
        "prediction_type": "quantile",
        "features": sample["features"],
        "runtime_ms": sample["runtime_ms"],
    })
    print(f"Collected: {response.json()['samples_collected']} samples")

# 2. Train with preprocessing chain
chain_config = {
    "steps": [
        {"name": "multiply", "params": {"feature_a": "width", "feature_b": "height", "output_feature": "pixels"}},
        {"name": "remove", "params": {"features_to_remove": ["width", "height"]}},
    ]
}

response = requests.post(f"{BASE_URL}/v2/train", json={
    "model_id": "my_model",
    "platform_info": {"software_name": "PyTorch", "software_version": "2.0", "hardware_name": "GPU"},
    "prediction_type": "quantile",
    "preprocess_chain": chain_config,
})
print(f"Trained: {response.json()['samples_trained']} samples, chain_stored: {response.json()['chain_stored']}")

# 3. Predict (chain applied automatically)
response = requests.post(f"{BASE_URL}/v2/predict", json={
    "model_id": "my_model",
    "platform_info": {"software_name": "PyTorch", "software_version": "2.0", "hardware_name": "GPU"},
    "prediction_type": "quantile",
    "features": {"width": 100, "height": 200, "channels": 3},
})
print(f"Prediction: {response.json()['result']}, chain_applied: {response.json()['chain_applied']}")
```

### Example 4: Adapting V1 Preprocessors

```python
from src.preprocessor.adapters import V1PreprocessorAdapter
from src.preprocessor.semantic_predictor import SemanticPredictor

# Load existing V1 semantic predictor
v1_semantic = SemanticPredictor(
    model_path="models/semantic.pt",
    config_path="models/semantic_config.json",
)

# Wrap for V2 usage
semantic_adapter = V1PreprocessorAdapter(
    v1_preprocessor=v1_semantic,
    input_feature="prompt",
    name="semantic_v2",
)

# Build chain with adapter
chain = (
    PreprocessorChainV2(name="nlp_pipeline")
    .add(TokenLengthPreprocessor("prompt", output_feature="input_length"))
    .add(semantic_adapter)
    .add(RemoveFeaturePreprocessor(["prompt"]))
)
```

---

## Error Handling

The V2 system uses fail-fast error handling. Errors are raised immediately with
detailed information.

### Common Errors

1. **Missing input feature** (raised during chain execution):
   ```
   ValueError: Preprocessor 'multiply' requires feature 'width' which is not available.
   Available features: ['height', 'prompt']
   ```

2. **Unknown preprocessor in registry**:
   ```
   ValueError: Unknown preprocessor 'unknown'. Available: ['multiply', 'remove', 'token_length']
   ```

3. **Chain validation failure** (HTTP 400):
   ```json
   {
     "error": "Chain validation failed",
     "message": "Preprocessor 'multiply' missing required features: ['width']. Available features: ['height', 'prompt']"
   }
   ```

4. **Model not found** (HTTP 404):
   ```json
   {
     "error": "Model not found",
     "message": "Model not found: my_model__PyTorch-2.0__GPU__quantile"
   }
   ```

---

## Migration from V1 to V2

### Step 1: Convert preprocess_config to chain config

**V1:**
```python
preprocess_config = {
    "prompt": ["semantic"],
}
```

**V2:**
```python
config = [
    {"name": "semantic", "params": {"input_feature": "prompt"}},  # Using V1 adapter
]
```

Or using built-in preprocessors:
```python
config = [
    {"name": "token_length", "params": {"input_feature": "prompt", "output_feature": "input_length"}},
    {"name": "remove", "params": {"features_to_remove": ["prompt"]}},
]
```

### Step 2: Use V2 endpoints

Replace `/train` with `/v2/train` and `/predict` with `/v2/predict`.

### Step 3: Remove chain from predict requests

The V2 predict endpoint uses the stored chain automatically. Remove any
`preprocess_config` or `preprocess_chain` from prediction requests.

### Step 4: Test thoroughly

Run both V1 and V2 workflows to ensure backward compatibility.

---

## Best Practices

1. **Validate chains before training**: Use `chain.validate()` to catch configuration
   errors early.

2. **Use the registry for configuration-driven chains**: The registry makes it easy
   to build chains from external configuration (JSON, YAML, database).

3. **Store chain config in version control**: Since chains affect model behavior,
   track chain configurations alongside model versions.

4. **Test chains independently**: Use `chain.transform()` to test preprocessing
   logic before integrating with training.

5. **Use meaningful chain names**: Names help with debugging and logging.

6. **Check `chain_applied` in responses**: The prediction response includes
   `chain_applied: true/false` to confirm preprocessing was applied.

7. **Handle missing features gracefully**: Use `RemoveFeaturePreprocessor` which
   silently ignores missing features, or validate inputs before calling chains.

---

## File Structure

```
src/preprocessor/
├── base_preprocessor.py          # V1 base (unchanged)
├── base_preprocessor_v2.py       # V2: FeatureContext, BasePreprocessorV2, OperationType
├── preprocessors_v2.py           # V2: Multiply, Remove, TokenLength
├── chain_v2.py                   # V2: PreprocessorChainV2
├── registry_v2.py                # V2: PreprocessorsRegistryV2
├── adapters.py                   # V1PreprocessorAdapter
├── semantic_predictor.py         # V1 semantic (unchanged)
└── preprocessors_registry.py     # V1 registry (unchanged)

src/api/routes/
├── training.py                   # V1: /train, /collect
├── training_v2.py                # V2: /v2/train, /v2/collect
├── prediction.py                 # V1: /predict
├── prediction_v2.py              # V2: /v2/predict
└── helpers.py                    # Shared exception handling

src/models.py                     # V2 Pydantic models (added)
```
