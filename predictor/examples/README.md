# V2 Preprocessor API Examples

Step-by-step examples demonstrating the V2 Preprocessor API for both
Library API and HTTP API usage.

## Prerequisites

```bash
cd predictor
uv sync  # Install dependencies
```

## Quick Start

### Run All Examples

```bash
# Library API examples (no server needed)
uv run python examples/library_api/run_all.py

# HTTP API examples (server auto-started)
uv run python examples/http_api/run_all.py
```

### Run Individual Examples

```bash
# Library API
uv run python examples/library_api/ex01_train_without_preprocessor.py
uv run python examples/library_api/ex02_train_with_preprocessor.py
uv run python examples/library_api/ex03_predict_without_preprocessor.py
uv run python examples/library_api/ex04_predict_with_preprocessor.py
uv run python examples/library_api/ex05_error_handling.py
uv run python examples/library_api/ex06_collect_update_same_chain.py
uv run python examples/library_api/ex07_collect_retrain_new_chain.py
uv run python examples/library_api/ex08_custom_preprocessor.py

# HTTP API
uv run python examples/http_api/ex01_train_without_preprocessor.py
uv run python examples/http_api/ex02_train_with_preprocessor.py
uv run python examples/http_api/ex03_predict_without_preprocessor.py
uv run python examples/http_api/ex04_predict_with_preprocessor.py
uv run python examples/http_api/ex05_error_handling.py
uv run python examples/http_api/ex06_collect_update_same_chain.py
uv run python examples/http_api/ex07_collect_retrain_new_chain.py
uv run python examples/http_api/ex08_custom_preprocessor.py
```

## Example Overview

| # | Name | Description |
|---|------|-------------|
| 01 | Train without preprocessor | Basic training with raw features |
| 02 | Train with preprocessor | Training with chain (compute_cost = batch × seq) |
| 03 | Predict without preprocessor | Prediction on model trained without chain |
| 04 | Predict with preprocessor | Prediction with auto-applied stored chain |
| 05 | Error handling | Common errors and how to handle them |
| 06 | Collect & update (same chain) | Incremental training with unchanged chain |
| 07 | Collect & retrain (new chain) | Full retrain when chain changes |
| 08 | Custom preprocessor | Define and use custom preprocessor class |

## Directory Structure

```
examples/
├── README.md                          # This file
├── library_api/                       # Library API examples
│   ├── __init__.py
│   ├── utils.py                       # Shared utilities
│   ├── ex01_train_without_preprocessor.py
│   ├── ex02_train_with_preprocessor.py
│   ├── ex03_predict_without_preprocessor.py
│   ├── ex04_predict_with_preprocessor.py
│   ├── ex05_error_handling.py
│   ├── ex06_collect_update_same_chain.py
│   ├── ex07_collect_retrain_new_chain.py
│   ├── ex08_custom_preprocessor.py
│   └── run_all.py                     # All-in-one runner
└── http_api/                          # HTTP API examples
    ├── __init__.py
    ├── utils.py                       # Server management, HTTP helpers
    ├── ex01_train_without_preprocessor.py
    ├── ex02_train_with_preprocessor.py
    ├── ex03_predict_without_preprocessor.py
    ├── ex04_predict_with_preprocessor.py
    ├── ex05_error_handling.py
    ├── ex06_collect_update_same_chain.py
    ├── ex07_collect_retrain_new_chain.py
    ├── ex08_custom_preprocessor.py
    └── run_all.py                     # All-in-one runner
```

## Data Domain

All examples use ML inference simulation:

- **Features**: `batch_size`, `sequence_length`, `hidden_size`
- **Preprocessing**: `compute_cost = batch_size × sequence_length`
- **Target**: `runtime_ms` (simulated with realistic correlation)

### Sample Training Data

```python
{
    "batch_size": 32,
    "sequence_length": 512,
    "hidden_size": 768,
    "runtime_ms": 45.2  # Correlated with features
}
```

### Preprocessing Chain Example

```python
# Transformation: {batch_size, sequence_length, hidden_size}
#              → {compute_cost, hidden_size}

chain = (
    PreprocessorChainV2(name="compute_chain")
    .add(MultiplyPreprocessor("batch_size", "sequence_length", "compute_cost"))
    .add(RemoveFeaturePreprocessor(["batch_size", "sequence_length"]))
)
```

## Expected Output

### Example 01 (Library API)

```
================================================================================
  Example 1: Train Without Preprocessor
================================================================================

Step 1: Prepare Training Data
  Generated 25 samples with features:
    batch_size, sequence_length, hidden_size, runtime_ms

Step 2: Train Model
  Model trained: no-preprocess-model
  Samples: 25
  Features: ['batch_size', 'hidden_size', 'sequence_length']
  Chain stored: None

Step 3: Verify Model
  Model ID: no-preprocess-model
  Platform: PyTorch-2.0-NVIDIA_A100
  Prediction Type: expect_error

✓ Example 1 completed successfully!
```

### Example 02 (Library API)

```
================================================================================
  Example 2: Train With Preprocessor
================================================================================

Step 1: Create Preprocessing Chain
  Chain Name: compute_chain
  Steps:
    1. multiply(batch_size, sequence_length) → compute_cost
    2. remove([batch_size, sequence_length])

Step 2: Transformation Preview
  Input:  {'batch_size': 32, 'sequence_length': 512, 'hidden_size': 768}
  Output: {'compute_cost': 16384, 'hidden_size': 768}

Step 3: Train Model with Chain
  Model trained: compute-predictor
  Samples: 25
  Features after chain: ['compute_cost', 'hidden_size']

Step 4: Verify Chain Storage
  Chain stored with model
  Features: compute_cost, hidden_size

✓ Example 2 completed successfully!
```

### All-in-One Runner Output

```
======================================================================
  V2 API Examples - Library API
  All-in-One Runner
======================================================================

  This script runs all 8 Library API examples in sequence.
  Each example uses isolated storage that is cleaned up after.

[01] Train without preprocessor
--------------------------------------------------
... (example output) ...

  Result: PASSED

----------------------------------------------------------------------

[02] Train with preprocessor
--------------------------------------------------
... (example output) ...

  Result: PASSED

... (more examples) ...

======================================================================
  Summary
======================================================================

  All 8 examples passed!

======================================================================
```

## API Comparison

### Library API

Use when:
- Need maximum flexibility
- Building Python applications
- Want custom preprocessing
- Performance is critical

```python
from src.core import PredictorCore
from src.low_level import PredictorLowLevel
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

# Create chain
chain = PreprocessorChainV2().add(MultiplyPreprocessor(...))

# Train
api = PredictorLowLevel()
predictor = api.train_predictor_with_pipeline_v2(features_list, "quantile", chain=chain)
api.save_model(model_id, platform, "quantile", predictor)

# Predict
core = PredictorCore()
result = core.inference_pipeline_v2(model_id, platform, "quantile", features, chain=chain)
```

### HTTP API

Use when:
- Using only built-in preprocessors
- Building microservices architecture
- Need language-agnostic access
- Deployment requires network isolation

```python
import requests

# Train with chain config
response = requests.post(f"{base_url}/v2/train", json={
    "model_id": "my-model",
    "platform_info": {"software_name": "PyTorch", ...},
    "prediction_type": "quantile",
    "features_list": [...],
    "preprocess_chain": {
        "steps": [
            {"name": "multiply", "params": {...}},
            {"name": "remove", "params": {...}},
        ]
    }
})

# Predict (chain auto-applied from stored)
response = requests.post(f"{base_url}/v2/predict", json={
    "model_id": "my-model",
    "platform_info": {...},
    "prediction_type": "quantile",
    "features": {"batch_size": 64, "sequence_length": 1024, ...}
})
```

## Troubleshooting

### Port Already in Use (HTTP API)

HTTP examples find an available port automatically. If you encounter issues:

```bash
# Check for processes on port 8765
lsof -i :8765

# Kill if needed
kill -9 <PID>
```

### Server Startup Timeout (HTTP API)

If the server takes too long to start, you can increase the timeout in the example:

```python
with ServerManager(example_name="...", startup_timeout=60) as base_url:
    ...
```

### Storage Directory Issues

Each example uses isolated storage in `/tmp/predictor_examples_*`. If you encounter
permission issues or stale data:

```bash
# Clean up example storage directories
rm -rf /tmp/predictor_examples_*
```

### Import Errors

Ensure you're running from the `predictor` directory:

```bash
cd /path/to/swarmpilot-refresh/predictor
uv run python examples/library_api/run_all.py
```

## Custom Preprocessor Guide

### Creating a Custom Preprocessor

```python
from src.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
    OperationType,
)

class LogTransformPreprocessor(BasePreprocessorV2):
    """Apply log10 transformation to a numeric feature."""

    def __init__(self, input_feature: str, output_feature: str = None):
        self._input_feature = input_feature
        self._output_feature = output_feature or f"log_{input_feature}"
        self._name = f"log_transform_{input_feature}"
        self._operation_type = OperationType.TRANSFORM

    @property
    def name(self) -> str:
        return self._name

    @property
    def operation_type(self) -> OperationType:
        return self._operation_type

    @property
    def input_features(self) -> list[str]:
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        return [self._output_feature]

    @property
    def removes_features(self) -> list[str]:
        return []  # Or [self._input_feature] if you want to remove input

    def transform(self, context: FeatureContext) -> None:
        import math
        value = context.get(self._input_feature)
        context.set(self._output_feature, math.log10(max(float(value), 1.0)))
```

### Using Custom Preprocessor in Chain

```python
chain = (
    PreprocessorChainV2(name="custom_chain")
    .add(MultiplyPreprocessor("batch_size", "sequence_length", "compute_cost"))
    .add(LogTransformPreprocessor("compute_cost", "log_compute"))
    .add(RemoveFeaturePreprocessor(["batch_size", "sequence_length", "compute_cost"]))
)
```

### Note on HTTP API

Custom preprocessors require server-side registration. For HTTP usage, you have two options:

1. **Use built-in preprocessors** in creative combinations
2. **Register custom preprocessors** on the server before starting:

```python
# In src/api/dependencies.py
from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

registry = PreprocessorsRegistryV2()
registry.register_factory(
    "log_transform",
    lambda **p: LogTransformPreprocessor(**p)
)
```

## Built-in Preprocessors

| Name | Description | Parameters |
|------|-------------|------------|
| `multiply` | Multiplies two features | `feature_a`, `feature_b`, `output_feature` |
| `remove` | Removes specified features | `features_to_remove` (list) |
| `token_length` | Counts tokens in text | `input_feature`, `output_feature` |

## License

Part of the Swarmpilot project.
