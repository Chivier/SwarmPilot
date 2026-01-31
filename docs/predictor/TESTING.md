# Testing Documentation

## Test Overview

The predictor service has comprehensive test coverage with **63 tests** achieving **88% code coverage**.

## Test Structure

### 1. Unit Tests (50 tests)

#### API Tests (`tests/test_api.py`) - 16 tests
- **Health Check**: Service availability
- **List Endpoint**: Model enumeration
- **Train Endpoint**: Model training workflows
- **Predict Endpoint**: Prediction functionality
- **Experiment Mode**: Synthetic predictions
- **Quantile Predictions**: Multi-quantile predictions

#### Experiment Mode Tests (`tests/test_experiment.py`) - 19 tests
- Experiment mode detection (exp_runtime, platform="exp")
- Runtime extraction and validation
- Synthetic prediction generation for both types
- Error handling for invalid inputs

#### Predictor Tests (`tests/test_predictors.py`) - 15 tests
- Expect/Error predictor training and prediction
- Quantile predictor with pinball loss
- Model serialization/deserialization
- Feature validation
- Monotonicity checks for quantiles

### 2. Integration Tests (`tests/test_integration.py`) - 13 tests

#### Complete Workflow Tests (2 tests)
- **test_expect_error_complete_workflow**: 
  - Service startup → Training → Prediction → Listing
  - Tests entire lifecycle with realistic data
  - Verifies data persistence
  
- **test_quantile_complete_workflow**:
  - End-to-end quantile prediction workflow
  - Custom quantile configuration
  - Monotonicity verification

#### Multi-Model Management (2 tests)
- **test_multiple_models_same_prediction_type**:
  - Same model_id on different platforms (CPU, GPU, TPU)
  - Independent model management
  - Cross-platform prediction verification

- **test_multiple_prediction_types**:
  - Different prediction types coexisting
  - Type-specific model behavior

#### Model Update & Retrain (2 tests)
- **test_incremental_training**:
  - Initial training with 20 samples
  - Update with 50 samples
  - Sample count verification

- **test_training_with_different_features**:
  - Feature set evolution
  - Old features fail after retrain
  - New features work correctly

#### Experiment Mode Integration (2 tests)
- **test_experiment_mode_without_trained_model**:
  - Experiment mode works without any models
  - No training required for testing

- **test_switch_between_normal_and_experiment_mode**:
  - Toggle between real and synthetic predictions
  - Same endpoint, different modes

#### Error Recovery (2 tests)
- **test_invalid_training_then_valid_training**:
  - Service recovers from insufficient samples
  - No partial model creation

- **test_prediction_with_wrong_platform_then_correct**:
  - Wrong platform returns 404
  - Correct platform succeeds

#### Real-World Scenarios (2 tests)
- **test_model_versioning_workflow**:
  - Deploy v1 → Use v1 → Deploy v2 → Both coexist
  - Simulates production version management

- **test_multi_platform_deployment**:
  - Same model on 4 platforms (CPU, V100, A100, TPU)
  - Performance characteristics reflected in predictions
  - Speed hierarchy verification

#### Concurrency (1 test)
- **test_concurrent_training_and_prediction**:
  - Train and predict simultaneously
  - No blocking or conflicts

## Test Data Generation

### Realistic Training Data
```python
generate_realistic_training_data(n_samples=30, base_runtime=100, noise_factor=0.1)
```

Features:
- **batch_size**: 8-64 (random)
- **sequence_length**: 64, 128, 256, 512 (choices)
- **hidden_size**: 256, 512, 768, 1024 (choices)
- **runtime_ms**: Correlated with features + noise

Formula:
```
runtime = base_runtime + (batch_size × 0.5 + sequence_length × 0.1 + hidden_size × 0.05) ± noise
```

## Coverage Report

```
Total Coverage: 88%

Component                    Coverage
-----------------------------------------
src/api.py                   73%
src/models.py                94%
src/predictor/base.py        82%
src/predictor/expect_error   97%
src/predictor/mlp.py         96%
src/predictor/quantile.py    95%
src/storage/model_storage    79%
src/utils/experiment.py      98%
```

### Uncovered Areas
Most uncovered lines are:
- Error handling branches for rare conditions
- Debug logging statements
- Startup/shutdown hooks
- Optional validation paths

## Running Tests

### All Tests
```bash
uv run pytest tests/ -v
```

### Specific Test Suites
```bash
# Unit tests only
uv run pytest tests/test_api.py tests/test_experiment.py tests/test_predictors.py -v

# Integration tests only
uv run pytest tests/test_integration.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

### Test Selection
```bash
# Run specific test class
uv run pytest tests/test_integration.py::TestCompleteWorkflow -v

# Run specific test
uv run pytest tests/test_integration.py::TestCompleteWorkflow::test_expect_error_complete_workflow -v

# Run tests matching pattern
uv run pytest tests/ -k "quantile" -v
```

## Test Fixtures

### Global Fixtures
- **setup_and_teardown**: Creates/cleans test storage directory
- **client**: FastAPI TestClient instance

### Data Fixtures
- **trained_model**: Pre-trained expect_error model
- **trained_quantile_model**: Pre-trained quantile model

## Test Isolation

Each test:
- Uses separate storage directory
- Cleans up after completion
- No test interference
- Deterministic results (seeded random)

## CI/CD Integration

Recommended GitHub Actions workflow:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          uv sync --all-extras
      
      - name: Run tests
        run: uv run pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Best Practices

### Writing Integration Tests

1. **Test Real Workflows**: Simulate actual user scenarios
2. **Use Realistic Data**: Generate data with correlations
3. **Verify State Changes**: Check storage, listings, etc.
4. **Test Error Recovery**: Ensure service resilience
5. **Test Concurrency**: Multiple operations simultaneously

### Test Naming Convention
```
test_<scenario>_<expected_outcome>
```

Examples:
- `test_invalid_training_then_valid_training`
- `test_prediction_with_wrong_platform_then_correct`
- `test_experiment_mode_without_trained_model`

## Performance

Test suite execution times:
- Unit tests: ~5-8 seconds
- Integration tests: ~10-12 seconds
- Total: ~13-15 seconds

All tests are fast enough for TDD workflows.
