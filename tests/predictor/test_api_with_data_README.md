# Real Data API Test - README

## Overview

`test_api_with_data.py` is a comprehensive integration test that validates the Predictor API using real production data from `pull_data.csv`.

## What This Test Does

1. **Data Loading**: Loads real OCR inference data from CSV file
2. **Data Splitting**: Splits data into 80% training and 20% test sets
3. **Service Management**: Automatically starts/stops the Predictor API service
4. **Model Training**: Trains a quantile prediction model via `/train` endpoint
5. **Prediction**: Makes predictions on test set via `/predict` endpoint
6. **Evaluation**: Calculates pinball loss metrics for each quantile

## Data Configuration

The test uses the following data mapping:

- **Feature**: `req_data_size` (request data size in bytes)
- **Target**: `serve_timecost_ms` (serving time in milliseconds)
- **Model ID**: `remote_name` (e.g., "video.text_match.TextMatchModelInfer.model_infer")
- **Platform**:
  - software_name: "TXOCR"
  - software_version: "0.0.1"
  - hardware_name: "tx_server"
- **Prediction Type**: quantile
- **Quantiles**: [0.5, 0.9, 0.95, 0.99]

## Running the Test

### Using pytest

```bash
# From predictor directory
cd /chivier-disk/yanweiye/Projects/swarmpilot-refresh/predictor

# Run with uv
uv run pytest tests/test_api_with_data.py -v

# Run with pytest directly
pytest tests/test_api_with_data.py -v

# Run with detailed output
uv run pytest tests/test_api_with_data.py -v -s
```

### Running Standalone

The test can be run as a standalone script with various options using argparse:

#### Get Help
```bash
cd /chivier-disk/yanweiye/Projects/swarmpilot-refresh/predictor

# Display all available options
uv run python tests/test_api_with_data.py --help
```

#### Test All Models (Default)
```bash
# Test top 3 models with default settings
# (1000 training samples, 500 test samples per model)
uv run python tests/test_api_with_data.py
```

#### Test Specific Model
```bash
# Test a specific model
uv run python tests/test_api_with_data.py --model "video.text_match.TextMatchModelInfer.model_infer"

# Or use short option
uv run python tests/test_api_with_data.py -m "video.text_match.TextMatchModelInfer.model_infer"

# Test multiple specific models
uv run python tests/test_api_with_data.py -m "video.text_match.TextMatchModelInfer.model_infer" -m "rec.rec.RecModel.OcrRecognize"
```

#### Configure Dataset Sizes
```bash
# Custom training and test sizes
uv run python tests/test_api_with_data.py --train-size 2000 --test-size 1000

# Train on all available data, test on 1000 samples
uv run python tests/test_api_with_data.py --train-size 0 --test-size 1000
```

#### Configure MLP Architecture
```bash
# Change hidden layer sizes (default: [64, 32])
uv run python tests/test_api_with_data.py --hidden-layers 128 64 32

# Use a deeper network
uv run python tests/test_api_with_data.py --hidden-layers 256 128 64 32

# Use a single hidden layer
uv run python tests/test_api_with_data.py --hidden-layers 128

# Configure training epochs (default: 1000)
uv run python tests/test_api_with_data.py --epochs 2000

# Configure learning rate (default: 0.01)
uv run python tests/test_api_with_data.py --learning-rate 0.001
```

#### Configure Quantiles
```bash
# Test different quantiles (default: [0.5, 0.9, 0.95, 0.99])
uv run python tests/test_api_with_data.py --quantiles 0.25 0.5 0.75 0.9 0.95 0.99

# Test only median
uv run python tests/test_api_with_data.py --quantiles 0.5
```

#### Combine Multiple Options
```bash
# Complete custom configuration
uv run python tests/test_api_with_data.py \
  --model "rec.rec.RecModel.OcrRecognize" \
  --train-size 5000 \
  --test-size 2000 \
  --hidden-layers 256 128 64 \
  --epochs 2000 \
  --learning-rate 0.001 \
  --quantiles 0.5 0.9 0.95 0.99
```

#### Available Models

The test automatically detects available models from the CSV data. The three models found in the data are:

1. **video.text_match.TextMatchModelInfer.model_infer** - Text matching model (~472K samples)
2. **det.det.DetModel.OcrDetectFullImageWithLayout** - OCR detection model (~277K samples)
3. **rec.rec.RecModel.OcrRecognize** - OCR recognition model (~170K samples)

## Configuration Reference

### Default Configuration

You can modify defaults by editing the configuration constants at the top of the test file:

```python
# In test_api_with_data.py
DEFAULT_TRAIN_SIZE = 1000         # Training samples per model
DEFAULT_TEST_SIZE = 500           # Test samples per model
DEFAULT_QUANTILES = [0.5, 0.9, 0.95, 0.99]  # Quantiles to predict
DEFAULT_HIDDEN_LAYERS = [64, 32]  # MLP architecture
DEFAULT_EPOCHS = 1000             # Training epochs
DEFAULT_LEARNING_RATE = 0.01      # Optimizer learning rate
```

### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--model` | `-m` | str | Top 3 models | Model ID to test (repeatable) |
| `--train-size` | | int | 1000 | Training samples per model |
| `--test-size` | | int | 500 | Test samples per model |
| `--quantiles` | | float[] | [0.5, 0.9, 0.95, 0.99] | Quantiles to predict |
| `--hidden-layers` | | int[] | [64, 32] | MLP hidden layer sizes |
| `--epochs` | | int | 1000 | Training epochs |
| `--learning-rate` | | float | 0.01 | Optimizer learning rate |

### MLP Architecture Tuning Guide

**Choosing Hidden Layer Sizes:**
- **Small datasets (< 1000 samples)**: `[32, 16]` or `[64, 32]`
- **Medium datasets (1000-10000 samples)**: `[64, 32]` or `[128, 64]` (default-like)
- **Large datasets (> 10000 samples)**: `[128, 64, 32]` or `[256, 128, 64]`

**Choosing Epochs:**
- Start with 500-1000 epochs
- Increase to 2000-5000 for better convergence on complex patterns
- Monitor for overfitting with very high epoch counts

**Choosing Learning Rate:**
- Default `0.01` works well for most cases
- Use `0.001` for more stable but slower training
- Use `0.1` for faster training (may be unstable)

**Example Configurations:**
```bash
# Fast training for quick tests
uv run python tests/test_api_with_data.py --hidden-layers 32 --epochs 500

# Balanced configuration (default)
uv run python tests/test_api_with_data.py --hidden-layers 64 32 --epochs 1000

# High-quality training for best results
uv run python tests/test_api_with_data.py --hidden-layers 256 128 64 --epochs 3000 --learning-rate 0.005
```

## Expected Output

### Multi-Model Test Output

When testing multiple models, you'll see output for each model followed by a summary:

```
======================================================================
STARTING TESTS FOR 3 MODEL(S)
======================================================================
Train size per model: 1000
Test size per model: 500
Quantiles: [0.5, 0.9, 0.95, 0.99]
======================================================================

[1/3] Testing model: video.text_match.TextMatchModelInfer.model_infer
...
[2/3] Testing model: det.det.DetModel.OcrDetectFullImageWithLayout
...
[3/3] Testing model: rec.rec.RecModel.OcrRecognize
...

======================================================================
SUMMARY OF ALL TESTS
======================================================================

Successfully tested 3/3 models:

Model                                              Pinball      Med.MAPE     Exp.MAPE
-------------------------------------------------- ------------ ------------ ------------
video.text_match.TextMatchModelInfer.model_infer   45.17        75.09        287.09
det.det.DetModel.OcrDetectFullImageWithLayout      64.95        40.13        65.99
rec.rec.RecModel.OcrRecognize                      167.42       55.08        234.55

Average                                            92.51        56.77        195.88

======================================================================
✓ All tests completed successfully!
======================================================================
```

### Single Model Output

The test will output detailed results for each model tested:

1. **Data Loading Info**:
   - Total rows loaded
   - Rows after filtering
   - Train/test split sizes
   - Feature statistics

2. **Training Info**:
   - Model ID being trained
   - Number of training samples
   - Quantiles being predicted
   - Training completion status

3. **Evaluation Results**:
   ```
   ============================================================
   EVALUATION RESULTS
   ============================================================
   Model: video.text_match.TextMatchModelInfer.model_infer
   Training samples: 1000
   Test samples evaluated: 500

   ============================================================
   PINBALL LOSS BY QUANTILE
   ============================================================
     q=0.5: 60.0000
     q=0.9: 72.0000
     q=0.95: 39.5900
     q=0.99: 9.1170

   Overall Average Pinball Loss: 45.1760

   ============================================================
   DISTRIBUTION PREDICTION METRICS
   ============================================================
   Median (q=0.5) Prediction:
     MAPE: 74.76%
     Bias: +14.01% (over-prediction)

   Expectation (mean of quantiles) Prediction:
     MAPE: 286.65%
     Bias: +264.44% (over-prediction)

   ============================================================
   INTERPRETATION
   ============================================================
   • MAPE (Mean Absolute Percentage Error):
     - Measures average magnitude of prediction errors
     - Lower is better (0% = perfect prediction)
     - Typical threshold: <20% is good, <10% is excellent

   • Bias (Average Percentage Difference):
     - Positive: model tends to over-predict
     - Negative: model tends to under-predict
     - Close to 0%: unbiased predictions
   ============================================================
   ```

## Evaluation Metrics

The test calculates several metrics to comprehensively evaluate prediction quality:

### 1. Pinball Loss (Quantile Loss)

The **pinball loss** evaluates how well each quantile is predicted:

```
Loss = max(q * (y - ŷ), (q - 1) * (y - ŷ))
```

Where:
- `y` = actual runtime
- `ŷ` = predicted runtime for quantile q
- `q` = quantile level (e.g., 0.5, 0.9, 0.95, 0.99)

**Properties**:
- Lower loss = better prediction
- Asymmetric loss that penalizes over/under-prediction differently based on quantile
- For q=0.5 (median), equivalent to Mean Absolute Error
- Optimized during model training

### 2. MAPE (Mean Absolute Percentage Error)

**MAPE** measures the average magnitude of prediction errors as a percentage:

```
MAPE = (1/n) * Σ |y_true - y_pred| / y_true * 100%
```

**Calculated for**:
- **Median Prediction** (q=0.5): The 50th percentile prediction
- **Expectation**: The expected value of the predicted distribution (computed via trapezoidal integration over quantiles)

**Interpretation**:
- 0% = perfect prediction
- < 10% = excellent
- < 20% = good
- < 30% = acceptable
- > 50% = poor

**Note**: For runtime prediction with high variance, MAPE values of 50-100% can still be acceptable.

### 3. Bias (Average Percentage Difference)

**Bias** measures the systematic tendency to over-predict or under-predict:

```
Bias = (1/n) * Σ (y_pred - y_true) / y_true * 100%
```

**Interpretation**:
- **Positive bias**: Model tends to over-predict (predictions > actual)
- **Negative bias**: Model tends to under-predict (predictions < actual)
- **~0% bias**: Unbiased predictions (balanced over/under-predictions)

**Calculated for**:
- **Median Prediction** (q=0.5)
- **Expectation** (mean of quantile predictions)

### 4. Expectation Calculation

The **expectation** (expected value) of the predicted distribution is computed using **trapezoidal integration**:

For a quantile function Q(p), the expectation is:
```
E[X] = ∫₀¹ Q(p) dp
```

We approximate this integral using the quantile predictions:
- Sort quantiles and their predicted values
- Use trapezoidal rule for numerical integration between quantile points
- Extrapolate linearly at the edges (0 to q_min and q_max to 1)

**Why this matters**:
- Quantile regression models are optimized for individual quantiles, not the mean
- The expectation MAPE can be higher than median MAPE
- It provides insight into whether the predicted distribution's center of mass aligns with actual runtimes

## Test Configuration

You can modify test behavior by editing constants at the top of the test file:

```python
# Data configuration
DATA_FILE = Path(__file__).parent / "data" / "pull_data.csv"
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
DEFAULT_QUANTILES = [0.5, 0.9, 0.95, 0.99]

# Model configuration
MODEL_ID_COLUMN = "remote_name"
FEATURE_COLUMN = "req_data_size"
TARGET_COLUMN = "serve_timecost_ms"
```

## Test Parameters

The test automatically:
- Selects the most common `remote_name` from the data
- Removes outliers (values > 3 standard deviations from mean)
- Limits training to 1000 samples (for speed)
- Limits evaluation to 500 test samples (for speed)

To change these, modify the test function directly.

## Troubleshooting

### Data File Not Found
Ensure `predictor/tests/data/pull_data.csv` exists and is readable.

### Service Startup Timeout
If the service takes too long to start:
- Check if port 8000 is already in use
- Increase timeout in `wait_for_service()` function

### High Pinball Loss
If the loss is very high (> 1000):
- Check if you need more training data
- Adjust training epochs in `train_model()` function
- Verify feature scaling is appropriate for your data range

### Memory Issues
If loading the full CSV causes memory issues:
- Reduce the number of rows loaded
- Use `pd.read_csv(DATA_FILE, nrows=100000)` to limit rows

## Implementation Details

### Key Functions

1. **`calculate_pinball_loss()`**: Computes pinball loss for a single prediction
2. **`load_and_split_data()`**: Loads CSV and creates train/test split
3. **`prepare_training_data()`**: Formats data for API training endpoint
4. **`PredictorServiceManager`**: Context manager for service lifecycle
5. **`train_model()`**: Trains model via `/train` endpoint
6. **`predict()`**: Makes predictions via `/predict` endpoint
7. **`evaluate_predictions()`**: Evaluates model on test set

### Service Management

The test uses a context manager to automatically:
- Start uvicorn server in subprocess
- Wait for health check to pass
- Run tests
- Gracefully shutdown service on completion or error

### Data Preprocessing

The test performs:
- Missing value removal
- Outlier detection (3σ rule)
- Random shuffling with seed for reproducibility
- Train/test stratification

## Performance Considerations

**Default settings** are optimized for quick testing:
- 1000 training samples
- 500 test samples
- 1000 training epochs

**For production evaluation**, increase:
- Training samples to 10,000+
- Test samples to 1,000+
- Training epochs to 2,000+

Edit the test function to adjust these values.

## Assertions

The test verifies:
1. ✓ Data file exists
2. ✓ Sufficient training data (≥100 samples)
3. ✓ Sufficient test data (≥20 samples)
4. ✓ Training succeeds
5. ✓ All pinball losses are finite and non-negative
6. ✓ Overall average loss is reasonable (< 1000)

## License

Part of the swarmpilot-refresh project.
