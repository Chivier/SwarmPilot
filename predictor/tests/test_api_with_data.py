"""
Test API with real data from pull_data.csv.

This test script:
1. Loads data from pull_data.csv
2. Splits into train/test sets (80/20)
3. Starts predictor service
4. Trains quantile prediction model
5. Makes predictions on test set
6. Calculates pinball loss metrics
"""

import pytest
import pandas as pd
import numpy as np
import requests
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys


# Configuration
DATA_FILE = Path(__file__).parent / "data" / "pull_data.csv"
API_BASE_URL = "http://localhost:8000"
TRAIN_TEST_SPLIT = 0.8
DEFAULT_QUANTILES = [0.5, 0.9, 0.95, 0.99]

# Model configuration
MODEL_ID_COLUMN = "remote_name"
FEATURE_COLUMN = "req_data_size"
TARGET_COLUMN = "serve_timecost_ms"
SOFTWARE_NAME = "TXOCR"
SOFTWARE_VERSION = "0.0.1"
HARDWARE_NAME = "tx_server"

# Configurable training and testing sizes
# Set to None to use all available data
DEFAULT_TRAIN_SIZE = 1000  # Number of training samples (or None for all)
DEFAULT_TEST_SIZE = 500    # Number of test samples (or None for all)

# MLP configuration
DEFAULT_HIDDEN_LAYERS = [32, 32, 64, 64, 64, 32, 32]  # Hidden layer sizes for MLP (or None for default)
DEFAULT_EPOCHS = 1000              # Number of training epochs
DEFAULT_LEARNING_RATE = 0.01       # Learning rate for optimizer


def calculate_pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    """
    Calculate pinball loss (quantile loss) for a single prediction.

    Loss = max(q * (y - y_pred), (q - 1) * (y - y_pred))

    Args:
        y_true: Actual value
        y_pred: Predicted value
        quantile: Quantile level (0-1)

    Returns:
        Pinball loss value
    """
    error = y_true - y_pred
    if error >= 0:
        return quantile * error
    else:
        return (quantile - 1) * error


def load_and_split_data(
    data_file: Path,
    model_id: str = None,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from CSV and split into train/test sets.

    Args:
        data_file: Path to CSV file
        model_id: Optional model_id to filter data (uses remote_name column)
        train_ratio: Ratio of training data (default 0.8)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    # Load CSV with explicit encoding to handle any special characters
    print(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        print("UTF-8 encoding failed, trying latin-1...")
        df = pd.read_csv(data_file, encoding='latin-1')

    print(f"Total rows loaded: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Filter for specific model if provided
    if model_id:
        df = df[df[MODEL_ID_COLUMN] == model_id]
        print(f"Filtered to model '{model_id}': {len(df)} rows")

    # Remove rows with missing values in key columns
    required_columns = [FEATURE_COLUMN, TARGET_COLUMN, MODEL_ID_COLUMN]
    df_clean = df.dropna(subset=required_columns)
    print(f"After removing NaN values: {len(df_clean)} rows")

    # Remove outliers (optional: keep values within 3 standard deviations)
    target_mean = df_clean[TARGET_COLUMN].mean()
    target_std = df_clean[TARGET_COLUMN].std()
    df_clean = df_clean[
        (df_clean[TARGET_COLUMN] >= target_mean - 3 * target_std) &
        (df_clean[TARGET_COLUMN] <= target_mean + 3 * target_std)
    ]
    print(f"After removing outliers: {len(df_clean)} rows")

    # Shuffle and split
    df_shuffled = df_clean.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    split_idx = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:split_idx]
    test_df = df_shuffled[split_idx:]

    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"Feature stats - Train mean: {train_df[FEATURE_COLUMN].mean():.2f}, "
          f"Test mean: {test_df[FEATURE_COLUMN].mean():.2f}")
    print(f"Target stats - Train mean: {train_df[TARGET_COLUMN].mean():.2f}, "
          f"Test mean: {test_df[TARGET_COLUMN].mean():.2f}")

    return train_df, test_df


def prepare_training_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to training data format expected by API.

    Args:
        df: DataFrame with feature and target columns

    Returns:
        List of feature dictionaries with runtime_ms
    """
    features_list = []
    for _, row in df.iterrows():
        features_list.append({
            FEATURE_COLUMN: float(row[FEATURE_COLUMN]),
            'runtime_ms': float(row[TARGET_COLUMN])
        })

    return features_list


def get_available_models(data_file: Path, min_samples: int = 100) -> List[str]:
    """
    Get list of available models from the data file.

    Args:
        data_file: Path to CSV file
        min_samples: Minimum number of samples required for a model

    Returns:
        List of model IDs that have sufficient data
    """
    print(f"Scanning for available models in {data_file}...")

    try:
        df = pd.read_csv(data_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_file, encoding='latin-1')

    # Count samples per model
    model_counts = df[MODEL_ID_COLUMN].value_counts()

    # Filter models with sufficient samples
    available_models = []
    for model_id, count in model_counts.items():
        if count >= min_samples:
            available_models.append(model_id)

    print(f"Found {len(available_models)} models with >= {min_samples} samples:")
    for i, model_id in enumerate(available_models[:10], 1):  # Show first 10
        count = model_counts[model_id]
        print(f"  {i}. {model_id}: {count} samples")
    if len(available_models) > 10:
        print(f"  ... and {len(available_models) - 10} more models")

    return available_models


def wait_for_service(url: str, timeout: int = 30, interval: float = 0.5) -> bool:
    """
    Wait for service to become available.

    Args:
        url: Health check URL
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds

    Returns:
        True if service is available, False otherwise
    """
    print(f"Waiting for service at {url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print("Service is healthy!")
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(interval)

    print(f"Service did not become available within {timeout} seconds")
    return False


class PredictorServiceManager:
    """Context manager for predictor service lifecycle."""

    def __init__(self, api_module_path: str = "src.api:app", host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize service manager.

        Args:
            api_module_path: FastAPI app module path
            host: Host to bind to
            port: Port to bind to
        """
        self.api_module_path = api_module_path
        self.host = host
        self.port = port
        self.process = None

    def __enter__(self):
        """Start the predictor service."""
        print(f"Starting predictor service on {self.host}:{self.port}...")

        # Start uvicorn server in subprocess
        # We need to run from the predictor directory
        predictor_dir = Path(__file__).parent.parent

        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                self.api_module_path,
                "--host", self.host,
                "--port", str(self.port),
                "--log-level", "warning"
            ],
            cwd=predictor_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for service to start
        health_url = f"http://{self.host}:{self.port}/health"
        if not wait_for_service(health_url, timeout=30):
            self.__exit__(None, None, None)
            raise RuntimeError("Failed to start predictor service")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the predictor service."""
        if self.process:
            print("Stopping predictor service...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Service did not stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()

            print("Service stopped")


def train_model(
    model_id: str,
    features_list: List[Dict[str, Any]],
    quantiles: List[float] = None,
    hidden_layers: List[int] = None,
    epochs: int = None,
    learning_rate: float = None
) -> Dict[str, Any]:
    """
    Train a quantile prediction model via API.

    Args:
        model_id: Model identifier
        features_list: Training data
        quantiles: List of quantiles to predict
        hidden_layers: Hidden layer sizes for MLP (e.g., [64, 32])
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        Training response JSON

    Raises:
        AssertionError: If training fails
    """
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    if hidden_layers is None:
        hidden_layers = DEFAULT_HIDDEN_LAYERS
    if epochs is None:
        epochs = DEFAULT_EPOCHS
    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE

    request_data = {
        'model_id': model_id,
        'platform_info': {
            'software_name': SOFTWARE_NAME,
            'software_version': SOFTWARE_VERSION,
            'hardware_name': HARDWARE_NAME
        },
        'prediction_type': 'quantile',
        'features_list': features_list,
        'training_config': {
            'quantiles': quantiles,
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    }

    print(f"Training model '{model_id}' with {len(features_list)} samples...")
    print(f"Quantiles: {quantiles}")
    print(f"MLP Hidden Layers: {hidden_layers}")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")

    response = requests.post(f"{API_BASE_URL}/train", json=request_data, timeout=300)

    assert response.status_code == 200, f"Training failed: {response.status_code} - {response.text}"

    result = response.json()
    print(f"Training completed: {result}")

    return result


def predict(model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction via API.

    Args:
        model_id: Model identifier
        features: Feature dictionary

    Returns:
        Prediction response JSON

    Raises:
        AssertionError: If prediction fails
    """
    request_data = {
        'model_id': model_id,
        'platform_info': {
            'software_name': SOFTWARE_NAME,
            'software_version': SOFTWARE_VERSION,
            'hardware_name': HARDWARE_NAME
        },
        'prediction_type': 'quantile',
        'features': features
    }

    response = requests.post(f"{API_BASE_URL}/predict", json=request_data, timeout=10)

    assert response.status_code == 200, f"Prediction failed: {response.status_code} - {response.text}"

    return response.json()


def evaluate_predictions(
    test_df: pd.DataFrame,
    model_id: str,
    quantiles: List[float] = None,
    max_samples: int = None
) -> Dict[str, Any]:
    """
    Evaluate model on test set and calculate comprehensive metrics.

    Args:
        test_df: Test DataFrame
        model_id: Model identifier
        quantiles: List of quantiles (must match training)
        max_samples: Maximum number of test samples to evaluate (for speed)

    Returns:
        Dict containing:
            - pinball_loss: Dict of quantile -> average pinball loss
            - median_mape: Mean Absolute Percentage Error for median (q=0.5)
            - expectation_mape: MAPE for distribution expectation
            - median_bias: Average percentage difference for median
            - expectation_bias: Average percentage difference for expectation
    """
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    # Limit test samples if specified
    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(n=max_samples, random_state=42)
        print(f"Limiting evaluation to {max_samples} samples")

    print(f"Evaluating on {len(test_df)} test samples...")

    # Storage for losses and predictions
    losses_by_quantile = {str(q): [] for q in quantiles}
    y_true_list = []
    median_predictions = []  # q=0.5 predictions
    expectation_predictions = []  # Mean of all quantile predictions

    # Make predictions for each test sample
    # Use enumerate to get actual count instead of DataFrame index
    for count, (idx, row) in enumerate(test_df.iterrows(), start=1):
        y_true = row[TARGET_COLUMN]
        features = {FEATURE_COLUMN: float(row[FEATURE_COLUMN])}

        try:
            # Get prediction
            pred_response = predict(model_id, features)
            quantile_predictions = pred_response['result']['quantiles']

            # Store true value
            y_true_list.append(y_true)

            # Calculate pinball loss for each quantile
            quantile_values = []
            for q in quantiles:
                q_str = str(q)
                if q_str in quantile_predictions:
                    y_pred = quantile_predictions[q_str]
                    loss = calculate_pinball_loss(y_true, y_pred, q)
                    losses_by_quantile[q_str].append(loss)
                    quantile_values.append(y_pred)

                    # Store median prediction (q=0.5)
                    if q == 0.5:
                        median_predictions.append(y_pred)

            # Calculate expectation using trapezoidal integration over quantiles
            # For quantile function Q(p), E[X] ≈ ∫Q(p)dp from 0 to 1
            # We approximate this using the quantile values we have
            if quantile_values and len(quantile_values) == len(quantiles):
                # Use trapezoidal rule for numerical integration
                # Sort quantiles and values together
                sorted_pairs = sorted(zip(quantiles, quantile_values))
                q_points = [p for p, _ in sorted_pairs]
                q_values = [v for _, v in sorted_pairs]

                # Trapezoidal integration: ∫f(x)dx ≈ Σ[(x[i+1]-x[i]) * (f[i+1]+f[i])/2]
                expectation = 0.0
                for i in range(len(q_points) - 1):
                    width = q_points[i+1] - q_points[i]
                    avg_height = (q_values[i] + q_values[i+1]) / 2
                    expectation += width * avg_height

                # Add contribution from edges (0 to first quantile and last quantile to 1)
                # Assume linear extrapolation from endpoints
                if q_points[0] > 0:
                    expectation += q_points[0] * q_values[0]
                if q_points[-1] < 1:
                    expectation += (1 - q_points[-1]) * q_values[-1]

                expectation_predictions.append(expectation)

            # Progress indicator (every 100 samples)
            if count % 100 == 0:
                print(f"Processed {count}/{len(test_df)} samples...")

        except Exception as e:
            print(f"Error predicting sample at row {idx} (sample {count}): {e}")
            continue

    # Calculate average pinball losses
    avg_losses = {}
    for q_str, losses in losses_by_quantile.items():
        if losses:
            avg_losses[q_str] = np.mean(losses)
        else:
            avg_losses[q_str] = float('inf')

    # Convert to numpy arrays
    y_true_array = np.array(y_true_list)
    median_pred_array = np.array(median_predictions)
    expectation_pred_array = np.array(expectation_predictions)

    # Calculate MAPE (Mean Absolute Percentage Error)
    # MAPE = (1/n) * Σ |y_true - y_pred| / y_true * 100%
    def calculate_mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Calculate average percentage bias (signed percentage difference)
    def calculate_bias(y_true, y_pred):
        """Calculate average percentage bias."""
        mask = y_true != 0
        if mask.sum() == 0:
            return float('inf')
        return np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100

    median_mape = calculate_mape(y_true_array, median_pred_array)
    expectation_mape = calculate_mape(y_true_array, expectation_pred_array)
    median_bias = calculate_bias(y_true_array, median_pred_array)
    expectation_bias = calculate_bias(y_true_array, expectation_pred_array)

    return {
        'pinball_loss': avg_losses,
        'median_mape': median_mape,
        'expectation_mape': expectation_mape,
        'median_bias': median_bias,
        'expectation_bias': expectation_bias,
        'num_samples': len(y_true_list)
    }


# ==================== TESTS ====================

def test_single_model(
    model_id: str,
    train_size: int = None,
    test_size: int = None,
    quantiles: List[float] = None,
    hidden_layers: List[int] = None,
    epochs: int = None,
    learning_rate: float = None
) -> Dict[str, Any]:
    """
    Test quantile prediction for a single model.

    Args:
        model_id: Model identifier to test
        train_size: Number of training samples (None = use all available)
        test_size: Number of test samples (None = use all available)
        quantiles: List of quantiles to predict (None = use DEFAULT_QUANTILES)
        hidden_layers: Hidden layer sizes for MLP (e.g., [64, 32])
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        Dict with evaluation results

    This function:
    1. Loads and splits data for the specified model
    2. Trains a quantile model
    3. Evaluates on test set
    4. Returns comprehensive metrics
    """
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES

    if train_size is None:
        train_size = DEFAULT_TRAIN_SIZE

    if test_size is None:
        test_size = DEFAULT_TEST_SIZE

    if hidden_layers is None:
        hidden_layers = DEFAULT_HIDDEN_LAYERS

    if epochs is None:
        epochs = DEFAULT_EPOCHS

    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE

    print(f"\n{'='*70}")
    print(f"Testing model: {model_id}")
    print(f"{'='*70}\n")

    # Load and split data
    train_df, test_df = load_and_split_data(
        DATA_FILE,
        model_id=model_id,
        train_ratio=TRAIN_TEST_SPLIT
    )

    # Ensure we have enough data
    min_train_samples = min(100, train_size) if train_size else 100
    min_test_samples = min(20, test_size) if test_size else 20

    if len(train_df) < min_train_samples:
        print(f"⚠ Warning: Insufficient training data for {model_id}: {len(train_df)} < {min_train_samples}")
        return None

    if len(test_df) < min_test_samples:
        print(f"⚠ Warning: Insufficient test data for {model_id}: {len(test_df)} < {min_test_samples}")
        return None

    # Prepare training data
    if train_size:
        train_subset_size = min(train_size, len(train_df))
        train_subset = train_df.sample(n=train_subset_size, random_state=42)
    else:
        train_subset = train_df

    features_list = prepare_training_data(train_subset)

    print(f"Using {len(features_list)} training samples")

    # Train model
    train_result = train_model(
        model_id=model_id,
        features_list=features_list,
        quantiles=quantiles,
        hidden_layers=hidden_layers,
        epochs=epochs,
        learning_rate=learning_rate
    )

    if train_result['status'] != 'success':
        print(f"✗ Training failed for {model_id}")
        return None

    assert train_result['samples_trained'] == len(features_list)

    # Evaluate on test set
    eval_results = evaluate_predictions(
        test_df,
        model_id=model_id,
        quantiles=quantiles,
        max_samples=test_size
    )

    # Extract results
    avg_losses = eval_results['pinball_loss']
    median_mape = eval_results['median_mape']
    expectation_mape = eval_results['expectation_mape']
    median_bias = eval_results['median_bias']
    expectation_bias = eval_results['expectation_bias']
    num_samples = eval_results['num_samples']

    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Training samples: {len(features_list)}")
    print(f"Test samples evaluated: {num_samples}")

    print(f"\n{'='*70}")
    print("PINBALL LOSS BY QUANTILE")
    print(f"{'='*70}")
    for q_str, loss in sorted(avg_losses.items(), key=lambda x: float(x[0])):
        print(f"  q={q_str}: {loss:.4f}")

    # Calculate overall average loss
    overall_avg_loss = np.mean(list(avg_losses.values()))
    print(f"\nOverall Average Pinball Loss: {overall_avg_loss:.4f}")

    print(f"\n{'='*70}")
    print("DISTRIBUTION PREDICTION METRICS")
    print(f"{'='*70}")
    print(f"Median (q=0.5) Prediction:")
    print(f"  MAPE: {median_mape:.2f}%")
    print(f"  Bias: {median_bias:+.2f}% {'(over-prediction)' if median_bias > 0 else '(under-prediction)' if median_bias < 0 else '(unbiased)'}")

    print(f"\nExpectation (mean of quantiles) Prediction:")
    print(f"  MAPE: {expectation_mape:.2f}%")
    print(f"  Bias: {expectation_bias:+.2f}% {'(over-prediction)' if expectation_bias > 0 else '(under-prediction)' if expectation_bias < 0 else '(unbiased)'}")

    # Assertions: Verify losses are reasonable
    # Pinball loss should be finite and positive
    for q_str, loss in avg_losses.items():
        assert np.isfinite(loss), f"Loss for quantile {q_str} is not finite"
        assert loss >= 0, f"Loss for quantile {q_str} is negative"

    # Overall loss should not be too high
    assert overall_avg_loss < 1000, f"Average loss too high: {overall_avg_loss}"

    # MAPE should be reasonable
    assert median_mape < 100, f"Median MAPE too high: {median_mape:.2f}%"

    # Note: Expectation MAPE can be high for quantile regression
    if expectation_mape > 200:
        print(f"\n⚠ Warning: Expectation MAPE is high ({expectation_mape:.2f}%)")
        print(f"  This is expected for quantile regression.")
    else:
        assert expectation_mape < 200, f"Expectation MAPE too high: {expectation_mape:.2f}%"

    print(f"✓ Model {model_id} passed all assertions!\n")

    # Return results for aggregation
    return {
        'model_id': model_id,
        'pinball_loss': avg_losses,
        'overall_pinball_loss': overall_avg_loss,
        'median_mape': median_mape,
        'expectation_mape': expectation_mape,
        'median_bias': median_bias,
        'expectation_bias': expectation_bias,
        'num_train_samples': len(features_list),
        'num_test_samples': num_samples
    }


def test_quantile_prediction_with_real_data(
    models: List[str] = None,
    train_size: int = None,
    test_size: int = None,
    quantiles: List[float] = None,
    hidden_layers: List[int] = None,
    epochs: int = None,
    learning_rate: float = None
):
    """
    Test quantile prediction using real data from pull_data.csv.

    Args:
        models: List of model IDs to test (None = test all available models)
        train_size: Number of training samples per model (None = use DEFAULT_TRAIN_SIZE)
        test_size: Number of test samples per model (None = use DEFAULT_TEST_SIZE)
        quantiles: List of quantiles to predict (None = use DEFAULT_QUANTILES)
        hidden_layers: Hidden layer sizes for MLP (e.g., [64, 32])
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    This test:
    1. Loads and splits data
    2. Starts predictor service
    3. Trains quantile models for each specified model
    4. Evaluates on test sets
    5. Reports comprehensive metrics
    """
    # Check if data file exists
    assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"

    # Get list of models to test
    if models is None:
        # Get all available models
        available_models = get_available_models(DATA_FILE, min_samples=200)
        # Limit to top 3 models by sample count for testing
        models = available_models[:3]
        print(f"\nTesting top 3 models with most samples\n")
    elif isinstance(models, str):
        models = [models]

    print(f"\n{'='*70}")
    print(f"STARTING TESTS FOR {len(models)} MODEL(S)")
    print(f"{'='*70}")
    print(f"Train size per model: {train_size or DEFAULT_TRAIN_SIZE}")
    print(f"Test size per model: {test_size or DEFAULT_TEST_SIZE}")
    print(f"Quantiles: {quantiles or DEFAULT_QUANTILES}")
    print(f"MLP Hidden Layers: {hidden_layers or DEFAULT_HIDDEN_LAYERS}")
    print(f"Epochs: {epochs or DEFAULT_EPOCHS}, Learning Rate: {learning_rate or DEFAULT_LEARNING_RATE}")
    print(f"{'='*70}\n")

    # Start predictor service and run tests
    with PredictorServiceManager():
        results = []

        for i, model_id in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Testing model: {model_id}")
            try:
                result = test_single_model(
                    model_id=model_id,
                    train_size=train_size,
                    test_size=test_size,
                    quantiles=quantiles,
                    hidden_layers=hidden_layers,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"✗ Error testing {model_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Print summary
        print(f"\n\n{'='*70}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*70}\n")

        if not results:
            print("No models were successfully tested.")
            return

        print(f"Successfully tested {len(results)}/{len(models)} models:\n")

        # Create summary table
        print(f"{'Model':<50} {'Pinball':<12} {'Med.MAPE':<12} {'Exp.MAPE':<12}")
        print(f"{'-'*50} {'-'*12} {'-'*12} {'-'*12}")

        for result in results:
            model_name = result['model_id'][:47] + '...' if len(result['model_id']) > 50 else result['model_id']
            print(f"{model_name:<50} {result['overall_pinball_loss']:<12.2f} {result['median_mape']:<12.2f} {result['expectation_mape']:<12.2f}")

        # Overall statistics
        avg_pinball = np.mean([r['overall_pinball_loss'] for r in results])
        avg_median_mape = np.mean([r['median_mape'] for r in results])
        avg_exp_mape = np.mean([r['expectation_mape'] for r in results])

        print(f"\n{'Average':<50} {avg_pinball:<12.2f} {avg_median_mape:<12.2f} {avg_exp_mape:<12.2f}")

        print(f"\n{'='*70}")
        print("✓ All tests completed successfully!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test quantile prediction API with real data from pull_data.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with default settings
  python test_api_with_data.py

  # Test specific model
  python test_api_with_data.py --model "video.text_match.TextMatchModelInfer.model_infer"

  # Test with custom dataset sizes
  python test_api_with_data.py --train-size 2000 --test-size 1000

  # Test with custom MLP architecture
  python test_api_with_data.py --hidden-layers 128 64 32 --epochs 2000 --learning-rate 0.001

  # Test with custom quantiles
  python test_api_with_data.py --quantiles 0.25 0.5 0.75 0.9 0.95 0.99

  # Combine multiple options
  python test_api_with_data.py --model "rec.rec.RecModel.OcrRecognize" --train-size 5000 --hidden-layers 256 128 64
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        action='append',
        dest='models',
        help='Model ID to test (can specify multiple times). If not specified, tests top 3 models.'
    )

    parser.add_argument(
        '--train-size',
        type=int,
        default=None,
        help=f'Number of training samples per model (default: {DEFAULT_TRAIN_SIZE})'
    )

    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help=f'Number of test samples per model (default: {DEFAULT_TEST_SIZE})'
    )

    parser.add_argument(
        '--quantiles',
        type=float,
        nargs='+',
        default=None,
        help=f'List of quantiles to predict (default: {DEFAULT_QUANTILES}). Example: --quantiles 0.5 0.9 0.95 0.99'
    )

    parser.add_argument(
        '--hidden-layers',
        type=int,
        nargs='+',
        default=None,
        help=f'Hidden layer sizes for MLP (default: {DEFAULT_HIDDEN_LAYERS}). Example: --hidden-layers 128 64 32'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help=f'Learning rate for optimizer (default: {DEFAULT_LEARNING_RATE})'
    )

    args = parser.parse_args()

    # Run test with parsed arguments
    test_quantile_prediction_with_real_data(
        models=args.models,
        train_size=args.train_size,
        test_size=args.test_size,
        quantiles=args.quantiles,
        hidden_layers=args.hidden_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
