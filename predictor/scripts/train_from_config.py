#!/usr/bin/env python3
"""
Train predictor models using training_config.json data.

This script:
1. Loads llm_samples and t2vid_samples from training_config.json
2. Prepares features (removes non-predictive fields)
3. Sends training requests to the predictor service
4. Reports pinball loss and other metrics
"""

import json
import requests
import sys
from pathlib import Path


def load_training_data(config_path: str) -> tuple:
    """Load training samples from config file."""
    with open(config_path, 'r') as f:
        data = json.load(f)

    llm_samples = data.get('llm_samples', [])
    t2vid_samples = data.get('t2vid_samples', [])

    # Get platform info from config
    llm_platform = None
    t2vid_platform = None

    if 'config' in data:
        if 'llm_service' in data['config'] and data['config']['llm_service'].get('instances'):
            inst = data['config']['llm_service']['instances'][0]
            llm_platform = {
                'hardware_name': inst.get('hardware_name', 'unknown'),
                'software_name': inst.get('software_name', 'unknown'),
                'software_version': inst.get('software_version', 'unknown')
            }

        if 't2vid_service' in data['config'] and data['config']['t2vid_service'].get('instances'):
            inst = data['config']['t2vid_service']['instances'][0]
            t2vid_platform = {
                'hardware_name': inst.get('hardware_name', 'unknown'),
                'software_name': inst.get('software_name', 'unknown'),
                'software_version': inst.get('software_version', 'unknown')
            }

    return llm_samples, t2vid_samples, llm_platform, t2vid_platform


def prepare_llm_features(samples: list) -> list:
    """Prepare LLM training samples by extracting relevant features."""
    prepared = []

    # Fields to exclude (non-predictive or metadata)
    exclude_fields = {'sentence', '_raw_output', '_entry_id'}

    for sample in samples:
        features = {}
        for key, value in sample.items():
            if key not in exclude_fields:
                features[key] = value
        prepared.append(features)

    return prepared


def prepare_t2vid_features(samples: list) -> list:
    """Prepare T2VID training samples by extracting relevant features."""
    prepared = []

    # Fields to exclude (non-predictive or metadata)
    exclude_fields = {'_raw_output', '_entry_id'}

    for sample in samples:
        features = {}
        for key, value in sample.items():
            if key not in exclude_fields:
                features[key] = value
        prepared.append(features)

    return prepared


def train_model(
    predictor_url: str,
    model_id: str,
    platform_info: dict,
    features: list,
    config: dict = None,
    prediction_type: str = 'quantile'
) -> dict:
    """Send training request to predictor service."""

    payload = {
        'model_id': model_id,
        'platform_info': platform_info,
        'prediction_type': prediction_type,
        'features_list': features
    }

    if config:
        payload['training_config'] = config

    print(f"\n{'='*60}")
    print(f"Training model: {model_id}")
    print(f"Platform: {platform_info}")
    print(f"Samples: {len(features)}")
    print(f"{'='*60}")

    # Print sample feature keys
    if features:
        print(f"Feature keys: {list(features[0].keys())}")

    try:
        response = requests.post(
            f"{predictor_url}/train",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"\nTraining Result:")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Message: {result.get('message', 'N/A')}")

        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n  Training Metrics:")
            print(f"    Final Loss: {metrics.get('final_loss', 'N/A'):.6f}")
            print(f"    Epochs: {metrics.get('epochs_trained', 'N/A')}")
            if 'loss_history' in metrics:
                history = metrics['loss_history']
                print(f"    Loss History: start={history[0]:.4f}, end={history[-1]:.4f}")

        if 'pinball_loss' in result:
            pinball = result['pinball_loss']
            print(f"\n  Pinball Loss by Quantile:")
            for q, loss in pinball.items():
                print(f"    q{float(q)*100:.0f}: {loss:.4f}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error training model: {e}")
        return {'status': 'error', 'error': str(e)}


def test_predictions(
    predictor_url: str,
    model_id: str,
    platform_info: dict,
    test_features: list,
    num_tests: int = 5
) -> None:
    """Test predictions from trained model."""

    print(f"\n{'='*60}")
    print(f"Testing predictions for: {model_id}")
    print(f"{'='*60}")

    for i, sample in enumerate(test_features[:num_tests]):
        # Extract features without runtime_ms for prediction
        pred_features = {k: v for k, v in sample.items() if k != 'runtime_ms'}
        actual_runtime = sample.get('runtime_ms', 0)

        payload = {
            'model_id': model_id,
            'platform_info': platform_info,
            'features': pred_features,
            'prediction_type': 'quantile'
        }

        try:
            response = requests.post(
                f"{predictor_url}/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if 'quantiles' in result:
                q = result['quantiles']
                q50 = float(q.get('0.5', 0))
                q90 = float(q.get('0.9', 0))
                q99 = float(q.get('0.99', 0))

                print(f"\n  Sample {i+1}:")
                print(f"    Actual:  {actual_runtime:.1f} ms")
                print(f"    q50:     {q50:.1f} ms")
                print(f"    q90:     {q90:.1f} ms")
                print(f"    q99:     {q99:.1f} ms")
                print(f"    q99/q50: {q99/q50:.2f}x" if q50 > 0 else "")

                # Check if actual falls within reasonable range
                if actual_runtime < q50:
                    print(f"    Status:  Actual < q50 (fast execution)")
                elif actual_runtime < q90:
                    print(f"    Status:  q50 < Actual < q90 (normal)")
                elif actual_runtime < q99:
                    print(f"    Status:  q90 < Actual < q99 (slow)")
                else:
                    print(f"    Status:  Actual > q99 (very slow)")

        except requests.exceptions.RequestException as e:
            print(f"  Sample {i+1}: Error - {e}")


def main():
    # Configuration
    predictor_url = "http://localhost:8101"
    config_path = Path(__file__).parent.parent.parent / "experiments/13.workflow_benchmark/type1_text2video/data/training_config.json"

    if len(sys.argv) > 1:
        predictor_url = sys.argv[1]
    if len(sys.argv) > 2:
        config_path = sys.argv[2]

    print(f"Predictor URL: {predictor_url}")
    print(f"Config Path: {config_path}")

    # Check predictor health
    try:
        health = requests.get(f"{predictor_url}/health", timeout=5)
        health.raise_for_status()
        print(f"Predictor Status: {health.json()}")
    except Exception as e:
        print(f"Error connecting to predictor: {e}")
        print("Make sure the predictor service is running at the specified URL.")
        sys.exit(1)

    # Load training data
    llm_samples, t2vid_samples, llm_platform, t2vid_platform = load_training_data(config_path)

    print(f"\nLoaded {len(llm_samples)} LLM samples")
    print(f"Loaded {len(t2vid_samples)} T2VID samples")

    # Default platform info if not found in config
    if not llm_platform:
        llm_platform = {
            'hardware_name': 'NVIDIA H20',
            'software_name': 'sglang',
            'software_version': '0.5.5.post2'
        }

    if not t2vid_platform:
        t2vid_platform = {
            'hardware_name': 'NVIDIA H20',
            'software_name': 'diffuser',
            'software_version': 'diffuser'
        }

    # Prepare features
    llm_features = prepare_llm_features(llm_samples)
    t2vid_features = prepare_t2vid_features(t2vid_samples)

    # Training configuration with data augmentation and residual calibration
    training_config = {
        'data_augmentation': {
            'enabled': True,
            'cv': 0.4,              # 40% CV for augmentation
            'samples_per_point': 5,  # Generate 5 synthetic samples per real sample
            'distribution': 'lognormal'
        },
        'residual_calibration': {
            'enabled': True,
            'min_sigma': 0.1
        },
        'epochs': 200,
        'learning_rate': 0.001
    }

    # Train LLM model
    if llm_features:
        llm_result = train_model(
            predictor_url=predictor_url,
            model_id='llm_task',
            platform_info=llm_platform,
            features=llm_features,
            config=training_config
        )

        # Test LLM predictions
        if llm_result.get('status') == 'success':
            test_predictions(
                predictor_url=predictor_url,
                model_id='llm_task',
                platform_info=llm_platform,
                test_features=llm_features
            )

    # Train T2VID model
    if t2vid_features:
        t2vid_result = train_model(
            predictor_url=predictor_url,
            model_id='t2vid_task',
            platform_info=t2vid_platform,
            features=t2vid_features,
            config=training_config
        )

        # Test T2VID predictions
        if t2vid_result.get('status') == 'success':
            test_predictions(
                predictor_url=predictor_url,
                model_id='t2vid_task',
                platform_info=t2vid_platform,
                test_features=t2vid_features
            )

    # List trained models
    print(f"\n{'='*60}")
    print("Listing all trained models:")
    print(f"{'='*60}")
    try:
        response = requests.get(f"{predictor_url}/list", timeout=10)
        response.raise_for_status()
        models = response.json()
        for model in models.get('models', []):
            print(f"  - {model}")
    except Exception as e:
        print(f"Error listing models: {e}")


if __name__ == '__main__':
    main()
