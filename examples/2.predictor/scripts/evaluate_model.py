#!/usr/bin/env python3
"""
Evaluate trained predictor models using holdout data.

Calculates:
1. Pinball loss for each quantile
2. Prediction accuracy metrics
3. Distribution spread analysis
"""

import json
import requests
import sys
import numpy as np
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


def prepare_features(sample: dict, exclude_fields: set) -> dict:
    """Prepare features by removing excluded fields."""
    return {k: v for k, v in sample.items() if k not in exclude_fields}


def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    """Calculate pinball loss for a single prediction."""
    error = y_true - y_pred
    if error >= 0:
        return quantile * error
    else:
        return (quantile - 1) * error


def evaluate_model(
    predictor_url: str,
    model_id: str,
    platform_info: dict,
    samples: list,
    exclude_fields: set,
    sample_limit: int = None
) -> dict:
    """Evaluate model predictions on all samples."""

    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_id}")
    print(f"Platform: {platform_info}")
    print(f"Total samples: {len(samples)}")
    print(f"{'='*60}")

    if sample_limit:
        samples = samples[:sample_limit]
        print(f"Using {len(samples)} samples for evaluation")

    predictions = []
    actuals = []
    quantiles_dict = {'0.5': [], '0.9': [], '0.95': [], '0.99': []}
    errors = []

    for i, sample in enumerate(samples):
        # Extract features without runtime_ms
        pred_features = prepare_features(sample, exclude_fields | {'runtime_ms'})
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

            if 'result' in result and 'quantiles' in result['result']:
                q = result['result']['quantiles']
                for qname in quantiles_dict.keys():
                    if qname in q:
                        quantiles_dict[qname].append(float(q[qname]))

                actuals.append(actual_runtime)
                predictions.append(float(q.get('0.5', 0)))

        except Exception as e:
            errors.append(str(e))

    if errors:
        print(f"\n  Errors: {len(errors)} requests failed")

    if not actuals:
        print("  No successful predictions")
        return {}

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    # Calculate statistics
    print(f"\n  Prediction Statistics:")
    print(f"    Samples evaluated: {len(actuals)}")

    # Actual runtime distribution
    print(f"\n  Actual Runtime Distribution:")
    print(f"    Mean:   {np.mean(actuals):.1f} ms")
    print(f"    Std:    {np.std(actuals):.1f} ms")
    print(f"    CV:     {np.std(actuals)/np.mean(actuals):.2%}")
    print(f"    Min:    {np.min(actuals):.1f} ms")
    print(f"    Median: {np.median(actuals):.1f} ms")
    print(f"    Max:    {np.max(actuals):.1f} ms")

    # Predicted quantile distribution
    print(f"\n  Predicted Quantile Distribution (averages):")
    for qname, qvals in quantiles_dict.items():
        if qvals:
            print(f"    q{float(qname)*100:.0f}:    {np.mean(qvals):.1f} ms")

    # Quantile spread
    if quantiles_dict['0.5'] and quantiles_dict['0.99']:
        q50_mean = np.mean(quantiles_dict['0.5'])
        q99_mean = np.mean(quantiles_dict['0.99'])
        spread = q99_mean / q50_mean if q50_mean > 0 else 0
        print(f"\n  Quantile Spread (q99/q50): {spread:.2f}x")

    # Pinball loss for each quantile
    print(f"\n  Pinball Loss by Quantile:")
    pinball_results = {}
    for qname, qvals in quantiles_dict.items():
        if qvals:
            q = float(qname)
            losses = [pinball_loss(a, p, q) for a, p in zip(actuals, qvals)]
            avg_loss = np.mean(losses)
            pinball_results[qname] = avg_loss
            print(f"    q{q*100:.0f}: {avg_loss:.2f} ms")

    # Calculate mean pinball loss (normalized by actual mean)
    total_pinball = sum(pinball_results.values()) / len(pinball_results)
    normalized_pinball = total_pinball / np.mean(actuals)
    print(f"\n  Mean Pinball Loss: {total_pinball:.2f} ms ({normalized_pinball:.2%} of mean)")

    # Coverage analysis
    print(f"\n  Coverage Analysis:")
    for qname, qvals in quantiles_dict.items():
        if qvals:
            q = float(qname)
            coverage = np.mean(actuals <= np.array(qvals))
            expected = q
            print(f"    q{q*100:.0f}: {coverage:.2%} covered (expected: {expected:.0%})")

    # Prediction accuracy (using q50 as point estimate)
    if predictions.any():
        mape = np.mean(np.abs(actuals - predictions) / actuals) * 100
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        print(f"\n  Point Prediction Accuracy (q50):")
        print(f"    MAE:  {mae:.1f} ms")
        print(f"    RMSE: {rmse:.1f} ms")
        print(f"    MAPE: {mape:.1f}%")

    return {
        'pinball_loss': pinball_results,
        'mean_pinball': total_pinball,
        'normalized_pinball': normalized_pinball,
        'spread': spread if quantiles_dict['0.5'] and quantiles_dict['0.99'] else None
    }


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

    # Load data
    llm_samples, t2vid_samples, llm_platform, t2vid_platform = load_training_data(config_path)

    # Default platform info if not found
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

    # Evaluate LLM model
    llm_exclude = {'sentence', '_raw_output', '_entry_id'}
    llm_results = evaluate_model(
        predictor_url=predictor_url,
        model_id='llm_task',
        platform_info=llm_platform,
        samples=llm_samples,
        exclude_fields=llm_exclude
    )

    # Evaluate T2VID model
    t2vid_exclude = {'_raw_output', '_entry_id'}
    t2vid_results = evaluate_model(
        predictor_url=predictor_url,
        model_id='t2vid_task',
        platform_info=t2vid_platform,
        samples=t2vid_samples,
        exclude_fields=t2vid_exclude
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nLLM Model:")
    if llm_results:
        print(f"  Mean Pinball Loss: {llm_results.get('mean_pinball', 'N/A'):.2f} ms")
        print(f"  Normalized Pinball: {llm_results.get('normalized_pinball', 0):.2%}")
        print(f"  q99/q50 Spread: {llm_results.get('spread', 'N/A'):.2f}x")

    print("\nT2VID Model:")
    if t2vid_results:
        print(f"  Mean Pinball Loss: {t2vid_results.get('mean_pinball', 'N/A'):.2f} ms")
        print(f"  Normalized Pinball: {t2vid_results.get('normalized_pinball', 0):.2%}")
        print(f"  q99/q50 Spread: {t2vid_results.get('spread', 'N/A'):.2f}x")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
Pinball Loss Guidelines:
  - Lower is better
  - Normalized pinball < 10% is good
  - Normalized pinball < 20% is acceptable

Coverage Analysis:
  - Ideally, q90 should cover ~90% of actuals, q99 should cover ~99%
  - Under-coverage (actual coverage < expected) means predictions are too optimistic
  - Over-coverage means predictions are too conservative

Quantile Spread (q99/q50):
  - Spread > 1.5x indicates meaningful uncertainty estimation
  - Spread close to 1.0x means model is essentially point prediction
  - Real-world runtime distributions typically have spread 1.5x - 3.0x
""")


if __name__ == '__main__':
    main()
