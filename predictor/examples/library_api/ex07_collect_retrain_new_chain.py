"""Example 7: Retrain with different chain - triggers full retrain.

Demonstrates chain change behavior:
1. Train initial model with chain A
2. Collect new samples
3. Retrain with chain B - old model is discarded, full retrain happens

When the chain changes, the system cannot incrementally update
because the feature space is different. A full retrain is required.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.library_api.utils import cleanup_storage
from examples.library_api.utils import create_core_with_storage
from examples.library_api.utils import create_low_level_with_storage
from examples.library_api.utils import generate_training_data
from examples.library_api.utils import get_platform_info
from examples.library_api.utils import print_chain_info
from examples.library_api.utils import print_result
from examples.library_api.utils import print_section
from examples.library_api.utils import print_subsection
from examples.library_api.utils import setup_storage
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_v2 import MultiplyPreprocessor
from swarmpilot.predictor.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor


def create_chain_a() -> PreprocessorChainV2:
    """Create chain A: compute_cost from batch_size * sequence_length."""
    return (
        PreprocessorChainV2(name="chain_a")
        .add(
            MultiplyPreprocessor(
                "batch_size", "sequence_length", "compute_cost"
            )
        )
        .add(RemoveFeaturePreprocessor(["batch_size", "sequence_length"]))
    )


def create_chain_b() -> PreprocessorChainV2:
    """Create chain B: different computation using all features."""
    return (
        PreprocessorChainV2(name="chain_b")
        .add(MultiplyPreprocessor("batch_size", "hidden_size", "batch_hidden"))
        .add(RemoveFeaturePreprocessor(["batch_size", "hidden_size"]))
    )


def main() -> None:
    """Run Example 7: Collect and retrain with new chain."""
    print_section("Example 7: Collect & Retrain (New Chain)")

    # Setup isolated storage
    storage = setup_storage("ex07_collect_new_chain")
    platform = get_platform_info()

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        model_id = "evolving-model"

        # Step 1: Initial training with Chain A
        print_subsection("Step 1: Train with Chain A")
        chain_a = create_chain_a()
        print_chain_info(chain_a)

        initial_data = generate_training_data(n_samples=20)

        predictor_a = api.train_predictor_with_pipeline_v2(
            features_list=initial_data,
            prediction_type="quantile",
            config=None,
            chain=chain_a,
        )
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            predictor=predictor_a,
        )

        info_a = api.get_model_info(model_id, platform, "quantile")
        print("\n  Model trained with Chain A")
        print(f"  Samples: {info_a.samples_count}")
        print(f"  Features: {info_a.feature_names}")

        # Show sample transformation with Chain A
        sample_input = {
            "batch_size": 32,
            "sequence_length": 512,
            "hidden_size": 768,
        }
        transformed_a = chain_a.transform(sample_input)
        print("\n  Chain A transformation:")
        print(f"    Input:  {sample_input}")
        print(f"    Output: {transformed_a}")

        # Step 2: Collect new samples
        print_subsection("Step 2: Collect New Samples")
        new_samples = generate_training_data(n_samples=15)

        for sample in new_samples:
            runtime_ms = sample.pop("runtime_ms")
            core.collect(
                model_id=model_id,
                platform_info=platform,
                prediction_type="quantile",
                features=sample,
                runtime_ms=runtime_ms,
            )

        collected = core.get_collected_count(model_id, platform, "quantile")
        print(f"  Collected {collected} new samples")

        # Step 3: Retrain with Chain B (DIFFERENT chain)
        print_subsection("Step 3: Retrain with Chain B (Different!)")
        chain_b = create_chain_b()
        print_chain_info(chain_b)

        # Show sample transformation with Chain B
        transformed_b = chain_b.transform(sample_input)
        print("\n  Chain B transformation:")
        print(f"    Input:  {sample_input}")
        print(f"    Output: {transformed_b}")

        print("\n  ⚠️  Chain changed! This triggers FULL RETRAIN.")
        print("     Old model is discarded, training from scratch.")

        # Get all collected data for full retrain
        accumulated = core._accumulated.get(
            core._make_accumulator_key(model_id, platform, "quantile"), []
        )
        retrain_data = []
        for sample in accumulated:
            retrain_data.append(
                {**sample.features, "runtime_ms": sample.runtime_ms}
            )

        # For demonstration, combine with some of original data
        # In real scenario, you'd use all available raw data
        all_data = initial_data + retrain_data

        # Retrain with new chain
        predictor_b = api.train_predictor_with_pipeline_v2(
            features_list=all_data,
            prediction_type="quantile",
            config=None,
            chain=chain_b,  # Different chain!
        )
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            predictor=predictor_b,
        )

        core.clear_collected(model_id, platform, "quantile")

        # Verify
        print_subsection("Verification")
        info_b = api.get_model_info(model_id, platform, "quantile")
        print_result(
            "Model Info After Retrain",
            {
                "model_id": info_b.model_id,
                "samples_count": info_b.samples_count,
                "feature_names": info_b.feature_names,
            },
        )

        print("\n  Comparison:")
        print("    Chain A features: compute_cost, hidden_size")
        print(f"    Chain B features: {info_b.feature_names}")

        # Test prediction with new chain
        print_subsection("Test Prediction with Chain B")
        test_features = {
            "batch_size": 64,
            "sequence_length": 1024,
            "hidden_size": 1024,
        }
        result = core.inference_pipeline_v2(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            features=test_features,
            chain=chain_b,  # Must use Chain B now!
        )
        print(f"  Input: {test_features}")
        transformed = chain_b.transform(test_features)
        print(f"  After Chain B: {transformed}")
        print_result("  Prediction", result.result)

        # Summary
        print_subsection("Chain Change Summary")
        print("  When preprocessing chain changes:")
        print("    1. Feature space is different")
        print("    2. Old model predictions would be invalid")
        print("    3. Full retrain from raw data is required")
        print("    4. Model is replaced, not updated")
        print("")
        print("  Best practice: Keep raw data available for retraining")
        print("  when you need to experiment with different chains.")

        print("\n✓ Example 7 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
