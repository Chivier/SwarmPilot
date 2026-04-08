"""Example 6: Collect new data and update model without changing chain.

Demonstrates incremental training workflow:
1. Train initial model with preprocessor chain
2. Collect new samples via collect() API
3. Retrain using the same chain - model is updated incrementally
4. Verify samples_trained increased

When the chain is unchanged (or None to use stored), the model
is updated with the new data rather than retrained from scratch.
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


def create_compute_chain() -> PreprocessorChainV2:
    """Create the standard compute chain for examples."""
    return (
        PreprocessorChainV2(name="compute_chain")
        .add(
            MultiplyPreprocessor(
                "batch_size", "sequence_length", "compute_cost"
            )
        )
        .add(RemoveFeaturePreprocessor(["batch_size", "sequence_length"]))
    )


def main() -> None:
    """Run Example 6: Collect and update with same chain."""
    print_section("Example 6: Collect & Update (Same Chain)")

    # Setup isolated storage
    storage = setup_storage("ex06_collect_same_chain")
    platform = get_platform_info()

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        # Create the preprocessing chain
        chain = create_compute_chain()
        model_id = "incremental-model"

        # Step 1: Initial training
        print_subsection("Step 1: Initial Training")
        initial_data = generate_training_data(n_samples=20)

        predictor = api.train_predictor_with_pipeline_v2(
            features_list=initial_data,
            prediction_type="quantile",
            config=None,
            chain=chain,
        )
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            predictor=predictor,
        )

        initial_info = api.get_model_info(model_id, platform, "quantile")
        print(f"  Model trained: {model_id}")
        print(f"  Initial samples: {initial_info.samples_count}")
        print_chain_info(chain)

        # Step 2: Collect new samples
        print_subsection("Step 2: Collect New Samples")
        new_samples = generate_training_data(n_samples=10)

        for i, sample in enumerate(new_samples):
            # Separate runtime_ms from features
            runtime_ms = sample.pop("runtime_ms")
            core.collect(
                model_id=model_id,
                platform_info=platform,
                prediction_type="quantile",
                features=sample,
                runtime_ms=runtime_ms,
            )

        collected_count = core.get_collected_count(
            model_id, platform, "quantile"
        )
        print(f"  Samples collected: {collected_count}")

        # Step 3: Retrain with SAME chain
        print_subsection("Step 3: Retrain with Same Chain")
        print("  Using the same chain as initial training")
        print("  This triggers incremental update, not full retrain")

        # Get accumulated samples and retrain.
        # Get the accumulated data and combine with new training.
        accumulated = core._accumulated.get(
            core._make_accumulator_key(model_id, platform, "quantile"), []
        )

        # Prepare combined features_list
        combined_data = []
        for sample in accumulated:
            combined_data.append(
                {**sample.features, "runtime_ms": sample.runtime_ms}
            )

        # Add to original data (simulating incremental)
        all_data = initial_data + combined_data

        # Retrain with same chain
        updated_predictor = api.train_predictor_with_pipeline_v2(
            features_list=all_data,
            prediction_type="quantile",
            config=None,
            chain=chain,  # Same chain!
        )
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            predictor=updated_predictor,
        )

        # Clear accumulated after training
        core.clear_collected(model_id, platform, "quantile")

        # Verify
        print_subsection("Verification")
        updated_info = api.get_model_info(model_id, platform, "quantile")
        print_result(
            "Model Info",
            {
                "model_id": updated_info.model_id,
                "samples_count": updated_info.samples_count,
                "feature_names": updated_info.feature_names,
            },
        )
        print(f"\n  Initial samples: {initial_info.samples_count}")
        print(f"  Updated samples: {updated_info.samples_count}")
        diff = updated_info.samples_count - initial_info.samples_count
        print(f"  Difference: +{diff}")

        # Test prediction
        print_subsection("Test Prediction")
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
            chain=chain,
        )
        print(f"  Input: {test_features}")
        print_result("  Prediction", result.result)

        print("\n✓ Example 6 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
