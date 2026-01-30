"""Example 1: Train a model without preprocessor chain.

Demonstrates basic training workflow using the Library API:
1. Generate training data with batch_size, sequence_length, hidden_size
2. Train expect_error predictor directly without preprocessing
3. Verify model is saved and can be listed

This is the simplest training scenario - raw features are used directly.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.library_api.utils import cleanup_storage
from examples.library_api.utils import create_low_level_with_storage
from examples.library_api.utils import generate_training_data
from examples.library_api.utils import get_platform_info
from examples.library_api.utils import print_features_summary
from examples.library_api.utils import print_result
from examples.library_api.utils import print_section
from examples.library_api.utils import print_subsection
from examples.library_api.utils import setup_storage


def main() -> None:
    """Run Example 1: Train without preprocessor."""
    print_section("Example 1: Train Without Preprocessor")

    # Setup isolated storage
    storage = setup_storage("ex01_no_preprocess")
    platform = get_platform_info()

    try:
        # Create low-level API with storage
        api = create_low_level_with_storage(storage)

        # Generate training data
        print_subsection("Training Data")
        features_list = generate_training_data(n_samples=25)
        print_features_summary(features_list)

        # Train without preprocessor
        print_subsection("Training Model")
        print("  Prediction type: expect_error")
        print("  Preprocessor chain: None")

        predictor = api.train_predictor_with_pipeline_v2(
            features_list=features_list,
            prediction_type="expect_error",
            config=None,
            chain=None,  # No preprocessing
        )

        print("  Training completed successfully!")

        # Save the model
        print_subsection("Saving Model")
        model_id = "inference-model"
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
            predictor=predictor,
        )
        print(f"  Model saved: {model_id}")

        # Verify model exists
        print_subsection("Verification")
        loaded = api.load_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
        )
        print(f"  Model loaded successfully: {loaded is not None}")

        # Get model info
        info = api.get_model_info(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
        )
        print_result(
            "Model Info",
            {
                "model_id": info.model_id,
                "samples_count": info.samples_count,
                "feature_names": info.feature_names,
                "last_trained": info.last_trained,
            },
        )

        print("\n✓ Example 1 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
