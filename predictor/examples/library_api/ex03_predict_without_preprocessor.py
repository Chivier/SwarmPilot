"""Example 3: Make predictions without preprocessor.

Demonstrates prediction workflow without preprocessing:
1. Train a model without preprocessor (from Example 1)
2. Make predictions using the trained model
3. Interpret expect_error prediction results (mean, std)

This example shows the standard predict workflow for models
trained without any preprocessing chain.
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
from examples.library_api.utils import print_result
from examples.library_api.utils import print_section
from examples.library_api.utils import print_subsection
from examples.library_api.utils import setup_storage


def main() -> None:
    """Run Example 3: Predict without preprocessor."""
    print_section("Example 3: Predict Without Preprocessor")

    # Setup isolated storage
    storage = setup_storage("ex03_predict_no_preprocess")
    platform = get_platform_info()

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        # First, train a model (same as Example 1)
        print_subsection("Training Model First")
        features_list = generate_training_data(n_samples=25)
        model_id = "inference-model"

        predictor = api.train_predictor_with_pipeline_v2(
            features_list=features_list,
            prediction_type="expect_error",
            config=None,
            chain=None,
        )
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
            predictor=predictor,
        )
        print(f"  Model trained and saved: {model_id}")

        # Make predictions
        print_subsection("Making Predictions")

        # Test case 1: Small batch
        features_small = {
            "batch_size": 16,
            "sequence_length": 256,
            "hidden_size": 512,
        }
        print(f"\n  Input 1: {features_small}")

        result1 = core.predict(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
            features=features_small,
        )
        print_result("  Prediction 1", result1.result)

        # Test case 2: Medium batch
        features_medium = {
            "batch_size": 32,
            "sequence_length": 512,
            "hidden_size": 768,
        }
        print(f"\n  Input 2: {features_medium}")

        result2 = core.predict(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
            features=features_medium,
        )
        print_result("  Prediction 2", result2.result)

        # Test case 3: Large batch
        features_large = {
            "batch_size": 128,
            "sequence_length": 2048,
            "hidden_size": 1024,
        }
        print(f"\n  Input 3: {features_large}")

        result3 = core.predict(
            model_id=model_id,
            platform_info=platform,
            prediction_type="expect_error",
            features=features_large,
        )
        print_result("  Prediction 3", result3.result)

        # Explanation
        print_subsection("Understanding Results")
        print("  expect_error predictions return:")
        print("    - mean: Expected runtime in ms")
        print("    - std: Standard deviation (uncertainty)")
        print("")
        print("  As batch_size and sequence_length increase,")
        print("  the predicted runtime should also increase.")

        print("\n✓ Example 3 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
