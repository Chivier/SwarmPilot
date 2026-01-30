"""Example 5: Error handling scenarios.

Demonstrates common error scenarios and how to handle them:
1. Training with insufficient data (< 10 samples)
2. Chain validation failure (missing required features)
3. Prediction on non-existent model
4. Invalid feature values

Each error type has a corresponding exception class that you can catch.
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
from examples.library_api.utils import print_section
from examples.library_api.utils import print_subsection
from examples.library_api.utils import setup_storage
from swarmpilot.predictor.api.core import ModelNotFoundError
from swarmpilot.predictor.api.core import TrainingError
from swarmpilot.predictor.api.core import ValidationError
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_v2 import MultiplyPreprocessor


def demo_insufficient_data(api) -> None:
    """Demonstrate training failure with insufficient data."""
    print_subsection("Error 1: Insufficient Training Data")
    print("  Attempting to train with only 5 samples...")
    print("  (Minimum required: 10 samples)")

    # Generate only 5 samples
    features_list = generate_training_data(n_samples=5)

    try:
        api.train_predictor_with_pipeline_v2(
            features_list=features_list,
            prediction_type="expect_error",
            config=None,
            chain=None,
        )
        print("  ERROR: Should have raised an exception!")
    except (ValidationError, TrainingError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
        print(f"    Message: {e}")


def demo_chain_validation_failure(api) -> None:
    """Demonstrate chain validation failure."""
    print_subsection("Error 2: Chain Validation Failure")
    print("  Training data has: batch_size, sequence_length, hidden_size")
    print("  Chain requires: width, height")
    print("  This mismatch will cause validation to fail.")

    # Chain that requires features not in our data
    chain = PreprocessorChainV2(name="invalid_chain").add(
        MultiplyPreprocessor("width", "height", "pixels")
    )

    features_list = generate_training_data(n_samples=25)

    try:
        api.train_predictor_with_pipeline_v2(
            features_list=features_list,
            prediction_type="expect_error",
            config=None,
            chain=chain,
        )
        print("  ERROR: Should have raised an exception!")
    except (ValidationError, KeyError, ValueError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
        print(f"    Message: {e}")


def demo_model_not_found(core) -> None:
    """Demonstrate model not found error."""
    print_subsection("Error 3: Model Not Found")
    print("  Attempting to predict with non-existent model...")

    platform = get_platform_info()

    try:
        core.predict(
            model_id="non-existent-model",
            platform_info=platform,
            prediction_type="expect_error",
            features={
                "batch_size": 32,
                "sequence_length": 512,
                "hidden_size": 768,
            },
        )
        print("  ERROR: Should have raised an exception!")
    except ModelNotFoundError as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
        print(f"    Message: {e}")


def demo_empty_features_list(api) -> None:
    """Demonstrate empty features list error."""
    print_subsection("Error 4: Empty Features List")
    print("  Attempting to train with empty features_list...")

    try:
        api.train_predictor_with_pipeline_v2(
            features_list=[],  # Empty list
            prediction_type="expect_error",
            config=None,
            chain=None,
        )
        print("  ERROR: Should have raised an exception!")
    except (ValidationError, TrainingError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
        print(f"    Message: {e}")


def main() -> None:
    """Run Example 5: Error handling."""
    print_section("Example 5: Error Handling")

    # Setup isolated storage
    storage = setup_storage("ex05_error_handling")

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        # Run all error demos
        demo_insufficient_data(api)
        demo_chain_validation_failure(api)
        demo_model_not_found(core)
        demo_empty_features_list(api)

        # Summary
        print_subsection("Error Handling Summary")
        print("  Exception Types:")
        print("    - ValidationError: Invalid input data or parameters")
        print("    - TrainingError: Training process failure")
        print("    - ModelNotFoundError: Model doesn't exist")
        print("    - PredictionError: Prediction process failure")
        print("")
        print("  Best Practices:")
        print("    1. Always validate data before training")
        print("    2. Use try/except to handle expected errors")
        print("    3. Log errors for debugging")
        print("    4. Provide clear error messages to users")

        print("\n✓ Example 5 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
