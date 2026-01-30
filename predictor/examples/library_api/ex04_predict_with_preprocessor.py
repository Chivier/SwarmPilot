"""Example 4: Predict with stored preprocessor chain.

Demonstrates prediction with V2 preprocessing:
1. Train a model with preprocessor chain (from Example 2)
2. Make predictions - chain is applied to input features
3. Show how features are transformed before prediction

The key insight: when using inference_pipeline_v2(), the chain
is applied to input features automatically, so you provide
raw features (batch_size, sequence_length) and the chain
transforms them to what the model expects (compute_cost).
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
from src.preprocessor.chain_v2 import PreprocessorChainV2
from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor


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
    """Run Example 4: Predict with preprocessor."""
    print_section("Example 4: Predict With Preprocessor")

    # Setup isolated storage
    storage = setup_storage("ex04_predict_with_preprocess")
    platform = get_platform_info()

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        # Create the preprocessing chain
        chain = create_compute_chain()

        # First, train a model with the chain (same as Example 2)
        print_subsection("Training Model With Chain")
        features_list = generate_training_data(n_samples=25)
        model_id = "compute-predictor"

        predictor = api.train_predictor_with_pipeline_v2(
            features_list=features_list,
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
        print(f"  Model trained with chain: {model_id}")
        print_chain_info(chain)

        # Make predictions with chain
        print_subsection("Making Predictions")
        print("  Using inference_pipeline_v2() with chain")

        # Test case 1: Small computation
        features1 = {
            "batch_size": 16,
            "sequence_length": 256,
            "hidden_size": 512,
        }
        print(f"\n  Raw Input: {features1}")

        # Show transformation
        transformed1 = chain.transform(features1)
        print(f"  After Chain: {transformed1}")

        result1 = core.inference_pipeline_v2(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            features=features1,
            chain=chain,  # Chain applied to features
        )
        print_result("  Prediction", result1.result)

        # Test case 2: Large computation
        features2 = {
            "batch_size": 64,
            "sequence_length": 1024,
            "hidden_size": 1024,
        }
        print(f"\n  Raw Input: {features2}")

        transformed2 = chain.transform(features2)
        print(f"  After Chain: {transformed2}")
        bs = features2["batch_size"]
        sl = features2["sequence_length"]
        cc = transformed2["compute_cost"]
        print(f"  compute_cost = {bs} x {sl} = {cc}")

        result2 = core.inference_pipeline_v2(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            features=features2,
            chain=chain,
        )
        print_result("  Prediction", result2.result)

        # Explanation
        print_subsection("Understanding Chain Application")
        print(
            "  The preprocessing chain transforms features before prediction:"
        )
        print("    1. MultiplyPreprocessor:")
        print("       compute_cost = batch_size x sequence_length")
        print("    2. RemoveFeaturePreprocessor:")
        print("       removes batch_size, sequence_length")
        print("")
        print("  The model only sees: {compute_cost, hidden_size}")
        print("  But you provide raw features for convenience!")

        print("\n✓ Example 4 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
