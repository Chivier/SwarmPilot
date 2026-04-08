"""Example 2: Train with preprocessor chain.

Demonstrates training with V2 preprocessor chain:
1. Create PreprocessorChainV2 with MultiplyPreprocessor
2. Chain computes: compute_cost = batch_size * sequence_length
3. Remove original features, keeping only compute_cost and hidden_size
4. Train on processed features
5. Verify chain is applied during training

After preprocessing, features transform from:
  {batch_size, sequence_length, hidden_size} → {compute_cost, hidden_size}
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
from examples.library_api.utils import print_chain_info
from examples.library_api.utils import print_features_summary
from examples.library_api.utils import print_result
from examples.library_api.utils import print_section
from examples.library_api.utils import print_subsection
from examples.library_api.utils import setup_storage
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_v2 import MultiplyPreprocessor
from swarmpilot.predictor.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor


def create_compute_chain() -> PreprocessorChainV2:
    """Create the standard compute chain for examples.

    Chain:
    1. Multiply batch_size * sequence_length → compute_cost
    2. Remove batch_size and sequence_length

    Returns:
        PreprocessorChainV2 instance.
    """
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
    """Run Example 2: Train with preprocessor."""
    print_section("Example 2: Train With Preprocessor")

    # Setup isolated storage
    storage = setup_storage("ex02_with_preprocess")
    platform = get_platform_info()

    try:
        # Create low-level API with storage
        api = create_low_level_with_storage(storage)

        # Create preprocessor chain
        print_subsection("Preprocessor Chain")
        chain = create_compute_chain()
        print_chain_info(chain)

        # Show transformation example
        print_subsection("Transformation Example")
        sample_input = {
            "batch_size": 32,
            "sequence_length": 512,
            "hidden_size": 768,
        }
        sample_output = chain.transform(sample_input)
        print(f"  Input:  {sample_input}")
        print(f"  Output: {sample_output}")

        # Generate training data
        print_subsection("Training Data")
        features_list = generate_training_data(n_samples=25)
        print_features_summary(features_list)

        # Train with preprocessor chain
        print_subsection("Training Model")
        print("  Prediction type: quantile")
        print("  Preprocessor chain: compute_chain")

        predictor = api.train_predictor_with_pipeline_v2(
            features_list=features_list,
            prediction_type="quantile",
            config=None,
            chain=chain,  # V2 preprocessing chain
        )

        print("  Training completed successfully!")
        print("  Features after chain: compute_cost, hidden_size")

        # Save the model
        print_subsection("Saving Model")
        model_id = "compute-predictor"
        api.save_model(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
            predictor=predictor,
        )
        print(f"  Model saved: {model_id}")

        # Verify model info
        print_subsection("Verification")
        info = api.get_model_info(
            model_id=model_id,
            platform_info=platform,
            prediction_type="quantile",
        )
        print_result(
            "Model Info",
            {
                "model_id": info.model_id,
                "samples_count": info.samples_count,
                "feature_names": info.feature_names,
            },
        )

        # Note: feature_names should be compute_cost, hidden_size (after chain)
        print("\n✓ Example 2 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
