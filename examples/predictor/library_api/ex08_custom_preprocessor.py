"""Example 8: Define and use custom preprocessor.

Demonstrates creating custom preprocessors:
1. Subclass BasePreprocessorV2
2. Implement required abstract properties
3. Implement transform() method
4. Use in a preprocessing chain
5. Train and predict with custom preprocessor

This example creates a LogTransformPreprocessor that applies
log10 transformation to numeric features.
"""

from __future__ import annotations

import math
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
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import BasePreprocessorV2
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import FeatureContext
from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import OperationType
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_v2 import MultiplyPreprocessor
from swarmpilot.predictor.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor


# =============================================================================
# Custom Preprocessor Definition
# =============================================================================


class LogTransformPreprocessor(BasePreprocessorV2):
    """Apply log10 transformation to a numeric feature.

    This custom preprocessor demonstrates how to:
    - Subclass BasePreprocessorV2
    - Implement required abstract properties
    - Implement the transform() method
    - Handle optional input removal

    The log transform is useful for features with large ranges
    (like compute_cost which can be batch_size * sequence_length).
    """

    def __init__(
        self,
        input_feature: str,
        output_feature: str | None = None,
        remove_input: bool = False,
    ):
        """Initialize LogTransformPreprocessor.

        Args:
            input_feature: Name of feature to transform.
            output_feature: Name for output feature.
                Defaults to f"log_{input_feature}".
            remove_input: Whether to remove input feature after transform.
        """
        self._input_feature = input_feature
        self._output_feature = output_feature or f"log_{input_feature}"
        self._remove_input = remove_input
        self._name = f"log_transform_{input_feature}"
        self._operation_type = OperationType.TRANSFORM

    @property
    def name(self) -> str:
        """Preprocessor name."""
        return self._name

    @property
    def operation_type(self) -> OperationType:
        """Type of operation performed."""
        return self._operation_type

    @property
    def input_features(self) -> list[str]:
        """Features required as input."""
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        """Features produced as output."""
        return [self._output_feature]

    @property
    def removes_features(self) -> list[str]:
        """Features removed by this preprocessor."""
        return [self._input_feature] if self._remove_input else []

    def transform(self, context: FeatureContext) -> None:
        """Apply log10 transformation.

        Uses log10(max(value, 1)) to handle values <= 0 safely.

        Args:
            context: Feature context to modify in-place.
        """
        value = context.get(self._input_feature)

        # Apply log10, ensuring we don't take log of 0 or negative
        log_value = math.log10(max(float(value), 1.0))

        context.set(self._output_feature, round(log_value, 4))

        if self._remove_input:
            context.remove(self._input_feature)


class NormalizePreprocessor(BasePreprocessorV2):
    """Normalize a feature to [0, 1] range using min-max scaling.

    Another example of a custom preprocessor that requires
    additional state (min/max values).
    """

    def __init__(
        self,
        input_feature: str,
        min_val: float,
        max_val: float,
        output_feature: str | None = None,
        remove_input: bool = False,
    ):
        """Initialize NormalizePreprocessor.

        Args:
            input_feature: Feature to normalize.
            min_val: Expected minimum value.
            max_val: Expected maximum value.
            output_feature: Output feature name.
            remove_input: Whether to remove input.
        """
        self._input_feature = input_feature
        self._min_val = min_val
        self._max_val = max_val
        self._output_feature = output_feature or f"norm_{input_feature}"
        self._remove_input = remove_input
        self._name = f"normalize_{input_feature}"
        self._operation_type = OperationType.TRANSFORM

    @property
    def name(self) -> str:
        """Preprocessor name."""
        return self._name

    @property
    def operation_type(self) -> OperationType:
        """Type of operation performed."""
        return self._operation_type

    @property
    def input_features(self) -> list[str]:
        """Features required as input."""
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        """Features produced as output."""
        return [self._output_feature]

    @property
    def removes_features(self) -> list[str]:
        """Features removed by this preprocessor."""
        return [self._input_feature] if self._remove_input else []

    def transform(self, context: FeatureContext) -> None:
        """Apply min-max normalization."""
        value = float(context.get(self._input_feature))

        # Clamp to range and normalize
        clamped = max(self._min_val, min(self._max_val, value))
        normalized = (clamped - self._min_val) / (self._max_val - self._min_val)

        context.set(self._output_feature, round(normalized, 4))

        if self._remove_input:
            context.remove(self._input_feature)


def main() -> None:
    """Run Example 8: Custom preprocessor."""
    print_section("Example 8: Custom Preprocessor")

    # Setup isolated storage
    storage = setup_storage("ex08_custom_preprocessor")
    platform = get_platform_info()

    try:
        # Create APIs with storage
        api = create_low_level_with_storage(storage)
        core = create_core_with_storage(storage)

        model_id = "custom-preprocess-model"

        # Step 1: Define the custom preprocessor
        print_subsection("Step 1: Custom Preprocessor Definition")
        print("  LogTransformPreprocessor:")
        print("    - Applies log10 to numeric features")
        print("    - Useful for large-range features")
        print("    - Example: compute_cost (can be millions)")
        print("")
        print("  Required implementations:")
        print("    - name (property): Unique identifier")
        print("    - operation_type (property): OperationType enum")
        print("    - input_features (property): List of required inputs")
        print("    - output_features (property): List of outputs")
        print("    - transform(context): Modify FeatureContext in-place")

        # Step 2: Create chain with custom preprocessor
        print_subsection("Step 2: Create Chain with Custom Preprocessor")

        chain = (
            PreprocessorChainV2(name="custom_chain")
            # First: compute batch_size * sequence_length
            .add(
                MultiplyPreprocessor(
                    "batch_size", "sequence_length", "compute_cost"
                )
            )
            # Second: apply log10 to compute_cost
            .add(
                LogTransformPreprocessor(
                    "compute_cost", "log_compute", remove_input=True
                )
            )
            # Third: normalize hidden_size (expected range 256-1024)
            .add(
                NormalizePreprocessor(
                    "hidden_size", 256, 1024, "norm_hidden", remove_input=True
                )
            )
            # Fourth: remove original features
            .add(RemoveFeaturePreprocessor(["batch_size", "sequence_length"]))
        )

        print_chain_info(chain)

        # Show transformation example
        print_subsection("Step 3: Transformation Example")
        sample_input = {
            "batch_size": 64,
            "sequence_length": 2048,
            "hidden_size": 768,
        }
        sample_output = chain.transform(sample_input)

        print(f"  Input:  {sample_input}")
        print(f"  Output: {sample_output}")
        print("")
        compute_cost = (
            sample_input["batch_size"] * sample_input["sequence_length"]
        )
        print("  Breakdown:")
        print(f"    compute_cost = 64 * 2048 = {compute_cost}")
        log_val = math.log10(compute_cost)
        norm_val = (768 - 256) / (1024 - 256)
        print(f"    log_compute = log10({compute_cost}) = {log_val:.4f}")
        print(f"    norm_hidden = (768-256)/(1024-256) = {norm_val:.4f}")

        # Step 4: Train with custom chain
        print_subsection("Step 4: Train with Custom Chain")
        features_list = generate_training_data(n_samples=25)

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

        info = api.get_model_info(model_id, platform, "quantile")
        print(f"  Model trained: {model_id}")
        print(f"  Samples: {info.samples_count}")
        print(f"  Features after chain: {info.feature_names}")

        # Step 5: Make predictions
        print_subsection("Step 5: Make Predictions")

        test_cases = [
            {"batch_size": 16, "sequence_length": 256, "hidden_size": 512},
            {"batch_size": 64, "sequence_length": 1024, "hidden_size": 768},
            {"batch_size": 128, "sequence_length": 2048, "hidden_size": 1024},
        ]

        for i, features in enumerate(test_cases, 1):
            transformed = chain.transform(features)
            result = core.inference_pipeline_v2(
                model_id=model_id,
                platform_info=platform,
                prediction_type="quantile",
                features=features,
                chain=chain,
            )

            print(f"\n  Test {i}:")
            print(f"    Raw: {features}")
            print(f"    Transformed: {transformed}")
            print_result("    Prediction", result.result)

        # Summary
        print_subsection("Custom Preprocessor Summary")
        print("  To create a custom preprocessor:")
        print("    1. Subclass BasePreprocessorV2")
        print("    2. Implement required properties:")
        print("       - name, operation_type")
        print("       - input_features, output_features")
        print("    3. Implement transform(context) method")
        print("    4. Optionally implement removes_features")
        print("")
        print("  The transform() method receives a FeatureContext")
        print("  that provides get(), set(), remove() methods for")
        print("  modifying features in-place.")

        print("\n✓ Example 8 completed successfully!")

    finally:
        # Cleanup
        cleanup_storage(storage)


if __name__ == "__main__":
    main()
