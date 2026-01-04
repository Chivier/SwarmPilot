"""Unit tests for MultiplyPreprocessor - TDD tests written first."""

from __future__ import annotations

import pytest

from src.preprocessor.base_preprocessor_v2 import FeatureContext


class TestMultiplyPreprocessorBasic:
    """Basic tests for MultiplyPreprocessor."""

    def test_multiplies_two_features(self) -> None:
        """Should multiply two features and store result."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="width",
            feature_b="height",
            output_feature="pixel_num",
        )
        context = FeatureContext(features={"width": 100, "height": 200})

        prep.transform(context)

        assert context.get("pixel_num") == 20000

    def test_preserves_inputs_by_default(self) -> None:
        """Should preserve input features by default."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="width",
            feature_b="height",
            output_feature="pixel_num",
        )
        context = FeatureContext(features={"width": 100, "height": 200})

        prep.transform(context)

        assert context.has("width")
        assert context.has("height")

    def test_removes_inputs_when_configured(self) -> None:
        """Should remove inputs when remove_inputs=True."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="width",
            feature_b="height",
            output_feature="pixel_num",
            remove_inputs=True,
        )
        context = FeatureContext(features={"width": 100, "height": 200})

        prep.transform(context)

        assert not context.has("width")
        assert not context.has("height")
        assert context.get("pixel_num") == 20000


class TestMultiplyPreprocessorProperties:
    """Tests for MultiplyPreprocessor properties."""

    def test_input_features_property(self) -> None:
        """input_features should return both inputs."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="a", feature_b="b", output_feature="c"
        )

        assert set(prep.input_features) == {"a", "b"}

    def test_output_features_property(self) -> None:
        """output_features should return the output feature."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="a", feature_b="b", output_feature="result"
        )

        assert prep.output_features == ["result"]

    def test_removes_features_empty_by_default(self) -> None:
        """removes_features should be empty when remove_inputs=False."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="a", feature_b="b", output_feature="c"
        )

        assert prep.removes_features == []

    def test_removes_features_when_configured(self) -> None:
        """removes_features should list inputs when remove_inputs=True."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="a", feature_b="b", output_feature="c", remove_inputs=True
        )

        assert set(prep.removes_features) == {"a", "b"}


class TestMultiplyPreprocessorValidation:
    """Tests for MultiplyPreprocessor input validation."""

    def test_raises_on_missing_input(self) -> None:
        """Should raise ValueError if input feature missing."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="width",
            feature_b="height",
            output_feature="pixel_num",
        )
        context = FeatureContext(features={"width": 100})  # Missing height

        with pytest.raises(ValueError, match="missing required features"):
            prep(context)  # Uses __call__ which validates

    def test_error_shows_missing_feature_name(self) -> None:
        """Error should indicate which feature is missing."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="width",
            feature_b="height",
            output_feature="pixel_num",
        )
        context = FeatureContext(features={"width": 100})

        with pytest.raises(ValueError) as exc_info:
            prep(context)

        assert "height" in str(exc_info.value)


class TestMultiplyPreprocessorEdgeCases:
    """Edge case tests for MultiplyPreprocessor."""

    def test_multiplies_floats(self) -> None:
        """Should work with float values."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="x", feature_b="y", output_feature="product"
        )
        context = FeatureContext(features={"x": 2.5, "y": 4.0})

        prep.transform(context)

        assert context.get("product") == 10.0

    def test_multiplies_same_feature_twice(self) -> None:
        """Should work when multiplying feature by itself."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="x", feature_b="x", output_feature="x_squared"
        )
        context = FeatureContext(features={"x": 5})

        prep.transform(context)

        assert context.get("x_squared") == 25

    def test_preserves_other_features(self) -> None:
        """Should preserve features not involved in multiplication."""
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        prep = MultiplyPreprocessor(
            feature_a="a", feature_b="b", output_feature="c"
        )
        context = FeatureContext(features={"a": 2, "b": 3, "other": 999})

        prep.transform(context)

        assert context.get("other") == 999
