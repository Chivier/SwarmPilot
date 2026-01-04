"""Unit tests for RemoveFeaturePreprocessor - TDD tests written first."""

from __future__ import annotations

import pytest

from src.preprocessor.base_preprocessor_v2 import FeatureContext


class TestRemoveFeaturePreprocessorBasic:
    """Basic tests for RemoveFeaturePreprocessor."""

    def test_removes_single_feature(self) -> None:
        """Should remove a single feature."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["to_remove"])
        context = FeatureContext(features={"to_remove": 1, "keep": 2})

        prep.transform(context)

        assert not context.has("to_remove")
        assert context.has("keep")

    def test_removes_multiple_features(self) -> None:
        """Should remove multiple features."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["a", "b", "c"])
        context = FeatureContext(features={"a": 1, "b": 2, "c": 3, "keep": 4})

        prep.transform(context)

        assert not context.has("a")
        assert not context.has("b")
        assert not context.has("c")
        assert context.has("keep")


class TestRemoveFeaturePreprocessorProperties:
    """Tests for RemoveFeaturePreprocessor properties."""

    def test_input_features_equals_features_to_remove(self) -> None:
        """input_features should match features_to_remove."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["x", "y"])

        assert set(prep.input_features) == {"x", "y"}

    def test_output_features_is_empty(self) -> None:
        """output_features should be empty."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["x"])

        assert prep.output_features == []

    def test_removes_features_property(self) -> None:
        """removes_features should match features_to_remove."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["a", "b"])

        assert set(prep.removes_features) == {"a", "b"}


class TestRemoveFeaturePreprocessorValidation:
    """Tests for RemoveFeaturePreprocessor validation."""

    def test_does_not_raise_on_missing_feature(self) -> None:
        """Should not raise if feature to remove is missing."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["nonexistent"])
        context = FeatureContext(features={"other": 1})

        # Should not raise - silently skip missing features
        prep.transform(context)

        assert context.has("other")


class TestRemoveFeaturePreprocessorTracking:
    """Tests for tracking removed features."""

    def test_tracks_removed_in_context(self) -> None:
        """Should track removed features in context."""
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        prep = RemoveFeaturePreprocessor(features_to_remove=["x"])
        context = FeatureContext(features={"x": 1, "y": 2})

        prep.transform(context)

        assert "x" in context.removed_features
        assert "y" not in context.removed_features
