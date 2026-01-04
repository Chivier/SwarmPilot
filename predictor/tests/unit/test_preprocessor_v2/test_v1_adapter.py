"""Unit tests for V1PreprocessorAdapter - TDD tests written first."""

from __future__ import annotations

from typing import Any

import pytest

from src.preprocessor.base_preprocessor import BasePreprocessor
from src.preprocessor.base_preprocessor_v2 import FeatureContext
from src.preprocessor.base_preprocessor_v2 import OperationType


class MockV1Preprocessor(BasePreprocessor):
    """Mock V1 preprocessor for testing."""

    def __init__(
        self,
        output_dict: dict[str, Any],
        remove_origin: bool = False,
    ) -> None:
        """Initialize mock with predetermined output."""
        super().__init__()
        self.output_dict = output_dict
        self.remove_origin = remove_origin
        self.call_count = 0
        self.last_input: list[str] | None = None

    def __call__(
        self,
        input_text: list[str],
    ) -> tuple[dict[str, Any], bool]:
        """Return predetermined output."""
        self.call_count += 1
        self.last_input = input_text
        return self.output_dict, self.remove_origin


class TestV1PreprocessorAdapterBasic:
    """Basic tests for V1PreprocessorAdapter."""

    def test_wraps_v1_preprocessor(self) -> None:
        """Should wrap a V1 preprocessor and call it."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello world"})

        adapter.transform(context)

        assert v1_prep.call_count == 1
        assert v1_prep.last_input == ["Hello world"]

    def test_adds_output_features_to_context(self) -> None:
        """Should add V1 output dict to context features."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor(
            {"output_length": 42, "complexity": 0.75},
            remove_origin=False,
        )
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello world"})

        adapter.transform(context)

        assert context.get("output_length") == 42
        assert context.get("complexity") == 0.75

    def test_preserves_input_when_remove_is_false(self) -> None:
        """Should preserve input feature when V1 returns remove_origin=False."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello world"})

        adapter.transform(context)

        assert context.has("prompt")
        assert context.get("prompt") == "Hello world"

    def test_removes_input_when_remove_is_true(self) -> None:
        """Should remove input feature when V1 returns remove_origin=True."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=True)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello world"})

        adapter.transform(context)

        assert not context.has("prompt")

    def test_preserves_other_features(self) -> None:
        """Should preserve features not related to the adapter."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=True)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(
            features={"prompt": "Hello", "width": 100, "height": 200}
        )

        adapter.transform(context)

        assert context.get("width") == 100
        assert context.get("height") == 200


class TestV1PreprocessorAdapterProperties:
    """Tests for V1PreprocessorAdapter properties."""

    def test_input_features_property(self) -> None:
        """input_features should return the input feature name."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
        )

        assert adapter.input_features == ["text"]

    def test_output_features_property(self) -> None:
        """output_features should be empty (dynamic from V1)."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
        )

        # V1 output is dynamic, so we can't declare it statically
        assert adapter.output_features == []

    def test_removes_features_empty_by_default(self) -> None:
        """removes_features should be empty (dynamic from V1)."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
        )

        # Removal is decided at runtime by V1 preprocessor
        assert adapter.removes_features == []

    def test_custom_name(self) -> None:
        """Should use custom name when provided."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
            name="semantic_predictor_adapter",
        )

        assert adapter.name == "semantic_predictor_adapter"

    def test_default_name(self) -> None:
        """Should generate default name from preprocessor class."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
        )

        assert "MockV1Preprocessor" in adapter.name

    def test_operation_type_is_transform(self) -> None:
        """Operation type should be TRANSFORM."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="text",
        )

        assert adapter.operation_type == OperationType.TRANSFORM


class TestV1PreprocessorAdapterValidation:
    """Tests for V1PreprocessorAdapter validation."""

    def test_raises_on_missing_input(self) -> None:
        """Should raise ValueError if input feature is missing."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=False)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"other": "value"})

        with pytest.raises(ValueError, match="missing required features"):
            adapter(context)  # Uses __call__ which validates


class TestV1PreprocessorAdapterTracking:
    """Tests for feature tracking in adapter."""

    def test_tracks_added_features(self) -> None:
        """Should track features added by V1 preprocessor."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor(
            {"output_length": 42, "complexity": 0.5},
            remove_origin=False,
        )
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello"})

        adapter.transform(context)

        assert "output_length" in context.added_features
        assert "complexity" in context.added_features

    def test_tracks_removed_features(self) -> None:
        """Should track input feature removal."""
        from src.preprocessor.adapters import V1PreprocessorAdapter

        v1_prep = MockV1Preprocessor({"output_length": 42}, remove_origin=True)
        adapter = V1PreprocessorAdapter(
            v1_preprocessor=v1_prep,
            input_feature="prompt",
        )
        context = FeatureContext(features={"prompt": "Hello"})

        adapter.transform(context)

        assert "prompt" in context.removed_features
