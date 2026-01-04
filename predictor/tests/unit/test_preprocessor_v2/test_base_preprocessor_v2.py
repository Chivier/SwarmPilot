"""Unit tests for BasePreprocessorV2 - TDD tests.

These tests verify the abstract base class behavior and contract.
"""

from __future__ import annotations

import pytest

from src.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
    FeatureContext,
    OperationType,
)


class SimpleDoublePreprocessor(BasePreprocessorV2):
    """Concrete preprocessor for testing: doubles a value."""

    def __init__(
        self,
        input_feature: str = "value",
        output_feature: str = "doubled",
    ) -> None:
        super().__init__(name="simple_double", operation_type=OperationType.ADD)
        self._input_feature = input_feature
        self._output_feature = output_feature

    @property
    def input_features(self) -> list[str]:
        return [self._input_feature]

    @property
    def output_features(self) -> list[str]:
        return [self._output_feature]

    def transform(self, context: FeatureContext) -> None:
        value = context.get(self._input_feature)
        context.set(self._output_feature, value * 2)


class TestOperationType:
    """Tests for OperationType enum."""

    def test_operation_types_exist(self) -> None:
        """All expected operation types should exist."""
        assert OperationType.MODIFY.value == "modify"
        assert OperationType.ADD.value == "add"
        assert OperationType.REMOVE.value == "remove"
        assert OperationType.TRANSFORM.value == "transform"


class TestBasePreprocessorV2Init:
    """Tests for BasePreprocessorV2 initialization."""

    def test_init_sets_name(self) -> None:
        """Preprocessor should store its name."""
        prep = SimpleDoublePreprocessor()

        assert prep.name == "simple_double"

    def test_init_sets_operation_type(self) -> None:
        """Preprocessor should store its operation type."""
        prep = SimpleDoublePreprocessor()

        assert prep.operation_type == OperationType.ADD

    def test_default_operation_type_is_transform(self) -> None:
        """Default operation type should be TRANSFORM."""

        class DefaultOpPreprocessor(BasePreprocessorV2):
            @property
            def input_features(self) -> list[str]:
                return []

            @property
            def output_features(self) -> list[str]:
                return []

            def transform(self, context: FeatureContext) -> None:
                pass

        prep = DefaultOpPreprocessor(name="default_op")

        assert prep.operation_type == OperationType.TRANSFORM


class TestBasePreprocessorV2Properties:
    """Tests for BasePreprocessorV2 property methods."""

    def test_input_features_returns_list(self) -> None:
        """input_features should return list of feature names."""
        prep = SimpleDoublePreprocessor(input_feature="x")

        assert prep.input_features == ["x"]

    def test_output_features_returns_list(self) -> None:
        """output_features should return list of feature names."""
        prep = SimpleDoublePreprocessor(output_feature="y")

        assert prep.output_features == ["y"]

    def test_removes_features_default_empty(self) -> None:
        """removes_features should default to empty list."""
        prep = SimpleDoublePreprocessor()

        assert prep.removes_features == []


class TestBasePreprocessorV2Transform:
    """Tests for BasePreprocessorV2.transform() method."""

    def test_transform_modifies_context(self) -> None:
        """transform() should modify the context."""
        prep = SimpleDoublePreprocessor(input_feature="x", output_feature="doubled_x")
        context = FeatureContext(features={"x": 5})

        prep.transform(context)

        assert context.get("doubled_x") == 10

    def test_transform_preserves_original(self) -> None:
        """transform() should preserve original features."""
        prep = SimpleDoublePreprocessor(input_feature="x", output_feature="doubled_x")
        context = FeatureContext(features={"x": 5, "y": 10})

        prep.transform(context)

        assert context.get("x") == 5
        assert context.get("y") == 10
        assert context.get("doubled_x") == 10


class TestBasePreprocessorV2ValidateInputs:
    """Tests for BasePreprocessorV2.validate_inputs() method."""

    def test_validate_inputs_returns_empty_when_all_present(self) -> None:
        """validate_inputs() should return empty list when all inputs present."""
        prep = SimpleDoublePreprocessor(input_feature="x")
        context = FeatureContext(features={"x": 5})

        missing = prep.validate_inputs(context)

        assert missing == []

    def test_validate_inputs_returns_missing_features(self) -> None:
        """validate_inputs() should return list of missing features."""
        prep = SimpleDoublePreprocessor(input_feature="x")
        context = FeatureContext(features={"y": 10})

        missing = prep.validate_inputs(context)

        assert missing == ["x"]


class TestBasePreprocessorV2Call:
    """Tests for BasePreprocessorV2.__call__() method."""

    def test_call_executes_transform(self) -> None:
        """__call__() should execute transform."""
        prep = SimpleDoublePreprocessor(input_feature="x", output_feature="doubled_x")
        context = FeatureContext(features={"x": 5})

        prep(context)

        assert context.get("doubled_x") == 10

    def test_call_raises_on_missing_inputs(self) -> None:
        """__call__() should raise ValueError on missing inputs."""
        prep = SimpleDoublePreprocessor(input_feature="x")
        context = FeatureContext(features={"y": 10})

        with pytest.raises(ValueError, match="missing required features"):
            prep(context)

    def test_call_error_includes_available_features(self) -> None:
        """__call__() error should include available features."""
        prep = SimpleDoublePreprocessor(input_feature="x")
        context = FeatureContext(features={"a": 1, "b": 2})

        with pytest.raises(ValueError) as exc_info:
            prep(context)

        error_msg = str(exc_info.value)
        assert "Available features" in error_msg

    def test_call_error_includes_preprocessor_name(self) -> None:
        """__call__() error should include preprocessor name."""
        prep = SimpleDoublePreprocessor(input_feature="x")
        context = FeatureContext(features={"y": 10})

        with pytest.raises(ValueError) as exc_info:
            prep(context)

        assert "simple_double" in str(exc_info.value)


class TestBasePreprocessorV2Abstract:
    """Tests verifying abstract method requirements."""

    def test_cannot_instantiate_without_input_features(self) -> None:
        """Cannot instantiate without implementing input_features."""

        class IncompletePreprocessor(BasePreprocessorV2):
            @property
            def output_features(self) -> list[str]:
                return []

            def transform(self, context: FeatureContext) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompletePreprocessor(name="incomplete")

    def test_cannot_instantiate_without_output_features(self) -> None:
        """Cannot instantiate without implementing output_features."""

        class IncompletePreprocessor(BasePreprocessorV2):
            @property
            def input_features(self) -> list[str]:
                return []

            def transform(self, context: FeatureContext) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompletePreprocessor(name="incomplete")

    def test_cannot_instantiate_without_transform(self) -> None:
        """Cannot instantiate without implementing transform."""

        class IncompletePreprocessor(BasePreprocessorV2):
            @property
            def input_features(self) -> list[str]:
                return []

            @property
            def output_features(self) -> list[str]:
                return []

        with pytest.raises(TypeError, match="abstract"):
            IncompletePreprocessor(name="incomplete")
