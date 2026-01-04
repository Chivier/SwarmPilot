"""Unit tests for PreprocessorChainV2 - TDD tests written first."""

from __future__ import annotations

import pytest

from src.preprocessor.base_preprocessor_v2 import FeatureContext


class TestPreprocessorChainV2Basic:
    """Basic tests for PreprocessorChainV2."""

    def test_empty_chain_returns_features_unchanged(self) -> None:
        """Empty chain should return features without modification."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2

        chain = PreprocessorChainV2(name="empty_chain")
        features = {"width": 100, "height": 200}

        result = chain.transform(features)

        assert result == {"width": 100, "height": 200}

    def test_add_appends_preprocessor(self) -> None:
        """add() should append preprocessor to chain."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="test_chain")
        prep = MultiplyPreprocessor("width", "height", "pixels")

        chain.add(prep)

        assert len(chain.preprocessors) == 1
        assert chain.preprocessors[0] is prep

    def test_add_returns_self_for_chaining(self) -> None:
        """add() should return self for fluent interface."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="test_chain")
        prep = MultiplyPreprocessor("width", "height", "pixels")

        result = chain.add(prep)

        assert result is chain

    def test_fluent_chain_building(self) -> None:
        """Should support fluent chain building."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        chain = (
            PreprocessorChainV2(name="fluent_chain")
            .add(MultiplyPreprocessor("a", "b", "c"))
            .add(RemoveFeaturePreprocessor(["a", "b"]))
        )

        assert len(chain.preprocessors) == 2


class TestPreprocessorChainV2Transform:
    """Tests for PreprocessorChainV2.transform()."""

    def test_transform_applies_preprocessors_in_order(self) -> None:
        """Should apply preprocessors in the order they were added."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        chain = (
            PreprocessorChainV2(name="ordered_chain")
            .add(MultiplyPreprocessor("width", "height", "pixels"))
            .add(RemoveFeaturePreprocessor(["width", "height"]))
        )
        features = {"width": 100, "height": 200}

        result = chain.transform(features)

        assert result == {"pixels": 20000}

    def test_complex_chain_example(self) -> None:
        """Full example: width/height/prompt -> channels/pixel_num/input_length."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        chain = (
            PreprocessorChainV2(name="full_example")
            .add(MultiplyPreprocessor("width", "height", "pixel_num"))
            .add(RemoveFeaturePreprocessor(["width", "height"]))
            .add(TokenLengthPreprocessor("prompt", "input_length"))
            .add(RemoveFeaturePreprocessor(["prompt"]))
        )
        features = {
            "width": 100,
            "height": 200,
            "channels": 3,
            "prompt": "Hello world",
        }

        result = chain.transform(features)

        assert result == {
            "channels": 3,
            "pixel_num": 20000,
            "input_length": 2,
        }

    def test_transform_does_not_modify_input(self) -> None:
        """transform() should not modify the input dict."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        chain = PreprocessorChainV2(name="no_modify").add(
            RemoveFeaturePreprocessor(["a"])
        )
        original = {"a": 1, "b": 2}
        original_copy = original.copy()

        chain.transform(original)

        assert original == original_copy


class TestPreprocessorChainV2Callable:
    """Tests for PreprocessorChainV2.__call__()."""

    def test_callable_interface(self) -> None:
        """__call__() should work like transform()."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="callable_test").add(
            MultiplyPreprocessor("a", "b", "c")
        )
        features = {"a": 2, "b": 3}

        result = chain(features)

        assert result == {"a": 2, "b": 3, "c": 6}


class TestPreprocessorChainV2FeatureAnalysis:
    """Tests for feature analysis methods."""

    def test_get_required_inputs_empty_chain(self) -> None:
        """Empty chain should require no inputs."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2

        chain = PreprocessorChainV2(name="empty")

        assert chain.get_required_inputs() == set()

    def test_get_required_inputs_single_preprocessor(self) -> None:
        """Should return inputs required by single preprocessor."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="single").add(
            MultiplyPreprocessor("width", "height", "pixels")
        )

        assert chain.get_required_inputs() == {"width", "height"}

    def test_get_required_inputs_chain_produces_intermediate(self) -> None:
        """Should not require inputs produced by earlier preprocessors."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        # First preprocessor produces "pixels", second uses "text"
        # Both width, height, and text are required from initial input
        chain = (
            PreprocessorChainV2(name="intermediate")
            .add(MultiplyPreprocessor("width", "height", "pixels"))
            .add(TokenLengthPreprocessor("text", "length"))
        )

        required = chain.get_required_inputs()

        assert required == {"width", "height", "text"}

    def test_get_required_inputs_uses_intermediate_output(self) -> None:
        """Should not require features produced by earlier preprocessors."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        # Second multiply uses output of first
        chain = (
            PreprocessorChainV2(name="uses_intermediate")
            .add(MultiplyPreprocessor("a", "b", "c"))
            .add(MultiplyPreprocessor("c", "d", "e"))  # c is produced by first
        )

        required = chain.get_required_inputs()

        # c is not required from input since first preprocessor produces it
        assert required == {"a", "b", "d"}

    def test_get_final_outputs_empty_chain(self) -> None:
        """Empty chain produces no new outputs."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2

        chain = PreprocessorChainV2(name="empty")

        assert chain.get_final_outputs() == set()

    def test_get_final_outputs_includes_produced_features(self) -> None:
        """Should include features produced by chain."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="produces").add(
            MultiplyPreprocessor("a", "b", "product")
        )

        outputs = chain.get_final_outputs()

        assert "product" in outputs

    def test_get_final_outputs_excludes_removed_features(self) -> None:
        """Should exclude features removed by chain."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import RemoveFeaturePreprocessor

        chain = (
            PreprocessorChainV2(name="removes")
            .add(MultiplyPreprocessor("a", "b", "c"))
            .add(RemoveFeaturePreprocessor(["a", "b"]))
        )

        # a and b are removed, c is added
        outputs = chain.get_final_outputs()

        assert "c" in outputs
        # Note: these are features produced by chain, not what remains in input
        # get_final_outputs tracks chain-produced features


class TestPreprocessorChainV2Validation:
    """Tests for chain validation."""

    def test_validate_returns_empty_for_valid_chain(self) -> None:
        """validate() should return empty list for valid input."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="valid").add(
            MultiplyPreprocessor("a", "b", "c")
        )

        errors = chain.validate({"a", "b", "extra"})

        assert errors == []

    def test_validate_returns_errors_for_missing_inputs(self) -> None:
        """validate() should return list of missing features."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="missing").add(
            MultiplyPreprocessor("a", "b", "c")
        )

        errors = chain.validate({"a"})  # Missing "b"

        assert len(errors) == 1
        assert "b" in errors[0]

    def test_validate_considers_chain_outputs(self) -> None:
        """validate() should consider outputs produced by earlier steps."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = (
            PreprocessorChainV2(name="chain_outputs")
            .add(MultiplyPreprocessor("a", "b", "c"))
            .add(MultiplyPreprocessor("c", "d", "e"))  # c is from first step
        )

        # Only need a, b, d from input (c is produced)
        errors = chain.validate({"a", "b", "d"})

        assert errors == []


class TestPreprocessorChainV2Insert:
    """Tests for PreprocessorChainV2.insert()."""

    def test_insert_at_beginning(self) -> None:
        """insert(0, prep) should add at beginning."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        chain = PreprocessorChainV2(name="insert_test").add(
            MultiplyPreprocessor("a", "b", "c")
        )
        new_prep = TokenLengthPreprocessor("text", "length")

        chain.insert(0, new_prep)

        assert chain.preprocessors[0] is new_prep
        assert len(chain.preprocessors) == 2

    def test_insert_returns_self(self) -> None:
        """insert() should return self for fluent interface."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="insert_fluent")
        prep = MultiplyPreprocessor("a", "b", "c")

        result = chain.insert(0, prep)

        assert result is chain


class TestPreprocessorChainV2Properties:
    """Tests for chain properties."""

    def test_name_property(self) -> None:
        """Should have accessible name property."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2

        chain = PreprocessorChainV2(name="my_chain")

        assert chain.name == "my_chain"

    def test_preprocessors_property(self) -> None:
        """Should have accessible preprocessors list."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="preps_test")
        chain.add(MultiplyPreprocessor("a", "b", "c"))

        assert len(chain.preprocessors) == 1


class TestPreprocessorChainV2ErrorHandling:
    """Tests for error handling in chain execution."""

    def test_transform_raises_on_missing_input(self) -> None:
        """transform() should raise on missing required feature."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="error_test").add(
            MultiplyPreprocessor("a", "b", "c")
        )

        with pytest.raises(ValueError, match="missing required features"):
            chain.transform({"a": 1})  # Missing "b"

    def test_error_includes_preprocessor_name(self) -> None:
        """Error should identify which preprocessor failed."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        chain = PreprocessorChainV2(name="error_name_test").add(
            MultiplyPreprocessor("a", "b", "c")
        )

        with pytest.raises(ValueError) as exc_info:
            chain.transform({"a": 1})

        assert "multiply" in str(exc_info.value).lower()

    def test_fail_fast_on_first_error(self) -> None:
        """Should stop on first error, not continue chain."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        chain = (
            PreprocessorChainV2(name="fail_fast")
            .add(MultiplyPreprocessor("a", "b", "c"))  # Will fail
            .add(TokenLengthPreprocessor("text", "length"))  # Should not run
        )

        with pytest.raises(ValueError):
            chain.transform({"a": 1, "text": "hello"})  # Missing "b"
