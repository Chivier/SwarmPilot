"""Unit tests for TokenLengthPreprocessor - TDD tests written first."""

from __future__ import annotations

import pytest

from src.preprocessor.base_preprocessor_v2 import FeatureContext


class TestTokenLengthPreprocessorBasic:
    """Basic tests for TokenLengthPreprocessor."""

    def test_calculates_token_length_default_tokenizer(self) -> None:
        """Should calculate token length with default whitespace tokenizer."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="text",
            output_feature="token_count",
        )
        context = FeatureContext(features={"text": "Hello world test"})

        prep.transform(context)

        assert context.get("token_count") == 3

    def test_uses_custom_tokenizer(self) -> None:
        """Should use custom tokenizer when provided."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        # Tokenizer that splits on commas
        comma_tokenizer = lambda x: x.split(",")

        prep = TokenLengthPreprocessor(
            input_feature="text",
            output_feature="token_count",
            tokenizer=comma_tokenizer,
        )
        context = FeatureContext(features={"text": "a,b,c,d,e"})

        prep.transform(context)

        assert context.get("token_count") == 5

    def test_removes_input_when_configured(self) -> None:
        """Should remove input when remove_input=True."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="text",
            output_feature="token_count",
            remove_input=True,
        )
        context = FeatureContext(features={"text": "Hello world"})

        prep.transform(context)

        assert not context.has("text")
        assert context.get("token_count") == 2


class TestTokenLengthPreprocessorProperties:
    """Tests for TokenLengthPreprocessor properties."""

    def test_input_features_property(self) -> None:
        """input_features should return input feature name."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="prompt", output_feature="length"
        )

        assert prep.input_features == ["prompt"]

    def test_output_features_property(self) -> None:
        """output_features should return output feature name."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="prompt", output_feature="input_length"
        )

        assert prep.output_features == ["input_length"]

    def test_default_output_feature(self) -> None:
        """Default output_feature should be 'input_length'."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(input_feature="text")

        assert prep.output_features == ["input_length"]

    def test_removes_features_empty_by_default(self) -> None:
        """removes_features should be empty when remove_input=False."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(input_feature="text")

        assert prep.removes_features == []

    def test_removes_features_when_configured(self) -> None:
        """removes_features should list input when remove_input=True."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(input_feature="text", remove_input=True)

        assert prep.removes_features == ["text"]


class TestTokenLengthPreprocessorEdgeCases:
    """Edge case tests for TokenLengthPreprocessor."""

    def test_empty_string_returns_zero(self) -> None:
        """Should return 0 for empty string."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="text", output_feature="count"
        )
        context = FeatureContext(features={"text": ""})

        prep.transform(context)

        # Empty string split gives [""], but we should handle this properly
        # Actually, "".split() returns [], so count should be 0
        assert context.get("count") == 0

    def test_single_word_returns_one(self) -> None:
        """Should return 1 for single word."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="text", output_feature="count"
        )
        context = FeatureContext(features={"text": "hello"})

        prep.transform(context)

        assert context.get("count") == 1

    def test_preserves_other_features(self) -> None:
        """Should preserve features not being processed."""
        from src.preprocessor.preprocessors_v2 import TokenLengthPreprocessor

        prep = TokenLengthPreprocessor(
            input_feature="text", output_feature="count"
        )
        context = FeatureContext(features={"text": "hello", "other": 999})

        prep.transform(context)

        assert context.get("other") == 999
