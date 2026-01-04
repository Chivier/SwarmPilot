"""Unit tests for PreprocessorsRegistryV2 - TDD tests written first."""

from __future__ import annotations

from typing import Any

import pytest

from src.preprocessor.base_preprocessor_v2 import BasePreprocessorV2
from src.preprocessor.base_preprocessor_v2 import FeatureContext
from src.preprocessor.base_preprocessor_v2 import OperationType


class MockPreprocessor(BasePreprocessorV2):
    """Mock preprocessor for testing registry."""

    def __init__(self, name: str = "mock") -> None:
        super().__init__(name=name, operation_type=OperationType.TRANSFORM)

    @property
    def input_features(self) -> list[str]:
        return []

    @property
    def output_features(self) -> list[str]:
        return []

    def transform(self, context: FeatureContext) -> None:
        pass


class TestPreprocessorsRegistryV2Basic:
    """Basic tests for PreprocessorsRegistryV2."""

    def test_register_and_get_preprocessor(self) -> None:
        """Should register and retrieve a preprocessor by name."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()
        prep = MockPreprocessor("test_prep")

        registry.register("my_prep", prep)
        retrieved = registry.get("my_prep")

        assert retrieved is prep

    def test_get_returns_none_for_unknown(self) -> None:
        """get() should return None for unknown name."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        result = registry.get("nonexistent")

        assert result is None


class TestPreprocessorsRegistryV2Factories:
    """Tests for factory registration."""

    def test_register_factory(self) -> None:
        """Should register a factory function."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        def factory(**kwargs: Any) -> MockPreprocessor:
            return MockPreprocessor(name=kwargs.get("prep_name", "default"))

        registry.register_factory("mock", factory)
        prep = registry.get("mock", prep_name="custom")

        assert prep is not None
        assert prep.name == "custom"

    def test_factory_creates_new_instance_each_time(self) -> None:
        """Factory should create new instance on each call."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        def factory(**kwargs: Any) -> MockPreprocessor:
            return MockPreprocessor()

        registry.register_factory("mock", factory)

        prep1 = registry.get("mock")
        prep2 = registry.get("mock")

        assert prep1 is not prep2


class TestPreprocessorsRegistryV2BuiltinPreprocessors:
    """Tests for built-in preprocessor registration."""

    def test_multiply_is_auto_registered(self) -> None:
        """MultiplyPreprocessor should be auto-registered."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        prep = registry.get(
            "multiply", feature_a="a", feature_b="b", output_feature="c"
        )

        assert prep is not None
        assert prep.input_features == ["a", "b"]
        assert prep.output_features == ["c"]

    def test_remove_is_auto_registered(self) -> None:
        """RemoveFeaturePreprocessor should be auto-registered."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        prep = registry.get("remove", features_to_remove=["a", "b"])

        assert prep is not None
        assert set(prep.removes_features) == {"a", "b"}

    def test_token_length_is_auto_registered(self) -> None:
        """TokenLengthPreprocessor should be auto-registered."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        prep = registry.get("token_length", input_feature="text")

        assert prep is not None
        assert prep.input_features == ["text"]
        assert prep.output_features == ["input_length"]


class TestPreprocessorsRegistryV2Chains:
    """Tests for chain registration."""

    def test_register_and_get_chain(self) -> None:
        """Should register and retrieve a chain by name."""
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()
        chain = PreprocessorChainV2(name="test_chain")

        registry.register_chain("my_chain", chain)
        retrieved = registry.get_chain("my_chain")

        assert retrieved is chain

    def test_get_chain_returns_none_for_unknown(self) -> None:
        """get_chain() should return None for unknown name."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        result = registry.get_chain("nonexistent")

        assert result is None


class TestPreprocessorsRegistryV2CreateChainFromConfig:
    """Tests for creating chains from configuration."""

    def test_create_chain_from_config(self) -> None:
        """Should create chain from list of step configs."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        config = [
            {
                "name": "multiply",
                "params": {"feature_a": "w", "feature_b": "h", "output_feature": "p"},
            },
            {"name": "remove", "params": {"features_to_remove": ["w", "h"]}},
        ]

        chain = registry.create_chain_from_config(config)

        assert chain is not None
        assert len(chain.preprocessors) == 2
        result = chain.transform({"w": 10, "h": 20})
        assert result == {"p": 200}

    def test_create_chain_with_chain_name(self) -> None:
        """Should set chain name from config."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        config = [{"name": "token_length", "params": {"input_feature": "text"}}]

        chain = registry.create_chain_from_config(config, chain_name="my_pipeline")

        assert chain.name == "my_pipeline"

    def test_create_chain_raises_for_unknown_preprocessor(self) -> None:
        """Should raise ValueError for unknown preprocessor name."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        config = [{"name": "unknown_preprocessor", "params": {}}]

        with pytest.raises(ValueError, match="Unknown preprocessor"):
            registry.create_chain_from_config(config)

    def test_create_empty_chain_from_empty_config(self) -> None:
        """Empty config should create empty chain."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        chain = registry.create_chain_from_config([])

        assert len(chain.preprocessors) == 0


class TestPreprocessorsRegistryV2ComplexScenarios:
    """Tests for complex registry scenarios."""

    def test_full_pipeline_from_config(self) -> None:
        """Test full pipeline creation from config."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        config = [
            {
                "name": "multiply",
                "params": {
                    "feature_a": "width",
                    "feature_b": "height",
                    "output_feature": "pixel_num",
                },
            },
            {"name": "remove", "params": {"features_to_remove": ["width", "height"]}},
            {
                "name": "token_length",
                "params": {"input_feature": "prompt", "output_feature": "input_length"},
            },
            {"name": "remove", "params": {"features_to_remove": ["prompt"]}},
        ]

        chain = registry.create_chain_from_config(config, chain_name="image_pipeline")
        result = chain.transform(
            {
                "width": 100,
                "height": 200,
                "channels": 3,
                "prompt": "Hello world",
            }
        )

        assert result == {
            "channels": 3,
            "pixel_num": 20000,
            "input_length": 2,
        }

    def test_static_and_factory_precedence(self) -> None:
        """Static registration should take precedence over factory."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()
        static_prep = MockPreprocessor(name="static")

        def factory(**kwargs: Any) -> MockPreprocessor:
            return MockPreprocessor(name="from_factory")

        registry.register_factory("test", factory)
        registry.register("test", static_prep)

        retrieved = registry.get("test")

        assert retrieved is static_prep


class TestPreprocessorsRegistryV2ListAvailable:
    """Tests for listing available preprocessors."""

    def test_list_available_preprocessors(self) -> None:
        """Should list all registered preprocessor names."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()

        available = registry.list_available()

        # Should include built-in preprocessors
        assert "multiply" in available
        assert "remove" in available
        assert "token_length" in available

    def test_list_includes_custom_registered(self) -> None:
        """Should include custom registered preprocessors."""
        from src.preprocessor.registry_v2 import PreprocessorsRegistryV2

        registry = PreprocessorsRegistryV2()
        registry.register("custom", MockPreprocessor())

        available = registry.list_available()

        assert "custom" in available
