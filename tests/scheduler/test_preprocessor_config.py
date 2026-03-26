"""Unit tests for PreprocessorChainBuilder.

Tests config loading, rule matching, chain building, strict validation,
and edge cases.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from swarmpilot.scheduler.clients.preprocessor_config import (
    PreprocessorChainBuilder,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_registry_v1():
    """Create a mock V1 PreprocessorsRegistry."""
    registry = MagicMock()
    mock_preprocessor = MagicMock()
    mock_preprocessor.__call__ = MagicMock(return_value=({"output_length": 42}, True))
    registry.get_preprocessor.return_value = mock_preprocessor
    return registry


@pytest.fixture
def sample_config():
    """Sample preprocessor config with one rule."""
    return {
        "rules": [
            {
                "model_id_contains": ["llm_service", "model"],
                "chain": [
                    {
                        "type": "v1_adapter",
                        "name": "semantic",
                        "input_feature": "sentence",
                    }
                ],
            }
        ]
    }


@pytest.fixture
def multi_rule_config():
    """Config with multiple rules for testing precedence."""
    return {
        "rules": [
            {
                "model_id_contains": ["llm_service", "model"],
                "chain": [
                    {
                        "type": "v1_adapter",
                        "name": "semantic",
                        "input_feature": "sentence",
                    }
                ],
            },
            {
                "model_id_contains": ["vision"],
                "chain": [
                    {
                        "type": "v1_adapter",
                        "name": "image_embed",
                        "input_feature": "image",
                    }
                ],
            },
        ]
    }


def _write_config(config: dict) -> str:
    """Write config to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f)
    return path


# ============================================================================
# Config Loading Tests
# ============================================================================


class TestConfigLoading:
    """Tests for loading config from JSON files."""

    def test_empty_config_file_path(self, mock_registry_v1):
        """Empty config_file should produce no rules."""
        builder = PreprocessorChainBuilder(
            config_file="",
            registry_v1=mock_registry_v1,
        )
        assert not builder.has_rules
        assert builder.get_chain("any_model_id") is None

    def test_nonexistent_file(self, mock_registry_v1):
        """Missing config file should log warning, produce no rules."""
        builder = PreprocessorChainBuilder(
            config_file="/nonexistent/path.json",
            registry_v1=mock_registry_v1,
        )
        assert not builder.has_rules

    def test_invalid_json(self, mock_registry_v1, tmp_path):
        """Invalid JSON should log error, produce no rules."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{")

        builder = PreprocessorChainBuilder(
            config_file=str(bad_file),
            registry_v1=mock_registry_v1,
        )
        assert not builder.has_rules

    def test_valid_config_loads_rules(self, mock_registry_v1, sample_config):
        """Valid config should load rules correctly."""
        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            assert builder.has_rules
        finally:
            os.unlink(path)


# ============================================================================
# Rule Matching Tests
# ============================================================================


class TestRuleMatching:
    """Tests for model_id pattern matching."""

    def test_matching_model_id(self, mock_registry_v1, sample_config):
        """Model ID containing all patterns should match."""
        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("llm_service__model__v1")
            assert chain is not None
        finally:
            os.unlink(path)

    def test_non_matching_model_id(self, mock_registry_v1, sample_config):
        """Model ID missing a pattern should not match."""
        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("other_service")
            assert chain is None
        finally:
            os.unlink(path)

    def test_partial_match_fails(self, mock_registry_v1, sample_config):
        """Model ID with only one of two patterns should not match."""
        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("llm_service__other")
            assert chain is None
        finally:
            os.unlink(path)

    def test_first_matching_rule_wins(self, mock_registry_v1, multi_rule_config):
        """First matching rule should be used."""
        path = _write_config(multi_rule_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            # This matches first rule
            chain = builder.get_chain("llm_service__model__v1")
            assert chain is not None
            assert "llm_service" in chain.name
        finally:
            os.unlink(path)

    def test_second_rule_match(self, mock_registry_v1, multi_rule_config):
        """Second rule should match when first doesn't."""
        path = _write_config(multi_rule_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("vision_service")
            assert chain is not None
            assert "vision" in chain.name
        finally:
            os.unlink(path)


# ============================================================================
# Strict Validation Tests
# ============================================================================


class TestStrictValidation:
    """Tests for strict validation of preprocessor availability."""

    def test_strict_validation_passes(self, mock_registry_v1, sample_config):
        """No error when preprocessor is available."""
        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
                strict=True,
            )
            assert builder.has_rules
        finally:
            os.unlink(path)

    def test_strict_validation_fails_on_missing_model(self, sample_config):
        """RuntimeError when strict=True and preprocessor unavailable."""
        registry = MagicMock()
        registry.get_preprocessor.side_effect = RuntimeError("Model file not found")

        path = _write_config(sample_config)
        try:
            with pytest.raises(RuntimeError, match="unavailable"):
                PreprocessorChainBuilder(
                    config_file=path,
                    registry_v1=registry,
                    strict=True,
                )
        finally:
            os.unlink(path)

    def test_non_strict_logs_warning_on_missing(self, sample_config):
        """Non-strict mode should not raise on missing preprocessor."""
        registry = MagicMock()
        registry.get_preprocessor.side_effect = RuntimeError("Model file not found")

        path = _write_config(sample_config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=registry,
                strict=False,
            )
            assert builder.has_rules
        finally:
            os.unlink(path)


# ============================================================================
# Empty / Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_rules_list(self, mock_registry_v1):
        """Config with empty rules list."""
        path = _write_config({"rules": []})
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            assert not builder.has_rules
        finally:
            os.unlink(path)

    def test_rule_with_empty_chain(self, mock_registry_v1):
        """Rule with empty chain should return an empty chain."""
        config = {
            "rules": [
                {
                    "model_id_contains": ["test"],
                    "chain": [],
                }
            ]
        }
        path = _write_config(config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("test_model")
            assert chain is not None
        finally:
            os.unlink(path)

    def test_unknown_step_type_logged(self, mock_registry_v1):
        """Unknown step type should be logged and skipped."""
        config = {
            "rules": [
                {
                    "model_id_contains": ["test"],
                    "chain": [
                        {
                            "type": "unknown_type",
                            "name": "something",
                        }
                    ],
                }
            ]
        }
        path = _write_config(config)
        try:
            builder = PreprocessorChainBuilder(
                config_file=path,
                registry_v1=mock_registry_v1,
            )
            chain = builder.get_chain("test_model")
            assert chain is not None
        finally:
            os.unlink(path)
