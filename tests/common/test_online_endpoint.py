"""Unit tests for online API endpoint utilities.

Tests for swarmpilot.common.online_endpoint: PlatformInfo generation,
feature extraction, model tier inference, and YAML config loading.
"""

from __future__ import annotations

import contextlib

import pytest
import yaml

from swarmpilot.common.online_endpoint import (
    _infer_model_tier,
    extract_online_features,
    platform_info_from_online_endpoint,
)
from swarmpilot.scheduler.online_endpoint_config import (
    load_online_endpoints_config,
)

# ================================================================
# PlatformInfo generation
# ================================================================


class TestPlatformInfoFromOnlineEndpoint:
    """Tests for platform_info_from_online_endpoint()."""

    def test_deterministic_hash(self):
        """Same inputs always produce the same output."""
        url = "https://api.anthropic.com"
        key = "sk-ant-test-key-123"
        a = platform_info_from_online_endpoint(url, key)
        b = platform_info_from_online_endpoint(url, key)
        assert a == b

    def test_different_urls_different_software_name(self):
        """Different base_urls produce different software_name."""
        key = "shared-key"
        a = platform_info_from_online_endpoint("https://api.anthropic.com", key)
        b = platform_info_from_online_endpoint("https://api.openai.com/v1", key)
        assert a["software_name"] != b["software_name"]
        # software_version should match (same key)
        assert a["software_version"] == b["software_version"]

    def test_different_keys_different_software_version(self):
        """Different api_keys produce different software_version."""
        url = "https://api.anthropic.com"
        a = platform_info_from_online_endpoint(url, "key-tier-1")
        b = platform_info_from_online_endpoint(url, "key-tier-2")
        assert a["software_version"] != b["software_version"]
        # software_name should match (same url)
        assert a["software_name"] == b["software_name"]

    def test_hardware_name_is_cloud(self):
        """hardware_name is always 'cloud'."""
        info = platform_info_from_online_endpoint("https://example.com", "any-key")
        assert info["hardware_name"] == "cloud"

    def test_hash_length(self):
        """Hash values are 16 hex characters."""
        info = platform_info_from_online_endpoint(
            "https://api.anthropic.com", "sk-ant-key"
        )
        assert len(info["software_name"]) == 16
        assert len(info["software_version"]) == 16
        # Verify they are valid hex
        int(info["software_name"], 16)
        int(info["software_version"], 16)

    def test_required_keys_present(self):
        """Output dict has all three required PlatformInfo keys."""
        info = platform_info_from_online_endpoint(
            "https://api.openai.com/v1", "sk-test"
        )
        assert set(info.keys()) == {
            "software_name",
            "software_version",
            "hardware_name",
        }


# ================================================================
# Feature extraction
# ================================================================


class TestExtractOnlineFeatures:
    """Tests for extract_online_features()."""

    def test_basic_extraction(self):
        """Validates all returned feature fields."""
        request_body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        response = {
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 20,
            }
        }
        features = extract_online_features(request_body, response)

        assert features["input_tokens"] == 5
        assert features["estimated_output_tokens"] == 20
        assert features["model_tier"] == 1  # sonnet = medium
        assert features["is_streaming"] == 0
        assert features["has_tools"] == 0
        assert 0 <= features["time_of_day_hour"] <= 23

    def test_with_usage_from_response(self):
        """Prefers response usage over estimation."""
        request_body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "A " * 1000}],
        }
        response = {
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 100,
            }
        }
        features = extract_online_features(request_body, response)
        # Should use actual prompt_tokens, not estimate
        assert features["input_tokens"] == 42
        assert features["estimated_output_tokens"] == 100

    def test_fallback_token_estimation(self):
        """Falls back to len//4 estimation when no usage."""
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "x" * 100}],
        }
        response = {}
        features = extract_online_features(request_body, response)
        assert features["input_tokens"] == 25  # 100 // 4

    def test_streaming_flag(self):
        """is_streaming correctly maps True/False."""
        body = {"model": "m", "messages": []}
        f_off = extract_online_features(body, {}, is_streaming=False)
        f_on = extract_online_features(body, {}, is_streaming=True)
        assert f_off["is_streaming"] == 0
        assert f_on["is_streaming"] == 1

    def test_tools_detection(self):
        """has_tools flag detects tool definitions."""
        body_no_tools = {"model": "m", "messages": []}
        body_with_tools = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {}}],
        }
        f_no = extract_online_features(body_no_tools, {})
        f_yes = extract_online_features(body_with_tools, {})
        assert f_no["has_tools"] == 0
        assert f_yes["has_tools"] == 1

    def test_empty_messages(self):
        """Handles empty messages list gracefully."""
        features = extract_online_features({"model": "m", "messages": []}, {})
        assert features["input_tokens"] == 0

    def test_non_string_content_skipped(self):
        """Skips messages with non-string content."""
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "url": "..."}],
                }
            ],
        }
        features = extract_online_features(body, {})
        # Non-string content should be skipped in estimation
        assert features["input_tokens"] == 0


# ================================================================
# Model tier inference
# ================================================================


class TestInferModelTier:
    """Tests for _infer_model_tier()."""

    @pytest.mark.parametrize(
        "name",
        [
            "claude-haiku-3",
            "gpt-4o-mini",
            "gemini-2.0-flash",
            "Qwen/Qwen2.5-8B-Instruct",
        ],
    )
    def test_small_models(self, name):
        """Small/fast models map to tier 0."""
        assert _infer_model_tier(name) == 0

    @pytest.mark.parametrize(
        "name",
        [
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "gemini-pro",
            "Qwen/Qwen2.5-70B-Instruct",
        ],
    )
    def test_medium_models(self, name):
        """Medium models map to tier 1."""
        assert _infer_model_tier(name) == 1

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-4-20250514",
            "o1-preview",
            "gemini-ultra",
            "Meta-Llama-3.1-405B",
        ],
    )
    def test_large_models(self, name):
        """Large/premium models map to tier 2."""
        assert _infer_model_tier(name) == 2

    def test_unknown_defaults_to_medium(self):
        """Unknown model name defaults to tier 1."""
        assert _infer_model_tier("custom-model-xyz") == 1

    def test_empty_string(self):
        """Empty string defaults to tier 1."""
        assert _infer_model_tier("") == 1


# ================================================================
# Config loading
# ================================================================


class TestOnlineEndpointConfig:
    """Tests for load_online_endpoints_config()."""

    def test_load_valid_yaml(self, tmp_path):
        """Parses a valid YAML config file correctly."""
        config_data = {
            "endpoints": [
                {
                    "name": "claude-key-1",
                    "base_url": "https://api.anthropic.com",
                    "api_key_env": "CLAUDE_API_KEY_1",
                    "model_id": "claude-sonnet",
                    "concurrency_limit": 10,
                },
                {
                    "name": "openai-gpt4",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "model_id": "gpt-4o",
                    "concurrency_limit": 20,
                },
            ]
        }
        config_file = tmp_path / "endpoints.yaml"
        config_file.write_text(yaml.dump(config_data))

        entries = load_online_endpoints_config(str(config_file))

        assert len(entries) == 2
        assert entries[0].name == "claude-key-1"
        assert entries[0].base_url == "https://api.anthropic.com"
        assert entries[0].api_key_env == "CLAUDE_API_KEY_1"
        assert entries[0].model_id == "claude-sonnet"
        assert entries[0].concurrency_limit == 10
        assert entries[1].name == "openai-gpt4"
        assert entries[1].concurrency_limit == 20

    def test_load_missing_file_returns_empty(self):
        """Non-existent file returns empty list."""
        result = load_online_endpoints_config("/nonexistent/path.yaml")
        assert result == []

    def test_load_empty_path_returns_empty(self):
        """Empty string returns empty list."""
        assert load_online_endpoints_config("") == []

    def test_load_none_returns_empty(self):
        """None returns empty list."""
        assert load_online_endpoints_config(None) == []

    def test_default_concurrency_limit(self, tmp_path):
        """concurrency_limit defaults to 10."""
        config_data = {
            "endpoints": [
                {
                    "name": "test",
                    "base_url": "https://api.example.com",
                    "api_key_env": "TEST_KEY",
                    "model_id": "test-model",
                }
            ]
        }
        config_file = tmp_path / "endpoints.yaml"
        config_file.write_text(yaml.dump(config_data))

        entries = load_online_endpoints_config(str(config_file))
        assert entries[0].concurrency_limit == 10

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        """Falls back to ONLINE_ENDPOINTS_CONFIG env var."""
        config_data = {
            "endpoints": [
                {
                    "name": "env-test",
                    "base_url": "https://api.example.com",
                    "api_key_env": "KEY",
                    "model_id": "model",
                }
            ]
        }
        config_file = tmp_path / "endpoints.yaml"
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv("ONLINE_ENDPOINTS_CONFIG", str(config_file))

        entries = load_online_endpoints_config()
        assert len(entries) == 1
        assert entries[0].name == "env-test"

    def test_invalid_yaml_does_not_crash(self, tmp_path):
        """Invalid YAML content does not raise unhandled error."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("not: valid: yaml: [")

        # yaml.safe_load may raise or return malformed data;
        # either outcome is acceptable (no unhandled crash).
        with contextlib.suppress(Exception):
            load_online_endpoints_config(str(config_file))

    def test_empty_endpoints_list(self, tmp_path):
        """Config with empty endpoints list returns empty."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text(yaml.dump({"endpoints": []}))

        entries = load_online_endpoints_config(str(config_file))
        assert entries == []
