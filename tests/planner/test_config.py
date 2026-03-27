"""Tests for configuration validation."""

import os
from unittest.mock import patch

import pytest


class TestPlannerConfigValidation:
    """Tests for PlannerConfig.validate() method."""

    def test_valid_config(self):
        """Test that valid configuration passes validation."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            # Should not raise
            config.validate()

    def test_instance_timeout_zero_fails(self):
        """Test that INSTANCE_TIMEOUT=0 raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "0",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="INSTANCE_TIMEOUT must be positive"
            ):
                config.validate()

    def test_instance_timeout_negative_fails(self):
        """Test that negative INSTANCE_TIMEOUT raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "-5",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="INSTANCE_TIMEOUT must be positive"
            ):
                config.validate()

    def test_instance_max_retries_negative_fails(self):
        """Test that negative INSTANCE_MAX_RETRIES raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "-1",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="INSTANCE_MAX_RETRIES must be non-negative"
            ):
                config.validate()

    def test_instance_retry_delay_negative_fails(self):
        """Test that negative INSTANCE_RETRY_DELAY raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "-0.5",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="INSTANCE_RETRY_DELAY must be non-negative"
            ):
                config.validate()

    def test_planner_port_zero_fails(self):
        """Test that PLANNER_PORT=0 raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "0",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="PLANNER_PORT must be 1-65535"
            ):
                config.validate()

    def test_planner_port_too_high_fails(self):
        """Test that PLANNER_PORT=65536 raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "65536",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="PLANNER_PORT must be 1-65535"
            ):
                config.validate()

    def test_planner_port_valid_boundaries(self):
        """Test that valid port boundaries (1, 65535) pass validation."""
        # Test port 1
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "1",
                "AUTO_OPTIMIZE_INTERVAL": "60.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            config.validate()  # Should not raise

        # Test port 65535
        with patch.dict(os.environ, {"PLANNER_PORT": "65535"}, clear=False):
            config = PlannerConfig()
            config.validate()  # Should not raise

    def test_auto_optimize_interval_zero_fails(self):
        """Test that AUTO_OPTIMIZE_INTERVAL=0 raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="AUTO_OPTIMIZE_INTERVAL must be positive"
            ):
                config.validate()

    def test_auto_optimize_interval_negative_fails(self):
        """Test that negative AUTO_OPTIMIZE_INTERVAL raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "INSTANCE_TIMEOUT": "30",
                "INSTANCE_MAX_RETRIES": "3",
                "INSTANCE_RETRY_DELAY": "1.0",
                "PLANNER_PORT": "8000",
                "AUTO_OPTIMIZE_INTERVAL": "-10.0",
            },
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            with pytest.raises(
                ValueError, match="AUTO_OPTIMIZE_INTERVAL must be positive"
            ):
                config.validate()


class TestPlannerConfigGetSchedulerUrl:
    """Tests for PlannerConfig.get_scheduler_url() method."""

    def test_get_scheduler_url_with_override(self):
        """Test that override takes precedence."""
        with patch.dict(
            os.environ,
            {"SCHEDULER_URL": "http://env-scheduler:8100"},
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            result = config.get_scheduler_url(override="http://override:9000")
            assert result == "http://override:9000"

    def test_get_scheduler_url_from_env(self):
        """Test that env var is used when no override."""
        with patch.dict(
            os.environ,
            {"SCHEDULER_URL": "http://env-scheduler:8100"},
            clear=False,
        ):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            result = config.get_scheduler_url()
            assert result == "http://env-scheduler:8100"

    def test_get_scheduler_url_none_when_not_set(self):
        """Test that None is returned when no env var or override."""
        env_copy = os.environ.copy()
        if "SCHEDULER_URL" in env_copy:
            del env_copy["SCHEDULER_URL"]
        with patch.dict(os.environ, env_copy, clear=True):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            config.scheduler_url = None  # Ensure it's None
            result = config.get_scheduler_url()
            assert result is None


class TestPlannerConfigInit:
    """Tests for PlannerConfig initialization."""

    def test_auto_optimize_enabled_true_values(self):
        """Test that various true values enable auto-optimize."""
        for value in ["true", "TRUE", "True", "1", "yes", "YES"]:
            with patch.dict(
                os.environ, {"AUTO_OPTIMIZE_ENABLED": value}, clear=False
            ):
                from swarmpilot.planner.config import PlannerConfig

                config = PlannerConfig()
                assert config.auto_optimize_enabled is True, (
                    f"Failed for value: {value}"
                )

    def test_auto_optimize_enabled_false_values(self):
        """Test that other values disable auto-optimize."""
        for value in ["false", "FALSE", "0", "no", "anything"]:
            with patch.dict(
                os.environ, {"AUTO_OPTIMIZE_ENABLED": value}, clear=False
            ):
                from swarmpilot.planner.config import PlannerConfig

                config = PlannerConfig()
                assert config.auto_optimize_enabled is False, (
                    f"Failed for value: {value}"
                )

    def test_default_values(self):
        """Test default configuration values."""
        env_minimal = {
            k: v
            for k, v in os.environ.items()
            if k
            not in [
                "INSTANCE_TIMEOUT",
                "INSTANCE_MAX_RETRIES",
                "INSTANCE_RETRY_DELAY",
                "PLANNER_PORT",
                "AUTO_OPTIMIZE_ENABLED",
                "AUTO_OPTIMIZE_INTERVAL",
            ]
        }
        with patch.dict(os.environ, env_minimal, clear=True):
            from swarmpilot.planner.config import PlannerConfig

            config = PlannerConfig()
            assert config.instance_timeout == 30
            assert config.instance_max_retries == 3
            assert config.instance_retry_delay == 1.0
            assert config.planner_port == 8000
            assert config.auto_optimize_enabled is False
            assert config.auto_optimize_interval == 60.0
