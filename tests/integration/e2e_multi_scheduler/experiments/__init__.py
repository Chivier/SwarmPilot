"""Experiment configurations for multi-scheduler E2E tests.

This package contains pre-configured experiment setups for different
workload types: sleep models and mock LLMs.
"""

from .sleep_model_config import create_sleep_model_config
from .mock_llm_config import create_mock_llm_config

__all__ = [
    "create_sleep_model_config",
    "create_mock_llm_config",
]
