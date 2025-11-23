"""
Common infrastructure for workflow experiments.

This package provides reusable components for implementing multi-model workflow
experiments with different patterns (Text2Video, Deep Research, etc.).

Core Components:
- BaseTaskSubmitter: Abstract base for task submission threads
- BaseTaskReceiver: Abstract base for result receiving threads
- RateLimiter: Thread-safe QPS control with token bucket algorithm
- WorkflowState: Unified state tracking for all workflow types
- Utilities: JSON, logging, HTTP, and timestamp helpers
"""

from .base_classes import BaseTaskReceiver, BaseTaskSubmitter
from .data_structures import TaskStatus, WorkflowState, WorkflowType
from .rate_limiter import PoissonRateLimiter, RateLimiter
from .utils import (
    calculate_duration,
    configure_logging,
    create_retry_session,
    ensure_directory,
    estimate_token_length,
    format_duration,
    format_timestamp,
    get_timestamp,
    http_request_with_retry,
    load_json,
    parse_json_response,
    safe_divide,
    save_json,
    setup_console_handler,
    setup_file_handler,
    to_json,
)

__version__ = "0.1.0"

__all__ = [
    # Base classes
    "BaseTaskSubmitter",
    "BaseTaskReceiver",
    # Rate limiting
    "RateLimiter",
    "PoissonRateLimiter",
    # Data structures
    "WorkflowState",
    "WorkflowType",
    "TaskStatus",
    # JSON utilities
    "to_json",
    "save_json",
    "load_json",
    "parse_json_response",
    # Logging utilities
    "configure_logging",
    "setup_file_handler",
    "setup_console_handler",
    # HTTP utilities
    "create_retry_session",
    "http_request_with_retry",
    # Timestamp utilities
    "get_timestamp",
    "format_timestamp",
    "format_duration",
    "calculate_duration",
    # Miscellaneous utilities
    "estimate_token_length",
    "ensure_directory",
    "safe_divide",
]
