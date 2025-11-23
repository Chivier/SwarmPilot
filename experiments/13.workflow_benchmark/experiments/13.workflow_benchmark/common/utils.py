"""
Common utility functions for workflow experiments.

This module provides reusable utilities for:
- JSON serialization with custom type handlers
- Logging configuration with consistent formatting
- HTTP requests with retry logic
- Timestamp and duration calculations
"""

import datetime
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# JSON Serialization Utilities
# ============================================================================

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    JSON encoder with support for custom types.

    Handles:
    - datetime objects → ISO format strings
    - dataclasses → dictionaries
    - Enums → values
    - Path objects → strings
    """

    def default(self, obj):
        """Encode custom types to JSON-serializable forms."""
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, **kwargs) -> str:
    """
    Serialize object to JSON string with custom encoder.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    kwargs.setdefault('cls', EnhancedJSONEncoder)
    kwargs.setdefault('indent', 2)
    return json.dumps(obj, **kwargs)


def save_json(obj: Any, filepath: Union[str, Path], **kwargs):
    """
    Save object to JSON file.

    Args:
        obj: Object to save
        filepath: Path to output file
        **kwargs: Additional arguments for json.dumps
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(obj, f, cls=EnhancedJSONEncoder, indent=2, **kwargs)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load object from JSON file.

    Args:
        filepath: Path to input file

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_json_response(response: requests.Response) -> Dict[str, Any]:
    """
    Parse JSON response with error handling.

    Args:
        response: HTTP response object

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If response is not valid JSON
    """
    try:
        return response.json()
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}\nContent: {response.text[:200]}")


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with consistent formatting.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file
        format_string: Optional custom format string

    Returns:
        Root logger instance
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(levelname)-8s - '
            '[%(threadName)-15s] %(name)-25s - %(message)s'
        )

    # Create formatter
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def setup_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    level: int = logging.DEBUG
):
    """
    Add file handler to existing logger.

    Args:
        logger: Logger instance
        log_file: Path to log file
        level: Log level for file handler
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - [%(threadName)-15s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def setup_console_handler(
    logger: logging.Logger,
    level: int = logging.INFO
):
    """
    Add console handler to existing logger.

    Args:
        logger: Logger instance
        level: Log level for console handler
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# ============================================================================
# HTTP Request Utilities
# ============================================================================

def create_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple = (500, 502, 503, 504),
    session: Optional[requests.Session] = None
) -> requests.Session:
    """
    Create requests session with retry logic.

    Args:
        retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        status_forcelist: HTTP status codes to retry on
        session: Optional existing session to configure

    Returns:
        Configured session with retry adapter
    """
    session = session or requests.Session()

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def http_request_with_retry(
    method: str,
    url: str,
    max_retries: int = 3,
    timeout: float = 10.0,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> requests.Response:
    """
    Make HTTP request with retry logic and comprehensive error handling.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        logger: Optional logger for error messages
        **kwargs: Additional arguments for requests

    Returns:
        HTTP response object

    Raises:
        requests.RequestException: If request fails after all retries
    """
    logger = logger or logging.getLogger(__name__)
    session = create_retry_session(retries=max_retries)

    try:
        response = session.request(
            method=method,
            url=url,
            timeout=timeout,
            **kwargs
        )
        response.raise_for_status()
        return response

    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout after {timeout}s: {url}")
        raise

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {url} - {e}")
        raise

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code}: {url}")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {url} - {e}")
        raise

    finally:
        session.close()


# ============================================================================
# Timestamp and Duration Utilities
# ============================================================================

def get_timestamp() -> float:
    """
    Get current timestamp in seconds.

    Returns:
        Current time as float
    """
    return time.time()


def format_timestamp(timestamp: float, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to human-readable string.

    Args:
        timestamp: Unix timestamp
        format_str: strftime format string

    Returns:
        Formatted timestamp string
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime(format_str)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration (e.g., "1h 23m 45s", "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def calculate_duration(start_time: float, end_time: Optional[float] = None) -> float:
    """
    Calculate duration between timestamps.

    Args:
        start_time: Start timestamp
        end_time: End timestamp (defaults to current time)

    Returns:
        Duration in seconds
    """
    end_time = end_time or get_timestamp()
    return end_time - start_time


# ============================================================================
# Miscellaneous Utilities
# ============================================================================

def estimate_token_length(text: Optional[str]) -> int:
    """
    Estimate token length from text.

    Uses a simple heuristic: approximately 4 characters per token for English text.

    Args:
        text: Input text string

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple heuristic: ~4 characters per token
    return max(1, len(text) // 4)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero

    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default
