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
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger


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
# Logging Configuration (loguru)
# ============================================================================

def configure_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
):
    """
    Configure loguru logging with consistent formatting.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG", "WARNING")
        log_file: Optional path to log file
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    if format_string is None:
        format_string = (
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <6}</level> | '
            '<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - '
            '<level>{message}</level>'
        )

    # Add console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True
    )

    # Add file handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler without colors
        file_format = (
            '{time:YYYY-MM-DD HH:mm:ss} | '
            '{level: <8} | '
            '[{thread.name: <15}] '
            '{name}:{function}:{line} - '
            '{message}'
        )

        logger.add(
            str(log_file),
            format=file_format,
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="1 week"  # Keep logs for 1 week
        )

    return logger


def setup_file_handler(
    log_file: Union[str, Path],
    level: str = "DEBUG"
):
    """
    Add file handler to loguru logger.

    Args:
        log_file: Path to log file
        level: Log level for file handler
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_format = (
        '{time:YYYY-MM-DD HH:mm:ss} | '
        '{level: <8} | '
        '[{thread.name: <15}] '
        '{name}:{function}:{line} - '
        '{message}'
    )

    logger.add(
        str(log_file),
        format=file_format,
        level=level,
        rotation="10 MB",
        retention="1 week"
    )


def setup_console_handler(
    level: str = "INFO"
):
    """
    Add console handler to loguru logger.

    Args:
        level: Log level for console handler
    """
    console_format = (
        '<green>{time:HH:mm:ss}</green> | '
        '<level>{level: <8}</level> | '
        '<cyan>{name}</cyan> - '
        '<level>{message}</level>'
    )

    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True
    )


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
    custom_logger: Optional[Any] = None,
    **kwargs
) -> requests.Response:
    """
    Make HTTP request with retry logic and comprehensive error handling.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        custom_logger: Optional custom logger (defaults to loguru logger)
        **kwargs: Additional arguments for requests

    Returns:
        HTTP response object

    Raises:
        requests.RequestException: If request fails after all retries
    """
    log = custom_logger or logger
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
        log.error(f"Request timeout after {timeout}s: {url}")
        raise

    except requests.exceptions.ConnectionError as e:
        log.error(f"Connection error: {url} - {e}")
        raise

    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error {e.response.status_code}: {url}")
        raise

    except requests.exceptions.RequestException as e:
        log.error(f"Request failed: {url} - {e}")
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
# Result Extraction Utilities
# ============================================================================

def extract_task_result(data: Dict[str, Any], field: str = "output") -> str:
    """
    Extract task result from WebSocket message data.

    Supports multiple result formats from different model services:
    - Simulation mode (sleep_model): {"output": "..."}
    - Real mode LLM: {"result": {"output": "..."}} or {"output": "..."}
    - Nested format: {"result": {"result": {"output": "..."}}}

    Args:
        data: WebSocket message data containing "result" field
        field: Field name to extract (default: "output")

    Returns:
        Extracted result string, or empty string if not found
    """
    result = data.get("result", {})

    if isinstance(result, dict):
        # Try nested format first: result.result.{field}
        if "result" in result and isinstance(result.get("result"), dict):
            nested = result.get("result", {})
            if field in nested:
                return str(nested.get(field, ""))
            # Fallback: return entire nested result as string
            return str(nested) if nested else ""

        # Then try direct format: result.{field}
        if field in result:
            return str(result.get(field, ""))

        # Fallback: return entire result as string (useful for complex types)
        return str(result) if result else ""

    # If result is not a dict, convert to string
    return str(result) if result else ""


def extract_workflow_id_from_task_id(task_id: str) -> Optional[str]:
    """
    Extract workflow ID from task ID.

    Task ID formats:
    - Type1: task-A1-{strategy}-workflow-{num}
    - Type1: task-A2-{strategy}-workflow-{num}
    - Type1: task-B{N}-{strategy}-workflow-{num}
    - Type2: task-A-{strategy}-workflow-{num}
    - Type2: task-B1-{strategy}-workflow-{num}-{b_index}
    - Type2: task-B2-{strategy}-workflow-{num}-{b_index}
    - Type2: task-merge-{strategy}-workflow-{num}

    Args:
        task_id: Task ID string

    Returns:
        Workflow ID in format "workflow-{num}", or None if not found
    """
    if not task_id:
        return None

    parts = task_id.split("-")
    if "workflow" in parts:
        idx = parts.index("workflow")
        if idx + 1 < len(parts):
            return f"workflow-{parts[idx + 1]}"

    return None


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


# ============================================================================
# Scheduler Strategy Management
# ============================================================================

def clear_scheduler_tasks(scheduler_url: str, custom_logger: Optional[Any] = None) -> bool:
    """
    Clear all tasks from a scheduler.

    Args:
        scheduler_url: Scheduler endpoint URL
        custom_logger: Optional custom logger (defaults to loguru logger)

    Returns:
        True if successful, False otherwise
    """
    log = custom_logger or logger
    log.info(f"Clearing tasks from scheduler: {scheduler_url}")

    try:
        response = requests.post(
            f"{scheduler_url}/task/clear", timeout=300 # clear tasks may take a while
        )
        response.raise_for_status()
        result = response.json()

        log.info(f"Cleared {result.get('cleared_count', 0)} tasks from scheduler")

        return True
    except Exception as e:
        log.error(f"Failed to clear tasks from scheduler {scheduler_url}: {e}")
        return False


def set_scheduler_strategy(
    scheduler_url: str,
    strategy_name: str,
    target_quantile: Optional[float] = None,
    quantiles: Optional[list] = None,
    custom_logger: Optional[Any] = None
) -> bool:
    """
    Set scheduling strategy for a scheduler.

    Args:
        scheduler_url: Scheduler endpoint URL
        strategy_name: Strategy name (min_time, probabilistic, round_robin, random, po2, serverless)
        target_quantile: Target quantile for probabilistic strategy (default: 0.9)
        quantiles: Custom quantiles for probabilistic strategy (default: [0.1, 0.25, 0.5, 0.75, 0.99])
        custom_logger: Optional custom logger (defaults to loguru logger)

    Returns:
        True if successful, False otherwise
    """
    # Default quantiles for probabilistic strategy
    if quantiles is None and strategy_name == "probabilistic":
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.99]

    # Build request payload
    payload = {"strategy_name": strategy_name}
    if target_quantile is not None:
        payload["target_quantile"] = target_quantile
    if quantiles is not None:
        payload["quantiles"] = quantiles

    log = custom_logger or logger
    log.info(f"Setting strategy '{strategy_name}' on scheduler: {scheduler_url}")
    if quantiles:
        log.info(f"  Quantiles: {quantiles}")
    if target_quantile:
        log.info(f"  Target quantile: {target_quantile}")

    try:
        response = requests.post(
            f"{scheduler_url}/strategy/set",
            json=payload,
            timeout=300 # set strategy may take a while
        )
        response.raise_for_status()
        result = response.json()

        log.info(f"Successfully set strategy '{strategy_name}' on scheduler")
        if result.get("previous_strategy"):
            log.info(f"  Previous strategy: {result['previous_strategy']}")

        return True
    except Exception as e:
        log.error(f"Failed to set strategy on scheduler {scheduler_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            log.error(f"  Response: {e.response.text}")
        return False


def setup_scheduler_strategies(
    strategy_name: str,
    scheduler_a_url: str,
    scheduler_b_url: Optional[str] = None,
    target_quantile: Optional[float] = None,
    quantiles: Optional[list] = None,
    custom_logger: Optional[Any] = None
) -> Dict[str, bool]:
    """
    Setup scheduling strategy on schedulers.

    This function sets the strategy on all schedulers.
    NOTE: Task queue clearing must be done explicitly before calling this function.

    Args:
        strategy_name: Strategy name to set
        scheduler_a_url: Scheduler A endpoint URL
        scheduler_b_url: Optional Scheduler B endpoint URL
        target_quantile: Target quantile for probabilistic strategy
        quantiles: Custom quantiles for probabilistic strategy
        custom_logger: Optional custom logger (defaults to loguru logger)

    Returns:
        Dict mapping strategy names to success status
    """
    results = {}

    log = custom_logger or logger
    log.info(f"Setting strategy: {strategy_name}")

    # Step 1: Set strategy on Scheduler A
    if not set_scheduler_strategy(
        scheduler_a_url,
        strategy_name,
        target_quantile,
        quantiles,
        log
    ):
        results[strategy_name] = False
        return results

    # Step 2: Set strategy on Scheduler B (if provided)
    if scheduler_b_url and not set_scheduler_strategy(
        scheduler_b_url,
        strategy_name,
        target_quantile,
        quantiles,
        log
    ):
        results[strategy_name] = False
        return results

    results[strategy_name] = True

    return results
