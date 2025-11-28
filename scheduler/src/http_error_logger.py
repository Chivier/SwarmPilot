"""
Centralized HTTP error logging utility.

Provides consistent, comprehensive logging for API request errors
across all HTTP clients in the scheduler, capturing full request/response
details while filtering sensitive data.
"""

from typing import Any, Dict, Optional, Union
import json
import httpx
from loguru import logger


# Headers that should never be logged (case-insensitive matching)
SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
    "api-key",
    "bearer",
}

# Maximum body length to log (characters)
MAX_BODY_LENGTH = 4096


def _sanitize_headers(
    headers: Union[httpx.Headers, Dict[str, str], None]
) -> Dict[str, str]:
    """
    Remove sensitive headers from a headers dict/object.

    Args:
        headers: Headers object or dict to sanitize

    Returns:
        Dict with sensitive headers replaced by "[REDACTED]"
    """
    if headers is None:
        return {}

    sanitized = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    return sanitized


def _truncate_body(body: Optional[Union[str, bytes, Dict[str, Any]]]) -> str:
    """
    Truncate body content for logging.

    Args:
        body: Request or response body

    Returns:
        Truncated string representation
    """
    if body is None:
        return "<empty>"

    # Convert to string if needed
    if isinstance(body, bytes):
        try:
            body_str = body.decode("utf-8")
        except UnicodeDecodeError:
            return f"<binary data, {len(body)} bytes>"
    elif isinstance(body, dict):
        try:
            body_str = json.dumps(body, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            body_str = str(body)
    else:
        body_str = str(body)

    if len(body_str) > MAX_BODY_LENGTH:
        return body_str[:MAX_BODY_LENGTH] + f"... [truncated, {len(body_str)} total]"

    return body_str


def log_http_error(
    error: Exception,
    *,
    request_url: Optional[str] = None,
    request_method: Optional[str] = None,
    request_headers: Optional[Union[httpx.Headers, Dict[str, str]]] = None,
    request_body: Optional[Union[str, bytes, Dict[str, Any]]] = None,
    response: Optional[httpx.Response] = None,
    context: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log comprehensive HTTP error information.

    This function provides consistent error logging for all API request failures,
    capturing full request/response details while filtering sensitive data.

    Args:
        error: The exception that occurred
        request_url: The URL that was requested (extracted from error if available)
        request_method: HTTP method (GET, POST, etc.)
        request_headers: Request headers (will be sanitized)
        request_body: Request body (will be truncated if large)
        response: httpx.Response object if available
        context: Optional context string (e.g., "predictor /predict")
        extra: Optional additional context as key-value pairs

    Example:
        try:
            response = await client.post(url, json=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log_http_error(
                e,
                request_body=data,
                context="predictor prediction request",
            )
            raise
    """
    # Extract request info from error if not provided
    # Note: httpx exceptions may raise RuntimeError when accessing .request
    # if the property has not been set (common in test mocks)
    if isinstance(error, httpx.HTTPStatusError):
        try:
            request = error.request
            if request_url is None:
                request_url = str(request.url)
            if request_method is None:
                request_method = request.method
            if request_headers is None:
                request_headers = dict(request.headers)
        except RuntimeError:
            pass  # .request property not set
        try:
            if response is None:
                response = error.response
        except RuntimeError:
            pass  # .response property not set
    elif isinstance(error, httpx.RequestError):
        try:
            request = error.request
            if request is not None:
                if request_url is None:
                    request_url = str(request.url)
                if request_method is None:
                    request_method = request.method
                if request_headers is None:
                    request_headers = dict(request.headers)
        except RuntimeError:
            pass  # .request property not set

    # Build log message components
    parts = [f"HTTP API Error [{context or 'unknown'}] - {type(error).__name__}: {error}"]

    # Request details
    if request_method or request_url:
        parts.append(f"\n  Request: {request_method or '?'} {request_url or '?'}")

    # Sanitized headers
    if request_headers:
        sanitized = _sanitize_headers(request_headers)
        parts.append(f"\n  Request Headers: {sanitized}")

    # Request body
    if request_body is not None:
        parts.append(f"\n  Request Body: {_truncate_body(request_body)}")

    # Response details
    if response is not None:
        parts.append(f"\n  Response Status: {response.status_code}")
        parts.append(f"\n  Response Headers: {_sanitize_headers(dict(response.headers))}")

        try:
            response_body = response.text
        except Exception:
            response_body = "<unable to read response body>"
        parts.append(f"\n  Response Body: {_truncate_body(response_body)}")

    # Extra context
    if extra:
        parts.append(f"\n  Extra: {extra}")

    # Log at ERROR level
    log_message = "".join(parts)
    logger.error(log_message)
