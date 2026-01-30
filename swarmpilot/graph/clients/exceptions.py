"""
Custom exception hierarchy for the Scheduler Client.

Provides specific exception types for different error scenarios when communicating
with the scheduler service.
"""

from typing import Optional


class SchedulerClientError(Exception):
    """Base exception for all scheduler client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            status_code: HTTP status code if applicable
            response_body: Raw response body from the server if available
        """
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with additional details."""
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"(status={self.status_code})")
        if self.response_body:
            parts.append(f"Response: {self.response_body}")
        return " ".join(parts)


class SchedulerNotFoundError(SchedulerClientError):
    """Raised when a requested resource is not found (404 errors).

    Examples:
        - Task ID does not exist
        - Instance ID does not exist
    """

    pass


class SchedulerValidationError(SchedulerClientError):
    """Raised when request validation fails (400 errors).

    Examples:
        - Duplicate task/instance ID
        - Missing required fields
        - Invalid state transitions (e.g., switching strategy while tasks are running)
        - Cannot remove instance that is not in DRAINING state
    """

    pass


class SchedulerServiceError(SchedulerClientError):
    """Raised when the scheduler service is unavailable or encounters an error (503 errors).

    Examples:
        - Predictor service is unavailable
        - Service is unhealthy
        - Internal server error (500/502)
    """

    pass


class SchedulerConnectionError(SchedulerClientError):
    """Raised when network connection to the scheduler fails.

    Examples:
        - Cannot connect to scheduler host
        - Connection timeout
        - Network unreachable
    """

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        """Initialize the connection error.

        Args:
            message: Error message describing the connection failure
            original_exception: Original exception that caused the connection error
        """
        self.original_exception = original_exception
        super().__init__(message)


class SchedulerTimeoutError(SchedulerClientError):
    """Raised when a request to the scheduler times out.

    Examples:
        - REST API request exceeds timeout
        - WebSocket receive timeout
    """

    def __init__(self, message: str, timeout_seconds: float):
        """Initialize the timeout error.

        Args:
            message: Error message describing the timeout
            timeout_seconds: The timeout value that was exceeded
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{message} (timeout={timeout_seconds}s)")


class SchedulerWebSocketError(SchedulerClientError):
    """Raised when WebSocket communication fails.

    Examples:
        - WebSocket connection closed unexpectedly
        - Invalid message format received
        - Protocol error
    """

    pass
