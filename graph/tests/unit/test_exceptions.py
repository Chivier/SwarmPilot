"""
Unit tests for custom exception hierarchy.

Tests cover exception initialization, formatting, and inheritance.
"""

import pytest

from src.clients.exceptions import (
    SchedulerClientError,
    SchedulerConnectionError,
    SchedulerNotFoundError,
    SchedulerServiceError,
    SchedulerTimeoutError,
    SchedulerValidationError,
    SchedulerWebSocketError,
)


class TestSchedulerClientError:
    """Test suite for SchedulerClientError base exception."""

    def test_init_with_message_only(self):
        """Test initialization with message only."""
        error = SchedulerClientError("Test error")

        assert error.message == "Test error"
        assert error.status_code is None
        assert error.response_body is None
        assert str(error) == "Test error"

    def test_init_with_status_code(self):
        """Test initialization with status code."""
        error = SchedulerClientError("Test error", status_code=404)

        assert error.message == "Test error"
        assert error.status_code == 404
        assert "status=404" in str(error)

    def test_init_with_response_body(self):
        """Test initialization with response body."""
        error = SchedulerClientError("Test error", response_body='{"detail": "Not found"}')

        assert error.message == "Test error"
        assert error.response_body == '{"detail": "Not found"}'
        assert 'Response: {"detail": "Not found"}' in str(error)

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        error = SchedulerClientError("Test error", status_code=500, response_body="Internal error")

        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.response_body == "Internal error"
        assert "status=500" in str(error)
        assert "Response: Internal error" in str(error)

    def test_format_message_structure(self):
        """Test message formatting structure."""
        error = SchedulerClientError("Error", status_code=400, response_body="Bad request")

        formatted = str(error)
        assert formatted.startswith("Error")
        assert "status=400" in formatted
        assert "Response: Bad request" in formatted


class TestSchedulerNotFoundError:
    """Test suite for SchedulerNotFoundError."""

    def test_inherits_from_base(self):
        """Test SchedulerNotFoundError inherits from SchedulerClientError."""
        error = SchedulerNotFoundError("Resource not found", status_code=404)

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerNotFoundError)

    def test_init_and_formatting(self):
        """Test initialization and message formatting."""
        error = SchedulerNotFoundError("Task not found", status_code=404, response_body='{"task_id": "123"}')

        assert error.message == "Task not found"
        assert error.status_code == 404
        assert "status=404" in str(error)
        assert "Task not found" in str(error)


class TestSchedulerValidationError:
    """Test suite for SchedulerValidationError."""

    def test_inherits_from_base(self):
        """Test SchedulerValidationError inherits from SchedulerClientError."""
        error = SchedulerValidationError("Validation failed", status_code=400)

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerValidationError)

    def test_with_validation_details(self):
        """Test with validation error details."""
        error = SchedulerValidationError(
            "Invalid request", status_code=400, response_body='{"errors": ["Missing field: task_id"]}'
        )

        assert error.message == "Invalid request"
        assert error.status_code == 400
        assert error.response_body == '{"errors": ["Missing field: task_id"]}'


class TestSchedulerServiceError:
    """Test suite for SchedulerServiceError."""

    def test_inherits_from_base(self):
        """Test SchedulerServiceError inherits from SchedulerClientError."""
        error = SchedulerServiceError("Service unavailable", status_code=503)

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerServiceError)

    def test_with_service_error_details(self):
        """Test with service error details."""
        error = SchedulerServiceError(
            "Predictor service down", status_code=503, response_body='{"detail": "Predictor unavailable"}'
        )

        assert error.message == "Predictor service down"
        assert error.status_code == 503
        assert "status=503" in str(error)


class TestSchedulerConnectionError:
    """Test suite for SchedulerConnectionError."""

    def test_inherits_from_base(self):
        """Test SchedulerConnectionError inherits from SchedulerClientError."""
        error = SchedulerConnectionError("Connection failed")

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerConnectionError)

    def test_init_with_message_only(self):
        """Test initialization with message only."""
        error = SchedulerConnectionError("Cannot connect to scheduler")

        assert error.message == "Cannot connect to scheduler"
        assert error.original_exception is None
        assert str(error) == "Cannot connect to scheduler"

    def test_init_with_original_exception(self):
        """Test initialization with original exception."""
        original = ConnectionError("Network unreachable")
        error = SchedulerConnectionError("Connection failed", original_exception=original)

        assert error.message == "Connection failed"
        assert error.original_exception == original
        assert str(error) == "Connection failed"

    def test_preserves_original_exception(self):
        """Test that original exception is preserved for debugging."""
        original = TimeoutError("Connection timeout")
        error = SchedulerConnectionError("Failed to connect", original_exception=original)

        assert error.original_exception is original
        assert isinstance(error.original_exception, TimeoutError)


class TestSchedulerTimeoutError:
    """Test suite for SchedulerTimeoutError."""

    def test_inherits_from_base(self):
        """Test SchedulerTimeoutError inherits from SchedulerClientError."""
        error = SchedulerTimeoutError("Request timed out", timeout_seconds=30.0)

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerTimeoutError)

    def test_init_with_timeout(self):
        """Test initialization with timeout value."""
        error = SchedulerTimeoutError("Request timed out", timeout_seconds=15.5)

        assert error.timeout_seconds == 15.5
        assert "timeout=15.5s" in str(error)
        assert "Request timed out" in str(error)

    def test_message_includes_timeout(self):
        """Test that message includes timeout information."""
        error = SchedulerTimeoutError("API call timeout", timeout_seconds=60.0)

        message = str(error)
        assert "API call timeout" in message
        assert "timeout=60.0s" in message


class TestSchedulerWebSocketError:
    """Test suite for SchedulerWebSocketError."""

    def test_inherits_from_base(self):
        """Test SchedulerWebSocketError inherits from SchedulerClientError."""
        error = SchedulerWebSocketError("WebSocket connection closed")

        assert isinstance(error, SchedulerClientError)
        assert isinstance(error, SchedulerWebSocketError)

    def test_init_with_websocket_error(self):
        """Test initialization with WebSocket error message."""
        error = SchedulerWebSocketError("Connection closed unexpectedly")

        assert error.message == "Connection closed unexpectedly"
        assert str(error) == "Connection closed unexpectedly"

    def test_with_protocol_error(self):
        """Test with WebSocket protocol error."""
        error = SchedulerWebSocketError("Invalid message format", response_body='{"type": "invalid"}')

        assert error.message == "Invalid message format"
        assert error.response_body == '{"type": "invalid"}'
        assert "Invalid message format" in str(error)


class TestExceptionInheritance:
    """Test suite for exception hierarchy and inheritance."""

    def test_all_inherit_from_base(self):
        """Test all exceptions inherit from SchedulerClientError."""
        exceptions = [
            SchedulerNotFoundError("test"),
            SchedulerValidationError("test"),
            SchedulerServiceError("test"),
            SchedulerConnectionError("test"),
            SchedulerTimeoutError("test", 30.0),
            SchedulerWebSocketError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, SchedulerClientError)
            assert isinstance(exc, Exception)

    def test_base_inherits_from_exception(self):
        """Test SchedulerClientError inherits from Exception."""
        error = SchedulerClientError("test")

        assert isinstance(error, Exception)

    def test_exceptions_can_be_caught_as_base(self):
        """Test specific exceptions can be caught as base exception."""
        try:
            raise SchedulerNotFoundError("Resource not found", status_code=404)
        except SchedulerClientError as e:
            assert isinstance(e, SchedulerNotFoundError)
            assert e.status_code == 404

    def test_exceptions_can_be_caught_specifically(self):
        """Test exceptions can be caught by their specific type."""
        try:
            raise SchedulerValidationError("Invalid input", status_code=400)
        except SchedulerValidationError as e:
            assert e.message == "Invalid input"
            assert e.status_code == 400

    def test_connection_error_special_handling(self):
        """Test SchedulerConnectionError has special original_exception attribute."""
        original = RuntimeError("Network error")

        try:
            raise SchedulerConnectionError("Connection failed", original_exception=original)
        except SchedulerConnectionError as e:
            assert e.original_exception is original
            assert isinstance(e.original_exception, RuntimeError)

    def test_timeout_error_special_handling(self):
        """Test SchedulerTimeoutError has special timeout_seconds attribute."""
        try:
            raise SchedulerTimeoutError("Timeout occurred", timeout_seconds=45.0)
        except SchedulerTimeoutError as e:
            assert e.timeout_seconds == 45.0
            assert "timeout=45.0s" in str(e)


class TestExceptionUsagePatterns:
    """Test suite for common exception usage patterns."""

    def test_reraise_with_context(self):
        """Test reraising exceptions with context."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as ve:
                raise SchedulerConnectionError("Failed to connect", original_exception=ve)
        except SchedulerConnectionError as e:
            assert e.original_exception is not None
            assert isinstance(e.original_exception, ValueError)

    def test_exception_chaining(self):
        """Test exception chaining using from keyword."""
        original = ConnectionError("Network unreachable")

        try:
            raise SchedulerConnectionError("Cannot connect", original_exception=original) from original
        except SchedulerConnectionError as e:
            assert e.__cause__ is original

    def test_multiple_exception_types(self):
        """Test handling multiple exception types."""
        exceptions = [
            SchedulerNotFoundError("404 error", status_code=404),
            SchedulerValidationError("400 error", status_code=400),
            SchedulerServiceError("503 error", status_code=503),
        ]

        for exc in exceptions:
            if isinstance(exc, SchedulerNotFoundError):
                assert exc.status_code == 404
            elif isinstance(exc, SchedulerValidationError):
                assert exc.status_code == 400
            elif isinstance(exc, SchedulerServiceError):
                assert exc.status_code == 503
