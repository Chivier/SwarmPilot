"""Unit tests for HTTP error logging utility.

Tests header sanitization, body truncation, and error logging functionality.
"""

from unittest.mock import MagicMock, patch

import httpx

from src.http_error_logger import (
    MAX_BODY_LENGTH,
    _sanitize_headers,
    _truncate_body,
    log_http_error,
)


class TestSanitizeHeaders:
    """Tests for header sanitization."""

    def test_sanitizes_authorization(self):
        """Test that Authorization header is redacted."""
        headers = {
            "Authorization": "Bearer secret123",
            "Content-Type": "application/json",
        }
        result = _sanitize_headers(headers)
        assert result["Authorization"] == "[REDACTED]"
        assert result["Content-Type"] == "application/json"

    def test_sanitizes_x_api_key(self):
        """Test that X-API-Key header is redacted."""
        headers = {"X-API-Key": "my-secret-key", "Accept": "application/json"}
        result = _sanitize_headers(headers)
        assert result["X-API-Key"] == "[REDACTED]"
        assert result["Accept"] == "application/json"

    def test_sanitizes_cookie(self):
        """Test that Cookie headers are redacted."""
        headers = {"Cookie": "session=abc123", "Set-Cookie": "token=xyz"}
        result = _sanitize_headers(headers)
        assert result["Cookie"] == "[REDACTED]"
        assert result["Set-Cookie"] == "[REDACTED]"

    def test_case_insensitive(self):
        """Test that header matching is case-insensitive."""
        headers = {
            "AUTHORIZATION": "secret",
            "x-api-key": "secret",
            "Cookie": "secret",
        }
        result = _sanitize_headers(headers)
        assert result["AUTHORIZATION"] == "[REDACTED]"
        assert result["x-api-key"] == "[REDACTED]"
        assert result["Cookie"] == "[REDACTED]"

    def test_handles_none(self):
        """Test that None input returns empty dict."""
        assert _sanitize_headers(None) == {}

    def test_handles_httpx_headers(self):
        """Test that httpx.Headers objects are handled."""
        headers = httpx.Headers(
            {"Authorization": "secret", "Accept": "application/json"}
        )
        result = _sanitize_headers(headers)
        # httpx normalizes header names to lowercase
        assert result["authorization"] == "[REDACTED]"
        assert result["accept"] == "application/json"

    def test_preserves_non_sensitive_headers(self):
        """Test that non-sensitive headers are preserved."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "test-client/1.0",
            "X-Request-ID": "req-123",
        }
        result = _sanitize_headers(headers)
        assert result == headers


class TestTruncateBody:
    """Tests for body truncation."""

    def test_truncates_long_string(self):
        """Test that long strings are truncated."""
        long_string = "x" * (MAX_BODY_LENGTH + 1000)
        result = _truncate_body(long_string)
        assert len(result) < len(long_string)
        assert "truncated" in result
        assert str(len(long_string)) in result

    def test_preserves_short_string(self):
        """Test that short strings are preserved."""
        short_string = "short body"
        result = _truncate_body(short_string)
        assert result == short_string

    def test_handles_dict(self):
        """Test that dictionaries are JSON serialized."""
        body = {"key": "value", "nested": {"a": 1}}
        result = _truncate_body(body)
        assert "key" in result
        assert "value" in result
        assert "nested" in result

    def test_handles_bytes(self):
        """Test that UTF-8 bytes are decoded."""
        body = b'{"test": "data"}'
        result = _truncate_body(body)
        assert "test" in result
        assert "data" in result

    def test_handles_binary_bytes(self):
        """Test that non-UTF8 binary bytes show as binary data."""
        body = bytes([0x00, 0x01, 0x02, 0xFF])
        result = _truncate_body(body)
        assert "binary data" in result
        assert "4 bytes" in result

    def test_handles_none(self):
        """Test that None returns '<empty>'."""
        assert _truncate_body(None) == "<empty>"

    def test_handles_empty_string(self):
        """Test that empty string is returned as-is."""
        assert _truncate_body("") == ""

    def test_handles_non_serializable_dict(self):
        """Test that non-serializable dicts are converted via str()."""

        # Create a dict with non-serializable value
        class NonSerializable:
            def __str__(self):
                return "non-serializable-object"

        body = {"key": NonSerializable()}
        result = _truncate_body(body)
        assert "key" in result
        assert "non-serializable-object" in result


class TestLogHttpError:
    """Tests for the main logging function."""

    @patch("src.http_error_logger.logger")
    def test_logs_generic_exception(self, mock_logger):
        """Test logging a generic exception."""
        error = Exception("Test error")
        log_http_error(
            error,
            request_url="http://example.com/api",
            request_method="POST",
            context="test context",
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "test context" in log_message
        assert "Exception" in log_message
        assert "Test error" in log_message
        assert "POST" in log_message
        assert "http://example.com/api" in log_message

    @patch("src.http_error_logger.logger")
    def test_logs_http_status_error(self, mock_logger):
        """Test logging an httpx.HTTPStatusError."""
        request = MagicMock(spec=httpx.Request)
        request.url = "http://example.com/api"
        request.method = "POST"
        request.headers = httpx.Headers(
            {"Authorization": "secret", "Content-Type": "application/json"}
        )

        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.headers = httpx.Headers({"Content-Type": "application/json"})
        response.text = '{"error": "Internal server error"}'

        error = httpx.HTTPStatusError(
            "Server error", request=request, response=response
        )

        log_http_error(
            error, request_body={"data": "test"}, context="test context"
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "test context" in log_message
        assert "HTTPStatusError" in log_message
        assert "500" in log_message
        assert "[REDACTED]" in log_message  # Authorization should be redacted
        assert "Internal server error" in log_message

    @patch("src.http_error_logger.logger")
    def test_logs_timeout_exception(self, mock_logger):
        """Test logging an httpx.TimeoutException."""
        request = MagicMock(spec=httpx.Request)
        request.url = "http://example.com/slow"
        request.method = "GET"
        request.headers = httpx.Headers({})

        error = httpx.TimeoutException("Connection timeout", request=request)

        log_http_error(error, context="timeout test")

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "TimeoutException" in log_message
        assert "timeout test" in log_message
        assert "http://example.com/slow" in log_message

    @patch("src.http_error_logger.logger")
    def test_logs_with_extra_context(self, mock_logger):
        """Test logging with extra context dictionary."""
        error = Exception("Test error")
        log_http_error(
            error,
            request_url="http://example.com",
            extra={"task_id": "123", "instance_id": "abc"},
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "task_id" in log_message
        assert "123" in log_message
        assert "instance_id" in log_message
        assert "abc" in log_message

    @patch("src.http_error_logger.logger")
    def test_logs_with_request_body(self, mock_logger):
        """Test logging with request body included."""
        error = Exception("Test error")
        request_body = {"model_id": "gpt-4", "features": {"size": 100}}

        log_http_error(
            error,
            request_url="http://example.com/predict",
            request_method="POST",
            request_body=request_body,
            context="prediction",
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "model_id" in log_message
        assert "gpt-4" in log_message
        assert "Request Body" in log_message

    @patch("src.http_error_logger.logger")
    def test_logs_without_context(self, mock_logger):
        """Test logging without context shows 'unknown'."""
        error = Exception("Test error")
        log_http_error(error, request_url="http://example.com")

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "[unknown]" in log_message

    @patch("src.http_error_logger.logger")
    def test_handles_response_text_error(self, mock_logger):
        """Test handling when response.text raises an error."""
        request = MagicMock(spec=httpx.Request)
        request.url = "http://example.com/api"
        request.method = "POST"
        request.headers = httpx.Headers({})

        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.headers = httpx.Headers({})
        # Make response.text raise an exception
        type(response).text = property(
            lambda self: (_ for _ in ()).throw(Exception("Cannot read"))
        )

        error = httpx.HTTPStatusError(
            "Server error", request=request, response=response
        )

        log_http_error(error, context="test")

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "unable to read response body" in log_message

    @patch("src.http_error_logger.logger")
    def test_extracts_info_from_http_status_error(self, mock_logger):
        """Test that request info is extracted from HTTPStatusError automatically."""
        request = MagicMock(spec=httpx.Request)
        request.url = "http://auto-extracted.com/api"
        request.method = "DELETE"
        request.headers = httpx.Headers({"X-Custom": "value"})

        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.headers = httpx.Headers({})
        response.text = "Not found"

        error = httpx.HTTPStatusError(
            "Not found", request=request, response=response
        )

        # Don't provide request_url or request_method - should be extracted
        log_http_error(error, context="auto-extract test")

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "http://auto-extracted.com/api" in log_message
        assert "DELETE" in log_message
        assert "404" in log_message

    @patch("src.http_error_logger.logger")
    def test_connection_error_without_response(self, mock_logger):
        """Test logging connection error without response object."""
        error = ConnectionError("Connection refused")

        log_http_error(
            error,
            request_url="http://unreachable.com/api",
            request_method="POST",
            request_body={"test": "data"},
            context="connection error",
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]

        assert "connection error" in log_message
        assert "ConnectionError" in log_message
        assert "Connection refused" in log_message
        assert "http://unreachable.com/api" in log_message
        # Should not have response section
        assert "Response Status" not in log_message

    @patch("src.http_error_logger.logger")
    def test_http_status_error_request_property_raises_runtime_error(
        self, mock_logger
    ):
        """Test handling when HTTPStatusError.request raises RuntimeError."""

        # Create an HTTPStatusError subclass that raises RuntimeError on property access
        class ErrorWithBrokenRequest(httpx.HTTPStatusError):
            @property
            def request(self):
                raise RuntimeError("Request not set")

            @property
            def response(self):
                raise RuntimeError("Response not set")

        # Create instance using parent's __new__ to avoid constructor issues
        error = Exception.__new__(ErrorWithBrokenRequest)
        error.args = ("Mocked error",)

        # Should not raise - the RuntimeError should be caught
        log_http_error(
            error,
            request_url="http://fallback.com",
            request_method="GET",
            context="runtime error test",
        )

        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        assert "runtime error test" in log_message

    @patch("src.http_error_logger.logger")
    def test_http_status_error_response_property_raises_runtime_error(
        self, mock_logger
    ):
        """Test handling when HTTPStatusError.response raises RuntimeError but request works."""

        # Create an HTTPStatusError subclass that works for request but fails on response
        class ErrorWithBrokenResponse(httpx.HTTPStatusError):
            def __init__(self):
                self._request = MagicMock(spec=httpx.Request)
                self._request.url = "http://example.com/api"
                self._request.method = "POST"
                self._request.headers = httpx.Headers({})
                self.args = ("Test error",)

            @property
            def request(self):
                return self._request

            @property
            def response(self):
                raise RuntimeError("Response not set")

        error = ErrorWithBrokenResponse()
        log_http_error(error, context="response error test")

        mock_logger.error.assert_called_once()


class TestTruncateBodyEdgeCases:
    """Additional edge case tests for body truncation."""

    def test_dict_with_circular_reference_fallback_to_str(self):
        """Test that circular reference dict falls back to str()."""
        # Create dict with circular reference - json.dumps will fail
        circular = {}
        circular["self"] = circular

        result = _truncate_body(circular)
        # Should fallback to str(body) - won't show full structure but won't crash
        assert result is not None
        assert len(result) > 0
