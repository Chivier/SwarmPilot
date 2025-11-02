"""
Unit tests for scheduler_client module
"""

import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.scheduler_client import (
    SchedulerClient,
    get_scheduler_client,
    initialize_scheduler_client,
)


class TestSchedulerClient:
    """Test suite for SchedulerClient class"""

    def test_init_with_defaults(self):
        """Test SchedulerClient initialization with default values"""
        client = SchedulerClient()
        assert client.scheduler_url is None
        assert client.instance_id == "instance-default"
        assert client.timeout == 10.0
        assert client.max_retries == 3
        assert client.retry_delay == 2.0
        assert not client._registered

    def test_init_with_custom_values(self):
        """Test SchedulerClient initialization with custom values"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
            instance_endpoint="http://localhost:5000",
            timeout=5.0,
            max_retries=5,
            retry_delay=1.0,
        )
        assert client.scheduler_url == "http://scheduler:8000"
        assert client.instance_id == "test-instance"
        assert client.instance_endpoint == "http://localhost:5000"
        assert client.timeout == 5.0
        assert client.max_retries == 5
        assert client.retry_delay == 1.0

    def test_init_with_env_vars(self, monkeypatch):
        """Test SchedulerClient initialization from environment variables"""
        monkeypatch.setenv("SCHEDULER_URL", "http://env-scheduler:8000")
        monkeypatch.setenv("INSTANCE_ID", "env-instance")
        monkeypatch.setenv("INSTANCE_ENDPOINT", "http://env-instance:5000")

        client = SchedulerClient()
        assert client.scheduler_url == "http://env-scheduler:8000"
        assert client.instance_id == "env-instance"
        assert client.instance_endpoint == "http://env-instance:5000"

    def test_is_enabled_true(self):
        """Test is_enabled returns True when scheduler_url is set"""
        client = SchedulerClient(scheduler_url="http://scheduler:8000")
        assert client.is_enabled is True

    def test_is_enabled_false(self):
        """Test is_enabled returns False when scheduler_url is not set"""
        client = SchedulerClient(scheduler_url=None)
        assert client.is_enabled is False

    def test_get_platform_info(self):
        """Test _get_platform_info returns correct platform information"""
        client = SchedulerClient()
        info = client._get_platform_info()

        assert "software_name" in info
        assert "software_version" in info
        assert "hardware_name" in info
        assert "python_version" in info
        assert "detected_at" in info
        assert info["detected_at"].endswith("Z")

    @pytest.mark.asyncio
    async def test_register_instance_disabled(self, capsys):
        """Test register_instance returns False when scheduler is disabled"""
        client = SchedulerClient(scheduler_url=None)
        result = await client.register_instance("test-model")

        assert result is False
        captured = capsys.readouterr()
        assert "Scheduler integration disabled" in captured.out

    @pytest.mark.asyncio
    async def test_register_instance_success(self):
        """Test successful instance registration"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance("test-model")

            assert result is True
            assert client._registered is True
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://scheduler:8000/instance/register"
            assert "model_id" in call_args[1]["json"]
            assert call_args[1]["json"]["model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_register_instance_with_custom_platform_info(self):
        """Test registration with custom platform info"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )

        custom_info = {"custom_key": "custom_value"}

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance("test-model", platform_info=custom_info)

            assert result is True
            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["platform_info"] == custom_info

    @pytest.mark.asyncio
    async def test_register_instance_failure_response(self, capsys):
        """Test registration with failure response from scheduler"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )

        mock_response = Mock()
        mock_response.json.return_value = {"success": False, "error": "Registration failed"}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.register_instance("test-model")

            assert result is False
            assert client._registered is False
            captured = capsys.readouterr()
            assert "Registration failed" in captured.out

    @pytest.mark.asyncio
    async def test_register_instance_http_error_with_retry(self, capsys):
        """Test registration retries on HTTP error"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
            max_retries=2,
            retry_delay=0.1,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await client.register_instance("test-model")

            assert result is False
            assert client._registered is False
            assert mock_client.post.call_count == 2
            captured = capsys.readouterr()
            assert "Registration attempt" in captured.out
            assert "Failed to register instance after" in captured.out

    @pytest.mark.asyncio
    async def test_register_instance_unexpected_error(self, capsys):
        """Test registration handles unexpected errors"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client.register_instance("test-model")

            assert result is False
            captured = capsys.readouterr()
            assert "Unexpected error during registration" in captured.out

    @pytest.mark.asyncio
    async def test_deregister_instance_disabled(self):
        """Test deregister returns False when scheduler is disabled"""
        client = SchedulerClient(scheduler_url=None)
        result = await client.deregister_instance()

        assert result is False

    @pytest.mark.asyncio
    async def test_deregister_instance_not_registered(self):
        """Test deregister returns False when not registered"""
        client = SchedulerClient(scheduler_url="http://scheduler:8000")
        client._registered = False
        result = await client.deregister_instance()

        assert result is False

    @pytest.mark.asyncio
    async def test_deregister_instance_success(self, capsys):
        """Test successful instance deregistration"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )
        client._registered = True

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()

            assert result is True
            assert client._registered is False
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://scheduler:8000/instance/remove"
            captured = capsys.readouterr()
            assert "deregistered from scheduler" in captured.out

    @pytest.mark.asyncio
    async def test_deregister_instance_failure_response(self, capsys):
        """Test deregistration with failure response from scheduler"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )
        client._registered = True

        mock_response = Mock()
        mock_response.json.return_value = {"success": False, "error": "Not found"}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()

            assert result is False
            captured = capsys.readouterr()
            assert "Deregistration failed" in captured.out

    @pytest.mark.asyncio
    async def test_deregister_instance_http_error(self, capsys):
        """Test deregistration handles HTTP errors"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )
        client._registered = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()

            assert result is False
            captured = capsys.readouterr()
            assert "Failed to deregister instance" in captured.out

    @pytest.mark.asyncio
    async def test_deregister_instance_unexpected_error(self, capsys):
        """Test deregistration handles unexpected errors"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            instance_id="test-instance",
        )
        client._registered = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client.deregister_instance()

            assert result is False
            captured = capsys.readouterr()
            assert "Unexpected error during deregistration" in captured.out

    @pytest.mark.asyncio
    async def test_send_task_result_disabled(self):
        """Test send_task_result returns False when scheduler is disabled and no callback URL"""
        client = SchedulerClient(scheduler_url=None)
        result = await client.send_task_result("task-123", "completed")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_task_result_with_custom_callback(self):
        """Test send_task_result uses custom callback URL"""
        client = SchedulerClient(scheduler_url=None)

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(
                "task-123",
                "completed",
                callback_url="http://custom-callback:8000/callback"
            )

            assert result is True
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://custom-callback:8000/callback"

    @pytest.mark.asyncio
    async def test_send_task_result_derives_callback_from_scheduler_url(self):
        """Test send_task_result derives callback URL from scheduler_url when not provided"""
        client = SchedulerClient(scheduler_url="http://scheduler:8000")

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            # Call without callback_url parameter
            result = await client.send_task_result(
                "task-123",
                "completed",
                result={"output": "test"}
            )

            assert result is True
            call_args = mock_client.post.call_args
            # Verify it derived callback URL from scheduler_url
            assert call_args[0][0] == "http://scheduler:8000/callback/task_result"

    @pytest.mark.asyncio
    async def test_send_task_result_success_with_all_fields(self):
        """Test send_task_result with all optional fields"""
        client = SchedulerClient(scheduler_url="http://scheduler:8000")

        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result(
                "task-123",
                "completed",
                result={"output": "test"},
                error="test error",
                execution_time_ms=123.45
            )

            assert result is True
            call_args = mock_client.post.call_args
            callback_data = call_args[1]["json"]
            assert callback_data["task_id"] == "task-123"
            assert callback_data["status"] == "completed"
            assert callback_data["result"] == {"output": "test"}
            assert callback_data["error"] == "test error"
            assert callback_data["execution_time_ms"] == 123.45

    @pytest.mark.asyncio
    async def test_send_task_result_failure_response_with_retry(self, capsys):
        """Test send_task_result retries on failure response"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            max_retries=2,
            retry_delay=0.1,
        )

        mock_response = Mock()
        mock_response.json.return_value = {"success": False, "error": "Callback failed"}
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await client.send_task_result("task-123", "completed")

            assert result is False
            assert mock_client.post.call_count == 2
            captured = capsys.readouterr()
            assert "Callback failed" in captured.out

    @pytest.mark.asyncio
    async def test_send_task_result_http_error_with_retry(self, capsys):
        """Test send_task_result retries on HTTP error"""
        client = SchedulerClient(
            scheduler_url="http://scheduler:8000",
            max_retries=2,
            retry_delay=0.1,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
            mock_client_class.return_value = mock_client

            result = await client.send_task_result("task-123", "completed")

            assert result is False
            assert mock_client.post.call_count == 2
            captured = capsys.readouterr()
            assert "Callback attempt" in captured.out
            assert "Failed to send callback after" in captured.out

    @pytest.mark.asyncio
    async def test_send_task_result_unexpected_error(self, capsys):
        """Test send_task_result handles unexpected errors"""
        client = SchedulerClient(scheduler_url="http://scheduler:8000")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post = AsyncMock(side_effect=ValueError("Unexpected error"))
            mock_client_class.return_value = mock_client

            result = await client.send_task_result("task-123", "completed")

            assert result is False
            captured = capsys.readouterr()
            assert "Unexpected error during callback" in captured.out


class TestSchedulerClientGlobalFunctions:
    """Test suite for global scheduler client functions"""

    def test_get_scheduler_client_creates_instance(self):
        """Test get_scheduler_client creates a new instance if none exists"""
        # Reset global client
        import src.scheduler_client
        src.scheduler_client._scheduler_client = None

        client = get_scheduler_client()
        assert client is not None
        assert isinstance(client, SchedulerClient)

    def test_get_scheduler_client_returns_same_instance(self):
        """Test get_scheduler_client returns the same instance on multiple calls"""
        import src.scheduler_client
        src.scheduler_client._scheduler_client = None

        client1 = get_scheduler_client()
        client2 = get_scheduler_client()
        assert client1 is client2

    def test_initialize_scheduler_client(self):
        """Test initialize_scheduler_client creates configured instance"""
        import src.scheduler_client
        src.scheduler_client._scheduler_client = None

        client = initialize_scheduler_client(
            scheduler_url="http://test-scheduler:8000",
            instance_id="test-instance",
            instance_endpoint="http://test-endpoint:5000"
        )

        assert client.scheduler_url == "http://test-scheduler:8000"
        assert client.instance_id == "test-instance"
        assert client.instance_endpoint == "http://test-endpoint:5000"

    def test_initialize_scheduler_client_replaces_existing(self):
        """Test initialize_scheduler_client replaces existing instance"""
        import src.scheduler_client

        client1 = initialize_scheduler_client(
            scheduler_url="http://old-scheduler:8000",
            instance_id="old-instance"
        )

        client2 = initialize_scheduler_client(
            scheduler_url="http://new-scheduler:8000",
            instance_id="new-instance"
        )

        assert client1 is not client2
        assert get_scheduler_client() is client2
        assert get_scheduler_client().scheduler_url == "http://new-scheduler:8000"
