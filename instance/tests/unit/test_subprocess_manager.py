"""
Unit tests for src/subprocess_manager.py
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import httpx
from src.subprocess_manager import SubprocessManager, get_subprocess_manager, get_docker_manager
from src.models import ModelInfo, ModelRegistryEntry


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManager:
    """Test suite for SubprocessManager class"""

    async def test_init(self):
        """Test SubprocessManager initialization"""
        manager = SubprocessManager()

        assert manager.current_model is None
        assert manager.http_client is not None
        assert isinstance(manager.http_client, httpx.AsyncClient)
        assert manager.uv_run_process is None

    async def test_start_model_success(self, mock_model_registry, mock_config, temp_model_directory):
        """Test successful model start"""
        manager = SubprocessManager()

        # Mock registry
        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock subprocess for uv sync and uv run
        mock_sync_process = AsyncMock()
        mock_sync_process.returncode = 0
        mock_sync_process.communicate.return_value = (b"Successfully synced", b"")

        mock_run_process = AsyncMock()
        mock_run_process.returncode = None  # Process still running
        mock_run_process.pid = 12345
        # Mock stdout and stderr with proper stream readers
        mock_run_process.stdout = AsyncMock()
        mock_run_process.stdout.readline = AsyncMock(return_value=b"")
        mock_run_process.stderr = AsyncMock()
        mock_run_process.stderr.readline = AsyncMock(return_value=b"")
        manager.uv_run_process = mock_run_process

        # Mock httpx client for health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                    # First call: uv sync, Second call: uv run
                    mock_subprocess.side_effect = [mock_sync_process, mock_run_process]
                    model_info = await manager.start_model("test-model", {"temperature": 0.7})

        assert model_info is not None
        assert model_info.model_id == "test-model"
        assert model_info.parameters == {"temperature": 0.7}
        assert manager.current_model == model_info
        assert manager.uv_run_process == mock_run_process

    async def test_start_model_not_in_registry(self, mock_model_registry, mock_config):
        """Test starting a model not in registry raises ValueError"""
        manager = SubprocessManager()

        # Mock registry to return None
        mock_model_registry.get_model.return_value = None

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("non-existent-model")

        assert "not found in registry" in str(exc_info.value)

    async def test_start_model_directory_not_found(self, mock_model_registry, mock_config):
        """Test starting model with missing directory raises ValueError"""
        manager = SubprocessManager()

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = Path("/non/existent/path")

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("test-model")

        assert "directory not found" in str(exc_info.value)

    async def test_start_model_no_pyproject_toml(self, mock_model_registry, mock_config, tmp_path):
        """Test starting model without pyproject.toml raises ValueError"""
        manager = SubprocessManager()

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        # Create main.py but no pyproject.toml
        (model_dir / "main.py").write_text("print('test')")

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = model_dir

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("test-model")

        assert "main.py and pyproject.toml" in str(exc_info.value)

    async def test_start_model_no_main_py(self, mock_model_registry, mock_config, tmp_path):
        """Test starting model without main.py raises ValueError"""
        manager = SubprocessManager()

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        # Create pyproject.toml but no main.py
        (model_dir / "pyproject.toml").write_text("[project]\nname = 'test'")

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = model_dir

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("test-model")

        assert "main.py and pyproject.toml" in str(exc_info.value)

    async def test_start_model_uv_sync_failure(self, mock_model_registry, mock_config, temp_model_directory):
        """Test handling of uv sync failure"""
        manager = SubprocessManager()

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock uv sync failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Failed to sync dependencies")

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    with pytest.raises(RuntimeError) as exc_info:
                        await manager.start_model("test-model")

        assert "Failed to run uv sync" in str(exc_info.value)

    async def test_start_model_health_check_timeout(self, mock_model_registry, mock_config, temp_model_directory):
        """Test cleanup when health check fails"""
        manager = SubprocessManager()

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock successful uv sync
        mock_sync_process = AsyncMock()
        mock_sync_process.returncode = 0
        mock_sync_process.communicate.return_value = (b"", b"")

        # Mock uv run process
        mock_run_process = AsyncMock()
        mock_run_process.returncode = None
        mock_run_process.pid = 12345
        mock_run_process.wait = AsyncMock(return_value=0)
        mock_run_process.terminate = Mock()
        mock_run_process.kill = Mock()
        # Mock stdout and stderr with proper stream readers
        mock_run_process.stdout = AsyncMock()
        mock_run_process.stdout.readline = AsyncMock(return_value=b"")
        mock_run_process.stderr = AsyncMock()
        mock_run_process.stderr.readline = AsyncMock(return_value=b"")

        # Mock health check to always fail
        manager.http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                    mock_subprocess.side_effect = [mock_sync_process, mock_run_process]
                    with patch("asyncio.sleep", return_value=None):  # Speed up test
                        with pytest.raises(RuntimeError) as exc_info:
                            await manager.start_model("test-model")

        assert "did not become healthy" in str(exc_info.value)
        assert manager.uv_run_process is None  # Should be cleaned up

    async def test_stop_model_success(self):
        """Test successful model stop"""
        manager = SubprocessManager()

        # Set current model and process
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = None  # Still running
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.terminate = Mock()
        manager.uv_run_process = mock_process

        stopped_id = await manager.stop_model()

        assert stopped_id == "test-model"
        assert manager.current_model is None
        assert manager.uv_run_process is None
        mock_process.terminate.assert_called_once()

    async def test_stop_model_none_running(self):
        """Test stopping when no model is running returns None"""
        manager = SubprocessManager()

        result = await manager.stop_model()

        assert result is None

    async def test_stop_model_process_already_finished(self):
        """Test stopping when process already finished"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Mock process that already finished
        mock_process = AsyncMock()
        mock_process.returncode = 0  # Already finished
        mock_process.pid = 12345
        manager.uv_run_process = mock_process

        stopped_id = await manager.stop_model()

        assert stopped_id == "test-model"
        assert manager.current_model is None

    async def test_stop_model_process_termination_failure(self):
        """Test stopping when process termination fails"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Mock subprocess that fails to terminate
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.terminate = Mock(side_effect=OSError("Permission denied"))
        manager.uv_run_process = mock_process

        with pytest.raises(RuntimeError) as exc_info:
            await manager.stop_model()

        assert "Failed to stop subprocess" in str(exc_info.value)

    async def test_stop_model_force_kill(self):
        """Test stopping with force kill when graceful termination times out"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Mock subprocess that doesn't respond to terminate but responds to kill
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.terminate = Mock()
        mock_process.kill = Mock()
        # First wait (after terminate) times out, second wait (after kill) succeeds
        mock_process.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), None])
        manager.uv_run_process = mock_process

        # Mock wait_for to let the TimeoutError propagate from process.wait()
        original_wait_for = asyncio.wait_for
        async def mock_wait_for(coro, timeout):
            return await original_wait_for(coro, timeout)

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            stopped_id = await manager.stop_model()

        assert stopped_id == "test-model"
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    async def test_restart_model_success(self, mock_model_registry, mock_config, temp_model_directory):
        """Test successful model restart"""
        manager = SubprocessManager()

        # Set current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={"temperature": 0.7},
            container_name="model_instance_test-model"
        )

        # Mock old process
        mock_old_process = Mock()
        mock_old_process.returncode = None
        mock_old_process.pid = 12345
        mock_old_process.wait = AsyncMock(return_value=0)
        mock_old_process.terminate = Mock()
        # Mock stdout and stderr
        mock_old_process.stdout = AsyncMock()
        mock_old_process.stdout.readline = AsyncMock(return_value=b"")
        mock_old_process.stderr = AsyncMock()
        mock_old_process.stderr.readline = AsyncMock(return_value=b"")
        manager.uv_run_process = mock_old_process

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock uv sync and run (for restart)
        mock_sync_process = AsyncMock()
        mock_sync_process.returncode = 0
        mock_sync_process.communicate.return_value = (b"", b"")

        mock_new_run_process = AsyncMock()
        mock_new_run_process.returncode = None
        mock_new_run_process.pid = 67890
        # Mock stdout and stderr with proper stream readers
        mock_new_run_process.stdout = AsyncMock()
        mock_new_run_process.stdout.readline = AsyncMock(return_value=b"")
        mock_new_run_process.stderr = AsyncMock()
        mock_new_run_process.stderr.readline = AsyncMock(return_value=b"")
        manager.uv_run_process = mock_new_run_process

        # Mock health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.get_registry", return_value=mock_model_registry):
            with patch("src.subprocess_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                    # First two calls: uv sync and run for restart
                    mock_subprocess.side_effect = [mock_sync_process, mock_new_run_process]
                    restarted_id = await manager.restart_model()

        assert restarted_id == "test-model"
        assert manager.current_model is not None
        assert manager.current_model.model_id == "test-model"
        assert manager.current_model.parameters == {"temperature": 0.7}

    async def test_restart_model_no_model_running(self):
        """Test restarting when no model is running returns None"""
        manager = SubprocessManager()

        result = await manager.restart_model()

        assert result is None

    async def test_restart_model_stop_failure(self):
        """Test restart fails when stop fails"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Mock process that fails to stop
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.terminate = Mock(side_effect=OSError("Permission denied"))
        manager.uv_run_process = mock_process

        with pytest.raises(RuntimeError) as exc_info:
            await manager.restart_model()

        assert "Failed to restart subprocess" in str(exc_info.value)
        assert manager.current_model is None
        assert manager.uv_run_process is None

    async def test_get_current_model(self):
        """Test getting current model"""
        manager = SubprocessManager()

        # No model running
        assert await manager.get_current_model() is None

        # Set a model
        model_info = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager.current_model = model_info

        # Get current model
        result = await manager.get_current_model()
        assert result == model_info

    async def test_is_model_running_true(self):
        """Test is_model_running returns True when model is running"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        assert await manager.is_model_running() is True

    async def test_is_model_running_false(self):
        """Test is_model_running returns False when no model is running"""
        manager = SubprocessManager()

        assert await manager.is_model_running() is False

    async def test_check_model_health_healthy(self, mock_config):
        """Test check_model_health returns True for healthy model"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock healthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is True

    async def test_check_model_health_unhealthy(self, mock_config):
        """Test check_model_health returns False for unhealthy model"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock unhealthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "unhealthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is False

    async def test_check_model_health_no_model(self):
        """Test check_model_health returns False when no model is running"""
        manager = SubprocessManager()

        result = await manager.check_model_health()

        assert result is False

    async def test_check_model_health_non_200_status(self, mock_config):
        """Test check_model_health returns False for non-200 status code"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock non-200 response
        mock_response = Mock()
        mock_response.status_code = 500
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is False

    async def test_check_model_health_exception(self, mock_config):
        """Test check_model_health returns False on exception"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock exception
        manager.http_client.get = AsyncMock(side_effect=Exception("Network error"))

        with patch("src.subprocess_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is False

    async def test_invoke_inference_success(self, mock_config):
        """Test successful inference invocation"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock inference response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"output": "inference result"})
        manager.http_client.post = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            result = await manager.invoke_inference({"prompt": "test"})

        assert result == {"output": "inference result"}

    async def test_invoke_inference_no_model(self):
        """Test inference fails when no model is running"""
        manager = SubprocessManager()

        with pytest.raises(RuntimeError) as exc_info:
            await manager.invoke_inference({"prompt": "test"})

        assert "No model is currently running" in str(exc_info.value)

    async def test_invoke_inference_timeout(self, mock_config):
        """Test inference timeout handling"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        manager.http_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch("src.subprocess_manager.config", mock_config):
            with pytest.raises(RuntimeError) as exc_info:
                await manager.invoke_inference({"prompt": "test"})

        assert "Inference timeout" in str(exc_info.value)

    async def test_invoke_inference_error(self, mock_config):
        """Test inference error handling"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json = Mock(return_value={"error": "Model error"})
        manager.http_client.post = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            with pytest.raises(RuntimeError) as exc_info:
                await manager.invoke_inference({"prompt": "test"})

        assert "Inference failed" in str(exc_info.value)

    async def test_invoke_inference_request_error(self, mock_config):
        """Test inference request error handling"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        manager.http_client.post = AsyncMock(side_effect=httpx.RequestError("Connection refused"))

        with patch("src.subprocess_manager.config", mock_config):
            with pytest.raises(RuntimeError) as exc_info:
                await manager.invoke_inference({"prompt": "test"})

        assert "Inference request failed" in str(exc_info.value)

    async def test_wait_for_health_success(self, mock_config):
        """Test successful health check wait"""
        manager = SubprocessManager()

        # Mock healthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.subprocess_manager.config", mock_config):
            # Should not raise
            await manager._wait_for_health(8080, timeout=5, interval=1)

    async def test_wait_for_health_timeout(self, mock_config):
        """Test health check timeout"""
        manager = SubprocessManager()

        # Mock unhealthy response
        manager.http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

        with patch("src.subprocess_manager.config", mock_config):
            with patch("asyncio.sleep", return_value=None):  # Speed up test
                with pytest.raises(RuntimeError) as exc_info:
                    await manager._wait_for_health(8080, timeout=2, interval=1)

        assert "did not become healthy" in str(exc_info.value)

    async def test_stop_subprocess(self):
        """Test stopping subprocess cleanly"""
        manager = SubprocessManager()

        # Mock subprocess that terminates gracefully
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.terminate = Mock()

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.return_value = None  # wait() returns immediately
            await manager._stop_subprocess(mock_process)

        mock_process.terminate.assert_called_once()

    async def test_stop_subprocess_already_finished(self):
        """Test stopping subprocess that already finished"""
        manager = SubprocessManager()

        # Mock process that already finished
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.pid = 12345

        # Should not raise and should return early
        await manager._stop_subprocess(mock_process)

    async def test_stop_subprocess_force_kill(self):
        """Test force killing subprocess when graceful termination fails"""
        manager = SubprocessManager()

        # Mock subprocess that doesn't respond to terminate
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.terminate = Mock()
        mock_process.kill = Mock()
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await manager._stop_subprocess(mock_process)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    async def test_close(self):
        """Test resource cleanup"""
        manager = SubprocessManager()

        # Mock aclose
        manager.http_client.aclose = AsyncMock()

        await manager.close()

        manager.http_client.aclose.assert_called_once()


@pytest.mark.unit
class TestSubprocessManagerHelpers:
    """Test suite for SubprocessManager helper methods (synchronous)"""

    def test_build_env_vars(self, mock_config):
        """Test building environment variables"""
        manager = SubprocessManager()

        with patch("src.subprocess_manager.config", mock_config):
            with patch.dict("os.environ", {}, clear=True):
                env_vars = manager._build_env_vars(
                    "test-model",
                    {"temperature": 0.7, "max_tokens": 100}
                )

        assert env_vars["MODEL_ID"] == "test-model"
        assert env_vars["INSTANCE_ID"] == mock_config.instance_id
        assert env_vars["LOG_LEVEL"] == mock_config.log_level
        assert env_vars["MODEL_TEMPERATURE"] == "0.7"
        assert env_vars["MODEL_MAX_TOKENS"] == "100"

    def test_build_env_vars_complex_types(self, mock_config):
        """Test building environment variables with complex types (dict, list)"""
        manager = SubprocessManager()

        with patch("src.subprocess_manager.config", mock_config):
            with patch.dict("os.environ", {}, clear=True):
                env_vars = manager._build_env_vars(
                    "test-model",
                    {
                        "config": {"nested": "value"},
                        "list_param": [1, 2, 3],
                        "string_param": "simple"
                    }
                )

        assert env_vars["MODEL_CONFIG"] == json.dumps({"nested": "value"})
        assert env_vars["MODEL_LIST_PARAM"] == json.dumps([1, 2, 3])
        assert env_vars["MODEL_STRING_PARAM"] == "simple"

    def test_build_env_vars_inherits_parent_env(self, mock_config):
        """Test that environment variables inherit from parent process"""
        manager = SubprocessManager()

        with patch("src.subprocess_manager.config", mock_config):
            with patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/home/test"}, clear=False):
                env_vars = manager._build_env_vars("test-model", {})

        # Should include parent environment
        assert "PATH" in env_vars
        assert "HOME" in env_vars
        assert env_vars["MODEL_ID"] == "test-model"


@pytest.mark.unit
class TestGetSubprocessManager:
    """Test suite for get_subprocess_manager function"""

    def test_get_subprocess_manager_singleton(self):
        """Test that get_subprocess_manager returns a singleton instance"""
        # Reset the global manager
        import src.subprocess_manager
        src.subprocess_manager._subprocess_manager = None

        # First call creates the instance
        manager1 = get_subprocess_manager()
        assert manager1 is not None

        # Second call returns the same instance
        manager2 = get_subprocess_manager()
        assert manager2 is manager1

    def test_get_subprocess_manager_creates_instance(self):
        """Test that get_subprocess_manager creates manager on first call"""
        # Reset the global manager
        import src.subprocess_manager
        src.subprocess_manager._subprocess_manager = None

        manager = get_subprocess_manager()

        assert manager is not None
        assert isinstance(manager, SubprocessManager)


@pytest.mark.unit
class TestGetDockerManager:
    """Test suite for get_docker_manager backward compatibility alias"""

    def test_get_docker_manager_backward_compatibility(self):
        """Test that get_docker_manager returns SubprocessManager instance"""
        # Reset the global manager
        import src.subprocess_manager
        src.subprocess_manager._subprocess_manager = None

        manager = get_docker_manager()

        assert manager is not None
        assert isinstance(manager, SubprocessManager)

    def test_get_docker_manager_singleton(self):
        """Test that get_docker_manager returns singleton"""
        # Reset the global manager
        import src.subprocess_manager
        src.subprocess_manager._subprocess_manager = None

        manager1 = get_docker_manager()
        manager2 = get_docker_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SubprocessManager)
