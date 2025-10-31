"""
Unit tests for src/docker_manager.py
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import httpx
from src.docker_manager import DockerManager, get_docker_manager
from src.models import ModelInfo, ModelRegistryEntry


@pytest.mark.unit
@pytest.mark.asyncio
class TestDockerManager:
    """Test suite for DockerManager class"""

    async def test_init(self):
        """Test DockerManager initialization"""
        manager = DockerManager()

        assert manager.current_model is None
        assert manager.http_client is not None
        assert isinstance(manager.http_client, httpx.AsyncClient)

    async def test_start_model_success(self, mock_model_registry, mock_config, temp_model_directory):
        """Test successful model start"""
        manager = DockerManager()

        # Mock registry
        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")

        # Mock httpx client for health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    model_info = await manager.start_model("test-model", {"temperature": 0.7})

        assert model_info is not None
        assert model_info.model_id == "test-model"
        assert model_info.parameters == {"temperature": 0.7}
        assert manager.current_model == model_info

    async def test_start_model_not_in_registry(self, mock_model_registry, mock_config):
        """Test starting a model not in registry raises ValueError"""
        manager = DockerManager()

        # Mock registry to return None
        mock_model_registry.get_model.return_value = None

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("non-existent-model")

        assert "not found in registry" in str(exc_info.value)

    async def test_start_model_directory_not_found(self, mock_model_registry, mock_config):
        """Test starting model with missing directory raises ValueError"""
        manager = DockerManager()

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = Path("/non/existent/path")

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("test-model")

        assert "directory not found" in str(exc_info.value)

    async def test_start_model_no_docker_compose(self, mock_model_registry, mock_config, tmp_path):
        """Test starting model without docker-compose.yaml raises ValueError"""
        manager = DockerManager()

        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        # Don't create docker-compose.yaml

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = model_dir

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with pytest.raises(ValueError) as exc_info:
                    await manager.start_model("test-model")

        assert "docker-compose.yaml not found" in str(exc_info.value)

    async def test_start_model_health_check_timeout(self, mock_model_registry, mock_config, temp_model_directory):
        """Test cleanup when health check fails"""
        manager = DockerManager()

        mock_model_registry.get_model.return_value = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )
        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock successful docker-compose up
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")

        # Mock health check to always fail
        manager.http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    with patch("asyncio.sleep", return_value=None):  # Speed up test
                        with pytest.raises(RuntimeError) as exc_info:
                            await manager.start_model("test-model")

        assert "did not become healthy" in str(exc_info.value)

    async def test_stop_model_success(self, mock_model_registry, mock_config, temp_model_directory):
        """Test successful model stop"""
        manager = DockerManager()

        # Set current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_test-instance_test-model"
        )

        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock docker-compose down
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    stopped_id = await manager.stop_model()

        assert stopped_id == "test-model"
        assert manager.current_model is None

    async def test_stop_model_none_running(self):
        """Test stopping when no model is running returns None"""
        manager = DockerManager()

        result = await manager.stop_model()

        assert result is None

    async def test_stop_model_cleanup_failure(self, mock_model_registry, mock_config, temp_model_directory):
        """Test fallback to force removal when docker-compose fails"""
        manager = DockerManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_test-instance_test-model"
        )

        mock_model_registry.get_model_directory.return_value = temp_model_directory

        # Mock docker-compose down to fail
        mock_process_fail = AsyncMock()
        mock_process_fail.returncode = 1
        mock_process_fail.communicate.return_value = (b"", b"Error stopping container")

        # Mock force remove to succeed
        mock_process_success = AsyncMock()
        mock_process_success.returncode = 0
        mock_process_success.communicate.return_value = (b"", b"")

        with patch("src.docker_manager.get_registry", return_value=mock_model_registry):
            with patch("src.docker_manager.config", mock_config):
                with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                    # First call (docker-compose down) fails, subsequent calls (force remove) succeed
                    mock_subprocess.side_effect = [mock_process_fail, mock_process_success, mock_process_success]
                    await manager.stop_model()

        assert manager.current_model is None

    async def test_get_current_model(self):
        """Test getting current model"""
        manager = DockerManager()

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
        manager = DockerManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        assert await manager.is_model_running() is True

    async def test_is_model_running_false(self):
        """Test is_model_running returns False when no model is running"""
        manager = DockerManager()

        assert await manager.is_model_running() is False

    async def test_check_model_health_healthy(self, mock_config):
        """Test check_model_health returns True for healthy model"""
        manager = DockerManager()

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

        with patch("src.docker_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is True

    async def test_check_model_health_unhealthy(self, mock_config):
        """Test check_model_health returns False for unhealthy model"""
        manager = DockerManager()

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

        with patch("src.docker_manager.config", mock_config):
            result = await manager.check_model_health()

        assert result is False

    async def test_check_model_health_no_model(self):
        """Test check_model_health returns False when no model is running"""
        manager = DockerManager()

        result = await manager.check_model_health()

        assert result is False

    async def test_invoke_inference_success(self, mock_config):
        """Test successful inference invocation"""
        manager = DockerManager()

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

        with patch("src.docker_manager.config", mock_config):
            result = await manager.invoke_inference({"prompt": "test"})

        assert result == {"output": "inference result"}

    async def test_invoke_inference_no_model(self):
        """Test inference fails when no model is running"""
        manager = DockerManager()

        with pytest.raises(RuntimeError) as exc_info:
            await manager.invoke_inference({"prompt": "test"})

        assert "No model is currently running" in str(exc_info.value)

    async def test_invoke_inference_timeout(self, mock_config):
        """Test inference timeout handling"""
        manager = DockerManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        manager.http_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with patch("src.docker_manager.config", mock_config):
            with pytest.raises(RuntimeError) as exc_info:
                await manager.invoke_inference({"prompt": "test"})

        assert "Inference timeout" in str(exc_info.value)

    async def test_invoke_inference_error(self, mock_config):
        """Test inference error handling"""
        manager = DockerManager()

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

        with patch("src.docker_manager.config", mock_config):
            with pytest.raises(RuntimeError) as exc_info:
                await manager.invoke_inference({"prompt": "test"})

        assert "Inference failed" in str(exc_info.value)

    def test_build_env_vars(self, mock_config):
        """Test building environment variables"""
        manager = DockerManager()

        with patch("src.docker_manager.config", mock_config):
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
        manager = DockerManager()

        with patch("src.docker_manager.config", mock_config):
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

    async def test_wait_for_health_success(self, mock_config):
        """Test successful health check wait"""
        manager = DockerManager()

        # Mock healthy response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"status": "healthy"})
        manager.http_client.get = AsyncMock(return_value=mock_response)

        with patch("src.docker_manager.config", mock_config):
            # Should not raise
            await manager._wait_for_health(8080, timeout=5, interval=1)

    async def test_wait_for_health_timeout(self, mock_config):
        """Test health check timeout"""
        manager = DockerManager()

        # Mock unhealthy response
        manager.http_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

        with patch("src.docker_manager.config", mock_config):
            with patch("asyncio.sleep", return_value=None):  # Speed up test
                with pytest.raises(RuntimeError) as exc_info:
                    await manager._wait_for_health(8080, timeout=2, interval=1)

        assert "did not become healthy" in str(exc_info.value)

    async def test_run_docker_compose_success(self, temp_model_directory):
        """Test successful docker-compose execution"""
        manager = DockerManager()

        # Mock successful process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Container started\n", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Should not raise
            await manager._run_docker_compose(
                temp_model_directory,
                ["up", "-d"],
                {"MODEL_PORT": "8080"}
            )

    async def test_run_docker_compose_failure(self, temp_model_directory):
        """Test docker-compose failure"""
        manager = DockerManager()

        # Mock failed process
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Container failed to start\n")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with pytest.raises(RuntimeError) as exc_info:
                await manager._run_docker_compose(
                    temp_model_directory,
                    ["up", "-d"],
                    {"MODEL_PORT": "8080"}
                )

        assert "docker-compose command failed" in str(exc_info.value)

    async def test_force_remove_container(self):
        """Test forcing container removal"""
        manager = DockerManager()

        # Mock successful processes
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Should not raise
            await manager._force_remove_container("test-container")

    async def test_close(self):
        """Test resource cleanup"""
        manager = DockerManager()

        # Mock aclose
        manager.http_client.aclose = AsyncMock()

        await manager.close()

        manager.http_client.aclose.assert_called_once()


@pytest.mark.unit
class TestGetDockerManager:
    """Test suite for get_docker_manager function"""

    def test_get_docker_manager_singleton(self):
        """Test that get_docker_manager returns a singleton instance"""
        # Reset the global manager
        import src.docker_manager
        src.docker_manager._docker_manager = None

        # First call creates the instance
        manager1 = get_docker_manager()
        assert manager1 is not None

        # Second call returns the same instance
        manager2 = get_docker_manager()
        assert manager2 is manager1

    def test_get_docker_manager_creates_instance(self):
        """Test that get_docker_manager creates manager on first call"""
        # Reset the global manager
        import src.docker_manager
        src.docker_manager._docker_manager = None

        manager = get_docker_manager()

        assert manager is not None
        assert isinstance(manager, DockerManager)
