"""Unit tests for service management components."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from service_management.service_launcher import ServiceConfig, ServiceLauncher
from service_management.deployment_manager import ModelConfig, DeploymentManager
from service_management.health_checker import HealthChecker, HealthStatus
from service_management.resource_binder import ResourceBinder, ResourceAllocation


# ============================================================================
# ServiceLauncher Tests
# ============================================================================

def test_service_config():
    """Test ServiceConfig dataclass."""
    config = ServiceConfig(
        name="scheduler_a",
        service_type="scheduler",
        host="127.0.0.1",
        port=8100,
        cpu_cores="0-3",
        gpu_id=0,
        env_vars={"LOG_LEVEL": "INFO"}
    )

    assert config.name == "scheduler_a"
    assert config.service_type == "scheduler"
    assert config.host == "127.0.0.1"
    assert config.port == 8100
    assert config.cpu_cores == "0-3"
    assert config.gpu_id == 0
    assert config.env_vars["LOG_LEVEL"] == "INFO"


def test_service_launcher_initialization():
    """Test ServiceLauncher initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        launcher = ServiceLauncher(log_dir=tmpdir)

        assert launcher.log_dir == Path(tmpdir)
        assert launcher.log_dir.exists()
        assert len(launcher.processes) == 0


def test_service_launcher_get_local_ip():
    """Test local IP detection."""
    launcher = ServiceLauncher()

    # Should return some IP (either from interface or hostname)
    ip = launcher.get_local_ip()

    assert ip is not None
    assert isinstance(ip, str)
    # Basic IP format check
    assert len(ip.split('.')) == 4


# ============================================================================
# DeploymentManager Tests
# ============================================================================

def test_model_config():
    """Test ModelConfig dataclass."""
    config = ModelConfig(
        model_id="llm_service",
        instances=["http://localhost:8200", "http://localhost:8201"],
        metadata={"type": "llm"}
    )

    assert config.model_id == "llm_service"
    assert len(config.instances) == 2
    assert config.metadata["type"] == "llm"


def test_deployment_manager_initialization():
    """Test DeploymentManager initialization."""
    manager = DeploymentManager(
        max_workers=5,
        max_retries=3,
        retry_delay=1.0,
        timeout=10.0
    )

    assert manager.max_workers == 5
    assert manager.max_retries == 3
    assert manager.retry_delay == 1.0
    assert manager.timeout == 10.0


@patch('service_management.deployment_manager.requests.post')
def test_deploy_model_success(mock_post):
    """Test successful model deployment."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    manager = DeploymentManager(max_retries=1)

    model_config = ModelConfig(
        model_id="test_model",
        instances=["http://localhost:8200"]
    )

    result = manager.deploy_model(
        "http://localhost:8100",
        model_config
    )

    assert result is True
    assert mock_post.called


@patch('service_management.deployment_manager.requests.post')
def test_deploy_model_retry(mock_post):
    """Test model deployment with retry."""
    # First attempt fails, second succeeds
    mock_response_fail = Mock()
    mock_response_fail.status_code = 500

    # Make raise_for_status raise an exception
    from requests.exceptions import HTTPError
    mock_response_fail.raise_for_status.side_effect = HTTPError("Server error")

    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.raise_for_status.return_value = None  # Success

    mock_post.side_effect = [mock_response_fail, mock_response_success]

    manager = DeploymentManager(max_retries=2, retry_delay=0.1)

    model_config = ModelConfig(
        model_id="test_model",
        instances=["http://localhost:8200"]
    )

    result = manager.deploy_model(
        "http://localhost:8100",
        model_config
    )

    assert result is True
    assert mock_post.call_count == 2


# ============================================================================
# HealthChecker Tests
# ============================================================================

def test_health_checker_initialization():
    """Test HealthChecker initialization."""
    checker = HealthChecker(
        check_interval=5.0,
        timeout=3.0
    )

    assert checker.check_interval == 5.0
    assert checker.timeout == 3.0
    assert len(checker.health_status) == 0


@patch('service_management.health_checker.requests.get')
def test_check_http_endpoint_healthy(mock_get):
    """Test HTTP endpoint health check (healthy)."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    checker = HealthChecker()

    status = checker.check_http_endpoint(
        "test_service",
        "http://localhost:8100/health"
    )

    assert status == HealthStatus.HEALTHY
    assert checker.get_status("test_service") == HealthStatus.HEALTHY


@patch('service_management.health_checker.requests.get')
def test_check_http_endpoint_unhealthy(mock_get):
    """Test HTTP endpoint health check (unhealthy)."""
    # Mock failed response
    mock_response = Mock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    checker = HealthChecker()

    status = checker.check_http_endpoint(
        "test_service",
        "http://localhost:8100/health"
    )

    assert status == HealthStatus.UNHEALTHY
    assert checker.get_status("test_service") == HealthStatus.UNHEALTHY


@patch('service_management.health_checker.requests.get')
def test_check_model_instances(mock_get):
    """Test model instance health check."""
    # Mock response with instances
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "instances": [
            {"url": "http://localhost:8200"},
            {"url": "http://localhost:8201"}
        ]
    }
    mock_get.return_value = mock_response

    checker = HealthChecker()

    status = checker.check_model_instances(
        "test_model",
        "http://localhost:8100",
        min_instances=2
    )

    assert status == HealthStatus.HEALTHY


def test_health_checker_is_healthy():
    """Test is_healthy method."""
    checker = HealthChecker()

    checker.health_status["service1"] = HealthStatus.HEALTHY
    checker.health_status["service2"] = HealthStatus.UNHEALTHY

    assert checker.is_healthy("service1") is True
    assert checker.is_healthy("service2") is False
    assert checker.is_healthy("service3") is False


# ============================================================================
# ResourceBinder Tests
# ============================================================================

def test_resource_binder_initialization():
    """Test ResourceBinder initialization."""
    binder = ResourceBinder()

    assert binder.total_cpus > 0
    assert binder.total_gpus >= 0


def test_allocate_cpus():
    """Test CPU allocation."""
    binder = ResourceBinder()

    cores = binder.allocate_cpus(num_cores=4, start_core=0)

    assert len(cores) == 4
    assert cores == [0, 1, 2, 3]


def test_allocate_cpus_invalid():
    """Test invalid CPU allocation."""
    binder = ResourceBinder()

    # Try to allocate more cores than available
    with pytest.raises(ValueError, match="Cannot allocate"):
        binder.allocate_cpus(num_cores=binder.total_cpus + 1, start_core=0)

    # Invalid start core
    with pytest.raises(ValueError, match="Invalid start_core"):
        binder.allocate_cpus(num_cores=1, start_core=binder.total_cpus + 1)


def test_cpu_list_to_taskset_format():
    """Test taskset format conversion."""
    binder = ResourceBinder()

    # Contiguous range
    assert binder.cpu_list_to_taskset_format([0, 1, 2, 3]) == "0-3"

    # Non-contiguous
    assert binder.cpu_list_to_taskset_format([0, 2, 4]) == "0,2,4"

    # Mixed
    assert binder.cpu_list_to_taskset_format([0, 1, 3, 4, 6]) == "0-1,3-4,6"

    # Single core
    assert binder.cpu_list_to_taskset_format([5]) == "5"

    # Empty
    assert binder.cpu_list_to_taskset_format([]) == ""


def test_gpu_list_to_env_format():
    """Test CUDA_VISIBLE_DEVICES format conversion."""
    binder = ResourceBinder()

    assert binder.gpu_list_to_env_format([0, 1, 2]) == "0,1,2"
    assert binder.gpu_list_to_env_format([0]) == "0"
    assert binder.gpu_list_to_env_format([]) == ""


def test_create_allocation():
    """Test resource allocation creation."""
    binder = ResourceBinder()

    allocation = binder.create_allocation(
        num_cpus=4,
        num_gpus=0,
        cpu_offset=0
    )

    assert isinstance(allocation, ResourceAllocation)
    assert len(allocation.cpu_cores) == 4
    assert allocation.cpu_cores == [0, 1, 2, 3]
    assert len(allocation.gpu_ids) == 0


def test_validate_allocation():
    """Test allocation validation."""
    binder = ResourceBinder()

    # Valid allocation
    allocation = ResourceAllocation(
        cpu_cores=[0, 1, 2],
        gpu_ids=[]
    )

    assert binder.validate_allocation(allocation) is True

    # Invalid CPU
    allocation = ResourceAllocation(
        cpu_cores=[binder.total_cpus + 1],
        gpu_ids=[]
    )

    with pytest.raises(ValueError, match="Invalid CPU core"):
        binder.validate_allocation(allocation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
