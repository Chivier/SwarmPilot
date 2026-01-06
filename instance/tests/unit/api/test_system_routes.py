"""Unit tests for System API routes.

Tests follow TDD principle - written before implementation.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a FastAPI app with system routes."""
    from src.api.routes.system import router

    app = FastAPI()
    app.include_router(router, prefix="/v1/system")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_system_metrics():
    """Mock system metrics for testing."""
    with (
        patch("src.api.routes.system.psutil") as mock_psutil,
        patch("src.api.routes.system.get_gpu_info") as mock_gpu,
        patch("src.api.routes.system.get_instance_id") as mock_instance_id,
        patch("src.api.routes.system.get_uptime_seconds") as mock_uptime,
    ):
        # Mock CPU
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.cpu_percent.return_value = 45.2

        # Mock Memory
        mock_mem = type("Memory", (), {})()
        mock_mem.total = 64 * 1024**3  # 64 GB
        mock_mem.used = 32.5 * 1024**3
        mock_mem.percent = 50.8
        mock_psutil.virtual_memory.return_value = mock_mem

        # Mock Disk
        mock_disk = type("Disk", (), {})()
        mock_disk.total = 500 * 1024**3  # 500 GB
        mock_disk.used = 120 * 1024**3
        mock_disk.free = 380 * 1024**3
        mock_disk.percent = 24.0
        mock_psutil.disk_usage.return_value = mock_disk

        # Mock GPU info
        mock_gpu.return_value = [
            {
                "index": 0,
                "name": "NVIDIA A100",
                "memory_total_gb": 80.0,
                "memory_used_gb": 45.0,
                "utilization_percent": 78.5,
                "temperature_celsius": 65,
            }
        ]

        # Mock instance info
        mock_instance_id.return_value = "inst_test123"
        mock_uptime.return_value = 3600

        yield {
            "psutil": mock_psutil,
            "gpu": mock_gpu,
            "instance_id": mock_instance_id,
            "uptime": mock_uptime,
        }


class TestHealthEndpoint:
    """Tests for GET /v1/system/health endpoint."""

    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/v1/system/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/v1/system/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_returns_timestamp(self, client):
        """Test that health endpoint returns a valid timestamp."""
        response = client.get("/v1/system/health")
        data = response.json()
        assert "timestamp" in data
        # Timestamp should be ISO format ending with Z
        assert data["timestamp"].endswith("Z")

    def test_health_response_matches_schema(self, client):
        """Test that response matches HealthResponse schema."""
        from src.api.schemas import HealthResponse

        response = client.get("/v1/system/health")
        data = response.json()

        # Should be able to parse as HealthResponse
        health = HealthResponse(**data)
        assert health.status == "healthy"


class TestInfoEndpoint:
    """Tests for GET /v1/system/info endpoint."""

    def test_info_returns_200(self, client, mock_system_metrics):
        """Test that info endpoint returns 200 OK."""
        response = client.get("/v1/system/info")
        assert response.status_code == 200

    def test_info_returns_instance_id(self, client, mock_system_metrics):
        """Test that info includes instance_id."""
        response = client.get("/v1/system/info")
        data = response.json()
        assert data["instance_id"] == "inst_test123"

    def test_info_returns_uptime(self, client, mock_system_metrics):
        """Test that info includes uptime_seconds."""
        response = client.get("/v1/system/info")
        data = response.json()
        assert data["uptime_seconds"] == 3600

    def test_info_returns_supported_model_types(self, client, mock_system_metrics):
        """Test that info includes supported model types."""
        response = client.get("/v1/system/info")
        data = response.json()
        assert "llm" in data["supported_model_types"]

    def test_info_returns_inference_server(self, client, mock_system_metrics):
        """Test that info includes inference server info."""
        response = client.get("/v1/system/info")
        data = response.json()
        assert "inference_server" in data
        assert data["inference_server"]["type"] == "vllm"

    def test_info_returns_cpu_info(self, client, mock_system_metrics):
        """Test that info includes CPU information."""
        response = client.get("/v1/system/info")
        data = response.json()
        cpu = data["resources"]["cpu"]
        assert cpu["cores"] == 8
        assert cpu["usage_percent"] == 45.2

    def test_info_returns_memory_info(self, client, mock_system_metrics):
        """Test that info includes memory information."""
        response = client.get("/v1/system/info")
        data = response.json()
        mem = data["resources"]["memory"]
        assert mem["total_gb"] == 64.0
        assert mem["usage_percent"] == 50.8

    def test_info_returns_disk_info(self, client, mock_system_metrics):
        """Test that info includes disk information."""
        response = client.get("/v1/system/info")
        data = response.json()
        disk = data["resources"]["disk"]
        assert disk["total_gb"] == 500.0
        assert disk["available_gb"] == 380.0

    def test_info_returns_gpu_info(self, client, mock_system_metrics):
        """Test that info includes GPU information."""
        response = client.get("/v1/system/info")
        data = response.json()
        gpus = data["resources"]["gpu"]
        assert len(gpus) == 1
        assert gpus[0]["name"] == "NVIDIA A100"
        assert gpus[0]["memory_total_gb"] == 80.0

    def test_info_response_matches_schema(self, client, mock_system_metrics):
        """Test that response matches SystemInfo schema."""
        from src.api.schemas import SystemInfo

        response = client.get("/v1/system/info")
        data = response.json()

        # Should be able to parse as SystemInfo
        info = SystemInfo(**data)
        assert info.instance_id == "inst_test123"
        assert len(info.resources.gpu) == 1

    def test_info_graceful_when_no_gpu(self, client):
        """Test graceful handling when no GPU is present."""
        with (
            patch("src.api.routes.system.psutil") as mock_psutil,
            patch("src.api.routes.system.get_gpu_info") as mock_gpu,
            patch("src.api.routes.system.get_instance_id") as mock_id,
            patch("src.api.routes.system.get_uptime_seconds") as mock_up,
        ):
            # Setup basic mocks
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.cpu_percent.return_value = 10.0

            mock_mem = type("Memory", (), {})()
            mock_mem.total = 16 * 1024**3
            mock_mem.used = 8 * 1024**3
            mock_mem.percent = 50.0
            mock_psutil.virtual_memory.return_value = mock_mem

            mock_disk = type("Disk", (), {})()
            mock_disk.total = 256 * 1024**3
            mock_disk.used = 128 * 1024**3
            mock_disk.free = 128 * 1024**3
            mock_disk.percent = 50.0
            mock_psutil.disk_usage.return_value = mock_disk

            # No GPU available
            mock_gpu.return_value = []
            mock_id.return_value = "inst_nogpu"
            mock_up.return_value = 100

            response = client.get("/v1/system/info")
            assert response.status_code == 200
            data = response.json()
            assert data["resources"]["gpu"] == []
