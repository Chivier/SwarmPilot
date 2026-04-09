"""Integration tests for online API endpoint scheduling.

Tests the end-to-end flow of registering online API endpoints,
routing requests through the scheduler, and collecting training
samples — using mock HTTP backends.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from swarmpilot.common.online_endpoint import (
    platform_info_from_online_endpoint,
)
from swarmpilot.scheduler.models.core import Instance
from swarmpilot.scheduler.models.status import InstanceStatus
from swarmpilot.scheduler.registry.instance_registry import (
    InstanceRegistry,
)
from swarmpilot.scheduler.services.worker_queue_manager import (
    WorkerQueueManager,
)


@pytest.mark.integration
class TestOnlineEndpointRegistration:
    """Tests for registering online endpoints as instances."""

    @pytest.fixture
    def instance_registry(self):
        """Fresh instance registry."""
        return InstanceRegistry()

    @pytest.fixture
    def mock_callback(self):
        """Mock result callback."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_callback):
        """WorkerQueueManager with short timeouts."""
        return WorkerQueueManager(
            callback=mock_callback,
            http_timeout=5.0,
            max_retries=1,
            retry_delay=0.01,
        )

    async def test_register_two_online_endpoints(
        self, instance_registry, manager
    ):
        """Two endpoints for the same model register correctly."""
        url = "https://api.anthropic.com"
        key1, key2 = "sk-key-tier-1", "sk-key-tier-2"

        for i, key in enumerate([key1, key2], start=1):
            instance_id = f"online-claude-key-{i}"
            pinfo = platform_info_from_online_endpoint(url, key)
            instance = Instance(
                instance_id=instance_id,
                model_id="claude-sonnet",
                endpoint=url,
                platform_info=pinfo,
                status=InstanceStatus.ACTIVE,
            )
            await instance_registry.register(instance)
            manager.register_worker(
                worker_id=instance_id,
                worker_endpoint=url,
                model_id="claude-sonnet",
                extra_headers={
                    "Authorization": f"Bearer {key}",
                },
            )

        # Both registered
        assert manager.get_worker_count() == 2
        assert manager.has_worker("online-claude-key-1")
        assert manager.has_worker("online-claude-key-2")

        # Both active in registry
        active = await instance_registry.list_active()
        assert len(active) == 2

        # Different platform_info (different keys)
        i1 = await instance_registry.get("online-claude-key-1")
        i2 = await instance_registry.get("online-claude-key-2")
        assert (
            i1.platform_info["software_version"]
            != i2.platform_info["software_version"]
        )
        # Same provider
        assert (
            i1.platform_info["software_name"]
            == i2.platform_info["software_name"]
        )

        manager.shutdown()

    async def test_online_endpoint_status_is_active(self, instance_registry):
        """Online endpoints start as ACTIVE (no initialization)."""
        pinfo = platform_info_from_online_endpoint(
            "https://api.openai.com/v1", "sk-openai-key"
        )
        instance = Instance(
            instance_id="online-openai",
            model_id="gpt-4o",
            endpoint="https://api.openai.com/v1",
            platform_info=pinfo,
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry.register(instance)

        retrieved = await instance_registry.get("online-openai")
        assert retrieved.status == InstanceStatus.ACTIVE

    async def test_online_endpoint_hardware_name_is_cloud(
        self, instance_registry
    ):
        """Online endpoint platform_info has hardware_name=cloud."""
        pinfo = platform_info_from_online_endpoint(
            "https://api.anthropic.com", "sk-key"
        )
        instance = Instance(
            instance_id="online-test",
            model_id="test-model",
            endpoint="https://api.anthropic.com",
            platform_info=pinfo,
            status=InstanceStatus.ACTIVE,
        )
        await instance_registry.register(instance)

        retrieved = await instance_registry.get("online-test")
        assert retrieved.platform_info["hardware_name"] == "cloud"


@pytest.mark.integration
class TestOnlineEndpointExtraHeaders:
    """Tests that auth headers are injected into requests."""

    @pytest.fixture
    def mock_callback(self):
        """Mock result callback."""
        return MagicMock()

    def test_extra_headers_stored_on_thread(self, mock_callback):
        """WorkerQueueThread stores extra_headers for merging."""
        manager = WorkerQueueManager(
            callback=mock_callback,
            http_timeout=5.0,
            max_retries=1,
            retry_delay=0.01,
        )

        auth_headers = {
            "Authorization": "Bearer sk-ant-test-key",
            "Content-Type": "application/json",
        }

        manager.register_worker(
            worker_id="online-claude",
            worker_endpoint="https://api.anthropic.com",
            model_id="claude-sonnet",
            extra_headers=auth_headers,
        )

        thread = manager.get_worker("online-claude")
        assert thread._extra_headers == auth_headers

        manager.shutdown()

    def test_self_hosted_no_extra_headers(self, mock_callback):
        """Self-hosted workers default to empty extra_headers."""
        manager = WorkerQueueManager(
            callback=mock_callback,
            http_timeout=5.0,
        )

        manager.register_worker(
            worker_id="vllm-worker-1",
            worker_endpoint="http://10.0.0.2:8001",
            model_id="qwen-8b",
        )

        thread = manager.get_worker("vllm-worker-1")
        assert thread._extra_headers == {}

        manager.shutdown()
