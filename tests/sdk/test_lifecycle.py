"""Tests for SDK model lifecycle methods.

Validates that Instance, InstanceGroup, Process, and
DeploymentResult lifecycle methods (wait_ready, terminate, scale)
correctly call the planner API and handle error cases.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from swarmpilot.errors import (
    DeployError,
    SwarmPilotError,
    SwarmPilotTimeoutError,
)
from swarmpilot.sdk.client import SwarmPilotClient
from swarmpilot.sdk.models import (
    Instance,
    InstanceGroup,
    Process,
)

PLANNER = "http://planner:8002"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def client() -> SwarmPilotClient:
    """Create a client pointed at test planner URL."""
    return SwarmPilotClient(
        planner_url=PLANNER,
        timeout=5.0,
    )


def _make_instance(
    name: str = "worker-0",
    status: str = "pending",
    client: SwarmPilotClient | None = None,
) -> Instance:
    """Build an Instance with common defaults.

    Args:
        name: Instance name.
        status: Instance status.
        client: Optional client reference.

    Returns:
        An Instance dataclass.
    """
    return Instance(
        name=name,
        model="llama-7b",
        command="vllm serve llama-7b",
        endpoint=None,
        scheduler=None,
        status=status,
        gpu=1,
        _client=client,
    )


# ------------------------------------------------------------------
# Instance.wait_ready()
# ------------------------------------------------------------------


class TestInstanceWaitReady:
    """Instance.wait_ready() polls until status is ready."""

    @respx.mock
    async def test_polls_until_running(
        self, client: SwarmPilotClient
    ) -> None:
        """wait_ready returns once status becomes 'running'."""
        # First call returns pending, second returns running.
        respx.get(
            f"{PLANNER}/v1/instances/worker-0"
        ).mock(
            side_effect=[
                httpx.Response(
                    200, json={"status": "pending"}
                ),
                httpx.Response(
                    200, json={"status": "running"}
                ),
            ]
        )

        inst = _make_instance(client=client)

        with patch(
            "swarmpilot.sdk.models.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await inst.wait_ready(timeout=300)

        assert inst.status == "running"

    @respx.mock
    async def test_accepts_active_status(
        self, client: SwarmPilotClient
    ) -> None:
        """wait_ready returns for 'active' status too."""
        respx.get(
            f"{PLANNER}/v1/instances/worker-0"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "active"}
            )
        )

        inst = _make_instance(client=client)

        with patch(
            "swarmpilot.sdk.models.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await inst.wait_ready(timeout=300)

        assert inst.status == "active"

    @respx.mock
    async def test_timeout_raises_swarm_pilot_timeout_error(
        self, client: SwarmPilotClient
    ) -> None:
        """wait_ready raises SwarmPilotTimeoutError on timeout."""
        respx.get(
            f"{PLANNER}/v1/instances/worker-0"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "pending"}
            )
        )

        inst = _make_instance(client=client)

        # Simulate time moving past the deadline by
        # advancing event loop time.
        call_count = 0

        def fast_time() -> float:
            """Return increasing time to force timeout."""
            nonlocal call_count
            call_count += 1
            # First call sets the deadline; subsequent calls
            # exceed it after a few iterations.
            return call_count * 100.0

        loop = asyncio.get_event_loop()
        with (
            patch.object(
                loop,
                "time",
                side_effect=fast_time,
            ),
            patch(
                "swarmpilot.sdk.models.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(SwarmPilotTimeoutError),
        ):
            await inst.wait_ready(timeout=10)

    @respx.mock
    async def test_failed_status_raises_deploy_error(
        self, client: SwarmPilotClient
    ) -> None:
        """wait_ready raises DeployError on 'failed' status."""
        respx.get(
            f"{PLANNER}/v1/instances/worker-0"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "failed"}
            )
        )

        inst = _make_instance(client=client)

        with (
            patch(
                "swarmpilot.sdk.models.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(
                DeployError, match="worker-0"
            ),
        ):
            await inst.wait_ready(timeout=300)


# ------------------------------------------------------------------
# Instance.terminate()
# ------------------------------------------------------------------


class TestInstanceTerminate:
    """Instance.terminate() calls POST /v1/terminate."""

    @respx.mock
    async def test_terminate_calls_correct_endpoint(
        self, client: SwarmPilotClient
    ) -> None:
        """Terminate sends name in the payload."""
        route = respx.post(
            f"{PLANNER}/v1/terminate"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "terminated"}
            )
        )

        inst = _make_instance(client=client)
        await inst.terminate()

        assert route.called
        payload = route.calls[0].request.content
        import json

        body = json.loads(payload)
        assert body == {"name": "worker-0"}


# ------------------------------------------------------------------
# Instance methods require _client
# ------------------------------------------------------------------


class TestInstanceRequiresClient:
    """Instance methods raise SwarmPilotError without _client."""

    async def test_wait_ready_raises_without_client(self) -> None:
        """wait_ready raises when _client is None."""
        inst = _make_instance(client=None)

        with pytest.raises(SwarmPilotError, match="No client"):
            await inst.wait_ready()

    async def test_terminate_raises_without_client(self) -> None:
        """Terminate raises when _client is None."""
        inst = _make_instance(client=None)

        with pytest.raises(SwarmPilotError, match="No client"):
            await inst.terminate()


# ------------------------------------------------------------------
# InstanceGroup.wait_ready()
# ------------------------------------------------------------------


class TestInstanceGroupWaitReady:
    """InstanceGroup.wait_ready() delegates to instances."""

    @respx.mock
    async def test_delegates_to_each_instance(
        self, client: SwarmPilotClient
    ) -> None:
        """wait_ready calls each instance's wait_ready."""
        # Both instances return running immediately.
        respx.get(
            f"{PLANNER}/v1/instances/w-0"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "running"}
            )
        )
        respx.get(
            f"{PLANNER}/v1/instances/w-1"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "running"}
            )
        )

        instances = [
            _make_instance(name="w-0", client=client),
            _make_instance(name="w-1", client=client),
        ]
        group = InstanceGroup(
            name="llama-group",
            model="llama-7b",
            command="vllm serve llama-7b",
            instances=instances,
            _client=client,
        )

        with patch(
            "swarmpilot.sdk.models.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await group.wait_ready(timeout=300)

        assert instances[0].status == "running"
        assert instances[1].status == "running"


# ------------------------------------------------------------------
# InstanceGroup.terminate()
# ------------------------------------------------------------------


class TestInstanceGroupTerminate:
    """InstanceGroup.terminate() calls POST /v1/terminate."""

    @respx.mock
    async def test_terminate_sends_model(
        self, client: SwarmPilotClient
    ) -> None:
        """Terminate sends model in the payload."""
        route = respx.post(
            f"{PLANNER}/v1/terminate"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "terminated"}
            )
        )

        group = InstanceGroup(
            name="llama-group",
            model="llama-7b",
            command="vllm serve llama-7b",
            _client=client,
        )
        await group.terminate()

        assert route.called
        import json

        body = json.loads(
            route.calls[0].request.content
        )
        assert body == {"model": "llama-7b"}


# ------------------------------------------------------------------
# InstanceGroup.scale()
# ------------------------------------------------------------------


class TestInstanceGroupScale:
    """InstanceGroup.scale() calls POST /v1/scale."""

    @respx.mock
    async def test_scale_sends_model_and_replicas(
        self, client: SwarmPilotClient
    ) -> None:
        """Scale sends model and replicas in the payload."""
        route = respx.post(f"{PLANNER}/v1/scale").mock(
            return_value=httpx.Response(
                200, json={"status": "scaled"}
            )
        )

        group = InstanceGroup(
            name="llama-group",
            model="llama-7b",
            command="vllm serve llama-7b",
            _client=client,
        )
        await group.scale(replicas=3)

        assert route.called
        import json

        body = json.loads(
            route.calls[0].request.content
        )
        assert body == {
            "model": "llama-7b",
            "replicas": 3,
        }


# ------------------------------------------------------------------
# Process.terminate()
# ------------------------------------------------------------------


class TestProcessTerminate:
    """Process.terminate() calls POST /v1/terminate."""

    @respx.mock
    async def test_terminate_sends_name(
        self, client: SwarmPilotClient
    ) -> None:
        """Terminate sends process name in the payload."""
        route = respx.post(
            f"{PLANNER}/v1/terminate"
        ).mock(
            return_value=httpx.Response(
                200, json={"status": "terminated"}
            )
        )

        proc = Process(
            name="etl-job",
            command="python etl.py",
            endpoint=None,
            status="running",
            gpu=0,
            _client=client,
        )
        await proc.terminate()

        assert route.called
        import json

        body = json.loads(
            route.calls[0].request.content
        )
        assert body == {"name": "etl-job"}


# ------------------------------------------------------------------
# Client injection: serve() returns objects with _client set
# ------------------------------------------------------------------


class TestClientInjection:
    """Client methods inject _client into returned objects."""

    @respx.mock
    async def test_serve_sets_client_on_group(
        self, client: SwarmPilotClient
    ) -> None:
        """serve() returns InstanceGroup with _client set."""
        respx.post(f"{PLANNER}/v1/serve").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "llama-group",
                    "model": "llama-7b",
                    "command": "vllm serve llama-7b",
                    "replicas": [
                        {
                            "name": "w-0",
                            "endpoint": "http://a:8080",
                            "status": "running",
                        }
                    ],
                    "scheduler": None,
                    "status": "deployed",
                },
            )
        )

        group = await client.serve("llama-7b", gpu=1)

        assert group._client is client
        assert group.instances[0]._client is client

    @respx.mock
    async def test_run_sets_client_on_process(
        self, client: SwarmPilotClient
    ) -> None:
        """run() returns Process with _client set."""
        respx.post(f"{PLANNER}/v1/run").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "etl-job",
                    "command": "python etl.py",
                    "endpoint": None,
                    "status": "running",
                },
            )
        )

        proc = await client.run(
            "python etl.py", name="etl-job", gpu=0
        )

        assert proc._client is client

    @respx.mock
    async def test_deploy_sets_client_on_result(
        self, client: SwarmPilotClient
    ) -> None:
        """deploy() returns DeploymentResult with _client set."""
        respx.post(f"{PLANNER}/v1/deploy").mock(
            return_value=httpx.Response(
                200,
                json={
                    "plan": {"llama-7b": 1},
                    "groups": {
                        "llama-7b": {
                            "name": "llama-group",
                            "model": "llama-7b",
                            "command": "vllm serve",
                            "replicas": [
                                {
                                    "name": "w-0",
                                    "endpoint": "http://a:8080",
                                    "status": "running",
                                }
                            ],
                            "scheduler": None,
                        }
                    },
                    "status": "deployed",
                },
            )
        )

        result = await client.deploy()

        assert result._client is client
        group = result["llama-7b"]
        assert group._client is client
        assert group.instances[0]._client is client
