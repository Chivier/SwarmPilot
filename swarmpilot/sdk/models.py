"""Dataclass-based data models for the SwarmPilot SDK.

These models represent the core domain objects exchanged between
the SDK client layer and the SwarmPilot microservices (Scheduler,
Predictor, Planner).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from swarmpilot.errors import (
    DeployError,
    SwarmPilotError,
    SwarmPilotTimeoutError,
)

if TYPE_CHECKING:
    from swarmpilot.sdk.client import SwarmPilotClient


@dataclass
class Instance:
    """A single deployed instance managed by the scheduler.

    Attributes:
        name: Unique instance identifier.
        model: Model served by this instance, or ``None`` for
            generic workloads.
        command: Shell command used to launch the instance.
        endpoint: HTTP endpoint once the instance is ready, or
            ``None`` while still pending.
        scheduler: URL of the scheduler this instance is
            registered with, or ``None`` if unregistered.
        status: Current lifecycle status (e.g. ``"running"``,
            ``"pending"``).
        gpu: Number of GPUs allocated to this instance.
    """

    name: str
    model: str | None
    command: str
    endpoint: str | None
    scheduler: str | None
    status: str
    gpu: int
    _client: SwarmPilotClient | None = field(
        default=None, repr=False, compare=False
    )

    def _require_client(self) -> SwarmPilotClient:
        """Return the attached client or raise.

        Returns:
            The :class:`SwarmPilotClient` bound to this instance.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        if self._client is None:
            raise SwarmPilotError(
                "No client attached to this Instance. "
                "Use SwarmPilotClient to create instances."
            )
        return self._client

    async def wait_ready(self, timeout: int = 300) -> None:
        """Poll until the instance is running or active.

        Polls ``GET /v1/instances/{name}`` every 2 seconds until the
        status becomes ``"running"`` or ``"active"``.

        Args:
            timeout: Maximum seconds to wait before giving up.

        Raises:
            SwarmPilotTimeoutError: If *timeout* elapses without
                reaching a ready state.
            DeployError: If the instance enters ``"failed"`` status.
            SwarmPilotError: If no client is attached.
        """
        client = self._require_client()
        url = (
            f"{client.planner_url}"
            f"/v1/instances/{self.name}"
        )
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            resp = await client._client.get(url)
            data = resp.json()
            status = data.get("status", "unknown")
            if status in ("running", "active"):
                self.status = status
                self.endpoint = data.get("endpoint", self.endpoint)
                return
            if status == "failed":
                raise DeployError(
                    f"Instance '{self.name}' failed"
                )
            await asyncio.sleep(2)
        raise SwarmPilotTimeoutError(
            timeout=timeout, name=self.name
        )

    async def terminate(self) -> None:
        """Terminate this instance.

        Sends ``POST /v1/terminate`` with ``{"name": self.name}``
        to the planner.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        client = self._require_client()
        url = f"{client.planner_url}/v1/terminate"
        await client._client.post(
            url, json={"name": self.name}
        )


@dataclass
class InstanceGroup:
    """A group of replicas serving the same model/command.

    Attributes:
        name: Logical group name.
        model: Model served by every replica in the group, or
            ``None`` for generic workloads.
        command: Shell command shared by all replicas.
        instances: Individual :class:`Instance` objects that
            belong to this group.
        scheduler: URL of the scheduler managing the group, or
            ``None`` if unregistered.
    """

    name: str
    model: str | None
    command: str
    instances: list[Instance] = field(default_factory=list)
    scheduler: str | None = None
    _client: SwarmPilotClient | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def endpoints(self) -> list[str]:
        """Return non-``None`` endpoints from all instances.

        Returns:
            List of endpoint URLs for instances that have an
            endpoint assigned.
        """
        return [
            inst.endpoint
            for inst in self.instances
            if inst.endpoint is not None
        ]

    def _require_client(self) -> SwarmPilotClient:
        """Return the attached client or raise.

        Returns:
            The :class:`SwarmPilotClient` bound to this group.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        if self._client is None:
            raise SwarmPilotError(
                "No client attached to this InstanceGroup. "
                "Use SwarmPilotClient to create groups."
            )
        return self._client

    async def wait_ready(self, timeout: int = 300) -> None:
        """Wait until every instance in the group is ready.

        Delegates to each instance's :meth:`Instance.wait_ready`.

        Args:
            timeout: Maximum seconds to wait **per instance**.

        Raises:
            SwarmPilotTimeoutError: If any instance times out.
            DeployError: If any instance enters ``"failed"``
                status.
            SwarmPilotError: If no client is attached.
        """
        self._require_client()
        for inst in self.instances:
            await inst.wait_ready(timeout=timeout)

    async def scale(self, replicas: int) -> None:
        """Scale this group to a target replica count.

        Sends ``POST /v1/scale`` with the group's model and the
        desired *replicas* count.

        Args:
            replicas: Desired number of replicas.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        client = self._require_client()
        url = f"{client.planner_url}/v1/scale"
        await client._client.post(
            url,
            json={"model": self.model, "replicas": replicas},
        )

    async def terminate(self) -> None:
        """Terminate all instances in this group.

        Sends ``POST /v1/terminate`` with
        ``{"model": self.model}``.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        client = self._require_client()
        url = f"{client.planner_url}/v1/terminate"
        await client._client.post(
            url, json={"model": self.model}
        )


@dataclass
class Process:
    """A custom workload that runs without a scheduler.

    Attributes:
        name: Unique process identifier.
        command: Shell command used to launch the process.
        endpoint: HTTP endpoint once the process is ready, or
            ``None`` while still pending.
        status: Current lifecycle status (e.g. ``"running"``,
            ``"pending"``).
        gpu: Number of GPUs allocated to this process.
        scheduler: Always ``None``; processes are unmanaged.
    """

    name: str
    command: str
    endpoint: str | None
    status: str
    gpu: int
    scheduler: None = field(default=None, init=False)
    _client: SwarmPilotClient | None = field(
        default=None, repr=False, compare=False
    )

    def _require_client(self) -> SwarmPilotClient:
        """Return the attached client or raise.

        Returns:
            The :class:`SwarmPilotClient` bound to this process.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        if self._client is None:
            raise SwarmPilotError(
                "No client attached to this Process. "
                "Use SwarmPilotClient to create processes."
            )
        return self._client

    async def terminate(self) -> None:
        """Terminate this process.

        Sends ``POST /v1/terminate`` with ``{"name": self.name}``
        to the planner.

        Raises:
            SwarmPilotError: If no client is attached.
        """
        client = self._require_client()
        url = f"{client.planner_url}/v1/terminate"
        await client._client.post(
            url, json={"name": self.name}
        )


@dataclass
class DeploymentResult:
    """Result of an optimized :func:`deploy` call.

    Supports dict-like access via ``result["model_name"]`` to
    retrieve the :class:`InstanceGroup` for a given model.

    Attributes:
        plan: Raw optimization plan returned by the planner.
        groups: Mapping from model name to :class:`InstanceGroup`.
        status: Overall deployment status string.
    """

    plan: dict
    groups: dict[str, InstanceGroup]
    status: str
    _client: SwarmPilotClient | None = field(
        default=None, repr=False, compare=False
    )

    def __getitem__(self, model: str) -> InstanceGroup:
        """Return the :class:`InstanceGroup` for *model*.

        Args:
            model: Model name to look up.

        Returns:
            The corresponding :class:`InstanceGroup`.

        Raises:
            KeyError: If *model* is not present in ``groups``.
        """
        return self.groups[model]

    async def wait_ready(self, timeout: int = 600) -> None:
        """Wait until every group in the deployment is ready.

        Delegates to each group's :meth:`InstanceGroup.wait_ready`.

        Args:
            timeout: Maximum seconds to wait **per group**.

        Raises:
            SwarmPilotTimeoutError: If any group times out.
            DeployError: If any instance enters ``"failed"``
                status.
            SwarmPilotError: If no client is attached.
        """
        if self._client is None:
            raise SwarmPilotError(
                "No client attached to this DeploymentResult. "
                "Use SwarmPilotClient to create deployments."
            )
        for group in self.groups.values():
            await group.wait_ready(timeout=timeout)


@dataclass
class ClusterState:
    """Current state of all instances and processes in the cluster.

    Attributes:
        instances: All individual instances across all groups.
        processes: All standalone processes.
        groups: Logical instance groups.
    """

    instances: list[Instance] = field(default_factory=list)
    processes: list[Process] = field(default_factory=list)
    groups: list[InstanceGroup] = field(default_factory=list)


@dataclass
class PreprocessorInfo:
    """Metadata about a feature preprocessor registered with the predictor.

    Attributes:
        name: Preprocessor identifier.
        feature: Name of the feature this preprocessor handles.
        path: Filesystem path to the serialized preprocessor.
    """

    name: str
    feature: str
    path: str


@dataclass
class ModelStatus:
    """Predictor training status for a specific model.

    Attributes:
        model: Model identifier.
        samples_collected: Number of samples collected so far.
        last_trained: ISO-8601 timestamp of the last training
            run, or ``None`` if never trained.
        prediction_types: Available prediction types (e.g.
            ``["expect_error", "quantile"]``).
        metrics: Evaluation metrics from the last training run.
        strategy: Active prediction strategy name.
        preprocessors: List of registered preprocessors.
    """

    model: str
    samples_collected: int
    last_trained: str | None
    prediction_types: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    strategy: str = "expect_error"
    preprocessors: list[PreprocessorInfo] = field(
        default_factory=list
    )


@dataclass
class TrainResult:
    """Result of a training or retrain operation.

    Attributes:
        model: Model identifier that was trained.
        samples_trained: Number of samples used in training.
        metrics: Evaluation metrics from the training run.
        strategy: Prediction strategy used for training.
    """

    model: str
    samples_trained: int
    metrics: dict = field(default_factory=dict)
    strategy: str = "expect_error"


@dataclass
class PredictResult:
    """Result of a manual prediction request.

    Depending on the prediction strategy, either the
    ``expected_runtime_ms`` / ``error_margin_ms`` pair or the
    ``quantiles`` dict will be populated (the other set to
    ``None``).

    Attributes:
        model: Model identifier the prediction was made for.
        expected_runtime_ms: Point estimate in milliseconds, or
            ``None`` when using quantile mode.
        error_margin_ms: Error margin in milliseconds, or
            ``None`` when using quantile mode.
        quantiles: Mapping from quantile level to predicted
            runtime in milliseconds, or ``None`` when using
            expect-error mode.
    """

    model: str
    expected_runtime_ms: float | None = None
    error_margin_ms: float | None = None
    quantiles: dict[float, float] | None = None
