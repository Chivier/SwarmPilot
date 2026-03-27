"""SDK client for the SwarmPilot API.

Provides :class:`SwarmPilotClient`, a thin async HTTP wrapper around the
Planner and Scheduler REST endpoints.  All heavy lifting (deployment,
scaling, prediction) is delegated to the backend services; this module
only marshals requests and converts responses into SDK dataclasses.
"""

from __future__ import annotations

from typing import Any

import httpx

from swarmpilot.errors import ModelNotDeployed, SchedulerNotFound
from swarmpilot.sdk.models import (
    ClusterState,
    DeploymentResult,
    Instance,
    InstanceGroup,
    ModelStatus,
    PredictResult,
    PreprocessorInfo,
    Process,
    TrainResult,
)


class SwarmPilotClient:
    """Client for the SwarmPilot API.

    Provides methods for deploying models, managing instances,
    and interacting with the predictor.

    The client must be used as an async context manager **or**
    explicitly closed via :meth:`close`::

        async with SwarmPilotClient() as sp:
            group = await sp.serve("llama-7b", gpu=1)

    Args:
        planner_url: Base URL of the Planner service.
        scheduler_url: Base URL of the Scheduler service.  When
            ``None``, predictor-related methods will raise
            :class:`ValueError`.
        timeout: Default HTTP timeout in seconds.
    """

    def __init__(
        self,
        planner_url: str = "http://localhost:8002",
        scheduler_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._planner_url = planner_url.rstrip("/")
        self._scheduler_url = (
            scheduler_url.rstrip("/") if scheduler_url else None
        )
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
        )

    @property
    def planner_url(self) -> str:
        """Return the base URL of the Planner service.

        Returns:
            The planner URL string (without trailing slash).
        """
        return self._planner_url

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> SwarmPilotClient:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager and close the HTTP client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict:
        """Send an HTTP request and return the JSON response.

        Args:
            method: HTTP method (``GET``, ``POST``, etc.).
            url: Fully-qualified URL to call.
            **kwargs: Forwarded to :meth:`httpx.AsyncClient.request`.

        Returns:
            Parsed JSON response body as a dictionary.

        Raises:
            SchedulerNotFound: On 404 when the target is the
                scheduler.
            ModelNotDeployed: On 404 when the target is the
                planner.
            ValueError: On 400 (bad request).
            httpx.HTTPStatusError: On any other non-2xx status.
        """
        response = await self._client.request(method, url, **kwargs)

        if response.status_code == 404:
            # Determine which error to raise based on the URL
            if self._scheduler_url and url.startswith(self._scheduler_url):
                raise SchedulerNotFound(model=url)
            raise ModelNotDeployed(model=url)

        if response.status_code == 400:
            detail = response.text
            raise ValueError(f"Bad request: {detail}")

        # Let httpx raise for any other non-2xx status
        response.raise_for_status()

        return response.json()

    def _planner(self, path: str) -> str:
        """Build a full planner URL for *path*.

        Args:
            path: API path (e.g. ``/v1/serve``).

        Returns:
            Absolute URL string.
        """
        return f"{self._planner_url}{path}"

    def _scheduler(self, path: str) -> str:
        """Build a full scheduler URL for *path*.

        Args:
            path: API path (e.g. ``/v1/predictor/train``).

        Returns:
            Absolute URL string.

        Raises:
            ValueError: If no scheduler URL was configured.
        """
        if self._scheduler_url is None:
            raise ValueError(
                "scheduler_url is required for predictor operations"
            )
        return f"{self._scheduler_url}{path}"

    # ------------------------------------------------------------------
    # Planner endpoints
    # ------------------------------------------------------------------

    async def serve(
        self,
        model_or_command: str,
        *,
        name: str | None = None,
        replicas: int = 1,
        gpu: int = 1,
        scheduler: str | None = "auto",
    ) -> InstanceGroup:
        """Deploy a model service.

        Sends a ``POST /v1/serve`` request to the Planner and
        converts the response into an :class:`InstanceGroup`.

        Args:
            model_or_command: Model name or shell command to serve.
            name: Optional instance-group name.
            replicas: Number of replicas to launch.
            gpu: GPUs required per replica.
            scheduler: Scheduler URL, ``'auto'`` for automatic
                selection, or ``None`` to skip registration.

        Returns:
            An :class:`InstanceGroup` describing the deployed
            replicas.
        """
        payload: dict[str, Any] = {
            "model_or_command": model_or_command,
            "name": name,
            "replicas": replicas,
            "gpu_count": gpu,
            "scheduler": scheduler,
        }
        data = await self._request(
            "POST",
            self._planner("/v1/serve"),
            json=payload,
        )
        return self._parse_instance_group(data)

    async def run(
        self,
        command: str,
        *,
        name: str,
        gpu: int = 0,
    ) -> Process:
        """Start a custom process.

        Sends a ``POST /v1/run`` request to the Planner and
        converts the response into a :class:`Process`.

        Args:
            command: Shell command to execute.
            name: Unique process name.
            gpu: GPUs allocated to the process.

        Returns:
            A :class:`Process` describing the launched workload.
        """
        payload: dict[str, Any] = {
            "command": command,
            "name": name,
            "gpu_count": gpu,
        }
        data = await self._request(
            "POST",
            self._planner("/v1/run"),
            json=payload,
        )
        return Process(
            name=data["name"],
            command=data["command"],
            endpoint=data.get("endpoint"),
            status=data.get("status", "running"),
            gpu=gpu,
            _client=self,
        )

    async def register(
        self,
        model: str,
        *,
        gpu: int = 1,
        replicas: int = 1,
    ) -> None:
        """Register a model for optimized deployment.

        Sends a ``POST /v1/register`` request to the Planner.

        Args:
            model: Model identifier.
            gpu: GPUs required per instance.
            replicas: Desired number of replicas.
        """
        payload: dict[str, Any] = {
            "model": model,
            "gpu_count": gpu,
            "replicas": replicas,
        }
        await self._request(
            "POST",
            self._planner("/v1/register"),
            json=payload,
        )

    async def deploy(self) -> DeploymentResult:
        """Trigger optimized deployment.

        Sends a ``POST /v1/deploy`` request to the Planner and
        converts the response into a :class:`DeploymentResult`.

        Returns:
            A :class:`DeploymentResult` describing the full
            deployment outcome.
        """
        data = await self._request(
            "POST",
            self._planner("/v1/deploy"),
        )
        groups: dict[str, InstanceGroup] = {}
        for model_name, group_data in data.get("groups", {}).items():
            groups[model_name] = self._parse_instance_group(group_data)
        return DeploymentResult(
            plan=data.get("plan", {}),
            groups=groups,
            status=data.get("status", "unknown"),
            _client=self,
        )

    async def instances(self) -> ClusterState:
        """List all instances in the cluster.

        Sends a ``GET /v1/instances`` request to the Planner and
        converts the response into a :class:`ClusterState`.

        Returns:
            A :class:`ClusterState` with current instances,
            processes, and groups.
        """
        data = await self._request(
            "GET",
            self._planner("/v1/instances"),
        )
        # The API may return a flat list of instances or a dict
        # with "instances", "processes", and "groups" keys.
        if isinstance(data, list):
            raw_instances = data
            raw_processes: list = []
            raw_groups: list = []
        else:
            raw_instances = data.get("instances", [])
            raw_processes = data.get("processes", [])
            raw_groups = data.get("groups", [])
        instances_list = [self._parse_instance(inst) for inst in raw_instances]
        processes_list = [
            Process(
                name=p["name"],
                command=p["command"],
                endpoint=p.get("endpoint"),
                status=p["status"],
                gpu=p.get("gpu", 0),
                _client=self,
            )
            for p in raw_processes
        ]
        groups_list = [self._parse_instance_group(g) for g in raw_groups]
        return ClusterState(
            instances=instances_list,
            processes=processes_list,
            groups=groups_list,
        )

    async def scale(self, model: str, replicas: int) -> InstanceGroup:
        """Scale a model to a target replica count.

        Sends a ``POST /v1/scale`` request to the Planner.

        Args:
            model: Model identifier to scale.
            replicas: Desired replica count.

        Returns:
            An :class:`InstanceGroup` reflecting the new state.
        """
        payload: dict[str, Any] = {
            "model": model,
            "replicas": replicas,
        }
        data = await self._request(
            "POST",
            self._planner("/v1/scale"),
            json=payload,
        )
        return self._parse_instance_group(data)

    async def terminate(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        all: bool = False,
    ) -> None:
        """Terminate instances.

        At least one of *name*, *model*, or *all* must be set.

        Sends a ``POST /v1/terminate`` request to the Planner.

        Args:
            name: Terminate a specific instance by name.
            model: Terminate all instances of a given model.
            all: Terminate every managed instance.
        """
        payload: dict[str, Any] = {
            "name": name,
            "model": model,
            "all": all,
        }
        await self._request(
            "POST",
            self._planner("/v1/terminate"),
            json=payload,
        )

    async def schedulers(self) -> dict[str, str]:
        """Get the model-to-scheduler mapping.

        Sends a ``GET /v1/schedulers`` request to the Planner.

        Returns:
            A dictionary mapping model identifiers to scheduler
            URLs.
        """
        data = await self._request(
            "GET",
            self._planner("/v1/schedulers"),
        )
        return data.get("schedulers", {})

    # ------------------------------------------------------------------
    # Scheduler / predictor endpoints
    # ------------------------------------------------------------------

    async def train(
        self,
        model: str,
        *,
        prediction_type: str = "expect_error",
    ) -> TrainResult:
        """Train the predictor for a model.

        Sends a ``POST /v1/predictor/train`` request to the
        Scheduler.

        Args:
            model: Model identifier.
            prediction_type: Prediction strategy to train
                (``'expect_error'`` or ``'quantile'``).

        Returns:
            A :class:`TrainResult` with training outcome.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prediction_type": prediction_type,
        }
        data = await self._request(
            "POST",
            self._scheduler("/v1/predictor/train"),
            json=payload,
        )
        return TrainResult(
            model=data.get("model", model),
            samples_trained=data.get("samples_trained", 0),
            metrics=data.get("metrics", {}),
            strategy=data.get("strategy", prediction_type),
        )

    async def predict(
        self,
        model: str,
        *,
        features: dict,
        prediction_type: str = "expect_error",
    ) -> PredictResult:
        """Run a manual prediction.

        Sends a ``POST /v1/predictor/predict`` request to the
        Scheduler.

        Args:
            model: Model identifier.
            features: Feature dictionary for the prediction.
            prediction_type: Prediction strategy to use.

        Returns:
            A :class:`PredictResult` with predicted values.
        """
        payload: dict[str, Any] = {
            "model": model,
            "features": features,
            "prediction_type": prediction_type,
        }
        data = await self._request(
            "POST",
            self._scheduler("/v1/predictor/predict"),
            json=payload,
        )
        return PredictResult(
            model=data.get("model", model),
            expected_runtime_ms=data.get("expected_runtime_ms"),
            error_margin_ms=data.get("error_margin_ms"),
            quantiles=data.get("quantiles"),
        )

    async def predictor_status(self, model: str) -> ModelStatus:
        """Get predictor status for a model.

        Sends a ``GET /v1/predictor/status/{model}`` request to
        the Scheduler.

        Args:
            model: Model identifier.

        Returns:
            A :class:`ModelStatus` with training and prediction
            metadata.
        """
        data = await self._request(
            "GET",
            self._scheduler(f"/v1/predictor/status/{model}"),
        )
        preprocessors = [
            PreprocessorInfo(
                name=p["name"],
                feature=p["feature"],
                path=p["path"],
            )
            for p in data.get("preprocessors", [])
        ]
        return ModelStatus(
            model=data.get("model", model),
            samples_collected=data.get("samples_collected", 0),
            last_trained=data.get("last_trained"),
            prediction_types=data.get("prediction_types", []),
            metrics=data.get("metrics", {}),
            strategy=data.get("strategy", "expect_error"),
            preprocessors=preprocessors,
        )

    # ------------------------------------------------------------------
    # Private conversion helpers
    # ------------------------------------------------------------------

    def _parse_instance(self, data: dict) -> Instance:
        """Convert a raw JSON dict to an :class:`Instance`.

        Args:
            data: Dictionary with instance fields.

        Returns:
            An :class:`Instance` dataclass with ``_client`` set.
        """
        return Instance(
            name=data.get("name") or data.get("instance_id", ""),
            model=data.get("model") or data.get("model_id"),
            command=data.get("command", ""),
            endpoint=data.get("endpoint"),
            scheduler=data.get("scheduler"),
            status=data.get("status", "unknown"),
            gpu=data.get("gpu") or data.get("gpu_count", 0),
            _client=self,
        )

    def _parse_instance_group(self, data: dict) -> InstanceGroup:
        """Convert a raw JSON dict to an :class:`InstanceGroup`.

        Handles both the ``replicas`` key (from ``ServeResponse``)
        and the ``instances`` key (from cluster-state listings).

        Args:
            data: Dictionary with instance-group fields.

        Returns:
            An :class:`InstanceGroup` dataclass with ``_client``
            set.
        """
        # Several response formats co-exist:
        #   - "replicas" as list[dict] (legacy SDK convention in tests)
        #   - "replicas" as int + "instances" as list[str] (ServeResponse)
        #   - "instances" as list[dict] (cluster-state listings)
        # Resolve to a list of dicts or strings.
        raw_replicas = data.get("replicas", data.get("instances", []))
        if isinstance(raw_replicas, int):
            # replicas is a count; use instances list instead
            raw_replicas = data.get("instances", [])

        parsed: list[Instance] = []
        for r in raw_replicas:
            if isinstance(r, str):
                # ServeResponse: instance IDs as strings
                parsed.append(
                    Instance(
                        name=r,
                        model=data.get("model"),
                        command=data.get("command", ""),
                        endpoint=None,
                        scheduler=data.get("scheduler_url"),
                        status="pending",
                        gpu=data.get("gpu_count", 0),
                        _client=self,
                    )
                )
            elif isinstance(r, dict) and "name" in r:
                parsed.append(self._parse_instance(r))
            elif isinstance(r, dict):
                parsed.append(
                    Instance(
                        name=r.get("name", ""),
                        model=data.get("model"),
                        command=data.get("command", ""),
                        endpoint=r.get("endpoint"),
                        scheduler=data.get("scheduler"),
                        status=r.get("status", "unknown"),
                        gpu=r.get("gpu", 0),
                        _client=self,
                    )
                )
        return InstanceGroup(
            name=data.get("name", ""),
            model=data.get("model"),
            command=data.get("command", ""),
            instances=parsed,
            scheduler=data.get("scheduler"),
            _client=self,
        )
