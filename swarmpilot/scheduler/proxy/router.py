"""Catch-all transparent proxy router.

This module implements the transparent proxy that forwards any unmatched
request to a backend instance, using the scheduling algorithm to select
which instance receives the request.

The proxy is mounted AFTER all scheduler-internal routes so that
/health, /instance/*, /task/*, /strategy/*, /callback/* take priority.

Flow:
1. Create task record + register Future
2. Run scheduling algorithm to select instance
3. Enqueue task to WorkerQueueManager
4. Await Future (with timeout)
5. Return backend's response transparently
"""

import asyncio
import json
import time
import uuid
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask

from swarmpilot.api import Translation, TranslationError, plan_request
from swarmpilot.api.base import DEFAULT_MAX_OUTPUT_TOKENS
from swarmpilot.api.registry import style_for_path
from swarmpilot.scheduler.services.async_worker_queue import StreamStart
from swarmpilot.scheduler.services.worker_queue_manager import (
    resolve_adapter,
)
from swarmpilot.scheduler.services.worker_queue_thread import QueuedTask
from swarmpilot.scheduler.shadow import ShadowMirror
from swarmpilot.scheduler.utils.arrival_counter import (
    PARENT_HEADER,
    PARENT_KEY,
    arrival_counter,
)

if TYPE_CHECKING:
    from swarmpilot.scheduler.algorithms.base import SchedulingStrategy
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
    from swarmpilot.scheduler.registry.task_registry import TaskRegistry
    from swarmpilot.scheduler.services.task_result_callback import (
        TaskResultCallback,
    )
    from swarmpilot.scheduler.services.worker_queue_manager import (
        WorkerQueueManager,
    )


# Lightweight paths that should be prioritized when the engine's adapter
# does not declare them.
# These are fast metadata queries that should not be blocked by
# long-running inference requests in the worker queue.
PRIORITY_PATHS: set[str] = {
    "v1/models",
    "health",
    "ping",
    "version",
}


def _extract_features(
    platform_info: dict[str, str], path: str, body: dict[str, Any]
) -> dict[str, float]:
    """Return the adapter's predictor features for one request.

    Args:
        platform_info: Platform info of the instances serving the model.
        path: Request path without the leading slash.
        body: Parsed JSON body.

    Returns:
        Feature name to value, or nothing if the body is malformed.
    """
    adapter = resolve_adapter(platform_info.get("software_name"))
    try:
        return adapter.features(path, body)
    except (TypeError, ValueError, AttributeError) as e:
        logger.debug(f"[PROXY] No features for /{path}: {e}")
        return {}


def _is_lightweight(platform_info: dict[str, str], path: str) -> bool:
    """Whether ``path`` is a metadata route that skips the queue.

    The engine's adapter decides for routes it declares; other paths fall
    back to :data:`PRIORITY_PATHS`.
    """
    route = resolve_adapter(platform_info.get("software_name")).route(path)
    if route is not None:
        return route.lightweight
    return path in PRIORITY_PATHS


def _prediction_fields(prediction: Any) -> dict[str, Any]:
    """Task-record prediction fields from a scheduling prediction."""
    if prediction is None:
        return {}
    return {
        "predicted_time_ms": prediction.predicted_time_ms,
        "predicted_error_margin_ms": prediction.error_margin_ms,
        "predicted_quantiles": prediction.quantiles,
    }


class ProxyRouter:
    """Transparent proxy router that forwards requests to backend instances.

    This class encapsulates the proxy logic and provides an APIRouter
    that can be mounted on the main FastAPI application.

    Attributes:
        router: FastAPI APIRouter with the catch-all route.
    """

    def __init__(
        self,
        scheduling_strategy: "SchedulingStrategy",
        instance_registry: "InstanceRegistry",
        task_registry: "TaskRegistry",
        task_result_callback: "TaskResultCallback",
        worker_queue_manager: "WorkerQueueManager",
        proxy_timeout: float = 300.0,
    ):
        """Initialize the proxy router.

        Args:
            scheduling_strategy: Algorithm for selecting instances.
            instance_registry: Registry of available instances.
            task_registry: Registry for task records.
            task_result_callback: Callback handler with Future pool.
            worker_queue_manager: Manager for per-instance worker queues.
            proxy_timeout: Timeout in seconds for proxy requests.
        """
        self._scheduling_strategy = scheduling_strategy
        self._instance_registry = instance_registry
        self._task_registry = task_registry
        self._callback = task_result_callback
        self._queue_manager = worker_queue_manager
        self._proxy_timeout = proxy_timeout
        # Set at startup when SCHEDULER_GATEWAY_SHADOW_URL is.
        self.shadow: ShadowMirror | None = None

        self.router = APIRouter()
        self.router.add_api_route(
            "/{path:path}",
            self._proxy_handler,
            methods=[
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "OPTIONS",
                "HEAD",
            ],
        )

    async def _proxy_handler(self, path: str, request: Request) -> Response:
        """Handle proxied requests.

        Args:
            path: The request path to forward.
            request: The incoming FastAPI request.

        Returns:
            Response from the backend instance.
        """
        method = request.method
        task_id = f"proxy-{uuid.uuid4().hex[:12]}"
        tr: Translation | None = None

        logger.debug(f"[PROXY] {method} /{path} -> task_id={task_id}")

        # 1. Check for available instances
        # Get all active instances (proxy doesn't filter by model_id)
        all_instances = await self._instance_registry.list_active()
        if not all_instances:
            logger.warning(
                f"[PROXY] No instances available for {method} /{path}"
            )
            # Still an arrival: the Planner sizes the model by its
            # arrivals, and one it never sees again it never brings back.
            arrival_counter.record(request.headers.get(PARENT_HEADER))
            return _error_response(
                path,
                503,
                "No backend instances available",
                "service_unavailable",
                tr,
            )

        # 2. Parse request body
        body: dict[str, Any] = {}
        if method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                # Non-JSON body or empty body
                body = {}

        # 2b. A route of a known API style the engine does not speak is
        #     translated; one it speaks passes through unchanged.
        adapter = resolve_adapter(
            all_instances[0].platform_info.get("software_name")
        )
        if method == "POST":
            try:
                tr = plan_request(
                    path,
                    body,
                    adapter.native_styles(),
                    default_max_tokens=getattr(
                        adapter, "default_max_tokens", DEFAULT_MAX_OUTPUT_TOKENS
                    ),
                )
            except TranslationError as e:
                return _error_response(
                    path, e.status, e.message, e.error_type, param=e.param
                )
        client_body = body
        client_path = path
        if tr is not None and not tr.passthrough:
            path, body = tr.path, tr.body
            if tr.stream and not getattr(
                self._queue_manager, "supports_streaming", False
            ):
                # Ask for one body and stream it to the client afterwards.
                body.pop("stream", None)
                body.pop("stream_options", None)

        # 3. Extract request headers to forward
        forward_headers = {}
        for key, value in request.headers.items():
            lower_key = key.lower()
            # Skip hop-by-hop headers and host
            if lower_key not in (
                "host",
                "connection",
                "transfer-encoding",
                "content-length",
            ):
                forward_headers[key] = value

        # 4. Select instance using scheduling algorithm
        try:
            # Use model_id from first instance (all instances serve same model)
            model_id = all_instances[0].model_id
            metadata: dict[str, Any] = {
                "path": path,
                "method": method,
                "proxy": True,
            }
            parent = request.headers.get(PARENT_HEADER)
            if parent:
                metadata[PARENT_KEY] = parent
            # The engine's adapter turns the body into predictor features,
            # so the scheduling strategy can call the trained predictor and
            # training samples use the same features later.
            metadata.update(
                _extract_features(all_instances[0].platform_info, path, body)
            )
            # Forward experiment mode fields from body to scheduling metadata
            for _exp_key in (
                "exp_runtime",
                "exp_cv",
                "exp_modes",
                "exp_skewness",
            ):
                if _exp_key in client_body:
                    metadata[_exp_key] = client_body[_exp_key]

            with (
                self.shadow.recording()
                if self.shadow is not None
                else nullcontext([])
            ) as draws:
                started = time.perf_counter()
                schedule_result = await self._scheduling_strategy.schedule_task(
                    model_id=model_id,
                    metadata=metadata,
                    available_instances=all_instances,
                    task_id=task_id,
                )
                decide_us = (time.perf_counter() - started) * 1e6
            selected_instance_id = schedule_result.selected_instance_id
            prediction = schedule_result.selected_prediction
            if self.shadow is not None:
                self.shadow.decided(
                    task_id=task_id,
                    path=client_path,
                    body=client_body,
                    instance_id=selected_instance_id,
                    draws=draws,
                    decide_us=decide_us,
                    strategy=type(self._scheduling_strategy).__name__,
                )

        except Exception as e:
            logger.opt(exception=True).error(f"[PROXY] Scheduling failed: {e}")
            # Fallback: pick first instance, with no prediction
            selected_instance_id = all_instances[0].instance_id
            prediction = None
            logger.info(
                f"[PROXY] Falling back to instance {selected_instance_id}"
            )

        # 5. Verify selected instance has a worker queue
        if not self._queue_manager.has_worker(selected_instance_id):
            await self._instance_registry.release_charge(task_id)
            logger.error(
                f"[PROXY] Instance {selected_instance_id} has no worker queue"
            )
            return _error_response(
                path,
                503,
                "Selected instance not ready",
                "service_unavailable",
                tr,
            )

        # 6. Register task in TaskRegistry so that handle_result
        #    can find it for stats, training-sample collection, etc.
        await self._task_registry.create_task(
            task_id=task_id,
            model_id=model_id,
            task_input=body,
            metadata=metadata,
            assigned_instance=selected_instance_id,
            # The result callback hands these to the trainer, whose drift
            # detector needs each task's prediction.
            **_prediction_fields(prediction),
        )
        arrival_counter.record(metadata.get(PARENT_KEY))

        # 7. Streamed requests get the engine's answer through a sink as
        #    soon as it starts; the rest wait for the completed result.
        stream = (
            isinstance(body, dict) and bool(body.get("stream"))
        ) and getattr(self._queue_manager, "supports_streaming", False)
        sink: asyncio.Future | None = None
        if stream:
            sink = asyncio.get_running_loop().create_future()
            future = sink
        else:
            future = self._callback.register_future(task_id)

        # 9. Create and enqueue the task
        queued_task = QueuedTask(
            task_id=task_id,
            model_id=model_id,
            task_input=body,
            metadata={
                "path": path,
                "method": method,
                "headers": forward_headers,
                "proxy": True,
            },
            enqueue_time=time.time(),
            stream_sink=sink,
        )

        is_priority = _is_lightweight(all_instances[0].platform_info, path)

        try:
            if is_priority:
                self._queue_manager.enqueue_priority_task(
                    selected_instance_id, queued_task
                )
            else:
                self._queue_manager.enqueue_task(
                    selected_instance_id, queued_task
                )
        except (ValueError, RuntimeError) as e:
            self._callback.cleanup_future(task_id)
            await self._instance_registry.release_charge(task_id)
            logger.error(f"[PROXY] Failed to enqueue task: {e}")
            return _error_response(
                path,
                503,
                "Failed to enqueue request",
                "service_unavailable",
                tr,
            )

        logger.debug(
            f"[PROXY] {method} /{path} enqueued to {selected_instance_id} "
            f"(task_id={task_id})"
        )

        # 10. Await the Future with timeout
        try:
            result = await asyncio.wait_for(
                future,
                timeout=self._proxy_timeout,
            )
        except TimeoutError:
            self._callback.cleanup_future(task_id)
            logger.error(
                f"[PROXY] Timeout after {self._proxy_timeout}s for "
                f"{method} /{path} (task_id={task_id})"
            )
            return _error_response(
                path, 504, "Request timed out", "gateway_timeout", tr
            )

        if isinstance(result, StreamStart):
            return self._stream_response(result, tr)

        # 11. Return the backend's response
        status_code = result.http_status_code or 200
        response_body = result.result or {}

        if result.status == "failed" and not result.http_status_code:
            # Internal failure (connection error, etc.)
            error_message = result.error or "Backend request failed"
            return _error_response(path, 502, error_message, "bad_gateway", tr)

        return _body_response(
            status_code, response_body, result.response_headers, tr
        )

    @staticmethod
    def _stream_response(
        start: StreamStart, tr: Translation | None = None
    ) -> Response:
        """Turn the engine's first answer into the client's response."""
        if start.error is not None:
            return _error_response("", 502, start.error, "bad_gateway", tr)
        if start.chunks is None:
            return _body_response(
                start.status_code,
                start.body if start.body is not None else {},
                start.headers,
                tr,
            )
        headers = _filter_headers(start.headers)
        headers.pop("content-type", None)
        chunks = start.chunks
        if tr is not None and not tr.passthrough:
            headers.update(tr.headers())
            chunks = tr.stream_response(start.chunks)
        return StreamingResponse(
            chunks,
            status_code=start.status_code,
            headers=headers,
            media_type=start.headers.get("content-type", "text/event-stream"),
            # Release the engine connection even if the client is gone
            # before the first chunk.
            background=BackgroundTask(start.chunks.aclose),
        )


def _error_response(
    path: str,
    status: int,
    message: str,
    error_type: str,
    tr: Translation | None = None,
    *,
    param: str | None = None,
) -> JSONResponse:
    """Return SwarmPilot's own error in the style of the route called.

    Routes of no known style get the OpenAI-shaped body they always had.
    """
    style = tr.client if tr is not None else style_for_path(path)
    if style is None:
        content: dict[str, Any] = {
            "error": {"message": message, "type": error_type}
        }
    else:
        content = style.encode_error(status, message, error_type, param)
    return JSONResponse(status_code=status, content=content)


def _body_response(
    status: int,
    body: Any,
    headers: dict[str, str] | None,
    tr: Translation | None,
) -> Response:
    """Return an engine's one-body answer, translated when needed."""
    out = _filter_headers(headers)
    if tr is None or tr.passthrough:
        return JSONResponse(status_code=status, content=body, headers=out)
    out.update(tr.headers())
    out.pop("content-type", None)
    if tr.stream and status < 400 and isinstance(body, dict):
        # The client asked for a stream; the engine answered with one body.
        return StreamingResponse(
            tr.stream_from_body(body),
            status_code=status,
            headers=out,
            media_type="text/event-stream",
        )
    status, content = tr.response(status, body)
    return JSONResponse(status_code=status, content=content, headers=out)


def _filter_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Drop hop-by-hop and length/encoding headers from an engine response."""
    return {
        k: v
        for k, v in (headers or {}).items()
        if k.lower()
        not in ("content-length", "transfer-encoding", "content-encoding")
    }
