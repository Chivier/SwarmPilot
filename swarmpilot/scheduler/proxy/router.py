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
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from loguru import logger

from swarmpilot.scheduler.services.worker_queue_thread import QueuedTask

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


# Lightweight OpenAI-compatible paths that should be prioritized.
# These are fast metadata queries that should not be blocked by
# long-running inference requests in the worker queue.
PRIORITY_PATHS: set[str] = {
    "v1/models",
    "health",
    "ping",
    "version",
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

        logger.debug(f"[PROXY] {method} /{path} -> task_id={task_id}")

        # 1. Check for available instances
        # Get all active instances (proxy doesn't filter by model_id)
        all_instances = await self._instance_registry.list_active()
        if not all_instances:
            logger.warning(
                f"[PROXY] No instances available for {method} /{path}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "No backend instances available",
                        "type": "service_unavailable",
                    }
                },
            )

        # 2. Parse request body
        body: dict[str, Any] = {}
        if method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                # Non-JSON body or empty body
                body = {}

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
            # Extract predictor features from request body so the
            # scheduling strategy can call the trained predictor.
            if "max_tokens" in body:
                metadata["max_tokens"] = body["max_tokens"]
            messages = body.get("messages")
            if isinstance(messages, list):
                metadata["prompt_length"] = sum(
                    len(m.get("content", ""))
                    for m in messages
                    if isinstance(m, dict)
                )
            elif "prompt" in body:
                metadata["prompt_length"] = len(body["prompt"])
            # Forward experiment mode fields from body to scheduling metadata
            for _exp_key in ("exp_runtime", "exp_cv", "exp_modes", "exp_skewness"):
                if _exp_key in body:
                    metadata[_exp_key] = body[_exp_key]

            schedule_result = await self._scheduling_strategy.schedule_task(
                model_id=model_id,
                metadata=metadata,
                available_instances=all_instances,
            )
            selected_instance_id = schedule_result.selected_instance_id

        except Exception as e:
            logger.opt(exception=True).error(f"[PROXY] Scheduling failed: {e}")
            # Fallback: pick first instance
            selected_instance_id = all_instances[0].instance_id
            logger.info(
                f"[PROXY] Falling back to instance {selected_instance_id}"
            )

        # 5. Verify selected instance has a worker queue
        if not self._queue_manager.has_worker(selected_instance_id):
            logger.error(
                f"[PROXY] Instance {selected_instance_id} has no worker queue"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Selected instance not ready",
                        "type": "service_unavailable",
                    }
                },
            )

        # 6. Register task in TaskRegistry so that handle_result
        #    can find it for stats, training-sample collection, etc.
        await self._task_registry.create_task(
            task_id=task_id,
            model_id=model_id,
            task_input=body,
            metadata=metadata,
            assigned_instance=selected_instance_id,
        )

        # 7. Register Future for this task
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
        )

        is_priority = path in PRIORITY_PATHS

        try:
            if is_priority:
                self._queue_manager.enqueue_priority_task(
                    selected_instance_id, queued_task
                )
            else:
                self._queue_manager.enqueue_task(
                    selected_instance_id, queued_task
                )
        except ValueError as e:
            self._callback.cleanup_future(task_id)
            logger.error(f"[PROXY] Failed to enqueue task: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Failed to enqueue request",
                        "type": "service_unavailable",
                    }
                },
            )

        logger.info(
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
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "message": "Request timed out",
                        "type": "gateway_timeout",
                    }
                },
            )

        # 11. Return the backend's response
        status_code = result.http_status_code or 200
        response_body = result.result or {}

        if result.status == "failed" and not result.http_status_code:
            # Internal failure (connection error, etc.)
            error_message = result.error or "Backend request failed"
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": error_message,
                        "type": "bad_gateway",
                    }
                },
            )

        # Build response headers (filter out problematic ones)
        response_headers = {}
        for key, value in (result.response_headers or {}).items():
            lower_key = key.lower()
            if lower_key not in (
                "content-length",
                "transfer-encoding",
                "content-encoding",
            ):
                response_headers[key] = value

        return JSONResponse(
            status_code=status_code,
            content=response_body,
            headers=response_headers,
        )
