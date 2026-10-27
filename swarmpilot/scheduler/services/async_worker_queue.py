"""Async per-instance dispatch queue (plan item 1.3).

:class:`AsyncWorkerQueue` replaces the one-thread, one-blocking-request
:class:`~swarmpilot.scheduler.services.worker_queue_thread.WorkerQueueThread`
with an asyncio queue on the Scheduler's event loop. Each instance keeps a
FIFO queue, but up to ``max_in_flight`` requests (taken from the engine
adapter's ``capacity()``) are sent at once over one shared
``aiohttp.ClientSession``, so continuous-batching engines such as vLLM and
SGLang see real concurrency. Streamed responses are handed to the caller as
soon as the engine answers, chunk by chunk.

The public methods match ``WorkerQueueThread`` so the manager, the proxy and
the API endpoints work with either.
"""

from __future__ import annotations

import asyncio
import time
import urllib.request
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from loguru import logger

from swarmpilot.engines.base import (
    EngineAdapter,
    EngineRequest,
    EngineResponse,
)
from swarmpilot.scheduler.services.worker_queue_thread import (
    QueuedTask,
    TaskResult,
)

_IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

# The request never reached the engine: the connection was refused or
# never opened. Typical of a crashed engine or a lost node.
_UNREACHABLE = (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError)


class StreamStart:
    """What a streamed task delivers to the caller when the engine answers.

    Attributes:
        status_code: Engine HTTP status.
        headers: Engine response headers.
        chunks: Byte chunks of the body, or None when the engine answered
            with a complete (non-streamed) body.
        body: The complete body when ``chunks`` is None.
        error: Transport error message when the request never got an
            answer (``status_code`` is 0).
    """

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        chunks: AsyncIterator[bytes] | None = None,
        body: Any = None,
        error: str | None = None,
    ):
        self.status_code = status_code
        self.headers = headers
        self.chunks = chunks
        self.body = body
        self.error = error


def env_proxy_for(url: str) -> str | None:
    """Return the HTTP(S)_PROXY that applies to ``url``, honouring NO_PROXY.

    Resolved once per instance: asking the environment on every request
    (aiohttp's ``trust_env``) cost about 0.5 ms per request with a long
    NO_PROXY list.

    Args:
        url: Instance base URL.

    Returns:
        Proxy URL, or None to connect directly.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not host or urllib.request.proxy_bypass(host):
        return None
    proxies = urllib.request.getproxies()
    return proxies.get(parts.scheme or "http")


class AsyncWorkerQueue:
    """FIFO queue for one instance with bounded concurrent dispatch.

    Must be created and used on the event loop that runs it.
    """

    def __init__(
        self,
        worker_id: str,
        worker_endpoint: str,
        model_id: str,
        callback: Callable[[TaskResult], None],
        session: aiohttp.ClientSession,
        adapter: EngineAdapter,
        max_in_flight: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        on_task_started: Callable[[str], None] | None = None,
        extra_headers: dict[str, str] | None = None,
        on_unreachable: Callable[[str, list[QueuedTask]], None] | None = None,
    ):
        """Create the queue.

        Args:
            worker_id: Instance id.
            worker_endpoint: Instance base URL.
            model_id: Model the instance serves.
            callback: Called with the TaskResult when a task finishes.
            session: Shared aiohttp session.
            adapter: Engine adapter used to send requests.
            max_in_flight: Requests sent to the instance at the same time.
            max_retries: Attempts for connection errors.
            retry_delay: First retry delay; doubles each attempt.
            on_task_started: Called with the task id when a task is sent.
            extra_headers: Headers merged into every request (task headers
                win on conflict).
            on_unreachable: Called with the instance id and its unsent
                tasks when a connection to the instance cannot be opened.
                The failing task comes first; none of them are reported
                through ``callback``, so the caller must re-route or fail
                them. Without it, the queue retries the same instance.
        """
        self.worker_id = worker_id
        self.worker_endpoint = worker_endpoint
        self.model_id = model_id
        self.max_in_flight = max(1, max_in_flight)
        self._callback = callback
        self._session = session
        self._adapter = adapter
        self._max_retries = max(1, max_retries)
        self._retry_delay = retry_delay
        self._on_task_started = on_task_started
        self._extra_headers = extra_headers or {}
        self._on_unreachable = on_unreachable
        self._proxy = env_proxy_for(worker_endpoint)

        self._queue: deque[QueuedTask] = deque()
        self._priority: deque[QueuedTask] = deque()
        self._running: dict[str, tuple[QueuedTask, float]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._wakeup: asyncio.Event | None = None
        self._dispatcher: asyncio.Task | None = None
        self._stopped = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start dispatching (needs a running event loop)."""
        if self._dispatcher is not None:
            raise RuntimeError(f"Worker {self.worker_id} already started")
        self._wakeup = asyncio.Event()
        self._dispatcher = asyncio.get_running_loop().create_task(
            self._dispatch_loop(), name=f"dispatch-{self.worker_id}"
        )
        logger.info(
            f"Started async worker queue for {self.worker_id} "
            f"(max_in_flight={self.max_in_flight}, engine={self._adapter.name})"
        )

    def stop(self, timeout: float = 10.0) -> list[QueuedTask]:
        """Stop dispatching and return tasks that were never sent.

        Requests already in flight keep running and report through the
        callback as usual.

        Args:
            timeout: Unused; kept for interface compatibility.

        Returns:
            Queued (unsent) tasks, priority tasks first.
        """
        self._stopped = True
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            self._dispatcher = None
        pending = self.take_pending()
        logger.info(
            f"Worker {self.worker_id} stopped with {len(pending)} pending "
            f"and {len(self._running)} in-flight tasks"
        )
        return pending

    # -- queue interface (same as WorkerQueueThread) -----------------------

    def enqueue(self, task: QueuedTask) -> int:
        """Append a task and return the queue size."""
        if self._stopped:
            raise RuntimeError(f"Worker {self.worker_id} is stopped")
        self._queue.append(task)
        self._kick()
        return len(self._queue)

    def enqueue_priority(self, task: QueuedTask) -> None:
        """Put a lightweight task ahead of the normal queue."""
        if self._stopped:
            raise RuntimeError(f"Worker {self.worker_id} is stopped")
        task.priority = True
        self._priority.append(task)
        self._kick()

    def take_pending(self) -> list[QueuedTask]:
        """Remove and return the tasks not sent yet, priority tasks first."""
        pending = [*self._priority, *self._queue]
        self._priority.clear()
        self._queue.clear()
        return pending

    def queue_size(self) -> int:
        """Return the number of normal tasks waiting to be sent."""
        return len(self._queue)

    def in_flight(self) -> int:
        """Return the number of requests currently sent to the instance."""
        return len(self._running)

    def outstanding(self) -> tuple[int, int]:
        """Return (queued including priority, in flight) task counts."""
        return len(self._queue) + len(self._priority), len(self._running)

    def has_running_task(self) -> bool:
        """Return whether any request is in flight."""
        return bool(self._running)

    def get_current_task_id(self) -> str | None:
        """Return the oldest in-flight task id, if any."""
        return next(iter(self._running), None)

    def get_estimated_wait_time(self, avg_exec_time_ms: float) -> float:
        """Estimate wait time for a new task in milliseconds.

        With ``k`` requests served at once, the queue drains about ``k``
        times faster than one at a time.

        Args:
            avg_exec_time_ms: Average execution time per task.

        Returns:
            Estimated wait in milliseconds.
        """
        now = time.time()
        remaining = sum(
            max(
                0.0,
                (t.predicted_time_ms or avg_exec_time_ms) - (now - s) * 1000,
            )
            for t, s in self._running.values()
        )
        backlog = len(self._queue) * avg_exec_time_ms + remaining
        return backlog / self.max_in_flight

    # -- dispatch ----------------------------------------------------------

    def _kick(self) -> None:
        if self._wakeup is not None:
            self._wakeup.set()

    def _next(self) -> QueuedTask | None:
        if self._priority:
            return self._priority.popleft()
        if self._queue:
            return self._queue.popleft()
        return None

    async def _dispatch_loop(self) -> None:
        assert self._wakeup is not None
        while True:
            # Priority (metadata) requests bypass the in-flight limit, as
            # they did with the thread queue.
            while self._priority or (
                self._queue and len(self._running) < self.max_in_flight
            ):
                task = self._next()
                if task is None:
                    break
                self._running[task.task_id] = (task, time.time())
                t = asyncio.get_running_loop().create_task(self._execute(task))
                self._tasks.add(t)
                t.add_done_callback(self._tasks.discard)
            self._wakeup.clear()
            await self._wakeup.wait()

    def _finish(self, task: QueuedTask, result: TaskResult) -> None:
        self._running.pop(task.task_id, None)
        self._kick()
        try:
            self._callback(result)
        except Exception as e:
            logger.opt(exception=True).error(
                f"Result callback failed for {task.task_id}: {e}"
            )

    def _request(self, task: QueuedTask) -> EngineRequest:
        headers = {**self._extra_headers, **task.metadata.get("headers", {})}
        body = task.task_input if isinstance(task.task_input, dict) else {}
        return EngineRequest(
            path=task.metadata.get("path", "v1/completions"),
            method=task.metadata.get("method", "POST").upper(),
            body=body,
            headers=headers,
            stream=bool(task.stream_sink is not None and body.get("stream")),
            proxy=self._proxy,
        )

    def _retryable(self, request: EngineRequest, exc: Exception) -> bool:
        """Whether resending ``request`` after ``exc`` is harmless.

        A refused connection never reached the engine, so it is always
        retried. A connection lost after sending may have been processed,
        so only routes the adapter marks ``retry_safe`` are resent; for
        routes it does not declare, only idempotent methods are.
        """
        if isinstance(exc, _UNREACHABLE):
            return True
        route = self._adapter.route(request.path)
        if route is not None:
            return route.retry_safe
        return request.method in _IDEMPOTENT

    async def _send(self, request: EngineRequest):
        for attempt in range(self._max_retries):
            try:
                return await self._adapter.dispatch(
                    self._session, self.worker_endpoint, request
                )
            except (
                *_UNREACHABLE,
                aiohttp.ServerDisconnectedError,
            ) as e:
                if isinstance(e, _UNREACHABLE) and self._on_unreachable:
                    # Another instance takes it now, instead of this one
                    # after seconds of backoff.
                    raise
                if attempt == self._max_retries - 1 or not self._retryable(
                    request, e
                ):
                    raise
                delay = self._retry_delay * (2**attempt)
                logger.warning(
                    f"Worker {self.worker_id} connection error ({e}), "
                    f"retrying in {delay}s (attempt {attempt + 1})"
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    def _hand_back(self, task: QueuedTask, e: Exception) -> None:
        """Give the failed task and all unsent ones to ``on_unreachable``."""
        self._running.pop(task.task_id, None)
        pending = [task, *self.take_pending()]
        logger.warning(
            f"Worker {self.worker_id} unreachable ({type(e).__name__}: {e}); "
            f"handing back {len(pending)} unsent task(s)"
        )
        self._kick()
        try:
            self._on_unreachable(self.worker_id, pending)
        except Exception as cb_err:
            logger.opt(exception=True).error(
                f"on_unreachable failed for {self.worker_id}: {cb_err}"
            )

    async def _execute(self, task: QueuedTask) -> None:
        started = time.time()
        if self._on_task_started:
            try:
                self._on_task_started(task.task_id)
            except Exception as e:
                logger.error(f"on_task_started failed for {task.task_id}: {e}")

        sink = task.stream_sink
        try:
            resp = await self._send(self._request(task))
        except Exception as e:
            if isinstance(e, _UNREACHABLE) and self._on_unreachable:
                self._hand_back(task, e)
                return
            elapsed = (time.time() - started) * 1000
            logger.warning(
                f"Task {task.task_id} failed after {elapsed:.2f}ms: "
                f"{type(e).__name__}: {e}"
            )
            if sink is not None and not sink.done():
                sink.set_result(StreamStart(0, {}, error=str(e) or repr(e)))
            self._finish(
                task,
                TaskResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status="failed",
                    error=str(e) or repr(e),
                    execution_time_ms=elapsed,
                ),
            )
            return

        status = "completed" if resp.status_code < 400 else "failed"
        if resp.chunks is None:
            elapsed = (time.time() - started) * 1000
            if sink is not None and not sink.done():
                sink.set_result(
                    StreamStart(resp.status_code, resp.headers, body=resp.body)
                )
            self._finish(
                task,
                TaskResult(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status=status,
                    result=resp.body,
                    error=(
                        None
                        if status == "completed"
                        else f"HTTP {resp.status_code}"
                    ),
                    execution_time_ms=elapsed,
                    http_status_code=resp.status_code,
                    response_headers=resp.headers,
                ),
            )
            return

        # Streamed: hand the chunks to the caller now, report completion
        # when the stream ends or is closed (client went away).
        relay = _Relay(resp, asyncio.get_running_loop().create_future())
        if sink is None or sink.done():
            # Nobody is listening; drain so the engine's slot is freed.
            async for _ in relay:
                pass
        else:
            sink.set_result(
                StreamStart(resp.status_code, resp.headers, chunks=relay)
            )
        ok = await relay.done
        self._finish(
            task,
            TaskResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                status="completed" if ok else "failed",
                result={"streamed": True},
                error=None if ok else "stream closed before the end",
                execution_time_ms=(time.time() - started) * 1000,
                http_status_code=resp.status_code,
                response_headers=resp.headers,
            ),
        )


class _Relay:
    """Async iterator over a streamed engine response.

    ``done`` resolves to True when the stream was read to the end and to
    False when it was closed early. ``aclose`` always releases the engine
    connection, even if iteration never started.
    """

    def __init__(self, resp: EngineResponse, done: asyncio.Future):
        self._resp = resp
        self._it = resp.chunks.__aiter__()
        self.done = done
        self._answered = False

    def __aiter__(self) -> _Relay:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self._it.__anext__()
        except StopAsyncIteration:
            await self._close(ok=True)
            raise
        except BaseException:
            await self._close(ok=False)
            raise

    def answered(self) -> None:
        """Count the stream as complete even if it is closed early.

        Called once the client has the whole answer, so a body that the
        engine keeps open after its last event can be cut off.
        """
        self._answered = True

    async def aclose(self) -> None:
        """Stop reading and release the connection."""
        await self._close(ok=False)

    async def _close(self, ok: bool) -> None:
        if self.done.done():
            return
        try:
            if self._resp.close is not None:
                await self._resp.close()
        finally:
            self.done.set_result(ok or self._answered)
