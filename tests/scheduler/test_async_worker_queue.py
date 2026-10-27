"""Tests for the async per-instance dispatch queue (plan item 1.3)."""

import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from swarmpilot.engines import RouteSpec, get_adapter
from swarmpilot.engines.base import EngineResponse
from swarmpilot.engines.passthrough import PassthroughAdapter
from swarmpilot.scheduler.services.async_worker_queue import (
    AsyncWorkerQueue,
    StreamStart,
)
from swarmpilot.scheduler.services.worker_queue_manager import (
    WorkerQueueManager,
    resolve_adapter,
)
from swarmpilot.scheduler.services.worker_queue_thread import (
    QueuedTask,
    TaskResult,
)


def _task(i: int, body: dict | None = None, sink=None) -> QueuedTask:
    return QueuedTask(
        task_id=f"t{i}",
        model_id="m",
        task_input=body or {"prompt": "x"},
        metadata={"path": "v1/completions", "headers": {"x-req": "1"}},
        enqueue_time=time.time(),
        stream_sink=sink,
    )


class _Engine:
    """Fake engine server that records concurrency."""

    def __init__(self):
        self.delay = 0.0
        self.active = 0
        self.peak = 0
        self.seen: list[web.Request] = []
        self.url = ""

    async def handler(self, req: web.Request) -> web.StreamResponse:
        self.seen.append(req)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        body = json.loads(await req.read() or b"{}")
        if body.get("stream"):
            resp = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(req)
            for i in range(3):
                await resp.write(f"data: {i}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        return web.json_response({"echo": body})


class _Key:
    host, port, ssl = "engine", 1, False


def _refused() -> aiohttp.ClientConnectorError:
    return aiohttp.ClientConnectorError(
        _Key(), ConnectionRefusedError(111, "x")
    )


class _FlakyAdapter(PassthroughAdapter):
    """Pass-through whose first ``fail`` sends raise ``exc()``."""

    def __init__(self, fail: int = 0, exc=_refused, routes=()):
        self.fail = fail
        self.exc = exc
        self.sent = 0
        self._routes = routes

    def routes(self):
        return self._routes

    async def dispatch(self, session, endpoint, request):
        self.sent += 1
        if self.fail:
            self.fail -= 1
            raise self.exc()
        return await super().dispatch(session, endpoint, request)


@pytest.fixture
async def engine():
    eng = _Engine()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", eng.handler)
    server = TestServer(app)
    await server.start_server()
    eng.url = str(server.make_url("")).rstrip("/")
    yield eng
    await server.close()


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


def _queue(engine: _Engine, session, results: list, **kw) -> AsyncWorkerQueue:
    return AsyncWorkerQueue(
        worker_id="w",
        worker_endpoint=engine.url,
        model_id="m",
        callback=results.append,
        session=session,
        adapter=kw.pop("adapter", PassthroughAdapter()),
        retry_delay=0.0,
        extra_headers={"authorization": "Bearer k"},
        **kw,
    )


async def _wait(cond, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise TimeoutError
        await asyncio.sleep(0.005)


class TestAsyncWorkerQueue:
    """Tests for Async Worker Queue."""

    async def test_respects_max_in_flight(self, engine, session):
        engine.delay, results = 0.05, []
        q = _queue(engine, session, results, max_in_flight=4)
        q.start()
        for i in range(12):
            q.enqueue(_task(i))
        await _wait(lambda: len(results) == 12)
        assert engine.peak == 4
        assert all(isinstance(r, TaskResult) for r in results)
        assert {r.status for r in results} == {"completed"}
        assert results[0].result == {"echo": {"prompt": "x"}}
        q.stop()

    async def test_one_in_flight_keeps_fifo_order(self, engine, session):
        engine.delay, results = 0.001, []
        q = _queue(engine, session, results, max_in_flight=1)
        q.start()
        for i in range(5):
            q.enqueue(_task(i))
        await _wait(lambda: len(results) == 5)
        assert [r.task_id for r in results] == [f"t{i}" for i in range(5)]
        assert engine.peak == 1
        q.stop()

    async def test_headers_merge_and_priority_first(self, engine, session):
        engine.delay, results = 0.01, []
        q = _queue(engine, session, results, max_in_flight=1)
        for i in range(3):
            q.enqueue(_task(i))
        q.enqueue_priority(_task(99))
        q.start()
        await _wait(lambda: len(results) == 4)
        assert results[0].task_id == "t99"
        assert engine.seen[0].headers["Authorization"] == "Bearer k"
        assert engine.seen[0].headers["x-req"] == "1"
        q.stop()

    async def test_stop_returns_unsent_tasks(self, engine, session):
        engine.delay, results = 0.2, []
        q = _queue(engine, session, results, max_in_flight=1)
        q.start()
        for i in range(4):
            q.enqueue(_task(i))
        await asyncio.sleep(0.02)
        pending = q.stop()
        assert [t.task_id for t in pending] == ["t1", "t2", "t3"]
        with pytest.raises(RuntimeError):
            q.enqueue(_task(9))
        # The in-flight request still reports back.
        await _wait(lambda: len(results) == 1)
        assert results[0].task_id == "t0"

    async def test_connect_error_retries_then_fails(self, engine, session):
        results = []
        adapter = _FlakyAdapter(fail=2)
        q = _queue(engine, session, results, max_retries=3, adapter=adapter)
        q.start()
        q.enqueue(_task(0))
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "completed"

        adapter.fail = 5
        q.enqueue(_task(1))
        await _wait(lambda: len(results) == 2)
        assert results[1].status == "failed"
        assert results[1].http_status_code == 0
        q.stop()

    async def test_lost_connection_not_resent_for_unsafe_route(
        self, engine, session
    ):
        results = []
        adapter = _FlakyAdapter(
            fail=1,
            exc=aiohttp.ServerDisconnectedError,
            routes=(RouteSpec("v1/completions", retry_safe=False),),
        )
        q = _queue(engine, session, results, max_retries=3, adapter=adapter)
        q.start()
        q.enqueue(_task(0))
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "failed"
        assert adapter.sent == 1
        q.stop()

    async def test_lost_connection_resent_for_safe_route(self, engine, session):
        results = []
        adapter = _FlakyAdapter(
            fail=1,
            exc=aiohttp.ServerDisconnectedError,
            routes=(RouteSpec("v1/completions", retry_safe=True),),
        )
        q = _queue(engine, session, results, max_retries=3, adapter=adapter)
        q.start()
        q.enqueue(_task(0))
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "completed"
        assert adapter.sent == 2
        q.stop()

    async def test_lost_connection_goes_to_on_lost(self, engine, session):
        results, lost = [], []
        adapter = _FlakyAdapter(
            fail=1,
            exc=aiohttp.ServerDisconnectedError,
            routes=(RouteSpec("v1/completions", rerun_safe=True),),
        )
        q = _queue(
            engine,
            session,
            results,
            adapter=adapter,
            on_lost=lambda wid, tasks: lost.extend(tasks),
        )
        q.start()
        q.enqueue(_task(0))
        await _wait(lambda: len(lost) == 1)
        assert results == [] and lost[0].reroutes == 1
        assert q.in_flight() == 0
        q.stop()

    async def test_lost_connection_not_rerun_unless_rerun_safe(
        self, engine, session
    ):
        results, lost = [], []
        adapter = _FlakyAdapter(
            fail=1,
            exc=aiohttp.ServerDisconnectedError,
            routes=(RouteSpec("v1/completions"),),
        )
        q = _queue(
            engine,
            session,
            results,
            adapter=adapter,
            on_lost=lambda wid, tasks: lost.extend(tasks),
        )
        q.start()
        q.enqueue(_task(0))
        await _wait(lambda: len(results) == 1)
        assert lost == [] and results[0].status == "failed"
        q.stop()

    async def test_503_goes_to_on_lost_until_max_reroutes(
        self, engine, session
    ):
        results, lost = [], []

        class _Stopped(PassthroughAdapter):
            async def dispatch(self, session, endpoint, request):
                return EngineResponse(503, body={"error": "engine stopped"})

        q = _queue(
            engine,
            session,
            results,
            adapter=_Stopped(),
            on_lost=lambda wid, tasks: lost.extend(tasks),
            max_reroutes=1,
        )
        q.start()
        task = _task(0)
        q.enqueue(task)
        await _wait(lambda: len(lost) == 1)
        # Sent again (here: to the same queue) after one loss, it fails.
        q.enqueue(task)
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "failed"
        assert results[0].http_status_code == 503
        q.stop()

    async def test_lost_connection_undeclared_route_by_method(
        self, engine, session
    ):
        results = []
        adapter = _FlakyAdapter(fail=2, exc=aiohttp.ServerDisconnectedError)
        q = _queue(engine, session, results, max_retries=3, adapter=adapter)
        q.start()
        get = _task(0)
        get.metadata["method"] = "GET"
        q.enqueue(get)
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "completed"
        adapter.fail = 1
        q.enqueue(_task(1))
        await _wait(lambda: len(results) == 2)
        assert results[1].status == "failed"
        q.stop()

    async def test_stream_is_relayed_then_completed(self, engine, session):
        results = []
        q = _queue(engine, session, results)
        q.start()
        sink = asyncio.get_running_loop().create_future()
        q.enqueue(_task(0, {"prompt": "x", "stream": True}, sink=sink))
        start: StreamStart = await asyncio.wait_for(sink, 1)
        assert start.status_code == 200 and start.chunks is not None
        assert results == []  # not complete until the stream ends
        data = b"".join([c async for c in start.chunks])
        assert data.endswith(b"data: [DONE]\n\n")
        await _wait(lambda: len(results) == 1)
        assert results[0].status == "completed"
        assert q.in_flight() == 0
        q.stop()

    async def test_stream_closed_early_is_failed_and_frees_slot(
        self, engine, session
    ):
        results = []
        q = _queue(engine, session, results, max_in_flight=1)
        q.start()
        sink = asyncio.get_running_loop().create_future()
        q.enqueue(_task(0, {"stream": True}, sink=sink))
        q.enqueue(_task(1))
        start = await asyncio.wait_for(sink, 1)
        await start.chunks.aclose()  # client went away before reading
        await _wait(lambda: len(results) == 2)
        assert results[0].status == "failed"
        assert results[1].status == "completed"

    async def test_stream_without_sink_is_sent_as_json(self, engine, session):
        results = []
        q = _queue(engine, session, results)
        q.start()
        q.enqueue(_task(0, {"stream": True}))
        await _wait(lambda: len(results) == 1)
        # Without a sink the request is not marked as streamed, but the
        # body still carries stream=true; the engine streams and the
        # queue drains it.
        assert results[0].status == "completed"
        q.stop()

    def test_estimated_wait_scales_with_concurrency(self):
        q = _queue(_Engine(), None, [], max_in_flight=4)
        for i in range(8):
            q._queue.append(_task(i))
        assert q.get_estimated_wait_time(100.0) == pytest.approx(200.0)


class TestManagerAsyncMode:
    """Tests for Manager Async Mode."""

    async def test_register_uses_adapter_capacity(self):
        mgr = WorkerQueueManager(callback=lambda r: None, async_dispatch=True)
        mgr.register_worker("a", "http://x:1", "m", engine="vllm")
        mgr.register_worker("b", "http://x:2", "m", engine="some-online-api")
        assert mgr.get_worker("a").max_in_flight == 256
        assert mgr.get_worker("b").max_in_flight == 1
        assert mgr.supports_streaming
        mgr.shutdown()
        await mgr.aclose()

    async def test_max_in_flight_override(self):
        mgr = WorkerQueueManager(
            callback=lambda r: None, async_dispatch=True, max_in_flight=8
        )
        mgr.register_worker("a", "http://x:1", "m", engine="vllm")
        mgr.register_worker("b", "http://x:2", "m")  # online endpoint slot
        assert mgr.get_worker("a").max_in_flight == 8
        assert mgr.get_worker("b").max_in_flight == 1
        mgr.shutdown()
        await mgr.aclose()

    def test_resolve_adapter(self):
        assert resolve_adapter("vllm") is get_adapter("vllm")
        assert isinstance(resolve_adapter(None), PassthroughAdapter)
        assert isinstance(resolve_adapter("nope"), PassthroughAdapter)
