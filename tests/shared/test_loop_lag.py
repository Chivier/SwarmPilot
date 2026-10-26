"""Event-loop lag probe lifecycle across app lifespans."""

import asyncio

from swarmpilot.shared.loop_lag import EventLoopLagMonitor


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_start_is_idempotent_on_one_loop():
    async def go():
        m = EventLoopLagMonitor(interval_s=0.005)
        m.start()
        first = m._task
        m.start()
        assert m._task is first
        await asyncio.sleep(0.03)
        await m.stop()
        assert not m.running

    _run(go())


def test_restart_on_new_loop_after_old_loop_closed():
    m = EventLoopLagMonitor(interval_s=0.005)

    async def first_lifespan():
        m.start()
        await asyncio.sleep(0.03)
        # Lifespan ends without stop(): the loop closes under the task.

    _run(first_lifespan())
    assert m.snapshot()["samples"] > 0
    since = m.snapshot()["since"]

    async def second_lifespan():
        m.start()
        assert m.running
        assert m.snapshot()["samples"] == 0
        assert m.snapshot()["since"] >= since
        await asyncio.sleep(0.03)
        await m.stop()

    _run(second_lifespan())
    assert m.snapshot()["samples"] > 0


def test_stop_with_task_on_closed_loop_is_safe():
    m = EventLoopLagMonitor(interval_s=0.005)

    async def lifespan():
        m.start()

    _run(lifespan())

    async def shutdown():
        await m.stop()

    _run(shutdown())
    assert m._task is None
