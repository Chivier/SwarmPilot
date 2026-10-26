"""Arrival counting and the extended Planner report (item 3.4)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from swarmpilot.scheduler.models import (
    Instance,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
from swarmpilot.scheduler.utils.arrival_counter import (
    ArrivalCounter,
    parent_of,
)
from swarmpilot.scheduler.utils.planner_reporter import PlannerReporter


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_counter_windows_and_resets():
    clock = _Clock()
    counter = ArrivalCounter(clock=clock)
    for parent in (None, "llm", "llm", "vlm"):
        counter.record(parent)
    clock.t += 2.0
    window = counter.take()
    assert window.rate == 2.0
    assert window.call_rates() == {"llm": 1.0, "vlm": 0.5}
    clock.t += 1.0
    assert counter.take().arrivals == 0


@pytest.mark.parametrize(
    "metadata,parent",
    [
        (None, None),
        ({}, None),
        ({"parent_model": "a"}, "a"),
        ({"parent_model": 3}, None),
        ({"parent_model": ""}, None),
    ],
)
def test_parent_of(metadata, parent):
    assert parent_of(metadata) == parent


def _instance(iid: str, model: str = "m") -> Instance:
    return Instance(
        instance_id=iid,
        model_id=model,
        endpoint=f"http://{iid}",
        platform_info={
            "software_name": "vllm",
            "software_version": "1",
            "hardware_name": "h",
        },
    )


async def test_report_carries_rate_calls_and_tail_work():
    registry = InstanceRegistry()
    for iid in ("a", "b", "other"):
        await registry.register(_instance(iid, "x" if iid == "other" else "m"))
    await registry.set_capacity("b", 4)
    await registry.update_queue_info(
        "a",
        InstanceQueueProbabilistic(
            instance_id="a", quantiles=[0.5, 0.95], values=[1000.0, 3000.0]
        ),
    )
    await registry.update_queue_info(
        "b",
        InstanceQueueExpectError(
            instance_id="b", expected_time_ms=400.0, error_margin_ms=100.0
        ),
    )
    await registry.update_queue_info(
        "other",
        InstanceQueueExpectError(
            instance_id="other", expected_time_ms=9e6, error_margin_ms=0
        ),
    )
    clock = _Clock()
    counter = ArrivalCounter(clock=clock)
    for parent in (None, None, "llm", "llm"):
        counter.record(parent)
    clock.t += 4.0

    tasks = MagicMock()
    tasks.get_count_by_status = AsyncMock(return_value=3)
    reporter = PlannerReporter(
        task_registry=tasks,
        planner_url="http://planner",
        interval=4.0,
        instance_registry=registry,
        arrivals=counter,
    )
    reporter.set_model_id("m")
    reporter._http_client = MagicMock()
    reporter._http_client.post = AsyncMock(
        return_value=MagicMock(raise_for_status=MagicMock())
    )
    await reporter._report_to_planner()

    body = reporter._http_client.post.call_args.kwargs["json"]
    assert body["model_id"] == "m" and body["value"] == 6.0
    assert body["arrival_rate"] == 1.0
    assert body["calls_from"] == {"llm": 0.5}
    # a: q0.9 interpolated between q0.5 and q0.95 = 2777.8 ms.
    # b: (400 + 100) ms per server, 4 servers. "other" serves model x.
    assert body["work_s"] == pytest.approx((1000 + 2000 * 0.4 / 0.45) / 1000 + 2.0)


async def test_report_without_new_sources_keeps_old_body():
    tasks = MagicMock()
    tasks.get_count_by_status = AsyncMock(return_value=1)
    reporter = PlannerReporter(
        task_registry=tasks, planner_url="http://planner", interval=1.0
    )
    reporter.set_model_id("m")
    reporter._http_client = MagicMock()
    reporter._http_client.post = AsyncMock(
        return_value=MagicMock(raise_for_status=MagicMock())
    )
    await reporter._report_to_planner()
    body = reporter._http_client.post.call_args.kwargs["json"]
    assert body == {"model_id": "m", "value": 2.0}


def test_submit_counts_parent_from_metadata():
    from fastapi.testclient import TestClient

    from swarmpilot.scheduler import api
    from swarmpilot.scheduler.utils.arrival_counter import arrival_counter

    with TestClient(api.app) as client:
        arrival_counter.take()
        for i, md in enumerate(({"parent_model": "llm"}, {})):
            r = client.post(
                "/v1/task/submit",
                json={
                    "task_id": f"arrival-{i}",
                    "model_id": "m",
                    "task_input": {"prompt": "x"},
                    "metadata": md,
                },
            )
            assert r.status_code == 200, r.text
        window = arrival_counter.take()
        client.post("/v1/task/clear")
    assert window.arrivals == 2
    assert window.calls_from == {"llm": 1}
