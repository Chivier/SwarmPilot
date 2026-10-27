"""Re-predictions ask for the prediction kind the queue sketches keep.

/v1/task/update_metadata and /v1/task/repredict replace a task's charge on
its instance's sketch. They used to ask for expect-error predictions only
under min_time, so po2 and serverless got quantile predictions without
error margins charged into their expect-error sketches.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from swarmpilot.scheduler.algorithms import queue_info_type_for
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import TaskStatus

STRATEGIES = [
    "min_time",
    "po2",
    "serverless",
    "probabilistic",
    "adaptive_bootstrap",
    "round_robin",
    "random",
]

INSTANCE_ID = "inst-repredict-kind"


def _prediction() -> Prediction:
    return Prediction(
        instance_id=INSTANCE_ID,
        predicted_time_ms=300.0,
        confidence=0.9,
        error_margin_ms=30.0,
        quantiles={0.5: 250.0, 0.9: 350.0, 0.95: 400.0, 0.99: 500.0},
    )


@pytest.fixture
def client_with_strategy(request):
    """Switch to the requested strategy and restore the previous one."""
    import swarmpilot.scheduler.api as api_module

    registry = api_module.instance_registry
    saved = (api_module.scheduling_strategy, registry._queue_info_type)
    client = TestClient(api_module.app)
    r = client.post("/v1/strategy/set", json={"strategy_name": request.param})
    assert r.status_code == 200, r.text
    try:
        yield client, request.param
    finally:
        _retire_tasks()
        client.post("/v1/instance/remove", json={"instance_id": INSTANCE_ID})
        api_module.scheduling_strategy, queue_kind = saved
        registry._queue_info_type = queue_kind


def _retire_tasks() -> None:
    """Complete this module's tasks so removing the instance moves none."""
    from swarmpilot.scheduler.api import task_registry

    async def retire() -> None:
        tasks, _ = await task_registry.list_all(instance_id=INSTANCE_ID)
        for task in tasks:
            if task.assigned_instance == INSTANCE_ID:
                task.status = TaskStatus.COMPLETED

    asyncio.run(retire())


def _queued_task(client: TestClient, task_id: str) -> None:
    from swarmpilot.scheduler.api import task_registry

    client.post(
        "/v1/instance/register",
        json={
            "instance_id": INSTANCE_ID,
            "model_id": "model-kind",
            "endpoint": "http://localhost:8001",
            "platform_info": {
                "software_name": "docker",
                "software_version": "20.10",
                "hardware_name": "test-hw",
            },
        },
    )
    with patch(
        "swarmpilot.scheduler.api.predictor_client.predict",
        new=AsyncMock(return_value=[]),
    ):
        client.post(
            "/v1/task/submit",
            json={
                "task_id": task_id,
                "model_id": "model-kind",
                "task_input": {"prompt": "test"},
                "metadata": {"height": 512},
            },
        )

    async def assign() -> None:
        task = await task_registry.get(task_id)
        assert task is not None
        task.assigned_instance = INSTANCE_ID
        task.status = TaskStatus.PENDING

    asyncio.run(assign())


def _expected_type(strategy: str) -> str:
    if queue_info_type_for(strategy) == "expect_error":
        return "expect_error"
    return "quantile"


@pytest.mark.parametrize("client_with_strategy", STRATEGIES, indirect=True)
def test_repredict_asks_for_sketch_kind(client_with_strategy) -> None:
    client, strategy = client_with_strategy
    _queued_task(client, f"task-repredict-kind-{strategy}")

    predict = AsyncMock(return_value=[_prediction()])
    with patch("swarmpilot.scheduler.api.predictor_client.predict", predict):
        r = client.post("/v1/task/repredict")

    assert r.status_code == 200, r.text
    assert r.json()["repredicted"] >= 1
    kinds = {c.kwargs["prediction_type"] for c in predict.call_args_list}
    assert kinds == {_expected_type(strategy)}
    for call in predict.call_args_list:
        if _expected_type(strategy) == "expect_error":
            assert call.kwargs["quantiles"] is None
        else:
            assert call.kwargs["quantiles"]


@pytest.mark.parametrize("client_with_strategy", STRATEGIES, indirect=True)
def test_update_metadata_asks_for_sketch_kind(client_with_strategy) -> None:
    client, strategy = client_with_strategy
    task_id = f"task-update-kind-{strategy}"
    _queued_task(client, task_id)

    predict = AsyncMock(return_value=[_prediction()])
    with patch("swarmpilot.scheduler.api.predictor_client.predict", predict):
        r = client.post(
            "/v1/task/update_metadata",
            json={"updates": [{"task_id": task_id, "metadata": {"h": 1}}]},
        )

    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["queue_updated"] is True
    predict.assert_awaited_once()
    assert predict.call_args.kwargs["prediction_type"] == _expected_type(strategy)


@pytest.mark.parametrize("client_with_strategy", ["po2", "serverless"], indirect=True)
def test_repredict_keeps_error_margin_in_expect_error_sketch(
    client_with_strategy,
) -> None:
    # Before the fix po2 and serverless charged quantile predictions, and
    # the expect-error sketch lost the task's error margin.
    import swarmpilot.scheduler.api as api_module

    client, strategy = client_with_strategy
    task_id = f"task-margin-{strategy}"
    _queued_task(client, task_id)

    async def predict(**kwargs) -> list[Prediction]:
        # Like the real predictor: only expect-error answers carry a margin.
        pred = _prediction()
        if kwargs["prediction_type"] != "expect_error":
            pred.error_margin_ms = None
        return [pred]

    with patch(
        "swarmpilot.scheduler.api.predictor_client.predict",
        new=AsyncMock(side_effect=predict),
    ):
        r = client.post(
            "/v1/task/update_metadata",
            json={"updates": [{"task_id": task_id, "metadata": {"h": 2}}]},
        )
    assert r.status_code == 200, r.text

    async def margin() -> float:
        queue = await api_module.instance_registry.get_queue_info(INSTANCE_ID)
        return queue.error_margin_ms

    assert asyncio.run(margin()) > 0
