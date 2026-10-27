"""Demand reports and the auto-optimize loop (plan items 3.2 and 3.3)."""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from swarmpilot.planner.services.auto_optimizer import AutoOptimizer
from swarmpilot.planner.services.demand import DemandStore, get_demand_store


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass
class _Inst:
    model_id: str
    endpoint: str | None


class _Service:
    """Applies a target state at once, keeping surviving endpoints."""

    def __init__(self, counts: dict[str, int]):
        self.instances: list[_Inst] = []
        self.applied: list[dict[str, int]] = []
        self._next = 0
        self._set(counts)

    def _set(self, counts: dict[str, int]) -> None:
        keep: list[_Inst] = []
        for m, n in counts.items():
            have = [i for i in self.instances if i.model_id == m][:n]
            while len(have) < n:
                self._next += 1
                have.append(_Inst(m, f"http://{m}-{self._next}:80"))
            keep.extend(have)
        self.instances = keep

    def get_active_instances(self) -> list[_Inst]:
        return list(self.instances)

    def apply_deployment(self, target_state, wait_for_ready=True):
        self.applied.append(dict(target_state))
        self._set(target_state)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.instances:
            out[i.model_id] = out.get(i.model_id, 0) + 1
        return out


def _optimizer(store, service, **kw) -> AutoOptimizer:
    kw.setdefault("interval", 10.0)
    return AutoOptimizer(store, service, **kw)


class TestDemandStore:
    """Smoothing and staleness of reports."""

    def test_smooths_and_expires(self):
        clock = _Clock()
        store = DemandStore(alpha=0.5, clock=clock)
        assert store.submit_target("a", 10) == 10
        assert store.submit_target("a", 20) == 15
        clock.t = 31
        assert store.targets(max_age=30) == {}
        assert store.targets() == {"a": 15}

    def test_execution_time_by_normalized_url(self):
        store = DemandStore(alpha=1.0)
        store.submit_throughput("http://x:80/", 2.0)
        assert store.execution_time("http://x:80") == 2.0
        assert store.execution_time("http://y:80") is None

    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            DemandStore(alpha=0)


class TestEndpoints:
    """The Planner endpoints the Scheduler reporter calls."""

    def test_reports_change_planner_state(self):
        from swarmpilot.planner.api import app

        get_demand_store().clear()
        with TestClient(app) as client:
            r = client.post(
                "/v1/submit_target", json={"model_id": "m1", "value": 4}
            )
            assert r.status_code == 200 and r.json()["smoothed"] == 4
            r = client.post(
                "/v1/submit_throughput",
                json={"instance_url": "http://i:1", "avg_execution_time": 0.5},
            )
            assert r.status_code == 200
            snap = client.get("/v1/demand").json()
        assert snap["targets"] == {"m1": 4}
        assert snap["execution_times"] == {"http://i:1": 0.5}
        get_demand_store().clear()

    @pytest.mark.parametrize(
        "path,body",
        [
            ("/v1/submit_target", {"model_id": "m", "value": -1}),
            ("/v1/submit_target", {"model_id": "", "value": 1}),
            (
                "/v1/submit_throughput",
                {"instance_url": "http://i:1", "avg_execution_time": 0},
            ),
        ],
    )
    def test_rejects_invalid_reports(self, path, body):
        from swarmpilot.planner.api import app

        with TestClient(app) as client:
            assert client.post(path, json=body).status_code == 422


class TestPlan:
    """One round's decision."""

    def test_no_demand_no_plan(self):
        service = _Service({"a": 2, "b": 2})
        opt = _optimizer(DemandStore(), service)
        assert opt.plan(service.get_active_instances()) is None

    def test_moves_toward_demand_within_cap(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 9)
        store.submit_target("b", 1)
        service = _Service({"a": 5, "b": 5})
        opt = _optimizer(store, service, change_factor=0.2)
        plan = opt.plan(service.get_active_instances())
        assert plan is not None
        assert plan.target_state == {"a": 7, "b": 3}
        assert plan.changes == 2
        assert plan.score_after < plan.score_before

    def test_small_gain_is_ignored(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 52)
        store.submit_target("b", 48)
        service = _Service({"a": 5, "b": 5})
        assert _optimizer(store, service).plan(service.instances) is None

    def test_quiet_model_keeps_a_replica(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 10)
        store.submit_target("b", 0)
        service = _Service({"a": 2, "b": 2})
        plan = _optimizer(store, service, change_factor=1.0).plan(
            service.instances
        )
        assert plan is not None
        assert plan.target_state == {"a": 3, "b": 1}

    def test_execution_time_weights_capacity(self):
        # Equal backlog, but b's tasks take 3x longer: b needs 3x replicas.
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 10)
        store.submit_target("b", 10)
        service = _Service({"a": 4, "b": 4})
        for inst in service.instances:
            store.submit_throughput(
                inst.endpoint, 1.0 if inst.model_id == "a" else 3.0
            )
        plan = _optimizer(store, service, change_factor=1.0).plan(
            service.instances
        )
        assert plan is not None
        assert plan.target_state == {"a": 2, "b": 6}

    def test_new_model_with_demand_gets_replicas(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 1)
        store.submit_target("c", 1)
        service = _Service({"a": 4})
        plan = _optimizer(store, service, change_factor=1.0).plan(
            service.instances
        )
        assert plan is not None
        assert plan.target_state == {"a": 2, "c": 2}


async def test_step_change_converges_without_oscillation():
    """Fluid simulation of plan item 3.3's acceptance check.

    Two models on 8 machines, 1 s per task. Demand steps from even to 7:1.
    The Scheduler reports outstanding work (backlog plus tasks in service)
    each interval. Replicas must settle within 3 intervals, then hold.
    """
    clock = _Clock()
    store = DemandStore(alpha=0.5, clock=clock)
    service = _Service({"a": 4, "b": 4})
    opt = _optimizer(store, service, change_factor=0.3)
    interval = opt.interval
    backlog = {"a": 0.0, "b": 0.0}
    history = []
    for step in range(16):
        rate = {"a": 3.0, "b": 3.0} if step < 3 else {"a": 5.25, "b": 0.75}
        counts = service.counts()
        for m in backlog:
            served = counts.get(m, 0) * interval
            backlog[m] = max(0.0, backlog[m] + (rate[m] * interval) - served)
            in_service = min(rate[m], counts.get(m, 0))
            store.submit_target(m, backlog[m] + in_service)
        clock.t += interval
        await opt.run_once()
        history.append(service.counts())

    settled = history[3 + 2]  # at most 3 intervals after the step
    assert all(h == settled for h in history[3 + 2 :]), history
    assert settled == {"a": 7, "b": 1}, history
    assert sum(backlog.values()) == 0.0
    assert len(service.applied) <= 3


async def test_loop_starts_and_stops():
    store = DemandStore(alpha=1.0)
    service = _Service({"a": 1})
    opt = _optimizer(store, service, interval=0.01)
    opt.start()
    opt.start()  # idempotent
    await opt.stop()
    assert opt._task is None
