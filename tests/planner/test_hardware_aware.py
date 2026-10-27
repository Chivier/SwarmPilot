"""Hardware-aware auto-optimizer: learned speeds, victims and placement."""

from dataclasses import dataclass

import pytest

from swarmpilot.planner.pylet.deployment_executor import DeploymentPlan
from swarmpilot.planner.services.auto_optimizer import AutoOptimizer
from swarmpilot.planner.services.demand import DemandStore
from swarmpilot.planner.services.hardware_speed import HardwareSpeeds

FAST, SLOW = "A100", "A10G"


@dataclass
class _Inst:
    model_id: str
    endpoint: str
    hardware: str
    worker_id: str
    pylet_id: str
    created_at: float = 0.0


class _Service:
    def __init__(self, layout: list[tuple[str, str]]):
        self.instances = [
            _Inst(m, f"http://w{i}:80", hw, f"w{i}", f"p{i}", float(i))
            for i, (m, hw) in enumerate(layout)
        ]
        self.applied: list[tuple] = []

    def get_active_instances(self):
        return list(self.instances)

    def apply_deployment(
        self, target_state, wait_for_ready=True, remove=(), place=None
    ):
        self.applied.append((dict(target_state), tuple(remove), place))


def _store(service, exec_s, rates, backlog=None) -> DemandStore:
    """Execution time per instance from ``exec_s[(model, hardware)]``."""
    store = DemandStore(alpha=1.0)
    for inst in service.instances:
        store.submit_throughput(
            inst.endpoint, exec_s[(inst.model_id, inst.hardware)]
        )
    for m, r in rates.items():
        n = (backlog or {}).get(m, 0)
        store.submit_target(m, n, arrival_rate=r)
    return store


def _opt(store, service, **kw) -> AutoOptimizer:
    kw.setdefault("interval", 10.0)
    kw.setdefault("change_factor", 1.0)
    kw.setdefault("hardware_aware", True)
    return AutoOptimizer(store, service, **kw)


# A10G is 4x slower for both models.
EXEC = {("a", FAST): 1.0, ("a", SLOW): 4.0, ("b", FAST): 2.0, ("b", SLOW): 8.0}


class TestHardwareSpeeds:
    """Fitting hardware speed from execution times."""

    def test_separates_model_weight_from_hardware(self):
        speeds = HardwareSpeeds(alpha=1.0)
        speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0), ("b", FAST, 3.0)])
        assert speeds.speed(FAST) == pytest.approx(1.0)
        assert speeds.speed(SLOW) == pytest.approx(0.25)
        assert speeds.speed("H100") is None

    def test_one_hardware_model_keeps_what_was_learned(self):
        speeds = HardwareSpeeds(alpha=1.0)
        speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0)])
        # Each model now runs on one kind only: no new information.
        speeds.observe([("a", FAST, 1.5), ("b", SLOW, 9.0)])
        assert speeds.speed(SLOW) == pytest.approx(0.25)

    def test_speed_has_a_floor(self):
        speeds = HardwareSpeeds(alpha=1.0, floor=0.1)
        speeds.observe([("a", FAST, 1.0), ("a", SLOW, 1000.0)])
        assert speeds.speed(SLOW) == 0.1


class TestShrink:
    """Which instances a shrinking pool drops."""

    def test_removes_the_slowest_instance(self):
        service = _Service(
            [("a", SLOW), ("a", FAST), ("a", FAST), ("b", FAST), ("b", SLOW)]
        )
        # a needs 1 s * 1/s = 1 task in flight; b 2. Plenty of surplus.
        store = _store(service, EXEC, {"a": 1.0, "b": 1.0})
        opt = _opt(
            store,
            service,
            elastic=True,
            target_concurrency=4,
            scale_down_rounds=1,
            change_factor=0.2,
        )
        plan = opt.plan(service.instances, slots=[])
        assert plan is not None and plan.size_after == 4
        slow = {i.pylet_id for i in service.instances if i.hardware == SLOW}
        assert set(plan.remove) <= slow and len(plan.remove) == 1

    def test_floor_keeps_the_fastest(self):
        service = _Service([("a", FAST), ("a", SLOW), ("b", FAST)])
        store = _store(service, EXEC, {"a": 0.0, "b": 0.0})
        store.submit_target("a", 0, arrival_rate=0.0)
        opt = _opt(
            store, service, elastic=True, scale_down_rounds=1, min_instances=1
        )
        plan = opt.plan(service.instances, slots=[])
        assert plan is not None
        assert plan.remove == ("p1",)
        assert plan.target_state == {"a": 1, "b": 1}


class TestGrow:
    """Which free GPUs a growing pool takes."""

    def test_grows_into_the_fastest_free_gpu_first(self):
        service = _Service([("a", FAST), ("b", FAST)])
        store = _store(service, EXEC, {"a": 3.0, "b": 0.5})
        opt = _opt(store, service, elastic=True, target_concurrency=1)
        opt.speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0)])
        # Need is 3 + 1 = 4 units; the pool has 2. One fast GPU (1 unit)
        # and three slow ones (0.25 each) are free: the fast one and all
        # three slow ones still fall short, so all four are used.
        slots = [("s0", SLOW), ("f0", FAST), ("s1", SLOW), ("s2", SLOW)]
        plan = opt.plan(service.instances, slots=slots)
        assert plan is not None and plan.size_after == 2 + 4
        # Growing moves nothing that already runs.
        assert plan.remove == ()
        # With fewer units needed, the fast GPU is taken first.
        store.submit_target("a", 0, arrival_rate=2.0)
        plan = opt.plan(service.instances, slots=slots)
        assert plan is not None and plan.size_after == 3
        assert plan.place == {"a": ("f0",)}

    def test_counts_capacity_not_instances(self):
        # Four slow instances are one fast one: the pool must grow.
        service = _Service([("a", SLOW)] * 4 + [("b", FAST)])
        exec_s = dict(EXEC)
        store = _store(service, exec_s, {"a": 0.5, "b": 0.5})
        # Learn that A10G is slow from b once running on both.
        opt = _opt(store, service, elastic=True, target_concurrency=1)
        opt.speeds.observe([("b", FAST, 2.0), ("b", SLOW, 8.0)])
        # a needs 0.5 * mean(4 s) = 2 units, b 1: 3 units, the pool has 2.
        plan = opt.plan(service.instances, slots=[("f1", FAST)])
        assert plan is not None and plan.size_after == 6


class TestOverloaded:
    """Allocation when the pool cannot carry all the need."""

    def test_arrivals_before_backlog(self):
        # a has a huge backlog; b only its steady arrivals. The pool
        # cannot carry both, so b must keep what its arrivals need.
        service = _Service([("a", FAST)] * 3 + [("b", FAST)] * 3)
        exec_s = {("a", FAST): 1.0, ("b", FAST): 1.0}
        store = _store(
            service, exec_s, {"a": 2.0, "b": 2.0}, backlog={"a": 1000}
        )
        opt = _opt(store, service, target_concurrency=1)
        plan = opt.plan(service.instances)
        assert plan is not None
        # Supply is 6 units; b's arrivals need 2 of them.
        assert plan.target_state["b"] >= 2
        assert plan.demand["b"] == pytest.approx(2.0)

    def test_count_share_without_flag_starves_b(self):
        service = _Service([("a", FAST)] * 3 + [("b", FAST)] * 3)
        exec_s = {("a", FAST): 1.0, ("b", FAST): 1.0}
        store = _store(
            service, exec_s, {"a": 2.0, "b": 2.0}, backlog={"a": 1000}
        )
        opt = _opt(store, service, target_concurrency=1, hardware_aware=False)
        plan = opt.plan(service.instances)
        assert plan is not None and plan.target_state["b"] == 1


class TestTrade:
    """Models trading machines of different speed."""

    def test_models_can_trade_machines(self):
        # b is heavy and sits on slow GPUs; a is light and on fast ones.
        service = _Service([("a", FAST), ("a", FAST), ("b", SLOW), ("b", SLOW)])
        store = _store(service, EXEC, {"a": 1.0, "b": 2.0})
        opt = _opt(store, service, target_concurrency=1)
        plan = opt.plan(service.instances)
        assert plan is not None
        # Counts may stay 2:2, but b should end up on a fast GPU.
        assert any(w in ("w0", "w1") for w in plan.place.get("b", ()))
        assert plan.remove


class TestWaitForReports:
    """The first rounds wait for every model's report."""

    def test_waits_until_every_model_reported(self):
        service = _Service([("a", FAST)] * 2 + [("b", FAST)] * 2)
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 10)
        opt = AutoOptimizer(store, service, interval=10.0)
        assert opt.plan(service.instances) is None
        store.submit_target("b", 0)
        assert opt.plan(service.instances) is not None

    def test_stops_waiting_after_stale_after(self):
        service = _Service([("a", FAST)] * 2 + [("b", FAST)] * 2)
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 10)
        opt = AutoOptimizer(store, service, interval=10.0)
        assert opt.plan(service.instances) is None
        opt._first_plan_at -= opt.stale_after
        assert opt.plan(service.instances) is not None


class TestDeploymentPlan:
    """Named victims and workers in the executor's plan."""

    @staticmethod
    def _by_model(service):
        out: dict = {}
        for inst in service.instances:
            out.setdefault(inst.model_id, []).append(inst)
        return out

    def test_named_victims_and_workers(self):
        service = _Service([("a", FAST), ("a", SLOW), ("b", FAST)])
        plan = DeploymentPlan.from_diff(
            {"a": 2, "b": 1},
            {"a": 1, "b": 2},
            self._by_model(service),
            remove=("p0",),
            place={"b": ("w0",)},
        )
        removes = [a for a in plan.actions if a.action_type == "remove"]
        adds = [a for a in plan.actions if a.action_type == "add"]
        assert removes[0].pylet_ids == ["p0"]
        assert [(a.model_id, a.target_worker, a.count) for a in adds] == [
            ("b", "w0", 1)
        ]

    def test_trade_keeps_counts(self):
        service = _Service([("a", FAST), ("b", SLOW)])
        plan = DeploymentPlan.from_diff(
            {"a": 1, "b": 1},
            {"a": 1, "b": 1},
            self._by_model(service),
            remove=("p0", "p1"),
            place={"a": ("w1",), "b": ("w0",)},
        )
        assert plan.total_removes == 2 and plan.total_adds == 2

    def test_without_victims_removes_the_oldest(self):
        service = _Service([("a", FAST), ("a", SLOW)])
        plan = DeploymentPlan.from_diff(
            {"a": 2}, {"a": 1}, self._by_model(service)
        )
        assert plan.actions[0].pylet_ids == ["p0"]


def test_removals_drain_side_by_side():
    """Three removals take one drain, not three."""
    import time
    from unittest.mock import MagicMock

    from swarmpilot.planner.pylet.deployment_executor import (
        DeploymentAction,
        DeploymentExecutor,
    )

    manager = MagicMock()
    manager.terminate_instance.side_effect = lambda pid: time.sleep(0.3) or True
    executor = DeploymentExecutor(manager)
    plan = DeploymentPlan(
        actions=[
            DeploymentAction("remove", "a", 2, pylet_ids=["p0", "p1"]),
            DeploymentAction("remove", "b", 1, pylet_ids=["p2"]),
        ],
        total_removes=3,
    )
    start = time.monotonic()
    result = executor.execute_plan(plan, wait_for_ready=False)
    assert time.monotonic() - start < 0.6
    assert sorted(result.removed_instances) == ["p0", "p1", "p2"]
