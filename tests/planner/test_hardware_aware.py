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
        opt.speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0)])
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


class TestChurn:
    """Moves must pay for themselves."""

    def test_new_instances_stay_put(self):
        import time

        service = _Service([("a", FAST)] * 2 + [("b", FAST)] * 2)
        store = _store(
            service, {("a", FAST): 1.0, ("b", FAST): 1.0}, {"a": 9, "b": 1}
        )
        for inst in service.instances:
            inst.created_at = time.time()
        assert _opt(store, service).plan(service.instances) is None
        opt = _opt(store, service, min_age=0)
        assert opt.plan(service.instances) is not None

    def test_each_move_must_clear_the_threshold(self):
        service = _Service([("a", FAST)] * 5 + [("b", FAST)] * 5)
        store = _store(
            service, {("a", FAST): 1.0, ("b", FAST): 1.0}, {"a": 6, "b": 4}
        )
        # One move gains 0.2 of share error: enough at 0.1, not at 0.25.
        assert _opt(store, service, change_threshold=0.1).plan(
            service.instances
        )
        assert (
            _opt(store, service, change_threshold=0.25).plan(service.instances)
            is None
        )


def test_light_model_is_not_starved_by_a_heavy_one():
    # b's arrivals alone need far more than the pool; a's need little.
    service = _Service([("a", FAST)] * 2 + [("b", FAST)] * 8)
    store = _store(
        service, {("a", FAST): 1.0, ("b", FAST): 1.0}, {"a": 2.0, "b": 100.0}
    )
    plan = _opt(store, service, target_concurrency=1).plan(service.instances)
    # Proportional shares would give a 2% of 10 machines; max-min gives
    # it the 2 it needs.
    assert plan is None or plan.target_state.get("a", 0) >= 2


class TestTradeSafety:
    """A trade never leaves a model with nothing serving."""

    def test_one_machine_model_does_not_trade_away_its_only_one(self):
        # The best shares put heavy b on a's fast GPU and a on a slow one,
        # but removals drain first: a would serve nothing in between.
        service = _Service([("a", FAST), ("b", SLOW), ("b", SLOW)])
        store = _store(service, EXEC, {"a": 0.2, "b": 3.0})
        opt = _opt(store, service, target_concurrency=1)
        opt.speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0)])
        assert opt.plan(service.instances) is None

    def test_trade_keeps_a_floor_on_unmoved_machines(self):
        service = _Service([("a", FAST), ("a", FAST), ("b", SLOW), ("b", SLOW)])
        store = _store(service, EXEC, {"a": 1.0, "b": 2.0})
        opt = _opt(store, service, target_concurrency=1)
        opt.speeds.observe([("a", FAST, 1.0), ("a", SLOW, 4.0)])
        plan = opt.plan(service.instances)
        assert plan is not None and plan.remove
        for model in ("a", "b"):
            mine = [i for i in service.instances if i.model_id == model]
            staying = [i for i in mine if i.pylet_id not in plan.remove]
            assert len(staying) >= 1, model

    def test_min_age_counts_from_serving(self):
        import time

        service = _Service([("a", FAST)] * 2 + [("b", FAST)] * 2)
        store = _store(
            service, {("a", FAST): 1.0, ("b", FAST): 1.0}, {"a": 9, "b": 1}
        )
        for inst in service.instances:
            inst.created_at = 0.0  # submitted long ago...
            inst.active_since = time.time()  # ...but serving only now
        assert _opt(store, service).plan(service.instances) is None


@dataclass
class _Sized(_Inst):
    gpu_count: int = 1


class _ShapedService(_Service):
    """Models of different GPU counts and hardware patterns."""

    def __init__(self, layout, reqs):
        self.instances = [
            _Sized(m, f"http://w{i}:80", hw, f"w{i}", f"p{i}", 0.0, reqs[m][0])
            for i, (m, hw) in enumerate(layout)
        ]
        self.applied = []
        self.reqs = reqs

    def model_requirements(self, model_id):
        return self.reqs.get(model_id, (1, None))


# "big" needs 4 GPUs of A100 (the 4-GPU workers); "small" one of any.
SHAPES = {"small": (1, None), "big": (4, "A100*")}
SHAPED_EXEC = {
    ("small", FAST): 1.0,
    ("small", SLOW): 4.0,
    ("big", FAST): 2.0,
    ("big", SLOW): 8.0,
}


class TestShapes:
    """Models that take several GPUs or one kind of hardware."""

    def test_big_model_grows_only_where_it_fits(self):
        service = _ShapedService([("small", FAST), ("big", FAST)], SHAPES)
        store = _store(service, SHAPED_EXEC, {"small": 0.1, "big": 4.0})
        opt = _opt(store, service, elastic=True, target_concurrency=1)
        # Single free GPUs everywhere, and one A100 worker with four.
        slots = [("a1", FAST)] + [("a4", FAST)] * 4 + [("s1", SLOW)] * 2
        plan = opt.plan(service.instances, slots=slots)
        assert plan is not None
        assert plan.place == {"big": ("a4",)}
        assert plan.target_state["big"] == 2

    def test_pinned_model_never_lands_on_other_hardware(self):
        service = _ShapedService([("small", FAST), ("big", FAST)], SHAPES)
        store = _store(service, SHAPED_EXEC, {"small": 0.1, "big": 4.0})
        opt = _opt(store, service, elastic=True, target_concurrency=1)
        plan = opt.plan(service.instances, slots=[("s4", SLOW)] * 4)
        # Four free A10Gs hold a 4-GPU instance, but not of A100: big
        # cannot grow, and small has all it needs.
        assert plan is None or "big" not in plan.place

    def test_machines_trade_only_between_equal_shapes(self):
        # big is starved, small has surplus; no free GPUs.
        service = _ShapedService(
            [("small", FAST)] * 4 + [("big", FAST)], SHAPES
        )
        store = _store(service, SHAPED_EXEC, {"small": 0.5, "big": 10.0})
        opt = _opt(store, service, target_concurrency=1)
        plan = opt.plan(service.instances)
        # A 1-GPU machine cannot become a 4-GPU instance.
        assert plan is None or plan.target_state.get("big", 0) <= 1

    def test_count_scaler_keeps_shapes_apart(self):
        service = _ShapedService(
            [("small", FAST)] * 4 + [("big", FAST)], SHAPES
        )
        store = _store(service, SHAPED_EXEC, {"small": 0.5, "big": 10.0})
        opt = _opt(store, service, hardware_aware=False)
        plan = opt.plan(service.instances)
        assert plan is None or plan.target_state.get("big", 0) <= 1

    def test_small_shortfall_takes_the_slowest_place_that_covers_it(self):
        service = _ShapedService([("small", FAST), ("big", FAST)], SHAPES)
        opt = _opt(
            _store(service, SHAPED_EXEC, {"small": 1.2, "big": 0.5}),
            service,
            elastic=True,
            target_concurrency=1,
        )
        opt.speeds.observe([("small", FAST, 1.0), ("small", SLOW, 4.0)])
        # small is 0.2 short: a slow GPU (0.25) covers it, so the fast
        # A100s stay free; of the slow ones, the single-GPU worker.
        slots = [("a4", FAST)] * 4 + [("s4", SLOW)] * 4 + [("s1", SLOW)]
        plan = opt.plan(service.instances, slots=slots)
        assert plan is not None
        assert plan.place == {"small": ("s1",)}


class TestShapeRegressions:
    """Cases from the review of the shape-aware loop."""

    def test_floor_swap_never_moves_a_pinned_model_off_its_hardware(self):
        shapes = {"m": (1, None), "n": (1, None), "pin": (1, "A100*")}
        exec_s = {}
        for model in shapes:
            exec_s[(model, FAST)] = 1.0
            exec_s[(model, SLOW)] = 4.0
        layouts = [
            [("m", FAST), ("n", SLOW), ("n", SLOW), ("n", FAST), ("pin", FAST)],
            [("m", FAST), ("n", SLOW), ("n", SLOW), ("pin", FAST)],
        ]
        for layout in layouts:
            for rates in (
                {"m": 0.3, "n": 0.1, "pin": 3.0},
                {"m": 1.0, "n": 0.1, "pin": 5.0},
            ):
                service = _ShapedService(layout, shapes)
                store = _store(service, exec_s, rates)
                opt = _opt(store, service, target_concurrency=1)
                opt.speeds.observe([("m", FAST, 1.0), ("m", SLOW, 4.0)])
                plan = opt.plan(service.instances, slots=[])
                if plan is None:
                    continue
                hw = {i.worker_id: i.hardware for i in service.instances}
                assert all(hw[w] == FAST for w in plan.place.get("pin", ()))

    def test_count_path_grows_models_of_any_shape(self):
        for shapes in (
            {"small": (2, None), "big": (2, None)},
            {"small": (1, None), "big": (1, "A100*")},
        ):
            service = _ShapedService([("small", FAST), ("big", FAST)], shapes)
            store = _store(service, SHAPED_EXEC, {"small": 0.1, "big": 4.0})
            opt = _opt(
                store,
                service,
                hardware_aware=False,
                elastic=True,
                target_concurrency=1,
            )
            plan = opt.plan(service.instances, spare=4)
            assert plan is not None and plan.target_state["big"] > 1
