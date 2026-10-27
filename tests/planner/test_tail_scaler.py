"""Tail- and call-structure-aware scaler (plan item 3.4)."""

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from swarmpilot.planner.services.auto_optimizer import AutoOptimizer
from swarmpilot.planner.services.demand import (
    MAX_CALL_RATIO,
    DemandStore,
    get_demand_store,
)

from .test_auto_optimizer import _Service


def _store(service: _Service, exec_s: dict[str, float]) -> DemandStore:
    store = DemandStore(alpha=1.0)
    for inst in service.instances:
        store.submit_throughput(inst.endpoint, exec_s[inst.model_id])
    return store


def _opt(store, service, **kw) -> AutoOptimizer:
    kw.setdefault("interval", 10.0)
    kw.setdefault("change_factor", 1.0)
    return AutoOptimizer(store, service, **kw)


class TestDemandStore:
    """Rates, tail work and call ratios."""

    def test_call_ratio_is_calls_per_caller_task(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("llm", 0, arrival_rate=2.0)
        store.submit_target(
            "tool", 0, arrival_rate=3.0, calls_from={"llm": 3.0}
        )
        assert store.call_ratios() == {"llm": {"tool": 1.5}}

    def test_caller_that_stops_calling_decays_to_zero(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("llm", 0, arrival_rate=2.0)
        store.submit_target("tool", 0, arrival_rate=1, calls_from={"llm": 1})
        store.submit_target("tool", 0, arrival_rate=1, calls_from={})
        assert store.call_ratios() == {}

    def test_ratio_is_capped(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("llm", 0, arrival_rate=0.001)
        store.submit_target("tool", 0, arrival_rate=9, calls_from={"llm": 9})
        assert store.call_ratios()["llm"]["tool"] == MAX_CALL_RATIO

    def test_old_report_leaves_rates_alone(self):
        store = DemandStore(alpha=1.0)
        store.submit_target("a", 3)
        assert store.arrival_rates() == {} and store.work() == {}


class TestEndpoint:
    """The extended /v1/submit_target body."""

    def test_new_fields_reach_the_snapshot(self):
        from swarmpilot.planner.api import app

        get_demand_store().clear()
        with TestClient(app) as client:
            for body in (
                {"model_id": "llm", "value": 2, "arrival_rate": 4.0},
                {
                    "model_id": "tool",
                    "value": 1,
                    "arrival_rate": 2.0,
                    "work_s": 7.5,
                    "calls_from": {"llm": 2.0},
                },
            ):
                r = client.post("/v1/submit_target", json=body)
                assert r.status_code == 200, r.text
            snap = client.get("/v1/demand").json()
        get_demand_store().clear()
        assert snap["arrival_rates"] == {"llm": 4.0, "tool": 2.0}
        assert snap["work_s"] == {"tool": 7.5}
        assert snap["call_ratios"] == {"llm": {"tool": 0.5}}

    @pytest.mark.parametrize(
        "extra",
        [
            {"arrival_rate": -1},
            {"work_s": -1},
            {"calls_from": {"llm": -1}},
            {"calls_from": {"": 1}},
        ],
    )
    def test_rejects_invalid_fields(self, extra):
        from swarmpilot.planner.api import app

        with TestClient(app) as client:
            r = client.post(
                "/v1/submit_target",
                json={"model_id": "m", "value": 1, **extra},
            )
            assert r.status_code == 422


class TestPlan:
    """How the tail scaler sizes models."""

    def test_model_that_keeps_up_keeps_its_share(self):
        # Same arrival rate; a keeps up (short queue), b is backlogged.
        # The count scaler reads a's short queue as no demand.
        service = _Service({"a": 4, "b": 4})
        store = _store(service, {"a": 1.0, "b": 1.0})
        store.submit_target("a", 0.5, arrival_rate=1.0)
        store.submit_target("b", 8.0, arrival_rate=1.0)

        count = _opt(store, service, scaler="count").plan(service.instances)
        assert count is not None and count.target_state == {"a": 1, "b": 7}

        tail = _opt(store, service).plan(service.instances)
        assert tail is not None and tail.scaler == "tail"
        # need = rate * s + backlog / H: a 1.05, b 1.8.
        assert tail.demand == pytest.approx({"a": 1.05, "b": 1.8})
        assert tail.target_state == {"a": 3, "b": 5}

    def test_tail_work_adds_headroom(self):
        service = _Service({"a": 4, "b": 4})
        store = _store(service, {"a": 1.0, "b": 1.0})
        store.submit_target("a", 5, arrival_rate=1.0, work_s=5.0)
        store.submit_target("b", 5, arrival_rate=1.0, work_s=45.0)
        plan = _opt(store, service).plan(service.instances)
        assert plan is not None
        assert plan.demand == pytest.approx({"a": 1.5, "b": 5.5})
        assert plan.target_state == {"a": 2, "b": 6}

    @pytest.mark.parametrize("llm_replicas,tool_need", [(6, 6.0), (2, 2.0)])
    def test_callee_follows_what_callers_can_finish(
        self, llm_replicas, tool_need
    ):
        # llm: 3/s arriving, 40 queued. Each llm task calls tool once.
        # Over the 10 s horizon llm can finish 3 + 40/10 = 7/s, but no
        # more than its replicas serve (1 task/s each).
        service = _Service({"llm": llm_replicas, "tool": 4})
        store = _store(service, {"llm": 1.0, "tool": 1.0})
        store.submit_target("llm", 40, arrival_rate=3.0)
        store.submit_target(
            "tool", 0, arrival_rate=3.0, calls_from={"llm": 3.0}
        )
        demand = _opt(store, service)._tail_demand(
            store.targets(),
            store.arrival_rates(),
            {"llm": 1.0, "tool": 1.0},
            {"llm": llm_replicas, "tool": 4},
        )
        assert demand["llm"] == pytest.approx(3.0 + 40 / 10)
        assert demand["tool"] == pytest.approx(tool_need)

    def test_call_cycle_converges(self):
        # External 1/s into llm; llm -> tool 0.8, tool -> llm 0.5.
        service = _Service({"llm": 10, "tool": 10})
        store = _store(service, {"llm": 1.0, "tool": 2.0})
        llm, tool = 1 / (1 - 0.4), 0.8 / (1 - 0.4)
        store.submit_target(
            "llm", 0, arrival_rate=llm, calls_from={"tool": 0.5 * tool}
        )
        store.submit_target(
            "tool", 0, arrival_rate=tool, calls_from={"llm": 0.8 * llm}
        )
        opt = _opt(store, service)
        demand = opt._tail_demand(
            store.targets(),
            store.arrival_rates(),
            {"llm": 1.0, "tool": 2.0},
            {"llm": 10, "tool": 10},
        )
        assert demand == pytest.approx({"llm": llm, "tool": 2 * tool})

    def test_falls_back_to_count_for_old_schedulers(self):
        service = _Service({"a": 4, "b": 4})
        store = _store(service, {"a": 1.0, "b": 1.0})
        store.submit_target("a", 9, arrival_rate=1.0)
        store.submit_target("b", 1)  # an older Scheduler: no rate
        plan = _opt(store, service).plan(service.instances)
        assert plan is not None and plan.scaler == "count"

    def test_rejects_unknown_scaler(self):
        with pytest.raises(ValueError):
            _opt(DemandStore(), _Service({}), scaler="magic")


def _agent_scaling():
    path = Path(__file__).parents[1] / "perf" / "agent_scaling.py"
    spec = importlib.util.spec_from_file_location("agent_scaling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scenario", ["short_steps", "long_steps"])
def test_fewer_slo_misses_than_count_scaler(scenario):
    """Acceptance for 3.4 on the agent-workload simulation."""
    sim = _agent_scaling()
    count = sim.simulate("count", 0, scenario)
    tail = sim.simulate("tail", 0, scenario)
    assert tail["slo_miss_rate"] < 0.5 * count["slo_miss_rate"], (
        count,
        tail,
    )


def test_backlog_never_below_outstanding_mean():
    # No trained predictor yet: the sketches report no work.
    service = _Service({"a": 2, "b": 2})
    store = _store(service, {"a": 1.0, "b": 2.0})
    store.submit_target("a", 0, arrival_rate=1.0, work_s=0.0)
    store.submit_target("b", 10, arrival_rate=1.0, work_s=0.0)
    demand = _opt(store, service)._tail_demand(
        store.targets(),
        store.arrival_rates(),
        {"a": 1.0, "b": 2.0},
        {"a": 2, "b": 2},
    )
    assert demand == pytest.approx({"a": 1.0, "b": 2.0 + 10 * 2.0 / 10})


def test_reused_port_starts_a_fresh_execution_time():
    # PyLet hands a port that served the diffusion model to the LLM.
    store = DemandStore(alpha=0.5)
    store.submit_throughput("http://w:15301", 0.8, model_id="diff")
    assert store.execution_time("http://w:15301", model_id="llm") is None
    assert store.submit_throughput("http://w:15301", 0.3, model_id="llm") == 0.3
    assert store.execution_time("http://w:15301", model_id="llm") == 0.3
    # Older Schedulers send no model: readings are averaged as before.
    store.submit_throughput("http://w:1", 1.0)
    assert store.submit_throughput("http://w:1", 2.0) == 1.5


def test_pylet_endpoint_matches_scheduler_url():
    # PyLet's endpoints carry no scheme; the Scheduler's reports do.
    store = DemandStore(alpha=1.0)
    store.submit_throughput("http://127.0.0.1:15301/", 0.3, model_id="llm")
    assert store.execution_time("127.0.0.1:15301", model_id="llm") == 0.3


class TestElasticPool:
    """Pool sizing from tasks in flight per instance."""

    @staticmethod
    def _loaded(rate_a: float, rate_b: float, n: int = 2) -> tuple:
        service = _Service({"a": n, "b": n})
        store = _store(service, {"a": 1.0, "b": 1.0})
        # need = rate * 1 s, no backlog.
        store.submit_target("a", 0, arrival_rate=rate_a)
        store.submit_target("b", 0, arrival_rate=rate_b)
        return service, store

    def test_fixed_pool_never_grows(self):
        service, store = self._loaded(30, 10)
        plan = _opt(store, service, target_concurrency=4).plan(
            service.instances, spare=10
        )
        assert plan is not None and plan.size_after == 4

    def test_grows_into_free_gpus(self):
        service, store = self._loaded(30, 10)
        opt = _opt(store, service, elastic=True, target_concurrency=4)
        plan = opt.plan(service.instances, spare=10)
        # 40 tasks in flight / 4 per instance = 10, split 3:1.
        assert plan is not None and plan.size_after == 10
        assert plan.target_state == {"a": 7, "b": 3} or plan.target_state == {
            "a": 8,
            "b": 2,
        }

    def test_growth_is_capped_by_spare_and_max(self):
        service, store = self._loaded(30, 10)
        opt = _opt(store, service, elastic=True, target_concurrency=4)
        assert opt.plan(service.instances, spare=2).size_after == 6
        opt = _opt(
            store, service, elastic=True, target_concurrency=4, max_instances=5
        )
        assert opt.plan(service.instances, spare=10).size_after == 5

    def test_grows_even_when_shares_already_match(self):
        # Shares are right (1:1), so a fixed pool has nothing to gain.
        service, store = self._loaded(20, 20)
        assert (
            _opt(store, service, target_concurrency=4).plan(
                service.instances, spare=10
            )
            is None
        )
        opt = _opt(store, service, elastic=True, target_concurrency=4)
        plan = opt.plan(service.instances, spare=10)
        assert plan is not None and plan.target_state == {"a": 5, "b": 5}

    def test_shrinks_after_quiet_rounds_from_the_surplus(self):
        service, store = self._loaded(4, 1, n=4)
        opt = _opt(
            store,
            service,
            elastic=True,
            target_concurrency=4,
            scale_down_rounds=2,
            change_factor=0.5,
        )
        # First quiet round: only a fixed-pool rebalance.
        first = opt.plan(service.instances)
        assert first is not None and first.size_after == 8
        second = opt.plan(service.instances)
        # Wants ceil(5 / 4) = 2, but sheds at most half the pool per round
        # and keeps a replica per model; b has the most surplus.
        assert second is not None and second.size_after == 4
        assert second.target_state["b"] < second.target_state["a"]
        assert min(second.target_state.values()) >= 1

    def test_busy_round_resets_the_scale_down_count(self):
        service, store = self._loaded(4, 1, n=4)
        opt = _opt(
            store,
            service,
            elastic=True,
            target_concurrency=4,
            scale_down_rounds=2,
        )
        opt.plan(service.instances)
        store.submit_target("a", 0, arrival_rate=40)
        opt.plan(service.instances)
        store.submit_target("a", 0, arrival_rate=4)
        plan = opt.plan(service.instances)
        assert plan is None or plan.size_after == 8

    def test_never_below_min_instances(self):
        service, store = self._loaded(0.1, 0.1, n=3)
        opt = _opt(
            store,
            service,
            elastic=True,
            target_concurrency=4,
            scale_down_rounds=1,
            change_factor=1.0,
            min_instances=4,
        )
        plan = opt.plan(service.instances)
        assert plan is not None and plan.size_after == 4

    def test_count_scaler_is_never_elastic(self):
        service, store = self._loaded(30, 10)
        opt = _opt(
            store, service, scaler="count", elastic=True, target_concurrency=4
        )
        plan = opt.plan(service.instances, spare=10)
        assert plan is None or plan.size_after == 4

    async def test_run_once_asks_the_service_for_spare(self):
        service, store = self._loaded(30, 10)
        service.spare_capacity = lambda: 1
        opt = _opt(store, service, elastic=True, target_concurrency=4)
        plan = await opt.run_once()
        assert plan is not None and plan.size_after == 5
        assert sum(service.counts().values()) == 5
