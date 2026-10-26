"""Queue sketches return to empty (plan item 2.2).

Every routed task holds one charge on its instance's sketch, and the
registry removes exactly that charge when the task finishes, fails or
moves. These tests drive the registry directly, through a strategy, and
through a strategy switch.
"""

import random

import pytest

from swarmpilot.scheduler.algorithms import (
    MinimumExpectedTimeStrategy,
    ProbabilisticSchedulingStrategy,
)
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    Instance,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
from swarmpilot.scheduler.registry.queue_sketch import (
    DEFAULT_QUANTILES,
    Charge,
)


def _instance(iid: str) -> Instance:
    return Instance(
        instance_id=iid,
        model_id="m",
        endpoint=f"http://{iid}",
        platform_info={
            "software_name": "x",
            "software_version": "1",
            "hardware_name": "h",
        },
    )


async def _registry(kind: str, *iids: str) -> InstanceRegistry:
    reg = InstanceRegistry(queue_info_type=kind)
    for iid in iids:
        await reg.register(_instance(iid))
    return reg


def _random_charge(rng: random.Random, iid: str) -> Charge:
    median = rng.uniform(10, 1000)
    spread = rng.uniform(0.1, 1.0)
    return Charge.of(
        iid,
        median,
        error_ms=median * spread,
        quantiles={
            0.1: median * (1 - spread / 2),
            0.5: median,
            0.9: median * (1 + spread),
            0.99: median * (1 + 3 * spread),
        },
    )


def _is_empty(queue) -> bool:
    if isinstance(queue, InstanceQueueProbabilistic):
        return all(v == 0.0 for v in queue.values)
    return queue.expected_time_ms == 0.0 and queue.error_margin_ms == 0.0


class TestCharge:
    """A task's contribution on a quantile grid."""

    def test_interpolates_and_holds_ends(self):
        c = Charge.of("a", 50.0, quantiles={0.1: 10.0, 0.5: 50.0, 0.9: 90.0})
        assert c.at([0.05, 0.1, 0.25, 0.5, 0.75, 0.99]) == pytest.approx(
            [10.0, 10.0, 25.0, 50.0, 75.0, 90.0]
        )

    def test_without_quantiles_uses_the_time(self):
        assert Charge.of("a", 7.0).at([0.1, 0.9]) == [7.0, 7.0]

    def test_lower_half_is_not_clamped_to_the_median(self):
        assert DEFAULT_QUANTILES[0] < 0.5
        c = Charge.of("a", 50.0, quantiles={0.05: 5.0, 0.5: 50.0, 0.99: 99.0})
        low = c.at(DEFAULT_QUANTILES)[: DEFAULT_QUANTILES.index(0.5)]
        assert all(v < 50.0 for v in low)


@pytest.mark.parametrize("kind", ["probabilistic", "expect_error"])
class TestReturnsToEmpty:
    """Charges come off exactly, for both sketch kinds."""

    async def test_charge_then_release(self, kind):
        reg = await _registry(kind, "a")
        rng = random.Random(1)
        ids = [f"t{i}" for i in range(50)]
        for t in ids:
            await reg.charge(t, _random_charge(rng, "a"))
        assert not _is_empty(await reg.get_queue_info("a"))
        for t in reversed(ids):
            assert await reg.release_charge(t) is not None
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_long_random_run_matches_outstanding_work(self, kind):
        """No drift: at every step the sketch equals its open charges."""
        reg = await _registry(kind, "a", "b")
        rng = random.Random(7)
        open_: dict[str, Charge] = {}
        for step in range(5000):
            if open_ and rng.random() < 0.5:
                t = rng.choice(list(open_))
                del open_[t]
                await reg.release_charge(t)
            else:
                t = f"t{step}"
                open_[t] = _random_charge(rng, rng.choice("ab"))
                await reg.charge(t, open_[t])
            if step % 500 == 0:
                await _assert_matches(reg, open_, kind)
        await _assert_matches(reg, open_, kind)
        for t in list(open_):
            await reg.release_charge(t)
        for iid in "ab":
            assert _is_empty(await reg.get_queue_info(iid))

    async def test_recharge_replaces_the_old_charge(self, kind):
        reg = await _registry(kind, "a")
        await reg.charge("t", Charge.of("a", 100.0, 10.0))
        await reg.charge("t", Charge.of("a", 30.0, 3.0))
        q = await reg.get_queue_info("a")
        want = 30.0
        got = q.values[0] if kind == "probabilistic" else q.expected_time_ms
        assert got == pytest.approx(want)
        await reg.release_charge("t")
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_move_charge(self, kind):
        reg = await _registry(kind, "a", "b")
        await reg.charge("t", Charge.of("a", 100.0, 10.0))
        assert await reg.move_charge("t", "b")
        assert _is_empty(await reg.get_queue_info("a"))
        assert not _is_empty(await reg.get_queue_info("b"))
        await reg.release_charge("t")
        assert _is_empty(await reg.get_queue_info("b"))

    async def test_release_unknown_task_is_noop(self, kind):
        reg = await _registry(kind, "a")
        assert await reg.release_charge("nope") is None
        assert await reg.move_charge("nope", "a") is False
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_removed_instance_drops_its_charges(self, kind):
        reg = await _registry(kind, "a", "b")
        await reg.charge("t1", Charge.of("a", 5.0))
        await reg.charge("t2", Charge.of("b", 5.0))
        await reg.remove("a")
        assert reg.get_charge("t1") is None
        assert await reg.release_charge("t1") is None
        await reg.release_charge("t2")
        assert _is_empty(await reg.get_queue_info("b"))


async def _assert_matches(reg, open_, kind):
    for iid in "ab":
        mine = [c for c in open_.values() if c.instance_id == iid]
        q = await reg.get_queue_info(iid)
        if kind == "probabilistic":
            want = [0.0] * len(q.quantiles)
            for c in mine:
                want = [w + v for w, v in zip(want, c.at(q.quantiles))]
            assert q.values == pytest.approx(want, rel=1e-9, abs=1e-6)
        else:
            assert q.expected_time_ms == pytest.approx(
                sum(c.time_ms for c in mine), rel=1e-9, abs=1e-6
            )
            assert q.error_margin_ms == pytest.approx(
                sum(c.error_ms**2 for c in mine) ** 0.5, rel=1e-6, abs=1e-3
            )


class TestThroughStrategies:
    """Strategies charge through the registry."""

    async def test_strategy_charges_by_task_and_release_empties(self):
        reg = await _registry("expect_error", "a")
        strategy = MinimumExpectedTimeStrategy(None, reg)
        pred = Prediction(
            instance_id="a", predicted_time_ms=100.0, error_margin_ms=30.0
        )
        await strategy.update_queue("a", pred, task_id="t1")
        await strategy.update_queue("a", pred, task_id="t2")
        q = await reg.get_queue_info("a")
        assert q.expected_time_ms == 200.0
        assert q.error_margin_ms == pytest.approx(30.0 * 2**0.5)
        await reg.release_charge("t1")
        await reg.release_charge("t2")
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_probabilistic_update_is_deterministic(self):
        reg = await _registry("probabilistic", "a")
        strategy = ProbabilisticSchedulingStrategy(None, reg)
        pred = Prediction(
            instance_id="a",
            predicted_time_ms=50.0,
            quantiles={0.05: 5.0, 0.5: 50.0, 0.99: 99.0},
        )
        await strategy.update_queue("a", pred, task_id="t")
        q = await reg.get_queue_info("a")
        assert q.values == pytest.approx(
            Charge.of("a", 50.0, quantiles=pred.quantiles).at(q.quantiles)
        )


class TestStrategySwitch:
    """A switch keeps the work still in flight."""

    async def test_switch_rebuilds_sketches_from_open_charges(self):
        reg = await _registry("probabilistic", "a", "b")
        await reg.charge("t1", Charge.of("a", 100.0, 10.0))
        await reg.charge("t2", Charge.of("b", 40.0, 4.0))

        await reg.reset_queues("expect_error")
        a, b = await reg.get_queue_info("a"), await reg.get_queue_info("b")
        assert isinstance(a, InstanceQueueExpectError)
        assert (a.expected_time_ms, a.error_margin_ms) == (100.0, 10.0)
        assert b.expected_time_ms == 40.0

        await reg.release_charge("t1")
        await reg.release_charge("t2")
        assert _is_empty(await reg.get_queue_info("a"))
        assert _is_empty(await reg.get_queue_info("b"))

    async def test_new_instance_after_switch_has_new_kind(self):
        reg = await _registry("probabilistic", "a")
        await reg.reset_queues("expect_error")
        await reg.register(_instance("b"))
        assert isinstance(
            await reg.get_queue_info("b"), InstanceQueueExpectError
        )

    async def test_switch_with_custom_quantiles(self):
        reg = await _registry("expect_error", "a")
        await reg.charge("t", Charge.of("a", 10.0, quantiles={0.5: 10.0}))
        await reg.reset_queues("probabilistic", [0.9, 0.5, 0.5])
        q = await reg.get_queue_info("a")
        assert q.quantiles == [0.5, 0.9]
        assert q.values == [10.0, 10.0]


class TestCapacity:
    """Sketches count the backlog as served ``capacity`` at a time (2.7)."""

    async def test_expect_error_divides_by_capacity(self):
        reg = await _registry("expect_error", "a")
        await reg.set_capacity("a", 4)
        for t in range(4):
            await reg.charge(f"t{t}", Charge.of("a", 100.0, error_ms=40.0))
        q = await reg.get_queue_info("a")
        assert q.expected_time_ms == pytest.approx(100.0)
        assert q.error_margin_ms == pytest.approx(20.0)
        for t in range(4):
            await reg.release_charge(f"t{t}")
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_probabilistic_divides_by_capacity(self):
        reg = await _registry("probabilistic", "a")
        await reg.set_capacity("a", 2)
        charge = Charge.of("a", 50.0, quantiles={0.05: 10.0, 0.99: 90.0})
        await reg.charge("t", charge)
        q = await reg.get_queue_info("a")
        assert q.values == pytest.approx(
            [v / 2 for v in charge.at(q.quantiles)]
        )

    async def test_capacity_change_rebuilds_open_charges(self):
        reg = await _registry("expect_error", "a", "b")
        await reg.charge("t1", Charge.of("a", 120.0))
        await reg.charge("t2", Charge.of("b", 50.0))
        await reg.set_capacity("a", 3)
        assert (await reg.get_queue_info("a")).expected_time_ms == 40.0
        assert (await reg.get_queue_info("b")).expected_time_ms == 50.0
        await reg.release_charge("t1")
        assert _is_empty(await reg.get_queue_info("a"))

    async def test_default_and_floor_are_one(self):
        reg = await _registry("expect_error", "a")
        assert reg.get_capacity("a") == 1
        await reg.set_capacity("a", 0)
        assert reg.get_capacity("a") == 1

    async def test_concurrent_instance_wins_with_same_backlog(self):
        reg = await _registry("expect_error", "serial", "batched")
        await reg.set_capacity("batched", 8)
        for iid in ("serial", "batched"):
            for t in range(4):
                await reg.charge(f"{iid}{t}", Charge.of(iid, 100.0))
        strategy = MinimumExpectedTimeStrategy(None, reg)
        preds = [
            Prediction(instance_id=i, predicted_time_ms=100.0)
            for i in ("serial", "batched")
        ]
        queues = {i: await reg.get_queue_info(i) for i in ("serial", "batched")}
        assert strategy.select_instance(preds, queues) == "batched"
