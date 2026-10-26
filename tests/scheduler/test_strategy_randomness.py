"""Strategies own their randomness (plan item 2.3)."""

import random

import numpy as np

from swarmpilot.scheduler.algorithms import get_strategy
from swarmpilot.scheduler.algorithms.power_of_two import PowerOfTwoStrategy
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry


def _preds(n: int) -> list[Prediction]:
    return [
        Prediction(instance_id=f"i{k}", predicted_time_ms=10.0)
        for k in range(n)
    ]


def _picks(strategy, n: int, rounds: int) -> list[str]:
    return [strategy.select_instance(_preds(n), {}) for _ in range(rounds)]


def test_same_seed_repeats_and_no_seed_differs():
    reg = InstanceRegistry()
    a = get_strategy("random", None, reg, seed=3)
    b = get_strategy("random", None, reg, seed=3)
    assert _picks(a, 8, 50) == _picks(b, 8, 50)
    c = get_strategy("random", None, reg)
    d = get_strategy("random", None, reg)
    assert _picks(c, 8, 50) != _picks(d, 8, 50)


def test_strategies_leave_global_random_state_alone():
    random.seed(123)
    np.random.seed(123)
    want = (random.random(), np.random.random())
    random.seed(123)
    np.random.seed(123)
    reg = InstanceRegistry()
    for name in ("random", "po2", "probabilistic"):
        _picks(get_strategy(name, None, reg), 4, 10)
    assert (random.random(), np.random.random()) == want


def test_po2_compares_two_distinct_replicas():
    strategy = PowerOfTwoStrategy(None, InstanceRegistry(), seed=0)
    queues = {
        "i0": InstanceQueueExpectError(
            instance_id="i0", expected_time_ms=0.0, error_margin_ms=0.0
        ),
        "i1": InstanceQueueExpectError(
            instance_id="i1", expected_time_ms=50.0, error_margin_ms=0.0
        ),
    }
    # With two replicas both are always compared, so the shorter queue wins.
    assert {strategy.select_instance(_preds(2), queues) for _ in range(50)} == {
        "i0"
    }


def test_po2_uses_second_queue_whatever_the_first_is():
    """The second replica's queue counts even if the first has none."""
    strategy = PowerOfTwoStrategy(None, InstanceRegistry(), seed=0)
    queues = {
        "i0": InstanceQueueProbabilistic(
            instance_id="i0", quantiles=[0.5], values=[0.0]
        ),
        "i1": InstanceQueueExpectError(
            instance_id="i1", expected_time_ms=50.0, error_margin_ms=0.0
        ),
    }
    picks = {strategy.select_instance(_preds(2), queues) for _ in range(50)}
    assert picks == {"i0"}


def test_po2_single_replica():
    strategy = PowerOfTwoStrategy(None, InstanceRegistry())
    assert strategy.select_instance(_preds(1), {}) == "i0"
