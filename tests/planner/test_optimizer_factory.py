from typing import TypedDict

import numpy as np
import pytest

from swarmpilot.planner.core.optimizer_factory import (
    create_optimizer,
    run_optimization,
)
from swarmpilot.planner.core.swarm_optimizer import (
    PULP_AVAILABLE,
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
)


class _OptimizerInputs(TypedDict):
    M: int
    N: int
    B: np.ndarray
    initial: np.ndarray
    a: float
    target: np.ndarray


def _make_simple_inputs() -> _OptimizerInputs:
    return {
        "M": 2,
        "N": 2,
        "B": np.array([[10.0, 5.0], [5.0, 10.0]]),
        "initial": np.array([0, 1]),
        "a": 0.5,
        "target": np.array([100.0, 100.0]),
    }


def test_create_sa_optimizer() -> None:
    """Create simulated annealing optimizer."""
    opt = create_optimizer("simulated_annealing", **_make_simple_inputs())
    assert isinstance(opt, SimulatedAnnealingOptimizer)


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_create_ip_optimizer() -> None:
    """Create integer programming optimizer."""
    opt = create_optimizer("integer_programming", **_make_simple_inputs())
    assert isinstance(opt, IntegerProgrammingOptimizer)


def test_create_unknown_algorithm_raises() -> None:
    """Raise on unknown optimization algorithm."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_optimizer("genetic_algorithm", **_make_simple_inputs())


def test_run_optimization_sa() -> None:
    """Run optimization through SA helper path."""
    inputs = _make_simple_inputs()
    opt = create_optimizer("simulated_annealing", **inputs)
    deployment, score, stats = run_optimization(
        opt,
        objective_method="relative_error",
        max_iterations=10,
    )
    assert isinstance(deployment, np.ndarray)
    assert isinstance(score, float)
    assert isinstance(stats, dict)


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
def test_run_optimization_ip() -> None:
    """Run optimization through IP helper path."""
    inputs = _make_simple_inputs()
    opt = create_optimizer("integer_programming", **inputs)
    deployment, score, stats = run_optimization(
        opt,
        objective_method="relative_error",
    )
    assert isinstance(deployment, np.ndarray)
    assert isinstance(score, float)
    assert isinstance(stats, dict)
