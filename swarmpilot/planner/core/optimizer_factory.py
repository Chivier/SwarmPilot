"""Factory for creating and running swarm optimizers."""

from __future__ import annotations

from typing import Any

import numpy as np

from swarmpilot.planner.core.swarm_optimizer import (
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
    SwarmOptimizer,
)


def create_optimizer(
    algorithm: str,
    *,
    M: int,
    N: int,
    B: np.ndarray,
    initial: np.ndarray,
    a: float,
    target: np.ndarray,
) -> SwarmOptimizer:
    """Create a swarm optimizer instance from an algorithm name."""
    if algorithm == "simulated_annealing":
        return SimulatedAnnealingOptimizer(
            M=M,
            N=N,
            B=B,
            initial=initial,
            a=a,
            target=target,
        )
    elif algorithm == "integer_programming":
        return IntegerProgrammingOptimizer(
            M=M,
            N=N,
            B=B,
            initial=initial,
            a=a,
            target=target,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_optimization(
    optimizer: SwarmOptimizer,
    *,
    objective_method: str,
    initial_temp: float | None = None,
    final_temp: float | None = None,
    cooling_rate: float | None = None,
    max_iterations: int | None = None,
    iterations_per_temp: int | None = None,
    solver_name: str | None = None,
    time_limit: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run optimization with algorithm-appropriate keyword arguments."""
    is_sa_optimizer = isinstance(
        SimulatedAnnealingOptimizer, type
    ) and isinstance(optimizer, SimulatedAnnealingOptimizer)
    is_ip_optimizer = isinstance(
        IntegerProgrammingOptimizer, type
    ) and isinstance(optimizer, IntegerProgrammingOptimizer)

    if is_sa_optimizer:
        kwargs: dict[str, Any] = {
            "objective_method": objective_method,
            "verbose": verbose,
        }
        if initial_temp is not None:
            kwargs["initial_temp"] = initial_temp
        if final_temp is not None:
            kwargs["final_temp"] = final_temp
        if cooling_rate is not None:
            kwargs["cooling_rate"] = cooling_rate
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        if iterations_per_temp is not None:
            kwargs["iterations_per_temp"] = iterations_per_temp
        return optimizer.optimize(**kwargs)
    elif is_ip_optimizer:
        kwargs = {"objective_method": objective_method, "verbose": verbose}
        if solver_name is not None:
            kwargs["solver_name"] = solver_name
        if time_limit is not None:
            kwargs["time_limit"] = time_limit
        return optimizer.optimize(**kwargs)
    elif hasattr(optimizer, "optimize"):
        has_sa_kwargs = any(
            value is not None
            for value in (
                initial_temp,
                final_temp,
                cooling_rate,
                max_iterations,
                iterations_per_temp,
            )
        )

        kwargs = {"objective_method": objective_method, "verbose": verbose}
        if has_sa_kwargs:
            if initial_temp is not None:
                kwargs["initial_temp"] = initial_temp
            if final_temp is not None:
                kwargs["final_temp"] = final_temp
            if cooling_rate is not None:
                kwargs["cooling_rate"] = cooling_rate
            if max_iterations is not None:
                kwargs["max_iterations"] = max_iterations
            if iterations_per_temp is not None:
                kwargs["iterations_per_temp"] = iterations_per_temp
        else:
            if solver_name is not None:
                kwargs["solver_name"] = solver_name
            if time_limit is not None:
                kwargs["time_limit"] = time_limit
        return optimizer.optimize(**kwargs)
    else:
        raise TypeError(
            f"Unsupported optimizer type: {type(optimizer).__name__}"
        )
