"""SwarmPilot model service optimization algorithms.

This module implements two optimization algorithms for solving SwarmPilot model
deployment optimization problems:

1. Integer Programming (IP)
2. Simulated Annealing (SA)

Problem Description:
    Optimize model deployment configurations across multiple machines under
    limited change constraints, so that the overall service capacity
    distribution matches the actual request distribution as closely as
    possible.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from loguru import logger

random.seed(42)


class SwarmOptimizer(ABC):
    """Abstract base class for SwarmPilot optimizers.

    Defines common interfaces and shared functionality for all optimization
    algorithms.

    Attributes:
        M: Number of machines.
        N: Number of models.
        B: Batch processing capacity matrix [M x N].
        initial: Initial deployment state vector [M].
        a: Change factor controlling maximum allowed changes.
        target: Target request distribution [N].
        max_changes: Maximum allowed number of changes.
        valid_assignments: Precomputed valid machine assignments per model.
    """

    def __init__(
        self,
        M: int,
        N: int,
        B: np.ndarray,
        initial: np.ndarray,
        a: float,
        target: np.ndarray,
    ):
        """Initialize the optimizer.

        Args:
            M: Number of machines.
            N: Number of models.
            B: Batch processing capacity matrix [M x N]. B[i][j] represents
                machine i's processing capacity for model j.
            initial: Initial state vector [M]. initial[i] represents the model
                initially deployed on machine i. Use -1 for undeployed.
            a: Change factor controlling the upper limit of machines that can
                be changed (0 < a <= 1).
            target: Target request distribution [N]. Expected request ratio
                for each model.
        """
        self.M = M
        self.N = N
        self.B = B.copy()
        self.initial = initial.copy()
        self.a = a
        self.target = target.copy()
        self.max_changes = int(a * M)

        self._validate_inputs()
        self.valid_assignments = self._precompute_valid_assignments()

    def _validate_inputs(self) -> None:
        """Validate input parameters.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if self.B.shape != (self.M, self.N):
            raise ValueError(
                f"Batch capacity matrix dimension error: "
                f"{self.B.shape} != ({self.M}, {self.N})"
            )
        if len(self.initial) != self.M:
            raise ValueError(
                f"Initial state vector length error: "
                f"{len(self.initial)} != {self.M}"
            )
        if len(self.target) != self.N:
            raise ValueError(
                f"Target distribution vector length error: "
                f"{len(self.target)} != {self.N}"
            )
        if not (0 < self.a <= 1):
            raise ValueError(
                f"Change factor out of range: {self.a} not in (0, 1]"
            )

        # Support -1 as "no model deployed" initial state
        # -1 means the planner should compute the optimal initial deployment
        if not all(-1 <= x < self.N for x in self.initial):
            raise ValueError("Initial state contains invalid model ID")

        # Only validate capacity for VMs with deployed models (not -1)
        for i in range(self.M):
            if self.initial[i] != -1 and self.B[i, self.initial[i]] <= 0:
                raise ValueError(
                    f"Initial state contains invalid deployment: "
                    f"machine {i} deploys model {self.initial[i]} "
                    f"but capacity is 0"
                )

    def _precompute_valid_assignments(self) -> dict[int, list[int]]:
        """Precompute valid machine assignments for each model.

        Returns:
            Dictionary mapping model_id to list of machine_ids that can
            deploy that model.
        """
        valid_assignments = {}
        for j in range(self.N):
            valid_assignments[j] = [
                i for i in range(self.M) if self.B[i, j] > 0
            ]
        return valid_assignments

    def generate_initial_deployment(self) -> np.ndarray:
        """Generate a valid initial deployment for states containing -1.

        For VMs with current_model_id = -1, assigns the model with highest
        capacity according to the batch matrix. This ensures the optimization
        algorithm has a valid starting point.

        Returns:
            Valid initial deployment array [M].
        """
        deployment = self.initial.copy()

        # Assign optimal model for each -1 position
        for i in range(self.M):
            if deployment[i] == -1:
                # Find model with highest capacity on this VM
                best_model = -1
                best_capacity = 0
                for j in range(self.N):
                    if self.B[i, j] > best_capacity:
                        best_capacity = self.B[i, j]
                        best_model = j

                if best_model != -1:
                    deployment[i] = best_model
                else:
                    # If VM has 0 capacity for all models, use model 0
                    # This case should be caught during config validation
                    logger.warning(
                        f"VM {i} has 0 capacity for all models, "
                        f"using model 0 as default"
                    )
                    deployment[i] = 0

        logger.info(f"Generated initial deployment: {deployment}")
        return deployment

    def compute_service_capacity(self, deployment: np.ndarray) -> np.ndarray:
        """Compute service capacity distribution for a deployment.

        Args:
            deployment: Deployment array [M]. deployment[i] represents the
                model deployed on machine i.

        Returns:
            Total service capacity for each model [N].
        """
        capacity = np.zeros(self.N)
        for i in range(self.M):
            model = deployment[i]
            # Skip VMs with no model deployed (model_id == -1)
            if model != -1:
                capacity[model] += self.B[i, model]
        return capacity

    def compute_changes(self, deployment: np.ndarray) -> int:
        """Compute number of changes from initial state.

        Args:
            deployment: Deployment array [M].

        Returns:
            Number of machines that changed deployment.
        """
        return int(np.sum(deployment != self.initial))

    def is_valid_deployment(self, deployment: np.ndarray) -> bool:
        """Check if a deployment is valid.

        Args:
            deployment: Deployment array [M].

        Returns:
            True if the deployment is valid, False otherwise.
        """
        # Check change limit
        if self.compute_changes(deployment) > self.max_changes:
            return False

        # Check each machine's deployment feasibility
        for i in range(self.M):
            model = deployment[i]
            # -1 means no model deployed, which is invalid for final
            # deployment but valid during optimization process
            if model == -1:
                return False
            if self.B[i, model] == 0:
                return False

        return True

    def objective_function(
        self, deployment: np.ndarray, method: str = "relative_error"
    ) -> float:
        """Compute objective function value (lower is better).

        Args:
            deployment: Deployment array [M].
            method: Objective function type. Options:
                - 'relative_error': Minimize relative error.
                - 'ratio_difference': Minimize ratio difference.
                - 'weighted_squared': Weighted squared error.

        Returns:
            Objective function value.

        Raises:
            ValueError: If method is unknown.
        """
        if not self.is_valid_deployment(deployment):
            return float("inf")

        capacity = self.compute_service_capacity(deployment)
        target_sum = np.sum(self.target)
        capacity_sum = np.sum(capacity)

        if capacity_sum == 0:
            return float("inf")

        if method == "relative_error":
            # Minimize relative error
            capacity_ratio = capacity / capacity_sum
            target_ratio = self.target / target_sum
            return float(np.sum(np.abs(capacity_ratio - target_ratio)))

        elif method == "ratio_difference":
            # L-infinity norm of proportion differences
            capacity_ratio = capacity / capacity_sum
            target_ratio = self.target / target_sum
            return float(np.max(np.abs(capacity_ratio - target_ratio)))

        elif method == "weighted_squared":
            # Weighted squared error
            scale_factor = capacity_sum / target_sum
            scaled_target = scale_factor * self.target
            weights = 1.0 / (self.target + 1e-8)  # Weight inversely to target
            return float(np.sum(weights * (capacity - scaled_target) ** 2))

        else:
            raise ValueError(f"Unknown objective function type: {method}")

    @abstractmethod
    def optimize(self, **kwargs) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Execute the optimization algorithm.

        Returns:
            Tuple of (optimal_deployment, optimal_objective_value, statistics).
        """
        pass


class SimulatedAnnealingOptimizer(SwarmOptimizer):
    """Simulated annealing optimizer.

    Strategy: Allows accepting worse solutions to escape local optima,
    with temperature gradually decreasing. Has stronger global search
    capability.

    Attributes:
        Inherits all attributes from SwarmOptimizer.
    """

    def optimize(
        self,
        objective_method: str = "relative_error",
        initial_temp: float = 100.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        max_iterations: int = 5000,
        iterations_per_temp: int = 100,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Execute simulated annealing optimization.

        Args:
            objective_method: Objective function type.
            initial_temp: Initial temperature.
            final_temp: Final temperature (stopping criterion).
            cooling_rate: Cooling rate (temperature multiplied by this each
                cooling step).
            max_iterations: Maximum total iterations.
            iterations_per_temp: Iterations per temperature level.
            verbose: Whether to output detailed information.

        Returns:
            Tuple of (optimal_deployment, optimal_objective_value, statistics).
        """
        import time

        start_time = time.time()

        # Log optimization input
        logger.info(
            f"[OPTIMIZE_INPUT] algorithm=SimulatedAnnealing "
            f"M={self.M} N={self.N} max_changes={self.max_changes} "
            f"target={self.target.tolist()}"
        )

        # Generate valid initial deployment if initial state contains -1
        if -1 in self.initial:
            logger.info(
                "Detected initial state contains -1 (undeployed), "
                "auto-generating initial deployment"
            )
            current_deployment = self.generate_initial_deployment()
            self.initial = current_deployment.copy()
        else:
            current_deployment = self.initial.copy()

        current_score = self.objective_function(
            current_deployment, objective_method
        )
        best_deployment = current_deployment.copy()
        best_score = current_score

        temperature = initial_temp
        iterations = 0
        acceptances = 0
        rejections = 0
        temperature_changes = 0

        if verbose:
            logger.info(
                f"Simulated annealing started, initial temp: {initial_temp}, "
                f"initial score: {current_score:.6f}"
            )

        while temperature > final_temp and iterations < max_iterations:
            temp_iterations = 0
            temp_acceptances = 0

            # Iterate at current temperature
            while (
                temp_iterations < iterations_per_temp
                and iterations < max_iterations
            ):
                # Generate neighbor solution (random single machine change)
                neighbor = self._generate_random_neighbor(current_deployment)

                if neighbor is not None and self.is_valid_deployment(neighbor):
                    neighbor_score = self.objective_function(
                        neighbor, objective_method
                    )
                    delta = neighbor_score - current_score

                    # Acceptance criterion: always accept better solutions,
                    # accept worse solutions with probability
                    if delta < 0 or random.random() < math.exp(
                        -delta / temperature
                    ):
                        current_deployment = neighbor.copy()
                        current_score = neighbor_score
                        temp_acceptances += 1
                        acceptances += 1

                        # Update global best solution
                        if neighbor_score < best_score:
                            best_deployment = neighbor.copy()
                            best_score = neighbor_score
                    else:
                        rejections += 1

                temp_iterations += 1
                iterations += 1

            # Cool down
            temperature *= cooling_rate
            temperature_changes += 1

            if verbose and temperature_changes % 10 == 0:
                acceptance_rate = (
                    temp_acceptances / iterations_per_temp
                    if iterations_per_temp > 0
                    else 0
                )
                # Log optimization iteration
                logger.debug(
                    f"[OPTIMIZE_ITER] iteration={iterations} temp={temperature:.4f} "
                    f"current_score={current_score:.6f} best_score={best_score:.6f} "
                    f"acceptance_rate={acceptance_rate:.3f}"
                )

        stats = {
            "algorithm": "simulated_annealing",
            "iterations": iterations,
            "temperature_changes": temperature_changes,
            "acceptances": acceptances,
            "rejections": rejections,
            "acceptance_rate": (
                acceptances / (acceptances + rejections)
                if (acceptances + rejections) > 0
                else 0
            ),
            "final_temperature": temperature,
            "initial_score": self.objective_function(
                self.initial, objective_method
            ),
            "final_score": best_score,
        }

        elapsed_time = time.time() - start_time
        changes_count = self.compute_changes(best_deployment)

        # Log optimization result
        logger.info(
            f"[OPTIMIZE_RESULT] algorithm=SimulatedAnnealing "
            f"deployment={best_deployment.tolist()} score={best_score:.6f} "
            f"changes={changes_count} iterations={iterations} "
            f"elapsed_time={elapsed_time:.2f}s"
        )

        return best_deployment, best_score, stats

    def _generate_random_neighbor(
        self, deployment: np.ndarray
    ) -> np.ndarray | None:
        """Generate a random neighbor solution.

        Args:
            deployment: Current deployment array.

        Returns:
            Random neighbor solution, or None if cannot generate.
        """
        # Randomly select a machine
        machine = random.randint(0, self.M - 1)
        current_model = deployment[machine]

        # Get other deployable models for this machine
        valid_models = [
            m
            for m in range(self.N)
            if m != current_model and self.B[machine, m] > 0
        ]

        if not valid_models:
            return None

        # Randomly select new model
        new_model = random.choice(valid_models)

        neighbor = deployment.copy()
        neighbor[machine] = new_model

        return neighbor


try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None


class IntegerProgrammingOptimizer(SwarmOptimizer):
    """Integer programming optimizer.

    Uses linear programming solver to solve mixed integer programming problems.
    Requires pulp library: pip install pulp

    Attributes:
        Inherits all attributes from SwarmOptimizer.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the integer programming optimizer.

        Raises:
            ImportError: If pulp library is not installed.
        """
        super().__init__(*args, **kwargs)
        if not PULP_AVAILABLE:
            raise ImportError(
                "pulp library required for integer programming optimizer: "
                "pip install pulp"
            )

    def optimize(
        self,
        objective_method: str = "relative_error",
        solver_name: str = "PULP_CBC_CMD",
        time_limit: int = 300,
        verbose: bool = True,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Execute integer programming optimization.

        Args:
            objective_method: Objective function type. Note: integer
                programming only supports linear objective functions.
            solver_name: Solver name to use.
            time_limit: Time limit in seconds.
            verbose: Whether to output detailed information.

        Returns:
            Tuple of (optimal_deployment, optimal_objective_value, statistics).
        """
        import time

        start_time = time.time()

        # Log optimization input
        logger.info(
            f"[OPTIMIZE_INPUT] algorithm=IntegerProgramming "
            f"M={self.M} N={self.N} max_changes={self.max_changes} "
            f"target={self.target.tolist()} solver={solver_name}"
        )

        # Create problem instance
        prob = pulp.LpProblem("SwarmPilot_Optimization", pulp.LpMinimize)

        # Decision variables: x[i][j] indicates if machine i deploys model j
        x = {}
        for i in range(self.M):
            for j in range(self.N):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

        # Change indicator variables: y[i] indicates if machine i changed
        y = {}
        for i in range(self.M):
            y[i] = pulp.LpVariable(f"y_{i}", cat="Binary")

        # Constraint 1: Each machine can only deploy one model
        for i in range(self.M):
            prob += pulp.lpSum([x[i, j] for j in range(self.N)]) == 1

        # Constraint 2: Machine capacity constraints
        for i in range(self.M):
            for j in range(self.N):
                if self.B[i, j] == 0:
                    prob += x[i, j] == 0

        # Constraint 3: Change detection
        # If initial_model is -1 (no model deployed), treat it as a change
        for i in range(self.M):
            initial_model = self.initial[i]
            if initial_model == -1:
                # No initial model, any deployment counts as a change
                # y[i] = 1 means changed (must deploy a model)
                prob += y[i] == 1
            else:
                # y[i] >= 1 - x[i, initial_model] means:
                # if x[i, initial_model] = 0 (changed), then y[i] >= 1
                prob += y[i] >= 1 - x[i, initial_model]

        # Constraint 4: Change count limit
        prob += pulp.lpSum([y[i] for i in range(self.M)]) <= self.max_changes

        # Objective function: using simplified linear objective
        # Integer programming struggles with complex nonlinear objectives,
        # so we use linear approximation
        if objective_method == "relative_error":
            # Minimize weighted deviation of service capacity from target
            target_sum = np.sum(self.target)
            target_ratio = self.target / target_sum

            # Compute service capacity for each model
            capacity = {}
            for j in range(self.N):
                capacity[j] = pulp.lpSum(
                    [self.B[i, j] * x[i, j] for i in range(self.M)]
                )

            total_capacity = pulp.lpSum([capacity[j] for j in range(self.N)])

            # Using linear approximation for objective
            # Minimize |capacity[j]/total_capacity - target_ratio[j]|
            # Since absolute value is not linear, we use auxiliary variables
            deviation_vars = {}
            for j in range(self.N):
                deviation_vars[j] = pulp.LpVariable(f"dev_{j}", lowBound=0)
                # Linearization of absolute value
                prob += (
                    deviation_vars[j]
                    >= capacity[j] - target_ratio[j] * total_capacity
                )
                prob += (
                    deviation_vars[j]
                    >= target_ratio[j] * total_capacity - capacity[j]
                )

            prob += pulp.lpSum([deviation_vars[j] for j in range(self.N)])

        else:
            # For other objectives, use basic load balancing objective
            logger.warning(
                f"Integer programming does not support objective "
                f"{objective_method}, using default linear objective"
            )
            capacity = {}
            for j in range(self.N):
                capacity[j] = pulp.lpSum(
                    [self.B[i, j] * x[i, j] for i in range(self.M)]
                )

            # Minimize number of changes (as simple objective)
            prob += pulp.lpSum([y[i] for i in range(self.M)])

        # Solve
        try:
            if solver_name == "PULP_CBC_CMD":
                solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
            else:
                solver = pulp.getSolver(
                    solver_name, timeLimit=time_limit, msg=verbose
                )

            prob.solve(solver)

            # Check solution status
            status = pulp.LpStatus[prob.status]

            if status == "Optimal":
                # Extract solution
                deployment = np.zeros(self.M, dtype=int)
                for i in range(self.M):
                    for j in range(self.N):
                        if (
                            x[i, j].varValue is not None
                            and x[i, j].varValue > 0.5
                        ):
                            deployment[i] = j
                            break

                final_score = self.objective_function(
                    deployment, objective_method
                )

                stats = {
                    "algorithm": "integer_programming",
                    "solver": solver_name,
                    "status": status,
                    "objective_value": pulp.value(prob.objective),
                    "solve_time": (
                        prob.solutionTime
                        if hasattr(prob, "solutionTime")
                        else None
                    ),
                    "initial_score": self.objective_function(
                        self.initial, objective_method
                    ),
                    "final_score": final_score,
                }

                elapsed_time = time.time() - start_time
                changes_count = self.compute_changes(deployment)

                # Log optimization result
                logger.info(
                    f"[OPTIMIZE_RESULT] algorithm=IntegerProgramming "
                    f"deployment={deployment.tolist()} score={final_score:.6f} "
                    f"changes={changes_count} status={status} "
                    f"elapsed_time={elapsed_time:.2f}s"
                )

                return deployment, final_score, stats

            else:
                elapsed_time = time.time() - start_time

                # Log optimization result (failed)
                logger.warning(
                    f"[OPTIMIZE_RESULT] algorithm=IntegerProgramming "
                    f"status={status} failed=true elapsed_time={elapsed_time:.2f}s"
                )

                # Return initial solution
                initial_score = self.objective_function(
                    self.initial, objective_method
                )
                stats = {
                    "algorithm": "integer_programming",
                    "solver": solver_name,
                    "status": status,
                    "objective_value": None,
                    "solve_time": None,
                    "initial_score": initial_score,
                    "final_score": initial_score,
                }

                return self.initial.copy(), initial_score, stats

        except Exception as e:
            logger.error(f"Error during integer programming solve: {e!s}")
            initial_score = self.objective_function(
                self.initial, objective_method
            )
            stats = {
                "algorithm": "integer_programming",
                "solver": solver_name,
                "status": "Error",
                "error": str(e),
                "initial_score": initial_score,
                "final_score": initial_score,
            }

            return self.initial.copy(), initial_score, stats
