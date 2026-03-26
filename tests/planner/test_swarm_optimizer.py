"""Tests for swarm optimizer algorithms."""

from unittest.mock import patch

import numpy as np
import pytest

from swarmpilot.planner.core.swarm_optimizer import (
    PULP_AVAILABLE,
    IntegerProgrammingOptimizer,
    SimulatedAnnealingOptimizer,
)


class TestSwarmOptimizerBase:
    """Tests for SwarmOptimizer base class functionality."""

    def test_generate_initial_deployment_with_minus_one(self):
        """Test that VMs with -1 get assigned the highest capacity model."""
        B = np.array(
            [
                [10.0, 5.0, 2.0],  # VM 0: model 0 has highest capacity
                [3.0, 15.0, 1.0],  # VM 1: model 1 has highest capacity
                [1.0, 2.0, 20.0],  # VM 2: model 2 has highest capacity
            ]
        )
        initial = np.array([-1, -1, -1])
        target = np.array([100.0, 100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=3, N=3, B=B, initial=initial, a=1.0, target=target
        )

        deployment = optimizer.generate_initial_deployment()

        assert deployment[0] == 0  # VM 0 should get model 0 (capacity 10)
        assert deployment[1] == 1  # VM 1 should get model 1 (capacity 15)
        assert deployment[2] == 2  # VM 2 should get model 2 (capacity 20)

    def test_generate_initial_deployment_all_zero_capacity(self):
        """Test fallback when VM has zero capacity for all models."""
        B = np.array(
            [
                [10.0, 5.0, 2.0],  # VM 0: normal
                [0.0, 0.0, 0.0],  # VM 1: all zero capacity (edge case)
                [1.0, 2.0, 3.0],  # VM 2: normal
            ]
        )
        initial = np.array([0, -1, 2])
        target = np.array([100.0, 100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=3, N=3, B=B, initial=initial, a=1.0, target=target
        )

        deployment = optimizer.generate_initial_deployment()

        # VM with all zero capacity should fall back to model 0
        assert deployment[1] == 0

    def test_generate_initial_deployment_partial_minus_one(self):
        """Test that only VMs with -1 are modified."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
                [8.0, 8.0],
            ]
        )
        initial = np.array([0, -1, 1])  # Only VM 1 needs assignment
        target = np.array([100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=3, N=2, B=B, initial=initial, a=1.0, target=target
        )

        deployment = optimizer.generate_initial_deployment()

        assert deployment[0] == 0  # Unchanged
        assert deployment[1] == 1  # Assigned model with highest capacity (10)
        assert deployment[2] == 1  # Unchanged


class TestObjectiveFunction:
    """Tests for different objective function methods."""

    @pytest.fixture
    def optimizer(self):
        """Create a basic optimizer for testing."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
                [8.0, 8.0],
            ]
        )
        initial = np.array([0, 1, 0])
        target = np.array([100.0, 100.0])

        return SimulatedAnnealingOptimizer(
            M=3, N=2, B=B, initial=initial, a=0.5, target=target
        )

    def test_relative_error_method(self, optimizer):
        """Test relative_error objective function."""
        deployment = np.array([0, 1, 0])
        score = optimizer.objective_function(
            deployment, method="relative_error"
        )

        # Should return a finite value
        assert np.isfinite(score)
        assert score >= 0

    def test_ratio_difference_method(self, optimizer):
        """Test ratio_difference objective function."""
        deployment = np.array([0, 1, 0])
        score = optimizer.objective_function(
            deployment, method="ratio_difference"
        )

        # Should return a finite value
        assert np.isfinite(score)
        assert score >= 0

    def test_weighted_squared_method(self, optimizer):
        """Test weighted_squared objective function."""
        deployment = np.array([0, 1, 0])
        score = optimizer.objective_function(
            deployment, method="weighted_squared"
        )

        # Should return a finite value
        assert np.isfinite(score)
        assert score >= 0

    def test_unknown_method_raises_error(self, optimizer):
        """Test that unknown method raises ValueError."""
        deployment = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="Unknown objective function type"):
            optimizer.objective_function(deployment, method="unknown_method")

    def test_invalid_deployment_returns_inf(self, optimizer):
        """Test that invalid deployment returns infinity."""
        # Deployment with -1 is invalid
        invalid_deployment = np.array([0, -1, 0])
        score = optimizer.objective_function(
            invalid_deployment, method="relative_error"
        )
        assert score == float("inf")

    def test_ratio_difference_no_degenerate_zero_allocation(self):
        """Regression: ratio_difference must not favor zero-allocation.

        Reproduces the mock_llm_cluster bug: 12 identical instances,
        2 models with B=[5.0, 1.0], target=[16.67, 83.33].
        Allocating 1 instance to model-0 and 11 to model-1 produces
        proportions [5/16, 11/16] ≈ [0.3125, 0.6875] which is closer
        to [0.1667, 0.8333] than putting all on model-1 ([0, 1]).
        """
        M, N = 12, 2
        B = np.tile(np.array([[5.0, 1.0]]), (M, 1))
        target = np.array([16.67, 83.33])
        initial = np.array([0] * 6 + [1] * 6)

        opt = SimulatedAnnealingOptimizer(
            M=M, N=N, B=B, initial=initial, a=1.0, target=target
        )

        balanced = np.array([0] + [1] * 11)  # 1 on model-0, 11 on model-1
        degenerate = np.array([1] * 12)  # all on model-1, model-0 starved

        score_balanced = opt.objective_function(
            balanced, method="ratio_difference"
        )
        score_degenerate = opt.objective_function(
            degenerate, method="ratio_difference"
        )

        assert score_balanced < score_degenerate, (
            f"Balanced ({score_balanced}) should beat degenerate ({score_degenerate})"
        )

    def test_ratio_difference_is_l_infinity_of_proportions(self):
        """Verify ratio_difference computes L-inf of proportion diffs."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
                [8.0, 8.0],
            ]
        )
        target = np.array([100.0, 100.0])
        initial = np.array([0, 1, 0])

        opt = SimulatedAnnealingOptimizer(
            M=3, N=2, B=B, initial=initial, a=0.5, target=target
        )

        deployment = np.array([0, 1, 0])
        # capacity: model 0 = B[0,0]+B[2,0] = 18, model 1 = B[1,1] = 10
        # capacity_ratio = [18/28, 10/28]
        # target_ratio = [0.5, 0.5]
        expected = max(abs(18 / 28 - 0.5), abs(10 / 28 - 0.5))

        score = opt.objective_function(deployment, method="ratio_difference")
        assert score == pytest.approx(expected, abs=1e-9)

    def test_zero_capacity_returns_inf(self):
        """Test that zero total capacity returns infinity."""
        # Create a scenario where capacity would be zero
        B = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        initial = np.array([0, 1])  # Invalid but for testing
        target = np.array([100.0, 100.0])

        # Need to bypass validation
        optimizer = SimulatedAnnealingOptimizer.__new__(
            SimulatedAnnealingOptimizer
        )
        optimizer.M = 2
        optimizer.N = 2
        optimizer.B = B
        optimizer.initial = initial
        optimizer.target = target
        optimizer.max_changes = 2
        optimizer.a = 1.0
        optimizer.valid_assignments = {0: [], 1: []}

        deployment = np.array([0, 1])
        # The is_valid_deployment check will fail, returning inf
        score = optimizer.objective_function(
            deployment, method="relative_error"
        )
        assert score == float("inf")


class TestSimulatedAnnealing:
    """Tests for SimulatedAnnealingOptimizer."""

    def test_sa_with_minus_one_initial(self):
        """Test SA optimizer with -1 initial states auto-generates deployment."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([-1, -1])
        target = np.array([100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=2, N=2, B=B, initial=initial, a=1.0, target=target
        )

        deployment, score, _stats = optimizer.optimize(
            max_iterations=10, verbose=False
        )

        # Should produce a valid deployment (no -1s)
        assert -1 not in deployment
        assert np.isfinite(score)

    def test_sa_verbose_output(self):
        """Test SA optimizer with verbose output."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=2, N=2, B=B, initial=initial, a=0.5, target=target
        )

        # Run with verbose=True, should not raise
        deployment, _score, stats = optimizer.optimize(
            max_iterations=100, iterations_per_temp=10, verbose=True
        )

        assert deployment is not None
        assert "algorithm" in stats
        assert stats["algorithm"] == "simulated_annealing"

    def test_sa_neighbor_generation_returns_none(self):
        """Test SA handles case when no valid neighbor exists."""
        # Create scenario where VM 0 can only deploy model 0
        B = np.array(
            [
                [10.0, 0.0],  # VM 0 can only deploy model 0
                [5.0, 10.0],  # VM 1 can deploy both
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=2, N=2, B=B, initial=initial, a=1.0, target=target
        )

        # Manually test neighbor generation for restricted VM
        deployment = np.array([0, 1])

        # Patch random.randint to always select VM 0
        with patch("random.randint", return_value=0):
            neighbor = optimizer._generate_random_neighbor(deployment)
            # Should return None because VM 0 has no other valid models
            assert neighbor is None

    def test_sa_stats_output(self):
        """Test SA returns proper statistics."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = SimulatedAnnealingOptimizer(
            M=2, N=2, B=B, initial=initial, a=0.5, target=target
        )

        _deployment, _score, stats = optimizer.optimize(
            max_iterations=50, verbose=False
        )

        assert "iterations" in stats
        assert "temperature_changes" in stats
        assert "acceptances" in stats
        assert "rejections" in stats
        assert "acceptance_rate" in stats
        assert "final_temperature" in stats
        assert "initial_score" in stats
        assert "final_score" in stats


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
class TestIntegerProgramming:
    """Tests for IntegerProgrammingOptimizer."""

    def test_ip_basic_optimization(self):
        """Test basic IP optimization."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = IntegerProgrammingOptimizer(
            M=2, N=2, B=B, initial=initial, a=0.5, target=target
        )

        deployment, score, stats = optimizer.optimize(
            verbose=False, time_limit=10
        )

        assert deployment is not None
        assert np.isfinite(score)
        assert "algorithm" in stats

    def test_ip_with_minus_one_initial(self):
        """Test IP optimizer with -1 initial states."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([-1, 1])  # VM 0 has no initial model
        target = np.array([100.0, 100.0])

        optimizer = IntegerProgrammingOptimizer(
            M=2, N=2, B=B, initial=initial, a=1.0, target=target
        )

        deployment, _score, _stats = optimizer.optimize(
            verbose=False, time_limit=10
        )

        # Should produce a valid deployment
        assert -1 not in deployment


class TestIPImportError:
    """Tests for IP optimizer import error handling."""

    def test_ip_import_error_when_pulp_not_available(self):
        """Test that ImportError is raised when pulp is not available."""
        B = np.array([[10.0, 5.0]])
        initial = np.array([0])
        target = np.array([100.0, 100.0])

        # Mock PULP_AVAILABLE to be False
        with (
            patch(
                "swarmpilot.planner.core.swarm_optimizer.PULP_AVAILABLE", False
            ),
            pytest.raises(ImportError, match="pulp library required"),
        ):
            IntegerProgrammingOptimizer(
                M=1, N=2, B=B, initial=initial, a=1.0, target=target
            )


class TestValidation:
    """Tests for input validation in SwarmOptimizer."""

    def test_invalid_B_shape(self):  # noqa: N802
        """Test validation fails for incorrect B matrix shape."""
        B = np.array([[10.0, 5.0]])  # Wrong shape
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        with pytest.raises(
            AssertionError, match="Batch capacity matrix dimension error"
        ):
            SimulatedAnnealingOptimizer(
                M=2, N=2, B=B, initial=initial, a=0.5, target=target
            )

    def test_invalid_initial_length(self):
        """Test validation fails for incorrect initial length."""
        B = np.array([[10.0, 5.0], [5.0, 10.0]])
        initial = np.array([0])  # Wrong length
        target = np.array([100.0, 100.0])

        with pytest.raises(
            AssertionError, match="Initial state vector length error"
        ):
            SimulatedAnnealingOptimizer(
                M=2, N=2, B=B, initial=initial, a=0.5, target=target
            )

    def test_invalid_target_length(self):
        """Test validation fails for incorrect target length."""
        B = np.array([[10.0, 5.0], [5.0, 10.0]])
        initial = np.array([0, 1])
        target = np.array([100.0])  # Wrong length

        with pytest.raises(
            AssertionError, match="Target distribution vector length error"
        ):
            SimulatedAnnealingOptimizer(
                M=2, N=2, B=B, initial=initial, a=0.5, target=target
            )

    def test_invalid_a_value(self):
        """Test validation fails for a value out of range."""
        B = np.array([[10.0, 5.0], [5.0, 10.0]])
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        with pytest.raises(AssertionError, match="Change factor out of range"):
            SimulatedAnnealingOptimizer(
                M=2,
                N=2,
                B=B,
                initial=initial,
                a=1.5,
                target=target,  # Invalid
            )

    def test_invalid_initial_model_id(self):
        """Test validation fails for invalid model ID in initial."""
        B = np.array([[10.0, 5.0], [5.0, 10.0]])
        initial = np.array([0, 5])  # 5 is out of range
        target = np.array([100.0, 100.0])

        with pytest.raises(
            AssertionError, match="Initial state contains invalid model ID"
        ):
            SimulatedAnnealingOptimizer(
                M=2, N=2, B=B, initial=initial, a=0.5, target=target
            )

    def test_zero_capacity_deployment(self):
        """Test validation fails when initial deploys model with 0 capacity."""
        B = np.array(
            [
                [10.0, 0.0],  # VM 0 can't run model 1
                [5.0, 10.0],
            ]
        )
        initial = np.array([1, 1])  # VM 0 tries to deploy model 1 (capacity 0)
        target = np.array([100.0, 100.0])

        with pytest.raises(
            AssertionError, match="Initial state contains invalid deployment"
        ):
            SimulatedAnnealingOptimizer(
                M=2, N=2, B=B, initial=initial, a=0.5, target=target
            )


@pytest.mark.skipif(not PULP_AVAILABLE, reason="pulp not installed")
class TestIPSolverFailurePaths:
    """Tests for IP solver failure and error handling paths."""

    def test_ip_solver_exception_handling(self):
        """Test IP solver handles exceptions gracefully."""
        import pulp

        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = IntegerProgrammingOptimizer(
            M=2, N=2, B=B, initial=initial, a=0.5, target=target
        )

        def mock_solve(self, *args, **kwargs):
            raise Exception("Solver crashed")

        with patch.object(pulp.LpProblem, "solve", mock_solve):
            deployment, _score, stats = optimizer.optimize(verbose=False)

            # Should return initial deployment on error
            assert np.array_equal(deployment, initial)
            assert stats["status"] == "Error"
            assert "error" in stats

    def test_ip_solver_verbose_output(self):
        """Test IP solver with verbose output."""
        B = np.array(
            [
                [10.0, 5.0],
                [5.0, 10.0],
            ]
        )
        initial = np.array([0, 1])
        target = np.array([100.0, 100.0])

        optimizer = IntegerProgrammingOptimizer(
            M=2, N=2, B=B, initial=initial, a=0.5, target=target
        )

        # Run with verbose=True
        deployment, _score, stats = optimizer.optimize(
            verbose=True, time_limit=10
        )

        # Should complete successfully
        assert deployment is not None
        assert "algorithm" in stats
