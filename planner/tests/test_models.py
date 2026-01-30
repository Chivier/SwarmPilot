"""Tests for data models and validation."""

import pytest
from pydantic import ValidationError

from src.models import (
    PlannerInput,
    PlannerOutput,
)


class TestPlannerInput:
    """Tests for PlannerInput model."""

    def test_valid_input(self, sample_planner_input):
        """Test creation with valid data."""
        planner_input = PlannerInput(**sample_planner_input)
        assert planner_input.M == 4
        assert planner_input.N == 3
        assert planner_input.a == 0.5
        assert planner_input.algorithm == "simulated_annealing"

    def test_invalid_M(self, sample_planner_input):
        """Test validation fails for M <= 0."""
        sample_planner_input["M"] = 0
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "M" in str(exc_info.value)

    def test_invalid_N(self, sample_planner_input):
        """Test validation fails for N <= 0."""
        sample_planner_input["N"] = -1
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "N" in str(exc_info.value)

    def test_invalid_a_too_small(self, sample_planner_input):
        """Test validation fails for a <= 0."""
        sample_planner_input["a"] = 0.0
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "a" in str(exc_info.value)

    def test_invalid_a_too_large(self, sample_planner_input):
        """Test validation fails for a > 1."""
        sample_planner_input["a"] = 1.5
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "a" in str(exc_info.value)

    def test_batch_capacity_wrong_rows(self, sample_planner_input):
        """Test validation fails when B has wrong number of rows."""
        sample_planner_input["B"] = [[10, 5, 0], [8, 6, 4]]  # Only 2 rows, need 4
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "must have 4 rows" in str(exc_info.value)

    def test_batch_capacity_wrong_columns(self, sample_planner_input):
        """Test validation fails when B has wrong number of columns."""
        sample_planner_input["B"] = [
            [10, 5],  # Only 2 columns, need 3
            [8, 6],
            [0, 10],
            [6, 0]
        ]
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "must have 3 columns" in str(exc_info.value)

    def test_batch_capacity_negative_values(self, sample_planner_input):
        """Test validation fails for negative capacities."""
        sample_planner_input["B"][0][0] = -5.0
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "negative values" in str(exc_info.value)

    def test_initial_wrong_length(self, sample_planner_input):
        """Test validation fails when initial has wrong length."""
        sample_planner_input["initial"] = [0, 1]  # Only 2, need 4
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "initial must have length 4" in str(exc_info.value)

    def test_initial_invalid_model_id(self, sample_planner_input):
        """Test validation fails for invalid model IDs in initial."""
        sample_planner_input["initial"] = [0, 1, 2, 5]  # 5 is out of range [0, 2]
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "invalid" in str(exc_info.value)

    def test_initial_with_negative_one(self, sample_planner_input):
        """Test -1 is allowed in initial (no model deployed)."""
        sample_planner_input["initial"] = [-1, 1, 2, 2]
        planner_input = PlannerInput(**sample_planner_input)
        assert planner_input.initial[0] == -1

    def test_initial_none_allowed(self, sample_planner_input):
        """Test None is allowed for initial (validator returns None)."""
        sample_planner_input["initial"] = None
        planner_input = PlannerInput(**sample_planner_input)
        assert planner_input.initial is None

    def test_target_wrong_length(self, sample_planner_input):
        """Test validation fails when target has wrong length."""
        sample_planner_input["target"] = [20.0, 30.0]  # Only 2, need 3
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "target must have length 3" in str(exc_info.value)

    def test_target_negative_values(self, sample_planner_input):
        """Test validation fails for negative target values."""
        sample_planner_input["target"] = [20.0, -10.0, 25.0]
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        assert "negative values" in str(exc_info.value)

    def test_temperature_validation(self, sample_planner_input):
        """Test validation fails when final_temp >= initial_temp."""
        sample_planner_input["initial_temp"] = 10.0
        sample_planner_input["final_temp"] = 20.0
        with pytest.raises(ValidationError) as exc_info:
            PlannerInput(**sample_planner_input)
        # Error message now includes specific values for better debugging
        assert "must be less than initial_temp" in str(exc_info.value)

    def test_default_values(self):
        """Test default values are set correctly."""
        minimal_input = {
            "M": 2,
            "N": 2,
            "B": [[1, 0], [0, 1]],
            "initial": [0, 1],
            "a": 0.5,
            "target": [10, 10]
        }
        planner_input = PlannerInput(**minimal_input)
        assert planner_input.algorithm == "simulated_annealing"
        assert planner_input.objective_method == "relative_error"
        assert planner_input.verbose is True
        assert planner_input.initial_temp == 100.0
        assert planner_input.max_iterations == 5000


class TestPlannerOutput:
    """Tests for PlannerOutput model."""

    def test_valid_output(self):
        """Test creation with valid data."""
        output = PlannerOutput(
            deployment=[0, 1, 1, 2],
            score=0.0667,
            stats={"algorithm": "simulated_annealing", "iterations": 100},
            service_capacity=[10.0, 16.0, 12.0],
            changes_count=1
        )
        assert output.deployment == [0, 1, 1, 2]
        assert output.score == 0.0667
        assert output.changes_count == 1
