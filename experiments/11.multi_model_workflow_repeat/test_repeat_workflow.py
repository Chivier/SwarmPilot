#!/usr/bin/env python3
"""
Unit tests for experiment 11: Multi-Model Workflow with Repeat Execution.

Tests the core data structures and logic for workflow repeat functionality.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_dynamic_workflow import (
    WorkflowTaskData,
    IterationState,
    WorkflowState,
    generate_task_ids
)
from workload_generator import generate_repeat_distribution


class TestWorkflowTaskData(unittest.TestCase):
    """Test WorkflowTaskData with repeat support."""

    def test_create_task_data_with_repeat(self):
        """Test creating WorkflowTaskData with repeat_num and iteration."""
        task = WorkflowTaskData(
            task_id="task-A-min_time-workflow-0042-3-A-iter-02",
            workflow_id="wf-min_time-0042-3",
            task_type="A",
            sleep_time=1.5,
            exp_runtime=1500.0,
            is_warmup=False,
            repeat_num=3,
            iteration=2
        )

        self.assertEqual(task.task_id, "task-A-min_time-workflow-0042-3-A-iter-02")
        self.assertEqual(task.workflow_id, "wf-min_time-0042-3")
        self.assertEqual(task.task_type, "A")
        self.assertEqual(task.repeat_num, 3)
        self.assertEqual(task.iteration, 2)
        self.assertFalse(task.is_warmup)

    def test_default_repeat_values(self):
        """Test that repeat_num defaults to 1 and iteration defaults to 1."""
        task = WorkflowTaskData(
            task_id="task-A-test-001",
            workflow_id="wf-test-001",
            task_type="A",
            sleep_time=1.0,
            exp_runtime=1000.0
        )

        self.assertEqual(task.repeat_num, 1)
        self.assertEqual(task.iteration, 1)


class TestIterationState(unittest.TestCase):
    """Test IterationState class."""

    def test_iteration_state_creation(self):
        """Test creating an IterationState."""
        state = IterationState(iteration=1, total_b_tasks=5)

        self.assertEqual(state.iteration, 1)
        self.assertEqual(state.total_b_tasks, 5)
        self.assertEqual(state.completed_b_tasks, 0)
        self.assertFalse(state.is_complete())

    def test_mark_b_task_complete(self):
        """Test marking B tasks as complete."""
        state = IterationState(iteration=1, total_b_tasks=3)

        # Mark first B task complete
        state.mark_b_task_complete("task-B-001", 1.0)
        self.assertEqual(state.completed_b_tasks, 1)
        self.assertFalse(state.is_complete())

        # Mark second B task complete
        state.mark_b_task_complete("task-B-002", 2.0)
        self.assertEqual(state.completed_b_tasks, 2)
        self.assertFalse(state.is_complete())

        # Mark third B task complete - should complete iteration
        state.mark_b_task_complete("task-B-003", 3.0)
        self.assertEqual(state.completed_b_tasks, 3)
        self.assertTrue(state.is_complete())
        self.assertEqual(state.iteration_complete_time, 3.0)

    def test_duplicate_b_task_ignored(self):
        """Test that marking the same B task twice doesn't increment counter."""
        state = IterationState(iteration=1, total_b_tasks=2)

        state.mark_b_task_complete("task-B-001", 1.0)
        self.assertEqual(state.completed_b_tasks, 1)

        # Try to mark same task again
        state.mark_b_task_complete("task-B-001", 1.5)
        self.assertEqual(state.completed_b_tasks, 1)  # Should still be 1
        self.assertFalse(state.is_complete())

    def test_iteration_complete_time_is_max(self):
        """Test that iteration_complete_time is the maximum of all B task times."""
        state = IterationState(iteration=1, total_b_tasks=3)

        state.mark_b_task_complete("task-B-001", 1.0)
        state.mark_b_task_complete("task-B-002", 5.0)  # Max
        state.mark_b_task_complete("task-B-003", 2.0)

        self.assertTrue(state.is_complete())
        self.assertEqual(state.iteration_complete_time, 5.0)


class TestWorkflowState(unittest.TestCase):
    """Test WorkflowState with repeat support."""

    def test_workflow_state_creation(self):
        """Test creating a WorkflowState with repeat support."""
        state = WorkflowState(
            workflow_id="wf-test-0001-3",
            strategy="min_time",
            repeat_num=3,
            current_iteration=1,
            a_task_ids={1: "a-1", 2: "a-2", 3: "a-3"},
            b_task_ids={1: ["b-1-1", "b-1-2"], 2: ["b-2-1", "b-2-2"], 3: ["b-3-1", "b-3-2"]},
            total_b_tasks=2,
            is_warmup=False
        )

        self.assertEqual(state.workflow_id, "wf-test-0001-3")
        self.assertEqual(state.repeat_num, 3)
        self.assertEqual(state.current_iteration, 1)
        self.assertEqual(state.total_b_tasks, 2)
        self.assertFalse(state.is_workflow_complete())

    def test_single_iteration_workflow(self):
        """Test a workflow that only executes once (repeat_num=1)."""
        state = WorkflowState(
            workflow_id="wf-test-0001-1",
            strategy="min_time",
            repeat_num=1,
            current_iteration=1,
            a_task_ids={1: "a-1"},
            b_task_ids={1: ["b-1-1", "b-1-2"]},
            total_b_tasks=2
        )

        # Initialize first iteration state
        state.iteration_states[1] = IterationState(iteration=1, total_b_tasks=2)

        # Mark B tasks complete
        state.mark_b_task_complete(1, "b-1-1", 1.0)
        state.mark_b_task_complete(1, "b-1-2", 2.0)

        # Should be complete (only 1 iteration)
        self.assertTrue(state.is_iteration_complete(1))
        self.assertTrue(state.is_workflow_complete())

    def test_multi_iteration_workflow(self):
        """Test a workflow with 3 iterations."""
        state = WorkflowState(
            workflow_id="wf-test-0001-3",
            strategy="min_time",
            repeat_num=3,
            current_iteration=1,
            a_task_ids={1: "a-1", 2: "a-2", 3: "a-3"},
            b_task_ids={1: ["b-1-1", "b-1-2"], 2: ["b-2-1", "b-2-2"], 3: ["b-3-1", "b-3-2"]},
            total_b_tasks=2
        )

        # Initialize first iteration
        state.iteration_states[1] = IterationState(iteration=1, total_b_tasks=2)

        # Complete iteration 1
        state.mark_b_task_complete(1, "b-1-1", 1.0)
        state.mark_b_task_complete(1, "b-1-2", 2.0)
        self.assertTrue(state.is_iteration_complete(1))
        self.assertFalse(state.is_workflow_complete())  # Still 2 more iterations

        # Move to iteration 2
        state.start_next_iteration()
        self.assertEqual(state.current_iteration, 2)
        self.assertIn(2, state.iteration_states)

        # Complete iteration 2
        state.mark_b_task_complete(2, "b-2-1", 3.0)
        state.mark_b_task_complete(2, "b-2-2", 4.0)
        self.assertTrue(state.is_iteration_complete(2))
        self.assertFalse(state.is_workflow_complete())  # Still 1 more iteration

        # Move to iteration 3
        state.start_next_iteration()
        self.assertEqual(state.current_iteration, 3)

        # Complete iteration 3
        state.mark_b_task_complete(3, "b-3-1", 5.0)
        state.mark_b_task_complete(3, "b-3-2", 6.0)
        self.assertTrue(state.is_iteration_complete(3))
        self.assertTrue(state.is_workflow_complete())  # All iterations done!

    def test_workflow_timing(self):
        """Test workflow start and complete time tracking."""
        state = WorkflowState(
            workflow_id="wf-test-0001-2",
            strategy="min_time",
            repeat_num=2,
            current_iteration=1,
            a_task_ids={1: "a-1", 2: "a-2"},
            b_task_ids={1: ["b-1-1"], 2: ["b-2-1"]},
            total_b_tasks=1
        )

        # Mark A task submit for iteration 1 (workflow start)
        state.mark_a_task_submit(1, 100.0)
        self.assertEqual(state.workflow_start_time, 100.0)

        # Initialize and complete iteration 1
        state.iteration_states[1] = IterationState(iteration=1, total_b_tasks=1)
        state.mark_a_task_complete(1, 101.0)
        state.mark_b_task_complete(1, "b-1-1", 102.0)
        self.assertTrue(state.is_iteration_complete(1))

        # Move to iteration 2
        state.start_next_iteration()
        state.mark_a_task_submit(2, 103.0)
        state.mark_a_task_complete(2, 104.0)

        # Complete iteration 2 (final)
        state.mark_b_task_complete(2, "b-2-1", 105.0)
        self.assertTrue(state.is_workflow_complete())
        self.assertEqual(state.workflow_complete_time, 105.0)

    def test_cannot_move_beyond_repeat_num(self):
        """Test that start_next_iteration() doesn't go beyond repeat_num."""
        state = WorkflowState(
            workflow_id="wf-test-0001-2",
            strategy="min_time",
            repeat_num=2,
            current_iteration=1,
            a_task_ids={1: "a-1", 2: "a-2"},
            b_task_ids={1: ["b-1"], 2: ["b-2"]},
            total_b_tasks=1
        )

        # Move to iteration 2
        state.start_next_iteration()
        self.assertEqual(state.current_iteration, 2)

        # Try to move beyond repeat_num
        state.start_next_iteration()
        self.assertEqual(state.current_iteration, 2)  # Should stay at 2


class TestTaskIDGeneration(unittest.TestCase):
    """Test task ID generation with repeat support."""

    def test_generate_task_ids_single_iteration(self):
        """Test task ID generation for workflows with repeat_num=1."""
        fanout_values = [2, 3]
        repeat_values = [1, 1]

        all_a_ids, all_b_ids, a_by_wf_iter, b_by_wf_iter = generate_task_ids(
            num_workflows=2,
            fanout_values=fanout_values,
            repeat_values=repeat_values,
            strategy="min_time"
        )

        # Should have 2 A tasks (2 workflows * 1 iteration each)
        self.assertEqual(len(all_a_ids), 2)

        # Should have 5 B tasks (2 + 3)
        self.assertEqual(len(all_b_ids), 5)

        # Check A task ID format
        self.assertEqual(all_a_ids[0], "task-A-min_time-workflow-0000-1-A-iter-01")
        self.assertEqual(all_a_ids[1], "task-A-min_time-workflow-0001-1-A-iter-01")

        # Check B task ID format
        self.assertEqual(all_b_ids[0], "task-B-min_time-workflow-0000-1-B-00-iter-01")
        self.assertEqual(all_b_ids[1], "task-B-min_time-workflow-0000-1-B-01-iter-01")

        # Check lookup dicts
        wf1_id = "wf-min_time-0000-1"
        self.assertIn((wf1_id, 1), a_by_wf_iter)
        self.assertIn((wf1_id, 1), b_by_wf_iter)
        self.assertEqual(len(b_by_wf_iter[(wf1_id, 1)]), 2)

    def test_generate_task_ids_multi_iteration(self):
        """Test task ID generation for workflows with repeat_num>1."""
        fanout_values = [2]
        repeat_values = [3]

        all_a_ids, all_b_ids, a_by_wf_iter, b_by_wf_iter = generate_task_ids(
            num_workflows=1,
            fanout_values=fanout_values,
            repeat_values=repeat_values,
            strategy="min_time"
        )

        # Should have 3 A tasks (1 workflow * 3 iterations)
        self.assertEqual(len(all_a_ids), 3)

        # Should have 6 B tasks (2 B tasks * 3 iterations)
        self.assertEqual(len(all_b_ids), 6)

        # Check iteration numbering
        self.assertEqual(all_a_ids[0], "task-A-min_time-workflow-0000-3-A-iter-01")
        self.assertEqual(all_a_ids[1], "task-A-min_time-workflow-0000-3-A-iter-02")
        self.assertEqual(all_a_ids[2], "task-A-min_time-workflow-0000-3-A-iter-03")

        # Check B task IDs for different iterations
        self.assertEqual(all_b_ids[0], "task-B-min_time-workflow-0000-3-B-00-iter-01")
        self.assertEqual(all_b_ids[1], "task-B-min_time-workflow-0000-3-B-01-iter-01")
        self.assertEqual(all_b_ids[2], "task-B-min_time-workflow-0000-3-B-00-iter-02")
        self.assertEqual(all_b_ids[3], "task-B-min_time-workflow-0000-3-B-01-iter-02")

        # Check all iterations exist in lookup dicts
        wf_id = "wf-min_time-0000-3"
        for iteration in [1, 2, 3]:
            self.assertIn((wf_id, iteration), a_by_wf_iter)
            self.assertIn((wf_id, iteration), b_by_wf_iter)
            self.assertEqual(len(b_by_wf_iter[(wf_id, iteration)]), 2)

    def test_generate_task_ids_mixed_repeats(self):
        """Test task ID generation with mixed repeat values."""
        fanout_values = [2, 3, 1]
        repeat_values = [1, 2, 3]

        all_a_ids, all_b_ids, a_by_wf_iter, b_by_wf_iter = generate_task_ids(
            num_workflows=3,
            fanout_values=fanout_values,
            repeat_values=repeat_values,
            strategy="round_robin"
        )

        # Total A tasks = 1 + 2 + 3 = 6
        self.assertEqual(len(all_a_ids), 6)

        # Total B tasks = (2*1) + (3*2) + (1*3) = 2 + 6 + 3 = 11
        self.assertEqual(len(all_b_ids), 11)

        # Check workflow IDs include repeat_num
        self.assertIn("task-A-round_robin-workflow-0000-1-A-iter-01", all_a_ids)
        self.assertIn("task-A-round_robin-workflow-0001-2-A-iter-01", all_a_ids)
        self.assertIn("task-A-round_robin-workflow-0001-2-A-iter-02", all_a_ids)
        self.assertIn("task-A-round_robin-workflow-0002-3-A-iter-01", all_a_ids)
        self.assertIn("task-A-round_robin-workflow-0002-3-A-iter-02", all_a_ids)
        self.assertIn("task-A-round_robin-workflow-0002-3-A-iter-03", all_a_ids)


class TestRepeatDistribution(unittest.TestCase):
    """Test repeat value distribution generation."""

    def test_generate_repeat_distribution(self):
        """Test generating repeat distribution."""
        repeat_values, config = generate_repeat_distribution(100, seed=42)

        # Should generate 100 values
        self.assertEqual(len(repeat_values), 100)

        # All values should be 1, 2, or 3
        for val in repeat_values:
            self.assertIn(val, [1, 2, 3])

        # Check config
        self.assertEqual(config.min_repeat, 1)
        self.assertEqual(config.max_repeat, 3)
        self.assertGreater(config.mean_repeat, 1.0)
        self.assertLess(config.mean_repeat, 3.0)

    def test_repeat_distribution_deterministic(self):
        """Test that repeat distribution is deterministic with same seed."""
        repeat1, _ = generate_repeat_distribution(50, seed=42)
        repeat2, _ = generate_repeat_distribution(50, seed=42)

        self.assertEqual(repeat1, repeat2)

    def test_repeat_distribution_different_seeds(self):
        """Test that different seeds produce different distributions."""
        repeat1, _ = generate_repeat_distribution(50, seed=42)
        repeat2, _ = generate_repeat_distribution(50, seed=43)

        self.assertNotEqual(repeat1, repeat2)


class TestWorkflowIDParsing(unittest.TestCase):
    """Test parsing workflow IDs and task IDs to extract repeat info."""

    def test_parse_a_task_id(self):
        """Test extracting workflow_id and iteration from A task ID."""
        # Format: task-A-{strategy}-workflow-{idx}-{repeat_num}-A-iter-{iter}
        task_id = "task-A-min_time-workflow-0042-3-A-iter-02"
        parts = task_id.split("-")

        # Extract components (0:task, 1:A, 2:strategy, 3:workflow, 4:idx, 5:repeat_num, 6:A, 7:iter, 8:iter_num)
        strategy = parts[2]
        wf_idx = parts[4]
        repeat_num = parts[5]
        iter_str = parts[8]

        workflow_id = f"wf-{strategy}-{wf_idx}-{repeat_num}"
        iteration = int(iter_str)

        self.assertEqual(workflow_id, "wf-min_time-0042-3")
        self.assertEqual(iteration, 2)

    def test_parse_b_task_id(self):
        """Test extracting workflow_id and iteration from B task ID."""
        # Format: task-B-{strategy}-workflow-{idx}-{repeat_num}-B-{b_idx}-iter-{iter}
        task_id = "task-B-round_robin-workflow-0123-2-B-05-iter-01"
        parts = task_id.split("-")

        # Extract components (0:task, 1:B, 2:strategy, 3:workflow, 4:idx, 5:repeat_num, 6:B, 7:b_idx, 8:iter, 9:iter_num)
        strategy = parts[2]
        wf_idx = parts[4]
        repeat_num = parts[5]
        b_idx = parts[7]
        iter_str = parts[9]

        workflow_id = f"wf-{strategy}-{wf_idx}-{repeat_num}"
        iteration = int(iter_str)
        b_index = int(b_idx)

        self.assertEqual(workflow_id, "wf-round_robin-0123-2")
        self.assertEqual(iteration, 1)
        self.assertEqual(b_index, 5)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
