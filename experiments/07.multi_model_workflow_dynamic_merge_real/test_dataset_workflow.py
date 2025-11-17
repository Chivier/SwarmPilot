#!/usr/bin/env python3
"""
Unit tests for dataset-based workflow generation and execution.

Tests the complete workflow from dataset loading to metrics calculation:
1. Load dataset.jsonl
2. Generate workflow tasks (A1, B1, B2, A2)
3. Verify data mapping (boot→A1, queries→B1/B2, summary→A2)
4. Verify max_tokens settings
5. Verify fanout calculation
6. Verify task structure and relationships
"""

import sys
import json
import pytest
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from workload_generator import (
    generate_workflow_from_dataset,
    load_dataset,
    DatasetEntry,
    WorkflowWorkload,
    WorkloadConfig
)


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_dataset_basic(self):
        """Test basic dataset loading."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        dataset = load_dataset(dataset_path)

        assert len(dataset) > 0, "Dataset should not be empty"
        assert all(isinstance(entry, DatasetEntry) for entry in dataset), \
            "All entries should be DatasetEntry instances"

    def test_dataset_entry_structure(self):
        """Test dataset entry has correct structure."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        dataset = load_dataset(dataset_path)

        # Check first entry
        entry = dataset[0]
        assert hasattr(entry, 'boot'), "Entry should have 'boot' field"
        assert hasattr(entry, 'queries'), "Entry should have 'queries' field"
        assert hasattr(entry, 'summary'), "Entry should have 'summary' field"
        assert hasattr(entry, 'fanout'), "Entry should have 'fanout' field"

        # Verify types
        assert isinstance(entry.boot, str), "boot should be string"
        assert isinstance(entry.queries, list), "queries should be list"
        assert isinstance(entry.summary, str), "summary should be string"
        assert isinstance(entry.fanout, int), "fanout should be int"

        # Verify fanout calculation
        assert entry.fanout == len(entry.queries), \
            "fanout should equal number of queries"

    def test_dataset_queries_format(self):
        """Test that queries are properly extracted."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        dataset = load_dataset(dataset_path)

        for entry in dataset[:5]:  # Test first 5 entries
            assert len(entry.queries) > 0, "Each entry should have at least one query"
            assert all(isinstance(q, str) for q in entry.queries), \
                "All queries should be strings"


class TestWorkflowGeneration:
    """Test workflow generation from dataset."""

    def test_generate_workflow_basic(self):
        """Test basic workflow generation."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=10,
            seed=42
        )

        assert isinstance(workflow, WorkflowWorkload), \
            "Should return WorkflowWorkload instance"
        assert isinstance(config, WorkloadConfig), \
            "Should return WorkloadConfig instance"

        # Verify workflow counts
        assert len(workflow.a1_times) == 10, "Should generate 10 A1 tasks"
        assert len(workflow.a2_times) == 10, "Should generate 10 A2 tasks"
        assert len(workflow.fanout_values) == 10, "Should have 10 fanout values"
        assert len(workflow.b1_times) == 10, "Should have B1 times for 10 workflows"
        assert len(workflow.b2_times) == 10, "Should have B2 times for 10 workflows"

    def test_workflow_with_replacement_sampling(self):
        """Test WITH REPLACEMENT sampling (num_workflows > dataset size)."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        dataset = load_dataset(dataset_path)
        dataset_size = len(dataset)

        # Request more workflows than dataset entries
        num_workflows = dataset_size + 50
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=num_workflows,
            seed=42
        )

        assert len(workflow.a1_times) == num_workflows, \
            f"Should generate {num_workflows} workflows (more than dataset size {dataset_size})"

    def test_workflow_times_are_zero(self):
        """Test that all execution times are set to 0.0."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=5,
            seed=42
        )

        # All A times should be 0.0
        assert all(t == 0.0 for t in workflow.a1_times), \
            "All A1 times should be 0.0"
        assert all(t == 0.0 for t in workflow.a2_times), \
            "All A2 times should be 0.0"

        # All B times should be 0.0
        for b1_workflow in workflow.b1_times:
            assert all(t == 0.0 for t in b1_workflow), \
                "All B1 times should be 0.0"
        for b2_workflow in workflow.b2_times:
            assert all(t == 0.0 for t in b2_workflow), \
                "All B2 times should be 0.0"

    def test_fanout_calculation(self):
        """Test that fanout matches query count from dataset."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        dataset = load_dataset(dataset_path)

        # Use seed to ensure reproducible sampling
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=5,
            seed=42
        )

        # Verify B task counts match fanout
        for i, fanout in enumerate(workflow.fanout_values):
            assert len(workflow.b1_times[i]) == fanout, \
                f"Workflow {i}: B1 task count should match fanout"
            assert len(workflow.b2_times[i]) == fanout, \
                f"Workflow {i}: B2 task count should match fanout"

    def test_workflow_config(self):
        """Test workflow configuration."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=10,
            seed=42
        )

        assert config.name == "dataset_based_workflow", \
            "Config name should be 'dataset_based_workflow'"
        assert config.min_time == 0.0, "min_time should be 0.0"
        assert config.max_time == 0.0, "max_time should be 0.0"
        assert config.mean_time == 0.0, "mean_time should be 0.0"
        assert config.std_time == 0.0, "std_time should be 0.0"


class TestTaskDataMapping:
    """Test that task data is correctly mapped from dataset."""

    def test_task_structure_creation(self):
        """Test creating tasks with proper data mapping."""
        from test_dynamic_workflow import WorkflowTaskData

        # Simulate dataset entry
        boot_text = "Test boot prompt"
        query_text = "Test query"
        summary_text = "Test summary"

        # Create A1 task
        a1_task = WorkflowTaskData(
            task_id="task-A-test-workflow-0000-A",
            workflow_id="wf-test-0000",
            task_type="A",
            sleep_time=0.0,
            exp_runtime=0.0,
            sentence=boot_text,
            max_tokens=4096
        )

        assert a1_task.sentence == boot_text, "A1 should use boot text"
        assert a1_task.max_tokens == 4096, "A1 max_tokens should be 4096"
        assert a1_task.sleep_time == 0.0, "A1 sleep_time should be 0.0"

        # Create A2 task
        a2_task = WorkflowTaskData(
            task_id="task-A-test-workflow-0000-merge",
            workflow_id="wf-test-0000",
            task_type="A",
            sleep_time=0.0,
            exp_runtime=0.0,
            sentence=summary_text,
            max_tokens=4096
        )

        assert a2_task.sentence == summary_text, "A2 should use summary text"
        assert a2_task.max_tokens == 4096, "A2 max_tokens should be 4096"

        # Create B1 task
        b1_task = WorkflowTaskData(
            task_id="task-B1-test-workflow-0000-B1-00",
            workflow_id="wf-test-0000",
            task_type="B1",
            sleep_time=0.0,
            exp_runtime=0.0,
            b_index=0,
            sentence=query_text,
            max_tokens=300
        )

        assert b1_task.sentence == query_text, "B1 should use query text"
        assert b1_task.max_tokens == 300, "B1 max_tokens should be 300"

        # Create B2 task
        b2_task = WorkflowTaskData(
            task_id="task-B2-test-workflow-0000-B2-00",
            workflow_id="wf-test-0000",
            task_type="B2",
            sleep_time=0.0,
            exp_runtime=0.0,
            b_index=0,
            sentence=query_text,
            max_tokens=1
        )

        assert b2_task.sentence == query_text, "B2 should use query text (same as B1)"
        assert b2_task.max_tokens == 1, "B2 max_tokens should be 1"


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_workflow_data_flow(self):
        """Test complete data flow from dataset to task creation."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"

        # Step 1: Load dataset
        dataset = load_dataset(dataset_path)
        assert len(dataset) > 0, "Dataset should load successfully"

        # Step 2: Generate workflow
        workflow, config = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=3,
            seed=42
        )

        # Step 3: Verify workflow structure
        assert len(workflow.a1_times) == 3
        assert len(workflow.a2_times) == 3
        assert len(workflow.fanout_values) == 3

        # Step 4: Verify fanout consistency
        total_b1_tasks = sum(len(b1) for b1 in workflow.b1_times)
        total_b2_tasks = sum(len(b2) for b2 in workflow.b2_times)
        total_fanout = sum(workflow.fanout_values)

        assert total_b1_tasks == total_fanout, \
            "Total B1 tasks should equal sum of fanouts"
        assert total_b2_tasks == total_fanout, \
            "Total B2 tasks should equal sum of fanouts"
        assert total_b1_tasks == total_b2_tasks, \
            "B1 and B2 task counts should match"

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"

        # Generate workflow twice with same seed
        workflow1, _ = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=5,
            seed=42
        )
        workflow2, _ = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=5,
            seed=42
        )

        # Should produce identical fanout sequences
        assert workflow1.fanout_values == workflow2.fanout_values, \
            "Same seed should produce same fanout sequence"

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different samples."""
        dataset_path = Path(__file__).parent / "data" / "dataset.jsonl"

        # Generate workflows with different seeds
        workflow1, _ = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=20,
            seed=42
        )
        workflow2, _ = generate_workflow_from_dataset(
            dataset_path=dataset_path,
            num_workflows=20,
            seed=999
        )

        # Should produce different fanout sequences (with high probability)
        assert workflow1.fanout_values != workflow2.fanout_values, \
            "Different seeds should likely produce different samples"


class TestErrorHandling:
    """Test error handling."""

    def test_empty_dataset_error(self, tmp_path):
        """Test that empty dataset raises error."""
        # Create empty dataset file
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            generate_workflow_from_dataset(
                dataset_path=empty_file,
                num_workflows=5,
                seed=42
            )

    def test_nonexistent_file_error(self):
        """Test that nonexistent file raises error."""
        fake_path = Path("/nonexistent/path/dataset.jsonl")

        with pytest.raises(FileNotFoundError):
            generate_workflow_from_dataset(
                dataset_path=fake_path,
                num_workflows=5,
                seed=42
            )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
