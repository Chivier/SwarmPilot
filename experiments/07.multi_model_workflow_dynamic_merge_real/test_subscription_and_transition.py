#!/usr/bin/env python3
"""
Integration tests for task subscription and phase transition with mocked schedulers.

Tests the core workflow mechanisms:
1. Task subscription: A1 → B1/B2, B1/B2 → A2
2. Phase transition: Verifying correct dependency handling
3. Complete workflow: From submission to completion to metrics

This test uses MOCKED scheduler responses to verify the workflow logic
without requiring actual scheduler instances to be running.
"""

import sys
import json
import time
import pytest
from pathlib import Path
from typing import List, Dict, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from workload_generator import (
    generate_workflow_from_dataset,
    load_dataset,
)

# Dataset path
DATASET_FILE = Path(__file__).parent / "data" / "dataset.jsonl"


@dataclass
class MockTaskState:
    """Mock task state for simulating scheduler behavior."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    output: str = ""
    subscribe_task_ids: List[str] = None

    def __post_init__(self):
        if self.subscribe_task_ids is None:
            self.subscribe_task_ids = []


class MockScheduler:
    """Mock scheduler that simulates task execution and subscription."""

    # Class-level registry of all schedulers for cross-scheduler subscription
    _all_schedulers: Dict[str, 'MockScheduler'] = {}

    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, MockTaskState] = {}
        self.strategy = "round_robin"
        self.completion_callbacks = []
        # Register this scheduler
        MockScheduler._all_schedulers[name] = self

    def clear_tasks(self):
        """Clear all tasks."""
        self.tasks.clear()
        self.completion_callbacks.clear()

    def set_strategy(self, strategy: str):
        """Set scheduling strategy."""
        self.strategy = strategy

    def submit_task(self, task_id: str, task_data: dict, subscribe_task_ids: List[str] = None):
        """Submit a task to the mock scheduler."""
        if subscribe_task_ids is None:
            subscribe_task_ids = []

        # Create task state
        self.tasks[task_id] = MockTaskState(
            task_id=task_id,
            status="pending",
            subscribe_task_ids=subscribe_task_ids
        )

        return {"success": True, "task_id": task_id}

    def query_task(self, task_id: str):
        """Query task status."""
        if task_id not in self.tasks:
            return {"status": "not_found", "task_id": task_id}

        task = self.tasks[task_id]
        return {
            "status": task.status,
            "task_id": task.task_id,
            "output": task.output
        }

    def complete_task(self, task_id: str, output: str = "mock output"):
        """Simulate task completion and trigger subscriptions."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = "completed"
        task.output = output

        # Trigger subscribed tasks (simulate scheduler's subscription mechanism)
        # Look for subscribed tasks in ALL schedulers (cross-scheduler subscription)
        for subscribed_id in task.subscribe_task_ids:
            triggered = False
            for scheduler in MockScheduler._all_schedulers.values():
                if subscribed_id in scheduler.tasks:
                    # Auto-trigger subscribed task (change from pending to running)
                    subscribed_task = scheduler.tasks[subscribed_id]
                    if subscribed_task.status == "pending":
                        subscribed_task.status = "running"
                    triggered = True
                    break

            if not triggered:
                print(f"Warning: Subscribed task {subscribed_id} not found in any scheduler")

    def check_and_trigger_merge(self, workflow_id: str, b2_task_ids: List[str], merge_task_id: str):
        """
        Check if all B2 tasks for a workflow are completed, and trigger merge task if so.

        This simulates the real scheduler's behavior where Thread 4 (B2TaskReceiver)
        monitors all B2 tasks and triggers the merge task when all are complete.

        Args:
            workflow_id: The workflow ID
            b2_task_ids: List of all B2 task IDs for this workflow
            merge_task_id: The merge (A2) task ID to trigger
        """
        # Check if all B2 tasks are completed
        all_complete = True
        for b2_id in b2_task_ids:
            if b2_id in self.tasks:
                if self.tasks[b2_id].status != "completed":
                    all_complete = False
                    break

        # If all B2 tasks complete, trigger merge task
        if all_complete:
            # Find merge task in all schedulers and trigger it
            for scheduler in MockScheduler._all_schedulers.values():
                if merge_task_id in scheduler.tasks:
                    merge_task = scheduler.tasks[merge_task_id]
                    if merge_task.status == "pending":
                        merge_task.status = "running"
                    break

    def start_task(self, task_id: str):
        """Manually start a task (change from pending to running)."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "running"


class TestSubscriptionAndTransition:
    """Test task subscription and phase transition mechanisms with mocked schedulers."""

    @pytest.fixture(autouse=True)
    def setup_schedulers(self):
        """Setup mock schedulers before each test."""
        self.scheduler_a = MockScheduler("scheduler_a")
        self.scheduler_b = MockScheduler("scheduler_b")

        yield

        # Cleanup
        self.scheduler_a.clear_tasks()
        self.scheduler_b.clear_tasks()

    def test_single_workflow_subscription_chain(self):
        """
        Test 1: Single workflow with complete subscription chain.

        Workflow structure:
        A1 (boot) → B1_0, B1_1 (queries) → B2_0, B2_1 (queries) → A2 (summary)

        Verification:
        1. A1 completes → B1 tasks are automatically triggered
        2. All B1 complete → B2 tasks are automatically triggered
        3. All B2 complete → A2 task is automatically triggered
        4. A2 completes → workflow finishes
        """
        print("\n" + "="*80)
        print("TEST 1: Single Workflow Subscription Chain (Mocked)")
        print("="*80)

        # Load dataset and use first entry
        dataset = load_dataset(DATASET_FILE)
        entry = dataset[0]

        print(f"\nDataset entry:")
        print(f"  Boot: {entry.boot[:50]}...")
        print(f"  Queries: {len(entry.queries)} queries")
        print(f"  Summary: {entry.summary[:50]}...")
        print(f"  Fanout: {entry.fanout}")

        # Generate task IDs (use only 2 queries for faster test)
        num_queries = min(2, entry.fanout)
        workflow_id = "wf-test-0000"
        a1_task_id = f"task-A-test-0000-A"
        a2_task_id = f"task-A-test-0000-merge"
        b1_task_ids = [f"task-B1-test-0000-B1-{i:02d}" for i in range(num_queries)]
        b2_task_ids = [f"task-B2-test-0000-B2-{i:02d}" for i in range(num_queries)]

        print(f"\nTask IDs generated:")
        print(f"  A1: {a1_task_id}")
        print(f"  B1: {b1_task_ids}")
        print(f"  B2: {b2_task_ids}")
        print(f"  A2: {a2_task_id}")

        # Phase 1: Submit A1 task
        print(f"\n--- Phase 1: Submit A1 task ---")
        a1_result = self.scheduler_a.submit_task(
            task_id=a1_task_id,
            task_data={
                "workflow_id": workflow_id,
                "sentence": entry.boot,
                "max_tokens": 4096,
            },
            subscribe_task_ids=b1_task_ids
        )
        print(f"A1 submitted: {a1_result}")
        assert a1_result["success"], "A1 submission should succeed"

        # Verify A1 is pending
        a1_status = self.scheduler_a.query_task(a1_task_id)
        assert a1_status["status"] == "pending", "A1 should be pending after submission"

        # Phase 2: Submit B1 tasks (they should remain pending until A1 completes)
        print(f"\n--- Phase 2: Submit B1 tasks (will be triggered by A1) ---")
        for i, b1_id in enumerate(b1_task_ids):
            query_text = entry.queries[i] if isinstance(entry.queries[i], str) else entry.queries[i].get("input", "")

            b1_result = self.scheduler_b.submit_task(
                task_id=b1_id,
                task_data={
                    "workflow_id": workflow_id,
                    "sentence": query_text,
                    "max_tokens": 300,
                },
                subscribe_task_ids=[b2_task_ids[i]]
            )
            print(f"B1_{i} submitted: {b1_result}")
            assert b1_result["success"], f"B1_{i} submission should succeed"

        # Verify B1 tasks are pending (not yet triggered)
        for b1_id in b1_task_ids:
            b1_status = self.scheduler_b.query_task(b1_id)
            assert b1_status["status"] == "pending", f"B1 {b1_id} should be pending before A1 completes"

        # Phase 3: Complete A1 → should trigger B1 tasks
        print(f"\n--- Phase 3: Complete A1 (triggers B1 tasks) ---")
        self.scheduler_a.start_task(a1_task_id)
        self.scheduler_a.complete_task(a1_task_id, output="A1 mock output")

        a1_status = self.scheduler_a.query_task(a1_task_id)
        print(f"A1 completed: status={a1_status['status']}")
        assert a1_status["status"] == "completed", "A1 should be completed"

        # Verify B1 tasks are now triggered (status changed to running)
        print(f"\n--- Phase 4: Verify B1 tasks auto-triggered ---")
        for b1_id in b1_task_ids:
            b1_status = self.scheduler_b.query_task(b1_id)
            print(f"B1 {b1_id}: status={b1_status['status']}")
            assert b1_status["status"] == "running", \
                f"B1 {b1_id} should be auto-triggered (running) after A1 completes"

        # Phase 5: Submit B2 tasks
        print(f"\n--- Phase 5: Submit B2 tasks (will be triggered by B1) ---")
        for i, b2_id in enumerate(b2_task_ids):
            query_text = entry.queries[i] if isinstance(entry.queries[i], str) else entry.queries[i].get("input", "")

            # NOTE: B2 tasks do NOT directly subscribe to A2
            # The merge task (A2) is triggered by Thread 4 (B2TaskReceiver)
            # when it detects ALL B2 tasks for a workflow are complete
            b2_result = self.scheduler_b.submit_task(
                task_id=b2_id,
                task_data={
                    "workflow_id": workflow_id,
                    "sentence": query_text,
                    "max_tokens": 1,
                },
                subscribe_task_ids=[]  # No direct subscription to A2
            )
            print(f"B2_{i} submitted: {b2_result}")
            assert b2_result["success"], f"B2_{i} submission should succeed"

        # Phase 6: Complete all B1 tasks → should trigger B2 tasks
        print(f"\n--- Phase 6: Complete all B1 tasks (triggers B2 tasks) ---")
        for i, b1_id in enumerate(b1_task_ids):
            self.scheduler_b.complete_task(b1_id, output=f"B1_{i} mock output")
            b1_status = self.scheduler_b.query_task(b1_id)
            print(f"B1_{i} completed: status={b1_status['status']}")
            assert b1_status["status"] == "completed", f"B1_{i} should be completed"

        # Verify B2 tasks are now triggered
        print(f"\n--- Phase 7: Verify B2 tasks auto-triggered ---")
        for i, b2_id in enumerate(b2_task_ids):
            b2_status = self.scheduler_b.query_task(b2_id)
            print(f"B2_{i}: status={b2_status['status']}")
            assert b2_status["status"] == "running", \
                f"B2_{i} should be auto-triggered (running) after B1_{i} completes"

        # Phase 8: Submit A2 task
        print(f"\n--- Phase 8: Submit A2 task (will be triggered by last B2) ---")
        a2_result = self.scheduler_a.submit_task(
            task_id=a2_task_id,
            task_data={
                "workflow_id": workflow_id,
                "sentence": entry.summary,
                "max_tokens": 4096,
            },
            subscribe_task_ids=[]
        )
        print(f"A2 submitted: {a2_result}")
        assert a2_result["success"], "A2 submission should succeed"

        # Verify A2 is pending
        a2_status = self.scheduler_a.query_task(a2_task_id)
        assert a2_status["status"] == "pending", "A2 should be pending before B2 completes"

        # Phase 9: Complete all B2 tasks
        print(f"\n--- Phase 9: Complete all B2 tasks ---")
        for i, b2_id in enumerate(b2_task_ids):
            self.scheduler_b.complete_task(b2_id, output=f"B2_{i} mock output")
            b2_status = self.scheduler_b.query_task(b2_id)
            print(f"B2_{i} completed: status={b2_status['status']}")
            assert b2_status["status"] == "completed", f"B2_{i} should be completed"

        # Phase 10: Simulate Thread 4 (B2TaskReceiver) checking if all B2 complete
        print(f"\n--- Phase 10: Check all B2 complete and trigger A2 (simulates Thread 4) ---")
        # In real implementation, Thread 4 monitors B2 completions and triggers merge when all done
        self.scheduler_b.check_and_trigger_merge(workflow_id, b2_task_ids, a2_task_id)

        # Verify A2 is now triggered (only after all B2 complete)
        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"A2: status={a2_status['status']}")
        assert a2_status["status"] == "running", \
            "A2 should be auto-triggered (running) after all B2 complete"

        # Phase 11: Complete A2
        print(f"\n--- Phase 11: Complete A2 (workflow finishes) ---")
        self.scheduler_a.complete_task(a2_task_id, output="A2 mock output")
        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"A2 completed: status={a2_status['status']}")
        assert a2_status["status"] == "completed", "A2 should be completed"

        # Verify complete workflow
        print(f"\n--- Phase 12: Verify Complete Workflow ---")
        print(f"✅ A1 completed: {a1_task_id}")
        print(f"✅ B1 tasks completed: {b1_task_ids}")
        print(f"✅ B2 tasks completed: {b2_task_ids}")
        print(f"✅ A2 completed: {a2_task_id}")
        print(f"\n🎉 Complete workflow subscription chain verified!")

    def test_dependency_transition_timing(self):
        """
        Test 2: Verify dependency transition behavior.

        Checks that:
        1. B tasks do NOT start before A1 completes
        2. A2 does NOT start before all B tasks complete
        3. Subscription triggers tasks immediately upon completion
        """
        print("\n" + "="*80)
        print("TEST 2: Dependency Transition Behavior (Mocked)")
        print("="*80)

        # Load dataset and use first entry
        dataset = load_dataset(DATASET_FILE)
        entry = dataset[0]

        # Generate task IDs (use only 1 query for simpler test)
        workflow_id = "wf-timing-0000"
        a1_task_id = f"task-A-timing-0000-A"
        a2_task_id = f"task-A-timing-0000-merge"
        b1_task_id = f"task-B1-timing-0000-B1-00"
        b2_task_id = f"task-B2-timing-0000-B2-00"

        # Submit A1 with subscription to B1
        print(f"\n--- Submit A1 (subscribes to B1) ---")
        self.scheduler_a.submit_task(
            task_id=a1_task_id,
            task_data={"sentence": entry.boot, "max_tokens": 4096},
            subscribe_task_ids=[b1_task_id]
        )

        # Submit B1 (should be pending)
        print(f"--- Submit B1 (subscribes to B2) ---")
        query_text = entry.queries[0] if isinstance(entry.queries[0], str) else entry.queries[0].get("input", "")
        self.scheduler_b.submit_task(
            task_id=b1_task_id,
            task_data={"sentence": query_text, "max_tokens": 300},
            subscribe_task_ids=[b2_task_id]
        )

        # Verify B1 is pending (NOT running before A1 completes)
        print(f"--- Verify B1 is pending before A1 completes ---")
        b1_status = self.scheduler_b.query_task(b1_task_id)
        print(f"B1 status before A1 completes: {b1_status['status']}")
        assert b1_status["status"] == "pending", \
            "B1 should be pending (NOT running) before A1 completes"

        # Complete A1
        print(f"\n--- Complete A1 ---")
        self.scheduler_a.start_task(a1_task_id)
        self.scheduler_a.complete_task(a1_task_id)

        # Verify B1 is now triggered (status changed to running)
        print(f"--- Verify B1 is triggered after A1 completes ---")
        b1_status = self.scheduler_b.query_task(b1_task_id)
        print(f"B1 status after A1 completes: {b1_status['status']}")
        assert b1_status["status"] == "running", \
            "B1 should be auto-triggered (running) immediately after A1 completes"

        # Submit B2 and A2
        print(f"\n--- Submit B2 and A2 ---")
        self.scheduler_b.submit_task(
            task_id=b2_task_id,
            task_data={"sentence": query_text, "max_tokens": 1},
            subscribe_task_ids=[]  # No direct subscription to A2
        )
        self.scheduler_a.submit_task(
            task_id=a2_task_id,
            task_data={"sentence": entry.summary, "max_tokens": 4096},
            subscribe_task_ids=[]
        )

        # Verify A2 is pending before B2 completes
        print(f"--- Verify A2 is pending before B2 completes ---")
        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"A2 status before B2 completes: {a2_status['status']}")
        assert a2_status["status"] == "pending", \
            "A2 should be pending before B2 completes"

        # Complete B1 (triggers B2)
        print(f"\n--- Complete B1 (triggers B2) ---")
        self.scheduler_b.complete_task(b1_task_id)
        b2_status = self.scheduler_b.query_task(b2_task_id)
        print(f"B2 status after B1 completes: {b2_status['status']}")
        assert b2_status["status"] == "running", "B2 should be triggered after B1 completes"

        # Complete B2 and check for merge
        print(f"\n--- Complete B2 and check for merge (simulates Thread 4) ---")
        self.scheduler_b.complete_task(b2_task_id)
        self.scheduler_b.check_and_trigger_merge(workflow_id, [b2_task_id], a2_task_id)

        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"A2 status after B2 completes: {a2_status['status']}")
        assert a2_status["status"] == "running", \
            "A2 should be auto-triggered (running) after all B2 complete"

        print(f"\n✅ Dependency transition behavior verified!")
        print(f"   - B1 waits for A1 ✓")
        print(f"   - B2 waits for B1 ✓")
        print(f"   - A2 waits for B2 ✓")
        print(f"   - Immediate trigger upon completion ✓")

    def test_fanout_subscription_correctness(self):
        """
        Test 3: Verify fanout subscription correctness.

        For a workflow with fanout=N:
        1. A1 should subscribe to N B1 tasks
        2. Each B1_i should subscribe to exactly one B2_i
        3. Last B2 should subscribe to A2
        4. All N B1 and N B2 tasks should complete
        """
        print("\n" + "="*80)
        print("TEST 3: Fanout Subscription Correctness (Mocked)")
        print("="*80)

        # Load dataset and find entry with fanout >= 3
        dataset = load_dataset(DATASET_FILE)
        entry = None
        for e in dataset:
            if e.fanout >= 3:
                entry = e
                break

        if entry is None:
            pytest.skip("No dataset entry with fanout >= 3 found")

        # Use first 3 queries only
        num_queries = min(3, entry.fanout)
        print(f"\nTesting with fanout={num_queries}")

        # Generate task IDs
        workflow_id = "wf-fanout-0000"
        a1_task_id = f"task-A-fanout-0000-A"
        a2_task_id = f"task-A-fanout-0000-merge"
        b1_task_ids = [f"task-B1-fanout-0000-B1-{i:02d}" for i in range(num_queries)]
        b2_task_ids = [f"task-B2-fanout-0000-B2-{i:02d}" for i in range(num_queries)]

        print(f"Task structure:")
        print(f"  A1: {a1_task_id} → subscribes to {len(b1_task_ids)} B1 tasks")
        print(f"  B1: {b1_task_ids}")
        print(f"  B2: {b2_task_ids}")
        print(f"  A2: {a2_task_id}")

        # Submit A1 with subscription to all B1 tasks
        print(f"\n--- Submit A1 (subscribes to {num_queries} B1 tasks) ---")
        self.scheduler_a.submit_task(
            task_id=a1_task_id,
            task_data={"sentence": entry.boot, "max_tokens": 4096},
            subscribe_task_ids=b1_task_ids
        )

        # Submit all B1 tasks
        print(f"--- Submit {num_queries} B1 tasks ---")
        for i in range(num_queries):
            query_text = entry.queries[i] if isinstance(entry.queries[i], str) else entry.queries[i].get("input", "")
            self.scheduler_b.submit_task(
                task_id=b1_task_ids[i],
                task_data={"sentence": query_text, "max_tokens": 300},
                subscribe_task_ids=[b2_task_ids[i]]
            )

        # Verify all B1 are pending
        for i, b1_id in enumerate(b1_task_ids):
            b1_status = self.scheduler_b.query_task(b1_id)
            assert b1_status["status"] == "pending", f"B1_{i} should be pending before A1 completes"

        # Complete A1
        print(f"\n--- Complete A1 ---")
        self.scheduler_a.start_task(a1_task_id)
        self.scheduler_a.complete_task(a1_task_id)

        # Verify ALL B1 tasks are triggered
        print(f"--- Verify all {num_queries} B1 tasks are triggered ---")
        for i, b1_id in enumerate(b1_task_ids):
            b1_status = self.scheduler_b.query_task(b1_id)
            print(f"B1_{i}: status={b1_status['status']}")
            assert b1_status["status"] == "running", \
                f"B1_{i} should be auto-triggered after A1 completes"

        # Submit all B2 tasks
        print(f"\n--- Submit {num_queries} B2 tasks ---")
        for i in range(num_queries):
            query_text = entry.queries[i] if isinstance(entry.queries[i], str) else entry.queries[i].get("input", "")
            self.scheduler_b.submit_task(
                task_id=b2_task_ids[i],
                task_data={"sentence": query_text, "max_tokens": 1},
                subscribe_task_ids=[]  # No direct subscription to A2
            )

        # Submit A2
        print(f"--- Submit A2 ---")
        self.scheduler_a.submit_task(
            task_id=a2_task_id,
            task_data={"sentence": entry.summary, "max_tokens": 4096},
            subscribe_task_ids=[]
        )

        # Complete all B1 tasks one by one
        print(f"\n--- Complete all {num_queries} B1 tasks ---")
        for i in range(num_queries):
            self.scheduler_b.complete_task(b1_task_ids[i])

            # Verify corresponding B2 is triggered
            b2_status = self.scheduler_b.query_task(b2_task_ids[i])
            print(f"After B1_{i} completes: B2_{i} status={b2_status['status']}")
            assert b2_status["status"] == "running", \
                f"B2_{i} should be triggered after B1_{i} completes"

        # Verify A2 is still pending (not all B2 completed yet)
        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"\nA2 status before B2 completion: {a2_status['status']}")
        assert a2_status["status"] == "pending", "A2 should still be pending"

        # Complete all B2 tasks
        print(f"\n--- Complete all {num_queries} B2 tasks ---")
        for i in range(num_queries):
            self.scheduler_b.complete_task(b2_task_ids[i])
            print(f"B2_{i} completed")

        # Simulate Thread 4 checking for all B2 complete and triggering merge
        print(f"\n--- Check all B2 complete and trigger A2 (simulates Thread 4) ---")
        self.scheduler_b.check_and_trigger_merge(workflow_id, b2_task_ids, a2_task_id)

        # Verify A2 is now triggered (after all B2 complete)
        a2_status = self.scheduler_a.query_task(a2_task_id)
        print(f"A2 status after all B2 complete: {a2_status['status']}")
        assert a2_status["status"] == "running", \
            "A2 should be triggered after all B2 complete"

        # Complete A2
        self.scheduler_a.complete_task(a2_task_id)

        print(f"\n✅ Fanout subscription correctness verified!")
        print(f"   - A1 → {num_queries}×B1 ✓")
        print(f"   - Each B1_i → B2_i ✓")
        print(f"   - Last B2 → A2 ✓")
        print(f"   - Complete workflow chain: A1 → {num_queries}×B1 → {num_queries}×B2 → A2 ✓")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
