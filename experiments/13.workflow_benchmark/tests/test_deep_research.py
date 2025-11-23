"""Unit tests for Deep Research workflow components."""

import pytest
import threading
from queue import Queue
from unittest.mock import Mock

from type2_deep_research.config import DeepResearchConfig
from type2_deep_research.workflow_data import DeepResearchWorkflowData
from type2_deep_research.submitters import (
    ATaskSubmitter,
    B1TaskSubmitter,
    B2TaskSubmitter,
    MergeTaskSubmitter
)
from type2_deep_research.receivers import (
    ATaskReceiver,
    B1TaskReceiver,
    B2TaskReceiver,
    MergeTaskReceiver
)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_config_defaults():
    """Test configuration default values."""
    config = DeepResearchConfig()

    assert config.qps == 1.0
    assert config.duration == 600
    assert config.num_workflows == 600
    assert config.fanout_count == 3
    assert config.model_a_id == "llm_service_small_model"
    assert config.model_b_id == "llm_service_small_model"
    assert config.model_merge_id == "llm_service_small_model"


def test_config_from_env(monkeypatch):
    """Test configuration from environment variables."""
    monkeypatch.setenv("QPS", "2.0")
    monkeypatch.setenv("DURATION", "300")
    monkeypatch.setenv("NUM_WORKFLOWS", "100")
    monkeypatch.setenv("FANOUT_COUNT", "5")

    config = DeepResearchConfig.from_env()

    assert config.qps == 2.0
    assert config.duration == 300
    assert config.num_workflows == 100
    assert config.fanout_count == 5


# ============================================================================
# Workflow Data Tests
# ============================================================================

def test_workflow_data_initialization():
    """Test workflow data initialization."""
    workflow = DeepResearchWorkflowData(
        workflow_id="test-001",
        fanout_count=3
    )

    assert workflow.workflow_id == "test-001"
    assert workflow.fanout_count == 3
    assert workflow.a_result is None
    assert len(workflow.b1_task_ids) == 0
    assert len(workflow.b2_task_ids) == 0
    assert len(workflow.b1_complete_times) == 0
    assert len(workflow.b2_complete_times) == 0
    assert workflow.merge_task_id is None
    assert workflow.merge_complete_time is None


def test_workflow_all_b1_complete():
    """Test B1 completion check."""
    workflow = DeepResearchWorkflowData(
        workflow_id="test-001",
        fanout_count=3
    )

    # Not complete initially
    assert workflow.all_b1_complete() is False

    # After 2 B1 completions
    workflow.b1_complete_times["test-001-B1-0"] = 100.0
    workflow.b1_complete_times["test-001-B1-1"] = 101.0
    assert workflow.all_b1_complete() is False

    # After 3 B1 completions (all done)
    workflow.b1_complete_times["test-001-B1-2"] = 102.0
    assert workflow.all_b1_complete() is True


def test_workflow_all_b2_complete():
    """Test B2 completion check."""
    workflow = DeepResearchWorkflowData(
        workflow_id="test-001",
        fanout_count=2
    )

    # Not complete initially
    assert workflow.all_b2_complete() is False

    # After 1 B2 completion
    workflow.b2_complete_times["test-001-B2-0"] = 200.0
    assert workflow.all_b2_complete() is False

    # After 2 B2 completions (all done)
    workflow.b2_complete_times["test-001-B2-1"] = 201.0
    assert workflow.all_b2_complete() is True


def test_workflow_is_complete():
    """Test workflow completion check."""
    workflow = DeepResearchWorkflowData(
        workflow_id="test-001",
        fanout_count=3
    )

    # Not complete initially
    assert workflow.is_complete() is False

    # Even with all B1 and B2 done, still not complete without Merge
    workflow.b1_complete_times["test-001-B1-0"] = 100.0
    workflow.b1_complete_times["test-001-B1-1"] = 101.0
    workflow.b1_complete_times["test-001-B1-2"] = 102.0
    workflow.b2_complete_times["test-001-B2-0"] = 200.0
    workflow.b2_complete_times["test-001-B2-1"] = 201.0
    workflow.b2_complete_times["test-001-B2-2"] = 202.0
    assert workflow.is_complete() is False

    # Complete after Merge
    workflow.merge_complete_time = 300.0
    assert workflow.is_complete() is True


def test_workflow_get_b2_for_b1():
    """Test B1-to-B2 mapping."""
    workflow = DeepResearchWorkflowData(
        workflow_id="test-001",
        fanout_count=3
    )

    # Pre-populate task IDs (simulates A receiver behavior)
    workflow.b1_task_ids = ["test-001-B1-0", "test-001-B1-1", "test-001-B1-2"]
    workflow.b2_task_ids = ["test-001-B2-0", "test-001-B2-1", "test-001-B2-2"]

    # Test 1:1 mapping
    assert workflow.get_b2_for_b1("test-001-B1-0") == "test-001-B2-0"
    assert workflow.get_b2_for_b1("test-001-B1-1") == "test-001-B2-1"
    assert workflow.get_b2_for_b1("test-001-B1-2") == "test-001-B2-2"

    # Test non-existent B1
    assert workflow.get_b2_for_b1("test-001-B1-9") is None


# ============================================================================
# Submitter Tests
# ============================================================================

def test_a_submitter_initialization():
    """Test A submitter initialization."""
    config = DeepResearchConfig(num_workflows=10, fanout_count=3)
    workflow_states = {}
    state_lock = threading.Lock()

    submitter = ATaskSubmitter(
        name="TestA",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url="http://localhost:8100",
        rate_limiter=Mock()
    )

    # Check workflow generation
    assert len(submitter.workflows) == 10
    assert submitter.workflows[0].workflow_id == "workflow-0000"
    assert submitter.workflows[0].fanout_count == 3

    # Check workflow states populated
    assert len(workflow_states) == 10


def test_a_submitter_payload_preparation():
    """Test A task payload preparation."""
    config = DeepResearchConfig()
    workflow_states = {}
    state_lock = threading.Lock()

    submitter = ATaskSubmitter(
        name="TestA",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url="http://localhost:8100",
        rate_limiter=Mock()
    )

    workflow = submitter.workflows[0]
    payload = submitter._prepare_task_payload(workflow)

    assert payload["task_id"] == "workflow-0000-A"
    assert payload["model_id"] == config.model_a_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0000"
    assert payload["metadata"]["fanout_count"] == config.fanout_count
    assert payload["metadata"]["task_type"] == "A"


def test_b1_submitter_payload_preparation():
    """Test B1 task payload preparation."""
    config = DeepResearchConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    a_result_queue = Queue()

    submitter = B1TaskSubmitter(
        name="TestB1",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a_result_queue=a_result_queue,
        scheduler_url="http://localhost:8101",
        rate_limiter=Mock()
    )

    task_data = ("workflow-0001", "a_result", 2)
    payload = submitter._prepare_task_payload(task_data)

    assert payload["task_id"] == "workflow-0001-B1-2"
    assert payload["model_id"] == config.model_b_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0001"
    assert payload["metadata"]["a_result"] == "a_result"
    assert payload["metadata"]["b1_index"] == 2
    assert payload["metadata"]["task_type"] == "B1"


def test_b2_submitter_payload_preparation():
    """Test B2 task payload preparation."""
    config = DeepResearchConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    b1_result_queue = Queue()

    submitter = B2TaskSubmitter(
        name="TestB2",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_result_queue=b1_result_queue,
        scheduler_url="http://localhost:8101",
        rate_limiter=Mock()
    )

    task_data = ("workflow-0001", "b1_result", 1)
    payload = submitter._prepare_task_payload(task_data)

    assert payload["task_id"] == "workflow-0001-B2-1"
    assert payload["model_id"] == config.model_b_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0001"
    assert payload["metadata"]["b1_result"] == "b1_result"
    assert payload["metadata"]["b1_index"] == 1
    assert payload["metadata"]["task_type"] == "B2"


def test_merge_submitter_payload_preparation():
    """Test Merge task payload preparation."""
    config = DeepResearchConfig(fanout_count=3)
    workflow_states = {
        "workflow-0001": DeepResearchWorkflowData(
            workflow_id="workflow-0001",
            fanout_count=3
        )
    }
    state_lock = threading.Lock()
    merge_trigger_queue = Queue()

    submitter = MergeTaskSubmitter(
        name="TestMerge",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_trigger_queue=merge_trigger_queue,
        scheduler_url="http://localhost:8100",
        rate_limiter=Mock()
    )

    payload = submitter._prepare_task_payload("workflow-0001")

    assert payload["task_id"] == "workflow-0001-Merge"
    assert payload["model_id"] == config.model_merge_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0001"
    assert len(payload["metadata"]["b2_results"]) == 3
    assert payload["metadata"]["task_type"] == "Merge"


# ============================================================================
# Receiver Tests
# ============================================================================

def test_a_receiver_subscription_payload():
    """Test A receiver WebSocket subscription payload."""
    config = DeepResearchConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    a_result_queue = Queue()

    receiver = ATaskReceiver(
        name="TestAReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_submitter=Mock(),
        a_result_queue=a_result_queue,
        scheduler_url="http://localhost:8100",
        model_id="llm_service_small_model"
    )

    payload = receiver._get_subscription_payload()

    assert payload["type"] == "subscribe"
    assert payload["model_id"] == "llm_service_small_model"


@pytest.mark.asyncio
async def test_a_receiver_process_result_fanout():
    """Test A receiver result processing and fanout logic."""
    config = DeepResearchConfig(fanout_count=3)
    workflow_states = {
        "workflow-0001": DeepResearchWorkflowData(
            workflow_id="workflow-0001",
            fanout_count=3
        )
    }
    state_lock = threading.Lock()
    a_result_queue = Queue()

    receiver = ATaskReceiver(
        name="TestAReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b1_submitter=Mock(),
        a_result_queue=a_result_queue,
        scheduler_url="http://localhost:8100",
        model_id="llm_service_small_model"
    )

    # Simulate A result
    data = {
        "metadata": {
            "workflow_id": "workflow-0001",
            "fanout_count": 3
        },
        "result": {"output": "a_result"}
    }

    await receiver._process_result(data)

    # Check workflow state updated
    assert workflow_states["workflow-0001"].a_result == "a_result"
    assert len(workflow_states["workflow-0001"].b1_task_ids) == 3
    assert len(workflow_states["workflow-0001"].b2_task_ids) == 3

    # Check 3 B1 tasks queued
    assert a_result_queue.qsize() == 3
    for i in range(3):
        workflow_id, a_result, b1_index = a_result_queue.get()
        assert workflow_id == "workflow-0001"
        assert a_result == "a_result"
        assert b1_index == i


@pytest.mark.asyncio
async def test_b2_receiver_synchronization():
    """Test B2 receiver synchronization logic."""
    config = DeepResearchConfig(fanout_count=3)
    workflow_states = {
        "workflow-0001": DeepResearchWorkflowData(
            workflow_id="workflow-0001",
            fanout_count=3
        )
    }
    state_lock = threading.Lock()
    merge_trigger_queue = Queue()

    # Mock merge submitter
    merge_submitter = Mock()

    receiver = B2TaskReceiver(
        name="TestB2Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        merge_submitter=merge_submitter,
        merge_trigger_queue=merge_trigger_queue,
        scheduler_url="http://localhost:8101",
        model_id="llm_service_small_model"
    )

    # Simulate B2-0 completion (should NOT trigger Merge)
    data = {
        "task_id": "workflow-0001-B2-0",
        "metadata": {"workflow_id": "workflow-0001"}
    }

    await receiver._process_result(data)

    assert len(workflow_states["workflow-0001"].b2_complete_times) == 1
    assert merge_trigger_queue.empty()

    # Simulate B2-1 completion (should NOT trigger Merge)
    data = {
        "task_id": "workflow-0001-B2-1",
        "metadata": {"workflow_id": "workflow-0001"}
    }

    await receiver._process_result(data)

    assert len(workflow_states["workflow-0001"].b2_complete_times) == 2
    assert merge_trigger_queue.empty()

    # Simulate B2-2 completion (should trigger Merge)
    data = {
        "task_id": "workflow-0001-B2-2",
        "metadata": {"workflow_id": "workflow-0001"}
    }

    await receiver._process_result(data)

    assert len(workflow_states["workflow-0001"].b2_complete_times) == 3
    assert workflow_states["workflow-0001"].all_b2_complete()
    # Check Merge was triggered
    assert merge_trigger_queue.qsize() == 1
    assert merge_trigger_queue.get() == "workflow-0001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
