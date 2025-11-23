"""Unit tests for Text2Video workflow components."""

import json
import pytest
import tempfile
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import Mock, MagicMock

from type1_text2video.config import Text2VideoConfig
from type1_text2video.workflow_data import (
    Text2VideoWorkflowData,
    load_captions
)
from type1_text2video.submitters import A1TaskSubmitter, A2TaskSubmitter, BTaskSubmitter
from type1_text2video.receivers import A1TaskReceiver, A2TaskReceiver, BTaskReceiver


# ============================================================================
# Configuration Tests
# ============================================================================

def test_config_defaults():
    """Test configuration default values."""
    config = Text2VideoConfig()

    assert config.qps == 1.0
    assert config.duration == 600
    assert config.num_workflows == 600
    assert config.max_b_loops == 4
    assert config.model_a_id == "llm_service_small_model"
    assert config.model_b_id == "t2vid"


def test_config_from_env(monkeypatch):
    """Test configuration from environment variables."""
    monkeypatch.setenv("QPS", "2.5")
    monkeypatch.setenv("DURATION", "300")
    monkeypatch.setenv("NUM_WORKFLOWS", "100")
    monkeypatch.setenv("MAX_B_LOOPS", "2")

    config = Text2VideoConfig.from_env()

    assert config.qps == 2.5
    assert config.duration == 300
    assert config.num_workflows == 100
    assert config.max_b_loops == 2


# ============================================================================
# Workflow Data Tests
# ============================================================================

def test_workflow_data_initialization():
    """Test workflow data initialization."""
    workflow = Text2VideoWorkflowData(
        workflow_id="test-001",
        caption="A man walking in a park",
        max_b_loops=4
    )

    assert workflow.workflow_id == "test-001"
    assert workflow.caption == "A man walking in a park"
    assert workflow.max_b_loops == 4
    assert workflow.a1_result is None
    assert workflow.a2_result is None
    assert workflow.b_loop_count == 0
    assert len(workflow.b_complete_times) == 0


def test_workflow_should_continue_b_loop():
    """Test B-loop continuation logic."""
    workflow = Text2VideoWorkflowData(
        workflow_id="test-001",
        caption="test",
        max_b_loops=3
    )

    # Initially should continue
    assert workflow.should_continue_b_loop() is True

    # After 2 iterations
    workflow.b_loop_count = 2
    assert workflow.should_continue_b_loop() is True

    # After 3 iterations (max reached)
    workflow.b_loop_count = 3
    assert workflow.should_continue_b_loop() is False


def test_workflow_is_complete():
    """Test workflow completion check."""
    workflow = Text2VideoWorkflowData(
        workflow_id="test-001",
        caption="test",
        max_b_loops=2
    )

    # Not complete initially
    assert workflow.is_complete() is False

    # After 1 B completion
    workflow.b_complete_times.append(100.0)
    assert workflow.is_complete() is False

    # After 2 B completions (max reached)
    workflow.b_complete_times.append(200.0)
    assert workflow.is_complete() is True


def test_load_captions_from_list():
    """Test loading captions from JSON array."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(["caption1", "caption2", "caption3"], f)
        temp_path = f.name

    try:
        captions = load_captions(temp_path)
        assert len(captions) == 3
        assert captions[0] == "caption1"
    finally:
        Path(temp_path).unlink()


def test_load_captions_from_dict():
    """Test loading captions from JSON object with 'captions' key."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"captions": ["caption1", "caption2"]}, f)
        temp_path = f.name

    try:
        captions = load_captions(temp_path)
        assert len(captions) == 2
    finally:
        Path(temp_path).unlink()


def test_load_captions_file_not_found():
    """Test loading captions from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_captions("/nonexistent/path.json")


def test_load_captions_invalid_format():
    """Test loading captions from invalid JSON format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"invalid": "format"}, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid caption file format"):
            load_captions(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_captions_empty():
    """Test loading empty captions file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump([], f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Caption file is empty"):
            load_captions(temp_path)
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Submitter Tests
# ============================================================================

def test_a1_submitter_initialization():
    """Test A1 submitter initialization."""
    config = Text2VideoConfig(num_workflows=10, max_b_loops=2)
    captions = ["caption1", "caption2", "caption3"]
    workflow_states = {}
    state_lock = threading.Lock()

    # Mock rate limiter
    rate_limiter = Mock()

    submitter = A1TaskSubmitter(
        name="TestA1",
        captions=captions,
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url="http://localhost:8100",
        rate_limiter=rate_limiter
    )

    # Check workflow generation
    assert len(submitter.workflows) == 10
    assert submitter.workflows[0].workflow_id == "workflow-0000"
    assert submitter.workflows[0].max_b_loops == 2

    # Check workflow states populated
    assert len(workflow_states) == 10


def test_a1_submitter_payload_preparation():
    """Test A1 task payload preparation."""
    config = Text2VideoConfig()
    workflow_states = {}
    state_lock = threading.Lock()

    submitter = A1TaskSubmitter(
        name="TestA1",
        captions=["test caption"],
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        scheduler_url="http://localhost:8100",
        rate_limiter=Mock()
    )

    workflow = submitter.workflows[0]
    payload = submitter._prepare_task_payload(workflow)

    assert payload["task_id"] == "workflow-0000-A1"
    assert payload["model_id"] == config.model_a_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0000"
    assert payload["metadata"]["caption"] == "test caption"
    assert payload["metadata"]["task_type"] == "A1"


def test_a2_submitter_payload_preparation():
    """Test A2 task payload preparation."""
    config = Text2VideoConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    a1_result_queue = Queue()

    submitter = A2TaskSubmitter(
        name="TestA2",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a1_result_queue=a1_result_queue,
        scheduler_url="http://localhost:8100",
        rate_limiter=Mock()
    )

    task_data = ("workflow-0001", "positive prompt result")
    payload = submitter._prepare_task_payload(task_data)

    assert payload["task_id"] == "workflow-0001-A2"
    assert payload["model_id"] == config.model_a_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0001"
    assert payload["metadata"]["positive_prompt"] == "positive prompt result"
    assert payload["metadata"]["task_type"] == "A2"


def test_b_submitter_payload_preparation():
    """Test B task payload preparation."""
    config = Text2VideoConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    a2_result_queue = Queue()

    submitter = BTaskSubmitter(
        name="TestB",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_result_queue=a2_result_queue,
        scheduler_url="http://localhost:8101",
        rate_limiter=Mock()
    )

    task_data = ("workflow-0001", "negative prompt result", 2)
    payload = submitter._prepare_task_payload(task_data)

    assert payload["task_id"] == "workflow-0001-B2"
    assert payload["model_id"] == config.model_b_id
    assert "sleep_time" in payload["task_input"]
    assert payload["metadata"]["workflow_id"] == "workflow-0001"
    assert payload["metadata"]["negative_prompt"] == "negative prompt result"
    assert payload["metadata"]["task_type"] == "B"
    assert payload["metadata"]["loop_iteration"] == 2


# ============================================================================
# Receiver Tests
# ============================================================================

def test_a1_receiver_subscription_payload():
    """Test A1 receiver WebSocket subscription payload."""
    config = Text2VideoConfig()
    workflow_states = {}
    state_lock = threading.Lock()
    a1_result_queue = Queue()

    receiver = A1TaskReceiver(
        name="TestA1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_submitter=Mock(),
        a1_result_queue=a1_result_queue,
        scheduler_url="http://localhost:8100",
        model_id="llm_service_small_model"
    )

    payload = receiver._get_subscription_payload()

    assert payload["type"] == "subscribe"
    assert payload["model_id"] == "llm_service_small_model"


@pytest.mark.asyncio
async def test_a1_receiver_process_result():
    """Test A1 receiver result processing."""
    config = Text2VideoConfig()
    workflow_states = {
        "workflow-0001": Text2VideoWorkflowData(
            workflow_id="workflow-0001",
            caption="test caption"
        )
    }
    state_lock = threading.Lock()
    a1_result_queue = Queue()

    receiver = A1TaskReceiver(
        name="TestA1Receiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        a2_submitter=Mock(),
        a1_result_queue=a1_result_queue,
        scheduler_url="http://localhost:8100",
        model_id="llm_service_small_model"
    )

    # Simulate A1 result
    data = {
        "metadata": {"workflow_id": "workflow-0001"},
        "result": {"output": "positive prompt result"}
    }

    await receiver._process_result(data)

    # Check workflow state updated
    assert workflow_states["workflow-0001"].a1_result == "positive prompt result"

    # Check A2 task queued
    assert not a1_result_queue.empty()
    workflow_id, a1_result = a1_result_queue.get()
    assert workflow_id == "workflow-0001"
    assert a1_result == "positive prompt result"


@pytest.mark.asyncio
async def test_b_receiver_loop_logic():
    """Test B receiver loop logic."""
    config = Text2VideoConfig(max_b_loops=3)
    workflow_states = {
        "workflow-0001": Text2VideoWorkflowData(
            workflow_id="workflow-0001",
            caption="test",
            max_b_loops=3,
            a2_result="negative prompt",
            b_loop_count=2  # On iteration 2
        )
    }
    state_lock = threading.Lock()

    # Mock B submitter
    b_submitter = Mock()

    receiver = BTaskReceiver(
        name="TestBReceiver",
        config=config,
        workflow_states=workflow_states,
        state_lock=state_lock,
        b_submitter=b_submitter,
        scheduler_url="http://localhost:8101",
        model_id="t2vid"
    )

    # Simulate B2 completion (should trigger B3)
    data = {
        "metadata": {
            "workflow_id": "workflow-0001",
            "loop_iteration": 2
        }
    }

    await receiver._process_result(data)

    # Check B3 was triggered
    b_submitter.add_task.assert_called_once_with(
        "workflow-0001",
        "negative prompt",
        3
    )

    # Check state updated
    assert workflow_states["workflow-0001"].b_loop_count == 3
    assert len(workflow_states["workflow-0001"].b_complete_times) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
