"""
Unit tests for Pydantic data models.

Tests all model validation, serialization, and enum functionality.
"""

import pytest
from pydantic import ValidationError

from src.model import (
    # Base Models
    Task,
    Instance,
    InstanceQueueBase,
    InstanceQueueProbabilistic,
    # Response Models
    SuccessResponse,
    ErrorResponse,
    # Instance Models
    InstanceRegisterRequest,
    InstanceRegisterResponse,
    InstanceRemoveRequest,
    InstanceRemoveResponse,
    InstanceListResponse,
    InstanceStats,
    InstanceInfoResponse,
    # Task Models
    TaskStatus,
    TaskSubmitRequest,
    TaskInfo,
    TaskSubmitResponse,
    TaskSummary,
    TaskListResponse,
    TaskTimestamps,
    TaskDetailInfo,
    TaskDetailResponse,
    # Health Models
    HealthStats,
    HealthResponse,
    HealthErrorResponse,
    # WebSocket Models
    WSMessageType,
    WSSubscribeMessage,
    WSUnsubscribeMessage,
    WSAckMessage,
    WSTaskResultMessage,
    WSErrorMessage,
)


# ============================================================================
# Base Models Tests
# ============================================================================

class TestTask:
    """Tests for Task model."""

    def test_valid_task(self):
        """Test creating a valid task."""
        task = Task(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"priority": "high"}
        )
        assert task.task_id == "task-1"
        assert task.model_id == "model-1"
        assert task.task_input == {"prompt": "test"}
        assert task.metadata == {"priority": "high"}

    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            Task(task_id="task-1", model_id="model-1")

    def test_empty_dicts(self):
        """Test task with empty input and metadata."""
        task = Task(
            task_id="task-1",
            model_id="model-1",
            task_input={},
            metadata={}
        )
        assert task.task_input == {}
        assert task.metadata == {}

    def test_serialization(self):
        """Test task serialization."""
        task = Task(
            task_id="task-1",
            model_id="model-1",
            task_input={"x": 1},
            metadata={"y": 2}
        )
        data = task.model_dump()
        assert data["task_id"] == "task-1"
        assert data["task_input"] == {"x": 1}


class TestInstance:
    """Tests for Instance model."""

    def test_valid_instance(self):
        """Test creating a valid instance."""
        instance = Instance(instance_id="inst-1", model_id="model-1", endpoint="http://localhost:8000", platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"})
        assert instance.instance_id == "inst-1"
        assert instance.endpoint == "http://localhost:8000"

    def test_missing_required_field(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError):
            Instance(instance_id="inst-1", model_id="model-1")

    def test_wrong_type(self):
        """Test wrong field type."""
        with pytest.raises(ValidationError):
            Instance(instance_id=123, model_id="model-1", endpoint="http://test")


class TestInstanceQueueBase:
    """Tests for InstanceQueueBase model."""

    def test_valid_queue_base(self):
        """Test creating valid base queue info."""
        queue = InstanceQueueBase(instance_id="inst-1")
        assert queue.instance_id == "inst-1"

    def test_missing_instance_id(self):
        """Test missing instance_id."""
        with pytest.raises(ValidationError):
            InstanceQueueBase()


class TestInstanceQueueProbabilistic:
    """Tests for InstanceQueueProbabilistic model."""

    def test_valid_probabilistic_queue(self):
        """Test creating valid probabilistic queue info."""
        queue = InstanceQueueProbabilistic(
            instance_id="inst-1",
            quantiles=[0.5, 0.9, 0.95],
            values=[100.0, 200.0, 300.0]
        )
        assert queue.instance_id == "inst-1"
        assert len(queue.quantiles) == 3
        assert len(queue.values) == 3

    def test_empty_lists(self):
        """Test with empty quantiles and values."""
        queue = InstanceQueueProbabilistic(
            instance_id="inst-1",
            quantiles=[],
            values=[]
        )
        assert queue.quantiles == []
        assert queue.values == []

    def test_mismatched_lengths(self):
        """Test that mismatched quantiles/values are accepted (no validation)."""
        # Pydantic doesn't validate list length matching unless explicitly defined
        queue = InstanceQueueProbabilistic(
            instance_id="inst-1",
            quantiles=[0.5, 0.9],
            values=[100.0, 200.0, 300.0]
        )
        assert len(queue.quantiles) == 2
        assert len(queue.values) == 3


# ============================================================================
# Response Models Tests
# ============================================================================

class TestSuccessResponse:
    """Tests for SuccessResponse model."""

    def test_success_with_message(self):
        """Test success response with message."""
        resp = SuccessResponse(success=True, message="Operation completed")
        assert resp.success is True
        assert resp.message == "Operation completed"

    def test_success_without_message(self):
        """Test success response without message."""
        resp = SuccessResponse(success=True)
        assert resp.success is True
        assert resp.message is None

    def test_optional_message_default(self):
        """Test that message defaults to None."""
        resp = SuccessResponse(success=False)
        assert resp.message is None


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response(self):
        """Test error response."""
        resp = ErrorResponse(success=False, error="Something went wrong")
        assert resp.success is False
        assert resp.error == "Something went wrong"

    def test_missing_error_field(self):
        """Test that error field is required."""
        with pytest.raises(ValidationError):
            ErrorResponse(success=False)


# ============================================================================
# Instance Management Models Tests
# ============================================================================

class TestInstanceRegisterRequest:
    """Tests for InstanceRegisterRequest model."""

    def test_valid_register_request(self):
        """Test valid registration request."""
        req = InstanceRegisterRequest(
            instance_id="inst-1",
            model_id="model-1",
            endpoint="http://localhost:8000",
            platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"}
        )
        assert req.instance_id == "inst-1"
        assert req.model_id == "model-1"
        assert req.endpoint == "http://localhost:8000"


class TestInstanceRegisterResponse:
    """Tests for InstanceRegisterResponse model."""

    def test_valid_register_response(self):
        """Test valid registration response."""
        instance = Instance(instance_id="inst-1", model_id="model-1", endpoint="http://localhost:8000", platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"})
        resp = InstanceRegisterResponse(
            success=True,
            message="Registered",
            instance=instance
        )
        assert resp.success is True
        assert resp.instance.instance_id == "inst-1"


class TestInstanceRemoveRequest:
    """Tests for InstanceRemoveRequest model."""

    def test_valid_remove_request(self):
        """Test valid remove request."""
        req = InstanceRemoveRequest(instance_id="inst-1")
        assert req.instance_id == "inst-1"


class TestInstanceRemoveResponse:
    """Tests for InstanceRemoveResponse model."""

    def test_valid_remove_response(self):
        """Test valid remove response."""
        resp = InstanceRemoveResponse(
            success=True,
            message="Removed",
            instance_id="inst-1"
        )
        assert resp.success is True
        assert resp.instance_id == "inst-1"


class TestInstanceListResponse:
    """Tests for InstanceListResponse model."""

    def test_list_response_with_instances(self):
        """Test list response with instances."""
        instances = [
            Instance(instance_id=f"inst-{i}", model_id="model-1", endpoint=f"http://localhost:800{i}")
            for i in range(3)
        ]
        resp = InstanceListResponse(success=True, count=3, instances=instances)
        assert resp.success is True
        assert resp.count == 3
        assert len(resp.instances) == 3

    def test_empty_list_response(self):
        """Test empty list response."""
        resp = InstanceListResponse(success=True, count=0, instances=[])
        assert resp.count == 0
        assert resp.instances == []


class TestInstanceStats:
    """Tests for InstanceStats model."""

    def test_valid_stats(self):
        """Test valid instance stats."""
        stats = InstanceStats(pending_tasks=5, completed_tasks=10, failed_tasks=2)
        assert stats.pending_tasks == 5
        assert stats.completed_tasks == 10
        assert stats.failed_tasks == 2

    def test_zero_stats(self):
        """Test stats with zero values."""
        stats = InstanceStats(pending_tasks=0, completed_tasks=0, failed_tasks=0)
        assert stats.pending_tasks == 0


class TestInstanceInfoResponse:
    """Tests for InstanceInfoResponse model."""

    def test_valid_info_response(self):
        """Test valid info response."""
        instance = Instance(instance_id="inst-1", model_id="model-1", endpoint="http://test", platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"})
        queue_info = InstanceQueueBase(instance_id="inst-1")
        stats = InstanceStats(pending_tasks=1, completed_tasks=2, failed_tasks=0)

        resp = InstanceInfoResponse(
            success=True,
            instance=instance,
            queue_info=queue_info,
            stats=stats
        )
        assert resp.success is True
        assert resp.instance.instance_id == "inst-1"
        assert resp.stats.pending_tasks == 1


# ============================================================================
# Task Models Tests
# ============================================================================

class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_status_values(self):
        """Test all enum values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_enum_membership(self):
        """Test enum membership."""
        assert "pending" in TaskStatus.__members__.values()
        assert TaskStatus.PENDING in TaskStatus

    def test_invalid_status(self):
        """Test that invalid status raises error."""
        with pytest.raises(ValueError):
            TaskStatus("invalid")


class TestTaskSubmitRequest:
    """Tests for TaskSubmitRequest model."""

    def test_valid_submit_request(self):
        """Test valid task submission."""
        req = TaskSubmitRequest(
            task_id="task-1",
            model_id="model-1",
            task_input={"prompt": "test"},
            metadata={"key": "value"}
        )
        assert req.task_id == "task-1"
        assert req.task_input == {"prompt": "test"}


class TestTaskInfo:
    """Tests for TaskInfo model."""

    def test_valid_task_info(self):
        """Test valid task info."""
        info = TaskInfo(
            task_id="task-1",
            status=TaskStatus.PENDING,
            assigned_instance="inst-1",
            submitted_at="2024-01-01T00:00:00"
        )
        assert info.task_id == "task-1"
        assert info.status == TaskStatus.PENDING
        assert info.assigned_instance == "inst-1"


class TestTaskSubmitResponse:
    """Tests for TaskSubmitResponse model."""

    def test_valid_submit_response(self):
        """Test valid submit response."""
        task_info = TaskInfo(
            task_id="task-1",
            status=TaskStatus.PENDING,
            assigned_instance="inst-1",
            submitted_at="2024-01-01T00:00:00"
        )
        resp = TaskSubmitResponse(
            success=True,
            message="Task submitted",
            task=task_info
        )
        assert resp.success is True
        assert resp.task.task_id == "task-1"


class TestTaskSummary:
    """Tests for TaskSummary model."""

    def test_summary_with_completed_at(self):
        """Test task summary with completion time."""
        summary = TaskSummary(
            task_id="task-1",
            model_id="model-1",
            status=TaskStatus.COMPLETED,
            assigned_instance="inst-1",
            submitted_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:01:00"
        )
        assert summary.completed_at == "2024-01-01T00:01:00"

    def test_summary_without_completed_at(self):
        """Test task summary without completion time."""
        summary = TaskSummary(
            task_id="task-1",
            model_id="model-1",
            status=TaskStatus.PENDING,
            assigned_instance="inst-1",
            submitted_at="2024-01-01T00:00:00"
        )
        assert summary.completed_at is None


class TestTaskListResponse:
    """Tests for TaskListResponse model."""

    def test_list_response_with_pagination(self):
        """Test list response with pagination info."""
        tasks = [
            TaskSummary(
                task_id=f"task-{i}",
                model_id="model-1",
                status=TaskStatus.PENDING,
                assigned_instance="inst-1",
                submitted_at="2024-01-01T00:00:00"
            )
            for i in range(10)
        ]
        resp = TaskListResponse(
            success=True,
            count=10,
            total=100,
            offset=0,
            limit=10,
            tasks=tasks
        )
        assert resp.count == 10
        assert resp.total == 100
        assert resp.offset == 0
        assert resp.limit == 10

    def test_empty_task_list(self):
        """Test empty task list."""
        resp = TaskListResponse(
            success=True,
            count=0,
            total=0,
            offset=0,
            limit=10,
            tasks=[]
        )
        assert resp.count == 0
        assert len(resp.tasks) == 0


class TestTaskTimestamps:
    """Tests for TaskTimestamps model."""

    def test_all_timestamps_present(self):
        """Test timestamps with all fields."""
        ts = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00",
            started_at="2024-01-01T00:00:01",
            completed_at="2024-01-01T00:00:02"
        )
        assert ts.submitted_at == "2024-01-01T00:00:00"
        assert ts.started_at == "2024-01-01T00:00:01"
        assert ts.completed_at == "2024-01-01T00:00:02"

    def test_optional_timestamps(self):
        """Test optional timestamp fields."""
        ts = TaskTimestamps(submitted_at="2024-01-01T00:00:00")
        assert ts.submitted_at == "2024-01-01T00:00:00"
        assert ts.started_at is None
        assert ts.completed_at is None


class TestTaskDetailInfo:
    """Tests for TaskDetailInfo model."""

    def test_completed_task_detail(self):
        """Test detailed info for completed task."""
        ts = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00",
            started_at="2024-01-01T00:00:01",
            completed_at="2024-01-01T00:00:02"
        )
        detail = TaskDetailInfo(
            task_id="task-1",
            model_id="model-1",
            status=TaskStatus.COMPLETED,
            assigned_instance="inst-1",
            task_input={"prompt": "test"},
            metadata={"key": "value"},
            result={"output": "result"},
            timestamps=ts,
            execution_time_ms=1000
        )
        assert detail.task_id == "task-1"
        assert detail.result == {"output": "result"}
        assert detail.error is None
        assert detail.execution_time_ms == 1000

    def test_failed_task_detail(self):
        """Test detailed info for failed task."""
        ts = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00",
            started_at="2024-01-01T00:00:01",
            completed_at="2024-01-01T00:00:02"
        )
        detail = TaskDetailInfo(
            task_id="task-1",
            model_id="model-1",
            status=TaskStatus.FAILED,
            assigned_instance="inst-1",
            task_input={"prompt": "test"},
            metadata={},
            error="Execution failed",
            timestamps=ts
        )
        assert detail.error == "Execution failed"
        assert detail.result is None
        assert detail.execution_time_ms is None


class TestTaskDetailResponse:
    """Tests for TaskDetailResponse model."""

    def test_valid_detail_response(self):
        """Test valid detail response."""
        ts = TaskTimestamps(submitted_at="2024-01-01T00:00:00")
        detail = TaskDetailInfo(
            task_id="task-1",
            model_id="model-1",
            status=TaskStatus.PENDING,
            assigned_instance="inst-1",
            task_input={},
            metadata={},
            timestamps=ts
        )
        resp = TaskDetailResponse(success=True, task=detail)
        assert resp.success is True
        assert resp.task.task_id == "task-1"


# ============================================================================
# Health Models Tests
# ============================================================================

class TestHealthStats:
    """Tests for HealthStats model."""

    def test_valid_health_stats(self):
        """Test valid health statistics."""
        stats = HealthStats(
            total_instances=10,
            active_instances=8,
            total_tasks=100,
            pending_tasks=5,
            running_tasks=3,
            completed_tasks=90,
            failed_tasks=2
        )
        assert stats.total_instances == 10
        assert stats.active_instances == 8
        assert stats.total_tasks == 100

    def test_zero_stats(self):
        """Test health stats with zeros."""
        stats = HealthStats(
            total_instances=0,
            active_instances=0,
            total_tasks=0,
            pending_tasks=0,
            running_tasks=0,
            completed_tasks=0,
            failed_tasks=0
        )
        assert stats.total_instances == 0


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_healthy_response(self):
        """Test healthy response."""
        stats = HealthStats(
            total_instances=1, active_instances=1, total_tasks=0,
            pending_tasks=0, running_tasks=0, completed_tasks=0, failed_tasks=0
        )
        resp = HealthResponse(
            success=True,
            status="healthy",
            timestamp="2024-01-01T00:00:00",
            version="0.1.0",
            stats=stats
        )
        assert resp.success is True
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"


class TestHealthErrorResponse:
    """Tests for HealthErrorResponse model."""

    def test_error_health_response(self):
        """Test error health response."""
        resp = HealthErrorResponse(
            success=False,
            status="unhealthy",
            error="Service unavailable",
            timestamp="2024-01-01T00:00:00"
        )
        assert resp.success is False
        assert resp.status == "unhealthy"
        assert resp.error == "Service unavailable"


# ============================================================================
# WebSocket Models Tests
# ============================================================================

class TestWSMessageType:
    """Tests for WSMessageType enum."""

    def test_all_message_types(self):
        """Test all WebSocket message type values."""
        assert WSMessageType.SUBSCRIBE == "subscribe"
        assert WSMessageType.UNSUBSCRIBE == "unsubscribe"
        assert WSMessageType.RESULT == "result"
        assert WSMessageType.ERROR == "error"
        assert WSMessageType.ACK == "ack"

    def test_enum_membership(self):
        """Test enum membership."""
        assert "subscribe" in WSMessageType.__members__.values()

    def test_invalid_type(self):
        """Test invalid message type."""
        with pytest.raises(ValueError):
            WSMessageType("invalid")


class TestWSSubscribeMessage:
    """Tests for WSSubscribeMessage model."""

    def test_valid_subscribe(self):
        """Test valid subscribe message."""
        msg = WSSubscribeMessage(task_ids=["task-1", "task-2"])
        assert msg.type == WSMessageType.SUBSCRIBE
        assert len(msg.task_ids) == 2

    def test_empty_task_ids(self):
        """Test subscribe with empty task list."""
        msg = WSSubscribeMessage(task_ids=[])
        assert msg.task_ids == []

    def test_default_type(self):
        """Test that type defaults to SUBSCRIBE."""
        msg = WSSubscribeMessage(task_ids=["task-1"])
        assert msg.type == WSMessageType.SUBSCRIBE


class TestWSUnsubscribeMessage:
    """Tests for WSUnsubscribeMessage model."""

    def test_valid_unsubscribe(self):
        """Test valid unsubscribe message."""
        msg = WSUnsubscribeMessage(task_ids=["task-1"])
        assert msg.type == WSMessageType.UNSUBSCRIBE
        assert msg.task_ids == ["task-1"]


class TestWSAckMessage:
    """Tests for WSAckMessage model."""

    def test_valid_ack(self):
        """Test valid ACK message."""
        msg = WSAckMessage(message="Subscribed", subscribed_tasks=["task-1", "task-2"])
        assert msg.type == WSMessageType.ACK
        assert msg.message == "Subscribed"
        assert len(msg.subscribed_tasks) == 2

    def test_empty_subscribed_tasks(self):
        """Test ACK with no subscribed tasks."""
        msg = WSAckMessage(message="OK", subscribed_tasks=[])
        assert msg.subscribed_tasks == []


class TestWSTaskResultMessage:
    """Tests for WSTaskResultMessage model."""

    def test_successful_result(self):
        """Test result message for successful task."""
        ts = TaskTimestamps(
            submitted_at="2024-01-01T00:00:00",
            started_at="2024-01-01T00:00:01",
            completed_at="2024-01-01T00:00:02"
        )
        msg = WSTaskResultMessage(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            result={"output": "test"},
            timestamps=ts,
            execution_time_ms=1000
        )
        assert msg.type == WSMessageType.RESULT
        assert msg.task_id == "task-1"
        assert msg.status == TaskStatus.COMPLETED
        assert msg.result == {"output": "test"}
        assert msg.error is None

    def test_failed_result(self):
        """Test result message for failed task."""
        ts = TaskTimestamps(submitted_at="2024-01-01T00:00:00")
        msg = WSTaskResultMessage(
            task_id="task-1",
            status=TaskStatus.FAILED,
            error="Task failed",
            timestamps=ts
        )
        assert msg.status == TaskStatus.FAILED
        assert msg.error == "Task failed"
        assert msg.result is None


class TestWSErrorMessage:
    """Tests for WSErrorMessage model."""

    def test_error_with_task_id(self):
        """Test error message with task ID."""
        msg = WSErrorMessage(error="Task not found", task_id="task-1")
        assert msg.type == WSMessageType.ERROR
        assert msg.error == "Task not found"
        assert msg.task_id == "task-1"

    def test_error_without_task_id(self):
        """Test error message without task ID."""
        msg = WSErrorMessage(error="Connection error")
        assert msg.error == "Connection error"
        assert msg.task_id is None


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for model serialization and deserialization."""

    def test_task_round_trip(self):
        """Test task serialization and deserialization."""
        original = Task(
            task_id="task-1",
            model_id="model-1",
            task_input={"key": "value"},
            metadata={"meta": "data"}
        )
        data = original.model_dump()
        restored = Task.model_validate(data)
        assert restored.task_id == original.task_id
        assert restored.task_input == original.task_input

    def test_instance_round_trip(self):
        """Test instance serialization and deserialization."""
        original = Instance(instance_id="inst-1", model_id="model-1", endpoint="http://test", platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"})
        json_str = original.model_dump_json()
        restored = Instance.model_validate_json(json_str)
        assert restored.instance_id == original.instance_id

    def test_nested_model_serialization(self):
        """Test serialization of nested models."""
        instance = Instance(instance_id="inst-1", model_id="model-1", endpoint="http://test", platform_info={"software_name": "docker", "software_version": "20.10", "hardware_name": "test-hardware"})
        queue = InstanceQueueBase(instance_id="inst-1")
        stats = InstanceStats(pending_tasks=1, completed_tasks=2, failed_tasks=0)

        resp = InstanceInfoResponse(
            success=True,
            instance=instance,
            queue_info=queue,
            stats=stats
        )
        data = resp.model_dump()
        assert data["instance"]["instance_id"] == "inst-1"
        assert data["stats"]["pending_tasks"] == 1

        # Restore from dict
        restored = InstanceInfoResponse.model_validate(data)
        assert restored.instance.instance_id == "inst-1"
