"""
Unit tests for src/models.py
"""

import pytest
from datetime import UTC, datetime
from pydantic import ValidationError
from src.models import Task, TaskStatus, InstanceStatus, ModelInfo, ModelRegistryEntry


@pytest.mark.unit
class TestTaskStatus:
    """Test suite for TaskStatus enum"""

    def test_task_status_values(self):
        """Test that TaskStatus has all required values"""
        assert TaskStatus.QUEUED == "queued"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_task_status_membership(self):
        """Test that all expected statuses are members of the enum"""
        expected_statuses = ["queued", "running", "completed", "failed"]
        actual_statuses = [status.value for status in TaskStatus]
        assert set(actual_statuses) == set(expected_statuses)


@pytest.mark.unit
class TestInstanceStatus:
    """Test suite for InstanceStatus enum"""

    def test_instance_status_values(self):
        """Test that InstanceStatus has all required values"""
        assert InstanceStatus.IDLE == "idle"
        assert InstanceStatus.RUNNING == "running"
        assert InstanceStatus.BUSY == "busy"
        assert InstanceStatus.ERROR == "error"

    def test_instance_status_membership(self):
        """Test that all expected statuses are members of the enum"""
        expected_statuses = ["idle", "running", "busy", "error"]
        actual_statuses = [status.value for status in InstanceStatus]
        assert set(actual_statuses) == set(expected_statuses)


@pytest.mark.unit
class TestTask:
    """Test suite for Task model"""

    def test_task_creation(self):
        """Test successful Task instantiation"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={"prompt": "Hello, world!"}
        )

        assert task.task_id == "task-123"
        assert task.model_id == "test-model"
        assert task.task_input == {"prompt": "Hello, world!"}
        assert task.status == TaskStatus.QUEUED
        assert task.submitted_at is not None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.result is None
        assert task.error is None

    def test_task_validation_required_fields(self):
        """Test that Task validation fails without required fields"""
        # Missing task_id
        with pytest.raises(ValidationError) as exc_info:
            Task(model_id="test-model", task_input={})
        assert "task_id" in str(exc_info.value)

        # Missing model_id
        with pytest.raises(ValidationError) as exc_info:
            Task(task_id="task-123", task_input={})
        assert "model_id" in str(exc_info.value)

        # Missing task_input
        with pytest.raises(ValidationError) as exc_info:
            Task(task_id="task-123", model_id="test-model")
        assert "task_input" in str(exc_info.value)

    def test_task_default_status(self):
        """Test that Task defaults to QUEUED status"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={}
        )
        assert task.status == TaskStatus.QUEUED

    def test_task_submitted_at_timestamp(self):
        """Test that submitted_at is automatically generated"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={}
        )

        # Check that submitted_at is a valid ISO format string
        assert task.submitted_at is not None
        assert task.submitted_at.endswith("Z")

        # Parse to verify it's a valid datetime
        datetime.fromisoformat(task.submitted_at.replace("Z", "+00:00"))

    def test_task_mark_started(self):
        """Test marking task as started"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={}
        )

        # Initially queued
        assert task.status == TaskStatus.QUEUED
        assert task.started_at is None

        # Mark as started
        task.mark_started()

        # Verify status change
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.started_at.endswith("Z")

        # Verify timestamp is valid
        datetime.fromisoformat(task.started_at.replace("Z", "+00:00"))

    def test_task_mark_completed(self):
        """Test marking task as completed with result"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={}
        )

        result = {"output": "Task completed successfully", "tokens": 42}

        # Mark as completed
        task.mark_completed(result)

        # Verify status change
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.completed_at.endswith("Z")
        assert task.result == result
        assert task.error is None

        # Verify timestamp is valid
        datetime.fromisoformat(task.completed_at.replace("Z", "+00:00"))

    def test_task_mark_failed(self):
        """Test marking task as failed with error"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={}
        )

        error_message = "Model inference failed: timeout"

        # Mark as failed
        task.mark_failed(error_message)

        # Verify status change
        assert task.status == TaskStatus.FAILED
        assert task.completed_at is not None
        assert task.completed_at.endswith("Z")
        assert task.error == error_message
        assert task.result is None

        # Verify timestamp is valid
        datetime.fromisoformat(task.completed_at.replace("Z", "+00:00"))

    def test_task_lifecycle_transitions(self):
        """Test complete task lifecycle: queued -> running -> completed"""
        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input={"prompt": "test"}
        )

        # Initial state
        assert task.status == TaskStatus.QUEUED
        submitted_time = task.submitted_at

        # Start task
        task.mark_started()
        assert task.status == TaskStatus.RUNNING
        assert task.submitted_at == submitted_time  # Should not change
        started_time = task.started_at

        # Complete task
        result = {"output": "result"}
        task.mark_completed(result)
        assert task.status == TaskStatus.COMPLETED
        assert task.submitted_at == submitted_time  # Should not change
        assert task.started_at == started_time  # Should not change
        assert task.completed_at is not None
        assert task.result == result

    def test_task_with_complex_input(self):
        """Test Task with complex nested input data"""
        complex_input = {
            "prompt": "Translate this text",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 100,
                "stop_sequences": ["\n", "END"]
            },
            "metadata": {
                "user_id": "user-123",
                "request_id": "req-456"
            }
        }

        task = Task(
            task_id="task-123",
            model_id="test-model",
            task_input=complex_input
        )

        assert task.task_input == complex_input
        assert task.task_input["parameters"]["temperature"] == 0.7
        assert task.task_input["metadata"]["user_id"] == "user-123"


@pytest.mark.unit
class TestModelInfo:
    """Test suite for ModelInfo model"""

    def test_model_info_creation(self):
        """Test successful ModelInfo instantiation"""
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        model_info = ModelInfo(
            model_id="test-model",
            started_at=started_at
        )

        assert model_info.model_id == "test-model"
        assert model_info.started_at == started_at
        assert model_info.parameters == {}
        assert model_info.container_name is None

    def test_model_info_with_parameters(self):
        """Test ModelInfo with parameters"""
        parameters = {"temperature": 0.8, "max_tokens": 200}
        model_info = ModelInfo(
            model_id="test-model",
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            parameters=parameters
        )

        assert model_info.parameters == parameters

    def test_model_info_with_container_name(self):
        """Test ModelInfo with container name"""
        model_info = ModelInfo(
            model_id="test-model",
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            container_name="model_instance_test-model"
        )

        assert model_info.container_name == "model_instance_test-model"

    def test_model_info_validation_required_fields(self):
        """Test that ModelInfo validation fails without required fields"""
        # Missing model_id
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"))
        assert "model_id" in str(exc_info.value)

        # Missing started_at
        with pytest.raises(ValidationError) as exc_info:
            ModelInfo(model_id="test-model")
        assert "started_at" in str(exc_info.value)


@pytest.mark.unit
class TestModelRegistryEntry:
    """Test suite for ModelRegistryEntry model"""

    def test_model_registry_entry_creation(self):
        """Test successful ModelRegistryEntry instantiation"""
        entry = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={"memory": "2Gi", "cpu": "1"}
        )

        assert entry.model_id == "test-model"
        assert entry.name == "Test Model"
        assert entry.directory == "test_model"
        assert entry.resource_requirements == {"memory": "2Gi", "cpu": "1"}

    def test_model_registry_entry_validation_required_fields(self):
        """Test that ModelRegistryEntry validation fails without required fields"""
        # Missing model_id
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistryEntry(
                name="Test Model",
                directory="test_model",
                resource_requirements={}
            )
        assert "model_id" in str(exc_info.value)

        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistryEntry(
                model_id="test-model",
                directory="test_model",
                resource_requirements={}
            )
        assert "name" in str(exc_info.value)

        # Missing directory
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistryEntry(
                model_id="test-model",
                name="Test Model",
                resource_requirements={}
            )
        assert "directory" in str(exc_info.value)

        # Missing resource_requirements
        with pytest.raises(ValidationError) as exc_info:
            ModelRegistryEntry(
                model_id="test-model",
                name="Test Model",
                directory="test_model"
            )
        assert "resource_requirements" in str(exc_info.value)

    def test_model_registry_entry_with_empty_resources(self):
        """Test ModelRegistryEntry with empty resource requirements"""
        entry = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={}
        )

        assert entry.resource_requirements == {}

    def test_model_registry_entry_json_serialization(self):
        """Test that ModelRegistryEntry can be serialized to JSON"""
        entry = ModelRegistryEntry(
            model_id="test-model",
            name="Test Model",
            directory="test_model",
            resource_requirements={"memory": "2Gi"}
        )

        # Convert to dict (which is JSON-serializable)
        entry_dict = entry.model_dump()

        assert entry_dict["model_id"] == "test-model"
        assert entry_dict["name"] == "Test Model"
        assert entry_dict["directory"] == "test_model"
        assert entry_dict["resource_requirements"] == {"memory": "2Gi"}
