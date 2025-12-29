"""
Unit tests for src/models.py
"""

import pytest
from datetime import UTC, datetime
from pydantic import ValidationError
from src.models import (
    Task, TaskStatus, InstanceStatus, ModelInfo, ModelRegistryEntry,
    RestartStatus, RestartOperation
)


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
        expected_statuses = ["queued", "running", "completed", "failed", "fetched"]
        actual_statuses = [status.value for status in TaskStatus]
        assert set(actual_statuses) == set(expected_statuses)

    def test_fetched_status(self):
        """Test that FETCHED status exists for work-stealing support"""
        assert TaskStatus.FETCHED == "fetched"


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


@pytest.mark.unit
class TestRestartStatus:
    """Test suite for RestartStatus enum"""

    def test_restart_status_values(self):
        """Test that RestartStatus has all required values"""
        assert RestartStatus.PENDING == "pending"
        assert RestartStatus.DRAINING == "draining"
        assert RestartStatus.EXTRACTING_TASKS == "extracting_tasks"
        assert RestartStatus.WAITING_RUNNING_TASK == "waiting_running_task"
        assert RestartStatus.STOPPING_MODEL == "stopping_model"
        assert RestartStatus.DEREGISTERING == "deregistering"
        assert RestartStatus.STARTING_MODEL == "starting_model"
        assert RestartStatus.REGISTERING == "registering"
        assert RestartStatus.COMPLETED == "completed"
        assert RestartStatus.FAILED == "failed"

    def test_restart_status_membership(self):
        """Test that all expected statuses are members of the enum"""
        expected_statuses = [
            "pending", "draining", "extracting_tasks", "waiting_running_task", "stopping_model",
            "deregistering", "starting_model", "registering", "completed", "failed"
        ]
        actual_statuses = [status.value for status in RestartStatus]
        assert set(actual_statuses) == set(expected_statuses)


@pytest.mark.unit
class TestRestartOperation:
    """Test suite for RestartOperation model"""

    def test_restart_operation_creation(self):
        """Test creating a RestartOperation with required fields"""
        operation = RestartOperation(
            operation_id="test-op-123",
            new_model_id="new-model"
        )

        assert operation.operation_id == "test-op-123"
        assert operation.new_model_id == "new-model"
        assert operation.status == RestartStatus.PENDING
        assert operation.old_model_id is None
        assert operation.new_parameters == {}
        assert operation.new_scheduler_url is None
        assert operation.pending_tasks_at_start == 0
        assert operation.pending_tasks_completed == 0
        assert operation.error is None

    def test_restart_operation_with_all_fields(self):
        """Test creating a RestartOperation with all fields"""
        operation = RestartOperation(
            operation_id="test-op-456",
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={"key": "value"},
            new_scheduler_url="http://scheduler:8000",
            pending_tasks_at_start=5,
            pending_tasks_completed=3
        )

        assert operation.operation_id == "test-op-456"
        assert operation.old_model_id == "old-model"
        assert operation.new_model_id == "new-model"
        assert operation.new_parameters == {"key": "value"}
        assert operation.new_scheduler_url == "http://scheduler:8000"
        assert operation.pending_tasks_at_start == 5
        assert operation.pending_tasks_completed == 3

    def test_restart_operation_update_status(self):
        """Test updating operation status"""
        operation = RestartOperation(
            operation_id="test-op-789",
            new_model_id="new-model"
        )

        # Update to draining
        operation.update_status(RestartStatus.DRAINING)
        assert operation.status == RestartStatus.DRAINING
        assert operation.error is None
        assert operation.completed_at is None

        # Update to completed
        operation.update_status(RestartStatus.COMPLETED)
        assert operation.status == RestartStatus.COMPLETED
        assert operation.completed_at is not None

    def test_restart_operation_update_status_with_error(self):
        """Test updating operation status with error"""
        operation = RestartOperation(
            operation_id="test-op-error",
            new_model_id="new-model"
        )

        # Update to failed with error
        error_msg = "Model not found in registry"
        operation.update_status(RestartStatus.FAILED, error=error_msg)

        assert operation.status == RestartStatus.FAILED
        assert operation.error == error_msg
        assert operation.completed_at is not None

    def test_restart_operation_timestamps(self):
        """Test that timestamps are set correctly"""
        operation = RestartOperation(
            operation_id="test-op-time",
            new_model_id="new-model"
        )

        # Check initiated_at is set
        assert operation.initiated_at is not None
        assert "Z" in operation.initiated_at  # ISO format with Z

        # completed_at should be None initially
        assert operation.completed_at is None

        # Update to completed
        operation.update_status(RestartStatus.COMPLETED)
        assert operation.completed_at is not None
        assert "Z" in operation.completed_at

    def test_restart_operation_validation(self):
        """Test RestartOperation field validation"""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            RestartOperation(
                operation_id="test-op"
                # Missing new_model_id
            )
        assert "new_model_id" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RestartOperation(
                # Missing operation_id
                new_model_id="new-model"
            )
        assert "operation_id" in str(exc_info.value)

    def test_restart_operation_json_serialization(self):
        """Test that RestartOperation can be serialized to JSON"""
        operation = RestartOperation(
            operation_id="test-op-json",
            old_model_id="old-model",
            new_model_id="new-model",
            new_parameters={"temp": 0.7},
            new_scheduler_url="http://scheduler:8000"
        )

        # Convert to dict
        op_dict = operation.model_dump()

        assert op_dict["operation_id"] == "test-op-json"
        assert op_dict["old_model_id"] == "old-model"
        assert op_dict["new_model_id"] == "new-model"
        assert op_dict["new_parameters"] == {"temp": 0.7}
        assert op_dict["new_scheduler_url"] == "http://scheduler:8000"
        assert op_dict["status"] == "pending"


@pytest.mark.unit
class TestRuntimeStandbyConfig:
    """Test suite for RuntimeStandbyConfig dataclass"""

    def test_runtime_standby_config_defaults(self):
        """Test RuntimeStandbyConfig with default values"""
        from src.models import RuntimeStandbyConfig

        config = RuntimeStandbyConfig()

        assert config.enabled is True  # Default is True
        assert config.port_offset == 1000
        assert config.max_retries == 3
        assert config.initial_delay == 5.0
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 2.0
        assert config.restart_delay == 30
        assert config.health_check_timeout == 600
        assert config.traditional_restart_delay == 30  # Default is 30

    def test_runtime_standby_config_custom_values(self):
        """Test RuntimeStandbyConfig with custom values"""
        from src.models import RuntimeStandbyConfig

        config = RuntimeStandbyConfig(
            enabled=True,
            port_offset=2000,
            max_retries=5,
            initial_delay=10.0,
            max_delay=60.0,
            backoff_multiplier=3.0,
            restart_delay=45,
            health_check_timeout=900,
            traditional_restart_delay=120
        )

        assert config.enabled is True
        assert config.port_offset == 2000
        assert config.max_retries == 5
        assert config.initial_delay == 10.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 3.0
        assert config.restart_delay == 45
        assert config.health_check_timeout == 900
        assert config.traditional_restart_delay == 120

    def test_from_config_and_overrides_no_overrides(self, mock_config):
        """Test from_config_and_overrides with no overrides"""
        from src.models import RuntimeStandbyConfig

        result = RuntimeStandbyConfig.from_config_and_overrides(mock_config)

        assert result.enabled == mock_config.standby_enabled
        assert result.port_offset == mock_config.standby_port_offset
        assert result.max_retries == mock_config.hot_standby_max_retries

    def test_from_config_and_overrides_standby_enabled(self, mock_config):
        """Test from_config_and_overrides with standby_enabled override"""
        from src.models import RuntimeStandbyConfig

        result = RuntimeStandbyConfig.from_config_and_overrides(
            mock_config,
            standby_enabled=True
        )

        assert result.enabled is True

    def test_from_config_and_overrides_all_overrides(self, mock_config):
        """Test from_config_and_overrides with all overrides"""
        from src.models import RuntimeStandbyConfig

        overrides = {
            "port_offset": 5000,
            "max_retries": 10,
            "initial_delay": 15.0,
            "max_delay": 120.0,
            "backoff_multiplier": 4.0,
            "restart_delay": 90,
            "health_check_timeout": 1800,
            "traditional_restart_delay": 180
        }

        result = RuntimeStandbyConfig.from_config_and_overrides(
            mock_config,
            standby_enabled=True,
            overrides=overrides
        )

        assert result.enabled is True
        assert result.port_offset == 5000
        assert result.max_retries == 10
        assert result.initial_delay == 15.0
        assert result.max_delay == 120.0
        assert result.backoff_multiplier == 4.0
        assert result.restart_delay == 90
        assert result.health_check_timeout == 1800
        assert result.traditional_restart_delay == 180
