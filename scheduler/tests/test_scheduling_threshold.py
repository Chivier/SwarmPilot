"""
Unit tests for scheduling threshold control feature.

Tests the new scheduling pause/resume functionality based on task load.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_registries():
    """Mock the instance and task registries."""
    with patch('src.api.instance_registry') as mock_instance_reg, \
         patch('src.api.task_registry') as mock_task_reg:
        yield mock_instance_reg, mock_task_reg


def test_get_submitted_task_count(mock_registries):
    """Test get_submitted_task_count() function."""
    from src.api import get_submitted_task_count
    from src.model import TaskStatus

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 10 pending, 5 running tasks
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 10,
        TaskStatus.RUNNING: 5,
    }.get(status, 0)

    result = get_submitted_task_count()

    assert result == 15, "Should return sum of pending and running tasks"
    assert mock_task_reg.get_count_by_status.call_count == 2


def test_should_pause_scheduling_below_threshold(mock_registries):
    """Test should_pause_scheduling() when below threshold."""
    from src.api import should_pause_scheduling
    from src.model import TaskStatus

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 2 instances, 15 tasks (below 10*2=20 threshold)
    mock_instance_reg.get_active_count.return_value = 2
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 10,
        TaskStatus.RUNNING: 5,
    }.get(status, 0)

    result = should_pause_scheduling()

    assert result is False, "Should not pause when below threshold"


def test_should_pause_scheduling_above_threshold_not_busy(mock_registries):
    """Test should_pause_scheduling() when above threshold but not all busy."""
    from src.api import should_pause_scheduling
    from src.model import TaskStatus, Instance, InstanceStats

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 2 instances, 25 tasks (above 10*2=20 threshold)
    mock_instance_reg.get_active_count.return_value = 2
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 15,
        TaskStatus.RUNNING: 10,
    }.get(status, 0)

    # But one instance is not busy (has no tasks)
    mock_instances = [
        Instance(instance_id="i1", model_id="m1", endpoint="http://i1", platform_info={}),
        Instance(instance_id="i2", model_id="m1", endpoint="http://i2", platform_info={}),
    ]
    mock_instance_reg.list_active.return_value = mock_instances
    mock_instance_reg.get_stats.side_effect = lambda iid: {
        "i1": InstanceStats(pending_tasks=5, completed_tasks=0, failed_tasks=0),
        "i2": InstanceStats(pending_tasks=0, completed_tasks=0, failed_tasks=0),  # Not busy!
    }.get(iid)
    mock_task_reg.list_all.return_value = ([], 0)  # No running tasks

    result = should_pause_scheduling()

    assert result is False, "Should not pause when not all instances are busy"


def test_should_resume_scheduling_below_threshold(mock_registries):
    """Test should_resume_scheduling() when below resume threshold."""
    from src.api import should_resume_scheduling
    from src.model import TaskStatus

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 2 instances, 15 tasks (below 8*2=16 threshold)
    mock_instance_reg.get_active_count.return_value = 2
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 10,
        TaskStatus.RUNNING: 5,
    }.get(status, 0)

    result = should_resume_scheduling()

    assert result is True, "Should resume when below resume threshold"


def test_should_resume_scheduling_above_threshold(mock_registries):
    """Test should_resume_scheduling() when above resume threshold."""
    from src.api import should_resume_scheduling
    from src.model import TaskStatus

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 2 instances, 20 tasks (above 8*2=16 threshold)
    mock_instance_reg.get_active_count.return_value = 2
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 12,
        TaskStatus.RUNNING: 8,
    }.get(status, 0)

    result = should_resume_scheduling()

    assert result is False, "Should not resume when above resume threshold"


def test_hysteresis_zone(mock_registries):
    """Test the hysteresis zone between pause and resume thresholds."""
    from src.api import should_pause_scheduling, should_resume_scheduling
    from src.model import TaskStatus, Instance, InstanceStats

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 3 instances, 20 tasks
    # Pause threshold: 10*3 = 30
    # Resume threshold: 8*3 = 24
    # 20 is in the hysteresis zone (< 30 and < 24)
    mock_instance_reg.get_active_count.return_value = 3
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 12,
        TaskStatus.RUNNING: 8,
    }.get(status, 0)

    # All instances busy
    mock_instances = [
        Instance(instance_id=f"i{i}", model_id="m1", endpoint=f"http://i{i}", platform_info={})
        for i in range(3)
    ]
    mock_instance_reg.list_active.return_value = mock_instances
    mock_instance_reg.get_stats.side_effect = lambda iid: InstanceStats(
        pending_tasks=2, completed_tasks=0, failed_tasks=0
    )
    mock_task_reg.list_all.return_value = ([Mock()], 1)  # Has running tasks

    # In hysteresis zone:
    # - Should not trigger pause (below 30)
    # - Should trigger resume (below 24)
    assert should_pause_scheduling() is False, "Should not pause in hysteresis zone"
    assert should_resume_scheduling() is True, "Should resume in hysteresis zone"


@pytest.mark.asyncio
async def test_dispatch_pending_tasks_batch(mock_registries):
    """Test dispatch_pending_tasks_batch() function."""
    from src.api import dispatch_pending_tasks_batch
    from src.model import TaskStatus
    import src.api as api_module

    mock_instance_reg, mock_task_reg = mock_registries

    # Setup: 5 pending tasks
    mock_tasks = [
        Mock(task_id=f"task{i}", status=TaskStatus.PENDING)
        for i in range(5)
    ]
    mock_task_reg.list_all.return_value = (mock_tasks, 5)

    # Mock get_submitted_task_count to return actual values
    mock_instance_reg.get_active_count.return_value = 2
    mock_task_reg.get_count_by_status.side_effect = lambda status: {
        TaskStatus.PENDING: 5,
        TaskStatus.RUNNING: 0,
    }.get(status, 0)
    mock_instance_reg.list_active.return_value = []

    # Mock task_dispatcher
    with patch.object(api_module, 'task_dispatcher') as mock_dispatcher:
        mock_dispatcher.dispatch_task_async = Mock()

        # Mock scheduling state
        api_module.scheduling_paused = False

        await dispatch_pending_tasks_batch()

        # Should have dispatched all 5 tasks
        assert mock_dispatcher.dispatch_task_async.call_count == 5


@pytest.mark.asyncio
async def test_scheduling_status_endpoint():
    """Test GET /scheduling/status endpoint."""
    from src.api import app
    from src.model import TaskStatus

    client = TestClient(app)

    with patch('src.api.instance_registry') as mock_instance_reg, \
         patch('src.api.task_registry') as mock_task_reg:

        # Setup mocks
        mock_instance_reg.get_active_count.return_value = 2
        mock_task_reg.get_count_by_status.side_effect = lambda status: {
            TaskStatus.PENDING: 10,
            TaskStatus.RUNNING: 5,
        }.get(status, 0)
        mock_instance_reg.list_active.return_value = []

        response = client.get("/scheduling/status")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["scheduling_paused"] is False
        assert data["metrics"]["active_instances"] == 2
        assert data["metrics"]["submitted_tasks"] == 15
        assert data["thresholds"]["pause_threshold"] == 20
        assert data["thresholds"]["resume_threshold"] == 16


@pytest.mark.asyncio
async def test_manual_pause_endpoint():
    """Test POST /scheduling/pause endpoint."""
    from src.api import app
    import src.api as api_module

    client = TestClient(app)

    # Reset state
    api_module.scheduling_paused = False

    response = client.post("/scheduling/pause")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["scheduling_paused"] is True
    assert "paused successfully" in data["message"].lower()

    # Verify state changed
    assert api_module.scheduling_paused is True


@pytest.mark.asyncio
async def test_manual_resume_endpoint():
    """Test POST /scheduling/resume endpoint."""
    from src.api import app
    import src.api as api_module

    client = TestClient(app)

    # Set to paused state
    api_module.scheduling_paused = True

    with patch.object(api_module, 'dispatch_pending_tasks_batch', new_callable=AsyncMock):
        response = client.post("/scheduling/resume")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["scheduling_paused"] is False
        assert "resumed successfully" in data["message"].lower()

        # Verify state changed
        assert api_module.scheduling_paused is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
