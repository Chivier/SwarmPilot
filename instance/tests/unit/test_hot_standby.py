"""
Unit tests for hot-standby port system in SubprocessManager.

Tests cover:
1. Data structures (PortInfo, DualPortState)
2. Port utilities (port_utils.py)
3. Hot-switch logic in SubprocessManager
4. API integration (/info endpoint standby status)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

from src.models import (
    PortRole, PortState, PortInfo, DualPortState,
    ModelInfo
)
from src.port_utils import (
    PortCheckResult, is_port_available, find_available_port,
    calculate_backoff_delay
)
from src.subprocess_manager import SubprocessManager
from src.config import Config


# =============================================================================
# Phase 1: Data Structure Tests
# =============================================================================

@pytest.mark.unit
class TestPortRole:
    """Test PortRole enum"""

    def test_port_role_values(self):
        """Test PortRole enum has correct values"""
        assert PortRole.PRIMARY.value == "primary"
        assert PortRole.STANDBY.value == "standby"

    def test_port_role_is_string_enum(self):
        """Test PortRole is a string enum"""
        assert isinstance(PortRole.PRIMARY, str)
        assert PortRole.PRIMARY == "primary"


@pytest.mark.unit
class TestPortState:
    """Test PortState enum"""

    def test_port_state_values(self):
        """Test PortState enum has all expected values"""
        assert PortState.UNINITIALIZED.value == "uninitialized"
        assert PortState.STARTING.value == "starting"
        assert PortState.HEALTHY.value == "healthy"
        assert PortState.STOPPING.value == "stopping"
        assert PortState.STOPPED.value == "stopped"
        assert PortState.FAILED.value == "failed"

    def test_port_state_is_string_enum(self):
        """Test PortState is a string enum"""
        assert isinstance(PortState.HEALTHY, str)
        assert PortState.HEALTHY == "healthy"


@pytest.mark.unit
class TestPortInfo:
    """Test PortInfo dataclass"""

    def test_port_info_initialization(self):
        """Test PortInfo initialization with defaults"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)

        assert port_info.port == 9000
        assert port_info.role == PortRole.PRIMARY
        assert port_info.state == PortState.UNINITIALIZED
        assert port_info.process is None
        assert port_info.started_at is None
        assert port_info.last_health_check is None
        assert port_info.health_check_failures == 0
        assert port_info.error_message is None
        assert port_info.startup_attempts == 0
        assert port_info.max_startup_attempts == 3

    def test_port_info_is_ready_healthy(self):
        """Test is_ready returns True when state is HEALTHY"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_info.state = PortState.HEALTHY

        assert port_info.is_ready() is True

    def test_port_info_is_ready_not_healthy(self):
        """Test is_ready returns False for non-HEALTHY states"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)

        for state in [PortState.UNINITIALIZED, PortState.STARTING,
                      PortState.STOPPING, PortState.STOPPED, PortState.FAILED]:
            port_info.state = state
            assert port_info.is_ready() is False, f"Expected False for state {state}"

    def test_port_info_can_start_valid_states(self):
        """Test can_start returns True for valid starting states"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)

        for state in [PortState.UNINITIALIZED, PortState.STOPPED, PortState.FAILED]:
            port_info.state = state
            assert port_info.can_start() is True, f"Expected True for state {state}"

    def test_port_info_can_start_invalid_states(self):
        """Test can_start returns False for invalid starting states"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)

        for state in [PortState.STARTING, PortState.HEALTHY, PortState.STOPPING]:
            port_info.state = state
            assert port_info.can_start() is False, f"Expected False for state {state}"

    def test_port_info_reset_for_restart(self):
        """Test reset_for_restart clears state and increments attempts"""
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_info.state = PortState.FAILED
        port_info.process = Mock()
        port_info.started_at = "2024-01-01T00:00:00Z"
        port_info.last_health_check = "2024-01-01T00:01:00Z"
        port_info.health_check_failures = 5
        port_info.error_message = "Some error"
        port_info.startup_attempts = 1

        port_info.reset_for_restart()

        assert port_info.state == PortState.UNINITIALIZED
        assert port_info.process is None
        assert port_info.started_at is None
        assert port_info.last_health_check is None
        assert port_info.health_check_failures == 0
        assert port_info.error_message is None
        assert port_info.startup_attempts == 2  # Incremented


@pytest.mark.unit
class TestDualPortState:
    """Test DualPortState dataclass"""

    def test_dual_port_state_initialization(self):
        """Test DualPortState initialization"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)

        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        assert dual_state.port_a == port_a
        assert dual_state.port_b == port_b
        assert dual_state._primary_port_name == "port_a"

    def test_dual_port_state_primary_property(self):
        """Test primary property returns correct port"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        assert dual_state.primary == port_a
        assert dual_state.primary.port == 9000

    def test_dual_port_state_standby_property(self):
        """Test standby property returns correct port"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        assert dual_state.standby == port_b
        assert dual_state.standby.port == 10000

    def test_dual_port_state_active_port(self):
        """Test active_port returns primary's port"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        assert dual_state.active_port == 9000

    def test_dual_port_state_swap_roles(self):
        """Test swap_roles exchanges PRIMARY and STANDBY"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        # Before swap
        assert dual_state.primary.port == 9000
        assert dual_state.standby.port == 10000

        dual_state.swap_roles()

        # After swap - primary is now port_b
        assert dual_state.primary.port == 10000
        assert dual_state.standby.port == 9000
        assert dual_state.active_port == 10000
        assert port_a.role == PortRole.STANDBY
        assert port_b.role == PortRole.PRIMARY

    def test_dual_port_state_swap_roles_twice(self):
        """Test swapping roles twice returns to original state"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        dual_state.swap_roles()
        dual_state.swap_roles()

        assert dual_state.primary.port == 9000
        assert dual_state.standby.port == 10000

    def test_dual_port_state_get_port_by_role(self):
        """Test get_port_by_role returns correct port"""
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        dual_state = DualPortState(port_a=port_a, port_b=port_b)

        assert dual_state.get_port_by_role(PortRole.PRIMARY).port == 9000
        assert dual_state.get_port_by_role(PortRole.STANDBY).port == 10000


# =============================================================================
# Phase 1: Port Utilities Tests
# =============================================================================

@pytest.mark.unit
class TestPortCheckResult:
    """Test PortCheckResult dataclass"""

    def test_port_check_result_available(self):
        """Test PortCheckResult for available port"""
        result = PortCheckResult(port=9000, available=True)

        assert result.port == 9000
        assert result.available is True
        assert result.error is None

    def test_port_check_result_unavailable(self):
        """Test PortCheckResult for unavailable port"""
        result = PortCheckResult(port=9000, available=False, error="Address in use")

        assert result.port == 9000
        assert result.available is False
        assert result.error == "Address in use"


@pytest.mark.unit
class TestIsPortAvailable:
    """Test is_port_available function"""

    def test_is_port_available_invalid_port_low(self):
        """Test is_port_available rejects port < 1"""
        result = is_port_available(0)

        assert result.available is False
        assert "out of valid range" in result.error

    def test_is_port_available_invalid_port_high(self):
        """Test is_port_available rejects port > 65535"""
        result = is_port_available(65536)

        assert result.available is False
        assert "out of valid range" in result.error

    def test_is_port_available_valid_port(self):
        """Test is_port_available with a high unlikely-to-be-used port"""
        # Use a high port that's unlikely to be in use
        result = is_port_available(59999, host="127.0.0.1")

        # Should succeed or fail with OSError, not validation error
        assert result.port == 59999
        if not result.available:
            assert result.error is not None
            assert "out of valid range" not in result.error


@pytest.mark.unit
class TestCalculateBackoffDelay:
    """Test calculate_backoff_delay function"""

    def test_backoff_delay_first_attempt(self):
        """Test backoff delay for first attempt (attempt=0)"""
        delay = calculate_backoff_delay(
            attempt=0, initial_delay=5.0, max_delay=30.0, multiplier=2.0
        )

        assert delay == 5.0

    def test_backoff_delay_second_attempt(self):
        """Test backoff delay for second attempt (attempt=1)"""
        delay = calculate_backoff_delay(
            attempt=1, initial_delay=5.0, max_delay=30.0, multiplier=2.0
        )

        assert delay == 10.0

    def test_backoff_delay_third_attempt(self):
        """Test backoff delay for third attempt (attempt=2)"""
        delay = calculate_backoff_delay(
            attempt=2, initial_delay=5.0, max_delay=30.0, multiplier=2.0
        )

        assert delay == 20.0

    def test_backoff_delay_capped_at_max(self):
        """Test backoff delay is capped at max_delay"""
        delay = calculate_backoff_delay(
            attempt=10, initial_delay=5.0, max_delay=30.0, multiplier=2.0
        )

        assert delay == 30.0  # Should not exceed max_delay


@pytest.mark.unit
class TestFindAvailablePort:
    """Test find_available_port function"""

    def test_find_available_port_returns_port(self):
        """Test find_available_port returns a port number"""
        # Use a high port range that's less likely to be occupied
        port = find_available_port(59000, max_attempts=100)

        # Should find an available port or return None
        if port is not None:
            assert 59000 <= port < 59100

    def test_find_available_port_exceeds_max_port(self):
        """Test find_available_port handles port exceeding 65535"""
        port = find_available_port(65530, max_attempts=10)

        # Should stop when port exceeds valid range
        assert port is None or port <= 65535


# =============================================================================
# Phase 2 & 3: SubprocessManager Hot-Standby Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManagerHotStandbyInit:
    """Test SubprocessManager initialization for hot-standby"""

    async def test_init_hot_standby_attributes(self):
        """Test SubprocessManager has hot-standby attributes"""
        manager = SubprocessManager()

        assert manager._dual_port_state is None
        assert manager._state_lock is not None
        assert manager._inference_lock is not None
        assert manager._standby_startup_task is None
        assert manager._standby_restart_task is None
        assert manager._model_dir is None
        assert manager._env_vars is None

    async def test_active_port_without_dual_state(self, mock_config):
        """Test active_port returns config.model_port when dual state not initialized"""
        manager = SubprocessManager()

        with patch("src.subprocess_manager.config", mock_config):
            assert manager.active_port == mock_config.model_port

    async def test_active_port_with_dual_state(self):
        """Test active_port returns dual_port_state.active_port when initialized"""
        manager = SubprocessManager()

        # Manually set up dual port state
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        assert manager.active_port == 9000

    async def test_is_standby_ready_no_dual_state(self):
        """Test is_standby_ready returns False when dual state not initialized"""
        manager = SubprocessManager()

        assert manager.is_standby_ready() is False

    async def test_is_standby_ready_standby_not_healthy(self):
        """Test is_standby_ready returns False when standby not healthy"""
        manager = SubprocessManager()

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STARTING)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        assert manager.is_standby_ready() is False

    async def test_is_standby_ready_standby_healthy(self):
        """Test is_standby_ready returns True when standby is healthy"""
        manager = SubprocessManager()

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        assert manager.is_standby_ready() is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManagerDualPortInit:
    """Test _initialize_dual_port_state method"""

    async def test_initialize_dual_port_state(self, mock_config):
        """Test _initialize_dual_port_state creates correct structure"""
        manager = SubprocessManager()

        # Mock config with standby_port_offset
        mock_config.standby_port_offset = 1000
        mock_config.hot_standby_max_retries = 3

        with patch("src.subprocess_manager.config", mock_config):
            dual_state = manager._initialize_dual_port_state()

        assert dual_state is not None
        assert dual_state.port_a.port == mock_config.model_port
        assert dual_state.port_a.role == PortRole.PRIMARY
        assert dual_state.port_b.port == mock_config.model_port + 1000
        assert dual_state.port_b.role == PortRole.STANDBY


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManagerHotSwitch:
    """Test hot-switch functionality"""

    async def test_perform_hot_switch_no_dual_state(self):
        """Test _perform_hot_switch fails without dual state"""
        manager = SubprocessManager()

        with pytest.raises(RuntimeError) as exc_info:
            await manager._perform_hot_switch()

        assert "dual port state not initialized" in str(exc_info.value)

    async def test_perform_hot_switch_standby_not_ready(self):
        """Test _perform_hot_switch fails when standby not healthy"""
        manager = SubprocessManager()

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STARTING)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        with pytest.raises(RuntimeError) as exc_info:
            await manager._perform_hot_switch()

        assert "standby is not healthy" in str(exc_info.value)

    async def test_perform_hot_switch_success(self, mock_config):
        """Test successful hot-switch swaps ports"""
        manager = SubprocessManager()

        # Set up dual port state with healthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        # Set up current model for background restart
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}

        # Mock _restart_standby_background to prevent actual restart
        with patch.object(manager, "_restart_standby_background", new_callable=AsyncMock):
            with patch("src.subprocess_manager.config", mock_config):
                await manager._perform_hot_switch()

        # Verify ports were swapped
        assert manager._dual_port_state.primary.port == 10000
        assert manager._dual_port_state.standby.port == 9000
        assert manager.active_port == 10000

    async def test_restart_model_uses_hot_switch_when_available(self, mock_config):
        """Test restart_model uses hot-switch when standby is ready"""
        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}

        # Set up dual port state with healthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        # Mock _restart_standby_background
        with patch.object(manager, "_restart_standby_background", new_callable=AsyncMock) as mock_restart:
            with patch("src.subprocess_manager.config", mock_config):
                result = await manager.restart_model()

        assert result == "test-model"
        # Verify hot-switch occurred (ports swapped)
        assert manager._dual_port_state.primary.port == 10000

    async def test_restart_model_falls_back_when_standby_not_ready(self, mock_config, mock_model_registry, temp_model_directory):
        """Test restart_model falls back to traditional restart when standby not ready"""
        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Set up dual port state with NOT healthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.FAILED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        # Mock stop_model and start_model
        with patch.object(manager, "stop_model", new_callable=AsyncMock) as mock_stop:
            with patch.object(manager, "start_model", new_callable=AsyncMock) as mock_start:
                mock_stop.return_value = "test-model"
                mock_start.return_value = ModelInfo(
                    model_id="test-model",
                    started_at="2024-01-01T00:00:00Z",
                    parameters={}
                )
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await manager.restart_model()

        assert result == "test-model"
        mock_stop.assert_called_once()
        mock_start.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManagerStopWithDualPort:
    """Test stop_model with dual-port state"""

    async def test_stop_model_cancels_background_tasks(self):
        """Test stop_model cancels standby startup/restart tasks"""
        import asyncio

        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={},
            container_name="model_instance_test-model"
        )

        # Set up dual port state
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        # Track if cancel was called
        startup_cancel_called = False
        restart_cancel_called = False

        # Create a real asyncio Task that we can cancel
        async def mock_long_running_task():
            try:
                await asyncio.sleep(100)  # Will be cancelled
            except asyncio.CancelledError:
                raise

        # Create actual tasks
        mock_startup_task = asyncio.create_task(mock_long_running_task())
        mock_restart_task = asyncio.create_task(mock_long_running_task())

        # Give tasks a chance to start
        await asyncio.sleep(0)

        manager._standby_startup_task = mock_startup_task
        manager._standby_restart_task = mock_restart_task

        # Mock _stop_port_process
        with patch.object(manager, "_stop_port_process", new_callable=AsyncMock):
            result = await manager.stop_model()

        assert result == "test-model"
        # Tasks should be cancelled
        assert mock_startup_task.cancelled()
        assert mock_restart_task.cancelled()
        assert manager._dual_port_state is None
        assert manager.current_model is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestSubprocessManagerBackwardCompatibility:
    """Test backward compatibility with legacy code"""

    async def test_uv_run_process_property_without_dual_state(self):
        """Test uv_run_process returns legacy process when no dual state"""
        manager = SubprocessManager()

        mock_process = Mock()
        manager._legacy_uv_run_process = mock_process

        assert manager.uv_run_process == mock_process

    async def test_uv_run_process_property_with_dual_state(self):
        """Test uv_run_process returns primary process when dual state exists"""
        manager = SubprocessManager()

        mock_process = Mock()
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_a.process = mock_process
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        assert manager.uv_run_process == mock_process

    async def test_uv_run_process_setter(self):
        """Test uv_run_process setter updates both legacy and dual state"""
        manager = SubprocessManager()

        mock_process = Mock()
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        manager.uv_run_process = mock_process

        assert manager._legacy_uv_run_process == mock_process
        assert manager._dual_port_state.primary.process == mock_process


# =============================================================================
# Phase 4: Inference Lock Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestInferenceLock:
    """Test inference lock behavior during hot-switch"""

    async def test_invoke_inference_acquires_lock(self, mock_config):
        """Test invoke_inference uses inference lock"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"output": "result"})
        manager.http_client.post = AsyncMock(return_value=mock_response)

        # Track lock acquisition
        lock_acquired = False
        original_acquire = manager._inference_lock.acquire

        async def track_acquire():
            nonlocal lock_acquired
            result = await original_acquire()
            lock_acquired = True
            return result

        manager._inference_lock.acquire = track_acquire

        with patch("src.subprocess_manager.config", mock_config):
            await manager.invoke_inference({"prompt": "test"})

        assert lock_acquired is True

    async def test_hot_switch_waits_for_inference_lock(self, mock_config):
        """Test hot-switch waits for inference to complete"""
        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}

        # Set up dual port state with healthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b)

        # Acquire inference lock to simulate ongoing inference
        async with manager._inference_lock:
            # Start hot-switch in background - it should wait for lock
            switch_started = asyncio.Event()
            switch_completed = asyncio.Event()

            async def do_switch():
                switch_started.set()
                with patch.object(manager, "_restart_standby_background", new_callable=AsyncMock):
                    with patch("src.subprocess_manager.config", mock_config):
                        await manager._perform_hot_switch()
                switch_completed.set()

            task = asyncio.create_task(do_switch())

            # Wait for switch to start (it will block on lock)
            await switch_started.wait()

            # Give it a moment to try acquiring the lock
            await asyncio.sleep(0.01)

            # Should not be completed yet (blocked on inference lock)
            assert not switch_completed.is_set()

        # Now lock is released, switch should complete
        await task
        assert switch_completed.is_set()


# =============================================================================
# Config Tests
# =============================================================================

@pytest.mark.unit
class TestHotStandbyConfig:
    """Test hot-standby configuration options"""

    def test_config_has_hot_standby_options(self, monkeypatch):
        """Test Config has all hot-standby configuration options"""
        monkeypatch.setenv("INSTANCE_STANDBY_PORT_OFFSET", "2000")
        monkeypatch.setenv("INSTANCE_HOT_STANDBY_MAX_RETRIES", "5")
        monkeypatch.setenv("INSTANCE_HOT_STANDBY_INITIAL_DELAY", "10.0")
        monkeypatch.setenv("INSTANCE_HOT_STANDBY_MAX_DELAY", "60.0")
        monkeypatch.setenv("INSTANCE_HOT_STANDBY_BACKOFF_MULTIPLIER", "3.0")
        monkeypatch.setenv("INSTANCE_STANDBY_RESTART_DELAY", "45")
        monkeypatch.setenv("INSTANCE_BACKUP_HEALTH_TIMEOUT", "900")

        config = Config()

        assert config.standby_port_offset == 2000
        assert config.hot_standby_max_retries == 5
        assert config.hot_standby_initial_delay == 10.0
        assert config.hot_standby_max_delay == 60.0
        assert config.hot_standby_backoff_multiplier == 3.0
        assert config.standby_restart_delay == 45
        assert config.backup_health_check_timeout == 900

    def test_config_hot_standby_defaults(self):
        """Test Config has correct default values for hot-standby"""
        config = Config()

        assert config.standby_port_offset == 1000
        assert config.hot_standby_max_retries == 3
        assert config.hot_standby_initial_delay == 5.0
        assert config.hot_standby_max_delay == 30.0
        assert config.hot_standby_backoff_multiplier == 2.0
        assert config.standby_restart_delay == 30
        assert config.backup_health_check_timeout == 600
