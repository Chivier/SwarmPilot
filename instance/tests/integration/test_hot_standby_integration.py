"""
Integration tests for SubprocessManager hot-standby functionality.

These tests exercise the hot-standby code paths including:
- Graceful hot-switch with inference lock
- Background standby restart
- Standby startup with retries
"""

import pytest
import asyncio
import signal
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, UTC

from src.subprocess_manager import SubprocessManager
from src.models import (
    ModelInfo, ModelRegistryEntry, RuntimeStandbyConfig,
    PortInfo, PortRole, PortState, DualPortState
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestGracefulHotSwitch:
    """Integration tests for graceful hot-switch with inference lock"""

    async def test_graceful_hot_switch_success(self, mock_config):
        """Test graceful hot-switch waits for inference lock and swaps roles"""
        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}
        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            restart_delay=0  # No delay for testing
        )

        # Set up dual port state with healthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock the restart task
        with patch.object(manager, "_restart_standby_background", new_callable=AsyncMock):
            with patch("src.subprocess_manager.config", mock_config):
                await manager._perform_graceful_hot_switch()

        # Verify roles were swapped
        assert manager._dual_port_state.primary.port == 10000
        assert manager._dual_port_state.standby.port == 9000

    async def test_graceful_hot_switch_no_dual_state(self, mock_config):
        """Test graceful hot-switch raises error when dual state not initialized"""
        manager = SubprocessManager()

        with pytest.raises(RuntimeError) as exc_info:
            await manager._perform_graceful_hot_switch()

        assert "dual port state not initialized" in str(exc_info.value)

    async def test_graceful_hot_switch_standby_not_ready(self, mock_config):
        """Test graceful hot-switch raises error when standby not healthy"""
        manager = SubprocessManager()

        # Set up dual port state with unhealthy standby
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STARTING)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        with pytest.raises(RuntimeError) as exc_info:
            await manager._perform_graceful_hot_switch()

        assert "standby is not healthy" in str(exc_info.value)

    async def test_graceful_hot_switch_waits_for_inference(self, mock_config):
        """Test graceful hot-switch waits for inference lock before swapping"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}
        manager._standby_config = RuntimeStandbyConfig(enabled=True, restart_delay=0)

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.HEALTHY)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Acquire inference lock to simulate ongoing inference
        switch_started = asyncio.Event()
        switch_completed = asyncio.Event()

        async def do_graceful_switch():
            switch_started.set()
            with patch.object(manager, "_restart_standby_background", new_callable=AsyncMock):
                with patch("src.subprocess_manager.config", mock_config):
                    await manager._perform_graceful_hot_switch()
            switch_completed.set()

        async with manager._inference_lock:
            task = asyncio.create_task(do_graceful_switch())
            await switch_started.wait()
            await asyncio.sleep(0.01)

            # Graceful hot-switch should be blocked by inference lock
            assert not switch_completed.is_set()

        # After releasing lock, switch should complete
        await task
        assert switch_completed.is_set()


@pytest.mark.integration
@pytest.mark.asyncio
class TestStandbyBackgroundRestart:
    """Integration tests for background standby restart"""

    async def test_restart_standby_background_success(self, mock_config):
        """Test background standby restart with minimal delay"""
        manager = SubprocessManager()

        # Set up current model
        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}
        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            restart_delay=0  # No delay for testing
        )

        # Set up dual port state
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock the stop and start methods
        with patch.object(manager, "_stop_port_process", new_callable=AsyncMock) as mock_stop:
            with patch.object(manager, "_start_standby_async", new_callable=AsyncMock) as mock_start:
                with patch("src.subprocess_manager.config", mock_config):
                    await manager._restart_standby_background()

        # Verify stop was called on standby port
        mock_stop.assert_called_once()
        # Verify startup was initiated
        mock_start.assert_called_once()

    async def test_restart_standby_background_no_dual_state(self, mock_config):
        """Test background restart returns early when no dual state"""
        manager = SubprocessManager()

        # Should return early without raising
        await manager._restart_standby_background()

    async def test_restart_standby_background_no_current_model(self, mock_config):
        """Test background restart returns early when no current model"""
        manager = SubprocessManager()

        # Set up dual port state but no model
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Should return early without raising
        await manager._restart_standby_background()

    async def test_restart_standby_background_no_model_context(self, mock_config):
        """Test background restart fails gracefully when model context missing"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._standby_config = RuntimeStandbyConfig(enabled=True, restart_delay=0)

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # No _model_dir or _env_vars set

        with patch.object(manager, "_stop_port_process", new_callable=AsyncMock):
            with patch("src.subprocess_manager.config", mock_config):
                await manager._restart_standby_background()

        # Standby should be marked as failed
        assert manager._dual_port_state.standby.state == PortState.FAILED

    async def test_restart_standby_background_stop_failure(self, mock_config):
        """Test background restart continues even if stop fails"""
        manager = SubprocessManager()

        manager.current_model = ModelInfo(
            model_id="test-model",
            started_at="2024-01-01T00:00:00Z",
            parameters={}
        )
        manager._model_dir = Path("/tmp/test")
        manager._env_vars = {"MODEL_ID": "test-model"}
        manager._standby_config = RuntimeStandbyConfig(enabled=True, restart_delay=0)

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock stop to fail, but start should still be called
        with patch.object(manager, "_stop_port_process", new_callable=AsyncMock) as mock_stop:
            mock_stop.side_effect = Exception("Stop failed")
            with patch.object(manager, "_start_standby_async", new_callable=AsyncMock) as mock_start:
                with patch("src.subprocess_manager.config", mock_config):
                    await manager._restart_standby_background()

        # Start should still be called even if stop failed
        mock_start.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
class TestStandbyAsyncStartup:
    """Integration tests for async standby startup with retries"""

    async def test_start_standby_async_success(self, mock_config):
        """Test successful standby startup"""
        manager = SubprocessManager()

        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            max_retries=3,
            initial_delay=0.1,
            max_delay=0.5,
            backoff_multiplier=2.0
        )

        # Set up dual port state
        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock the port process startup to succeed
        with patch.object(manager, "_start_port_process", new_callable=AsyncMock):
            with patch("src.subprocess_manager.config", mock_config):
                await manager._start_standby_async(
                    model_id="test-model",
                    model_dir=Path("/tmp/test"),
                    env_vars={"MODEL_ID": "test-model"}
                )

        # Standby should be in a good state
        assert manager._dual_port_state.standby.startup_attempts == 1

    async def test_start_standby_async_no_dual_state(self, mock_config):
        """Test standby startup returns early when no dual state"""
        manager = SubprocessManager()

        # Should return early without raising
        await manager._start_standby_async(
            model_id="test-model",
            model_dir=Path("/tmp/test"),
            env_vars={}
        )

    async def test_start_standby_async_retry_on_failure(self, mock_config):
        """Test standby startup retries on failure"""
        manager = SubprocessManager()

        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            max_retries=3,
            initial_delay=0.01,  # Very short for testing
            max_delay=0.05,
            backoff_multiplier=2.0
        )

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock startup to fail twice then succeed
        call_count = [0]

        async def mock_start(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception(f"Startup failed attempt {call_count[0]}")

        with patch.object(manager, "_start_port_process", side_effect=mock_start):
            with patch("src.subprocess_manager.config", mock_config):
                await manager._start_standby_async(
                    model_id="test-model",
                    model_dir=Path("/tmp/test"),
                    env_vars={"MODEL_ID": "test-model"}
                )

        # Should have attempted 3 times
        assert call_count[0] == 3

    async def test_start_standby_async_max_retries_exhausted(self, mock_config):
        """Test standby startup marks as failed after max retries"""
        manager = SubprocessManager()

        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            max_retries=2,
            initial_delay=0.01,
            max_delay=0.05,
            backoff_multiplier=2.0
        )

        port_a = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        port_b = PortInfo(port=10000, role=PortRole.STANDBY, state=PortState.STOPPED)
        manager._dual_port_state = DualPortState(port_a=port_a, port_b=port_b, _primary_port_name="port_a")

        # Mock startup to always fail
        with patch.object(manager, "_start_port_process", new_callable=AsyncMock) as mock_start:
            mock_start.side_effect = Exception("Always fails")
            with patch("src.subprocess_manager.config", mock_config):
                await manager._start_standby_async(
                    model_id="test-model",
                    model_dir=Path("/tmp/test"),
                    env_vars={"MODEL_ID": "test-model"}
                )

        # Standby should be marked as failed
        assert manager._dual_port_state.standby.state == PortState.FAILED
        assert "All startup attempts failed" in manager._dual_port_state.standby.error_message


@pytest.mark.integration
@pytest.mark.asyncio
class TestDualPortProcessManagement:
    """Integration tests for dual port process management"""

    async def test_stop_port_process_with_process(self, mock_config):
        """Test stopping a port process"""
        manager = SubprocessManager()

        # Create a port with a mock process
        port_info = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.HEALTHY)
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.wait = AsyncMock(return_value=0)
        port_info.process = mock_process

        with patch("os.killpg") as mock_killpg:
            await manager._stop_port_process(port_info)

        # Process should be killed
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
        assert port_info.state == PortState.STOPPED

    async def test_stop_port_process_already_stopped(self, mock_config):
        """Test stopping a port that's already stopped"""
        manager = SubprocessManager()

        port_info = PortInfo(port=9000, role=PortRole.PRIMARY, state=PortState.STOPPED)
        port_info.process = None

        # Should complete without error
        await manager._stop_port_process(port_info)
        assert port_info.state == PortState.STOPPED

    async def test_initialize_dual_port_state(self, mock_config):
        """Test initializing dual port state"""
        manager = SubprocessManager()

        manager._standby_config = RuntimeStandbyConfig(
            enabled=True,
            port_offset=1000,
            max_retries=3
        )

        # Method uses config.model_port for primary port
        mock_config.model_port = 9000

        with patch("src.subprocess_manager.config", mock_config):
            manager._dual_port_state = manager._initialize_dual_port_state()

        assert manager._dual_port_state is not None
        assert manager._dual_port_state.primary.port == 9000
        assert manager._dual_port_state.standby.port == 10000  # 9000 + 1000
        assert manager._dual_port_state.standby.max_startup_attempts == 3
