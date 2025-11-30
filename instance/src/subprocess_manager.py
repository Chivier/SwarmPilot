"""
Subprocess management for model execution with hot-standby support.

This module provides subprocess-based model management with a hot-standby
port system for zero-downtime restarts. When enabled, two model processes
run simultaneously - one active (primary) and one standby. During restart,
the system instantly switches to the standby port, then restarts the old
primary in the background.
"""

import asyncio
import json
import os
import signal
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger


def log_error_with_traceback(
    error: Exception,
    context: str,
    additional_info: str = "",
) -> None:
    """
    Log error with detailed information and traceback.

    Args:
        error: The exception that occurred
        context: Context description of where the error occurred
        additional_info: Additional context information
    """
    tb_str = traceback.format_exc()
    logger.error(
        f"[SubprocessManager] [{context}] Error occurred:\n"
        f"  Internal error: {type(error).__name__}: {error}\n"
        f"  {additional_info}\n"
        f"  Traceback:\n{tb_str}" if additional_info else
        f"[SubprocessManager] [{context}] Error occurred:\n"
        f"  Internal error: {type(error).__name__}: {error}\n"
        f"  Traceback:\n{tb_str}"
    )

from .config import config
from .model_registry import get_registry
from .models import ModelInfo, PortRole, PortState, PortInfo, DualPortState, RuntimeStandbyConfig
from .port_utils import find_available_port, calculate_backoff_delay


class SubprocessManager:
    """
    Manages subprocesses for model execution with hot-standby support.

    Made for TX server, which does not support docker-in-docker.

    Handles starting, stopping, and health checking of model subprocesses.
    Supports hot-standby mode for zero-downtime restarts.

    Port allocation:
    - Primary port:  config.model_port (instance_port + 1000)
    - Standby port:  config.model_port + config.standby_port_offset (instance_port + 2000)
    """

    def __init__(self):
        self.current_model: Optional[ModelInfo] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Legacy process reference (for backward compatibility)
        self._legacy_uv_run_process: Optional[asyncio.subprocess.Process] = None

        # Hot-standby dual-port state management
        self._dual_port_state: Optional[DualPortState] = None

        # Runtime standby configuration (set during start_model, used for restart)
        self._standby_config: Optional[RuntimeStandbyConfig] = None

        # Locks for thread safety during hot-switch
        self._state_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

        # Background task tracking
        self._standby_startup_task: Optional[asyncio.Task] = None
        self._standby_restart_task: Optional[asyncio.Task] = None

        # Model startup context (saved for standby startup)
        self._model_dir: Optional[Path] = None
        self._env_vars: Optional[Dict[str, str]] = None

    # =========================================================================
    # Port Management Properties
    # =========================================================================

    @property
    def active_port(self) -> int:
        """
        Get the currently active port for inference.

        This property abstracts away the dual-port system.
        External code should use this instead of config.model_port.

        Returns:
            Port number to use for inference/health checks
        """
        if self._dual_port_state is not None:
            return self._dual_port_state.active_port
        return config.model_port

    @property
    def uv_run_process(self) -> Optional[asyncio.subprocess.Process]:
        """
        Backward compatibility: return the primary port's process.

        Deprecated: Use _dual_port_state.primary.process instead for new code.
        """
        if self._dual_port_state is not None:
            return self._dual_port_state.primary.process
        return self._legacy_uv_run_process

    @uv_run_process.setter
    def uv_run_process(self, value: Optional[asyncio.subprocess.Process]):
        """Setter for backward compatibility."""
        if self._dual_port_state is not None:
            self._dual_port_state.primary.process = value
        self._legacy_uv_run_process = value

    def _initialize_dual_port_state(self) -> DualPortState:
        """
        Initialize the dual port state structure.

        Called during first model start. Sets up both port info objects
        with correct port numbers and initial roles.

        Uses self._standby_config for port offset and retry settings.

        Returns:
            Initialized DualPortState
        """
        # Use runtime standby config if available, otherwise fall back to global config
        standby_cfg = self._standby_config
        port_offset = standby_cfg.port_offset if standby_cfg else config.standby_port_offset
        max_retries = standby_cfg.max_retries if standby_cfg else config.hot_standby_max_retries

        primary_port = config.model_port
        standby_port = config.model_port + port_offset

        logger.info(f"Initializing dual-port state: primary={primary_port}, standby={standby_port}")

        return DualPortState(
            port_a=PortInfo(
                port=primary_port,
                role=PortRole.PRIMARY,
                state=PortState.UNINITIALIZED,
                max_startup_attempts=max_retries
            ),
            port_b=PortInfo(
                port=standby_port,
                role=PortRole.STANDBY,
                state=PortState.UNINITIALIZED,
                max_startup_attempts=max_retries
            ),
            _primary_port_name="port_a"
        )

    def is_standby_ready(self) -> bool:
        """Check if standby port is healthy and ready for hot-switch."""
        if self._dual_port_state is None:
            return False
        return self._dual_port_state.standby.is_ready()

    # =========================================================================
    # Core Port Process Management (Phase 3)
    # =========================================================================

    async def _start_port_process(
        self,
        port_info: PortInfo,
        model_id: str,
        model_dir: Path,
        env_vars: Dict[str, str]
    ) -> asyncio.subprocess.Process:
        """
        Start a model subprocess on a specific port.

        This is the low-level method for spawning a subprocess on a given port.
        It updates the PortInfo state throughout the lifecycle.

        Args:
            port_info: The PortInfo object to update with process state
            model_id: Model identifier for logging
            model_dir: Directory containing the model code
            env_vars: Base environment variables (port will be overridden)

        Returns:
            The started subprocess

        Raises:
            RuntimeError: If subprocess fails to start or become healthy
        """
        port = port_info.port
        role = port_info.role.value

        logger.info(f"[{role}] Starting process on port {port} for model {model_id}")

        # Update state to STARTING
        port_info.state = PortState.STARTING
        port_info.started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        port_info.error_message = None

        # Override port in environment
        port_env = env_vars.copy()
        # Note: The main.py accepts --port argument, not env var

        try:
            # Start the subprocess
            process = await asyncio.create_subprocess_exec(
                "uv", "run", "main.py", "--port", str(port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=port_env,
                cwd=str(model_dir),
                start_new_session=True  # Create new process group for proper cleanup
            )

            port_info.process = process

            # Start background tasks to stream output
            asyncio.create_task(
                self._stream_subprocess_output(
                    process.stdout,
                    "stdout",
                    f"{model_id}:{port}"
                )
            )
            asyncio.create_task(
                self._stream_subprocess_output(
                    process.stderr,
                    "stderr",
                    f"{model_id}:{port}"
                )
            )

            logger.info(f"[{role}] Process started (PID: {process.pid}), waiting for health...")

            # Wait for health check - use runtime config for standby timeout
            if port_info.role == PortRole.STANDBY:
                standby_cfg = self._standby_config
                timeout = standby_cfg.health_check_timeout if standby_cfg else config.backup_health_check_timeout
            else:
                timeout = 600
            await self._wait_for_health(port, timeout=timeout, interval=2)

            # Update state to HEALTHY
            port_info.state = PortState.HEALTHY
            port_info.last_health_check = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            port_info.health_check_failures = 0

            logger.info(f"[{role}] Process healthy on port {port} (PID: {process.pid})")

            return process

        except Exception as e:
            error_msg = str(e)
            log_error_with_traceback(
                error=e,
                context=f"_start_port_process({role}, port={port})",
                additional_info=f"Model: {model_id}, Directory: {model_dir}",
            )

            # Update state to FAILED
            port_info.state = PortState.FAILED
            port_info.error_message = error_msg

            # Clean up the process if it was started
            if port_info.process is not None:
                try:
                    await self._stop_subprocess(port_info.process)
                except Exception as stop_error:
                    logger.warning(f"[{role}] Error cleaning up failed process: {stop_error}")
                port_info.process = None

            raise RuntimeError(f"Failed to start {role} process on port {port}: {error_msg}")

    async def _start_standby_async(
        self,
        model_id: str,
        model_dir: Path,
        env_vars: Dict[str, str]
    ) -> None:
        """
        Start the standby process in the background with retry logic.

        This method is designed to run as a background task. It attempts to
        start the standby process with exponential backoff retry on failure.

        Args:
            model_id: Model identifier
            model_dir: Directory containing the model code
            env_vars: Environment variables for the subprocess
        """
        if self._dual_port_state is None:
            logger.error("Cannot start standby: dual port state not initialized")
            return

        standby = self._dual_port_state.standby

        logger.info(f"[standby] Background startup initiated for port {standby.port}")

        while standby.startup_attempts < standby.max_startup_attempts:
            attempt = standby.startup_attempts + 1
            logger.info(
                f"[standby] Startup attempt {attempt}/{standby.max_startup_attempts} "
                f"on port {standby.port}"
            )

            try:
                # Reset state for this attempt
                standby.reset_for_restart()

                # Try to start the process
                await self._start_port_process(
                    port_info=standby,
                    model_id=model_id,
                    model_dir=model_dir,
                    env_vars=env_vars
                )

                logger.info(f"[standby] Successfully started on port {standby.port}")
                return  # Success!

            except Exception as e:
                tb_str = traceback.format_exc()
                logger.warning(
                    f"[standby] Attempt {attempt} failed:\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback:\n{tb_str}"
                )

                # Check if we have more retries
                if standby.startup_attempts >= standby.max_startup_attempts:
                    logger.error(
                        f"[standby] All {standby.max_startup_attempts} attempts exhausted. "
                        f"Standby will not be available for hot-switch."
                    )
                    standby.state = PortState.FAILED
                    standby.error_message = f"All startup attempts failed. Last error: {e}"
                    return

                # Calculate backoff delay using runtime config
                standby_cfg = self._standby_config
                delay = calculate_backoff_delay(
                    attempt=standby.startup_attempts - 1,  # 0-indexed
                    initial_delay=standby_cfg.initial_delay if standby_cfg else config.hot_standby_initial_delay,
                    max_delay=standby_cfg.max_delay if standby_cfg else config.hot_standby_max_delay,
                    multiplier=standby_cfg.backoff_multiplier if standby_cfg else config.hot_standby_backoff_multiplier
                )

                logger.info(f"[standby] Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

    async def _stop_port_process(self, port_info: PortInfo) -> None:
        """
        Stop a process associated with a PortInfo.

        Updates the PortInfo state throughout the shutdown lifecycle.

        Args:
            port_info: The PortInfo whose process should be stopped
        """
        if port_info.process is None:
            logger.debug(f"[{port_info.role.value}] No process to stop on port {port_info.port}")
            port_info.state = PortState.STOPPED
            return

        role = port_info.role.value
        port = port_info.port

        logger.info(f"[{role}] Stopping process on port {port}")

        # Update state to STOPPING
        port_info.state = PortState.STOPPING

        try:
            await self._stop_subprocess(port_info.process)
            port_info.state = PortState.STOPPED
            port_info.process = None
            logger.info(f"[{role}] Process stopped on port {port}")

        except Exception as e:
            log_error_with_traceback(
                error=e,
                context=f"_stop_port_process({role}, port={port})",
            )
            port_info.state = PortState.FAILED
            port_info.error_message = str(e)
            port_info.process = None
            raise

    async def start_model(
        self,
        model_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        standby_enabled: Optional[bool] = None,
        standby_config: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Start a model subprocess with hot-standby support.

        This method:
        1. Validates the model and prepares environment
        2. Creates runtime standby configuration from env + API overrides
        3. Initializes the dual-port state
        4. Starts the primary port process (blocking until healthy)
        5. Schedules standby port startup in the background (non-blocking) if enabled

        Args:
            model_id: The model identifier from the registry
            parameters: Optional model-specific parameters
            standby_enabled: Override for standby enabled (None uses env default)
            standby_config: Dict of standby config overrides from API request

        Returns:
            ModelInfo object with subprocess details

        Raises:
            ValueError: If model not found in registry
            RuntimeError: If subprocess fails to start
        """
        # Build runtime standby configuration from env defaults + API overrides
        self._standby_config = RuntimeStandbyConfig.from_config_and_overrides(
            config=config,
            standby_enabled=standby_enabled,
            overrides=standby_config
        )
        # Validate model exists in registry
        registry = get_registry()
        model_entry = registry.get_model(model_id)
        if not model_entry:
            raise ValueError(f"Model not found in registry: {model_id}")

        # Get model directory
        model_dir = registry.get_model_directory(model_id)
        if not model_dir or not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # Check if uv-related file exists
        pyproject_file = model_dir / "pyproject.toml"
        entry_file = model_dir / "main.py"

        if not pyproject_file.exists() or not entry_file.exists():
            raise ValueError(f"A vaild model should contain at least a main.py and pyproject.toml")

        # Build environment variables
        env_vars = self._build_env_vars(model_id, parameters or {})

        standby_status = "enabled" if self._standby_config.enabled else "disabled"
        logger.info(f"Starting model with subprocess (standby {standby_status}): {model_id}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Primary port: {config.model_port}")
        if self._standby_config.enabled:
            standby_port = config.model_port + self._standby_config.port_offset
            logger.info(f"Standby port: {standby_port}")

        # Run uv sync first
        uv_sync_process = await asyncio.create_subprocess_exec(
            "uv", "sync",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env_vars,
            cwd=str(model_dir)
        )
        stdout, stderr = await uv_sync_process.communicate()
        if uv_sync_process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            raise RuntimeError(f"Failed to run uv sync: {error_msg}")

        # Save model context for standby startup (and potential restarts)
        self._model_dir = model_dir
        self._env_vars = env_vars

        # Create model info before starting
        model_info = ModelInfo(
            model_id=model_id,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            parameters=parameters or {},
            container_name=f"model_instance_{model_id}"
        )

        # Initialize dual-port state
        self._dual_port_state = self._initialize_dual_port_state()
        primary = self._dual_port_state.primary

        # Start primary port process (blocking)
        try:
            logger.info(f"[primary] Starting on port {primary.port}...")
            await self._start_port_process(
                port_info=primary,
                model_id=model_id,
                model_dir=model_dir,
                env_vars=env_vars
            )
            logger.info(f"[primary] Model subprocess is healthy on port {primary.port}")

        except Exception as e:
            log_error_with_traceback(
                error=e,
                context=f"start_model({model_id})",
                additional_info=f"Primary port: {primary.port}, Directory: {model_dir}",
            )
            # Clean up dual port state on failure
            self._dual_port_state = None
            self._standby_config = None
            self._model_dir = None
            self._env_vars = None
            raise RuntimeError(f"Model failed to start: {e}")

        # Set current model
        self.current_model = model_info

        # Schedule standby startup in background (non-blocking) only if enabled
        if self._standby_config.enabled:
            logger.info("[standby] Scheduling background startup...")
            self._standby_startup_task = asyncio.create_task(
                self._start_standby_async(
                    model_id=model_id,
                    model_dir=model_dir,
                    env_vars=env_vars
                )
            )
        else:
            logger.info("[standby] Standby mode is disabled, skipping standby startup")

        return model_info

    async def stop_model(self) -> Optional[str]:
        """
        Stop the currently running model subprocess.

        Stops both primary and standby ports if dual-port mode is active.

        Returns:
            The model_id that was stopped, or None if no model was running

        Raises:
            RuntimeError: If subprocess fails to stop
        """
        if not self.current_model:
            return None

        model_id = self.current_model.model_id
        container_name = self.current_model.container_name

        logger.info(f"Stopping model subprocess: {container_name}")

        # Cancel any pending standby tasks
        if self._standby_startup_task and not self._standby_startup_task.done():
            logger.info("Cancelling standby startup task...")
            self._standby_startup_task.cancel()
            try:
                await self._standby_startup_task
            except asyncio.CancelledError:
                pass
            self._standby_startup_task = None

        if self._standby_restart_task and not self._standby_restart_task.done():
            logger.info("Cancelling standby restart task...")
            self._standby_restart_task.cancel()
            try:
                await self._standby_restart_task
            except asyncio.CancelledError:
                pass
            self._standby_restart_task = None

        # Stop dual-port processes if active
        if self._dual_port_state is not None:
            logger.info("Stopping dual-port processes...")
            try:
                # Stop both ports
                await self._stop_port_process(self._dual_port_state.primary)
                await self._stop_port_process(self._dual_port_state.standby)
                logger.info(f"Model subprocess stopped: {container_name}")
            except Exception as e:
                log_error_with_traceback(
                    error=e,
                    context=f"stop_model({model_id})",
                    additional_info=f"Container: {container_name}",
                )
                raise RuntimeError(f"Failed to stop subprocess: {e}")
            finally:
                self._dual_port_state = None
                self._standby_config = None
                self._model_dir = None
                self._env_vars = None
        else:
            # Legacy single-process mode
            if self.uv_run_process:
                try:
                    await self._stop_subprocess(self.uv_run_process)
                    logger.info(f"Model subprocess stopped: {container_name}")
                except Exception as e:
                    log_error_with_traceback(
                        error=e,
                        context=f"stop_model({model_id})/legacy",
                        additional_info=f"Container: {container_name}",
                    )
                    raise RuntimeError(f"Failed to stop subprocess: {e}")
                finally:
                    self.uv_run_process = None

        self.current_model = None
        return model_id

    async def restart_model(self, force: bool = True) -> Optional[str]:
        """
        Restart the currently running model subprocess.

        Args:
            force: If True (default), force kill the primary process immediately
                   and switch to standby (if available), or perform traditional
                   restart without waiting for graceful shutdown.
                   If False, use graceful hot-switch that waits for in-flight
                   inference to complete.

        If hot-standby is available (standby is HEALTHY):
        - force=True: Kill primary immediately, switch to standby
        - force=False: Wait for inference lock, then switch to standby

        If hot-standby is not available, falls back to traditional restart.

        Returns:
            The model_id that was restarted, or None if no model was running

        Raises:
            RuntimeError: If restart fails
        """
        if not self.current_model:
            logger.warning("No model is running, nothing to restart")
            return None

        model_id = self.current_model.model_id
        container_name = self.current_model.container_name

        logger.info(f"Restarting model subprocess: {container_name} (force={force})")

        # Check if we can do hot-switch
        if self._dual_port_state is not None and self.is_standby_ready():
            if force:
                # Default: force hot-switch (kill primary immediately)
                logger.info("[hot-switch] Killing primary and switching to standby...")
                try:
                    await self._perform_hot_switch()
                    logger.info(f"[hot-switch] Model restarted via hot-switch: {container_name}")
                    return model_id
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"[hot-switch] Hot-switch failed, falling back to traditional restart:\n"
                        f"  Error: {type(e).__name__}: {e}\n"
                        f"  Traceback:\n{tb_str}"
                    )
                    # Fall through to traditional restart
            else:
                logger.info("[hot-switch] Standby is ready, performing graceful hot-switch...")
                try:
                    await self._perform_graceful_hot_switch()
                    logger.info(f"[hot-switch] Model restarted via graceful hot-switch: {container_name}")
                    return model_id
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(
                        f"[hot-switch] Graceful hot-switch failed, falling back to traditional restart:\n"
                        f"  Error: {type(e).__name__}: {e}\n"
                        f"  Traceback:\n{tb_str}"
                    )
                    # Fall through to traditional restart

        # Traditional restart (blocking)
        # Use runtime config delay when standby is disabled (default 30s for /task/clear)
        standby_cfg = self._standby_config
        if standby_cfg and not standby_cfg.enabled:
            restart_delay = standby_cfg.traditional_restart_delay
        else:
            restart_delay = 5  # Quick restart when standby was supposed to be available
        logger.info(
            f"Performing traditional restart (standby not available), "
            f"delay: {restart_delay}s..."
        )
        parameters = self.current_model.parameters

        try:
            # Stop all processes
            await self.stop_model()

            await asyncio.sleep(restart_delay)

            # Restart the model with same parameters
            await self.start_model(model_id, parameters)
            logger.info(f"Model subprocess restarted: {container_name}")

            return model_id

        except Exception as e:
            log_error_with_traceback(
                error=e,
                context=f"restart_model({model_id})",
                additional_info=f"Container: {container_name}",
            )
            raise RuntimeError(f"Failed to restart subprocess: {e}")

    # =========================================================================
    # Hot-Switch Implementation (Phase 4)
    # =========================================================================

    async def _perform_hot_switch(self) -> None:
        """
        Perform hot-switch: kill primary immediately and switch to standby.

        This is the default hot-switch behavior:
        1. Acquires state lock to prevent concurrent modifications
        2. Forcefully kills the primary process (does NOT wait for inference)
        3. Atomically swaps port roles
        4. Schedules background restart of old primary (now standby)

        Precondition: standby must be HEALTHY before calling.

        Raises:
            RuntimeError: If hot-switch fails or standby not ready
        """
        if self._dual_port_state is None:
            raise RuntimeError("Hot-switch not available: dual port state not initialized")

        if not self.is_standby_ready():
            raise RuntimeError("Hot-switch not available: standby is not healthy")

        async with self._state_lock:
            old_primary = self._dual_port_state.primary
            new_primary = self._dual_port_state.standby

            logger.info(
                f"[hot-switch] Switching from port {old_primary.port} to port {new_primary.port}"
            )

            # Force kill the old primary process immediately (don't wait for inference)
            logger.info(f"[hot-switch] Killing primary process on port {old_primary.port}...")
            try:
                await self._stop_port_process(old_primary)
                logger.info(f"[hot-switch] Primary process killed")
            except Exception as e:
                logger.warning(f"[hot-switch] Error killing primary process: {e}")
                # Continue anyway - process might already be dead

            # Atomic role swap
            self._dual_port_state.swap_roles()
            logger.info(
                f"[hot-switch] Roles swapped. New primary: {self._dual_port_state.primary.port}, "
                f"New standby: {self._dual_port_state.standby.port}"
            )

        # Schedule background restart of old primary (now standby)
        logger.info("[hot-switch] Scheduling background restart of old primary...")
        self._standby_restart_task = asyncio.create_task(
            self._restart_standby_background()
        )

    async def _perform_graceful_hot_switch(self) -> None:
        """
        Perform graceful hot-switch: wait for inference to complete before switching.

        Unlike the default _perform_hot_switch, this method:
        1. Waits for any in-flight inference to complete
        2. Then atomically swaps port roles
        3. Schedules background restart of old primary (now standby)

        Use this when you want zero-downtime switching without killing
        in-flight requests.

        Precondition: standby must be HEALTHY before calling.

        Raises:
            RuntimeError: If graceful hot-switch fails or standby not ready
        """
        if self._dual_port_state is None:
            raise RuntimeError("Graceful hot-switch not available: dual port state not initialized")

        if not self.is_standby_ready():
            raise RuntimeError("Graceful hot-switch not available: standby is not healthy")

        async with self._state_lock:
            old_primary = self._dual_port_state.primary
            new_primary = self._dual_port_state.standby

            logger.info(
                f"[graceful-hot-switch] Switching from port {old_primary.port} to port {new_primary.port}"
            )

            # Wait for any in-flight inference to complete
            logger.debug("[graceful-hot-switch] Waiting for inference lock...")
            async with self._inference_lock:
                # Atomic role swap
                self._dual_port_state.swap_roles()
                logger.info(
                    f"[graceful-hot-switch] Roles swapped. New primary: {self._dual_port_state.primary.port}, "
                    f"New standby: {self._dual_port_state.standby.port}"
                )

        # Schedule background restart of old primary (now standby)
        logger.info("[graceful-hot-switch] Scheduling background restart of old primary...")
        self._standby_restart_task = asyncio.create_task(
            self._restart_standby_background()
        )

    async def _restart_standby_background(self) -> None:
        """
        Restart the standby port in the background.

        Called after hot-switch to restart the old primary (now standby).
        Includes configurable delay before restart (30+ seconds).

        This method:
        1. Waits for the configured delay (default 30s)
        2. Stops the old process
        3. Starts a new process on the standby port with retry logic
        """
        if self._dual_port_state is None:
            logger.error("Cannot restart standby: dual port state not initialized")
            return

        if self.current_model is None:
            logger.error("Cannot restart standby: no current model")
            return

        standby = self._dual_port_state.standby
        model_id = self.current_model.model_id

        # Use runtime config for restart delay
        standby_cfg = self._standby_config
        restart_delay = standby_cfg.restart_delay if standby_cfg else config.standby_restart_delay

        logger.info(
            f"[standby-restart] Starting background restart for port {standby.port} "
            f"(delay: {restart_delay}s)"
        )

        # Wait for the configured delay before restarting
        # This allows the old process to fully drain any remaining connections
        await asyncio.sleep(restart_delay)

        # Stop the old process
        logger.info(f"[standby-restart] Stopping old process on port {standby.port}...")
        try:
            await self._stop_port_process(standby)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"[standby-restart] Failed to stop old process:\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{tb_str}"
            )
            # Continue anyway - the process might have crashed

        # Reset standby for new startup
        standby.startup_attempts = 0

        # Start new process using saved context
        if self._model_dir is None or self._env_vars is None:
            logger.error("[standby-restart] Cannot restart: model context not available")
            standby.state = PortState.FAILED
            standby.error_message = "Model context not available for restart"
            return

        logger.info(f"[standby-restart] Starting new process on port {standby.port}...")
        await self._start_standby_async(
            model_id=model_id,
            model_dir=self._model_dir,
            env_vars=self._env_vars
        )

    async def get_current_model(self) -> Optional[ModelInfo]:
        """Get information about the currently running model"""
        return self.current_model

    async def is_model_running(self) -> bool:
        """Check if a model is currently running"""
        return self.current_model is not None

    async def check_model_health(self) -> bool:
        """
        Check if the current model subprocess is healthy.

        Uses active_port to check the currently active process.

        Returns:
            True if healthy, False otherwise
        """
        if not self.current_model:
            return False

        try:
            url = f"http://localhost:{self.active_port}/health"
            response = await self.http_client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Health check failed:\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"  Traceback:\n{tb_str}"
            )
            return False

    async def invoke_inference(
        self,
        task_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke inference on the current model.

        Uses active_port to call the currently active process.
        Acquires inference_lock to prevent port switches during inference.

        Args:
            task_input: Input data for the model

        Returns:
            Inference result from the model

        Raises:
            RuntimeError: If no model is running or inference fails
        """
        if not self.current_model:
            raise RuntimeError("No model is currently running")

        # Acquire inference lock to prevent port switch during inference
        async with self._inference_lock:
            try:
                url = f"http://localhost:{self.active_port}/inference"
                response = await self.http_client.post(
                    url,
                    json=task_input,
                    timeout=2400.0  # 40min timeout for inference
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_detail = response.json().get("error", "Unknown error")
                    raise RuntimeError(f"Inference failed: {error_detail}")

            except httpx.TimeoutException:
                raise RuntimeError("Inference timeout")
            except httpx.RequestError as e:
                raise RuntimeError(f"Inference request failed: {e}")

    def _get_image_name(self, model_id: str) -> str:
        """
        Generate Docker image name from model_id.

        Args:
            model_id: The model identifier

        Returns:
            Docker image name with tag (e.g., "sleep_model:latest")
        """
        # Replace characters that aren't allowed in Docker image names
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return f"{safe_name}:latest"

    def _build_env_vars(
        self,
        model_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Build environment variables for the subprocess.

        Converts parameters to environment variables with MODEL_ prefix.
        For subprocesses, we need to inherit parent environment variables.
        """
        env_vars = {
            "MODEL_ID": model_id,
            "INSTANCE_ID": config.instance_id,
            "LOG_LEVEL": config.log_level,
        }

        # Convert parameters to environment variables
        for key, value in parameters.items():
            # On some system, we need http proxy to access the internet inorder to run the uv sync
            if key == "http_proxy" or key == "https_proxy" or key == "no_proxy":
                env_key = key.upper()
            else:
                env_key = f"MODEL_{key.upper()}"
            if isinstance(value, (dict, list)):
                env_vars[env_key] = json.dumps(value)
            else:
                env_vars[env_key] = str(value)

        # For subprocesses, inherit parent environment (needed for PATH, PYTHONPATH, etc.)
        # This differs from Docker where environment must be explicitly passed
        env_vars.update(os.environ)
        return env_vars


    async def _stream_subprocess_output(
        self,
        stream: asyncio.StreamReader,
        stream_name: str,
        model_id: str
    ):
        """
        Continuously read from subprocess stream and log to loguru.

        Args:
            stream: The stream reader (stdout or stderr)
            stream_name: Name of the stream for logging (e.g., "stdout", "stderr")
            model_id: The model identifier for context
        """
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break

                decoded_line = line.decode().rstrip()
                if decoded_line:
                    if stream_name == "stderr":
                        logger.error(f"[{model_id}] {decoded_line}")
                    else:
                        logger.info(f"[{model_id}] {decoded_line}")
        except Exception as e:
            logger.warning(f"Error streaming {stream_name} for {model_id}: {e}")

    async def _stop_subprocess(
        self,
        process: asyncio.subprocess.Process
    ):
        """
        Stop a subprocess and all its children using SIGKILL (signal 9).

        Since uv spawns child processes (Python interpreter), we need to kill
        the entire process group to ensure all related processes are terminated.

        Args:
            process: The subprocess to stop

        Raises:
            RuntimeError: If subprocess fails to stop
        """
        if process is None:
            return

        pid = process.pid
        logger.info(f"Stopping subprocess and its children with SIGKILL (PID: {pid})")

        # Check if process is still running
        if process.returncode is not None:
            logger.debug(f"Subprocess already finished with return code: {process.returncode}")
            return

        try:
            # Kill the entire process group using SIGKILL (signal 9)
            # The process was started with start_new_session=True, so its PID is the PGID
            try:
                os.killpg(pid, signal.SIGKILL)
                logger.info(f"Sent SIGKILL to process group (PGID: {pid})")
            except ProcessLookupError:
                # Process group already gone, try killing just the process
                logger.debug(f"Process group not found, trying single process kill (PID: {pid})")
                process.kill()

            await process.wait()
            logger.info(f"Subprocess and children killed with SIGKILL (PID: {pid})")

        except ProcessLookupError:
            # Process already finished
            logger.debug(f"Subprocess already finished (PID: {pid})")
        except Exception as e:
            log_error_with_traceback(
                error=e,
                context=f"_stop_subprocess(PID={pid})",
            )
            raise RuntimeError(f"Failed to stop subprocess: {e}")

    async def _wait_for_health(
        self,
        port: int,
        timeout: int = 30,
        interval: int = 2
    ):
        """
        Wait for the model subprocess to become healthy.

        Args:
            port: Port to check
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Raises:
            RuntimeError: If subprocess doesn't become healthy within timeout
        """
        url = f"http://localhost:{port}/health"
        elapsed = 0

        while elapsed < timeout:
            try:
                response = await self.http_client.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info("Model subprocess is healthy")
                        return
            except Exception as e:
                logger.debug(f"Health check attempt failed: {e}")

            await asyncio.sleep(interval)
            elapsed += interval

        raise RuntimeError(
            f"Model subprocess did not become healthy within {timeout} seconds"
        )


    async def close(self):
        """Clean up resources"""
        await self.http_client.aclose()


# Global subprocess manager instance
_subprocess_manager: Optional[SubprocessManager] = None


def get_subprocess_manager() -> SubprocessManager:
    """Get or create the global subprocess manager instance"""
    global _subprocess_manager
    if _subprocess_manager is None:
        _subprocess_manager = SubprocessManager()
    return _subprocess_manager


# Backward compatibility alias for easy migration from DockerManager
# Users can replace `from .docker_manager import get_docker_manager` 
# with `from .subprocess_manager import get_docker_manager` for drop-in replacement
def get_docker_manager() -> SubprocessManager:
    """
    Backward compatibility alias for get_subprocess_manager().
    
    This allows easy migration from DockerManager to SubprocessManager
    by simply changing the import:
    
    Before: from .docker_manager import get_docker_manager
    After:  from .subprocess_manager import get_docker_manager
    
    The interface is identical, so no code changes are needed.
    """
    return get_subprocess_manager()
