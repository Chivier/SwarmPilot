"""
Scheduler Client for Instance Service.

This module provides functionality for instances to communicate with the scheduler,
including registration, deregistration, task result callbacks, configuration
management, and automatic scheduler startup.

Features:
- Instance registration with scheduler
- Configuration of predictor and scheduling preferences
- Task result callbacks with automatic retry
- Instance draining support
- Platform auto-detection
- Environment variable configuration support
- Automatic scheduler startup with dependency checking
- Smart port detection and auto-discovery

Example usage:
    ```python
    # Initialize with custom configuration
    client = SchedulerClient(
        scheduler_url="http://scheduler:8000",
        instance_id="gpu-node-1",
        instance_endpoint="http://gpu-node-1:5000",
        predictor_config=PredictorConfig(
            url="http://predictor:8001",
            timeout=10.0
        ),
        scheduling_config=SchedulingConfig(
            default_strategy="probabilistic",
            probabilistic_quantile=0.95
        )
    )

    # Auto-start scheduler (with dependency checking)
    await client.start_scheduler(auto_find_port=True)

    # Register with scheduler
    await client.register_instance(
        model_id="gpt-4",
        platform_info={"software_name": "vllm", ...}
    )

    # Send task results
    await client.send_task_result(
        task_id="task-001",
        status="completed",
        result={"output": "..."},
        execution_time_ms=1234.5
    )
    ```
"""

import asyncio
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx


# ==================== GPU Detection Utility ====================


def _get_gpu0_name() -> str:
    """Get the name of GPU0, or 'CPU' if no GPU is available.
    
    Tries multiple methods to detect GPU:
    1. pynvml (NVIDIA Management Library Python bindings)
    2. nvidia-smi command
    3. Falls back to 'CPU' if no GPU is found
    
    Returns:
        GPU name string (e.g., "NVIDIA GeForce RTX 3090") or "CPU"
    """
    # Method 1: Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        # Decode bytes to string if needed
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        return name
    except (ImportError, Exception):
        pass
    
    # Method 2: Try nvidia-smi command
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '-i', '0'],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        if result.stdout:
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first line (GPU0)
            if gpu_name:
                return gpu_name
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception):
        pass
    
    # No GPU found, return CPU
    return "CPU"


# ==================== Configuration Classes ====================


@dataclass
class PredictorConfig:
    """Configuration for predictor service integration.

    These settings control how the scheduler communicates with the predictor
    service for runtime predictions.
    """

    url: str = field(default_factory=lambda: os.getenv("PREDICTOR_URL", "http://localhost:8001"))
    timeout: float = field(default_factory=lambda: float(os.getenv("PREDICTOR_TIMEOUT", "5.0")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("PREDICTOR_MAX_RETRIES", "3")))
    retry_delay: float = field(default_factory=lambda: float(os.getenv("PREDICTOR_RETRY_DELAY", "1.0")))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of predictor configuration
        """
        return {
            "url": self.url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }


@dataclass
class SchedulingConfig:
    """Configuration for scheduling behavior.

    These settings define how the scheduler should assign tasks to this instance.
    """

    default_strategy: str = field(default_factory=lambda: os.getenv("SCHEDULING_STRATEGY", "probabilistic"))
    probabilistic_quantile: float = field(
        default_factory=lambda: float(os.getenv("SCHEDULING_PROBABILISTIC_QUANTILE", "0.9"))
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of scheduling configuration
        """
        return {
            "default_strategy": self.default_strategy,
            "probabilistic_quantile": self.probabilistic_quantile,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline.

    These settings control automatic training behavior for predictor models.
    """

    enable_auto_training: bool = field(
        default_factory=lambda: os.getenv("TRAINING_ENABLE_AUTO", "false").lower() == "true"
    )
    batch_size: int = field(default_factory=lambda: int(os.getenv("TRAINING_BATCH_SIZE", "100")))
    frequency_seconds: int = field(default_factory=lambda: int(os.getenv("TRAINING_FREQUENCY", "3600")))
    min_samples: int = field(default_factory=lambda: int(os.getenv("TRAINING_MIN_SAMPLES", "10")))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of training configuration
        """
        return {
            "enable_auto_training": self.enable_auto_training,
            "batch_size": self.batch_size,
            "frequency_seconds": self.frequency_seconds,
            "min_samples": self.min_samples,
        }


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket communication.

    These settings control WebSocket connection behavior between instance and scheduler.
    """

    heartbeat_interval: int = field(default_factory=lambda: int(os.getenv("WEBSOCKET_HEARTBEAT_INTERVAL", "30")))
    heartbeat_timeout_threshold: int = field(
        default_factory=lambda: int(os.getenv("WEBSOCKET_HEARTBEAT_THRESHOLD", "3"))
    )
    ack_timeout: float = field(default_factory=lambda: float(os.getenv("WEBSOCKET_ACK_TIMEOUT", "10.0")))
    max_message_size: int = field(
        default_factory=lambda: int(os.getenv("WEBSOCKET_MAX_MESSAGE_SIZE", str(16 * 1024 * 1024)))
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of WebSocket configuration
        """
        return {
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout_threshold": self.heartbeat_timeout_threshold,
            "ack_timeout": self.ack_timeout,
            "max_message_size": self.max_message_size,
        }


# ==================== Scheduler Client ====================


class SchedulerClient:
    """Client for communicating with the scheduler service.

    Provides methods for instance registration, deregistration, task result callbacks,
    configuration management, and automatic scheduler startup with dependency checking.
    """

    def __init__(
        self,
        scheduler_url: Optional[str] = None,
        instance_id: Optional[str] = None,
        instance_endpoint: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        predictor_config: Optional[PredictorConfig] = None,
        scheduling_config: Optional[SchedulingConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        websocket_config: Optional[WebSocketConfig] = None,
        scheduler_module_path: Optional[str] = None,
    ):
        """Initialize scheduler client with configuration.

        Args:
            scheduler_url: Base URL of the scheduler service (e.g., http://localhost:8000)
                          Falls back to SCHEDULER_URL env var if not provided
            instance_id: Unique identifier for this instance
                        Falls back to INSTANCE_ID env var or "instance-default"
            instance_endpoint: HTTP endpoint where this instance is accessible
                              Falls back to INSTANCE_ENDPOINT env var or auto-generated
            timeout: Request timeout in seconds for scheduler communication
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds (exponential backoff)
            predictor_config: Configuration for predictor service integration
            scheduling_config: Configuration for scheduling behavior preferences
            training_config: Configuration for automatic training
            websocket_config: Configuration for WebSocket communication
            scheduler_module_path: Path to scheduler module (for auto-start)
                                  Falls back to SCHEDULER_MODULE_PATH env var
        """
        # Core connection settings
        self.scheduler_url = scheduler_url or os.getenv("SCHEDULER_URL", "http://localhost:8000")
        self.instance_id = instance_id or os.getenv("INSTANCE_ID", str(uuid.uuid4()))
        self.instance_endpoint = instance_endpoint or os.getenv(
            "INSTANCE_ENDPOINT", f"http://localhost:{os.getenv('INSTANCE_PORT', '8000')}"
        )

        # Request settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configuration objects
        self.predictor_config = predictor_config or PredictorConfig()
        self.scheduling_config = scheduling_config or SchedulingConfig()
        self.training_config = training_config or TrainingConfig()
        self.websocket_config = websocket_config or WebSocketConfig()

        # Scheduler module path for auto-start
        self.scheduler_module_path = scheduler_module_path or os.getenv(
            "SCHEDULER_MODULE_PATH", "../scheduler/src/main.py"
        )

        # Registration and process state
        self._registered = False
        self._scheduler_process: Optional[subprocess.Popen] = None

        print(f"Initializing scheduler client:")
        print(f"  Scheduler URL: {self.scheduler_url}")
        print(f"  Instance ID: {self.instance_id}")
        print(f"  Instance Endpoint: {self.instance_endpoint}")
        print(f"  Predictor URL: {self.predictor_config.url}")
        print(f"  Scheduling Strategy: {self.scheduling_config.default_strategy}")

    @property
    def is_enabled(self) -> bool:
        """Check if scheduler integration is enabled.

        Returns:
            True if scheduler URL is configured, False otherwise
        """
        return self.scheduler_url is not None

    @property
    def is_registered(self) -> bool:
        """Check if instance is currently registered with scheduler.

        Returns:
            True if registered, False otherwise
        """
        return self._registered

    @property
    def is_scheduler_running(self) -> bool:
        """Check if scheduler process is running (if started by this client).

        Returns:
            True if scheduler process is running, False otherwise
        """
        if self._scheduler_process is None:
            return False
        return self._scheduler_process.poll() is None

    def get_config_dict(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary.

        Returns:
            Dictionary containing all configuration settings
        """
        return {
            "predictor": self.predictor_config.to_dict(),
            "scheduling": self.scheduling_config.to_dict(),
            "training": self.training_config.to_dict(),
            "websocket": self.websocket_config.to_dict(),
        }

    # ==================== Port and Dependency Management ====================

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use.

        Args:
            port: Port number to check

        Returns:
            True if port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return False
            except OSError:
                return True

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> Optional[int]:
        """Find an available port starting from the given port.

        Args:
            start_port: Starting port number
            max_attempts: Maximum number of ports to try

        Returns:
            Available port number, or None if no port found
        """
        for port in range(start_port, start_port + max_attempts):
            if not self._is_port_in_use(port):
                return port
        return None

    def _extract_port_from_url(self, url: str) -> int:
        """Extract port number from URL.

        Args:
            url: URL string (e.g., "http://localhost:8000")

        Returns:
            Port number (defaults to 80 for http, 443 for https if not specified)
        """
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80

    def _update_scheduler_url_port(self, new_port: int) -> None:
        """Update scheduler URL with new port.

        Args:
            new_port: New port number
        """
        parsed = urlparse(self.scheduler_url)
        self.scheduler_url = f"{parsed.scheme}://{parsed.hostname}:{new_port}{parsed.path}"

    async def _check_predictor_available(self) -> bool:
        """Check if predictor service is available.

        Returns:
            True if predictor is reachable and healthy, False otherwise
        """
        print(f"Checking predictor availability at {self.predictor_config.url}...")

        try:
            async with httpx.AsyncClient(timeout=self.predictor_config.timeout) as client:
                response = await client.get(f"{self.predictor_config.url}/health")
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "healthy":
                    print(f"✓ Predictor service is healthy")
                    return True
                else:
                    print(f"✗ Predictor service returned unhealthy status: {data}")
                    return False

        except httpx.HTTPError as e:
            print(f"✗ Predictor service not reachable: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error checking predictor: {e}")
            return False

    async def _wait_for_scheduler_ready(self, timeout: float = 30.0, check_interval: float = 1.0) -> bool:
        """Wait for scheduler to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Interval between health checks in seconds

        Returns:
            True if scheduler is ready, False if timeout
        """
        print(f"Waiting for scheduler to become ready (timeout: {timeout}s)...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.scheduler_url}/health")
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy":
                            elapsed = time.time() - start_time
                            print(f"✓ Scheduler is ready (took {elapsed:.1f}s)")
                            return True
            except Exception:
                pass

            await asyncio.sleep(check_interval)

        print(f"✗ Scheduler did not become ready within {timeout}s")
        return False

    # ==================== Scheduler Auto-Start ====================

    async def start_scheduler(
        self,
        auto_find_port: bool = True,
        max_port_search: int = 100,
        wait_timeout: float = 30.0,
        check_predictor: bool = True,
    ) -> bool:
        """Start scheduler service automatically with dependency checking.

        This method performs the following steps:
        1. Check if predictor service is available (if check_predictor=True)
        2. Check if scheduler port is available
        3. If port is occupied and auto_find_port=True, find an available port
        4. Start scheduler process
        5. Wait for scheduler to become ready

        Args:
            auto_find_port: Automatically find available port if configured port is occupied
            max_port_search: Maximum number of ports to search for availability
            wait_timeout: Maximum time to wait for scheduler to become ready (seconds)
            check_predictor: Whether to check predictor availability before starting

        Returns:
            True if scheduler started successfully, False otherwise

        Raises:
            RuntimeError: If scheduler module path is not found

        Example:
            ```python
            client = SchedulerClient(
                scheduler_url="http://localhost:8000",
                predictor_config=PredictorConfig(url="http://localhost:8001")
            )

            # Start scheduler with auto port discovery
            success = await client.start_scheduler(auto_find_port=True)
            if success:
                print(f"Scheduler running at {client.scheduler_url}")
            ```
        """
        print("=" * 60)
        print("Starting Scheduler Service")
        print("=" * 60)

        # Step 1: Check predictor availability
        if check_predictor:
            if not await self._check_predictor_available():
                print("✗ Cannot start scheduler: Predictor service is not available")
                print("  Please ensure predictor is running before starting scheduler")
                return False

        # Step 2: Check if scheduler is already running
        current_port = self._extract_port_from_url(self.scheduler_url)
        print(f"Checking scheduler port {current_port}...")

        if self._is_port_in_use(current_port):
            # Port is in use, check if it's our scheduler
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{self.scheduler_url}/health")
                    if response.status_code == 200:
                        print(f"✓ Scheduler is already running at {self.scheduler_url}")
                        return True
            except Exception:
                pass

            # Port is occupied by something else
            if auto_find_port:
                print(f"⚠ Port {current_port} is occupied, searching for available port...")
                available_port = self._find_available_port(current_port + 1, max_port_search)

                if available_port is None:
                    print(f"✗ Could not find available port in range {current_port + 1} to {current_port + max_port_search}")
                    return False

                print(f"✓ Found available port: {available_port}")
                self._update_scheduler_url_port(available_port)
                current_port = available_port
            else:
                print(f"✗ Port {current_port} is already in use (auto_find_port=False)")
                return False

        # Step 3: Verify scheduler module exists
        if not os.path.exists(self.scheduler_module_path):
            raise RuntimeError(f"Scheduler module not found at: {self.scheduler_module_path}")

        # Step 4: Prepare environment variables
        env = os.environ.copy()
        env["SCHEDULER_PORT"] = str(current_port)
        env["PREDICTOR_URL"] = self.predictor_config.url
        env["SCHEDULING_STRATEGY"] = self.scheduling_config.default_strategy
        env["SCHEDULING_PROBABILISTIC_QUANTILE"] = str(self.scheduling_config.probabilistic_quantile)

        # Step 5: Start scheduler process
        print(f"Starting scheduler at port {current_port}...")
        print(f"  Module: {self.scheduler_module_path}")
        print(f"  Predictor: {self.predictor_config.url}")
        print(f"  Strategy: {self.scheduling_config.default_strategy}")

        try:
            # Use uvicorn to run the scheduler
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(current_port),
                "--log-level",
                "info",
            ]

            # Change to scheduler directory
            scheduler_dir = os.path.dirname(os.path.abspath(self.scheduler_module_path))

            self._scheduler_process = subprocess.Popen(
                cmd,
                cwd=scheduler_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print(f"✓ Scheduler process started (PID: {self._scheduler_process.pid})")

        except Exception as e:
            print(f"✗ Failed to start scheduler process: {e}")
            return False

        # Step 6: Wait for scheduler to become ready
        is_ready = await self._wait_for_scheduler_ready(timeout=wait_timeout)

        if not is_ready:
            print("✗ Scheduler failed to become ready")
            self.stop_scheduler()
            return False

        print("=" * 60)
        print(f"✓ Scheduler is running at {self.scheduler_url}")
        print("=" * 60)
        return True

    def stop_scheduler(self) -> bool:
        """Stop the scheduler process if it was started by this client.

        Returns:
            True if scheduler was stopped successfully, False otherwise
        """
        if self._scheduler_process is None:
            print("No scheduler process to stop")
            return False

        if not self.is_scheduler_running:
            print("Scheduler process is already stopped")
            self._scheduler_process = None
            return True

        print(f"Stopping scheduler process (PID: {self._scheduler_process.pid})...")

        try:
            self._scheduler_process.terminate()
            self._scheduler_process.wait(timeout=10.0)
            print("✓ Scheduler stopped successfully")
            self._scheduler_process = None
            return True
        except subprocess.TimeoutExpired:
            print("⚠ Scheduler did not stop gracefully, forcing termination...")
            self._scheduler_process.kill()
            self._scheduler_process.wait()
            print("✓ Scheduler killed")
            self._scheduler_process = None
            return True
        except Exception as e:
            print(f"✗ Error stopping scheduler: {e}")
            return False

    # ==================== Instance Registration API ====================

    async def register_instance(
        self,
        model_id: str,
        platform_info: Optional[Dict[str, Any]] = None,
        include_config: bool = True,
    ) -> bool:
        """Register this instance with the scheduler.

        Sends instance metadata and configuration to the scheduler for task routing.
        The scheduler will use the provided configuration when making decisions about
        this instance.

        Args:
            model_id: ID of the model this instance is serving (e.g., "gpt-4", "llama-2-70b")
            platform_info: Platform information (auto-detected if not provided)
                          Should include: software_name, software_version, hardware_name
            include_config: Whether to include configuration in registration request
                           Set to False if scheduler manages configuration globally

        Returns:
            True if registration successful, False otherwise
        """
        if not self.is_enabled:
            print("Scheduler integration disabled (SCHEDULER_URL not set)")
            return False

        # Auto-detect platform info if not provided
        if platform_info is None:
            platform_info = self._get_platform_info()

        # Build registration payload
        registration_data = {
            "instance_id": self.instance_id,
            "model_id": model_id,
            "endpoint": self.instance_endpoint,
            "platform_info": platform_info,
        }

        # Optionally include configuration
        if include_config:
            registration_data["config"] = self.get_config_dict()

        # Retry logic for registration
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.scheduler_url}/instance/register",
                        json=registration_data,
                    )
                    response.raise_for_status()
                    result = response.json()

                if result.get("success", False):
                    self._registered = True
                    print(f"✓ Instance {self.instance_id} registered with scheduler successfully")
                    print(f"  Model: {model_id}")
                    print(f"  Platform: {platform_info.get('software_name', 'unknown')}")
                    if include_config:
                        print(f"  Predictor: {self.predictor_config.url}")
                        print(f"  Strategy: {self.scheduling_config.default_strategy}")
                    return True
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"✗ Registration failed: {error_msg}")
                    return False

            except httpx.HTTPError as e:
                print(f"Registration attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    print(f"  Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"✗ Failed to register instance after {self.max_retries} attempts")
                    return False

            except Exception as e:
                print(f"✗ Unexpected error during registration: {str(e)}")
                return False

        return False

    async def deregister_instance(self) -> bool:
        """Deregister this instance from the scheduler.

        Removes the instance from the scheduler's instance pool. The scheduler will
        stop assigning new tasks to this instance.

        Returns:
            True if deregistration successful, False otherwise
        """
        if not self.is_enabled or not self._registered:
            print("Cannot deregister: scheduler disabled or instance not registered")
            return False

        deregistration_data = {
            "instance_id": self.instance_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/remove",
                    json=deregistration_data,
                )
                response.raise_for_status()
                result = response.json()

            if result.get("success", False):
                self._registered = False
                print(f"✓ Instance {self.instance_id} deregistered from scheduler")
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"✗ Deregistration failed: {error_msg}")
                return False

        except httpx.HTTPError as e:
            print(f"✗ Failed to deregister instance: {str(e)}")
            return False

        except Exception as e:
            print(f"✗ Unexpected error during deregistration: {str(e)}")
            return False

    async def drain_instance(self) -> Dict[str, Any]:
        """Request the scheduler to drain this instance.

        Stops the scheduler from assigning new tasks to this instance while allowing
        existing tasks to complete. Useful for graceful shutdown or maintenance.

        Returns:
            Dictionary with drain status information

        Raises:
            Exception: If drain request fails
        """
        if not self.is_enabled:
            raise Exception("Scheduler integration disabled (SCHEDULER_URL not set)")

        drain_data = {
            "instance_id": self.instance_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/drain",
                    json=drain_data,
                )
                response.raise_for_status()
                result = response.json()

            if result.get("success", False):
                print(f"✓ Instance {self.instance_id} is now draining")
                print(f"  Pending tasks: {result.get('pending_tasks', 0)}")
                print(f"  Running tasks: {result.get('running_tasks', 0)}")
                if "estimated_completion_time_ms" in result:
                    print(f"  Estimated completion: {result['estimated_completion_time_ms']}ms")
                return result
            else:
                error_msg = result.get("error", "Unknown error")
                raise Exception(f"Drain request failed: {error_msg}")

        except httpx.HTTPError as e:
            raise Exception(f"Failed to drain instance: {str(e)}")

        except Exception as e:
            raise Exception(f"Unexpected error during drain: {str(e)}")

    async def resubmit_task(
        self,
        task_id: str,
        model_id: str,
        task_input: Dict[str, Any],
        enqueue_time: Optional[float] = None,
        submitted_at: Optional[str] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resubmit a task back to the scheduler for redistribution.

        This method is used during instance restart/redeploy to send extracted
        pending tasks back to the scheduler for reassignment to other instances.

        Args:
            task_id: Original task ID
            model_id: Model ID required for the task
            task_input: Task input data
            enqueue_time: Original enqueue timestamp (for priority preservation)
            submitted_at: Original submission timestamp
            callback_url: Optional callback URL for task completion
            metadata: Optional task metadata

        Returns:
            True if task successfully resubmitted, False otherwise

        Raises:
            Exception: If scheduler integration is disabled or request fails
        """
        if not self.is_enabled:
            raise Exception("Cannot resubmit task: scheduler integration is disabled")

        endpoint = f"{self.scheduler_url}/task/submit"

        # Build task submission payload
        payload = {
            "task_id": task_id,
            "model_id": model_id,
            "task_input": task_input,
        }

        # Include optional fields for priority and context preservation
        if enqueue_time is not None:
            payload["enqueue_time"] = enqueue_time
        if submitted_at is not None:
            payload["submitted_at"] = submitted_at
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if metadata is not None:
            payload["metadata"] = metadata

        # Retry logic for reliability
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()

                if result.get("success", False):
                    return True
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"Task resubmission failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        return False

            except httpx.HTTPError as e:
                print(
                    f"Resubmit attempt {attempt + 1}/{self.max_retries} failed "
                    f"for task {task_id}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    print(
                        f"✗ Failed to resubmit task {task_id} after "
                        f"{self.max_retries} attempts"
                    )
                    return False

            except Exception as e:
                print(f"✗ Unexpected error during task resubmission: {str(e)}")
                return False

        return False

    # ==================== Task Result Callback API ====================

    async def send_task_result(
        self,
        task_id: str,
        status: str,
        callback_url: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
    ) -> bool:
        """Send task result to scheduler via callback.

        Args:
            task_id: ID of the completed task
            status: Task status - "completed" or "failed"
            callback_url: Callback URL for task result
            result: Task result data
            error: Error message
            execution_time_ms: Execution time in milliseconds
        Returns:
            True if callback successful, False otherwise
        """
        if not callback_url:
            raise ValueError("Callback URL is required")

        callback_data = {
            "task_id": task_id,
            "status": status,
        }

        if result is not None:
            callback_data["result"] = result
        if error is not None:
            callback_data["error"] = error
        if execution_time_ms is not None:
            callback_data["execution_time_ms"] = execution_time_ms

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        callback_url,
                        json=callback_data,
                    )
                    response.raise_for_status()
                    result_data = response.json()

                if result_data.get("success", False):
                    return True
                else:
                    error_msg = result_data.get("error", "Unknown error")
                    print(f"Callback failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        return False

            except httpx.HTTPError as e:
                print(f"Callback attempt {attempt + 1}/{self.max_retries} failed for task {task_id}: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    print(f"✗ Failed to send callback after {self.max_retries} attempts")
                    return False

            except Exception as e:
                print(f"✗ Unexpected error during callback: {str(e)}")
                return False

        return False

    # ==================== Utility Methods ====================

    def _get_platform_info(self) -> Dict[str, Any]:
        """Get platform information with support for user overrides.

        Checks config for platform overrides (from environment variables or CLI args).
        Falls back to auto-detection if overrides are not provided.

        hardware_name will be set to GPU0's name if available, otherwise "CPU".

        Returns:
            Dictionary containing platform information
        """
        from .config import config

        return {
            "software_name": config.platform_software_name or platform.system(),
            "software_version": config.platform_software_version or platform.release(),
            "hardware_name": config.platform_hardware_name or _get_gpu0_name(),
        }

    def update_predictor_config(self, predictor_config: PredictorConfig) -> None:
        """Update predictor configuration."""
        self.predictor_config = predictor_config
        print(f"Updated predictor config: {predictor_config.url}")

    def update_scheduling_config(self, scheduling_config: SchedulingConfig) -> None:
        """Update scheduling configuration."""
        self.scheduling_config = scheduling_config
        print(f"Updated scheduling config: {scheduling_config.default_strategy}")


# ==================== Global Client Instance ====================


_scheduler_client: Optional[SchedulerClient] = None


def get_scheduler_client() -> SchedulerClient:
    """Get the global scheduler client instance."""
    global _scheduler_client
    if _scheduler_client is None:
        _scheduler_client = SchedulerClient()
    return _scheduler_client


def initialize_scheduler_client(
    scheduler_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    instance_endpoint: Optional[str] = None,
    predictor_config: Optional[PredictorConfig] = None,
    scheduling_config: Optional[SchedulingConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    websocket_config: Optional[WebSocketConfig] = None,
    scheduler_module_path: Optional[str] = None,
) -> SchedulerClient:
    """Initialize and configure the global scheduler client."""
    global _scheduler_client
    _scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url,
        instance_id=instance_id,
        instance_endpoint=instance_endpoint,
        predictor_config=predictor_config,
        scheduling_config=scheduling_config,
        training_config=training_config,
        websocket_config=websocket_config,
        scheduler_module_path=scheduler_module_path,
    )
    return _scheduler_client
