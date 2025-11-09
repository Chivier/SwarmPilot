"""
Node implementation for graph execution.

A Node encapsulates a complete runtime environment for a model, including:
- An internal Scheduler (automatically started)
- Configured Predictor URL
- Multiple registered Instances

Usage:
    1. Start Predictor (external)
    2. Create and start Node (starts internal scheduler)
    3. Register Instances
    4. Execute tasks
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from clients.instance_client import InstanceClient
from clients.scheduler_client import SchedulerClient


class PossibleFanout(BaseModel):
    """Model for possible fanout configuration."""

    model_id: str
    min_fanout: int
    max_fanout: int


class Node:
    """
    Graph node that encapsulates a complete model runtime environment.

    Each Node contains:
    - An independent Scheduler (auto-started)
    - Configured Predictor URL
    - Multiple registered Instances

    Example:
        ```python
        # 1. Start Predictor (external)
        predictor = PredictorClient(predictor_url="http://localhost:8001")
        await predictor.start_predictor()

        # 2. Create and start Node
        node = Node(
            model_id="gpt-4",
            predictor_url="http://localhost:8001",
            scheduler_port=8100
        )
        await node.start()

        # 3. Register Instances
        instance1 = InstanceClient(base_url="http://instance-1:5000")
        await node.register_instance(instance1)

        # 4. Execute tasks
        result = await node.exec({"prompt": "Hello!"})

        # 5. Cleanup
        await node.stop()
        ```
    """

    def __init__(
        self,
        model_id: str,
        predictor_url: str,
        scheduler_host: str = "localhost",
        scheduler_port: int = 8100,
        scheduler_module_path: Optional[str] = None,
    ):
        """
        Initialize Node.

        Args:
            model_id: Model ID to run on this node
            predictor_url: Predictor service URL (e.g., "http://localhost:8001")
            scheduler_host: Host to bind scheduler to (default: "localhost")
            scheduler_port: Port for scheduler to listen on (default: 8100)
            scheduler_module_path: Path to scheduler module (auto-detected if None)
        """
        self.model_id = model_id
        self.predictor_url = predictor_url
        self.scheduler_host = scheduler_host
        self.scheduler_port = scheduler_port

        # Find scheduler module
        if scheduler_module_path:
            self.scheduler_module_path = Path(scheduler_module_path)
        else:
            self.scheduler_module_path = self._find_scheduler_module()

        # Internal state
        self._scheduler_process: Optional[subprocess.Popen] = None
        self._scheduler_config_file: Optional[Path] = None
        self.scheduler: Optional[SchedulerClient] = None
        self.instance_list: List[InstanceClient] = []
        self._started = False

    def _find_scheduler_module(self) -> Path:
        """
        Find scheduler module path.

        Tries the following in order:
        1. Relative to current file: graph/src/node.py -> scheduler/
        2. Environment variable: SCHEDULER_MODULE_PATH
        3. Current working directory: ../scheduler

        Returns:
            Path to scheduler module

        Raises:
            RuntimeError: If scheduler module cannot be found
        """
        current_file = Path(__file__).resolve()

        # Try: graph/src/node.py -> scheduler/
        scheduler_path = current_file.parent.parent.parent / "scheduler"
        if scheduler_path.exists() and (scheduler_path / "src" / "cli.py").exists():
            return scheduler_path

        # Try environment variable
        if "SCHEDULER_MODULE_PATH" in os.environ:
            path = Path(os.environ["SCHEDULER_MODULE_PATH"])
            if path.exists():
                return path

        # Try: current working directory
        scheduler_path = Path.cwd().parent / "scheduler"
        if scheduler_path.exists() and (scheduler_path / "src" / "cli.py").exists():
            return scheduler_path

        # Default to calculated path (will fail later if not exists)
        return current_file.parent.parent.parent / "scheduler"

    async def start(self):
        """
        Start the Node (starts internal scheduler).

        This will:
        1. Create scheduler configuration file with predictor URL
        2. Start scheduler process
        3. Wait for scheduler to be ready
        4. Create scheduler client

        Raises:
            RuntimeError: If already started or scheduler fails to start
            TimeoutError: If scheduler doesn't become ready in time
        """
        if self._started:
            raise RuntimeError("Node already started")

        print(f"Starting Node: {self.model_id}")

        # 1. Create scheduler configuration
        self._create_scheduler_config()

        # 2. Start scheduler process
        await self._start_scheduler_process()

        # 3. Wait for scheduler ready
        scheduler_url = f"http://{self.scheduler_host}:{self.scheduler_port}"
        await self._wait_for_scheduler_ready(scheduler_url)

        # 4. Create scheduler client
        self.scheduler = SchedulerClient(base_url=scheduler_url)

        self._started = True
        print(f"✅ Node started: {self.model_id}")
        print(f"   Scheduler: {scheduler_url}")
        print(f"   Predictor: {self.predictor_url}")

    def _create_scheduler_config(self):
        """Create scheduler configuration file with predictor URL."""
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "tomli_w is required for Node. Install it with: uv add tomli-w"
            )

        config = {
            "server": {"host": self.scheduler_host, "port": self.scheduler_port},
            "predictor": {
                "url": self.predictor_url,
                "timeout": 30,
                "max_retries": 3,
            },
            "scheduling": {"strategy": "min_time"},
        }

        # Create temporary configuration file
        self._scheduler_config_file = (
            Path(tempfile.gettempdir()) / f"scheduler_node_{id(self)}.toml"
        )

        with open(self._scheduler_config_file, "wb") as f:
            tomli_w.dump(config, f)

        print(f"   Config: {self._scheduler_config_file}")

    async def _start_scheduler_process(self):
        """
        Start scheduler subprocess.

        Raises:
            RuntimeError: If scheduler module not found or process fails immediately
        """
        if not self.scheduler_module_path.exists():
            raise RuntimeError(
                f"Scheduler module not found at: {self.scheduler_module_path}\n"
                f"Set SCHEDULER_MODULE_PATH environment variable or pass scheduler_module_path"
            )

        cli_path = self.scheduler_module_path / "src" / "cli.py"
        if not cli_path.exists():
            raise RuntimeError(
                f"Scheduler CLI not found at: {cli_path}\n"
                f"Expected structure: {self.scheduler_module_path}/src/cli.py"
            )

        cmd = [
            sys.executable,
            "-m",
            "src.cli",
            "start",
            "--config",
            str(self._scheduler_config_file),
        ]

        print(f"   Starting scheduler: {' '.join(cmd)}")

        self._scheduler_process = subprocess.Popen(
            cmd,
            cwd=self.scheduler_module_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        # Check if process immediately failed
        await asyncio.sleep(1.0)
        if self._scheduler_process.poll() is not None:
            stderr = self._scheduler_process.stderr.read().decode()
            stdout = self._scheduler_process.stdout.read().decode()
            raise RuntimeError(
                f"Scheduler process failed to start:\n"
                f"Exit code: {self._scheduler_process.returncode}\n"
                f"Stdout: {stdout}\n"
                f"Stderr: {stderr}"
            )

    async def _wait_for_scheduler_ready(
        self, scheduler_url: str, timeout: float = 30.0, check_interval: float = 0.5
    ):
        """
        Wait for scheduler to become ready.

        Args:
            scheduler_url: Scheduler base URL
            timeout: Maximum time to wait in seconds
            check_interval: Time between health checks in seconds

        Raises:
            TimeoutError: If scheduler doesn't become ready in time
        """
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(
                        f"{scheduler_url}/health", timeout=2.0
                    )
                    if response.status_code == 200:
                        return
                except Exception:
                    pass

                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    # Try to get error info from process
                    error_msg = f"Scheduler not ready after {timeout}s"
                    if self._scheduler_process:
                        if self._scheduler_process.poll() is not None:
                            # Process died
                            stderr = self._scheduler_process.stderr.read().decode()
                            stdout = self._scheduler_process.stdout.read().decode()
                            error_msg += (
                                f"\nProcess exited with code {self._scheduler_process.returncode}"
                                f"\nStdout: {stdout}\nStderr: {stderr}"
                            )
                    raise TimeoutError(error_msg)

                await asyncio.sleep(check_interval)

    async def register_instance(self, instance_client: InstanceClient):
        """
        Register an instance to this node's scheduler.

        This will call instance.start_model() which automatically registers
        the instance to the scheduler.

        Args:
            instance_client: Instance client to register

        Raises:
            RuntimeError: If node not started yet
        """
        if not self._started:
            raise RuntimeError("Node not started. Call await node.start() first")

        # Start model on instance, auto-registers to scheduler
        await instance_client.start_model(
            model_id=self.model_id, scheduler_url=self.scheduler.base_url
        )

        # Add to instance list
        self.instance_list.append(instance_client)

        print(f"✅ Instance registered: {instance_client.base_url}")
        print(f"   Total instances: {len(self.instance_list)}")

    async def exec(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on this node.

        Submits the task to the scheduler and waits for completion.

        Args:
            task_input: Task input data

        Returns:
            Task result

        Raises:
            RuntimeError: If node not started or task fails
        """
        if not self._started:
            raise RuntimeError("Node not started. Call await node.start() first")

        if not self.instance_list:
            raise RuntimeError(
                "No instances registered. Call register_instance() first"
            )

        task_id = str(uuid.uuid4())

        # Submit task to scheduler
        async with self.scheduler:
            response = await self.scheduler.submit_task(
                task_id=task_id,
                model_id=self.model_id,
                task_input=task_input,
                metadata={},
            )

            # Poll for task completion
            while True:
                task_info = await self.scheduler.get_task_info(task_id)
                task = task_info["task"]

                if task["status"] == "completed":
                    return task["result"]
                elif task["status"] == "failed":
                    error = task.get("error", "Unknown error")
                    raise RuntimeError(f"Task {task_id} failed: {error}")

                await asyncio.sleep(0.1)

    async def stop(self):
        """
        Stop the Node (stops internal scheduler).

        Cleans up:
        - Scheduler process
        - Temporary configuration file
        """
        if not self._started:
            return

        print(f"Stopping Node: {self.model_id}")

        # Stop scheduler process
        if self._scheduler_process:
            print("   Terminating scheduler process...")
            self._scheduler_process.terminate()

            try:
                self._scheduler_process.wait(timeout=10)
                print("   Scheduler stopped gracefully")
            except subprocess.TimeoutExpired:
                print("   Force killing scheduler process...")
                self._scheduler_process.kill()
                self._scheduler_process.wait()
                print("   Scheduler killed")

            self._scheduler_process = None

        # Delete temporary config file
        if self._scheduler_config_file and self._scheduler_config_file.exists():
            self._scheduler_config_file.unlink()

        self._started = False
        print(f"✅ Node stopped: {self.model_id}")

    def is_running(self) -> bool:
        """
        Check if node is running.

        Returns:
            True if node is started and scheduler is running
        """
        if not self._started:
            return False

        if self._scheduler_process is None:
            return False

        # Check if process is still alive
        return self._scheduler_process.poll() is None

    def __del__(self):
        """Destructor: ensure scheduler process is cleaned up."""
        if self._scheduler_process and self._scheduler_process.poll() is None:
            self._scheduler_process.terminate()
            try:
                self._scheduler_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._scheduler_process.kill()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
