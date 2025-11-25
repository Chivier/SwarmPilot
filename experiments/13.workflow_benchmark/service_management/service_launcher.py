"""Service launcher with CPU/GPU resource binding."""

import os
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger as loguru_logger


@dataclass
class ServiceConfig:
    """Configuration for a service instance."""
    name: str
    service_type: str  # "scheduler", "predictor", "planner", "sleep_model"
    host: str
    port: int
    cpu_cores: str  # e.g., "0-3" or "0,1,2,3"
    gpu_id: Optional[int] = None
    env_vars: Optional[Dict[str, str]] = None
    log_dir: Optional[str] = None


class ServiceLauncher:
    """
    Launches and manages services with resource binding.

    Supports:
    - CPU core affinity (via taskset)
    - GPU device assignment (via CUDA_VISIBLE_DEVICES)
    - Log file management
    - Process lifecycle management
    """

    def __init__(self, log_dir: str = "/tmp/workflow_benchmark", custom_logger: Optional[Any] = None):
        """
        Initialize service launcher.

        Args:
            log_dir: Directory for service logs
            custom_logger: Optional custom logger (defaults to loguru logger)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = custom_logger or loguru_logger

        # Track running processes
        self.processes: Dict[str, subprocess.Popen] = {}

    def get_local_ip(self, interface: str = "bond1") -> Optional[str]:
        """
        Extract local IP address from network interface.

        Args:
            interface: Network interface name

        Returns:
            IP address string, or None if not found
        """
        try:
            # Try using ip command
            cmd = f"ip addr show {interface} | grep 'inet ' | awk '{{print $2}}' | cut -d/ -f1"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                ip = result.stdout.strip()
                if ip:
                    return ip

        except Exception as e:
            self.logger.warning(f"Could not get IP from {interface}: {e}")

        # Fallback: use hostname
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return "127.0.0.1"

    def start_service(
        self,
        config: ServiceConfig,
        command: List[str],
        cwd: Optional[str] = None
    ) -> subprocess.Popen:
        """
        Start a service with resource binding.

        Args:
            config: Service configuration
            command: Command to execute (without taskset prefix)
            cwd: Working directory for the process

        Returns:
            Popen process object

        Raises:
            RuntimeError: If service fails to start
        """
        # Prepare environment variables
        env = os.environ.copy()

        # Set GPU device
        if config.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # Add custom env vars
        if config.env_vars:
            env.update(config.env_vars)

        # Prepare log file
        log_file = self.log_dir / f"{config.name}.log"
        log_handle = open(log_file, "w")

        # Build command with CPU affinity
        if config.cpu_cores:
            full_command = ["taskset", "-c", config.cpu_cores] + command
        else:
            full_command = command

        self.logger.info(
            f"Starting {config.name}: {' '.join(full_command)}"
        )
        self.logger.info(
            f"  Host: {config.host}:{config.port}"
        )
        self.logger.info(
            f"  CPU cores: {config.cpu_cores}"
        )
        if config.gpu_id is not None:
            self.logger.info(f"  GPU: {config.gpu_id}")
        self.logger.info(f"  Log: {log_file}")

        # Start process
        try:
            process = subprocess.Popen(
                full_command,
                env=env,
                cwd=cwd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Create new session to avoid signal propagation
            )

            self.processes[config.name] = process
            self.logger.info(f"Started {config.name} (PID: {process.pid})")

            return process

        except Exception as e:
            log_handle.close()
            raise RuntimeError(f"Failed to start {config.name}: {e}")

    def stop_service(self, name: str, timeout: int = 10) -> bool:
        """
        Stop a running service.

        Args:
            name: Service name
            timeout: Timeout in seconds for graceful shutdown

        Returns:
            True if stopped successfully
        """
        if name not in self.processes:
            self.logger.warning(f"Service {name} not found")
            return False

        process = self.processes[name]

        if process.poll() is not None:
            # Already terminated
            self.logger.info(f"{name} already terminated")
            del self.processes[name]
            return True

        try:
            self.logger.info(f"Stopping {name} (PID: {process.pid})...")

            # Try graceful termination
            process.terminate()

            try:
                process.wait(timeout=timeout)
                self.logger.info(f"{name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                self.logger.warning(f"{name} did not stop gracefully, killing...")
                process.kill()
                process.wait()
                self.logger.info(f"{name} killed")

            del self.processes[name]
            return True

        except Exception as e:
            self.logger.error(f"Error stopping {name}: {e}")
            return False

    def stop_all(self, timeout: int = 10):
        """
        Stop all running services.

        Args:
            timeout: Timeout in seconds for each service
        """
        self.logger.info("Stopping all services...")

        for name in list(self.processes.keys()):
            self.stop_service(name, timeout=timeout)

        self.logger.info("All services stopped")

    def is_running(self, name: str) -> bool:
        """
        Check if service is running.

        Args:
            name: Service name

        Returns:
            True if running
        """
        if name not in self.processes:
            return False

        return self.processes[name].poll() is None

    def get_status(self) -> Dict[str, Dict]:
        """
        Get status of all services.

        Returns:
            Dict mapping service name to status info
        """
        status = {}

        for name, process in self.processes.items():
            status[name] = {
                "pid": process.pid,
                "running": process.poll() is None,
                "returncode": process.returncode
            }

        return status

    def wait_for_port(
        self,
        host: str,
        port: int,
        timeout: int = 30,
        check_interval: float = 0.5
    ) -> bool:
        """
        Wait for a port to become available.

        Args:
            host: Host to check
            port: Port to check
            timeout: Maximum time to wait
            check_interval: Time between checks

        Returns:
            True if port is available within timeout
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=1):
                    self.logger.info(f"Port {host}:{port} is available")
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(check_interval)

        self.logger.error(f"Timeout waiting for {host}:{port}")
        return False
