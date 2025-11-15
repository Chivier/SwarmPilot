"""
Subprocess management for model execution
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from .config import config
from .model_registry import get_registry
from .models import ModelInfo


class SubprocessManager:
    """
    Manages subprocesses for model execution.
    
    Made for TX server, which does not support docker-in-docker.

    Handles starting, stopping, and health checking of model subprocesses.
    """

    def __init__(self):
        self.current_model: Optional[ModelInfo] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.uv_run_process: Optional[asyncio.subprocess.Process] = None

    async def start_model(
        self,
        model_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Start a model subprocess.

        Args:
            model_id: The model identifier from the registry
            parameters: Optional model-specific parameters

        Returns:
            ModelInfo object with subprocess details

        Raises:
            ValueError: If model not found in registry
            RuntimeError: If subprocess fails to start
        """
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
        # # Check if Dockerfile exists
        # dockerfile = model_dir / "Dockerfile"
        # if not dockerfile.exists():
        #     raise ValueError(f"Dockerfile not found in {model_dir}")

        # # Generate container and image names
        # container_name = config.get_model_container_name(model_id)
        # image_name = self._get_image_name(model_id)

        # Build environment variables
        env_vars = self._build_env_vars(model_id, parameters or {})

        logger.info(f"Starting model with subprocess: {model_id}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Container port: {config.model_port}")
        
        # Run uv sync at background
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
        
        # Create model info before starting (so we can track it even if startup fails)
        from datetime import UTC, datetime
        model_info = ModelInfo(
            model_id=model_id,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            parameters=parameters or {},
            container_name=f"model_instance_{model_id}"
        )

        # Run uv run at background (This is the main process, run it at the background then return, no need to wait for it)
        # Redirect the stdout and stderr to the logger
        # Change working directory to model_dir so uv can find the project
        logger.info(f"Running uv run with command: uv run main.py --port {config.model_port}")
        self.uv_run_process = await asyncio.create_subprocess_exec(
            "uv", "run", "main.py", "--port", str(config.model_port),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env_vars,
            cwd=str(model_dir)
        )

        # Start background tasks to stream stdout and stderr to loguru
        asyncio.create_task(
            self._stream_subprocess_output(
                self.uv_run_process.stdout,
                "stdout",
                model_id
            )
        )
        asyncio.create_task(
            self._stream_subprocess_output(
                self.uv_run_process.stderr,
                "stderr",
                model_id
            )
        )

        # Wait for the model to become healthy
        try:
            await self._wait_for_health(config.model_port, timeout=30, interval=2)
            logger.info("Model subprocess is healthy")
        except RuntimeError as e:
            # If health check fails, try to stop the process
            logger.error(f"Model failed to become healthy: {e}")
            if self.uv_run_process:
                try:
                    await self._stop_subprocess(self.uv_run_process)
                except Exception as stop_error:
                    logger.error(f"Error stopping failed subprocess: {stop_error}")
                self.uv_run_process = None
            raise RuntimeError(f"Model failed to start: {e}")

        self.current_model = model_info
        return model_info

    async def stop_model(self) -> Optional[str]:
        """
        Stop the currently running model subprocess.

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

        # Stop the subprocess
        if self.uv_run_process:
            try:
                await self._stop_subprocess(self.uv_run_process)
                logger.info(f"Model subprocess stopped: {container_name}")
            except Exception as e:
                logger.error(f"Error stopping subprocess: {e}")
                raise RuntimeError(f"Failed to stop subprocess: {e}")
            finally:
                self.uv_run_process = None

        self.current_model = None
        return model_id

    async def restart_model(self) -> Optional[str]:
        """
        Restart the currently running model subprocess.

        This will stop and restart the same model with the same parameters.

        Returns:
            The model_id that was restarted, or None if no model was running

        Raises:
            RuntimeError: If restart fails
        """
        if not self.current_model:
            logger.warning("No model is running, nothing to restart")
            return None

        # Save current model information
        model_id = self.current_model.model_id
        parameters = self.current_model.parameters
        container_name = self.current_model.container_name

        logger.info(f"Restarting model subprocess: {container_name}")

        try:
            # Stop the current subprocess
            if self.uv_run_process:
                await self._stop_subprocess(self.uv_run_process)
                self.uv_run_process = None
            logger.info(f"Model subprocess stopped: {container_name}")

            # Clear current model reference
            self.current_model = None

            # Restart the model with same parameters
            await self.start_model(model_id, parameters)
            logger.info(f"Model subprocess restarted: {container_name}")

            return model_id

        except Exception as e:
            logger.error(f"Failed to restart subprocess: {e}")
            # If restart fails, ensure current_model and process are cleared
            if self.uv_run_process:
                try:
                    await self._stop_subprocess(self.uv_run_process)
                except Exception:
                    pass
                self.uv_run_process = None
            self.current_model = None
            raise RuntimeError(f"Failed to restart subprocess: {e}")

    async def get_current_model(self) -> Optional[ModelInfo]:
        """Get information about the currently running model"""
        return self.current_model

    async def is_model_running(self) -> bool:
        """Check if a model is currently running"""
        return self.current_model is not None

    async def check_model_health(self) -> bool:
        """
        Check if the current model subprocess is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self.current_model:
            return False

        try:
            url = f"http://localhost:{config.model_port}/health"
            response = await self.http_client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def invoke_inference(
        self,
        task_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke inference on the current model.

        Args:
            task_input: Input data for the model

        Returns:
            Inference result from the model

        Raises:
            RuntimeError: If no model is running or inference fails
        """
        if not self.current_model:
            raise RuntimeError("No model is currently running")

        try:
            url = f"http://localhost:{config.model_port}/inference"
            response = await self.http_client.post(
                url,
                json=task_input,
                timeout=600.0  # 10 minutes timeout for inference
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

    async def _build_docker_image(
        self,
        model_dir: Path,
        image_name: str
    ):
        """
        Build Docker image from Dockerfile.

        Args:
            model_dir: Directory containing the Dockerfile
            image_name: Name and tag for the image

        Raises:
            RuntimeError: If image build fails
        """
        logger.info(f"Building Docker image: {image_name} from {model_dir}")

        cmd = [
            "docker", "build",
            "-t", image_name,
            str(model_dir)
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            logger.error(f"Docker build failed: {error_msg}")
            raise RuntimeError(f"Failed to build Docker image: {error_msg}")

        logger.info(f"Docker image built successfully: {image_name}")

    async def _run_docker_container(
        self,
        container_name: str,
        image_name: str,
        env_vars: Dict[str, str],
        port: int
    ):
        """
        Run Docker container with specified configuration.

        Args:
            container_name: Name for the container
            image_name: Docker image to use
            env_vars: Environment variables
            port: Host port to map to container port 8000

        Raises:
            RuntimeError: If container fails to start
        """
        logger.info(f"Running Docker container: {container_name}")

        # Build docker run command
        cmd = [
            "docker", "run",
            "--name", container_name,
            "--publish", f"{port}:8000",
            "--detach",
            "--restart", "unless-stopped",
            # Add healthcheck configuration
            "--health-cmd", "curl -f http://localhost:8000/health || exit 1",
            "--health-interval", "10s",
            "--health-timeout", "5s",
            "--health-retries", "3",
            "--health-start-period", "10s",
        ]

        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Add image name
        cmd.append(image_name)

        logger.debug(f"Docker run command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            logger.error(f"Docker run failed: {error_msg}")
            raise RuntimeError(f"Failed to run Docker container: {error_msg}")

        logger.info(f"Docker container started: {container_name}")

    async def _stop_docker_container(self, container_name: str):
        """
        Stop and remove a Docker container.

        Args:
            container_name: Name of the container to stop

        Raises:
            RuntimeError: If stop/remove fails
        """
        logger.info(f"Stopping Docker container: {container_name}")

        # Stop container
        stop_cmd = ["docker", "stop", container_name]
        process = await asyncio.create_subprocess_exec(
            *stop_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            if "No such container" not in error_msg:
                logger.error(f"Docker stop failed: {error_msg}")
                raise RuntimeError(f"Failed to stop Docker container: {error_msg}")
            else:
                logger.warning(f"Container {container_name} does not exist")

        # Remove container
        rm_cmd = ["docker", "rm", "-f", container_name]
        process = await asyncio.create_subprocess_exec(
            *rm_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            if "No such container" not in error_msg:
                logger.error(f"Docker rm failed: {error_msg}")
                raise RuntimeError(f"Failed to remove Docker container: {error_msg}")

        logger.info(f"Docker container stopped and removed: {container_name}")

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
        Stop a subprocess cleanly.

        Args:
            process: The subprocess to stop

        Raises:
            RuntimeError: If subprocess fails to stop
        """
        if process is None:
            return

        logger.info(f"Stopping subprocess (PID: {process.pid})")

        # Check if process is still running
        if process.returncode is not None:
            logger.debug(f"Subprocess already finished with return code: {process.returncode}")
            return

        try:
            # Try graceful termination first
            process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info(f"Subprocess terminated gracefully (PID: {process.pid})")
                return
            except asyncio.TimeoutError:
                logger.warning(f"Subprocess did not terminate gracefully, killing (PID: {process.pid})")

            # Force kill if graceful termination didn't work
            process.kill()
            await process.wait()
            logger.info(f"Subprocess killed (PID: {process.pid})")

        except ProcessLookupError:
            # Process already finished
            logger.debug(f"Subprocess already finished (PID: {process.pid})")
        except Exception as e:
            logger.error(f"Error stopping subprocess (PID: {process.pid}): {e}")
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

    async def _force_remove_container(self, container_name: str):
        """
        Force remove a Docker container.

        Args:
            container_name: Name of the container to remove

        Raises:
            RuntimeError: If removal fails
        """
        logger.info(f"Force removing container: {container_name}")

        # Stop container
        stop_cmd = ["docker", "stop", container_name]
        process = await asyncio.create_subprocess_exec(
            *stop_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

        # Remove container
        rm_cmd = ["docker", "rm", "-f", container_name]
        process = await asyncio.create_subprocess_exec(
            *rm_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            # Don't raise error if container doesn't exist
            if "No such container" not in error_msg:
                raise RuntimeError(f"Failed to remove container: {error_msg}")
            else:
                logger.info(f"Container {container_name} does not exist")

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
