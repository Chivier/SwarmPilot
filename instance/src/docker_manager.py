"""
Docker container management for model containers
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


class DockerManager:
    """
    Manages Docker containers for models.

    Handles starting, stopping, and health checking of model containers.
    """

    def __init__(self):
        self.current_model: Optional[ModelInfo] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def start_model(
        self,
        model_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Start a model container.

        Args:
            model_id: The model identifier from the registry
            parameters: Optional model-specific parameters

        Returns:
            ModelInfo object with container details

        Raises:
            ValueError: If model not found in registry
            RuntimeError: If container fails to start
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

        # Check if Dockerfile exists
        dockerfile = model_dir / "Dockerfile"
        if not dockerfile.exists():
            raise ValueError(f"Dockerfile not found in {model_dir}")

        # Generate container and image names
        container_name = config.get_model_container_name(model_id)
        image_name = self._get_image_name(model_id)

        # Build environment variables
        env_vars = self._build_env_vars(model_id, parameters or {})

        logger.info(f"Starting model container: {container_name}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Container port: {config.model_port}")
        logger.info(f"Image name: {image_name}")

        # Build Docker image
        try:
            await self._build_docker_image(model_dir, image_name)
        except Exception as e:
            raise RuntimeError(f"Failed to build Docker image: {e}")

        # Run Docker container
        try:
            await self._run_docker_container(
                container_name,
                image_name,
                env_vars,
                config.model_port
            )
            logger.info(f"Docker container started: {container_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to start Docker container: {e}")

        # Wait for container to be healthy
        try:
            await self._wait_for_health(config.model_port)
        except Exception as e:
            # If health check fails, try to stop the container
            logger.error(f"Health check failed, stopping container: {e}")
            try:
                await self._stop_docker_container(container_name)
            except Exception:
                pass
            raise

        # Create model info
        from datetime import UTC, datetime
        model_info = ModelInfo(
            model_id=model_id,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            parameters=parameters or {},
            container_name=container_name
        )

        self.current_model = model_info
        return model_info

    async def stop_model(self) -> Optional[str]:
        """
        Stop the currently running model container.

        Returns:
            The model_id that was stopped, or None if no model was running

        Raises:
            RuntimeError: If container fails to stop
        """
        if not self.current_model:
            return None

        model_id = self.current_model.model_id
        container_name = self.current_model.container_name

        logger.info(f"Stopping model container: {container_name}")

        # Stop and remove container using docker commands
        try:
            await self._stop_docker_container(container_name)
            logger.info(f"Docker container stopped: {container_name}")
        except Exception as e:
            logger.error(f"Error stopping container: {e}")
            # Try to force remove the container
            try:
                await self._force_remove_container(container_name)
            except Exception as e2:
                logger.error(f"Failed to force remove container: {e2}")
                raise RuntimeError(f"Failed to stop container: {e}")

        self.current_model = None
        return model_id

    async def restart_model(self) -> Optional[str]:
        """
        Restart the currently running model container.

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

        logger.info(f"Restarting model container: {container_name}")

        try:
            # Stop the current container
            await self._stop_docker_container(container_name)
            logger.info(f"Docker container stopped: {container_name}")

            # Clear current model reference
            self.current_model = None

            # Restart the model with same parameters
            await self.start_model(model_id, parameters)
            logger.info(f"Docker container restarted: {container_name}")

            return model_id

        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            # If restart fails, ensure current_model is cleared
            self.current_model = None
            raise RuntimeError(f"Failed to restart container: {e}")

    async def get_current_model(self) -> Optional[ModelInfo]:
        """Get information about the currently running model"""
        return self.current_model

    async def is_model_running(self) -> bool:
        """Check if a model is currently running"""
        return self.current_model is not None

    async def check_model_health(self) -> bool:
        """
        Check if the current model container is healthy.

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
        Build environment variables for the container.

        Converts parameters to environment variables with MODEL_ prefix.
        """
        env_vars = {
            "MODEL_ID": model_id,
            "INSTANCE_ID": config.instance_id,
            "LOG_LEVEL": config.log_level,
        }

        # Convert parameters to environment variables
        for key, value in parameters.items():
            env_key = f"MODEL_{key.upper()}"
            if isinstance(value, (dict, list)):
                env_vars[env_key] = json.dumps(value)
            else:
                env_vars[env_key] = str(value)

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
            # Add healthcheck configuration
            "--health-cmd", "wget -q --spider http://127.0.0.1:8000/health || exit 1",
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

    async def _wait_for_health(
        self,
        port: int,
        timeout: int = 30,
        interval: int = 2
    ):
        """
        Wait for the model container to become healthy.

        Args:
            port: Port to check
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Raises:
            RuntimeError: If container doesn't become healthy within timeout
        """
        url = f"http://localhost:{port}/health"
        elapsed = 0

        while elapsed < timeout:
            try:
                response = await self.http_client.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info("Model container is healthy")
                        return
            except Exception as e:
                logger.debug(f"Health check attempt failed: {e}")

            await asyncio.sleep(interval)
            elapsed += interval

        raise RuntimeError(
            f"Model container did not become healthy within {timeout} seconds"
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


# Global docker manager instance
_docker_manager: Optional[DockerManager] = None


def get_docker_manager() -> DockerManager:
    """Get or create the global docker manager instance"""
    global _docker_manager
    if _docker_manager is None:
        _docker_manager = DockerManager()
    return _docker_manager
