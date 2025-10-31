"""
Docker container management for model containers
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .config import config
from .model_registry import get_registry
from .models import ModelInfo

logger = logging.getLogger(__name__)


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

        # Check if docker-compose.yaml exists
        docker_compose_file = model_dir / "docker-compose.yaml"
        if not docker_compose_file.exists():
            raise ValueError(f"docker-compose.yaml not found in {model_dir}")

        # Generate container name
        container_name = config.get_model_container_name(model_id)

        # Build environment variables
        env_vars = self._build_env_vars(model_id, parameters or {})

        # Add container-specific environment variables
        env_vars["CONTAINER_NAME"] = container_name
        env_vars["MODEL_PORT"] = str(config.model_port)

        logger.info(f"Starting model container: {container_name}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Container port: {config.model_port}")

        # Start Docker container using docker-compose
        try:
            await self._run_docker_compose(
                model_dir,
                ["up", "-d"],
                env_vars
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
                await self._run_docker_compose(model_dir, ["down"], env_vars)
            except Exception:
                pass
            raise

        # Create model info
        from datetime import datetime
        model_info = ModelInfo(
            model_id=model_id,
            started_at=datetime.utcnow().isoformat() + "Z",
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

        # Get model directory
        registry = get_registry()
        model_dir = registry.get_model_directory(model_id)

        if model_dir and model_dir.exists():
            # Build environment variables (needed for docker-compose)
            env_vars = self._build_env_vars(model_id, self.current_model.parameters)
            env_vars["CONTAINER_NAME"] = container_name
            env_vars["MODEL_PORT"] = str(config.model_port)

            # Stop and remove container using docker-compose
            try:
                await self._run_docker_compose(
                    model_dir,
                    ["down", "--volumes", "--remove-orphans"],
                    env_vars
                )
                logger.info(f"Docker container stopped: {container_name}")
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
                # Try to force remove the container
                try:
                    await self._force_remove_container(container_name)
                except Exception as e2:
                    logger.error(f"Failed to force remove container: {e2}")
                    raise RuntimeError(f"Failed to stop container: {e}")
        else:
            logger.warning(f"Model directory not found, attempting force removal")
            try:
                await self._force_remove_container(container_name)
            except Exception as e:
                logger.error(f"Failed to force remove container: {e}")

        self.current_model = None
        return model_id

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
                timeout=300.0  # 5 minutes timeout for inference
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

    async def _run_docker_compose(
        self,
        working_dir: Path,
        args: list,
        env_vars: Dict[str, str]
    ):
        """
        Run docker-compose command.

        Args:
            working_dir: Directory containing docker-compose.yaml
            args: Arguments for docker-compose (e.g., ["up", "-d"])
            env_vars: Environment variables to set

        Raises:
            RuntimeError: If docker-compose command fails
        """
        # Prepare environment
        env = os.environ.copy()
        env.update(env_vars)

        # Build command
        cmd = ["docker-compose"] + args

        logger.info(f"Running docker-compose in {working_dir}: {' '.join(cmd)}")

        # Run command asynchronously
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(working_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            logger.error(f"docker-compose failed: {error_msg}")
            raise RuntimeError(f"docker-compose command failed: {error_msg}")

        if stdout:
            logger.debug(f"docker-compose output: {stdout.decode().strip()}")

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
