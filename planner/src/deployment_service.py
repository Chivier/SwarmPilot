"""Service layer for model deployment to instances."""

import asyncio
import time
from typing import Any, List, Dict, Optional
import httpx
from loguru import logger

from .models import DeploymentStatus, MigrationStatus
from .available_instance_store import AvailableInstance, AvailableInstanceStore


class ModelMapper:
    """Handles mapping between model names and integer IDs."""

    @staticmethod
    def create_mapping(model_names: List[str]) -> Dict[str, int]:
        """
        Create a mapping from model names to integer IDs.

        Args:
            model_names: List of model names (may contain duplicates)

        Returns:
            Dictionary mapping unique model names to IDs (0, 1, 2, ...)
        """
        unique_models = []
        seen = set()

        for name in model_names:
            if name not in seen:
                unique_models.append(name)
                seen.add(name)

        return {name: idx for idx, name in enumerate(unique_models)}

    @staticmethod
    def map_names_to_ids(names: List[str], mapping: Dict[str, int]) -> List[int]:
        """
        Convert model names to IDs using the mapping.

        Args:
            names: List of model names
            mapping: Name -> ID mapping

        Returns:
            List of model IDs

        Raises:
            ValueError: If a name is not in the mapping
        """
        result = []
        for name in names:
            if name not in mapping:
                error_msg = f"Model name '{name}' not found in mapping. Available mappings: {list(mapping.keys())}"
                logger.error(f"Model mapping failed: {error_msg}")
                raise ValueError(error_msg)
            result.append(mapping[name])
        return result

    @staticmethod
    def map_ids_to_names(ids: List[int], reverse_mapping: Dict[int, str]) -> List[str]:
        """
        Convert model IDs to names using reverse mapping.

        Args:
            ids: List of model IDs
            reverse_mapping: ID -> Name mapping

        Returns:
            List of model names

        Raises:
            ValueError: If an ID is not in the mapping
        """
        result = []
        for model_id in ids:
            if model_id not in reverse_mapping:
                error_msg = f"Model ID {model_id} not found in reverse mapping. Available IDs: {list(reverse_mapping.keys())}"
                logger.error(f"ID to name mapping failed: {error_msg}")
                raise ValueError(error_msg)
            result.append(reverse_mapping[model_id])
        return result


class InstanceDeployer:
    """Handles deployment of models to instances."""

    def __init__(
        self,
        timeout: int = 30,
        scheduler_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the deployer.

        Args:
            timeout: HTTP request timeout in seconds
            scheduler_url: Scheduler URL for instance registration (optional)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds (exponential backoff)
        """
        self.timeout = timeout
        self.scheduler_url = scheduler_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error should be retried, False otherwise
        """
        # Retry on connection errors, timeouts, and certain HTTP errors
        if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            # Retry on 5xx errors (server errors) except 501 (Not Implemented)
            status_code = error.response.status_code
            return 500 <= status_code < 600 and status_code != 501

        return False

    async def get_instance_info(self, endpoint: str) -> Optional[Dict]:
        """
        Get current status and model info from an instance.

        Args:
            endpoint: Instance endpoint URL

        Returns:
            Instance info dict, or None if request fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{endpoint}/info")
                response.raise_for_status()
                data = response.json()
                return data.get("instance", {})
        except Exception as e:
            logger.error(f"Failed to get info from {endpoint}: {e}", exc_info=True)
            return None

    async def deploy_model(
        self,
        endpoint: str,
        target_model: str,
        instance_index: int,
        previous_model: Optional[str] = None
    ) -> DeploymentStatus:
        """
        Deploy a model to a single instance.

        Workflow:
        1. Get current model from instance
        2. If different from target, stop current model
        3. Start target model
        4. Record timing and errors

        Args:
            endpoint: Instance endpoint URL
            target_model: Model name to deploy
            instance_index: Index of this instance
            previous_model: Previous model name (if known)

        Returns:
            DeploymentStatus with result
        """
        start_time = time.time()
        error_message = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: Get current model info
                info_response = await client.get(f"{endpoint}/info")
                info_response.raise_for_status()
                info_data = info_response.json()

                current_model_data = info_data.get("instance", {}).get("current_model")
                current_model_id = current_model_data.get("model_id") if current_model_data else None

                # Update previous_model if we got it from the instance
                if previous_model is None and current_model_id:
                    previous_model = current_model_id

                # Step 2: Check if deployment is needed
                if current_model_id == target_model:
                    logger.info(
                        f"Instance {instance_index} ({endpoint}) already running {target_model}, skipping"
                    )
                    deployment_time = time.time() - start_time
                    return DeploymentStatus(
                        instance_index=instance_index,
                        endpoint=endpoint,
                        target_model=target_model,
                        previous_model=previous_model,
                        success=True,
                        error_message=None,
                        deployment_time=deployment_time
                    )

                # Step 3: Stop current model if running (with retry)
                if current_model_id:
                    logger.info(f"Stopping model {current_model_id} on {endpoint}")
                    for attempt in range(self.max_retries):
                        try:
                            stop_response = await client.get(f"{endpoint}/model/stop")
                            stop_response.raise_for_status()
                            break  # Success, exit retry loop
                        except Exception as e:
                            if attempt < self.max_retries - 1 and self._is_retryable_error(e):
                                delay = self.retry_delay * (2 ** attempt)
                                logger.warning(
                                    f"Stop failed (attempt {attempt + 1}/{self.max_retries}), "
                                    f"retrying in {delay}s: {str(e)}"
                                )
                                await asyncio.sleep(delay)
                            else:
                                raise  # Not retryable or max retries exceeded

                # Step 4: Start target model (with retry)
                logger.info(f"Starting model {target_model} on {endpoint}")
                start_payload = {
                    "model_id": target_model,
                    "parameters": {}
                }
                # Add scheduler_url if available
                if self.scheduler_url:
                    start_payload["scheduler_url"] = self.scheduler_url
                    logger.debug(f"Including scheduler_url: {self.scheduler_url}")

                for attempt in range(self.max_retries):
                    try:
                        start_response = await client.post(
                            f"{endpoint}/model/start",
                            json=start_payload
                        )
                        start_response.raise_for_status()
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < self.max_retries - 1 and self._is_retryable_error(e):
                            delay = self.retry_delay * (2 ** attempt)
                            logger.warning(
                                f"Start failed (attempt {attempt + 1}/{self.max_retries}), "
                                f"retrying in {delay}s: {str(e)}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            raise  # Not retryable or max retries exceeded

                deployment_time = time.time() - start_time
                logger.info(
                    f"Successfully deployed {target_model} to instance {instance_index} "
                    f"in {deployment_time:.2f}s"
                )

                return DeploymentStatus(
                    instance_index=instance_index,
                    endpoint=endpoint,
                    target_model=target_model,
                    previous_model=previous_model,
                    success=True,
                    error_message=None,
                    deployment_time=deployment_time
                )

        except httpx.HTTPStatusError as e:
            # Try to extract error message from response JSON
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", error_data.get("detail", e.response.text))
            except Exception:
                error_msg = e.response.text or f"HTTP {e.response.status_code}"

            error_message = f"HTTP {e.response.status_code}: {error_msg}"
            logger.error(f"Failed to deploy to {endpoint}: {error_message}")
        except httpx.ConnectError as e:
            error_message = f"Connection failed: {str(e)}"
            logger.error(f"Failed to connect to {endpoint}: {error_message}")
        except httpx.TimeoutException as e:
            error_message = f"Request timed out after {self.timeout}s"
            logger.error(f"Timeout deploying to {endpoint}: {error_message}")
        except httpx.RequestError as e:
            error_message = f"Request error: {str(e)}"
            logger.error(f"Request failed for {endpoint}: {error_message}")
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to deploy to {endpoint}: {error_message}")

        deployment_time = time.time() - start_time
        return DeploymentStatus(
            instance_index=instance_index,
            endpoint=endpoint,
            target_model=target_model,
            previous_model=previous_model,
            success=False,
            error_message=error_message,
            deployment_time=deployment_time
        )

    async def restart_model(
        self,
        endpoint: str,
        target_model: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initiate a model restart operation on an instance.

        This calls the /model/restart endpoint which performs a graceful restart:
        1. Drains the current scheduler (stops accepting new tasks)
        2. Waits for pending tasks to complete
        3. Stops the current model
        4. Starts the new model
        5. Registers with the scheduler

        Args:
            endpoint: Instance endpoint URL
            target_model: Model name to restart to
            parameters: Optional model parameters

        Returns:
            Dict containing operation_id and initial status

        Raises:
            Exception: If restart initiation fails
        """
        payload = {
            "model_id": target_model,
            "parameters": parameters or {}
        }

        # Add scheduler_url if available
        if self.scheduler_url:
            payload["scheduler_url"] = self.scheduler_url

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{endpoint}/model/restart",
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to initiate restart on {endpoint}: {e}", exc_info=True)
            raise

    async def get_restart_status(
        self,
        endpoint: str,
        operation_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a restart operation.

        Args:
            endpoint: Instance endpoint URL
            operation_id: Restart operation ID

        Returns:
            Dict containing restart status information

        Raises:
            Exception: If status query fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{endpoint}/model/restart/status",
                    params={"operation_id": operation_id}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get restart status from {endpoint}: {e}", exc_info=True)
            raise

    async def deploy_to_instances(
        self,
        endpoints: List[str],
        target_models: List[str],
        previous_models: List[Optional[str]]
    ) -> List[DeploymentStatus]:
        """
        Deploy models to multiple instances concurrently.

        Args:
            endpoints: List of instance endpoint URLs
            target_models: List of target model names (one per instance)
            previous_models: List of previous model names (one per instance)

        Returns:
            List of DeploymentStatus objects
        """
        tasks = [
            self.deploy_model(endpoint, target_model, idx, previous_model)
            for idx, (endpoint, target_model, previous_model)
            in enumerate(zip(endpoints, target_models, previous_models))
        ]

        return await asyncio.gather(*tasks)


    def map_ids_to_names(ids: List[int], reverse_mapping: Dict[int, str]) -> List[str]:
        """
        Convert model IDs to names using reverse mapping.

        Args:
            ids: List of model IDs
            reverse_mapping: ID -> Name mapping

        Returns:
            List of model names

        Raises:
            ValueError: If an ID is not in the mapping
        """
        result = []
        for model_id in ids:
            if model_id not in reverse_mapping:
                raise ValueError(f"Model ID {model_id} not found in reverse mapping")
            result.append(reverse_mapping[model_id])
        return result


class InstanceMigrator:
    """Handles migration of models to instances."""

    def __init__(
        self,
        timeout: int = 30,
        scheduler_mapping: Dict[str, str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the deployer.

        Args:
            timeout: HTTP request timeout in seconds
            scheduler_url: Scheduler URL for instance registration (optional)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries in seconds (exponential backoff)
        """
        self.timeout = timeout
        self.scheduler_mapping = scheduler_mapping
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.

        Args:
            error: The exception to check

        Returns:
            True if the error should be retried, False otherwise
        """
        # Retry on connection errors, timeouts, and certain HTTP errors
        if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            # Retry on 5xx errors (server errors) except 501 (Not Implemented)
            status_code = error.response.status_code
            return 500 <= status_code < 600 and status_code != 501

        return False

    async def get_instance_info(self, endpoint: str) -> Optional[Dict]:
        """
        Get current status and model info from an instance.

        Args:
            endpoint: Instance endpoint URL

        Returns:
            Instance info dict, or None if request fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{endpoint}/info")
                response.raise_for_status()
                data = response.json()
                return data.get("instance", {})
        except Exception as e:
            logger.error(f"Failed to get info from {endpoint}: {e}", exc_info=True)
            return None

    async def migration_model(
        self,
        original_endpoint: str,
        target_endpoint: str,
        instance_index: int
    ) -> MigrationStatus:
        start_time = time.time()
        current_model_id = None
        target_model_id = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: get the information from original endpoint
                try:
                    info_response = await client.get(f"{original_endpoint}/info")
                    info_response.raise_for_status()
                    info_data = info_response.json()
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to get info from original endpoint {original_endpoint}: HTTP {e.response.status_code}"
                    logger.error(f"Migration failed: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error getting info from original endpoint {original_endpoint}: {str(e)}"
                    logger.error(f"Migration failed: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )

                current_model_data = info_data.get("instance", {}).get("current_model")
                current_instance_id = info_data.get("instance", {}).get("instance_id")
                current_model_id = current_model_data.get("model_id") if current_model_data else None

                # Step 2: get the information from target endpoint
                try:
                    info_response = await client.get(f"{target_endpoint}/info")
                    info_response.raise_for_status()
                    info_data = info_response.json()
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to get info from target endpoint {target_endpoint}: HTTP {e.response.status_code}"
                    logger.error(f"Migration failed: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error getting info from target endpoint {target_endpoint}: {str(e)}"
                    logger.error(f"Migration failed: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )

                target_model_data = info_data.get("instance", {}).get("current_model")
                target_instance_id = info_data.get('instance', {}).get("instance_id")
                target_model_id = target_model_data.get("model_id") if target_model_data else None

                if current_model_id == target_model_id:
                    logger.info(f"Instance {current_instance_id} have same model as {target_instance_id}, skip migration")
                    migration_time = time.time() - start_time
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=True,
                        error_message=None,
                        deployment_time=migration_time
                    )

                # Step 3: Validate scheduler_mapping has the target model
                if not self.scheduler_mapping or target_model_id not in self.scheduler_mapping:
                    error_msg = f"No scheduler mapping found for model {target_model_id}. Available models: {list(self.scheduler_mapping.keys()) if self.scheduler_mapping else []}"
                    logger.error(f"Migration failed: {error_msg}")
                    migration_time = time.time() - start_time
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=migration_time
                    )

                scheduler_url = self.scheduler_mapping[target_model_id]
                logger.info(f"Registering {target_endpoint} with model {target_model_id} to scheduler {scheduler_url}")

                payload = {
                    "scheduler_url": scheduler_url
                }

                # Step 4: Deregister original and register new instance in parallel
                try:
                    register_task = asyncio.create_task(
                        client.post(f"{target_endpoint}/model/register", json=payload)
                    )
                    deregister_task = asyncio.create_task(self.deregister_model(original_endpoint))
                    # Get the register and deregister result
                    deregister_response = await deregister_task
                    register_response = await register_task
                    register_response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to register/deregister: HTTP {e.response.status_code} - {e.response.text}"
                    logger.error(f"Migration failed during register/deregister: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error during register/deregister: {str(e)}"
                    logger.error(f"Migration failed during register/deregister: {error_msg}", exc_info=True)
                    return MigrationStatus(
                        instance_index=instance_index,
                        endpoint=original_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time
                    )

            # Now everything is done
            migration_time = time.time() - start_time
            logger.info(
                f"Successfully migrated from {original_endpoint} to {target_endpoint} "
                f"in {migration_time:.2f}s"
            )
            return MigrationStatus(
                instance_index=instance_index,
                endpoint=original_endpoint,
                target_model=target_model_id,
                previous_model=current_model_id,
                success=True,
                error_message=None,
                deployment_time=migration_time
            )
        except Exception as e:
            error_msg = f"Unexpected error during migration: {str(e)}"
            logger.error(f"Migration from {original_endpoint} to {target_endpoint} failed: {error_msg}", exc_info=True)
            return MigrationStatus(
                instance_index=instance_index,
                endpoint=original_endpoint,
                target_model=target_model_id,
                previous_model=current_model_id,
                success=False,
                error_message=error_msg,
                deployment_time=time.time() - start_time
            )
 

    async def deregister_model(
        self,
        endpoint: str,
    ) -> Dict[str, Any]:
        """
        Initiate a model deregister operation on an instance.

        This calls the /model/deregister endpoint which performs a graceful deregister from scheduler:
        1. Just call the deregiter method on the instance, then it will do everything itself.

        Args:
            endpoint: Instance endpoint URL

        Returns:
            Dict containing operation_id and initial status

        Raises:
            Exception: If restart initiation fails
        """

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{endpoint}/model/deregister",
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to deregister model on {endpoint}: {e}", exc_info=True)
            raise

    async def migration_instances(
        self,
        original_endpoints: List[str],
        target_endpoints: List[str],
    ) -> List[MigrationStatus]:
        """
        Migrate models across multiple instances concurrently.

        Args:
            original_endpoints: List of original instance endpoint URLs
            target_endpoints: List of target instance endpoint URLs (one per instance)

        Returns:
            List of MigrationStatus objects
        """
        tasks = [
            self.migration_model(original, target, idx)
            for idx, (original, target)
            in enumerate(zip(original_endpoints, target_endpoints))
        ]

        return await asyncio.gather(*tasks)
