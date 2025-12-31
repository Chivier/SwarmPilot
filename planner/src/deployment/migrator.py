"""Instance migration service."""

import asyncio
import time
import traceback
from typing import Any

import httpx
from loguru import logger

from ..models import MigrationStatus


class InstanceMigrator:
    """Handles migration of models to instances."""

    def __init__(
        self,
        timeout: int = 30,
        scheduler_mapping: dict[str, str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the migrator.

        Args:
            timeout: HTTP request timeout in seconds.
            scheduler_mapping: Mapping of model names to scheduler URLs.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Initial delay between retries (exponential backoff).
        """
        self.timeout = timeout
        self.scheduler_mapping = scheduler_mapping
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable.

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

    async def get_instance_info(self, endpoint: str) -> dict | None:
        """Get current status and model info from an instance.

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
            logger.error(
                f"Failed to get info from {endpoint}: {e}", exc_info=True
            )
            return None

    async def migration_model(
        self, original_endpoint: str, target_endpoint: str, instance_index: int
    ) -> MigrationStatus:
        """Migrate a model from one instance to another.

        Args:
            original_endpoint: Source instance endpoint URL.
            target_endpoint: Destination instance endpoint URL.
            instance_index: Index of this migration in the batch.

        Returns:
            MigrationStatus with success/failure information.
        """
        start_time = time.time()
        current_model_id = None
        target_model_id = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Step 1: get the information from original endpoint
                try:
                    info_response = await client.get(
                        f"{original_endpoint}/info"
                    )
                    info_response.raise_for_status()
                    info_data = info_response.json()
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to get info from original endpoint {original_endpoint}: HTTP {e.response.status_code}"
                    logger.error(
                        f"Migration failed: {error_msg}", exc_info=True
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error getting info from original endpoint {original_endpoint}: {str(e)}"
                    logger.error(
                        f"Migration failed: {error_msg}", exc_info=True
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )

                current_model_data = info_data.get("instance", {}).get(
                    "current_model"
                )
                current_instance_id = info_data.get("instance", {}).get(
                    "instance_id"
                )
                current_model_id = (
                    current_model_data.get("model_id")
                    if current_model_data
                    else None
                )

                # Step 2: get the information from target endpoint
                try:
                    info_response = await client.get(f"{target_endpoint}/info")
                    info_response.raise_for_status()
                    info_data = info_response.json()
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to get info from target endpoint {target_endpoint}: HTTP {e.response.status_code}"
                    logger.error(
                        f"Migration failed: {error_msg}", exc_info=True
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error getting info from target endpoint {target_endpoint}: {str(e)}"
                    logger.error(
                        f"Migration failed: {error_msg}", exc_info=True
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )

                target_model_data = info_data.get("instance", {}).get(
                    "current_model"
                )
                target_instance_id = info_data.get("instance", {}).get(
                    "instance_id"
                )
                target_model_id = (
                    target_model_data.get("model_id")
                    if target_model_data
                    else None
                )

                if current_model_id == target_model_id:
                    logger.info(
                        f"Instance {current_instance_id} have same model as {target_instance_id}, skip migration"
                    )
                    migration_time = time.time() - start_time
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=True,
                        error_message=None,
                        deployment_time=migration_time,
                    )

                # Step 3: Validate scheduler_mapping has the target model
                if (
                    not self.scheduler_mapping
                    or target_model_id not in self.scheduler_mapping
                ):
                    error_msg = f"No scheduler mapping found for model {target_model_id}. Available models: {list(self.scheduler_mapping.keys()) if self.scheduler_mapping else []}"
                    logger.error(f"Migration failed: {error_msg}")
                    migration_time = time.time() - start_time
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=migration_time,
                    )

                scheduler_url = self.scheduler_mapping[target_model_id]
                logger.info(
                    f"Migrating from {original_endpoint} to {target_endpoint} with model {target_model_id}"
                )

                payload = {"scheduler_url": scheduler_url}

                # Step 4: Deregister original endpoint (no timeout)
                try:
                    logger.info(f"Deregistering {original_endpoint}")
                    await self.deregister_model(original_endpoint)
                    logger.info(f"Deregister completed for {original_endpoint}")
                except Exception as e:
                    error_msg = (
                        f"Failed to deregister {original_endpoint}: {str(e)}"
                    )
                    logger.error(
                        f"Migration failed during deregister: {error_msg}",
                        exc_info=True,
                    )
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )

                # Step 5: Register new instance immediately after deregister completes
                try:
                    logger.info(
                        f"Registering {target_endpoint} after deregister"
                    )
                    register_response = await client.post(
                        f"{target_endpoint}/model/register", json=payload
                    )
                    register_response.raise_for_status()
                    logger.info(f"Register completed for {target_endpoint}")
                except httpx.HTTPStatusError as e:
                    error_msg = f"Failed to register {target_endpoint}: HTTP {e.response.status_code} - {e.response.text}"
                    logger.error(
                        f"Migration failed during register: {error_msg}",
                        exc_info=True,
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )
                except httpx.RequestError as e:
                    error_msg = f"Request error during register {target_endpoint}: {str(e)}"
                    logger.error(
                        f"Migration failed during register: {error_msg}",
                        exc_info=True,
                    )
                    return MigrationStatus(
                        instance_index=instance_index,
                        original_endpoint=original_endpoint,
                        target_endpoint=target_endpoint,
                        target_model=target_model_id,
                        previous_model=current_model_id,
                        success=False,
                        error_message=error_msg,
                        deployment_time=time.time() - start_time,
                    )

            # Now everything is done
            migration_time = time.time() - start_time
            logger.info(
                f"Successfully migrated from {original_endpoint} to {target_endpoint} "
                f"in {migration_time:.2f}s"
            )
            return MigrationStatus(
                instance_index=instance_index,
                original_endpoint=original_endpoint,
                target_endpoint=target_endpoint,
                target_model=target_model_id,
                previous_model=current_model_id,
                success=True,
                error_message=None,
                deployment_time=migration_time,
            )
        except Exception as e:
            error_msg = f"Unexpected error during migration: {str(e)}"
            logger.error(
                f"Migration from {original_endpoint} to {target_endpoint} failed: {error_msg}",
                exc_info=True,
            )
            return MigrationStatus(
                instance_index=instance_index,
                original_endpoint=original_endpoint,
                target_endpoint=target_endpoint,
                target_model=target_model_id,
                previous_model=current_model_id,
                success=False,
                error_message=error_msg,
                deployment_time=time.time() - start_time,
            )

    async def deregister_model(
        self,
        endpoint: str,
    ) -> dict[str, Any]:
        """Initiate a model deregister operation on an instance.

        This calls the /model/deregister endpoint which performs a graceful deregister from scheduler:
        1. Just call the deregister method on the instance, then it will do everything itself.

        Note: No timeout is set for deregister operations to allow graceful completion.

        Args:
            endpoint: Instance endpoint URL

        Returns:
            Dict containing operation_id and initial status

        Raises:
            Exception: If deregister initiation fails
        """
        try:
            # No timeout for deregister - allow graceful completion
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint}/model/deregister", timeout=None
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(
                f"Failed to deregister model on {endpoint}: {e}", exc_info=True
            )
            raise

    async def migration_instances(
        self,
        original_endpoints: list[str],
        target_endpoints: list[str],
    ) -> list[MigrationStatus]:
        """Migrate models across multiple instances concurrently.

        Args:
            original_endpoints: List of original instance endpoint URLs
            target_endpoints: List of target instance endpoint URLs (one per instance)

        Returns:
            List of MigrationStatus objects
        """
        tasks = [
            self.migration_model(original, target, idx)
            for idx, (original, target) in enumerate(
                zip(original_endpoints, target_endpoints)
            )
        ]

        return await asyncio.gather(*tasks)
