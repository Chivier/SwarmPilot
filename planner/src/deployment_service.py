"""Service layer for model deployment to instances."""

import time
from typing import List, Dict, Optional
import httpx
from loguru import logger

from .models import DeploymentStatus


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
                raise ValueError(f"Model name '{name}' not found in mapping")
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
                raise ValueError(f"Model ID {model_id} not found in reverse mapping")
            result.append(reverse_mapping[model_id])
        return result


class InstanceDeployer:
    """Handles deployment of models to instances."""

    def __init__(self, timeout: int = 30):
        """
        Initialize the deployer.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout

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
            logger.error(f"Failed to get info from {endpoint}: {e}")
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

                # Step 3: Stop current model if running
                if current_model_id:
                    logger.info(f"Stopping model {current_model_id} on {endpoint}")
                    stop_response = await client.get(f"{endpoint}/model/stop")
                    stop_response.raise_for_status()

                # Step 4: Start target model
                logger.info(f"Starting model {target_model} on {endpoint}")
                start_response = await client.post(
                    f"{endpoint}/model/start",
                    json={"model_id": target_model, "parameters": {}}
                )
                start_response.raise_for_status()

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
            error_message = f"HTTP {e.response.status_code}: {e.response.text}"
            logger.error(f"Failed to deploy to {endpoint}: {error_message}")
        except httpx.RequestError as e:
            error_message = f"Request failed: {str(e)}"
            logger.error(f"Failed to deploy to {endpoint}: {error_message}")
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
        import asyncio

        tasks = [
            self.deploy_model(endpoint, target_model, idx, previous_model)
            for idx, (endpoint, target_model, previous_model)
            in enumerate(zip(endpoints, target_models, previous_models))
        ]

        return await asyncio.gather(*tasks)
