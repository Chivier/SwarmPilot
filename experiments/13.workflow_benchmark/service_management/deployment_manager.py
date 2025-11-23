"""Model deployment manager with parallel deployment and retry logic."""

import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a model deployment."""
    model_id: str
    instances: List[str]  # List of instance URLs
    metadata: Optional[Dict] = None


class DeploymentManager:
    """
    Manages model deployment to schedulers.

    Supports:
    - Parallel deployment to multiple schedulers
    - Retry logic with exponential backoff
    - Health checking after deployment
    """

    def __init__(
        self,
        max_workers: int = 10,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 30.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize deployment manager.

        Args:
            max_workers: Maximum parallel deployment threads
            max_retries: Maximum retry attempts per deployment
            retry_delay: Base delay between retries (exponential backoff)
            timeout: Request timeout in seconds
            logger: Optional logger instance
        """
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

    def deploy_model(
        self,
        scheduler_url: str,
        model_config: ModelConfig,
        endpoint: str = "/model/start"
    ) -> bool:
        """
        Deploy a single model to a scheduler with retry logic.

        Args:
            scheduler_url: Scheduler base URL
            model_config: Model configuration
            endpoint: API endpoint for deployment

        Returns:
            True if deployment successful

        Raises:
            Exception: If all retries fail
        """
        url = f"{scheduler_url}{endpoint}"

        payload = {
            "model_id": model_config.model_id,
            "instances": model_config.instances
        }

        if model_config.metadata:
            payload["metadata"] = model_config.metadata

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self.logger.info(
                    f"Deploying {model_config.model_id} to {scheduler_url} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()

                self.logger.info(
                    f"Successfully deployed {model_config.model_id} to {scheduler_url}"
                )
                return True

            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(
                    f"Deployment attempt {attempt + 1} failed for {model_config.model_id}: {e}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)

        # All retries failed
        error_msg = f"Failed to deploy {model_config.model_id} after {self.max_retries} attempts"
        self.logger.error(f"{error_msg}: {last_exception}")
        raise Exception(error_msg) from last_exception

    def deploy_models_parallel(
        self,
        scheduler_url: str,
        models: List[ModelConfig],
        endpoint: str = "/model/start"
    ) -> Dict[str, bool]:
        """
        Deploy multiple models in parallel.

        Args:
            scheduler_url: Scheduler base URL
            models: List of model configurations
            endpoint: API endpoint for deployment

        Returns:
            Dict mapping model_id to deployment success (True/False)
        """
        self.logger.info(
            f"Deploying {len(models)} models to {scheduler_url} in parallel "
            f"(max_workers={self.max_workers})"
        )

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all deployment tasks
            future_to_model = {
                executor.submit(
                    self.deploy_model,
                    scheduler_url,
                    model,
                    endpoint
                ): model.model_id
                for model in models
            }

            # Collect results
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    success = future.result()
                    results[model_id] = success
                except Exception as e:
                    self.logger.error(f"Deployment failed for {model_id}: {e}")
                    results[model_id] = False

        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        self.logger.info(
            f"Deployment complete: {successful}/{total} successful"
        )

        return results

    def undeploy_model(
        self,
        scheduler_url: str,
        model_id: str,
        endpoint: str = "/model/stop"
    ) -> bool:
        """
        Undeploy a model from scheduler.

        Args:
            scheduler_url: Scheduler base URL
            model_id: Model identifier
            endpoint: API endpoint for undeployment

        Returns:
            True if successful
        """
        url = f"{scheduler_url}{endpoint}"

        try:
            self.logger.info(f"Undeploying {model_id} from {scheduler_url}")

            response = requests.post(
                url,
                json={"model_id": model_id},
                timeout=self.timeout
            )

            response.raise_for_status()

            self.logger.info(f"Successfully undeployed {model_id}")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to undeploy {model_id}: {e}")
            return False

    def check_model_health(
        self,
        scheduler_url: str,
        model_id: str,
        endpoint: str = "/model/query"
    ) -> bool:
        """
        Check if a model is healthy and available.

        Args:
            scheduler_url: Scheduler base URL
            model_id: Model identifier
            endpoint: API endpoint for health check

        Returns:
            True if model is healthy
        """
        url = f"{scheduler_url}{endpoint}"

        try:
            response = requests.get(
                url,
                params={"model_id": model_id},
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            # Check if model has active instances
            instances = data.get("instances", [])
            if not instances:
                self.logger.warning(f"No instances available for {model_id}")
                return False

            self.logger.info(
                f"Model {model_id} is healthy ({len(instances)} instances)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Health check failed for {model_id}: {e}")
            return False

    def wait_for_models(
        self,
        scheduler_url: str,
        model_ids: List[str],
        timeout: int = 60,
        check_interval: float = 2.0
    ) -> bool:
        """
        Wait for all models to become healthy.

        Args:
            scheduler_url: Scheduler base URL
            model_ids: List of model IDs to check
            timeout: Maximum time to wait
            check_interval: Time between checks

        Returns:
            True if all models are healthy within timeout
        """
        start_time = time.time()

        pending = set(model_ids)

        self.logger.info(
            f"Waiting for {len(pending)} models to become healthy "
            f"(timeout={timeout}s)..."
        )

        while pending and (time.time() - start_time) < timeout:
            for model_id in list(pending):
                if self.check_model_health(scheduler_url, model_id):
                    pending.remove(model_id)
                    self.logger.info(
                        f"{model_id} is ready ({len(model_ids) - len(pending)}/{len(model_ids)})"
                    )

            if pending:
                time.sleep(check_interval)

        if pending:
            self.logger.error(
                f"Timeout waiting for models: {', '.join(pending)}"
            )
            return False

        self.logger.info("All models are healthy")
        return True
