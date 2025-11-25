"""Health monitoring for services and models."""

import time
import requests
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from loguru import logger as loguru_logger


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthChecker:
    """
    Monitors health of services and models.

    Supports:
    - Periodic health checks
    - HTTP endpoint monitoring
    - Custom health check functions
    """

    def __init__(
        self,
        check_interval: float = 10.0,
        timeout: float = 5.0,
        custom_logger: Optional[Any] = None
    ):
        """
        Initialize health checker.

        Args:
            check_interval: Time between health checks
            timeout: Request timeout
            custom_logger: Optional custom logger (defaults to loguru logger)
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.logger = custom_logger or loguru_logger

        # Track health status
        self.health_status: Dict[str, HealthStatus] = {}

    def check_http_endpoint(
        self,
        name: str,
        url: str,
        expected_status: int = 200
    ) -> HealthStatus:
        """
        Check health via HTTP endpoint.

        Args:
            name: Service/model name
            url: Health check URL
            expected_status: Expected HTTP status code

        Returns:
            HealthStatus enum value
        """
        try:
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == expected_status:
                self.health_status[name] = HealthStatus.HEALTHY
                return HealthStatus.HEALTHY
            else:
                self.logger.warning(
                    f"{name} unhealthy: status code {response.status_code}"
                )
                self.health_status[name] = HealthStatus.UNHEALTHY
                return HealthStatus.UNHEALTHY

        except Exception as e:
            self.logger.error(f"{name} health check failed: {e}")
            self.health_status[name] = HealthStatus.UNHEALTHY
            return HealthStatus.UNHEALTHY

    def check_scheduler(self, name: str, scheduler_url: str) -> HealthStatus:
        """
        Check scheduler health.

        Args:
            name: Scheduler name
            scheduler_url: Scheduler base URL

        Returns:
            HealthStatus enum value
        """
        # Check /info endpoint
        return self.check_http_endpoint(
            name,
            f"{scheduler_url}/info"
        )

    def check_model_instances(
        self,
        model_id: str,
        scheduler_url: str,
        min_instances: int = 1
    ) -> HealthStatus:
        """
        Check if model has sufficient healthy instances.

        Args:
            model_id: Model identifier
            scheduler_url: Scheduler URL
            min_instances: Minimum required instances

        Returns:
            HealthStatus enum value
        """
        try:
            response = requests.get(
                f"{scheduler_url}/model/query",
                params={"model_id": model_id},
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            instances = data.get("instances", [])
            active_count = len(instances)

            if active_count >= min_instances:
                self.health_status[model_id] = HealthStatus.HEALTHY
                return HealthStatus.HEALTHY
            else:
                self.logger.warning(
                    f"{model_id} has {active_count} instances (need {min_instances})"
                )
                self.health_status[model_id] = HealthStatus.UNHEALTHY
                return HealthStatus.UNHEALTHY

        except Exception as e:
            self.logger.error(f"Failed to check {model_id} instances: {e}")
            self.health_status[model_id] = HealthStatus.UNHEALTHY
            return HealthStatus.UNHEALTHY

    def check_all(
        self,
        checks: Dict[str, Callable[[], HealthStatus]]
    ) -> Dict[str, HealthStatus]:
        """
        Run multiple health checks.

        Args:
            checks: Dict mapping name to health check function

        Returns:
            Dict mapping name to health status
        """
        results = {}

        for name, check_fn in checks.items():
            try:
                status = check_fn()
                results[name] = status
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthStatus.UNHEALTHY

        return results

    def get_status(self, name: str) -> HealthStatus:
        """
        Get cached health status.

        Args:
            name: Service/model name

        Returns:
            HealthStatus enum value
        """
        return self.health_status.get(name, HealthStatus.UNKNOWN)

    def get_all_status(self) -> Dict[str, HealthStatus]:
        """
        Get all cached health statuses.

        Returns:
            Dict mapping name to health status
        """
        return dict(self.health_status)

    def is_healthy(self, name: str) -> bool:
        """
        Check if service/model is healthy.

        Args:
            name: Service/model name

        Returns:
            True if healthy
        """
        return self.health_status.get(name) == HealthStatus.HEALTHY

    def wait_for_healthy(
        self,
        name: str,
        check_fn: Callable[[], HealthStatus],
        timeout: int = 60
    ) -> bool:
        """
        Wait for a service/model to become healthy.

        Args:
            name: Service/model name
            check_fn: Health check function
            timeout: Maximum time to wait

        Returns:
            True if became healthy within timeout
        """
        start_time = time.time()

        self.logger.info(f"Waiting for {name} to become healthy...")

        while (time.time() - start_time) < timeout:
            status = check_fn()

            if status == HealthStatus.HEALTHY:
                self.logger.info(f"{name} is healthy")
                return True

            time.sleep(self.check_interval)

        self.logger.error(f"Timeout waiting for {name} to become healthy")
        return False
