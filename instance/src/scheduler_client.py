"""
Scheduler Client for Instance Service

This module provides functionality for instances to communicate with the scheduler,
including registration, deregistration, and task result callbacks.
"""

import os
import platform
import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime


class SchedulerClient:
    """Client for communicating with the scheduler service."""

    def __init__(
        self,
        scheduler_url: Optional[str] = None,
        instance_id: Optional[str] = None,
        instance_endpoint: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize scheduler client.

        Args:
            scheduler_url: Base URL of the scheduler service (e.g., http://localhost:8000)
            instance_id: Unique identifier for this instance
            instance_endpoint: HTTP endpoint where this instance is accessible
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.scheduler_url = scheduler_url or os.getenv("SCHEDULER_URL")
        self.instance_id = instance_id or os.getenv("INSTANCE_ID", "instance-default")
        self.instance_endpoint = instance_endpoint or os.getenv(
            "INSTANCE_ENDPOINT",
            f"http://localhost:{os.getenv('INSTANCE_PORT', '5000')}"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._registered = False
        print(f"Starting scheduler client with endpoint: {self.scheduler_url}")

    @property
    def is_enabled(self) -> bool:
        """Check if scheduler integration is enabled."""
        return self.scheduler_url is not None

    async def register_instance(
        self,
        model_id: str,
        platform_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register this instance with the scheduler.

        Args:
            model_id: ID of the model this instance is serving
            platform_info: Platform information (auto-detected if not provided)

        Returns:
            True if registration successful, False otherwise
        """
        if not self.is_enabled:
            print("Scheduler integration disabled (SCHEDULER_URL not set)")
            return False

        # Auto-detect platform info if not provided
        if platform_info is None:
            platform_info = self._get_platform_info()

        registration_data = {
            "instance_id": self.instance_id,
            "model_id": model_id,
            "endpoint": self.instance_endpoint,
            "platform_info": platform_info,
        }

        # Retry logic for registration
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.scheduler_url}/instance/register",
                        json=registration_data,
                    )
                    response.raise_for_status()
                    result = response.json()

                if result.get("success", False):
                    self._registered = True
                    print(
                        f"Instance {self.instance_id} registered with scheduler successfully"
                    )
                    return True
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"Registration failed: {error_msg}")
                    return False

            except httpx.HTTPError as e:
                print(
                    f"Registration attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"Failed to register instance after {self.max_retries} attempts")
                    return False

            except Exception as e:
                print(f"Unexpected error during registration: {str(e)}")
                return False

        return False

    async def deregister_instance(self) -> bool:
        """
        Deregister this instance from the scheduler.

        Returns:
            True if deregistration successful, False otherwise
        """
        if not self.is_enabled or not self._registered:
            return False

        deregistration_data = {
            "instance_id": self.instance_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/remove",
                    json=deregistration_data,
                )
                response.raise_for_status()
                result = response.json()

            if result.get("success", False):
                self._registered = False
                print(f"Instance {self.instance_id} deregistered from scheduler")
                return True
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"Deregistration failed: {error_msg}")
                return False

        except httpx.HTTPError as e:
            print(f"Failed to deregister instance: {str(e)}")
            return False

        except Exception as e:
            print(f"Unexpected error during deregistration: {str(e)}")
            return False

    async def drain_instance(self) -> Dict[str, Any]:
        """
        Request the scheduler to drain this instance (stop assigning new tasks).

        Returns:
            Dictionary with drain status information including:
            - success: bool
            - status: str (should be "draining" after successful call)
            - pending_tasks: int
            - estimated_completion_time_ms: Optional[float]

        Raises:
            Exception: If drain request fails
        """
        if not self.is_enabled:
            raise Exception("Scheduler integration disabled (SCHEDULER_URL not set)")

        drain_data = {
            "instance_id": self.instance_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.scheduler_url}/instance/drain",
                    json=drain_data,
                )
                response.raise_for_status()
                result = response.json()

            if result.get("success", False):
                print(f"Instance {self.instance_id} is now draining")
                return result
            else:
                error_msg = result.get("error", "Unknown error")
                raise Exception(f"Drain request failed: {error_msg}")

        except httpx.HTTPError as e:
            raise Exception(f"Failed to drain instance: {str(e)}")

        except Exception as e:
            raise Exception(f"Unexpected error during drain: {str(e)}")

    async def send_task_result(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        callback_url: Optional[str] = None,
    ) -> bool:
        """
        Send task result to scheduler via callback.

        Args:
            task_id: ID of the completed task
            status: Task status ("completed" or "failed")
            result: Task result data (if completed)
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
            callback_url: Custom callback URL (defaults to scheduler's callback endpoint)

        Returns:
            True if callback successful, False otherwise
        """
        if not callback_url:
            if not self.is_enabled:
                return False
            callback_url = f"{self.scheduler_url}/callback/task_result"

        callback_data = {
            "task_id": task_id,
            "status": status,
        }

        if result is not None:
            callback_data["result"] = result
        if error is not None:
            callback_data["error"] = error
        if execution_time_ms is not None:
            callback_data["execution_time_ms"] = execution_time_ms

        # Retry logic for callback
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        callback_url,
                        json=callback_data,
                    )
                    response.raise_for_status()
                    result_data = response.json()

                if result_data.get("success", False):
                    return True
                else:
                    error_msg = result_data.get("error", "Unknown error")
                    print(f"Callback failed: {error_msg}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        return False

            except httpx.HTTPError as e:
                print(
                    f"Callback attempt {attempt + 1}/{self.max_retries} failed for task {task_id}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    print(f"Failed to send callback after {self.max_retries} attempts")
                    return False

            except Exception as e:
                print(f"Unexpected error during callback: {str(e)}")
                return False

        return False

    def _get_platform_info(self) -> Dict[str, Any]:
        """
        Auto-detect platform information.

        Returns:
            Dictionary containing platform information
        """
        return {
            "software_name": platform.system(),  # e.g., "Linux", "Darwin", "Windows"
            "software_version": platform.release(),  # e.g., "5.15.0-151-generic"
            "hardware_name": platform.machine(),  # e.g., "x86_64", "arm64"
        }


# Global scheduler client instance
_scheduler_client: Optional[SchedulerClient] = None


def get_scheduler_client() -> SchedulerClient:
    """
    Get the global scheduler client instance.

    Returns:
        SchedulerClient instance
    """
    global _scheduler_client
    if _scheduler_client is None:
        _scheduler_client = SchedulerClient()
    return _scheduler_client


def initialize_scheduler_client(
    scheduler_url: Optional[str] = None,
    instance_id: Optional[str] = None,
    instance_endpoint: Optional[str] = None,
) -> SchedulerClient:
    """
    Initialize and configure the global scheduler client.

    Args:
        scheduler_url: Base URL of the scheduler service
        instance_id: Unique identifier for this instance
        instance_endpoint: HTTP endpoint where this instance is accessible

    Returns:
        Configured SchedulerClient instance
    """
    global _scheduler_client
    _scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url,
        instance_id=instance_id,
        instance_endpoint=instance_endpoint,
    )
    return _scheduler_client
