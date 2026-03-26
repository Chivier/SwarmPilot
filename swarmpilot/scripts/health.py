"""Health monitoring for PyLet-managed models (PYLET-004).

This module provides health check and heartbeat functionality for model
instances deployed via PyLet.

Components:
- wait_for_health(): Wait for model to become healthy (supports vLLM/sglang)
- HeartbeatSender: Send periodic heartbeats to scheduler

Example:
    from swarmpilot.scripts.health import wait_for_health, HeartbeatSender

    # Wait for model to be ready
    if wait_for_health("localhost:8001"):
        print("Model is ready")

    # Start heartbeat
    sender = HeartbeatSender(
        scheduler_url="http://scheduler:8000",
        instance_id="my-instance",
    )
    sender.start()  # Runs in background thread
"""

from __future__ import annotations

import threading
import time

import httpx
from loguru import logger

# Health check endpoints for different backends
HEALTH_ENDPOINTS: list[str] = [
    "/health",  # vLLM standard
    "/v1/models",  # OpenAI-compatible (vLLM, sglang)
    "/healthz",  # Kubernetes-style
    "/",  # Basic connectivity check
]


def wait_for_health(
    endpoint: str,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
    health_endpoints: list[str] | None = None,
) -> bool:
    """Wait for model health check to pass.

    Tries multiple health endpoints to support different backends (vLLM, sglang).
    Returns True as soon as any endpoint returns 200 OK.

    Args:
        endpoint: Model endpoint in "host:port" format.
        timeout: Maximum wait time in seconds.
        poll_interval: Time between health check attempts.
        health_endpoints: Custom list of endpoints to try. If not provided,
            uses defaults: /health, /v1/models, /healthz, /

    Returns:
        True if health check passes, False if timeout.

    Example:
        if wait_for_health("localhost:8001", timeout=120):
            print("Model is ready")
        else:
            print("Model failed to become healthy")
    """
    endpoints_to_try = health_endpoints or HEALTH_ENDPOINTS

    # Normalize endpoint
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        base_url = f"http://{endpoint}"
    else:
        base_url = endpoint

    logger.info(f"Waiting for model at {base_url} (timeout: {timeout}s)")

    start_time = time.time()
    check_count = 0

    while time.time() - start_time < timeout:
        check_count += 1

        for health_path in endpoints_to_try:
            try:
                url = f"{base_url}{health_path}"
                response = httpx.get(url, timeout=5.0)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Model healthy at {base_url} "
                        f"(endpoint: {health_path}, "
                        f"check #{check_count}, "
                        f"elapsed: {elapsed:.1f}s)"
                    )
                    return True
            except httpx.RequestError:
                pass  # Try next endpoint

        # Progress logging every 10 checks
        if check_count % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Still waiting for model... "
                f"(check #{check_count}, elapsed: {elapsed:.0f}s)"
            )

        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    logger.warning(
        f"Health check timed out after {elapsed:.1f}s ({check_count} checks)"
    )
    return False


def check_health_once(
    endpoint: str,
    health_endpoints: list[str] | None = None,
) -> bool:
    """Perform a single health check.

    Args:
        endpoint: Model endpoint in "host:port" format.
        health_endpoints: Custom list of endpoints to try.

    Returns:
        True if healthy, False otherwise.
    """
    endpoints_to_try = health_endpoints or HEALTH_ENDPOINTS

    if not endpoint.startswith("http://"):
        base_url = f"http://{endpoint}"
    else:
        base_url = endpoint

    for health_path in endpoints_to_try:
        try:
            response = httpx.get(f"{base_url}{health_path}", timeout=5.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass

    return False


class HeartbeatSender:
    """Send periodic heartbeats to scheduler.

    Heartbeats indicate the instance is still alive and can be used by the
    scheduler to detect failed instances.

    The heartbeat runs in a background thread and can be stopped gracefully.

    Attributes:
        scheduler_url: Scheduler URL for heartbeat endpoint.
        instance_id: Instance identifier.
        interval: Heartbeat interval in seconds.
        running: True if heartbeat thread is running.
    """

    def __init__(
        self,
        scheduler_url: str,
        instance_id: str,
        interval: float = 30.0,
    ):
        """Initialize heartbeat sender.

        Args:
            scheduler_url: Scheduler URL for heartbeat endpoint.
            instance_id: Instance identifier.
            interval: Heartbeat interval in seconds (default: 30s).
        """
        self.scheduler_url = scheduler_url.rstrip("/")
        self.instance_id = instance_id
        self.interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._heartbeat_count = 0
        self._last_success = False

    def start(self) -> None:
        """Start heartbeat thread.

        Does nothing if already running.
        """
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            f"Heartbeat started for {self.instance_id} " f"(interval: {self.interval}s)"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop heartbeat thread.

        Args:
            timeout: Maximum time to wait for thread to stop.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Heartbeat thread did not stop cleanly")

        logger.info(f"Heartbeat stopped after {self._heartbeat_count} heartbeats")

    def _run(self) -> None:
        """Heartbeat loop (runs in background thread)."""
        while self._running and not self._stop_event.is_set():
            self._send_heartbeat()
            # Use event wait for interruptible sleep
            self._stop_event.wait(timeout=self.interval)

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat to scheduler."""
        try:
            response = httpx.post(
                f"{self.scheduler_url}/instance/heartbeat",
                json={"instance_id": self.instance_id},
                timeout=5.0,
            )

            self._heartbeat_count += 1

            if response.status_code == 200:
                self._last_success = True
                logger.debug(f"Heartbeat #{self._heartbeat_count} sent successfully")
            else:
                self._last_success = False
                logger.warning(
                    f"Heartbeat returned {response.status_code}: "
                    f"{response.text[:100]}"
                )

        except httpx.RequestError as e:
            self._last_success = False
            logger.warning(f"Heartbeat failed: {e}")

    @property
    def running(self) -> bool:
        """Check if heartbeat is running.

        Returns:
            True if heartbeat thread is active.
        """
        return self._running

    @property
    def heartbeat_count(self) -> int:
        """Get number of heartbeats sent.

        Returns:
            Total heartbeat count since start.
        """
        return self._heartbeat_count

    @property
    def last_heartbeat_success(self) -> bool:
        """Check if last heartbeat was successful.

        Returns:
            True if last heartbeat got 200 response.
        """
        return self._last_success


class HealthMonitor:
    """Combined health check and heartbeat monitor.

    Convenience class that manages both health checking and heartbeat sending.

    Example:
        monitor = HealthMonitor(
            endpoint="localhost:8001",
            scheduler_url="http://scheduler:8000",
            instance_id="my-instance",
        )

        if monitor.wait_for_ready():
            monitor.start_heartbeat()
            # ... do work ...
            monitor.stop()
    """

    def __init__(
        self,
        endpoint: str,
        scheduler_url: str,
        instance_id: str,
        health_timeout: float = 300.0,
        heartbeat_interval: float = 30.0,
    ):
        """Initialize health monitor.

        Args:
            endpoint: Model endpoint for health checks.
            scheduler_url: Scheduler URL for heartbeats.
            instance_id: Instance identifier.
            health_timeout: Health check timeout in seconds.
            heartbeat_interval: Heartbeat interval in seconds.
        """
        self.endpoint = endpoint
        self.health_timeout = health_timeout
        self._heartbeat = HeartbeatSender(
            scheduler_url=scheduler_url,
            instance_id=instance_id,
            interval=heartbeat_interval,
        )
        self._is_ready = False

    def wait_for_ready(self) -> bool:
        """Wait for model to become healthy.

        Returns:
            True if model is ready, False if timeout.
        """
        self._is_ready = wait_for_health(
            self.endpoint,
            timeout=self.health_timeout,
        )
        return self._is_ready

    def check_health(self) -> bool:
        """Perform a single health check.

        Returns:
            True if healthy, False otherwise.
        """
        return check_health_once(self.endpoint)

    def start_heartbeat(self) -> None:
        """Start sending heartbeats."""
        self._heartbeat.start()

    def stop(self) -> None:
        """Stop heartbeat."""
        self._heartbeat.stop()

    @property
    def is_ready(self) -> bool:
        """Check if model is ready.

        Returns:
            True if wait_for_ready() returned True.
        """
        return self._is_ready

    @property
    def heartbeat_running(self) -> bool:
        """Check if heartbeat is running.

        Returns:
            True if heartbeat thread is active.
        """
        return self._heartbeat.running
