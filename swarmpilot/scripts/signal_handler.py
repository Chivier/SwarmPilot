"""Signal handling for graceful shutdown (PYLET-003).

This module provides a GracefulShutdown class that manages graceful shutdown
on SIGTERM/SIGINT signals. When triggered, it deregisters from the scheduler
and runs cleanup callbacks before exiting.

Example:
    from swarmpilot.scripts.signal_handler import GracefulShutdown

    shutdown = GracefulShutdown(
        scheduler_url="http://scheduler:8000",
        instance_id="my-instance",
    )
    shutdown.add_cleanup_callback(cleanup_model)
    shutdown.setup()

    # Now SIGTERM/SIGINT will trigger graceful shutdown
"""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable

import httpx
from loguru import logger


class GracefulShutdown:
    """Manages graceful shutdown on signal.

    This class installs signal handlers for SIGTERM and SIGINT. When a signal
    is received, it:
    1. Deregisters from the scheduler
    2. Runs registered cleanup callbacks
    3. Exits cleanly

    If a second signal is received during shutdown, it forces immediate exit.

    Attributes:
        scheduler_url: URL of the scheduler for deregistration.
        instance_id: Instance identifier.
        grace_period: Maximum time for graceful shutdown in seconds.
        shutdown_requested: True if shutdown has been requested.
    """

    def __init__(
        self,
        scheduler_url: str,
        instance_id: str,
        grace_period: float = 25.0,
    ):
        """Initialize graceful shutdown handler.

        Args:
            scheduler_url: Scheduler URL for deregistration.
            instance_id: Instance identifier.
            grace_period: Maximum time for graceful shutdown (default: 25s).
                Should be less than PyLet's grace period (default 30s).
        """
        self.scheduler_url = scheduler_url
        self.instance_id = instance_id
        self.grace_period = grace_period
        self._shutdown_requested = False
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._original_sigterm = None
        self._original_sigint = None

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback to run on shutdown.

        Callbacks are run in order after deregistration. Exceptions in
        callbacks are logged but don't prevent other callbacks from running.

        Args:
            callback: Callable with no arguments.
        """
        self._cleanup_callbacks.append(callback)

    def setup(self) -> None:
        """Install signal handlers.

        After calling this method, SIGTERM and SIGINT will trigger graceful
        shutdown instead of the default behavior.
        """
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        logger.info(
            f"Signal handlers installed for graceful shutdown "
            f"(grace_period={self.grace_period}s)"
        )

    def teardown(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        logger.debug("Original signal handlers restored")

    def _handle_signal(self, signum: int, frame: object | None) -> None:
        """Handle received signal.

        Args:
            signum: Signal number.
            frame: Current stack frame (unused).
        """
        if self._shutdown_requested:
            logger.warning("Forced exit on second signal")
            sys.exit(1)
            return  # Explicit return for test scenarios where sys.exit is mocked

        self._shutdown_requested = True
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name}, initiating graceful shutdown...")

        self._graceful_shutdown()

    def _graceful_shutdown(self) -> None:
        """Execute graceful shutdown sequence."""
        start_time = time.time()

        # Step 1: Deregister from scheduler
        logger.info("Step 1/3: Deregistering from scheduler...")
        self._deregister()

        # Step 2: Run cleanup callbacks
        if self._cleanup_callbacks:
            logger.info(
                f"Step 2/3: Running {len(self._cleanup_callbacks)} cleanup callbacks..."
            )
            for i, callback in enumerate(self._cleanup_callbacks):
                try:
                    remaining = self.grace_period - (time.time() - start_time)
                    if remaining <= 0:
                        logger.warning(
                            "Grace period exceeded, skipping remaining callbacks"
                        )
                        break
                    callback()
                    logger.debug(f"Callback {i + 1} completed")
                except Exception as e:
                    logger.error(f"Cleanup callback {i + 1} error: {e}")
        else:
            logger.info("Step 2/3: No cleanup callbacks registered")

        # Step 3: Exit
        elapsed = time.time() - start_time
        logger.info(f"Step 3/3: Graceful shutdown complete in {elapsed:.1f}s")
        sys.exit(0)

    def _deregister(self) -> None:
        """Deregister from scheduler using drain + remove flow."""
        base_url = self.scheduler_url.rstrip("/")

        # Try drain first (preferred method)
        try:
            response = httpx.post(
                f"{base_url}/instance/drain",
                json={"instance_id": self.instance_id},
                timeout=5.0,
            )
            if response.status_code == 200:
                logger.debug("Instance draining started")
            elif response.status_code == 404:
                logger.debug("Instance not found in scheduler (already removed?)")
                return
        except httpx.RequestError as e:
            logger.warning(f"Drain request failed: {e}")

        # Try remove (may fail if not drained, but try anyway during shutdown)
        try:
            response = httpx.post(
                f"{base_url}/instance/remove",
                json={"instance_id": self.instance_id},
                timeout=5.0,
            )
            if response.status_code == 200:
                logger.info("Deregistered from scheduler")
            elif response.status_code == 404:
                logger.debug("Instance already removed")
            else:
                logger.warning(f"Remove returned {response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Deregistration failed: {e}")

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if SIGTERM/SIGINT has been received.
        """
        return self._shutdown_requested


def create_shutdown_handler(
    scheduler_url: str,
    instance_id: str,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
    grace_period: float = 25.0,
) -> GracefulShutdown:
    """Create and setup a graceful shutdown handler.

    Convenience function that creates a GracefulShutdown instance,
    adds cleanup callbacks, and installs signal handlers.

    Args:
        scheduler_url: Scheduler URL for deregistration.
        instance_id: Instance identifier.
        cleanup_callbacks: Optional list of cleanup callbacks.
        grace_period: Maximum time for graceful shutdown.

    Returns:
        Configured GracefulShutdown instance.

    Example:
        handler = create_shutdown_handler(
            scheduler_url="http://scheduler:8000",
            instance_id="my-instance",
            cleanup_callbacks=[stop_model, close_connections],
        )
    """
    handler = GracefulShutdown(
        scheduler_url=scheduler_url,
        instance_id=instance_id,
        grace_period=grace_period,
    )

    if cleanup_callbacks:
        for callback in cleanup_callbacks:
            handler.add_cleanup_callback(callback)

    handler.setup()
    return handler
