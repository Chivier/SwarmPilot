#!/usr/bin/env python3
r"""Start model and register with scheduler.

This is the main entry point for PyLet-deployed models. It:
1. Waits for the model service to be healthy
2. Registers the instance with the scheduler
3. Sets up signal handlers for graceful shutdown
4. Optionally starts heartbeat thread
5. Blocks until terminated

Usage (via PyLet submit):
    pylet submit "bash -c 'vllm serve model --port $PORT & python -m scripts.start_model'" \\
        --env MODEL_ID=Qwen/Qwen3-0.6B \\
        --env SCHEDULER_URL=http://localhost:8000

Environment Variables:
    MODEL_ID: Model identifier (required)
    SCHEDULER_URL: Scheduler URL for registration (default: http://localhost:8000)
    PORT: Port the model is running on (set by PyLet)
    HOSTNAME: Host IP/name (optional, defaults to localhost)
    PYLET_INSTANCE_ID: Instance ID (set by PyLet)
    MODEL_BACKEND: Backend type (default: vllm)
    HEALTH_PATH: Health check path (default: /health)
    HEALTH_TIMEOUT: Health check timeout in seconds (default: 300)
    HEARTBEAT_ENABLED: Enable heartbeat (default: true)
    HEARTBEAT_INTERVAL: Heartbeat interval in seconds (default: 30)
    GRACE_PERIOD: Graceful shutdown grace period in seconds (default: 25)
"""

from __future__ import annotations

import os
import sys
import time

from loguru import logger

from swarmpilot.scripts.health import HeartbeatSender, wait_for_health
from swarmpilot.scripts.register import (
    get_instance_info,
    register_with_scheduler,
)
from swarmpilot.scripts.signal_handler import GracefulShutdown


def main() -> int:
    """Main entry point for model startup script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    # Get configuration from environment
    scheduler_url = os.getenv("SCHEDULER_URL", "http://localhost:8000")
    health_timeout = float(os.getenv("HEALTH_TIMEOUT", "300"))
    heartbeat_enabled = os.getenv("HEARTBEAT_ENABLED", "true").lower() == "true"
    heartbeat_interval = float(os.getenv("HEARTBEAT_INTERVAL", "30"))
    grace_period = float(os.getenv("GRACE_PERIOD", "25"))

    # Get instance info from environment
    try:
        instance_info = get_instance_info()
    except ValueError as e:
        logger.error(f"Failed to get instance info: {e}")
        return 1

    instance_id = instance_info["instance_id"]
    model_id = instance_info["model_id"]
    endpoint = instance_info["endpoint"]

    logger.info(f"Starting model registration for {model_id}")
    logger.info(f"  Instance ID: {instance_id}")
    logger.info(f"  Endpoint: {endpoint}")
    logger.info(f"  Scheduler: {scheduler_url}")
    logger.info(
        f"  Heartbeat: {'enabled' if heartbeat_enabled else 'disabled'}"
    )

    # Step 1: Wait for model to be healthy
    logger.info("Step 1/4: Waiting for model to be healthy...")
    if not wait_for_health(endpoint, timeout=health_timeout):
        logger.error("Model health check failed, aborting registration")
        return 1

    # Step 2: Register with scheduler
    logger.info("Step 2/4: Registering with scheduler...")
    if not register_with_scheduler(
        scheduler_url=scheduler_url,
        instance_id=instance_id,
        model_id=model_id,
        endpoint=endpoint,
    ):
        logger.error("Failed to register with scheduler")
        return 1

    # Step 3: Start heartbeat if enabled
    heartbeat: HeartbeatSender | None = None
    if heartbeat_enabled:
        logger.info("Step 3/4: Starting heartbeat...")
        heartbeat = HeartbeatSender(
            scheduler_url=scheduler_url,
            instance_id=instance_id,
            interval=heartbeat_interval,
        )
        heartbeat.start()
    else:
        logger.info("Step 3/4: Heartbeat disabled, skipping...")

    # Step 4: Set up signal handlers for graceful shutdown
    logger.info("Step 4/4: Setting up signal handlers...")
    shutdown_handler = GracefulShutdown(
        scheduler_url=scheduler_url,
        instance_id=instance_id,
        grace_period=grace_period,
    )

    # Add heartbeat cleanup callback
    if heartbeat is not None:
        shutdown_handler.add_cleanup_callback(heartbeat.stop)

    shutdown_handler.setup()

    # Ready!
    logger.info("=" * 60)
    logger.info("Registration complete. Model is ready to serve requests.")
    logger.info(f"  Endpoint: http://{endpoint}")
    logger.info(f"  Model: {model_id}")
    logger.info("Waiting for termination signal (SIGTERM/SIGINT)...")
    logger.info("=" * 60)

    # Block until terminated
    try:
        while True:
            time.sleep(60)  # Wake up periodically to allow signal handling
    except KeyboardInterrupt:
        # Handler should have been called, but exit cleanly just in case
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
