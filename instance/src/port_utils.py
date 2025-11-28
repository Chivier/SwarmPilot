"""
Port management utilities for hot-standby system.

Provides functions for checking port availability and finding available ports.
"""

import socket
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class PortCheckResult:
    """Result of a port availability check."""
    port: int
    available: bool
    error: Optional[str] = None


def is_port_available(port: int, host: str = "127.0.0.1") -> PortCheckResult:
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check
        host: Host address to check (default: localhost)

    Returns:
        PortCheckResult with availability status
    """
    if port < 1 or port > 65535:
        return PortCheckResult(port=port, available=False, error="Port out of valid range (1-65535)")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return PortCheckResult(port=port, available=True)
    except OSError as e:
        return PortCheckResult(port=port, available=False, error=str(e))


def find_available_port(
    start_port: int,
    max_attempts: int = 100,
    host: str = "127.0.0.1"
) -> Optional[int]:
    """
    Find an available port starting from the given port.

    Searches sequentially from start_port until an available port is found
    or max_attempts is reached.

    Args:
        start_port: Starting port number to search from
        max_attempts: Maximum number of ports to try
        host: Host address to check (default: localhost)

    Returns:
        Available port number, or None if no port found within max_attempts
    """
    for offset in range(max_attempts):
        port = start_port + offset
        if port > 65535:
            logger.warning(f"Port search exceeded valid range at port {port}")
            break

        result = is_port_available(port, host)
        if result.available:
            logger.debug(f"Found available port: {port}")
            return port
        else:
            logger.debug(f"Port {port} unavailable: {result.error}")

    logger.warning(
        f"No available port found in range {start_port}-{start_port + max_attempts - 1}"
    )
    return None


def calculate_backoff_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    multiplier: float
) -> float:
    """
    Calculate delay for given attempt using exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        multiplier: Backoff multiplier

    Returns:
        Calculated delay in seconds, capped at max_delay
    """
    delay = initial_delay * (multiplier ** attempt)
    return min(delay, max_delay)
