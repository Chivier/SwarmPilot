"""
Rate limiter using token bucket algorithm with Poisson distribution support.

This module provides thread-safe rate limiting for controlling QPS across multiple
threads. Based on the reference implementation from experiments/03.Exp4.Text2Video/
test_dynamic_workflow_sim.py:84-132.
"""

import math
import random
import threading
import time
from typing import Optional
from loguru import logger as loguru_logger


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    The token bucket algorithm maintains a bucket that fills with tokens at a constant
    rate. Each request consumes one token. If no tokens are available, the request
    blocks until a token is added.

    Features:
    - Thread-safe operation for concurrent access
    - Configurable burst size (max_tokens)
    - Auto-refill based on elapsed time
    - Support for Poisson distribution intervals
    """

    def __init__(self, rate: float, burst_size: Optional[float] = None):
        """
        Initialize rate limiter.

        Args:
            rate: Target rate in requests per second (e.g., 10.0 for 10 QPS)
            burst_size: Maximum burst capacity (default: rate * 2)
        """
        self.rate = rate
        self.max_tokens = burst_size if burst_size is not None else rate * 2
        self.tokens = 0.0  # Start with 0 tokens to enforce strict rate limit
        self.last_update = time.time()
        self.lock = threading.Lock()
        self.logger = loguru_logger.bind(component="RateLimiter")

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens from the bucket, blocking if necessary.

        This method will block until enough tokens are available. It uses a
        sleep-and-retry pattern with fine-grained sleep intervals for accurate
        rate limiting.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            Time spent waiting in seconds
        """
        wait_start = time.time()

        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update

                # Refill tokens based on elapsed time
                new_tokens = elapsed * self.rate
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)
                self.last_update = now

                self.logger.debug(
                    f"Token state: {self.tokens:.2f}/{self.max_tokens:.2f} "
                    f"(refilled {new_tokens:.2f} over {elapsed:.3f}s)"
                )

                # If enough tokens available, consume and return
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    wait_time = time.time() - wait_start
                    if wait_time > 0.001:  # Log only if waited more than 1ms
                        self.logger.debug(f"Acquired {tokens} tokens after {wait_time:.3f}s wait")
                    return wait_time

            # Not enough tokens, sleep briefly and retry
            # Use small sleep to avoid excessive CPU usage
            time.sleep(0.01)

    def get_poisson_interval(self) -> float:
        """
        Get a Poisson-distributed inter-arrival time.

        The Poisson process models random arrivals with exponentially distributed
        intervals. The mean interval is 1/rate.

        Formula: -ln(U) / rate, where U ~ Uniform(0, 1)

        Returns:
            Interval in seconds until next arrival
        """
        # Using -1.0/rate * log(random()) formula from reference
        # This is mathematically equivalent to random.expovariate(rate)
        # but matches the reference implementation exactly
        interval = -1.0 / self.rate * math.log(random.random())
        self.logger.debug(f"Poisson interval: {interval:.3f}s (mean: {1.0/self.rate:.3f}s)")
        return interval

    def reset(self):
        """Reset the token bucket to empty state."""
        with self.lock:
            self.tokens = 0.0
            self.last_update = time.time()
            self.logger.info("Rate limiter reset")

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current state: {
                'rate': configured rate,
                'max_tokens': maximum burst size,
                'current_tokens': current token count,
                'utilization': token utilization (0.0-1.0)
            }
        """
        with self.lock:
            return {
                'rate': self.rate,
                'max_tokens': self.max_tokens,
                'current_tokens': self.tokens,
                'utilization': 1.0 - (self.tokens / self.max_tokens) if self.max_tokens > 0 else 0.0
            }


class PoissonRateLimiter(RateLimiter):
    """
    Rate limiter that automatically applies Poisson-distributed intervals.

    This is a convenience wrapper around RateLimiter that combines token bucket
    rate limiting with Poisson process intervals. Each acquire() call will
    automatically sleep for a Poisson-distributed interval.
    """

    def __init__(self, rate: float, burst_size: Optional[float] = None):
        """
        Initialize Poisson rate limiter.

        Args:
            rate: Target rate in requests per second (e.g., 10.0 for 10 QPS)
            burst_size: Maximum burst capacity (default: rate * 2)
        """
        super().__init__(rate, burst_size)

    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens and sleep for Poisson-distributed interval.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            Total time spent (waiting for tokens + Poisson interval)
        """
        # First acquire tokens (may block)
        wait_time = super().acquire(tokens)

        # Then sleep for Poisson interval
        interval = self.get_poisson_interval()
        time.sleep(interval)

        return wait_time + interval
