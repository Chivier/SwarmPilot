# PYLET-001: Create SWorker Wrapper Package

## Objective

Create the core sworker-wrapper package structure that will serve as the adapter between PyLet's process management and SwarmPilot's request-level task queue system.

## Prerequisites

- PyLet development environment set up
- Understanding of current Instance service architecture
- Familiarity with FastAPI and uvicorn

## Background

The SWorker wrapper is the bridge between:
- **PyLet**: Manages process lifecycle (start, stop, monitor)
- **SwarmPilot Scheduler**: Assigns request-level tasks to instances

The wrapper runs as the main process in a PyLet instance and internally manages the actual model service (vLLM, sglang, etc.).

## Files to Create

```
sworker-wrapper/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── main.py           # Entry point
│   └── config.py         # Configuration management
└── tests/
    └── test_config.py
```

## Implementation Steps

### Step 1: Create Package Structure

Create `sworker-wrapper/pyproject.toml`:

```toml
[project]
name = "sworker-wrapper"
version = "0.1.0"
description = "SwarmPilot worker wrapper for PyLet integration"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "httpx>=0.25.0",
    "loguru>=0.7.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
]

[project.scripts]
sworker-wrapper = "src.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Step 2: Implement Configuration

Create `sworker-wrapper/src/config.py`:

```python
"""SWorker Wrapper Configuration.

Environment variables:
    PORT: PyLet-assigned port for HTTP API
    SCHEDULER_URL: SwarmPilot scheduler URL
    MODEL_PORT_OFFSET: Offset for internal model port (default: 1)
    GRACE_PERIOD_SECONDS: Shutdown grace period (default: 30)
    LOG_LEVEL: Logging level (default: INFO)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for SWorker wrapper."""

    # Port settings
    port: int
    model_port_offset: int = 1

    # Scheduler settings
    scheduler_url: str = "http://localhost:8000"

    # Instance identification
    instance_id: Optional[str] = None
    model_id: Optional[str] = None

    # Command to wrap
    command: str = ""

    # Lifecycle settings
    grace_period_seconds: int = 30
    health_check_interval: int = 5

    # Logging
    log_level: str = "INFO"

    @property
    def model_port(self) -> int:
        """Calculate internal model service port."""
        return self.port + self.model_port_offset

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        port_str = os.getenv("PORT", "16000")
        return cls(
            port=int(port_str),
            model_port_offset=int(os.getenv("MODEL_PORT_OFFSET", "1")),
            scheduler_url=os.getenv("SCHEDULER_URL", "http://localhost:8000"),
            instance_id=os.getenv("INSTANCE_ID"),
            model_id=os.getenv("MODEL_ID"),
            command=os.getenv("SWORKER_COMMAND", ""),
            grace_period_seconds=int(os.getenv("GRACE_PERIOD_SECONDS", "30")),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "5")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
```

### Step 3: Implement Entry Point

Create `sworker-wrapper/src/main.py`:

```python
"""SWorker Wrapper Entry Point.

Usage:
    sworker-wrapper --command "vllm serve model" --port $PORT

    Or set via environment:
    PORT=16000 SWORKER_COMMAND="vllm serve model" sworker-wrapper
"""

import argparse
import asyncio
import os
import sys
import uuid

import uvicorn
from loguru import logger

from src.config import Config, set_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SWorker Wrapper - PyLet to SwarmPilot bridge"
    )
    parser.add_argument(
        "--command",
        type=str,
        help="Command to wrap (model service command)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "16000")),
        help="Port for HTTP API (default: $PORT or 16000)",
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default=os.getenv("SCHEDULER_URL", "http://localhost:8000"),
        help="SwarmPilot scheduler URL",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID"),
        help="Model identifier for scheduler registration",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default=os.getenv("INSTANCE_ID"),
        help="Instance identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=int(os.getenv("GRACE_PERIOD_SECONDS", "30")),
        help="Shutdown grace period in seconds",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Generate instance ID if not provided
    instance_id = args.instance_id or f"sworker-{uuid.uuid4().hex[:8]}"

    # Build configuration
    config = Config(
        port=args.port,
        scheduler_url=args.scheduler_url,
        instance_id=instance_id,
        model_id=args.model_id,
        command=args.command or os.getenv("SWORKER_COMMAND", ""),
        grace_period_seconds=args.grace_period,
        log_level=args.log_level,
    )
    set_config(config)

    logger.info(f"SWorker Wrapper starting")
    logger.info(f"  Instance ID: {config.instance_id}")
    logger.info(f"  API Port: {config.port}")
    logger.info(f"  Model Port: {config.model_port}")
    logger.info(f"  Scheduler: {config.scheduler_url}")
    logger.info(f"  Command: {config.command}")

    # Validate command
    if not config.command:
        logger.error("No command specified. Use --command or SWORKER_COMMAND env var")
        sys.exit(1)

    # Import and run the FastAPI app
    # (api.py will be implemented in PYLET-002)
    from src.api import create_app

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
```

### Step 4: Create Stub API Module

Create `sworker-wrapper/src/api.py` (stub for now):

```python
"""SWorker Wrapper HTTP API.

This is a stub implementation. Full implementation in PYLET-002.
"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="SWorker Wrapper",
        description="PyLet to SwarmPilot bridge",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/info")
    async def info():
        """Instance information."""
        from src.config import get_config
        config = get_config()
        return {
            "instance_id": config.instance_id,
            "model_id": config.model_id,
            "port": config.port,
            "model_port": config.model_port,
        }

    return app
```

### Step 5: Create __init__.py

Create `sworker-wrapper/src/__init__.py`:

```python
"""SWorker Wrapper - PyLet to SwarmPilot bridge."""

__version__ = "0.1.0"
```

### Step 6: Create Tests

Create `sworker-wrapper/tests/test_config.py`:

```python
"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from src.config import Config, get_config, set_config


class TestConfig:
    """Tests for Config class."""

    def test_model_port_calculation(self):
        """Test model port is calculated correctly."""
        config = Config(port=16000, model_port_offset=1)
        assert config.model_port == 16001

        config = Config(port=16000, model_port_offset=100)
        assert config.model_port == 16100

    def test_from_env_defaults(self):
        """Test default values from environment."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env()
            assert config.port == 16000
            assert config.scheduler_url == "http://localhost:8000"
            assert config.grace_period_seconds == 30

    def test_from_env_custom(self):
        """Test custom values from environment."""
        env = {
            "PORT": "17000",
            "SCHEDULER_URL": "http://scheduler:8000",
            "MODEL_PORT_OFFSET": "10",
            "INSTANCE_ID": "test-instance",
            "MODEL_ID": "test-model",
            "SWORKER_COMMAND": "echo hello",
            "GRACE_PERIOD_SECONDS": "60",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env, clear=True):
            config = Config.from_env()
            assert config.port == 17000
            assert config.model_port == 17010
            assert config.scheduler_url == "http://scheduler:8000"
            assert config.instance_id == "test-instance"
            assert config.model_id == "test-model"
            assert config.command == "echo hello"
            assert config.grace_period_seconds == 60
            assert config.log_level == "DEBUG"


class TestConfigSingleton:
    """Tests for global config singleton."""

    def test_get_config_creates_instance(self):
        """Test get_config creates instance on first call."""
        # Reset global state
        import src.config as config_module
        config_module._config = None

        with patch.dict(os.environ, {"PORT": "18000"}, clear=True):
            config = get_config()
            assert config.port == 18000

    def test_set_config(self):
        """Test set_config replaces global instance."""
        custom_config = Config(port=19000, command="test")
        set_config(custom_config)

        config = get_config()
        assert config.port == 19000
        assert config.command == "test"
```

## Test Strategy

### Unit Tests

```bash
cd sworker-wrapper
uv sync
uv run pytest tests/test_config.py -v
```

### Manual Testing

```bash
# Test basic startup
PORT=16000 SWORKER_COMMAND="sleep 3600" uv run sworker-wrapper

# In another terminal, test endpoints
curl http://localhost:16000/health
curl http://localhost:16000/info
```

## Acceptance Criteria

- [ ] Package structure created with pyproject.toml
- [ ] Configuration loads from environment and CLI args
- [ ] Entry point parses arguments correctly
- [ ] Health check endpoint responds
- [ ] Info endpoint returns instance configuration
- [ ] All unit tests pass
- [ ] Manual testing confirms HTTP server starts

## Next Steps

After completing this task:
1. Proceed to [PYLET-002](PYLET-002-implement-task-queue.md) to implement the task queue
2. The stub API will be expanded with full task submission endpoints

## Code References

- Current Instance config: [instance/src/config.py](../../instance/src/config.py)
- Current Instance API: [instance/src/api.py](../../instance/src/api.py)
