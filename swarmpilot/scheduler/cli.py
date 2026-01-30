#!/usr/bin/env python3
"""CLI entry point for the Scheduler service.

This module provides a command-line interface using Typer for managing
the scheduler service.
"""

import os
import sys
from pathlib import Path

import typer
import uvicorn

app = typer.Typer(
    name="sscheduler",
    help="Scheduler service command-line interface",
    add_completion=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

# Disable rich/colors globally for typer
os.environ["NO_COLOR"] = "1"


def load_config_file(config_path: Path) -> dict:
    """Load configuration from a file (JSON, TOML, or YAML).

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration values

    Raises:
        typer.BadParameter: If file format is not supported or file is invalid
    """
    if not config_path.exists():
        raise typer.BadParameter(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    try:
        if suffix == ".json":
            import json

            with open(config_path) as f:
                return json.load(f)
        elif suffix == ".toml":
            import tomllib

            with open(config_path, "rb") as f:
                return tomllib.load(f)
        elif suffix in [".yaml", ".yml"]:
            try:
                import yaml
            except ImportError as e:
                raise typer.BadParameter(
                    "PyYAML is not installed. Install it with: uv add pyyaml"
                ) from e
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            raise typer.BadParameter(
                f"Unsupported configuration file format: {suffix}. "
                f"Supported formats: .json, .toml, .yaml, .yml"
            )
    except Exception as e:
        raise typer.BadParameter(f"Error loading configuration file: {e}") from e


def apply_config(config_dict: dict, host: str | None, port: int | None) -> None:
    """Apply configuration from dictionary to environment variables.

    Command-line arguments take precedence over config file values.

    Args:
        config_dict: Configuration dictionary from file
        host: Host override from command line
        port: Port override from command line
    """
    # Map configuration keys to environment variables
    env_mappings = {
        "server.host": "SCHEDULER_HOST",
        "server.port": "SCHEDULER_PORT",
        "scheduling.strategy": "SCHEDULING_STRATEGY",
        "training.enable_auto": "TRAINING_ENABLE_AUTO",
        "training.batch_size": "TRAINING_BATCH_SIZE",
        "logging.level": "LOG_LEVEL",
    }

    # Apply config file values
    for config_key, env_var in env_mappings.items():
        parts = config_key.split(".")
        value = config_dict

        # Navigate nested dictionary
        try:
            for part in parts:
                value = value[part]

            # Set environment variable if not already set
            if env_var not in os.environ:
                os.environ[env_var] = str(value)
        except (KeyError, TypeError):
            # Config key not found, skip
            pass

    # Command-line arguments override everything
    if host is not None:
        os.environ["SCHEDULER_HOST"] = host
    if port is not None:
        os.environ["SCHEDULER_PORT"] = str(port)


@app.command()
def start(
    host: str | None = typer.Option(
        None,
        "--host",
        "-h",
        help="Server host address (overrides config and environment variables)",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port number (overrides config and environment variables)",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (JSON, TOML, or YAML)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Start the scheduler service.

    The configuration priority (highest to lowest):
    1. Command-line arguments (--host, --port)
    2. Configuration file (--config)
    3. Environment variables
    4. Default values

    Examples:
        # Start with default configuration
        $ sscheduler start

        # Start on custom host and port
        $ sscheduler start --host 127.0.0.1 --port 9000

        # Start with configuration file
        $ sscheduler start --config config.toml

        # Override config file with command-line arguments
        $ sscheduler start --config config.toml --port 9000
    """
    typer.echo("Starting Scheduler service...")

    # Load configuration file if provided
    if config:
        typer.echo(f"Loading configuration from: {config}")
        config_dict = load_config_file(config)
        apply_config(config_dict, host, port)
    else:
        # Apply command-line overrides even without config file
        apply_config({}, host, port)

    # Reload config to pick up environment variable changes
    import importlib

    from swarmpilot.scheduler import config as config_module

    importlib.reload(config_module)
    app_config = config_module.config

    final_host = app_config.server.host
    final_port = app_config.server.port

    typer.echo(f"Server will start at: http://{final_host}:{final_port}")
    typer.echo("")

    # Start the server
    try:
        uvicorn.run(
            "swarmpilot.scheduler.api:app",
            host=final_host,
            port=final_port,
            log_config=None,  # Disable uvicorn's default logging, use loguru instead
        )
    except KeyboardInterrupt:
        typer.echo("\nShutting down scheduler service...")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Error starting scheduler service: {e}", err=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show the scheduler version."""
    from swarmpilot.scheduler import __version__

    typer.echo(f"Scheduler version: {__version__}")


if __name__ == "__main__":
    app()
