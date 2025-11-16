"""
CLI entry point for the Instance Service.

This module provides the command-line interface using Typer.
"""

import typer
from typing import Optional
import uvicorn

from .config import config
from . import logger as _  # Import logger module to initialize logging

app = typer.Typer(
    name="sinstance",
    help="Instance Service CLI - Manage model instances and task queues",
    add_completion=False,
    no_args_is_help=True
)


@app.command()
def start(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the service to"
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to bind the service to (default: from config or 5000)"
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development"
    ),
    platform_software_name: Optional[str] = typer.Option(
        None,
        "--platform-software-name",
        help="Override platform software name (e.g., 'Linux', 'Darwin')"
    ),
    platform_software_version: Optional[str] = typer.Option(
        None,
        "--platform-software-version",
        help="Override platform software version (e.g., '5.15.0-151-generic')"
    ),
    platform_hardware_name: Optional[str] = typer.Option(
        None,
        "--platform-hardware-name",
        help="Override platform hardware name (e.g., 'NVIDIA GeForce RTX 4090', 'CPU')"
    )
):
    """
    Start the Instance Service.

    This will start the FastAPI application and begin listening for requests.
    The service will accept model management and task submission requests.

    Platform information can be overridden using CLI options or environment variables:
    - INSTANCE_PLATFORM_SOFTWARE_NAME
    - INSTANCE_PLATFORM_SOFTWARE_VERSION
    - INSTANCE_PLATFORM_HARDWARE_NAME

    CLI options take precedence over environment variables.
    """
    # Use config values as defaults
    service_port = port if port is not None else config.instance_port
    config.instance_port = service_port # let cli override the config

    # Update log level in config if provided via CLI
    if log_level is not None:
        config.log_level = log_level

    # Update platform overrides if provided via CLI (takes precedence over env vars)
    if platform_software_name is not None:
        config.platform_software_name = platform_software_name
    if platform_software_version is not None:
        config.platform_software_version = platform_software_version
    if platform_hardware_name is not None:
        config.platform_hardware_name = platform_hardware_name

    typer.echo(f"Starting Instance Service on {host}:{service_port}")
    typer.echo(f"Instance ID: {config.instance_id}")
    typer.echo(f"Log Level: {config.log_level}")

    # Show platform override information if any is set
    platform_overrides = []
    if config.platform_software_name:
        platform_overrides.append(f"Software: {config.platform_software_name}")
    if config.platform_software_version:
        platform_overrides.append(f"Version: {config.platform_software_version}")
    if config.platform_hardware_name:
        platform_overrides.append(f"Hardware: {config.platform_hardware_name}")

    if platform_overrides:
        typer.echo(f"Platform Overrides: {', '.join(platform_overrides)}")

    # Start the FastAPI application with uvicorn
    # Note: We don't pass log_level to uvicorn because loguru intercepts all logs
    uvicorn.run(
        "src.api:app",
        host=host,
        port=service_port,
        log_config=None,  # Disable uvicorn's default logging config
        reload=reload
    )


@app.command()
def version():
    """
    Show version information.
    """
    typer.echo("Instance Service version 0.1.0")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
