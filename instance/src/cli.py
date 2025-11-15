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
    )
):
    """
    Start the Instance Service.

    This will start the FastAPI application and begin listening for requests.
    The service will accept model management and task submission requests.
    """
    # Use config values as defaults
    service_port = port if port is not None else config.instance_port
    config.instance_port = service_port # let cli override the config

    # Update log level in config if provided via CLI
    if log_level is not None:
        config.log_level = log_level

    typer.echo(f"Starting Instance Service on {host}:{service_port}")
    typer.echo(f"Instance ID: {config.instance_id}")
    typer.echo(f"Log Level: {config.log_level}")

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
