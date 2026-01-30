"""CLI entry point for the Planner service.

This module provides a command-line interface using Typer for managing
the planner service.
"""

import os
import sys

import typer
import uvicorn

app = typer.Typer(
    name="splanner",
    help="Planner service command-line interface",
    add_completion=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

# Disable rich/colors globally for typer
os.environ["NO_COLOR"] = "1"


@app.command()
def start(
    host: str | None = typer.Option(
        None,
        "--host",
        "-h",
        help="Server host address (overrides PLANNER_HOST env var)",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port number (overrides PLANNER_PORT env var)",
    ),
) -> None:
    """Start the planner service.

    Examples:
        $ splanner start
        $ splanner start --host 127.0.0.1 --port 9000
    """
    typer.echo("Starting Planner service...")

    if host is not None:
        os.environ["PLANNER_HOST"] = host
    if port is not None:
        os.environ["PLANNER_PORT"] = str(port)

    from swarmpilot.planner.config import config

    final_host = config.planner_host
    final_port = config.planner_port

    typer.echo(f"Server will start at: http://{final_host}:{final_port}")
    typer.echo("")

    try:
        uvicorn.run(
            "swarmpilot.planner.api:app",
            host=final_host,
            port=final_port,
            log_config=None,
        )
    except KeyboardInterrupt:
        typer.echo("\nShutting down planner service...")
        sys.exit(0)
    except Exception as e:
        typer.echo(f"Error starting planner service: {e}", err=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show the planner version."""
    from swarmpilot.planner import __version__

    typer.echo(f"Planner version: {__version__}")


if __name__ == "__main__":
    app()
