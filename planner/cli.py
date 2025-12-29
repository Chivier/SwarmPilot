"""CLI entry point for the Planner service.

Provides the `splanner` command for managing the planner service.
"""

import os

import typer
import uvicorn

# Disable rich/colors globally for typer
os.environ["NO_COLOR"] = "1"

app = typer.Typer(
    name="splanner",
    help="Planner service CLI - Optimize model-to-instance deployment",
    add_completion=False,
    pretty_exceptions_enable=False,
)


@app.callback()
def callback():
    """Planner service CLI - Optimize model-to-instance deployment."""
    pass


@app.command(name="start")
def start(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the server to",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        "-l",
        help="Log level (debug, info, warning, error, critical)",
        case_sensitive=False,
    ),
):
    """Start the Planner service.

    Examples:
        splanner start
        splanner start --host 127.0.0.1 --port 9000
        splanner start --log-level debug
    """
    # Validate log level
    valid_log_levels = ["debug", "info", "warning", "error", "critical"]
    log_level_lower = log_level.lower()

    if log_level_lower not in valid_log_levels:
        typer.echo(
            f"Error: Invalid log level '{log_level}'. "
            f"Valid options: {', '.join(valid_log_levels)}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Starting Planner service on {host}:{port}")
    typer.echo(f"Log level: {log_level_lower.upper()}")

    # Set PLANNER_LOGURU_LEVEL if not already set
    if "PLANNER_LOGURU_LEVEL" not in os.environ:
        os.environ["PLANNER_LOGURU_LEVEL"] = log_level_lower.upper()

    # Start the FastAPI application with uvicorn
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        log_level=log_level_lower,
    )


@app.command(name="version")
def version():
    """Show version information."""
    typer.echo("planner 0.1.0")


if __name__ == "__main__":
    app()
