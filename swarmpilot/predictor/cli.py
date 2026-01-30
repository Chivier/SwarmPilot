"""Command-line interface for the predictor service.

Provides commands to start, manage, and configure the predictor service.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import typer
from typer import Option
from typing_extensions import Annotated

from swarmpilot.predictor.config import PredictorConfig
from swarmpilot.predictor.config import set_config
from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()

# Disable rich/colors globally for typer
os.environ["NO_COLOR"] = "1"

# Create the main Typer app
app = typer.Typer(
    name="spredictor",
    help="Runtime Predictor Service - MLP-based runtime prediction CLI",
    add_completion=False,
    pretty_exceptions_enable=False,
)

# Create config subcommand group
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")


@app.command()
def start(
    host: Annotated[
        str, Option("--host", "-h", help="Host to bind the server to")
    ] = "0.0.0.0",
    port: Annotated[
        int, Option("--port", "-p", help="Port to bind the server to")
    ] = 8000,
    reload: Annotated[
        bool, Option("--reload/--no-reload", help="Enable auto-reload")
    ] = False,
    workers: Annotated[
        int, Option("--workers", "-w", help="Number of worker processes")
    ] = 1,
    storage_dir: Annotated[
        str | None, Option("--storage-dir", "-s", help="Directory to store models")
    ] = None,
    config_file: Annotated[
        Path | None, Option("--config", "-c", help="Path to configuration file")
    ] = None,
    log_level: Annotated[
        str, Option("--log-level", "-l", help="Logging level")
    ] = "info",
    log_dir: Annotated[
        str | None, Option("--log-dir", help="Directory to store log files")
    ] = None,
) -> None:
    """Start the predictor service.

    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
        reload: Enable auto-reload for development.
        workers: Number of worker processes.
        storage_dir: Directory to store models.
        config_file: Path to configuration file.
        log_level: Logging level.
        log_dir: Directory to store log files.
    """
    import uvicorn

    # Load configuration
    if config_file:
        config = PredictorConfig.from_toml(config_file)
    else:
        config = PredictorConfig.from_toml()

    # Override with CLI arguments (only if explicitly provided)
    # We check sys.argv to see what was actually passed
    if "--host" in sys.argv or "-h" in sys.argv:
        config.host = host
    if "--port" in sys.argv or "-p" in sys.argv:
        config.port = port
    if "--reload" in sys.argv or "--no-reload" in sys.argv:
        config.reload = reload
    if "--workers" in sys.argv or "-w" in sys.argv:
        config.workers = workers
    if storage_dir is not None:
        config.storage_dir = storage_dir
    if "--log-level" in sys.argv or "-l" in sys.argv:
        config.log_level = log_level
    if log_dir is not None:
        config.log_dir = log_dir

    # Set global config
    set_config(config)

    # Initialize logging system before starting the server
    from swarmpilot.predictor.utils.logging import setup_logging

    setup_logging(
        log_dir=config.log_dir,
        log_level=config.log_level
    )

    # Ensure storage directory exists
    storage_path = config.ensure_storage_dir()

    typer.echo(f"🚀 Starting {config.app_name} v{config.app_version}")
    typer.echo(f"📁 Storage directory: {storage_path.absolute()}")
    typer.echo(f"📂 Log directory: {Path(config.log_dir).absolute()}")
    typer.echo(f"🌐 Server: http://{config.host}:{config.port}")
    typer.echo(f"📊 Log level: {config.log_level.upper()}")

    if config.reload:
        typer.echo("🔄 Auto-reload: ENABLED (development mode)")
    else:
        typer.echo(f"⚙️  Workers: {config.workers}")

    typer.echo("\n" + "=" * 50)

    # Import app here to allow config to be set first

    # Run the server
    uvicorn.run(
        "swarmpilot.predictor.api:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers if not config.reload else 1,
        log_level=config.log_level.lower(),
    )


@app.command()
def health(
    host: Annotated[
        str, Option("--host", "-h", help="Host to check")
    ] = "localhost",
    port: Annotated[
        int, Option("--port", "-p", help="Port to check")
    ] = 8000,
) -> None:
    """Check the health of a running predictor service.

    Args:
        host: Host to check.
        port: Port to check.
    """
    import httpx

    url = f"http://{host}:{port}/health"

    try:
        typer.echo(f"🔍 Checking health at {url}...")
        response = httpx.get(url, timeout=5.0)

        if response.status_code == 200:
            data = response.json()
            typer.echo(f"✅ Service is healthy!")
            typer.echo(f"   Status: {data.get('status', 'unknown')}")
            typer.echo(f"   Message: {data.get('message', 'N/A')}")
            if 'timestamp' in data:
                typer.echo(f"   Timestamp: {data['timestamp']}")
            sys.exit(0)
        else:
            typer.echo(f"❌ Service returned status {response.status_code}", err=True)
            sys.exit(1)

    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to {url}"
        logger.error(
            f"CLI health check failed\n"
            f"Error: {error_msg}\n"
            f"Host: {host}, Port: {port}\n"
            f"Exception: {type(e).__name__}: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        typer.echo(f"❌ {error_msg}", err=True)
        typer.echo(f"   Make sure the service is running on {host}:{port}", err=True)
        sys.exit(1)
    except httpx.TimeoutException as e:
        error_msg = "Request timed out"
        logger.error(
            f"CLI health check failed\n"
            f"Error: {error_msg}\n"
            f"URL: {url}\n"
            f"Exception: {type(e).__name__}: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        typer.echo(f"❌ {error_msg}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"CLI health check failed\n"
            f"Error: {str(e)}\n"
            f"URL: {url}\n"
            f"Exception: {type(e).__name__}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        typer.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show version information and dependencies."""
    import torch
    import fastapi
    import uvicorn
    import pydantic

    config = PredictorConfig()

    typer.echo(f"{'=' * 50}")
    typer.echo(f"🤖 {config.app_name}")
    typer.echo(f"{'=' * 50}")
    typer.echo(f"Version: {config.app_version}")
    typer.echo(f"\nDependencies:")
    typer.echo(f"  • Python: {sys.version.split()[0]}")
    typer.echo(f"  • FastAPI: {fastapi.__version__}")
    typer.echo(f"  • Uvicorn: {uvicorn.__version__}")
    typer.echo(f"  • Pydantic: {pydantic.__version__}")
    typer.echo(f"  • PyTorch: {torch.__version__}")
    typer.echo(f"  • Typer: {typer.__version__}")
    typer.echo(f"{'=' * 50}")


@app.command(name="list")
def list_models(
    storage_dir: Annotated[
        str | None, Option("--storage-dir", "-s", help="Storage directory")
    ] = None,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="Show detailed information")
    ] = False,
) -> None:
    """List all trained models in the storage directory.

    Args:
        storage_dir: Storage directory path.
        verbose: Show detailed information.
    """
    from swarmpilot.predictor.storage.model_storage import ModelStorage

    # Load config and override storage_dir if provided
    config = PredictorConfig.from_toml()
    if storage_dir:
        config.storage_dir = storage_dir

    storage = ModelStorage(storage_dir=config.storage_dir)
    storage_info = storage.get_storage_info()

    typer.echo(f"📁 Storage directory: {storage_info['storage_dir']}")
    typer.echo(f"📊 Total models: {storage_info['model_count']}")

    if storage_info['model_count'] == 0:
        typer.echo("\n   No models found.")
        return

    typer.echo(f"\n{'Model ID':<30} {'Type':<20} {'Samples':<10}")
    typer.echo("=" * 70)

    for model_id in storage_info['model_ids']:
        try:
            metadata = storage.get_metadata(model_id)
            if metadata:
                model_type = metadata.get('model_type', 'Unknown')
                samples = metadata.get('training_samples', 0)

                typer.echo(f"{model_id:<30} {model_type:<20} {samples:<10}")

                if verbose:
                    typer.echo(f"   Created: {metadata.get('created_at', 'Unknown')}")
                    typer.echo(f"   Updated: {metadata.get('updated_at', 'Unknown')}")
                    if 'model_config' in metadata:
                        config_info = metadata['model_config']
                        typer.echo(f"   Config: {config_info}")
                    typer.echo("")

        except Exception as e:
            logger.warning(
                f"Failed to load model metadata\n"
                f"Model ID: {model_id}\n"
                f"Exception: {type(e).__name__}: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            typer.echo(f"{model_id:<30} {'Error':<20} {str(e)}")


@config_app.command("show")
def config_show(
    config_file: Annotated[
        Path | None, Option("--config", "-c", help="Path to configuration file")
    ] = None,
) -> None:
    """Show current configuration.

    Args:
        config_file: Path to configuration file.
    """
    # Load configuration
    if config_file:
        config = PredictorConfig.from_toml(config_file)
        source = f"from {config_file}"
    else:
        config = PredictorConfig.from_toml()
        source = "from environment and defaults"

    typer.echo(f"{'=' * 50}")
    typer.echo(f"⚙️  Predictor Configuration ({source})")
    typer.echo(f"{'=' * 50}")

    config_dict = config.to_dict()
    max_key_length = max(len(key) for key in config_dict.keys())

    for key, value in config_dict.items():
        typer.echo(f"{key.replace('_', ' ').title():<{max_key_length + 5}}: {value}")

    typer.echo(f"{'=' * 50}")


@config_app.command("init")
def config_init(
    output: Annotated[
        Path, Option("--output", "-o", help="Output file path")
    ] = Path("predictor.toml"),
    force: Annotated[
        bool, Option("--force", "-f", help="Overwrite existing file")
    ] = False,
) -> None:
    """Initialize a default configuration file.

    Args:
        output: Output file path.
        force: Overwrite existing file.
    """
    if output.exists() and not force:
        typer.echo(f"❌ Configuration file already exists: {output}", err=True)
        typer.echo(f"   Use --force to overwrite", err=True)
        sys.exit(1)

    # Create default configuration content
    config_content = """# Predictor Service Configuration
# This file contains default configuration values.
# You can override these with environment variables (PREDICTOR_*)
# or command-line arguments.

[predictor]
# Server settings
host = "0.0.0.0"
port = 8000
reload = false
workers = 1

# Storage settings
storage_dir = "models"

# Logging settings
log_level = "info"

# Application metadata
app_name = "Runtime Predictor Service"
app_version = "0.1.0"
"""

    try:
        output.write_text(config_content)
        typer.echo(f"✅ Created configuration file: {output}")
        typer.echo(f"\nYou can now:")
        typer.echo(f"  1. Edit {output} to customize your settings")
        typer.echo(f"  2. Start the service with: spredictor start --config {output}")
    except Exception as e:
        logger.error(
            f"Failed to create configuration file\n"
            f"Output path: {output}\n"
            f"Force overwrite: {force}\n"
            f"Exception: {type(e).__name__}: {str(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        typer.echo(f"❌ Error creating configuration file: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
