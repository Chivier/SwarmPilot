"""CLI entry point for the Planner service.

This module provides a command-line interface using Typer for managing
the planner service, including deployment, scaling, and instance
management via the Planner REST API.
"""

from __future__ import annotations

import os
import sys

import httpx
import typer
import uvicorn

_DEFAULT_PLANNER_URL = "http://localhost:8002"

app = typer.Typer(
    name="splanner",
    help="Planner service command-line interface",
    add_completion=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

# Disable rich/colors globally for typer
os.environ["NO_COLOR"] = "1"


def _get_planner_url(url: str | None) -> str:
    """Resolve the planner base URL.

    Checks, in order: the explicit *url* argument, the
    ``PLANNER_URL`` environment variable, and the compiled-in
    default.

    Args:
        url: Explicit URL passed via ``--planner-url``, or None.

    Returns:
        The resolved planner base URL (no trailing slash).
    """
    resolved = (
        url
        or os.environ.get("PLANNER_URL")
        or _DEFAULT_PLANNER_URL
    )
    return resolved.rstrip("/")


def _request(
    method: str,
    url: str,
    **kwargs: object,
) -> dict:
    """Send an HTTP request and return the JSON body.

    Args:
        method: HTTP method (``GET``, ``POST``, etc.).
        url: Full request URL.
        **kwargs: Forwarded to ``httpx.request``.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        typer.Exit: On connection or HTTP errors, after
            printing a diagnostic to stderr.
    """
    try:
        resp = httpx.request(method, url, timeout=30.0, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        typer.echo(
            f"Error: cannot connect to planner at {url}",
            err=True,
        )
        raise typer.Exit(code=1)
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", "")
        except Exception:
            detail = exc.response.text
        typer.echo(
            f"Error ({exc.response.status_code}): {detail}",
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


# ------------------------------------------------------------------
# Existing commands
# ------------------------------------------------------------------


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

    typer.echo(
        f"Server will start at: http://{final_host}:{final_port}"
    )
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
        typer.echo(
            f"Error starting planner service: {e}", err=True
        )
        sys.exit(1)


@app.command()
def version() -> None:
    """Show the planner version."""
    from swarmpilot.planner import __version__

    typer.echo(f"Planner version: {__version__}")


# ------------------------------------------------------------------
# SDK / deployment commands
# ------------------------------------------------------------------


@app.command()
def serve(
    model_or_command: str = typer.Argument(
        ..., help="Model name or shell command to deploy"
    ),
    gpu: int = typer.Option(
        1, "--gpu", help="GPUs per replica"
    ),
    replicas: int = typer.Option(
        1, "--replicas", help="Number of replicas"
    ),
    name: str | None = typer.Option(
        None, "--name", help="Deployment name"
    ),
    scheduler: str | None = typer.Option(
        "auto",
        "--scheduler",
        help="Scheduler URL, 'auto', or 'none'",
    ),
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Deploy a model service.

    Examples:
        $ splanner serve Qwen/Qwen3-0.6B --gpu 1
        $ splanner serve "vllm serve my-model" --replicas 2
    """
    base = _get_planner_url(planner_url)
    sched_value = None if scheduler == "none" else scheduler
    payload = {
        "model_or_command": model_or_command,
        "gpu_count": gpu,
        "replicas": replicas,
        "scheduler": sched_value,
    }
    if name is not None:
        payload["name"] = name

    data = _request("POST", f"{base}/v1/serve", json=payload)

    typer.echo(f"Name:      {data.get('name', '')}")
    typer.echo(f"Model:     {data.get('model', '')}")
    typer.echo(f"Replicas:  {data.get('replicas', '')}")
    typer.echo(f"Success:   {data.get('success', '')}")
    instances = data.get("instances", [])
    if instances:
        typer.echo(f"Instances: {', '.join(instances)}")
    sched_url = data.get("scheduler_url")
    if sched_url:
        typer.echo(f"Scheduler: {sched_url}")
    error = data.get("error")
    if error:
        typer.echo(f"Error:     {error}")


@app.command()
def run(
    command: str = typer.Argument(
        ..., help="Shell command to execute"
    ),
    name: str = typer.Option(
        ..., "--name", help="Deployment name"
    ),
    gpu: int = typer.Option(
        1, "--gpu", help="GPUs per replica"
    ),
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Run a custom workload.

    Examples:
        $ splanner run "python train.py" --name my-job
    """
    base = _get_planner_url(planner_url)
    payload = {
        "command": command,
        "name": name,
        "gpu_count": gpu,
    }
    data = _request("POST", f"{base}/v1/run", json=payload)

    typer.echo(f"Name:      {data.get('name', '')}")
    typer.echo(f"Command:   {data.get('command', '')}")
    typer.echo(f"Replicas:  {data.get('replicas', '')}")
    typer.echo(f"Success:   {data.get('success', '')}")
    instances = data.get("instances", [])
    if instances:
        typer.echo(f"Instances: {', '.join(instances)}")
    error = data.get("error")
    if error:
        typer.echo(f"Error:     {error}")


@app.command()
def register(
    model: str = typer.Argument(
        ..., help="Model identifier"
    ),
    gpu: int = typer.Option(
        1, "--gpu", help="GPUs per replica"
    ),
    replicas: int = typer.Option(
        1, "--replicas", help="Desired replica count"
    ),
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Register model requirements for optimized deployment.

    Examples:
        $ splanner register Qwen/Qwen3-0.6B --gpu 1
        $ splanner register my-model --gpu 2 --replicas 3
    """
    base = _get_planner_url(planner_url)
    payload = {
        "model": model,
        "gpu_count": gpu,
        "replicas": replicas,
    }
    data = _request(
        "POST", f"{base}/v1/register", json=payload
    )

    typer.echo(f"Status: {data.get('status', '')}")
    typer.echo(f"Model:  {data.get('model', '')}")


@app.command()
def deploy(
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Trigger optimized deployment of registered models.

    Examples:
        $ splanner deploy
    """
    base = _get_planner_url(planner_url)
    data = _request("POST", f"{base}/v1/deploy")

    typer.echo(f"Success:   {data.get('success', '')}")
    models = data.get("deployed_models", [])
    if models:
        typer.echo(f"Models:    {', '.join(models)}")
    typer.echo(
        f"Instances: {data.get('total_instances', 0)}"
    )
    error = data.get("error")
    if error:
        typer.echo(f"Error:     {error}")


@app.command()
def ps(
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """List all managed instances.

    Examples:
        $ splanner ps
    """
    base = _get_planner_url(planner_url)
    instances = _request("GET", f"{base}/v1/instances")

    if not instances:
        typer.echo("No instances running.")
        return

    # Column headers and widths
    hdr = (
        f"{'PYLET_ID':<14}  {'INSTANCE_ID':<14}  "
        f"{'MODEL':<28}  {'STATUS':<10}  "
        f"{'GPU':<4}  {'ENDPOINT'}"
    )
    typer.echo(hdr)
    typer.echo("-" * len(hdr))
    for inst in instances:
        typer.echo(
            f"{inst.get('pylet_id', ''):<14}  "
            f"{inst.get('instance_id', ''):<14}  "
            f"{inst.get('model_id', ''):<28}  "
            f"{inst.get('status', ''):<10}  "
            f"{inst.get('gpu_count', ''):<4}  "
            f"{inst.get('endpoint', '') or ''}"
        )


@app.command()
def scale(
    model: str = typer.Argument(
        ..., help="Model identifier to scale"
    ),
    replicas: int = typer.Option(
        ..., "--replicas", help="Target replica count"
    ),
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Scale model replicas to a target count.

    Examples:
        $ splanner scale Qwen/Qwen3-0.6B --replicas 3
    """
    base = _get_planner_url(planner_url)
    payload = {
        "model": model,
        "replicas": replicas,
    }
    data = _request("POST", f"{base}/v1/scale", json=payload)

    typer.echo(f"Model:    {data.get('model', '')}")
    typer.echo(f"Previous: {data.get('previous_count', '')}")
    typer.echo(f"Current:  {data.get('current_count', '')}")
    typer.echo(f"Success:  {data.get('success', '')}")
    error = data.get("error")
    if error:
        typer.echo(f"Error:    {error}")


@app.command()
def terminate(
    name: str | None = typer.Argument(
        None, help="Deployment name to terminate"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Terminate by model identifier"
    ),
    all_instances: bool = typer.Option(
        False, "--all", help="Terminate all instances"
    ),
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Terminate instances by name, model, or all.

    Examples:
        $ splanner terminate my-deployment
        $ splanner terminate --model Qwen/Qwen3-0.6B
        $ splanner terminate --all
    """
    base = _get_planner_url(planner_url)
    payload: dict[str, object] = {}
    if name is not None:
        payload["name"] = name
    if model is not None:
        payload["model"] = model
    if all_instances:
        payload["all"] = True

    data = _request(
        "POST", f"{base}/v1/terminate", json=payload
    )

    typer.echo(f"Success:    {data.get('success', '')}")
    typer.echo(
        f"Terminated: {data.get('terminated_count', 0)}"
    )
    msg = data.get("message", "")
    if msg:
        typer.echo(f"Message:    {msg}")
    error = data.get("error")
    if error:
        typer.echo(f"Error:      {error}")


@app.command()
def schedulers(
    planner_url: str | None = typer.Option(
        None,
        "--planner-url",
        envvar="PLANNER_URL",
        help="Planner service URL",
    ),
) -> None:
    """Show scheduler-to-model mapping.

    Examples:
        $ splanner schedulers
    """
    base = _get_planner_url(planner_url)
    data = _request("GET", f"{base}/v1/schedulers")

    mapping = data.get("schedulers", {})
    total = data.get("total", 0)

    if not mapping:
        typer.echo("No schedulers registered.")
        return

    typer.echo(f"Registered schedulers ({total}):")
    for model_id, sched_url in mapping.items():
        typer.echo(f"  {model_id} -> {sched_url}")


if __name__ == "__main__":
    app()
