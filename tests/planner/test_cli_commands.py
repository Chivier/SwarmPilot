"""Tests for splanner CLI commands.

Validates that each CLI command sends the correct HTTP request to
the Planner REST API and formats the response for the terminal.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from swarmpilot.planner.cli import app

runner = CliRunner()


def _mock_response(
    json_data: dict | list,
    status_code: int = 200,
) -> httpx.Response:
    """Build a fake ``httpx.Response``.

    Args:
        json_data: JSON body to return.
        status_code: HTTP status code.

    Returns:
        A minimal ``httpx.Response`` suitable for mocking.
    """
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://test"),
    )
    return resp


# ------------------------------------------------------------------
# serve
# ------------------------------------------------------------------


class TestServeCommand:
    """Tests for ``splanner serve``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_serve_sends_correct_payload(self, mock_req: MagicMock) -> None:
        """Serve command posts model_or_command, gpu, replicas."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "name": "Qwen-Qwen3-0.6B",
                "model": "Qwen/Qwen3-0.6B",
                "replicas": 2,
                "instances": ["i-001", "i-002"],
                "scheduler_url": "http://sched:8000",
                "error": None,
            }
        )

        result = runner.invoke(
            app,
            ["serve", "Qwen/Qwen3-0.6B", "--gpu", "1", "--replicas", "2"],
        )

        assert result.exit_code == 0, result.output
        # Verify payload
        call_kwargs = mock_req.call_args
        assert call_kwargs[0][0] == "POST"
        assert call_kwargs[0][1].endswith("/v1/serve")
        body = call_kwargs[1]["json"]
        assert body["model_or_command"] == "Qwen/Qwen3-0.6B"
        assert body["gpu_count"] == 1
        assert body["replicas"] == 2
        assert body["scheduler"] == "auto"

        # Verify output
        assert "Qwen-Qwen3-0.6B" in result.output
        assert "i-001" in result.output
        assert "http://sched:8000" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_serve_scheduler_none(self, mock_req: MagicMock) -> None:
        """``--scheduler none`` sends null scheduler value."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "name": "test",
                "model": "m",
                "replicas": 1,
                "instances": [],
                "scheduler_url": None,
                "error": None,
            }
        )

        result = runner.invoke(
            app,
            [
                "serve",
                "my-model",
                "--gpu",
                "1",
                "--scheduler",
                "none",
            ],
        )

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["scheduler"] is None

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_serve_with_name(self, mock_req: MagicMock) -> None:
        """``--name`` is forwarded in the payload."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "name": "custom",
                "model": "m",
                "replicas": 1,
                "instances": [],
                "scheduler_url": None,
                "error": None,
            }
        )

        result = runner.invoke(
            app,
            [
                "serve",
                "Qwen/Qwen3-0.6B",
                "--gpu",
                "1",
                "--name",
                "custom",
            ],
        )

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["name"] == "custom"


# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------


class TestRunCommand:
    """Tests for ``splanner run``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_run_sends_correct_payload(self, mock_req: MagicMock) -> None:
        """Run command posts command, name, and gpu_count."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "name": "train-job",
                "command": "python train.py",
                "replicas": 1,
                "instances": ["i-100"],
                "error": None,
            }
        )

        result = runner.invoke(
            app,
            [
                "run",
                "python train.py",
                "--name",
                "train-job",
                "--gpu",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["command"] == "python train.py"
        assert body["name"] == "train-job"
        assert body["gpu_count"] == 2

        assert "train-job" in result.output
        assert "python train.py" in result.output


# ------------------------------------------------------------------
# register + deploy
# ------------------------------------------------------------------


class TestRegisterDeployFlow:
    """Tests for ``splanner register`` and ``splanner deploy``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_register_sends_correct_payload(self, mock_req: MagicMock) -> None:
        """Register posts model, gpu_count, replicas."""
        mock_req.return_value = _mock_response(
            {"status": "registered", "model": "Qwen/Qwen3-0.6B"}
        )

        result = runner.invoke(
            app,
            [
                "register",
                "Qwen/Qwen3-0.6B",
                "--gpu",
                "1",
                "--replicas",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["model"] == "Qwen/Qwen3-0.6B"
        assert body["gpu_count"] == 1
        assert body["replicas"] == 2

        assert "registered" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_deploy_sends_post(self, mock_req: MagicMock) -> None:
        """Deploy issues a POST with no payload."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "deployed_models": ["modelA", "modelB"],
                "total_instances": 4,
                "error": None,
            }
        )

        result = runner.invoke(app, ["deploy"])

        assert result.exit_code == 0, result.output
        call_kwargs = mock_req.call_args
        assert call_kwargs[0][0] == "POST"
        assert call_kwargs[0][1].endswith("/v1/deploy")

        assert "modelA" in result.output
        assert "4" in result.output


# ------------------------------------------------------------------
# ps
# ------------------------------------------------------------------


class TestPsCommand:
    """Tests for ``splanner ps``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_ps_lists_instances(self, mock_req: MagicMock) -> None:
        """Ps renders a table of active instances."""
        mock_req.return_value = _mock_response(
            [
                {
                    "pylet_id": "p-001",
                    "instance_id": "i-001",
                    "model_id": "Qwen/Qwen3-0.6B",
                    "endpoint": "http://w1:8080",
                    "status": "active",
                    "gpu_count": 1,
                    "error": None,
                },
                {
                    "pylet_id": "p-002",
                    "instance_id": "i-002",
                    "model_id": "meta/Llama-3",
                    "endpoint": "http://w2:8080",
                    "status": "active",
                    "gpu_count": 2,
                    "error": None,
                },
            ]
        )

        result = runner.invoke(app, ["ps"])

        assert result.exit_code == 0, result.output
        assert "p-001" in result.output
        assert "i-002" in result.output
        assert "Qwen/Qwen3-0.6B" in result.output
        assert "meta/Llama-3" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_ps_empty(self, mock_req: MagicMock) -> None:
        """Ps prints a message when no instances exist."""
        mock_req.return_value = _mock_response([])

        result = runner.invoke(app, ["ps"])

        assert result.exit_code == 0, result.output
        assert "No instances running" in result.output


# ------------------------------------------------------------------
# scale
# ------------------------------------------------------------------


class TestScaleCommand:
    """Tests for ``splanner scale``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_scale_sends_correct_payload(self, mock_req: MagicMock) -> None:
        """Scale posts model and target replicas."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "model": "Qwen/Qwen3-0.6B",
                "previous_count": 1,
                "current_count": 3,
                "error": None,
            }
        )

        result = runner.invoke(
            app,
            ["scale", "Qwen/Qwen3-0.6B", "--replicas", "3"],
        )

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["model"] == "Qwen/Qwen3-0.6B"
        assert body["replicas"] == 3

        assert "Previous: 1" in result.output
        assert "Current:  3" in result.output


# ------------------------------------------------------------------
# terminate
# ------------------------------------------------------------------


class TestTerminateCommand:
    """Tests for ``splanner terminate``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_terminate_by_name(self, mock_req: MagicMock) -> None:
        """Terminate with a positional name argument."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "terminated_count": 1,
                "message": "Terminated 1 instances matching 'my-dep'",
                "error": None,
            }
        )

        result = runner.invoke(app, ["terminate", "my-dep"])

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["name"] == "my-dep"
        assert "model" not in body
        assert "all" not in body

        assert "Terminated: 1" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_terminate_by_model(self, mock_req: MagicMock) -> None:
        """Terminate by --model flag."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "terminated_count": 2,
                "message": "Terminated 2 instances for model Qwen",
                "error": None,
            }
        )

        result = runner.invoke(app, ["terminate", "--model", "Qwen"])

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["model"] == "Qwen"
        assert "name" not in body

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_terminate_all(self, mock_req: MagicMock) -> None:
        """Terminate with --all flag."""
        mock_req.return_value = _mock_response(
            {
                "success": True,
                "terminated_count": 5,
                "message": "Terminated 5 instances",
                "error": None,
            }
        )

        result = runner.invoke(app, ["terminate", "--all"])

        assert result.exit_code == 0, result.output
        body = mock_req.call_args[1]["json"]
        assert body["all"] is True
        assert "name" not in body
        assert "model" not in body


# ------------------------------------------------------------------
# schedulers
# ------------------------------------------------------------------


class TestSchedulersCommand:
    """Tests for ``splanner schedulers``."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_schedulers_returns_mapping(self, mock_req: MagicMock) -> None:
        """Schedulers prints model-to-URL mapping."""
        mock_req.return_value = _mock_response(
            {
                "schedulers": {
                    "Qwen/Qwen3-0.6B": "http://sched1:8000",
                    "meta/Llama-3": "http://sched2:8000",
                },
                "total": 2,
            }
        )

        result = runner.invoke(app, ["schedulers"])

        assert result.exit_code == 0, result.output
        assert "Qwen/Qwen3-0.6B" in result.output
        assert "http://sched1:8000" in result.output
        assert "meta/Llama-3" in result.output
        assert "(2)" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_schedulers_empty(self, mock_req: MagicMock) -> None:
        """Empty scheduler map prints informational message."""
        mock_req.return_value = _mock_response({"schedulers": {}, "total": 0})

        result = runner.invoke(app, ["schedulers"])

        assert result.exit_code == 0, result.output
        assert "No schedulers registered" in result.output


# ------------------------------------------------------------------
# --planner-url option
# ------------------------------------------------------------------


class TestPlannerUrlOption:
    """Tests for the ``--planner-url`` override."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_planner_url_overrides_default(self, mock_req: MagicMock) -> None:
        """Explicit --planner-url is used for the request."""
        mock_req.return_value = _mock_response({"schedulers": {}, "total": 0})

        result = runner.invoke(
            app,
            [
                "schedulers",
                "--planner-url",
                "http://custom:9999",
            ],
        )

        assert result.exit_code == 0, result.output
        url = mock_req.call_args[0][1]
        assert url.startswith("http://custom:9999/")

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_planner_url_env_var(
        self, mock_req: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PLANNER_URL env var is respected."""
        monkeypatch.setenv("PLANNER_URL", "http://env:7777")
        mock_req.return_value = _mock_response({"schedulers": {}, "total": 0})

        result = runner.invoke(app, ["schedulers"])

        assert result.exit_code == 0, result.output
        url = mock_req.call_args[0][1]
        assert url.startswith("http://env:7777/")

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_planner_url_default(self, mock_req: MagicMock) -> None:
        """Default URL is http://localhost:8002."""
        mock_req.return_value = _mock_response({"schedulers": {}, "total": 0})

        result = runner.invoke(app, ["schedulers"])

        assert result.exit_code == 0, result.output
        url = mock_req.call_args[0][1]
        assert url.startswith("http://localhost:8002/")


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------


class TestErrorHandling:
    """Tests for HTTP error handling in _request."""

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_connection_error(self, mock_req: MagicMock) -> None:
        """Connection failures produce a readable message."""
        mock_req.side_effect = httpx.ConnectError("Connection refused")

        result = runner.invoke(app, ["ps"])

        assert result.exit_code == 1
        assert "cannot connect" in result.output

    @patch("swarmpilot.planner.cli.httpx.request")
    def test_http_error(self, mock_req: MagicMock) -> None:
        """HTTP errors surface the detail from the response."""
        mock_req.return_value = _mock_response(
            {"detail": "PyLet is not enabled"},
            status_code=503,
        )

        result = runner.invoke(app, ["deploy"])

        assert result.exit_code == 1
        assert "503" in result.output
        assert "PyLet is not enabled" in result.output
