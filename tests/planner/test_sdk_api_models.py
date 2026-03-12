"""Tests for SDK API request/response models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swarmpilot.planner.models.sdk_api import (
    DeployResponse,
    InstanceDetailResponse,
    RegisteredModelsResponse,
    RegisterRequest,
    RunRequest,
    RunResponse,
    ScaleRequest,
    ScaleResponse,
    SchedulerMapResponse,
    ServeRequest,
    ServeResponse,
    TerminateRequest,
    TerminateResponse,
)


# -------------------------------------------------------------------
# ServeRequest
# -------------------------------------------------------------------


class TestServeRequest:
    """Tests for ServeRequest model."""

    def test_valid_with_model_name(self) -> None:
        """Test creation with a model name."""
        req = ServeRequest(model_or_command="llama-7b")
        assert req.model_or_command == "llama-7b"
        assert req.name is None
        assert req.replicas == 1
        assert req.gpu_count == 1
        assert req.backend == "vllm"
        assert req.scheduler == "auto"

    def test_scheduler_accepts_auto(self) -> None:
        """Test scheduler field accepts 'auto'."""
        req = ServeRequest(
            model_or_command="model-a", scheduler="auto"
        )
        assert req.scheduler == "auto"

    def test_scheduler_accepts_none(self) -> None:
        """Test scheduler field accepts None."""
        req = ServeRequest(
            model_or_command="model-a", scheduler=None
        )
        assert req.scheduler is None

    def test_scheduler_accepts_url(self) -> None:
        """Test scheduler field accepts a URL string."""
        url = "http://scheduler:8000"
        req = ServeRequest(
            model_or_command="model-a", scheduler=url
        )
        assert req.scheduler == url

    def test_replicas_defaults_to_one(self) -> None:
        """Test replicas defaults to 1."""
        req = ServeRequest(model_or_command="m")
        assert req.replicas == 1

    def test_custom_values(self) -> None:
        """Test creation with all fields specified."""
        req = ServeRequest(
            model_or_command="Qwen/Qwen3-0.6B",
            name="my-instance",
            replicas=4,
            gpu_count=8,
            backend="sglang",
            scheduler="http://sched:8000",
        )
        assert req.name == "my-instance"
        assert req.replicas == 4
        assert req.gpu_count == 8
        assert req.backend == "sglang"


# -------------------------------------------------------------------
# RunRequest
# -------------------------------------------------------------------


class TestRunRequest:
    """Tests for RunRequest model."""

    def test_command_required(self) -> None:
        """Test that command is required."""
        with pytest.raises(ValidationError) as exc_info:
            RunRequest()
        assert "command" in str(exc_info.value)

    def test_gpu_defaults_to_one(self) -> None:
        """Test gpu_count defaults to 1."""
        req = RunRequest(command="train.py")
        assert req.gpu_count == 1

    def test_valid_creation(self) -> None:
        """Test valid RunRequest with all optional fields."""
        req = RunRequest(
            command="python train.py",
            name="trainer",
            replicas=2,
            gpu_count=4,
        )
        assert req.command == "python train.py"
        assert req.name == "trainer"
        assert req.replicas == 2
        assert req.gpu_count == 4


# -------------------------------------------------------------------
# RegisterRequest
# -------------------------------------------------------------------


class TestRegisterRequest:
    """Tests for RegisterRequest model."""

    def test_model_required(self) -> None:
        """Test that model is required."""
        with pytest.raises(ValidationError) as exc_info:
            RegisterRequest()
        assert "model" in str(exc_info.value)

    def test_valid_creation(self) -> None:
        """Test valid RegisterRequest creation."""
        req = RegisterRequest(
            model="llama-7b", replicas=3, gpu_count=2
        )
        assert req.model == "llama-7b"
        assert req.replicas == 3
        assert req.gpu_count == 2
        assert req.backend == "vllm"
        assert req.priority == 1.0


# -------------------------------------------------------------------
# ScaleRequest
# -------------------------------------------------------------------


class TestScaleRequest:
    """Tests for ScaleRequest model."""

    def test_valid_creation(self) -> None:
        """Test valid ScaleRequest creation."""
        req = ScaleRequest(model="llama-7b", replicas=3)
        assert req.model == "llama-7b"
        assert req.replicas == 3

    def test_replicas_zero_allowed(self) -> None:
        """Test scaling to zero replicas is permitted."""
        req = ScaleRequest(model="model-x", replicas=0)
        assert req.replicas == 0


# -------------------------------------------------------------------
# TerminateRequest
# -------------------------------------------------------------------


class TestTerminateRequest:
    """Tests for TerminateRequest model."""

    def test_name_only(self) -> None:
        """Test termination by name only."""
        req = TerminateRequest(name="instance-1")
        assert req.name == "instance-1"
        assert req.model is None
        assert req.all is False

    def test_model_only(self) -> None:
        """Test termination by model only."""
        req = TerminateRequest(model="llama-7b")
        assert req.model == "llama-7b"

    def test_all_flag(self) -> None:
        """Test termination with all=True."""
        req = TerminateRequest(all=True)
        assert req.all is True

    def test_empty_is_valid(self) -> None:
        """Test empty request is valid (no validator)."""
        req = TerminateRequest()
        assert req.name is None
        assert req.model is None
        assert req.all is False


# -------------------------------------------------------------------
# Response serialization
# -------------------------------------------------------------------


class TestResponseSerialization:
    """Tests for response model creation and serialization."""

    def test_serve_response(self) -> None:
        """Test ServeResponse creation."""
        resp = ServeResponse(
            success=True,
            name="llama-group",
            model="llama-7b",
            replicas=2,
            instances=["inst-1", "inst-2"],
            scheduler_url="http://sched:8000",
        )
        assert resp.name == "llama-group"
        assert resp.success is True
        assert len(resp.instances) == 2

        data = resp.model_dump()
        assert data["model"] == "llama-7b"

    def test_run_response(self) -> None:
        """Test RunResponse creation."""
        resp = RunResponse(
            success=True,
            name="trainer",
            command="python train.py",
            replicas=1,
        )
        assert resp.success is True
        assert resp.error is None

    def test_deploy_response(self) -> None:
        """Test DeployResponse creation."""
        resp = DeployResponse(
            success=True,
            deployed_models=["model-a", "model-b"],
            total_instances=5,
        )
        assert resp.success is True
        assert len(resp.deployed_models) == 2
        assert resp.total_instances == 5

    def test_instance_detail_response(self) -> None:
        """Test InstanceDetailResponse fields."""
        resp = InstanceDetailResponse(
            pylet_id="p-1",
            instance_id="inst-1",
            model_id="llama-7b",
            endpoint="http://host:8080",
            status="active",
            gpu_count=4,
        )
        assert resp.gpu_count == 4
        assert resp.endpoint == "http://host:8080"

    def test_scale_response(self) -> None:
        """Test ScaleResponse creation."""
        resp = ScaleResponse(
            success=True,
            model="llama-7b",
            previous_count=1,
            current_count=3,
        )
        assert resp.previous_count == 1
        assert resp.current_count == 3

    def test_terminate_response(self) -> None:
        """Test TerminateResponse creation."""
        resp = TerminateResponse(
            success=True,
            terminated_count=2,
            message="Terminated 2 instances",
        )
        assert resp.terminated_count == 2


# -------------------------------------------------------------------
# SchedulerMapResponse
# -------------------------------------------------------------------


class TestSchedulerMapResponse:
    """Tests for SchedulerMapResponse dict format."""

    def test_dict_format(self) -> None:
        """Test schedulers is a model->URL mapping."""
        resp = SchedulerMapResponse(
            schedulers={
                "llama-7b": "http://sched-a:8000",
                "mistral-7b": "http://sched-b:8000",
            },
            total=2,
        )
        assert resp.schedulers["llama-7b"] == "http://sched-a:8000"
        assert resp.total == 2

    def test_empty_schedulers(self) -> None:
        """Test empty scheduler map is valid."""
        resp = SchedulerMapResponse(schedulers={}, total=0)
        assert resp.schedulers == {}


# -------------------------------------------------------------------
# RegisteredModelsResponse
# -------------------------------------------------------------------


class TestRegisteredModelsResponse:
    """Tests for RegisteredModelsResponse model."""

    def test_models_dict(self) -> None:
        """Test models mapping contains RegisterRequest values."""
        reg = RegisterRequest(model="llama-7b", gpu_count=2)
        resp = RegisteredModelsResponse(
            models={"llama-7b": reg}, total=1
        )
        assert resp.models["llama-7b"].gpu_count == 2
        assert resp.total == 1

    def test_serialization(self) -> None:
        """Test nested RegisterRequest serialises correctly."""
        reg = RegisterRequest(model="mistral-7b", gpu_count=4)
        resp = RegisteredModelsResponse(
            models={"mistral-7b": reg}, total=1
        )
        data = resp.model_dump()
        assert data["models"]["mistral-7b"]["gpu_count"] == 4
