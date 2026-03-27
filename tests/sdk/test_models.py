"""Tests for SDK data models."""

from __future__ import annotations

import pytest

from swarmpilot.sdk.models import (
    DeploymentResult,
    Instance,
    InstanceGroup,
    ModelStatus,
    PredictResult,
    PreprocessorInfo,
    Process,
    TrainResult,
)


class TestInstance:
    """Instance dataclass stores all deployment metadata."""

    def test_create_with_all_fields(self) -> None:
        """All fields are preserved after construction."""
        inst = Instance(
            name="worker-1",
            model="llama-7b",
            command="vllm serve llama-7b",
            endpoint="http://localhost:8080",
            scheduler="http://scheduler:8000",
            status="running",
            gpu=1,
        )
        assert inst.name == "worker-1"
        assert inst.model == "llama-7b"
        assert inst.command == "vllm serve llama-7b"
        assert inst.endpoint == "http://localhost:8080"
        assert inst.scheduler == "http://scheduler:8000"
        assert inst.status == "running"
        assert inst.gpu == 1

    def test_pending_instance_has_none_endpoint(self) -> None:
        """A pending instance has no endpoint yet."""
        inst = Instance(
            name="worker-2",
            model="llama-7b",
            command="vllm serve llama-7b",
            endpoint=None,
            scheduler="http://scheduler:8000",
            status="pending",
            gpu=2,
        )
        assert inst.endpoint is None
        assert inst.status == "pending"

    def test_custom_command_with_none_model(self) -> None:
        """Generic workloads may have model=None."""
        inst = Instance(
            name="custom-1",
            model=None,
            command="python worker.py",
            endpoint="http://localhost:9000",
            scheduler=None,
            status="running",
            gpu=0,
        )
        assert inst.model is None
        assert inst.command == "python worker.py"


class TestInstanceGroup:
    """InstanceGroup aggregates replicas and exposes endpoints."""

    def test_endpoints_returns_non_none(self) -> None:
        """The endpoints property filters out None values."""
        instances = [
            Instance(
                name="w-1",
                model="gpt-4",
                command="serve",
                endpoint="http://a:8080",
                scheduler=None,
                status="running",
                gpu=1,
            ),
            Instance(
                name="w-2",
                model="gpt-4",
                command="serve",
                endpoint="http://b:8080",
                scheduler=None,
                status="running",
                gpu=1,
            ),
        ]
        group = InstanceGroup(
            name="gpt-4-group",
            model="gpt-4",
            command="serve",
            instances=instances,
        )
        assert group.endpoints == [
            "http://a:8080",
            "http://b:8080",
        ]

    def test_endpoints_filters_none(self) -> None:
        """Pending instances with endpoint=None are excluded."""
        instances = [
            Instance(
                name="w-1",
                model="gpt-4",
                command="serve",
                endpoint="http://a:8080",
                scheduler=None,
                status="running",
                gpu=1,
            ),
            Instance(
                name="w-2",
                model="gpt-4",
                command="serve",
                endpoint=None,
                scheduler=None,
                status="pending",
                gpu=1,
            ),
        ]
        group = InstanceGroup(
            name="gpt-4-group",
            model="gpt-4",
            command="serve",
            instances=instances,
        )
        assert group.endpoints == ["http://a:8080"]


class TestProcess:
    """Process always has scheduler=None."""

    def test_scheduler_always_none(self) -> None:
        """The scheduler field is not settable via __init__."""
        proc = Process(
            name="etl-job",
            command="python etl.py",
            endpoint="http://localhost:9090",
            status="running",
            gpu=0,
        )
        assert proc.scheduler is None
        assert proc.name == "etl-job"
        assert proc.status == "running"


class TestDeploymentResult:
    """DeploymentResult supports dict-like model lookup."""

    @pytest.fixture()
    def result(self) -> DeploymentResult:
        """Build a DeploymentResult with one group."""
        group = InstanceGroup(
            name="llama-group",
            model="llama-7b",
            command="vllm serve llama-7b",
            instances=[
                Instance(
                    name="w-1",
                    model="llama-7b",
                    command="vllm serve llama-7b",
                    endpoint="http://a:8080",
                    scheduler=None,
                    status="running",
                    gpu=1,
                ),
            ],
        )
        return DeploymentResult(
            plan={"replicas": 1},
            groups={"llama-7b": group},
            status="deployed",
        )

    def test_getitem_returns_group(self, result: DeploymentResult) -> None:
        """Subscript access returns the correct InstanceGroup."""
        group = result["llama-7b"]
        assert group.name == "llama-group"
        assert group.model == "llama-7b"

    def test_getitem_missing_raises_key_error(
        self, result: DeploymentResult
    ) -> None:
        """Missing model raises KeyError."""
        with pytest.raises(KeyError):
            result["nonexistent"]


class TestPreprocessorInfo:
    """PreprocessorInfo holds preprocessor metadata."""

    def test_basic_creation(self) -> None:
        """All fields are stored correctly."""
        info = PreprocessorInfo(
            name="tokenizer",
            feature="input_ids",
            path="/models/tokenizer.json",
        )
        assert info.name == "tokenizer"
        assert info.feature == "input_ids"
        assert info.path == "/models/tokenizer.json"


class TestModelStatus:
    """ModelStatus captures predictor state for a model."""

    def test_creation_with_all_fields(self) -> None:
        """Nested PreprocessorInfo list is preserved."""
        preprocessors = [
            PreprocessorInfo(
                name="tok",
                feature="input_ids",
                path="/p/tok.json",
            ),
        ]
        status = ModelStatus(
            model="llama-7b",
            samples_collected=500,
            last_trained="2026-01-15T10:30:00Z",
            prediction_types=["expect_error", "quantile"],
            metrics={"mae": 12.5, "rmse": 18.3},
            strategy="expect_error",
            preprocessors=preprocessors,
        )
        assert status.model == "llama-7b"
        assert status.samples_collected == 500
        assert status.last_trained == "2026-01-15T10:30:00Z"
        assert status.prediction_types == [
            "expect_error",
            "quantile",
        ]
        assert status.metrics == {"mae": 12.5, "rmse": 18.3}
        assert status.strategy == "expect_error"
        assert len(status.preprocessors) == 1
        assert status.preprocessors[0].name == "tok"


class TestTrainResult:
    """TrainResult records training outcome."""

    def test_creation_with_strategy(self) -> None:
        """Strategy field is stored correctly."""
        result = TrainResult(
            model="llama-7b",
            samples_trained=200,
            metrics={"mae": 10.0},
            strategy="quantile",
        )
        assert result.model == "llama-7b"
        assert result.samples_trained == 200
        assert result.metrics == {"mae": 10.0}
        assert result.strategy == "quantile"


class TestPredictResult:
    """PredictResult adapts to expect-error or quantile mode."""

    def test_expect_error_mode(self) -> None:
        """Quantiles are None in expect-error mode."""
        result = PredictResult(
            model="llama-7b",
            expected_runtime_ms=150.0,
            error_margin_ms=20.0,
            quantiles=None,
        )
        assert result.expected_runtime_ms == 150.0
        assert result.error_margin_ms == 20.0
        assert result.quantiles is None

    def test_quantile_mode(self) -> None:
        """Point estimates are None in quantile mode."""
        result = PredictResult(
            model="llama-7b",
            expected_runtime_ms=None,
            error_margin_ms=None,
            quantiles={0.5: 140.0, 0.9: 200.0, 0.99: 350.0},
        )
        assert result.expected_runtime_ms is None
        assert result.error_margin_ms is None
        assert result.quantiles == {
            0.5: 140.0,
            0.9: 200.0,
            0.99: 350.0,
        }
