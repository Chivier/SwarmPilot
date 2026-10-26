"""Adapter-driven predictor features and model keys (plan item 1.4)."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from swarmpilot.engines import (
    EngineAdapter,
    LaunchSpec,
    RouteSpec,
    WorkloadSpec,
    WorkloadType,
    register_adapter,
    unregister_adapter,
)
from swarmpilot.errors import ModelNotFoundError
from swarmpilot.predictor.storage.model_storage import ModelStorage
from swarmpilot.scheduler.clients.predictor_library_client import (
    PredictorClient,
)
from swarmpilot.scheduler.clients.training_library_client import (
    TrainingClient,
)
from swarmpilot.scheduler.models import Instance
from swarmpilot.scheduler.proxy.router import _extract_features
from swarmpilot.scheduler.services.task_result_callback import (
    training_features,
)
from swarmpilot.scheduler.services.worker_queue_manager import (
    complete_platform_info,
)


class _ToyDiffusionAdapter(EngineAdapter):
    """Image engine whose runtime depends on steps and pixels."""

    name: ClassVar[str] = "toy-diffusion"
    workload: ClassVar[WorkloadSpec] = WorkloadSpec(
        type=WorkloadType.DIFFUSION_IMAGE,
        feature_names=("steps", "megapixels"),
    )

    def launch_spec(
        self,
        model_id: str,
        *,
        gpus: int = 1,
        extra_args: tuple[str, ...] = (),
    ) -> LaunchSpec:
        return LaunchSpec(argv=("toy", model_id, *extra_args))

    def routes(self) -> tuple[RouteSpec, ...]:
        return (RouteSpec(path="v1/images/generations"),)

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        w, h = int(body.get("width", 512)), int(body.get("height", 512))
        return {
            "steps": float(body.get("num_inference_steps", 30)),
            "megapixels": w * h / 1e6,
        }


@pytest.fixture
def toy_diffusion():
    register_adapter(_ToyDiffusionAdapter)
    yield _ToyDiffusionAdapter()
    unregister_adapter("toy-diffusion")


def _pinfo(engine: str, version: str = "0.11") -> dict[str, str]:
    return {
        "software_name": engine,
        "software_version": version,
        "hardware_name": "cpu",
    }


class TestModelKey:
    """Model storage keys."""

    def test_legacy_key_unchanged(self):
        key = ModelStorage.generate_model_key(
            None, "m", _pinfo("x"), "quantile"
        )
        assert key == "m__x-0.11__cpu__quantile"

    def test_key_holds_workload_and_schema(self):
        pinfo = complete_platform_info(_pinfo("vllm"))
        key = ModelStorage.generate_model_key(None, "m", pinfo, "quantile")
        assert key == "m__llm_chat__vllm-0.11__cpu__fs1__quantile"

    def test_version_and_schema_change_the_key(self):
        base = complete_platform_info(_pinfo("vllm"))
        keys = {
            ModelStorage.generate_model_key(None, "m", p, "quantile")
            for p in (
                base,
                {**base, "software_version": "0.12"},
                {**base, "feature_schema": "2"},
            )
        }
        assert len(keys) == 3


class TestCompletePlatformInfo:
    """Workload keys filled in at registration."""

    def test_known_engine_gets_workload_keys(self):
        out = complete_platform_info(_pinfo("sglang"))
        assert out["workload"] == "llm_chat"
        assert out["feature_schema"] == "1"

    def test_unknown_engine_untouched(self):
        pinfo = _pinfo("openai-online")
        assert complete_platform_info(pinfo) is pinfo

    def test_existing_workload_kept(self):
        pinfo = {**_pinfo("vllm"), "workload": "custom"}
        assert complete_platform_info(pinfo)["workload"] == "custom"


class TestFeatureExtraction:
    """Features come from the engine adapter."""

    def test_llm_features_from_adapter(self):
        body = {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 7,
        }
        feats = _extract_features(_pinfo("vllm"), "v1/chat/completions", body)
        assert feats == {"prompt_length": 5, "max_tokens": 7}

    def test_unknown_engine_keeps_legacy_features(self):
        feats = _extract_features(_pinfo("online"), "v1/x", {"prompt": "abc"})
        assert feats == {"prompt_length": 3}

    def test_bad_max_tokens_falls_back_to_default(self):
        body = {"messages": [{"content": "x"}], "max_tokens": "lots"}
        feats = _extract_features(_pinfo("vllm"), "v1/chat/completions", body)
        assert feats == {"prompt_length": 1, "max_tokens": 1024}

    def test_adapter_error_gives_no_features(self, toy_diffusion):
        body = {"width": "wide"}
        pinfo = _pinfo("toy-diffusion")
        assert _extract_features(pinfo, "v1/images/generations", body) == {}

    def test_plugin_adapter_features(self, toy_diffusion):
        body = {"num_inference_steps": 20, "width": 1000, "height": 1000}
        feats = _extract_features(
            _pinfo("toy-diffusion"), "v1/images/generations", body
        )
        assert feats == {"steps": 20.0, "megapixels": 1.0}

    def test_training_uses_only_adapter_features(self):
        task = SimpleNamespace(
            metadata={
                "path": "v1/chat/completions",
                "method": "POST",
                "proxy": True,
                "prompt_length": 5,
                "max_tokens": 7,
            },
            task_input={},
        )
        assert training_features(task, _pinfo("vllm")) == {
            "prompt_length": 5,
            "max_tokens": 7,
        }

    def test_task_api_keeps_its_metadata(self):
        task = SimpleNamespace(
            metadata={"batch": 4}, task_input={"prompt": "abcd"}
        )
        assert training_features(task, _pinfo("vllm")) == {
            "batch": 4,
            "prompt_length": 4,
        }


def _collect(
    tmp_path, engine: str, bodies: list[dict], runtime
) -> tuple[PredictorClient, TrainingClient, dict[str, str]]:
    predictor = PredictorClient(storage_dir=str(tmp_path))
    trainer = TrainingClient(
        storage=predictor._low_level._storage,
        cache=predictor._low_level._cache,
        chain_builder=predictor._chain_builder,
        batch_size=10_000,
        min_samples=10,
        prediction_types=["expect_error"],
    )
    pinfo = complete_platform_info(_pinfo(engine))
    for body in bodies:
        feats = _extract_features(pinfo, "p", body)
        task = SimpleNamespace(metadata={"proxy": True, **feats}, task_input={})
        trainer.add_sample(
            model_id="m",
            platform_info=pinfo,
            features=training_features(task, pinfo),
            actual_runtime_ms=runtime(feats),
        )
    return predictor, trainer, pinfo


async def _predict(predictor, pinfo, body) -> float:
    feats = _extract_features(pinfo, "p", body)
    inst = Instance(
        instance_id="i0", model_id="m", endpoint="http://x", platform_info=pinfo
    )
    [pred] = await predictor.predict(
        "m", {"path": "p", "proxy": True, **feats}, [inst], "expect_error"
    )
    return pred.predicted_time_ms


@pytest.mark.slow
class TestTrainAndServe:
    """Each workload trains a model and predicts with it.

    A new engine version is cold instead of reusing that model.
    """

    async def test_llm_workload(self, tmp_path):
        bodies = [
            {"prompt": "x" * (10 * i), "max_tokens": 16 * (i % 5 + 1)}
            for i in range(40)
        ]
        predictor, trainer, pinfo = _collect(
            tmp_path,
            "vllm",
            bodies,
            lambda f: 1 + 0.1 * f["prompt_length"] + f["max_tokens"],
        )
        assert await trainer.flush()
        assert await _predict(predictor, pinfo, bodies[3]) > 0

        cold = {**pinfo, "software_version": "0.12"}
        with pytest.raises(ModelNotFoundError):
            await _predict(predictor, cold, bodies[3])

    async def test_diffusion_workload(self, tmp_path, toy_diffusion):
        bodies = [
            {"num_inference_steps": 10 + i, "width": 512, "height": 512}
            for i in range(40)
        ]
        predictor, trainer, pinfo = _collect(
            tmp_path,
            "toy-diffusion",
            bodies,
            lambda f: 40 * f["steps"] * f["megapixels"],
        )
        assert pinfo["workload"] == "diffusion_image"
        assert await trainer.flush()
        assert await _predict(predictor, pinfo, bodies[5]) > 0
