"""Adapter manifests and the portable predictor file (plan item 1.7)."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarmpilot.engines import adapter_names, get_adapter
from swarmpilot.engines.contract import check_adapter
from swarmpilot.engines.manifest import (
    SCHEMA_PATH,
    EngineManifest,
    json_schema,
    manifest_features,
)
from swarmpilot.engines.passthrough import PassthroughAdapter
from swarmpilot.engines.vllm import VLLMAdapter

EXAMPLE = Path(__file__).parents[2] / "examples" / "9.engine_plugin"


def test_schema_file_is_current():
    """Regenerate with python -m swarmpilot.engines.manifest --schema."""
    assert json.loads(SCHEMA_PATH.read_text()) == json_schema()


@pytest.mark.parametrize("name", adapter_names())
def test_builtin_manifests_validate(name):
    manifest = get_adapter(name).manifest()
    EngineManifest.model_validate(json.loads(json.dumps(manifest)))
    assert manifest["features"], f"{name} should declare extractors"
    assert manifest["python_features"] is False


def test_vllm_manifest_shape():
    m = get_adapter("vllm").manifest()
    assert m["transport"] == {"kind": "sse", "health_path": "/health"}
    assert m["capacity"]["batching"] == "continuous"
    assert [f["name"] for f in m["features"]] == ["prompt_length", "max_tokens"]
    chat = next(r for r in m["routes"] if r["path"] == "v1/chat/completions")
    assert chat["streaming"] is True
    assert "messages" in chat["request_schema"]["properties"]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 9},
        {
            "messages": [
                {"content": [{"type": "text", "text": "abc"}, {"x": 1}]},
                {"content": None},
                "junk",
            ],
            "max_completion_tokens": 3,
        },
        {"prompt": ["ab", "c", 7], "max_tokens": True},
        {"prompt": "hello", "messages": "nope", "max_tokens": float("inf")},
    ],
)
def test_llm_python_and_manifest_agree(body):
    adapter = get_adapter("sglang")
    want = adapter.features("v1/chat/completions", body)
    assert manifest_features(adapter.manifest(), body) == want


@pytest.mark.parametrize(
    "body",
    [
        {"size": "640x480", "num_inference_steps": 8, "n": 2},
        {"width": 512, "height": 256.0, "size": "1x1"},
        {"width": 512, "size": "big", "n": "3"},
    ],
)
def test_diffusion_python_and_manifest_agree(body):
    adapter = get_adapter("diffusers")
    want = adapter.features("v1/images/generations", body)
    assert manifest_features(adapter.manifest(), body) == want


def test_passthrough_runs_on_python_lane():
    m = PassthroughAdapter().manifest()
    assert m["features"] is None
    assert m["python_features"] is True


def test_plugin_without_extractors_runs_on_python_lane(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE / "toy_ranker_plugin"))
    from toy_ranker.adapter import ToyRankerAdapter

    m = ToyRankerAdapter().manifest()
    assert m["python_features"] is True
    assert m["routes"][0]["retry_safe"] is True


def test_yaml_engine_folds_defaults_into_expressions():
    from swarmpilot.engines.generic_http import make_adapter_class

    adapter = make_adapter_class(
        {
            "name": "y",
            "launch": {"argv": ["s", "--port", "{port}"]},
            "routes": [{"path": "run"}],
            "features": [
                {"name": "a", "expr": "len(items)", "default": 4},
                {"name": "b", "expr": "field(k)"},
            ],
        }
    )()
    m = adapter.manifest()
    assert m["features"] == [
        {"name": "a", "expr": "coalesce(len(items), 4.0)"},
        {"name": "b", "expr": "field(k)"},
    ]
    assert adapter.features("run", {}) == {"a": 4.0, "b": 0.0}
    assert manifest_features(m, {"items": [1], "k": 2}) == {"a": 1, "b": 2}


class _DriftingAdapter(VLLMAdapter):
    """Hand-written features that disagree with its manifest."""

    name = "drifting"

    def features(self, path: str, body: Mapping[str, Any]) -> dict[str, float]:
        feats = super().features(path, body)
        feats["prompt_length"] += 1
        return feats


def test_contract_catches_python_manifest_drift():
    problems = check_adapter(_DriftingAdapter())
    assert any("manifest features" in p for p in problems)


class _MisnamedAdapter(VLLMAdapter):
    name = "misnamed"

    def feature_extractors(self) -> tuple[tuple[str, str], ...]:
        return (("max_tokens", "field(max_tokens)"),)


def test_contract_catches_misnamed_extractors():
    problems = check_adapter(_MisnamedAdapter())
    assert any("feature_extractors() names" in p for p in problems)


# --- portable predictor file -------------------------------------------


def _forward(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Plain numpy MLP: Linear+ReLU blocks, linear output."""
    idx = sorted({int(k.split(".")[1]) for k in weights})
    for i, n in enumerate(idx):
        x = x @ weights[f"network.{n}.weight"].T + weights[f"network.{n}.bias"]
        if i < len(idx) - 1:
            x = np.maximum(x, 0)
    return x


@pytest.mark.slow
def test_portable_model_matches_torch(tmp_path):
    """The safetensors file plus sidecar reproduce torch predictions."""
    from safetensors.numpy import load_file

    from swarmpilot.predictor.predictor.expect_error import (
        ExpectErrorPredictor,
    )
    from swarmpilot.predictor.storage.model_storage import ModelStorage

    rng = np.random.default_rng(0)
    data = [
        {"a": float(a), "b": float(b), "runtime_ms": 3 * a + b * b}
        for a, b in rng.uniform(0, 10, size=(200, 2))
    ]
    predictor = ExpectErrorPredictor()
    predictor.train(features_list=data, config={"epochs": 50})
    storage = ModelStorage(str(tmp_path))
    pinfo = get_adapter("vllm").platform_info("0.11.0", "H100")
    key = storage.generate_model_key("org/m", pinfo, "expect_error")
    storage.save_model(
        key,
        predictor.get_model_state(),
        {"model_id": "org/m", "platform_info": pinfo, "prediction_type": "e"},
    )

    sidecar_path = tmp_path / f"{key}.json"
    sidecar = json.loads(sidecar_path.read_text())
    # weights_file is relative to the sidecar's directory.
    weights_path = sidecar_path.parent / sidecar["weights_file"]
    assert (
        hashlib.sha256(weights_path.read_bytes()).hexdigest()
        == sidecar["weights_sha256"]
    )
    assert sidecar["platform_info"]["workload"] == "llm_chat"
    assert sidecar["platform_info"]["software_version"] == "0.11"
    state = sidecar["state"]
    weights = load_file(str(weights_path))

    names = state["feature_names"]
    mean = np.asarray(state["feature_mean"], dtype=np.float32)
    std = np.asarray(state["feature_std"], dtype=np.float32)
    inputs = rng.uniform(0, 10, size=(1000, len(names))).astype(np.float32)
    ours = _forward(weights, (inputs - mean) / std)[:, 0]
    theirs = np.array(
        [
            predictor.predict(dict(zip(names, map(float, row))))[
                "expected_runtime_ms"
            ]
            for row in inputs
        ]
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-3)

    storage.delete_model(key)
    assert not list(tmp_path.rglob("*.safetensors"))
    assert not list(tmp_path.rglob("*.json"))
