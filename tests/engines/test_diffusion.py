"""Diffusion reference adapter and its server (plan item 1.6)."""

import base64
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from swarmpilot.engines import get_adapter
from swarmpilot.engines.contract import check_adapter
from swarmpilot.engines.diffusion import diffusion_features, parse_size
from swarmpilot.engines.servers.diffusers_server import (
    _dry_run_pipeline,
    create_app,
)

ROOT = Path(__file__).parents[2]


class TestFeatures:
    """Steps, megapixels and image count."""

    @pytest.mark.parametrize(
        ("body", "want"),
        [
            ({"size": "512x256"}, (512, 256)),
            ({"size": "1024X768"}, (1024, 768)),
            ({"width": 640, "height": 480, "size": "1x1"}, (640, 480)),
            ({"size": "big"}, (1024, 1024)),
            ({}, (1024, 1024)),
        ],
    )
    def test_size(self, body, want):
        assert parse_size(body) == want

    def test_features(self):
        body = {"size": "1000x1000", "num_inference_steps": 4, "n": 2}
        assert diffusion_features(body) == {
            "steps": 4.0,
            "megapixels": 1.0,
            "n_images": 2.0,
        }

    @pytest.mark.parametrize(
        "body",
        [{"num_inference_steps": "x", "n": True}, {"n": None}, {}],
    )
    def test_bad_values_use_defaults(self, body):
        feats = diffusion_features(body, default_steps=25)
        assert feats["steps"] == 25.0
        assert feats["n_images"] == 1.0


def test_adapter_passes_contract():
    adapter = get_adapter("diffusers")
    assert check_adapter(adapter) == []
    assert adapter.capacity().max_in_flight == 1
    assert adapter.platform_info("0.31.0", "A100")["workload"] == (
        "diffusion_image"
    )


def test_launch_spec_is_safe_argv():
    spec = get_adapter("diffusers").launch_spec("org/sdxl;reboot", gpus=1)
    assert spec.argv[:3] == (
        "python",
        "-m",
        "swarmpilot.engines.servers.diffusers_server",
    )
    assert "org/sdxl;reboot" in spec.argv
    assert "'org/sdxl;reboot'" in spec.render_shell()


def test_server_dry_run():
    app = create_app("toy", loader=lambda: _dry_run_pipeline(0.0))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/version").json() == {"version": "0.0.0-dryrun"}
        r = c.post(
            "/v1/images/generations",
            json={"prompt": "fox", "n": 2, "size": "64x64"},
        )
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data) == 2
        assert base64.b64decode(data[0]["b64_json"]).startswith(b"\x89PNG")
        assert c.post("/v1/images/generations", json={}).status_code == 400


def test_core_install_has_no_engine():
    """Pip install swarmpilot pulls no inference engine (plan 1.6)."""
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = " ".join(meta["dependencies"]).lower()
    for engine in ("vllm", "sglang", "diffusers"):
        assert engine not in core
        assert engine in " ".join(
            " ".join(v) for v in meta["optional-dependencies"].values()
        )


@pytest.mark.slow
async def test_diffusion_routes_by_predicted_runtime(tmp_path):
    """Diffusion requests route by predicted runtime.

    A trained predictor sends a big image job to the faster GPU and spills
    to the slower one once the fast queue is long enough.
    """
    from swarmpilot.scheduler.algorithms.factory import get_strategy
    from swarmpilot.scheduler.clients.predictor_library_client import (
        PredictorClient,
    )
    from swarmpilot.scheduler.clients.training_library_client import (
        TrainingClient,
    )
    from swarmpilot.scheduler.models import Instance
    from swarmpilot.scheduler.proxy.router import _extract_features
    from swarmpilot.scheduler.registry.instance_registry import (
        InstanceRegistry,
    )
    from swarmpilot.scheduler.services.task_result_callback import (
        training_features,
    )

    adapter = get_adapter("diffusers")
    predictor = PredictorClient(storage_dir=str(tmp_path))
    trainer = TrainingClient(
        storage=predictor._low_level._storage,
        cache=predictor._low_level._cache,
        chain_builder=predictor._chain_builder,
        min_samples=10,
        prediction_types=["expect_error"],
    )
    pinfos = {
        hw: adapter.platform_info("0.31.0", hw) for hw in ("fastgpu", "slowgpu")
    }
    ms_per_step_mp = {"fastgpu": 20.0, "slowgpu": 80.0}
    for i in range(60):
        body = {
            "prompt": "p",
            "num_inference_steps": 4 + i % 20,
            "size": f"{512 + 64 * (i % 8)}x512",
        }
        for hw, pinfo in pinfos.items():
            feats = _extract_features(pinfo, "v1/images/generations", body)
            task = type("T", (), {"metadata": {"proxy": True, **feats}})()
            f = training_features(task, pinfo)
            trainer.add_sample(
                model_id="sdxl",
                platform_info=pinfo,
                features=f,
                actual_runtime_ms=ms_per_step_mp[hw]
                * f["steps"]
                * f["megapixels"],
            )
    assert await trainer.flush()

    registry = InstanceRegistry(queue_info_type="expect_error")
    instances = [
        Instance(
            instance_id=hw,
            model_id="sdxl",
            endpoint=f"http://{hw}",
            platform_info=pinfo,
        )
        for hw, pinfo in pinfos.items()
    ]
    for inst in instances:
        await registry.register(inst)
    strategy = get_strategy("min_time", predictor, registry)

    big = {"prompt": "p", "num_inference_steps": 20, "size": "1024x1024"}
    meta = {
        "proxy": True,
        **_extract_features(pinfos["fastgpu"], "v1/images/generations", big),
    }
    picks = [
        (
            await strategy.schedule_task("sdxl", meta, instances)
        ).selected_instance_id
        for _ in range(6)
    ]
    assert picks[0] == "fastgpu"
    assert "slowgpu" in picks
    assert picks.count("fastgpu") > picks.count("slowgpu")
