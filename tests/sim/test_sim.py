"""Tests for the cluster simulator's data files and Python face.

The data checks run everywhere. The CLI checks need the Rust binary and
skip when it is not built (``cargo build --manifest-path rust/Cargo.toml``).
"""

import pytest
import yaml

from swarmpilot import sim

needs_binary = pytest.mark.skipif(
    sim.find_binary() is None, reason="swarmpilot-sim not built"
)


def _load(name):
    return yaml.safe_load((sim.DATA_DIR / name).read_text())


def test_default_spec_is_256_instances_on_9_configs():
    spec = yaml.safe_load(sim.DEFAULT_SPEC.read_text())
    groups = spec["node_groups"]
    gpus = _load("gpus.yaml")
    models = _load("models.yaml")
    engines = _load("engines.yaml")
    assert len(groups) == 9
    assert len({g["gpu"] for g in groups}) == 8
    instances = 0
    for g in groups:
        assert g["gpu"] in gpus
        for d in g["deployments"]:
            assert d["model"] in models
            assert d["engine"] in engines
            assert d["instances"] * d.get("tp", 1) <= (
                g["nodes"] * g["gpus_per_node"]
            )
            instances += d["instances"]
    assert instances == 256
    assert sum(g["nodes"] * g["gpus_per_node"] for g in groups) == 288


def test_every_catalog_entry_names_a_source():
    for name, gpu in _load("gpus.yaml").items():
        assert gpu["source"], name
    for name, engine in _load("engines.yaml").items():
        assert engine["source"], name
        assert isinstance(engine["uncalibrated"], bool), name


def test_missing_binary_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SWARMPILOT_SIM_BIN", str(tmp_path / "nope"))
    assert sim.find_binary() is None
    with pytest.raises(sim.SimulatorNotFoundError):
        sim.plan()


@needs_binary
def test_plan_default_spec():
    summary = sim.plan()
    assert summary["instances"] == 256
    assert summary["configurations"] == 9
    assert summary["gpus_used"] == 288
    # Only the vLLM profiles on H100 are fit to measurements.
    calibrated = summary["instances"] - summary["uncalibrated_instances"]
    assert calibrated == 32


@needs_binary
def test_plan_instances_have_unique_ids_and_profiles():
    plan = sim.plan(instances=True)
    assert len({p["id"] for p in plan}) == 256
    kinds = {p["profile"]["kind"]["kind"] for p in plan}
    assert kinds == {"llm", "diffusion", "generic"}


@needs_binary
def test_profile_and_bench_low_load_matches_vllm_nightly():
    p = sim.profile("vllm", "meta-llama/Llama-3.1-8B-Instruct", "H100-SXM-80GB")
    assert p["model"] == "llama-3.1-8b"
    assert not p["uncalibrated"]
    row = sim.bench("vllm", "llama-3.1-8b", "H100-SXM-80GB", qps=1)
    # vLLM nightly, ShareGPT at QPS 1 on H100: mean ITL 7.63 ms.
    assert row["itl_mean_ms"] == pytest.approx(7.63, rel=0.1)
    assert row["completed"] == 200


@needs_binary
def test_bench_reports_slo_misses():
    row = sim.bench(
        "diffusers", "sdxl", "L4-24GB", qps=0.5, requests=50, slo_ms=5_000
    )
    assert 0.0 <= row["slo_miss_rate"] <= 1.0
    # SDXL at 30 steps on an L4 takes well over 5 s.
    assert row["slo_miss_rate"] == 1.0


@needs_binary
def test_bad_spec_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nnode_groups:\n"
        "  - {name: a, gpu: L4-24GB, gpus_per_node: 2, nodes: 1,\n"
        "     deployments: [{engine: vllm, model: llama-3.1-70b, tp: 2,\n"
        "                    instances: 1}]}\n"
    )
    with pytest.raises(RuntimeError, match="does not fit"):
        sim.plan(bad)


def test_planner_sim_mode_implies_local_mode(monkeypatch):
    from swarmpilot.planner.config import PlannerConfig

    monkeypatch.setenv("PYLET_SIM_MODE", "true")
    monkeypatch.setenv("PYLET_LOCAL_PORT", "5155")
    monkeypatch.setenv("PYLET_SIM_TIME_SCALE", "0.1")
    monkeypatch.delenv("PYLET_HEAD_URL", raising=False)
    cfg = PlannerConfig()
    assert cfg.pylet_sim_mode and cfg.pylet_local_mode and cfg.pylet_enabled
    assert cfg.pylet_head_url == "http://localhost:5155"
    assert cfg.pylet_sim_time_scale == 0.1
    assert cfg.pylet_sim_spec is None
