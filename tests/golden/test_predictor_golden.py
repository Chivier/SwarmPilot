"""Replay the golden predictor models against the Python predictor.

``predictor/`` holds two trained models in the portable format and the
outputs torch gave when they were made (``make_models.py``). The Rust
core replays the same files in ``rust/crates/core/tests/golden.rs``; this
test keeps the Python side from drifting away from them.
"""

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor

DIR = Path(__file__).parent / "predictor"
EXPECTED = json.loads((DIR / "expected.json").read_text())
CLASSES = {"expect_error": ExpectErrorPredictor, "quantile": QuantilePredictor}


def load_portable(name: str):
    """Build a Python predictor from a portable model file."""
    doc = json.loads((DIR / f"{name}.json").read_text())
    state = dict(doc["state"])
    state["model_state_dict"] = load_file(str(DIR / doc["weights_file"]))
    predictor = CLASSES[name]()
    predictor.load_model_state(state)
    return predictor


def flatten(out: dict) -> list[float]:
    if "quantiles" in out:
        return [float(v) for v in out["quantiles"].values()]
    return [out["expected_runtime_ms"], out["error_margin_ms"]]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_python_predictor_matches_golden(name):
    predictor = load_portable(name)
    model = EXPECTED[name]
    assert predictor.feature_names == model["feature_names"]
    for case in model["cases"]:
        out = predictor.predict(
            dict(zip(model["feature_names"], case["x"], strict=True))
        )
        # torch on another CPU or BLAS, or the Rust core, is not bit-exact.
        assert flatten(out) == pytest.approx(flatten(case["out"]), rel=1e-5)
