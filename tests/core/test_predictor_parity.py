"""Predictor inference: the Rust core against torch.

Trains small expect-error and quantile models in several configurations
and compares ``predict()`` on the Python core (torch) with the Rust core,
right after training (float32 scaling), after a state round trip
(float64 scaling) and from the portable model file. The plan's bar is
1e-5 relative; measured differences are around 1e-6.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.native import native_model
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.storage.model_storage import ModelStorage

REL = 1e-5

CONFIGS = {
    "expect_error": (ExpectErrorPredictor, {"epochs": 60}),
    "expect_error_wide": (
        ExpectErrorPredictor,
        {"epochs": 60, "hidden_layers": [128, 64, 16]},
    ),
    "quantile": (QuantilePredictor, {"epochs": 60}),
    "quantile_log": (
        QuantilePredictor,
        {
            "epochs": 60,
            "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            "log_transform": {"enabled": True},
            "delta_scale": 2.0,
        },
    ),
    "quantile_calibrated": (
        QuantilePredictor,
        {
            "epochs": 60,
            "hidden_layers": [8],
            "residual_calibration": {"enabled": True},
            "data_augmentation": {"enabled": False},
        },
    ),
}


def _samples(seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(120):
        p = rng.randint(1, 3000)
        m = rng.choice([16, 128, 512])
        out.append(
            {
                "prompt_length": float(p),
                "max_tokens": float(m),
                "constant": 7.0,  # dropped by the quantile trainer
                "runtime_ms": 30 + 0.04 * p + 6 * m * rng.uniform(0.6, 1.5),
            }
        )
    return out


def _probes(seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = [
        {
            "prompt_length": rng.choice([0, rng.uniform(0, 6000), 1e5]),
            "max_tokens": rng.choice([1, 16, 128, 512, 4096]),
            "constant": 7.0,
        }
        for _ in range(150)
    ]
    # Non-finite inputs and values that overflow float32: torch gives NaN
    # or inf, and so must Rust.
    for odd in (float("nan"), float("inf"), -float("inf"), 1e39, -1e39):
        rows.append({"prompt_length": odd, "max_tokens": 128, "constant": 7.0})
        rows.append(
            {"prompt_length": 100.0, "max_tokens": odd, "constant": 7.0}
        )
    return rows


def _values(out: dict) -> list[float]:
    if "quantiles" in out:
        return list(out["quantiles"].values())
    return [out["expected_runtime_ms"], out["error_margin_ms"]]


def _assert_close(a: list[float], b: list[float]) -> float:
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if not (math.isfinite(x) and math.isfinite(y)):
            assert x == y or (math.isnan(x) and math.isnan(y)), (a, b)
            continue
        rel = abs(x - y) / max(abs(x), 1e-9)
        worst = max(worst, rel)
    assert worst <= REL, (a, b)
    return worst


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def trained(request):
    torch.manual_seed(1)
    np.random.seed(1)
    cls, config = CONFIGS[request.param]
    predictor = cls()
    predictor.train(_samples(len(request.param)), dict(config))
    return request.param, cls, predictor


def _predict_both(use_core, predictor, rows):
    use_core("python")
    want = [_values(predictor.predict(dict(r))) for r in rows]
    use_core("rust")
    assert native_model(predictor) is not None
    got = [_values(predictor.predict(dict(r))) for r in rows]
    return want, got


def test_after_training(use_core, trained):
    _, _, predictor = trained
    assert np.asarray(predictor.feature_mean).dtype == np.float32
    want, got = _predict_both(use_core, predictor, _probes(1))
    for w, g in zip(want, got, strict=True):
        _assert_close(w, g)


def test_after_reload(use_core, trained):
    _, cls, predictor = trained
    reloaded = cls()
    reloaded.load_model_state(predictor.get_model_state())
    want, got = _predict_both(use_core, reloaded, _probes(2))
    for w, g in zip(want, got, strict=True):
        _assert_close(w, g)


def test_portable_file(use_core, trained, tmp_path):
    import swarmpilot_core

    name, cls, predictor = trained
    state = predictor.get_model_state()
    sidecar = ModelStorage(str(tmp_path)).export_portable(
        name, state, {"prediction_type": name}
    )
    reloaded = cls()
    reloaded.load_model_state(state)
    native = swarmpilot_core.RuntimePredictor.load_portable(sidecar)
    assert native.feature_names == reloaded.feature_names
    use_core("python")
    for row in _probes(3):
        want = _values(reloaded.predict(dict(row)))
        x = np.array(
            [[row[n] for n in reloaded.feature_names]], dtype=np.float32
        )[0].tolist()
        out = native.predict(x)
        got = [v for _, v in out[1]] if out[0] == "quantile" else list(out[1:])
        _assert_close(want, got)


def test_retraining_rebuilds_the_rust_model(use_core, trained):
    _, cls, predictor = trained
    use_core("rust")
    assert native_model(predictor) is not None
    row = _probes(4)[0]
    before = _values(predictor.predict(dict(row)))
    fresh = cls()
    fresh.load_model_state(predictor.get_model_state())
    # Perturb the network the way a retrain would replace it.
    state = fresh.get_model_state()
    for t in state["model_state_dict"].values():
        t.mul_(1.5)
    fresh.load_model_state(state)
    after = _values(fresh.predict(dict(row)))
    use_core("python")
    torch_after = _values(fresh.predict(dict(row)))
    _assert_close(torch_after, after)
    assert after != before
