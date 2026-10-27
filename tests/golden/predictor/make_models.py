"""Write the golden predictor models and their torch predictions.

Two small trained models (expect_error and quantile with log transform)
in the portable format of ``ModelStorage.export_portable``, plus the
outputs torch gives on fixed inputs. The Rust core
(``rust/crates/core/tests/golden.rs``) and ``test_predictor_golden.py``
must reproduce the outputs to 1e-5 relative. Training is not bit-for-bit
repeatable across platforms, so the models are committed rather than
regenerated in CI; rerun this only to change them::

    uv run python tests/golden/predictor/make_models.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from swarmpilot.predictor.predictor.expect_error import ExpectErrorPredictor
from swarmpilot.predictor.predictor.quantile import QuantilePredictor
from swarmpilot.predictor.storage.model_storage import ModelStorage

HERE = Path(__file__).parent

MODELS = {
    "expect_error": (
        ExpectErrorPredictor,
        {"epochs": 200, "hidden_layers": [16, 8]},
    ),
    "quantile": (
        QuantilePredictor,
        {
            "epochs": 200,
            "hidden_layers": [32, 16],
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.9, 0.99],
            "log_transform": {"enabled": True},
            "delta_scale": 1.5,
            "data_augmentation": {"enabled": False},
        },
    ),
}


def samples(n: int, seed: int) -> list[dict[str, float]]:
    """Runtime samples over three features."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = rng.randint(1, 4000)
        m = rng.choice([16, 64, 256, 1024])
        out.append(
            {
                "prompt_length": float(p),
                "max_tokens": float(m),
                "batch": float(rng.randint(1, 8)),
                "runtime_ms": 20 + 0.05 * p + 8.0 * m * rng.uniform(0.7, 1.4),
            }
        )
    return out


def inputs() -> list[dict[str, float]]:
    """Fixed probe inputs, including ones outside the training data."""
    rng = random.Random(99)
    rows = [
        {
            "prompt_length": rng.uniform(0, 5000),
            "max_tokens": rng.choice([1, 16, 300, 2048]),
            "batch": rng.randint(1, 9),
        }
        for _ in range(40)
    ]
    extremes = [(0, 0, 0), (20000, 4096, 16), (-50, 3, 1)]
    return rows + [
        {"prompt_length": p, "max_tokens": m, "batch": b}
        for p, m, b in extremes
    ]


def main() -> None:
    """Train, export and record."""
    torch.manual_seed(0)
    np.random.seed(0)
    store = ModelStorage(str(HERE))
    expected = {}
    for name, (cls, config) in MODELS.items():
        predictor = cls()
        predictor.train(samples(300, seed=len(name)), config)
        state = predictor.get_model_state()
        # Reload as a model read from disk would be (float64 scaling).
        predictor = cls()
        predictor.load_model_state(state)
        store.export_portable(
            name,
            state,
            {
                "model_id": "golden",
                "prediction_type": name,
                "platform_info": {},
            },
        )
        names = predictor.feature_names
        expected[name] = {
            "feature_names": names,
            "cases": [
                {"x": [row[n] for n in names], "out": predictor.predict(row)}
                for row in inputs()
            ],
        }
    (HERE / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")


if __name__ == "__main__":
    main()
