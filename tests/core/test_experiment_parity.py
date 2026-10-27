"""Experiment-mode predictions: the Rust port against the Python original.

``generate_experiment_prediction`` draws 2,000 samples from
``RandomState(42)`` on every call. The Rust port takes those standard
normal draws as input (the gateway gets them from the Scheduler) and must
give the same predictions, or fail where Python raises.
"""

from __future__ import annotations

import json
import math
import random
import sys

import numpy as np
import pytest

from swarmpilot.predictor.utils.experiment import (
    generate_experiment_prediction,
)

NORMALS = np.random.RandomState(42).normal(size=2000).tolist()
GRIDS = [
    None,
    [0.5, 0.9, 0.95, 0.99],
    [0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    [0.0, 1.0],
]


def _fields(rng: random.Random) -> dict:
    out: dict = {"exp_runtime": rng.choice([1, 0.5, 37.25, 1e4, 3e6])}
    if rng.random() < 0.7:
        out["exp_cv"] = rng.choice([0, 0.05, 0.3, 1.2, 5])
    if rng.random() < 0.5:
        out["exp_skewness"] = rng.choice([0, -1, 0.5, 2, 8])
    if rng.random() < 0.4:
        out["exp_modes"] = [
            {
                k: v
                for k, v in {
                    "mean": rng.choice([3, 50.5, 800, 12000]),
                    "weight": rng.choice([None, 1, 0.2, 3]),
                    "cv": rng.choice([None, 0.1, 0.6]),
                    "skewness": rng.choice([None, 0, 1.5]),
                }.items()
                if v is not None
            }
            for _ in range(rng.randint(1, 4))
        ]
    return out


def _python(ptype, fields, grid):
    config = {} if grid is None else {"quantiles": grid}
    try:
        out = generate_experiment_prediction(ptype, dict(fields), config)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if ptype == "expect_error":
        return [out["expected_runtime_ms"], out["error_margin_ms"]]
    return list(out["quantiles"].values())


def _rust(rc, ptype, fields, grid):
    try:
        out = rc.experiment_predict(json.dumps(fields), ptype, grid, NORMALS)
    except ValueError:
        return None
    if out[0] == "expect_error":
        return [out[1], out[2]]
    return [v for _, v in out[1]]


# Rust follows CPython 3.12+'s compensated sum(); 3.11 sums floats
# naively and can differ in the last bit.
_REL = 0.0 if sys.version_info >= (3, 12) else 1e-15


def _same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= _REL * abs(a)


@pytest.mark.parametrize("ptype", ["expect_error", "quantile"])
def test_random_parameters_match(rc, ptype):
    rng = random.Random(7)
    mismatches = []
    for _ in range(600):
        fields = _fields(rng)
        grid = rng.choice(GRIDS)
        want = _python(ptype, fields, grid)
        got = _rust(rc, ptype, fields, grid)
        if want is None or got is None:
            assert want is None and got is None, (fields, want, got)
            continue
        if not all(_same(a, b) for a, b in zip(want, got, strict=True)):
            mismatches.append((fields, grid, want, got))
    assert not mismatches, mismatches[:3]


@pytest.mark.parametrize(
    "fields",
    [
        {"exp_runtime": 0},
        {"exp_runtime": -3},
        {"exp_runtime": "5"},
        {"exp_runtime": 10, "exp_cv": -0.5},
        {"exp_runtime": 10, "exp_modes": [{"weight": 1}]},
        {"exp_runtime": 10, "exp_modes": "ab"},
    ],
)
def test_bad_parameters_fail_on_both(rc, fields):
    assert _python("quantile", fields, None) is None
    assert _rust(rc, "quantile", fields, None) is None


def test_truthiness_and_empty_modes(rc):
    for fields in (
        {"exp_runtime": True},
        {"exp_runtime": 10, "exp_modes": []},
        {"exp_runtime": 10, "exp_modes": ""},
    ):
        for ptype in ("expect_error", "quantile"):
            assert _rust(rc, ptype, fields, None) == _python(
                ptype, fields, None
            )
