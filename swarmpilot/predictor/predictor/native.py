"""Run a trained predictor's inference in Rust (``SCHEDULER_CORE=rust``).

The predictor classes keep training, validation and feature ordering in
Python. With the Rust core active, :func:`native_model` hands back a
``swarmpilot_core.RuntimePredictor`` built from the same state the
portable model file holds, and ``predict()`` runs the network there
instead of in torch. ``tests/core/test_predictor_parity.py`` checks the
two agree to 1e-5 relative.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from swarmpilot.scheduler import core

_ATTR = "_native_model"


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def native_model(predictor: Any) -> Any | None:
    """Return the predictor's Rust twin, or None on the Python core.

    The twin is built on first use and rebuilt when the predictor's
    network object changes (training or loading replaces it).

    Args:
        predictor: A trained ``ExpectErrorPredictor`` or
            ``QuantilePredictor``.

    Returns:
        A ``swarmpilot_core.RuntimePredictor`` or None.
    """
    rc = core.rust()
    if rc is None or predictor.model is None:
        return None
    cached = predictor.__dict__.get(_ATTR)
    if cached is not None and cached[0] is predictor.model:
        return cached[1]
    state = predictor.get_model_state()
    tensors = {
        name: (
            list(t.shape),
            t.detach()
            .cpu()
            .contiguous()
            .float()
            .numpy()
            .astype("<f4")
            .tobytes(),
        )
        for name, t in state.pop("model_state_dict").items()
    }
    norm_f32 = np.asarray(predictor.feature_mean).dtype == np.float32
    native = rc.RuntimePredictor.from_state(
        json.dumps(state, default=_json_default), tensors, norm_f32
    )
    predictor.__dict__[_ATTR] = (predictor.model, native)
    return native


def forget_native_model(predictor: Any) -> None:
    """Drop a cached Rust twin (after training or loading)."""
    predictor.__dict__.pop(_ATTR, None)
