"""Measure QuantilePredictor training/prediction cost and calibration.

Mirrors what TrainingClient does on the scheduler event loop: train a fresh
QuantilePredictor with config={} on the last batch (default 100 samples).

Usage:
    uv run python tests/perf/bench_predictor.py
"""

import json
import time

import numpy as np

from swarmpilot.predictor.predictor.quantile import QuantilePredictor


def make(n, rng):
    plen = rng.integers(50, 4000, size=n)
    maxt = rng.choice([256, 512, 1024, 2048], size=n)
    mean = 50 + 0.05 * plen + 0.8 * maxt
    rt = mean * rng.lognormal(-0.5 * 0.4**2, 0.4, size=n)
    feats = [
        {"prompt_length": int(a), "max_tokens": int(b), "runtime_ms": float(r)}
        for a, b, r in zip(plen, maxt, rt)
    ]
    return feats, mean


def coverage(p, test):
    hits = dict.fromkeys((0.5, 0.9, 0.95, 0.99), 0)
    t0 = time.perf_counter()
    for f in test:
        out = p.predict({k: v for k, v in f.items() if k != "runtime_ms"})
        qs = out["quantiles"]
        for q in hits:
            key = q if q in qs else str(q)
            if f["runtime_ms"] <= qs[key]:
                hits[q] += 1
    dt = (time.perf_counter() - t0) / len(test) * 1000
    return {str(q): round(h / len(test), 3) for q, h in hits.items()}, dt


def main():
    from loguru import logger

    logger.remove()
    rng = np.random.default_rng(0)
    test, _ = make(500, rng)
    res = []
    for n, cfg_name, cfg in [
        (100, "default(config={}) as TrainingClient", {}),
        (100, "augmentation off", {"data_augmentation": {"enabled": False}}),
        (2000, "default, 2000 samples", {}),
        (
            2000,
            "augmentation off, 2000 samples",
            {"data_augmentation": {"enabled": False}},
        ),
    ]:
        train, _ = make(n, rng)
        p = QuantilePredictor()
        t0 = time.perf_counter()
        p.train(train, config=dict(cfg))
        tt = time.perf_counter() - t0
        cov, pms = coverage(p, test)
        res.append(
            {
                "case": cfg_name,
                "train_samples": n,
                "train_seconds": round(tt, 2),
                "predict_ms_per_call": round(pms, 3),
                "empirical_coverage": cov,
            }
        )
    for r in res:
        print(json.dumps(r))


main()
