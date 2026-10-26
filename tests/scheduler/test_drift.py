"""Drift detection and drift-driven retraining (plan item 3.1)."""

from typing import Any, ClassVar

import numpy as np
import pytest

from swarmpilot.scheduler.clients import training_library_client as tlc
from swarmpilot.scheduler.clients._predictor_lib import ModelCache, ModelStorage
from swarmpilot.scheduler.clients.training_library_client import TrainingClient
from swarmpilot.scheduler.services.drift import (
    DriftDetector,
    prompt_group,
    tail_estimate,
)

PLAT = {"software_name": "vllm", "software_version": "1", "hardware_name": "h"}
KEY = ("m", "p")


class TestTailEstimate:
    """The predicted 0.9 quantile from either prediction kind."""

    def test_interpolates_quantiles(self):
        assert tail_estimate(0.9, {0.5: 100, 0.95: 190}, None, None) == 180

    def test_holds_the_ends(self):
        assert tail_estimate(0.9, {0.95: 7.0, 0.99: 9.0}, None, None) == 7.0
        assert tail_estimate(0.9, {0.5: 3.0}, None, None) == 3.0

    def test_expect_error_reads_as_normal(self):
        assert tail_estimate(0.9, None, 100.0, 10.0) == pytest.approx(112.8155)

    def test_nothing_known(self):
        assert tail_estimate(0.9, None, None, None) is None


class TestPromptGroup:
    """Log2 buckets of the first feature."""

    def test_buckets(self):
        assert prompt_group({"prompt_length": 0}) == 0
        assert prompt_group({"prompt_length": 100}) == 6
        assert prompt_group({"prompt_length": 3000, "max_tokens": 9}) == 11
        assert prompt_group({"flag": True, "steps": 30}) == 4
        assert prompt_group({}) == 0


def _observe(det: DriftDetector, n: int, scale: float, rng, group=100.0):
    fired = []
    for _ in range(n):
        y = rng.uniform(50, 100) * scale
        fired.append(det.observe(KEY, {"len": group}, 95.0, y))
    return fired


class TestDriftDetector:
    """Window, baseline and the two drift signals."""

    def test_stable_never_drifts(self):
        det = DriftDetector()
        assert not any(_observe(det, 2000, 1.0, np.random.default_rng(0)))

    def test_shift_drifts_within_two_windows_of_min_samples(self):
        det = DriftDetector()
        rng = np.random.default_rng(1)
        _observe(det, 300, 1.0, rng)
        fired = _observe(det, 100, 2.0, rng)
        assert any(fired)
        assert fired.index(True) < 2 * det.min_samples
        assert det.drifted() == {KEY}
        # Fires once, not on every later task.
        assert sum(fired) == 1

    def test_over_prediction_drifts_on_loss(self):
        # Runtimes halve: misses vanish but relative loss grows.
        det = DriftDetector()
        rng = np.random.default_rng(2)
        _observe(det, 300, 1.0, rng)
        assert any(_observe(det, 200, 0.1, rng))

    def test_groups_are_separate(self):
        det = DriftDetector()
        rng = np.random.default_rng(3)
        _observe(det, 300, 1.0, rng, group=10.0)
        _observe(det, 300, 1.0, rng, group=5000.0)
        assert any(_observe(det, 100, 2.0, rng, group=5000.0))
        groups = {g["prompt_group"]: g["drifted"] for g in det.snapshot()}
        assert groups == {3: False, 12: True}

    def test_reset_and_rearm(self):
        det = DriftDetector()
        rng = np.random.default_rng(4)
        _observe(det, 300, 1.0, rng)
        _observe(det, 100, 2.0, rng)
        det.rearm(KEY)
        assert det.drifted() == set()
        # Still shifted against the old baseline: drifts again.
        assert any(_observe(det, 100, 2.0, rng))
        det.reset(KEY)
        assert det.snapshot() == []

    def test_ignores_bad_predictions(self):
        det = DriftDetector()
        assert not det.observe(KEY, {}, 0.0, 10.0)
        assert not det.observe(KEY, {}, float("nan"), 10.0)
        assert det.snapshot() == []


class Learned:
    """Constant-quantile model fitted to the training runtimes."""

    fits: ClassVar[list[int]] = []

    def train(self, features_list: list[dict], config: dict) -> None:
        Learned.fits.append(len(features_list))
        ys = np.array([f["runtime_ms"] for f in features_list])
        self.q = {
            str(q): float(np.quantile(ys, q)) for q in tlc.DEFAULT_QUANTILES
        }

    def predict(self, features: dict) -> dict[str, Any]:
        return {"quantiles": dict(self.q)}

    def get_model_state(self) -> dict[str, Any]:
        return {"q": self.q}

    def load_model_state(self, state: dict[str, Any]) -> None:
        self.q = state["q"]


@pytest.fixture
def learned(monkeypatch):
    Learned.fits = []
    monkeypatch.setitem(tlc.PREDICTOR_CLASSES, "quantile", Learned)


def _client(tmp_path, **kw) -> TrainingClient:
    kw.setdefault("batch_size", 100)
    kw.setdefault("window_size", 200)
    return TrainingClient(
        storage=ModelStorage(str(tmp_path)),
        cache=ModelCache(),
        min_samples=10,
        prediction_types=["quantile"],
        seed=0,
        warm_up=False,
        **kw,
    )


def _installed_q(client: TrainingClient) -> dict[float, float]:
    key = client._storage.generate_model_key("m", PLAT, "quantile")
    state = client._storage.load_model(key)["predictor_state"]["q"]
    return {float(q): v for q, v in state.items()}


async def _serve(client, n, scale, rng):
    """Complete n tasks, each routed with the installed model's quantiles."""
    for _ in range(n):
        q = _installed_q(client) if client._is_warm("m", PLAT) else None
        client.add_sample(
            "m",
            PLAT,
            {"prompt_length": 500.0},
            float(rng.uniform(50, 100) * scale),
            predicted_quantiles=q,
        )
        if client.start_flush_if_ready():
            await client._background


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
async def test_warm_model_retrains_only_on_drift(tmp_path, learned, seed):
    client = _client(tmp_path)
    rng = np.random.default_rng(seed)
    for _ in range(10):  # cold: full batches train until one installs
        await _serve(client, 100, 1.0, rng)
        if client._is_warm("m", PLAT):
            break
    assert client._is_warm("m", PLAT)
    await _serve(client, 200, 1.0, rng)  # refit once the data doubles
    n_cold = len(Learned.fits)
    assert Learned.fits[-1] == 160  # the full 200-sample window
    await _serve(client, 600, 1.0, rng)  # stable, window full: no retrain
    assert len(Learned.fits) == n_cold

    await _serve(client, 160, 3.0, rng)  # 3x slower
    assert len(Learned.fits) == n_cold + 1
    # 80 post-drift samples (20 held out) plus the 100 newest pre-drift
    # samples rescaled by the measured 3x shift fill the 200-sample window.
    assert Learned.fits[-1] == 80 + 100
    assert _installed_q(client)[0.9] > 250
    report = next(iter(client.last_reports.values()))
    assert report.installed, report

    await _serve(client, 600, 3.0, rng)
    # Refits as the post-drift data doubles to the full window (retried if
    # the gate turns one down), and no further drift.
    later = Learned.fits[n_cold + 1 :]
    assert 1 <= len(later) <= 3 and set(later) == {160}, Learned.fits


async def test_batch_mode_retrains_every_batch(tmp_path, learned):
    client = _client(tmp_path, retrain_mode="batch")
    rng = np.random.default_rng(0)
    await _serve(client, 300, 1.0, rng)
    assert len(Learned.fits) == 3


def test_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        _client(tmp_path, retrain_mode="sometimes")


async def test_drift_flagged_during_a_refit_is_kept(tmp_path, learned):
    """A detection while a fit runs must survive that fit's bookkeeping."""
    import asyncio
    import threading

    client = _client(tmp_path)
    rng = np.random.default_rng(0)
    for _ in range(10):
        await _serve(client, 100, 1.0, rng)
        if client._is_warm("m", PLAT):
            break
    assert client._is_warm("m", PLAT)
    key = client._key("m", PLAT)

    started, release = threading.Event(), threading.Event()
    real = client._train_model

    def slow(*args):
        started.set()
        release.wait(5)
        return real(*args)

    client._train_model = slow
    for _ in range(20):
        client.add_sample("m", PLAT, {"prompt_length": 500.0}, 75.0)
    fit = asyncio.create_task(client.flush(force=True))
    while not started.is_set():
        await asyncio.sleep(0.01)

    observe = client.drift.observe
    client.drift.observe = lambda *a: True
    client.add_sample(
        "m",
        PLAT,
        {"prompt_length": 500.0},
        300.0,
        predicted_quantiles=_installed_q(client),
    )
    client.drift.observe = observe
    for _ in range(5):
        client.add_sample("m", PLAT, {"prompt_length": 500.0}, 300.0)
    release.set()
    await fit

    assert client._recovering.get(key) == 5
    # The next absorb keeps the five post-detection samples, not none.
    client._absorb()
    assert len(client._windows[key]) == 5
    await client.close()
