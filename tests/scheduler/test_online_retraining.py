"""Online retraining: rolling window, holdout gate, off-loop fit (2.4)."""

import asyncio
from typing import Any, ClassVar

import pytest

from swarmpilot.scheduler.clients import training_library_client as tlc
from swarmpilot.scheduler.clients._predictor_lib import ModelCache, ModelStorage
from swarmpilot.scheduler.clients.training_library_client import TrainingClient

PLAT = {"software_name": "vllm", "software_version": "1", "hardware_name": "h"}


class FakeQuantile:
    """Predicts fixed q0.9 and q0.99, whatever the features."""

    q90 = 900.5
    q99 = 990.5
    fits: ClassVar[list[int]] = []

    def train(self, features_list: list[dict], config: dict) -> None:
        FakeQuantile.fits.append(len(features_list))
        self.config = config

    def predict(self, features: dict) -> dict[str, Any]:
        return {"quantiles": {"0.5": 500.0, "0.9": self.q90, "0.99": self.q99}}

    def get_model_state(self) -> dict[str, Any]:
        return {"q90": self.q90, "q99": self.q99}

    def load_model_state(self, state: dict[str, Any]) -> None:
        self.q90, self.q99 = state["q90"], state["q99"]


class FakeExpect:
    """Predicts a fixed runtime."""

    value = 500.0

    def train(self, features_list: list[dict], config: dict) -> None:
        self.v = FakeExpect.value

    def predict(self, features: dict) -> dict[str, Any]:
        return {"expected_runtime_ms": self.v, "error_margin_ms": 1.0}

    def get_model_state(self) -> dict[str, Any]:
        return {"v": self.v}

    def load_model_state(self, state: dict[str, Any]) -> None:
        self.v = state["v"]


@pytest.fixture
def fakes(monkeypatch):
    FakeQuantile.fits = []
    monkeypatch.setitem(tlc.PREDICTOR_CLASSES, "quantile", FakeQuantile)
    monkeypatch.setitem(tlc.PREDICTOR_CLASSES, "expect_error", FakeExpect)


def _client(tmp_path, types=("quantile",), **kw) -> TrainingClient:
    return TrainingClient(
        storage=ModelStorage(str(tmp_path)),
        cache=ModelCache(),
        batch_size=10_000,
        min_samples=10,
        prediction_types=list(types),
        seed=0,
        **kw,
    )


def _feed(client: TrainingClient, runtimes) -> None:
    for i, r in enumerate(runtimes):
        client.add_sample("m", PLAT, {"x": float(i)}, float(r))


def _key(client: TrainingClient, ptype: str) -> str:
    return client._storage.generate_model_key("m", PLAT, ptype)


async def test_quantile_installs_when_coverage_on_target(tmp_path, fakes):
    client = _client(tmp_path)
    _feed(client, range(1, 1001))  # q0.9 = 900.5 covers 90%
    assert await client.flush(force=True)
    report = client.last_reports[_key(client, "quantile")]
    assert report.installed, report
    assert (report.n_train, report.n_holdout) == (800, 200)
    assert client._storage.model_exists(_key(client, "quantile"))
    saved = client._storage.load_model(_key(client, "quantile"))
    assert saved["metadata"]["training_config"] == {
        "quantiles": list(tlc.DEFAULT_QUANTILES),
        "data_augmentation": {"enabled": False},
    }


async def test_quantile_rejected_keeps_installed_model(tmp_path, fakes):
    client = _client(tmp_path)
    _feed(client, range(1, 1001))
    await client.flush(force=True)
    key = _key(client, "quantile")
    path = client._storage.storage_dir / f"{key}.joblib"
    before = path.read_bytes()

    FakeQuantile.q90 = 500.5  # covers only 50%
    try:
        _feed(client, range(1, 1001))
        await client.flush(force=True)
    finally:
        FakeQuantile.q90 = 900.5
    report = client.last_reports[key]
    assert not report.installed
    assert "q0.9 coverage" in report.reason
    assert path.read_bytes() == before


async def test_low_q99_coverage_is_rejected(tmp_path, fakes):
    client = _client(tmp_path)
    FakeQuantile.q99 = 950.5  # q0.99 covers only 95%
    try:
        _feed(client, range(1, 1001))
        await client.flush(force=True)
    finally:
        FakeQuantile.q99 = 990.5
    report = client.last_reports[_key(client, "quantile")]
    assert not report.installed
    assert "q0.99 coverage" in report.reason


async def test_rolling_window(tmp_path, fakes):
    client = _client(tmp_path, window_size=150)
    _feed(client, range(1, 101))
    await client.flush(force=True)
    _feed(client, range(1, 101))
    await client.flush(force=True)
    # The second fit sees the window (150), not just the new batch (100).
    assert FakeQuantile.fits[-1] == 150 - int(150 * 0.2)


async def test_cold_platform_installs_without_holdout(tmp_path, fakes):
    client = _client(tmp_path)
    _feed(client, range(1, 31))  # 6 held out would be too few
    await client.flush(force=True)
    report = client.last_reports[_key(client, "quantile")]
    assert report.installed and report.n_holdout == 0
    # Once a model exists, a small window cannot replace it.
    _feed(client, range(1, 11))
    await client.flush(force=True)
    assert not client.last_reports[_key(client, "quantile")].installed


async def test_expect_error_must_not_be_worse(tmp_path, fakes):
    client = _client(tmp_path, types=("expect_error",))
    _feed(client, [500.0] * 200)
    await client.flush(force=True)
    assert client.last_reports[_key(client, "expect_error")].installed

    FakeExpect.value = 900.0
    try:
        _feed(client, [500.0] * 200)
        await client.flush(force=True)
    finally:
        FakeExpect.value = 500.0
    report = client.last_reports[_key(client, "expect_error")]
    assert not report.installed
    assert report.metrics["holdout_mae_ms"] > report.metrics["installed_mae_ms"]


async def test_background_flush_does_not_block(tmp_path, fakes, monkeypatch):
    client = _client(tmp_path)
    client.batch_size = 50
    started = asyncio.Event()
    real = client._train_model

    def slow(*args):
        started.set()
        import time

        time.sleep(0.3)
        return real(*args)

    monkeypatch.setattr(client, "_train_model", slow)
    _feed(client, range(1, 51))
    assert client.start_flush_if_ready()
    _feed(client, range(1, 51))
    # A retrain is running: no second one starts.
    assert not client.start_flush_if_ready()
    t = asyncio.get_running_loop().time()
    await asyncio.sleep(0.05)
    # The loop kept running while the fit sleeps on its thread.
    assert asyncio.get_running_loop().time() - t < 0.2
    await client.close()
    assert started.is_set()
    assert client.last_reports
