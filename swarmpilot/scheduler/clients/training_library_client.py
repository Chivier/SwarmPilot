"""Training client using direct library imports.

Shares ModelStorage and ModelCache instances with PredictorClient
so that trained models are immediately visible to predictions.
Applies V2 preprocessor chains during training for consistency
with prediction-time preprocessing.

Online retraining keeps the Scheduler responsive and its models sound:

- Each (model, platform) keeps a rolling window of its latest samples, and
  every retrain fits on that window, not only on the newest batch.
- Fitting runs on a worker thread, so the event loop keeps serving.
- A random holdout from the window validates the candidate. A quantile
  model installs only if its q0.9 coverage on the holdout is within
  ``coverage_tolerance`` of 0.9 and its q0.99 coverage is at least
  ``min_q99_coverage``. An expect-error model installs only if its holdout
  error is no worse than the installed model's.
- Installing writes the model file atomically, then drops the cache entry.
- Runtime augmentation is off: online samples carry real variance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from loguru import logger

from swarmpilot.scheduler.clients._predictor_lib import (
    PREDICTOR_CLASSES,
    ModelCache,
    ModelStorage,
)
from swarmpilot.scheduler.clients.models import TrainingSample
from swarmpilot.scheduler.clients.preprocessor_config import (
    PreprocessorChainBuilder,
)
from swarmpilot.scheduler.registry.queue_sketch import DEFAULT_QUANTILES

# Fewer holdout samples than this cannot judge a candidate; it then installs
# only where no model exists yet.
MIN_HOLDOUT = 20


@dataclass
class FitReport:
    """Outcome of one candidate model."""

    installed: bool
    reason: str
    n_train: int
    n_holdout: int
    metrics: dict[str, float]


class TrainingClient:
    """Training client using direct library imports.

    Shares ModelStorage and ModelCache with PredictorClient
    so that trained models are immediately visible to predictions.
    Applies the same V2 preprocessor chain used at prediction time
    to ensure feature consistency.
    """

    def __init__(
        self,
        storage: ModelStorage,
        cache: ModelCache,
        chain_builder: PreprocessorChainBuilder | None = None,
        batch_size: int = 100,
        min_samples: int = 10,
        prediction_types: list[str] | None = None,
        window_size: int = 2000,
        holdout_fraction: float = 0.2,
        coverage_tolerance: float = 0.03,
        min_q99_coverage: float = 0.97,
        seed: int | None = None,
        warm_up: bool = True,
    ):
        """Initialize training client.

        Args:
            storage: Shared ModelStorage instance.
            cache: Shared ModelCache instance.
            chain_builder: Shared PreprocessorChainBuilder for applying
                the same preprocessing at training time.
            batch_size: Samples to batch before training.
            min_samples: Minimum samples required for training.
            prediction_types: Prediction types to train.
            window_size: Latest samples kept per (model, platform).
            holdout_fraction: Share of the window held out to validate.
            coverage_tolerance: Allowed gap between q0.9 coverage and 0.9.
            min_q99_coverage: Lowest acceptable q0.99 coverage.
            seed: Seed for the holdout split (experiments only).
            warm_up: Run a tiny fit on the training thread now, so the
                first real retrain does not import anything.
        """
        self._storage = storage
        self._cache = cache
        self._chain_builder = chain_builder
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.prediction_types = prediction_types or [
            "expect_error",
            "quantile",
        ]

        self.window_size = window_size
        self.holdout_fraction = holdout_fraction
        self.coverage_tolerance = coverage_tolerance
        self.min_q99_coverage = min_q99_coverage
        self._rng = np.random.default_rng(seed)

        self._samples_buffer: list[TrainingSample] = []
        self._windows: dict[tuple[str, str], deque[TrainingSample]] = {}
        self._last_flush_samples: int = 0
        self.last_reports: dict[str, FitReport] = {}
        # One fit at a time, off the event loop.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sp-train"
        )
        self._flush_lock = asyncio.Lock()
        self._background: asyncio.Task | None = None
        if warm_up:
            # The first fit lazily imports hundreds of modules (torch's
            # optimizer and compiler stack, sympy). Imports and extension
            # loads hold the GIL, which stalled the event loop for ~100 ms
            # mid-retrain on cold machines. Pay that once, here.
            self._executor.submit(self._warm_up).result()

        logger.info(
            f"TrainingClient initialized "
            f"(batch_size={batch_size}, min_samples={min_samples}, "
            f"types={self.prediction_types}, "
            f"chain_builder={'yes' if chain_builder else 'no'})"
        )

    def add_sample(
        self,
        model_id: str,
        platform_info: dict[str, str],
        features: dict[str, Any],
        actual_runtime_ms: float,
    ) -> None:
        """Add a training sample to the buffer.

        Args:
            model_id: Model/tool identifier.
            platform_info: Platform information.
            features: Task features.
            actual_runtime_ms: Actual execution time in ms.
        """
        sample = TrainingSample(
            model_id=model_id,
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=actual_runtime_ms,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._samples_buffer.append(sample)

        logger.debug(
            f"Added training sample for {model_id} on "
            f"{platform_info.get('hardware_name', '?')}: "
            f"{actual_runtime_ms:.2f}ms "
            f"(buffer size: {len(self._samples_buffer)})"
        )

    async def flush_if_ready(self) -> bool:
        """Flush buffer to train models if batch size reached.

        Returns:
            True if training was triggered, False otherwise.
        """
        if len(self._samples_buffer) >= self.batch_size:
            return await self.flush()
        return False

    def start_flush_if_ready(self) -> bool:
        """Start a background flush if the batch is full and none is running.

        For the result callback: it must not wait for a retrain.

        Returns:
            True if a flush was started.
        """
        if len(self._samples_buffer) < self.batch_size:
            return False
        if self._background is not None and not self._background.done():
            return False
        self._background = asyncio.get_running_loop().create_task(self.flush())
        return True

    async def flush(self, force: bool = False) -> bool:
        """Train models with all buffered samples.

        Args:
            force: If True, train even below min_samples threshold.

        Returns:
            True if all training succeeded, False otherwise.
        """
        if not self._samples_buffer:
            logger.debug("No training samples to flush")
            return False

        if not force and len(self._samples_buffer) < self.min_samples:
            logger.debug(
                f"Buffer has {len(self._samples_buffer)} samples, "
                f"below minimum of {self.min_samples}. Skipping training."
            )
            return False

        batch, self._samples_buffer = self._samples_buffer, []
        touched: dict[tuple[str, str], deque[TrainingSample]] = {}
        for sample in batch:
            platform_key = json.dumps(sample.platform_info, sort_keys=True)
            key = (sample.model_id, platform_key)
            window = self._windows.get(key)
            if window is None:
                window = deque(maxlen=self.window_size)
                self._windows[key] = window
            window.append(sample)
            touched[key] = window

        logger.info(
            f"Flushing {len(batch)} training samples "
            f"across {len(touched)} model-platform combinations"
        )

        success_count = 0
        failure_count = 0
        loop = asyncio.get_running_loop()

        async with self._flush_lock:
            for (model_id, _platform_key), window in touched.items():
                samples = list(window)
                platform_info = samples[0].platform_info
                features_list = [
                    {**s.features, "runtime_ms": s.actual_runtime_ms}
                    for s in samples
                ]
                for prediction_type in self.prediction_types:
                    try:
                        report = await loop.run_in_executor(
                            self._executor,
                            self._train_model,
                            model_id,
                            platform_info,
                            prediction_type,
                            [dict(f) for f in features_list],
                        )
                        success_count += 1
                    except Exception as e:
                        logger.opt(exception=True).error(
                            f"Failed to train {model_id} ({prediction_type}) "
                            f"on {platform_info.get('hardware_name', '?')}: "
                            f"{e}"
                        )
                        failure_count += 1
                        continue
                    key = self._storage.generate_model_key(
                        model_id=model_id,
                        platform_info=platform_info,
                        prediction_type=prediction_type,
                    )
                    self.last_reports[key] = report
                    if report.installed:
                        # The file is already swapped; drop the stale copy.
                        self._cache.invalidate(key)

        logger.info(
            f"Training complete: {success_count} succeeded, "
            f"{failure_count} failed, {len(batch)} samples consumed"
        )

        self._last_flush_samples = len(batch)
        return failure_count == 0

    def _warm_up(self) -> None:
        """Fit a throwaway model of each type to load the code it needs."""
        rows = [{"x": float(i), "runtime_ms": 10.0 + i} for i in range(16)]
        with contextlib.suppress(ImportError):
            # Model files are also exported as safetensors on save.
            import safetensors.torch  # noqa: F401
        for prediction_type in self.prediction_types:
            cls = PREDICTOR_CLASSES.get(prediction_type)
            if cls is None:
                continue
            try:
                predictor = cls()
                predictor.train(
                    features_list=[dict(r) for r in rows],
                    config={**self._train_config(prediction_type), "epochs": 2},
                )
                predictor.predict({"x": 1.0})
                json.dumps(predictor.get_model_state(), default=str)
            except Exception as e:
                logger.warning(f"Training warm-up for {prediction_type}: {e}")

    def _train_model(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str,
        features_list: list[dict[str, Any]],
    ) -> FitReport:
        """Fit, validate and maybe install one model. Runs off the loop.

        Applies V2 preprocessor chain to each training sample before
        training, ensuring consistency with prediction-time preprocessing.

        Args:
            model_id: Model identifier.
            platform_info: Platform info dict.
            prediction_type: Type of prediction model.
            features_list: Training data with runtime_ms.

        Returns:
            What happened to the candidate.

        Raises:
            ValueError: If prediction type is invalid.
            Exception: On training or storage failure.
        """
        cls = PREDICTOR_CLASSES.get(prediction_type)
        if cls is None:
            raise ValueError(f"Invalid prediction type: {prediction_type}")

        # Apply V2 preprocessor chain if configured
        if self._chain_builder is not None:
            chain = self._chain_builder.get_chain(model_id)
            if chain is not None:
                processed_list = []
                for sample in features_list:
                    runtime_ms = sample.pop("runtime_ms")
                    processed = chain.transform(sample)
                    processed["runtime_ms"] = runtime_ms
                    processed_list.append(processed)
                features_list = processed_list

        model_key = self._storage.generate_model_key(
            model_id=model_id,
            platform_info=platform_info,
            prediction_type=prediction_type,
        )
        installed = self._storage.model_exists(model_key)

        order = self._rng.permutation(len(features_list))
        n_holdout = int(len(features_list) * self.holdout_fraction)
        if n_holdout < MIN_HOLDOUT:
            if installed:
                return FitReport(
                    installed=False,
                    reason=f"holdout of {n_holdout} is too small to replace "
                    "the installed model",
                    n_train=len(features_list),
                    n_holdout=n_holdout,
                    metrics={},
                )
            # Cold platform: a model fitted on everything beats none.
            n_holdout = 0
        holdout = [features_list[k] for k in order[:n_holdout]]
        train = [features_list[k] for k in order[n_holdout:]]

        logger.debug(
            f"Training {model_id} ({prediction_type}) on "
            f"{platform_info.get('hardware_name', '?')} "
            f"with {len(train)} samples, {len(holdout)} held out"
        )

        predictor = cls()
        predictor.train(
            features_list=train, config=self._train_config(prediction_type)
        )

        ok, reason, metrics = (
            self._validate(predictor, model_key, prediction_type, holdout)
            if holdout
            else (True, "no holdout; first model for this platform", {})
        )
        report = FitReport(
            installed=ok,
            reason=reason,
            n_train=len(train),
            n_holdout=len(holdout),
            metrics=metrics,
        )
        if not ok:
            logger.warning(
                f"Kept the installed {model_id} ({prediction_type}) model on "
                f"{platform_info.get('hardware_name', '?')}: {reason}"
            )
            return report

        metadata = {
            "model_id": model_id,
            "platform_info": platform_info,
            "prediction_type": prediction_type,
            "samples_count": len(train),
            "training_config": self._train_config(prediction_type),
            "validation": {"reason": reason, **metrics},
        }
        self._storage.save_model(
            model_key, predictor.get_model_state(), metadata
        )

        logger.info(
            f"Installed {model_id} ({prediction_type}) on "
            f"{platform_info.get('hardware_name', '?')} "
            f"({len(train)} samples; {reason})"
        )
        return report

    @staticmethod
    def _train_config(prediction_type: str) -> dict[str, Any]:
        if prediction_type == "quantile":
            return {
                "quantiles": list(DEFAULT_QUANTILES),
                "data_augmentation": {"enabled": False},
            }
        return {}

    def _validate(
        self,
        predictor: Any,
        model_key: str,
        prediction_type: str,
        holdout: list[dict[str, Any]],
    ) -> tuple[bool, str, dict[str, float]]:
        """Judge a candidate on the holdout.

        Returns:
            (installable, reason, metrics).
        """
        actual = np.array([h["runtime_ms"] for h in holdout])
        rows = [
            {k: v for k, v in h.items() if k != "runtime_ms"} for h in holdout
        ]
        if prediction_type == "quantile":
            preds = [predictor.predict(r)["quantiles"] for r in rows]
            cover = {
                q: float(
                    np.mean(
                        actual <= np.array([float(p[str(q)]) for p in preds])
                    )
                )
                for q in (0.9, 0.99)
            }
            metrics = {
                "q0.9_coverage": cover[0.9],
                "q0.99_coverage": cover[0.99],
            }
            if abs(cover[0.9] - 0.9) > self.coverage_tolerance:
                return False, f"q0.9 coverage {cover[0.9]:.3f}", metrics
            if cover[0.99] < self.min_q99_coverage:
                return False, f"q0.99 coverage {cover[0.99]:.3f}", metrics
            return True, "coverage within target", metrics

        def mae(model: Any) -> float:
            got = np.array(
                [model.predict(r)["expected_runtime_ms"] for r in rows]
            )
            return float(np.mean(np.abs(got - actual)))

        metrics = {"holdout_mae_ms": mae(predictor)}
        current = self._load_installed(model_key, prediction_type)
        if current is None:
            return True, "no installed model to beat", metrics
        try:
            metrics["installed_mae_ms"] = mae(current)
        except Exception as e:
            # The installed model cannot score this data (for example a
            # feature set change); the candidate replaces it.
            return True, f"installed model failed on holdout: {e}", metrics
        if metrics["holdout_mae_ms"] > metrics["installed_mae_ms"]:
            return False, "worse than the installed model", metrics
        return True, "no worse than the installed model", metrics

    def _load_installed(self, model_key: str, prediction_type: str) -> Any:
        state = self._storage.load_model(model_key)
        if state is None:
            return None
        model = PREDICTOR_CLASSES[prediction_type]()
        model.load_model_state(state["predictor_state"])
        return model

    def get_buffer_size(self) -> int:
        """Get current number of buffered training samples."""
        return len(self._samples_buffer)

    def get_last_flush_samples(self) -> int:
        """Get the number of samples consumed in the last flush."""
        return self._last_flush_samples

    def clear_buffer(self) -> None:
        """Clear all buffered training samples without training."""
        count = len(self._samples_buffer)
        self._samples_buffer.clear()
        logger.info(f"Cleared {count} training samples from buffer")

    async def close(self) -> None:
        """Wait for a running retrain, then stop the training thread."""
        if self._background is not None:
            await asyncio.gather(self._background, return_exceptions=True)
        self._executor.shutdown(wait=True)
        logger.info("TrainingClient closed")
