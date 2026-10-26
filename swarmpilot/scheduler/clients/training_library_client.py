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
  ``min_q99_coverage``, or if it beats the installed model on the holdout
  (lower q0.9 pinball loss, q0.9 coverage no further from 0.9), as after
  drift. An expect-error model installs only if its holdout error is no
  worse than the installed model's.
- Installing writes the model file atomically, then drops the cache entry.
- Runtime augmentation is off: online samples carry real variance.

When to retrain (plan item 3.1). A platform with no model trains as soon
as a batch is full. A platform with a model retrains only when the drift
detector (``swarmpilot.scheduler.services.drift``) sees its tail loss or
miss rate rise. Its window then restarts from the samples that showed the
drift, and it retrains once ``recovery_samples`` post-drift samples are in,
so the new model is not fitted on the old distribution. After any install
the model is refitted each time its data doubles, up to the window size,
so one first fitted on 100 samples keeps improving. ``retrain_mode``
``"batch"`` restores the old behaviour of retraining every full batch.
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
from swarmpilot.scheduler.services.drift import DriftDetector, tail_estimate

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
        retrain_mode: str = "drift",
        drift: DriftDetector | None = None,
        recovery_samples: int = 100,
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
            retrain_mode: "drift" retrains a platform that has a model only
                when it drifts; "batch" retrains on every full batch.
            drift: Drift detector; a default one when not given.
            recovery_samples: Post-drift samples to collect before the
                retrain. With the default holdout of 0.2 this keeps a
                holdout of at least 20.
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
        if retrain_mode not in ("drift", "batch"):
            raise ValueError(f"Unknown retrain_mode: {retrain_mode}")
        self.retrain_mode = retrain_mode
        self.drift = drift or DriftDetector()
        self.recovery_samples = recovery_samples
        # Drifted keys: samples seen since the drift was detected.
        self._recovering: dict[tuple[str, str], int] = {}
        # Drifted keys whose window must restart at the drift.
        self._trim: set[tuple[str, str]] = set()
        # Window size at the last install, and samples seen since, per key.
        # A model is refitted each time its data doubles, up to the window
        # size, so a model first fitted on a small sample keeps improving.
        self._fit_size: dict[tuple[str, str], int] = {}
        self._since_fit: dict[tuple[str, str], int] = {}
        # Drifted keys: runtime / predicted median of post-drift tasks, and
        # the pre-drift window with the scale learnt from those ratios.
        self._ratios: dict[tuple[str, str], list[float]] = {}
        self._transfer: dict[
            tuple[str, str], tuple[float, list[TrainingSample]]
        ] = {}
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

    @staticmethod
    def _key(model_id: str, platform_info: dict[str, str]) -> tuple[str, str]:
        return (model_id, json.dumps(platform_info, sort_keys=True))

    def add_sample(
        self,
        model_id: str,
        platform_info: dict[str, str],
        features: dict[str, Any],
        actual_runtime_ms: float,
        predicted_quantiles: dict[float, float] | None = None,
        predicted_time_ms: float | None = None,
        predicted_error_ms: float | None = None,
    ) -> None:
        """Add a training sample and score the prediction it was routed by.

        Args:
            model_id: Model/tool identifier.
            platform_info: Platform information.
            features: Task features.
            actual_runtime_ms: Actual execution time in ms.
            predicted_quantiles: Quantile prediction used to route the task.
            predicted_time_ms: Expected runtime used to route the task.
            predicted_error_ms: Error margin of ``predicted_time_ms``.
        """
        sample = TrainingSample(
            model_id=model_id,
            platform_info=platform_info,
            features=features,
            actual_runtime_ms=actual_runtime_ms,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._samples_buffer.append(sample)

        key = self._key(model_id, platform_info)
        self._since_fit[key] = self._since_fit.get(key, 0) + 1
        if key in self._recovering:
            self._recovering[key] += 1
            median = tail_estimate(
                0.5, predicted_quantiles, predicted_time_ms, None
            )
            if median and median > 0:
                self._ratios[key].append(actual_runtime_ms / median)
            return
        tail = tail_estimate(
            self.drift.tau,
            predicted_quantiles,
            predicted_time_ms,
            predicted_error_ms,
        )
        if (
            self.retrain_mode == "drift"
            and tail is not None
            and self.drift.observe(key, features, tail, actual_runtime_ms)
        ):
            # Only samples after the detection count: a large shift is
            # caught within a few tasks, so the ones before it are mostly
            # from the old distribution.
            self._recovering[key] = 0
            self._ratios[key] = []
            self._trim.add(key)
            logger.warning(
                f"Predictor drift for {model_id} on "
                f"{platform_info.get('hardware_name', '?')}; retraining "
                f"after {self.recovery_samples} recent samples"
            )

    def _is_warm(self, model_id: str, platform_info: dict[str, str]) -> bool:
        return all(
            self._storage.model_exists(
                self._storage.generate_model_key(
                    model_id=model_id,
                    platform_info=platform_info,
                    prediction_type=t,
                )
            )
            for t in self.prediction_types
        )

    def _due(self) -> set[tuple[str, str]] | None:
        """Keys to retrain now; None means every key in the buffer."""
        due = {
            k for k, n in self._recovering.items() if n >= self.recovery_samples
        }
        if self.retrain_mode == "drift":
            due |= {
                k
                for k, n in self._fit_size.items()
                if k not in self._recovering
                and n < self.window_size
                and self._since_fit.get(k, 0) >= n
            }
        if len(self._samples_buffer) >= self.batch_size:
            if self.retrain_mode == "batch":
                return None
            due |= {
                self._key(s.model_id, s.platform_info)
                for s in self._samples_buffer
                if not self._is_warm(s.model_id, s.platform_info)
            }
        return due

    async def flush_if_ready(self) -> bool:
        """Retrain what is due (see the module docstring).

        Returns:
            True if training ran and succeeded, False otherwise.
        """
        due = self._due()
        if due is None or due:
            return await self.flush(only=due)
        if len(self._samples_buffer) >= self.batch_size:
            self._absorb()
        return False

    def start_flush_if_ready(self) -> bool:
        """Start a background retrain if one is due and none is running.

        For the result callback: it must not wait for a retrain.

        Returns:
            True if a retrain was started.
        """
        if self._background is not None and not self._background.done():
            return False
        due = self._due()
        if due is not None and not due:
            if len(self._samples_buffer) >= self.batch_size:
                self._absorb()
            return False
        self._background = asyncio.get_running_loop().create_task(
            self.flush(only=due)
        )
        return True

    def _absorb(self) -> set[tuple[str, str]]:
        """Move buffered samples into their windows; return touched keys."""
        batch, self._samples_buffer = self._samples_buffer, []
        touched: set[tuple[str, str]] = set()
        for sample in batch:
            key = self._key(sample.model_id, sample.platform_info)
            window = self._windows.get(key)
            if window is None:
                window = deque(maxlen=self.window_size)
                self._windows[key] = window
            window.append(sample)
            touched.add(key)
        for key in self._trim:
            window = self._windows.get(key)
            if window is not None:
                n = self._recovering.get(key, 0)
                old = list(window)
                recent = old[-n:] if n else []
                window.clear()
                window.extend(recent)
                self._transfer[key] = (1.0, old[: len(old) - n])
        self._trim.clear()
        self._last_flush_samples = len(batch)
        return touched

    async def flush(
        self,
        force: bool = False,
        only: set[tuple[str, str]] | None = None,
    ) -> bool:
        """Absorb buffered samples and train models on their windows.

        Args:
            force: If True, train even below min_samples threshold.
            only: Train just these (model, platform) keys; None trains
                every key with buffered samples.

        Returns:
            True if all training succeeded, False otherwise.
        """
        if only is None:
            if not self._samples_buffer:
                logger.debug("No training samples to flush")
                return False
            if not force and len(self._samples_buffer) < self.min_samples:
                logger.debug(
                    f"Buffer has {len(self._samples_buffer)} samples, "
                    f"below minimum of {self.min_samples}. Skipping training."
                )
                return False

        n_batch = len(self._samples_buffer)
        touched = self._absorb()
        keys = touched if only is None else set(only)
        keys = {k for k in keys if self._windows.get(k)}

        logger.info(
            f"Flushing {n_batch} training samples; training "
            f"{len(keys)} model-platform combinations"
        )

        success_count = 0
        failure_count = 0
        loop = asyncio.get_running_loop()

        async with self._flush_lock:
            # Samples keep arriving while a fit runs on its thread, and one
            # may flag drift. Only a recovery that began before the fit is
            # finished by it; a newer one keeps its count and its trim.
            recovering = {k for k in keys if k in self._recovering}
            for key in keys:
                model_id = key[0]
                samples = list(self._windows[key])
                platform_info = samples[0].platform_info
                features_list = [
                    {**s.features, "runtime_ms": s.actual_runtime_ms}
                    for s in samples
                ]
                extra = self._transfer_rows(key, len(samples))
                installed_any = False
                for prediction_type in self.prediction_types:
                    try:
                        report = await loop.run_in_executor(
                            self._executor,
                            self._train_model,
                            model_id,
                            platform_info,
                            prediction_type,
                            [dict(f) for f in features_list],
                            [dict(f) for f in extra],
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
                    model_key = self._storage.generate_model_key(
                        model_id=model_id,
                        platform_info=platform_info,
                        prediction_type=prediction_type,
                    )
                    self.last_reports[model_key] = report
                    if report.installed:
                        installed_any = True
                        # The file is already swapped; drop the stale copy.
                        self._cache.invalidate(model_key)
                self._since_fit[key] = 0
                if installed_any:
                    self._fit_size[key] = len(samples)
                if key in recovering:
                    del self._recovering[key]
                    self._ratios.pop(key, None)
                    if installed_any:
                        self.drift.reset(key)
                    else:
                        self.drift.rearm(key)
                elif installed_any and key not in self._recovering:
                    self.drift.reset(key)

        logger.info(
            f"Training complete: {success_count} succeeded, "
            f"{failure_count} failed, {n_batch} samples consumed"
        )
        return failure_count == 0

    def _transfer_rows(
        self, key: tuple[str, str], n_real: int
    ) -> list[dict[str, Any]]:
        """Pre-drift samples rescaled to the new runtimes, as extra rows.

        After drift the window restarts, so the first refits see few real
        samples. Most shifts are close to multiplicative (a slower GPU,
        contention, a new engine version), so the pre-drift window, scaled
        by the median of runtime / predicted median over post-drift tasks,
        fills the window until real samples do. Validation uses real
        samples only.
        """
        if key not in self._transfer:
            return []
        scale, old = self._transfer[key]
        ratios = self._ratios.get(key)
        if ratios:
            scale = float(np.median(ratios))
            self._transfer[key] = (scale, old)
        room = self.window_size - n_real
        if room <= 0:
            del self._transfer[key]
            return []
        return [
            {**s.features, "runtime_ms": s.actual_runtime_ms * scale}
            for s in old[-room:]
        ]

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
        extra_train: list[dict[str, Any]] | None = None,
    ) -> FitReport:
        """Fit, validate and maybe install one model. Runs off the loop.

        Applies V2 preprocessor chain to each training sample before
        training, ensuring consistency with prediction-time preprocessing.

        Args:
            model_id: Model identifier.
            platform_info: Platform info dict.
            prediction_type: Type of prediction model.
            features_list: Training data with runtime_ms.
            extra_train: Rows used only for fitting, never held out.

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
        extra_train = list(extra_train or [])
        if self._chain_builder is not None:
            chain = self._chain_builder.get_chain(model_id)
            if chain is not None:

                def transform(rows: list[dict[str, Any]]) -> list[dict]:
                    out = []
                    for sample in rows:
                        runtime_ms = sample.pop("runtime_ms")
                        processed = chain.transform(sample)
                        processed["runtime_ms"] = runtime_ms
                        out.append(processed)
                    return out

                features_list = transform(features_list)
                extra_train = transform(extra_train)

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
        train = [features_list[k] for k in order[n_holdout:]] + extra_train

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

            def score(model: Any) -> tuple[dict[float, float], float]:
                preds = [model.predict(r)["quantiles"] for r in rows]
                cover = {
                    q: float(
                        np.mean(
                            actual
                            <= np.array([float(p[str(q)]) for p in preds])
                        )
                    )
                    for q in (0.9, 0.99)
                }
                q90 = np.array([float(p["0.9"]) for p in preds])
                diff = actual - q90
                pinball = float(np.mean(np.maximum(0.9 * diff, -0.1 * diff)))
                return cover, pinball

            cover, pinball = score(predictor)
            metrics = {
                "q0.9_coverage": cover[0.9],
                "q0.99_coverage": cover[0.99],
                "q0.9_pinball_ms": pinball,
            }
            on_target = abs(cover[0.9] - 0.9) <= self.coverage_tolerance and (
                cover[0.99] >= self.min_q99_coverage
            )
            if on_target:
                return True, "coverage within target", metrics
            # Off target, but the installed model may be worse still (it
            # has drifted): then the candidate is the better of the two.
            current = self._load_installed(model_key, prediction_type)
            if current is not None:
                try:
                    old_cover, old_pinball = score(current)
                except Exception:
                    old_cover, old_pinball = None, None
                if old_cover is not None:
                    metrics["installed_q0.9_coverage"] = old_cover[0.9]
                    metrics["installed_q0.9_pinball_ms"] = old_pinball
                    if pinball < old_pinball and abs(cover[0.9] - 0.9) <= abs(
                        old_cover[0.9] - 0.9
                    ):
                        return (
                            True,
                            "better than the installed model on the holdout",
                            metrics,
                        )
            if abs(cover[0.9] - 0.9) > self.coverage_tolerance:
                return False, f"q0.9 coverage {cover[0.9]:.3f}", metrics
            return False, f"q0.99 coverage {cover[0.99]:.3f}", metrics

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
