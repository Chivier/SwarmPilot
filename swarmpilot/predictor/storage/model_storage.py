"""Model storage layer for persisting and retrieving trained models.

Uses joblib for serialization and local filesystem for storage.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from swarmpilot.predictor.utils.logging import get_logger

logger = get_logger()


def _json_default(value: Any) -> Any:
    """Convert numpy scalars and arrays for the JSON sidecar."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class ModelStorage:
    """Manages model persistence using local filesystem.

    Models are stored with metadata including training date, sample count, etc.

    Attributes:
        storage_dir: Path to the directory where models are stored.
    """

    #: How long model_exists() trusts a miss before checking disk again.
    MISSING_TTL_S: float = 2.0

    def __init__(self, storage_dir: str = "models") -> None:
        """Initialize model storage.

        Args:
            storage_dir: Directory for storing models (created if not exists).
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[str, Any] = {}
        self._known: set[str] = set()
        self._missing: dict[str, float] = {}

    def generate_model_key(
        self,
        model_id: str,
        platform_info: dict[str, str],
        prediction_type: str = "expect_error",
    ) -> str:
        """Generate unique model key from model_id and platform_info.

        Format: {model_id}__{software}-{version}__{hardware}__{prediction_type}

        When ``platform_info`` carries a ``workload`` (set by engine
        adapters), the key also holds the workload and feature schema:
        {model_id}__{workload}__{software}-{version}__{hardware}__fs{N}__{type}
        so a model trained on other features is never reused.

        Args:
            model_id: Model identifier.
            platform_info: Platform information dict with software_name,
                software_version, hardware_name.
            prediction_type: Type of prediction model (expect_error or quantile).

        Returns:
            Unique model key string.
        """
        software = f"{platform_info['software_name']}-{platform_info['software_version']}"
        hardware = platform_info["hardware_name"]
        workload = platform_info.get("workload")
        if workload:
            schema = platform_info.get("feature_schema") or "0"
            return (
                f"{model_id}__{workload}__{software}__{hardware}"
                f"__fs{schema}__{prediction_type}"
            )
        return f"{model_id}__{software}__{hardware}__{prediction_type}"

    def save_model(
        self,
        model_key: str,
        predictor_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Save model and metadata to disk.

        Args:
            model_key: Unique key for the model.
            predictor_state: Complete predictor state from get_model_state().
            metadata: Additional metadata (model_id, platform_info, etc).

        Raises:
            Exception: If saving fails (joblib error, disk error, etc).
        """
        # Create complete state with metadata
        complete_state = {
            "predictor_state": predictor_state,
            "metadata": metadata,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        # Save to disk
        model_path = self.storage_dir / f"{model_key}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(complete_state, model_path)
            self._known.add(model_key)
            self._missing.pop(model_key, None)
            logger.debug(f"Model saved successfully: {model_path}")
        except (OSError, RuntimeError) as e:
            logger.opt(exception=True).error(
                f"Failed to save model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Metadata: {metadata}\n"
                f"Exception: {type(e).__name__}: {e!s}"
            )
            raise

        try:
            self.export_portable(model_key, predictor_state, metadata)
        except (OSError, RuntimeError, TypeError, ValueError, ImportError) as e:
            # joblib stays the source of truth; the portable copy is for
            # non-Python readers and must never block training.
            logger.warning(f"Portable export failed for {model_key}: {e}")

    def export_portable(
        self,
        model_key: str,
        predictor_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        """Write the model as safetensors weights plus a JSON sidecar.

        This is the predictor model file of the Rust migration plan: any
        language can load it without unpickling. ``{key}.safetensors``
        holds the network weights; ``{key}.json`` holds the model key, the
        platform it was trained on (workload, engine, version, hardware,
        feature schema), the architecture, the feature and target scaling
        and every other non-tensor field of the predictor state, plus the
        SHA-256 of the weights file (named relative to the sidecar's
        directory). Both are written to temporary names
        and renamed, weights first, so a reader that sees the sidecar
        always finds matching weights.

        Args:
            model_key: Unique key for the model.
            predictor_state: State from ``get_model_state()``.
            metadata: Model metadata (model_id, platform_info, ...).

        Returns:
            Path of the JSON sidecar.

        Raises:
            TypeError: If a non-tensor state field is not JSON-serializable.
        """
        import hashlib

        from safetensors.torch import save_file

        tensors = {
            name: t.detach().cpu().contiguous()
            for name, t in predictor_state.get("model_state_dict", {}).items()
        }
        rest = {
            k: v for k, v in predictor_state.items() if k != "model_state_dict"
        }
        weights = self.storage_dir / f"{model_key}.safetensors"
        sidecar = self.storage_dir / f"{model_key}.json"
        weights.parent.mkdir(parents=True, exist_ok=True)

        tmp = weights.with_name(weights.name + ".tmp")
        save_file(tensors, str(tmp), metadata={"model_key": model_key})
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, weights)

        doc = {
            "format": "swarmpilot-predictor",
            "format_version": 1,
            "model_key": model_key,
            "model_id": metadata.get("model_id"),
            "prediction_type": metadata.get("prediction_type"),
            "platform_info": metadata.get("platform_info"),
            "samples_count": metadata.get("samples_count"),
            "weights_file": weights.name,
            "weights_sha256": digest,
            "activation": "relu",
            "state": rest,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2, default=_json_default))
        os.replace(tmp, sidecar)
        logger.debug(f"Portable model written: {sidecar}")
        return sidecar

    def load_model(self, model_key: str) -> dict[str, Any] | None:
        """Load model and metadata from disk.

        Args:
            model_key: Unique key for the model.

        Returns:
            Dict with predictor_state and metadata keys, or None if not found.

        Raises:
            Exception: If loading fails (corrupted file, deserialization error).
        """
        model_path = self.storage_dir / f"{model_key}.joblib"

        if not model_path.exists():
            logger.debug(f"Model not found at path: {model_path}")
            return None

        try:
            result = joblib.load(model_path)
            logger.debug(f"Model loaded successfully: {model_path}")
            return result
        except (OSError, EOFError, RuntimeError) as e:
            logger.opt(exception=True).error(
                f"Failed to load model\n"
                f"Model key: {model_key}\n"
                f"Model path: {model_path}\n"
                f"Exception: {type(e).__name__}: {e!s}"
            )
            raise

    def model_exists(self, model_key: str) -> bool:
        """Check if a model exists in storage.

        Args:
            model_key: Unique key for the model.

        Returns:
            True if model exists, False otherwise.
        """
        # Called for every instance on every routed request, so answers are
        # cached: a model seen once stays known until delete_model, and a
        # miss is remembered for MISSING_TTL_S so a cold platform does not
        # cost a filesystem stat per request. Models written by another
        # process show up within that TTL.
        if model_key in self._known:
            return True
        now = time.monotonic()
        expiry = self._missing.get(model_key)
        if expiry is not None and now < expiry:
            return False
        if (self.storage_dir / f"{model_key}.joblib").exists():
            self._known.add(model_key)
            self._missing.pop(model_key, None)
            return True
        self._missing[model_key] = now + self.MISSING_TTL_S
        return False

    def list_models(self) -> list[dict[str, Any]]:
        """List all stored models with their metadata.

        Returns:
            List of model metadata dicts.
        """
        models = []

        for model_file in self.storage_dir.glob("*.joblib"):
            try:
                complete_state = joblib.load(model_file)
                metadata = complete_state.get("metadata", {})

                # Extract key information
                model_info = {
                    "model_id": metadata.get("model_id"),
                    "platform_info": metadata.get("platform_info"),
                    "prediction_type": metadata.get("prediction_type"),
                    "samples_count": metadata.get("samples_count"),
                    "last_trained": complete_state.get("saved_at"),
                }

                models.append(model_info)

            except (OSError, EOFError, RuntimeError, KeyError) as e:
                # Log corrupted files with full details
                logger.opt(exception=True).warning(
                    f"Failed to load model file (skipping)\n"
                    f"File: {model_file}\n"
                    f"Exception: {type(e).__name__}: {e!s}"
                )
                continue

        return models

    def delete_model(self, model_key: str) -> bool:
        """Delete a model from storage.

        Args:
            model_key: Unique key for the model.

        Returns:
            True if model was deleted, False if not found.
        """
        model_path = self.storage_dir / f"{model_key}.joblib"
        self._known.discard(model_key)

        if not model_path.exists():
            return False

        model_path.unlink()
        for suffix in (".safetensors", ".json"):
            (self.storage_dir / f"{model_key}{suffix}").unlink(missing_ok=True)
        return True

    def get_storage_info(self) -> dict[str, Any]:
        """Get information about the storage system.

        Returns:
            Dict with storage statistics (dir, count, size, accessibility).
        """
        model_files = list(self.storage_dir.glob("*.joblib"))
        total_size = sum(f.stat().st_size for f in model_files)

        return {
            "storage_dir": str(self.storage_dir.absolute()),
            "model_count": len(model_files),
            "total_size_bytes": total_size,
            "is_accessible": self.storage_dir.exists()
            and os.access(self.storage_dir, os.W_OK),
        }
