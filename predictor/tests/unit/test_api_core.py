"""Unit tests for the predictor library API.

Following TDD approach - these tests are written BEFORE implementation.
Tests cover both PredictorLowLevel (low-level API) and PredictorCore (high-level API).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.models import PlatformInfo


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def platform_info() -> PlatformInfo:
    """Create a standard platform info for testing."""
    return PlatformInfo(
        software_name="PyTorch",
        software_version="2.0",
        hardware_name="NVIDIA A100",
    )


@pytest.fixture
def sample_features() -> dict[str, Any]:
    """Create sample features for testing."""
    return {
        "batch_size": 32,
        "image_size": 224,
    }


@pytest.fixture
def training_data() -> list[dict[str, Any]]:
    """Create minimum training data (10 samples required)."""
    return [
        {"batch_size": 8, "image_size": 224, "runtime_ms": 50.0},
        {"batch_size": 16, "image_size": 224, "runtime_ms": 95.0},
        {"batch_size": 32, "image_size": 224, "runtime_ms": 180.0},
        {"batch_size": 64, "image_size": 224, "runtime_ms": 350.0},
        {"batch_size": 128, "image_size": 224, "runtime_ms": 690.0},
        {"batch_size": 8, "image_size": 112, "runtime_ms": 25.0},
        {"batch_size": 16, "image_size": 112, "runtime_ms": 48.0},
        {"batch_size": 32, "image_size": 112, "runtime_ms": 90.0},
        {"batch_size": 64, "image_size": 112, "runtime_ms": 175.0},
        {"batch_size": 128, "image_size": 112, "runtime_ms": 345.0},
    ]


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "models"
    storage_dir.mkdir()
    return storage_dir


# =============================================================================
# PredictorLowLevel Tests
# =============================================================================


class TestPredictorLowLevel:
    """Tests for low-level API."""

    # -------------------------------------------------------------------------
    # Model Management Tests
    # -------------------------------------------------------------------------

    def test_train_predictor_creates_valid_predictor(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """train_predictor should return a trained predictor instance."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        assert predictor is not None
        assert hasattr(predictor, "predict")
        assert hasattr(predictor, "feature_names")
        assert set(predictor.feature_names) == {"batch_size", "image_size"}

    def test_train_predictor_minimum_samples(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Should raise ValidationError if < 10 samples."""
        from src.api.core import PredictorLowLevel, ValidationError

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Only 5 samples - should fail
        insufficient_data = [
            {"batch_size": i * 8, "image_size": 224, "runtime_ms": i * 50.0}
            for i in range(1, 6)
        ]

        with pytest.raises(ValidationError, match="minimum.*10.*samples"):
            low.train_predictor(
                features_list=insufficient_data,
                prediction_type="expect_error",
            )

    def test_save_and_load_model(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """save_model + load_model should roundtrip correctly."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Train and save
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("test_model", platform_info, "expect_error", predictor)

        # Load and verify
        loaded = low.load_model("test_model", platform_info, "expect_error")
        assert loaded is not None
        assert loaded.feature_names == predictor.feature_names

        # Predictions should be identical
        test_features = {"batch_size": 32, "image_size": 224}
        original_result = predictor.predict(test_features)
        loaded_result = loaded.predict(test_features)
        assert original_result == loaded_result

    def test_delete_model_removes_from_storage(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """delete_model should remove model file."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Train and save
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("test_model", platform_info, "expect_error", predictor)

        # Verify exists
        assert low.model_exists("test_model", platform_info, "expect_error")

        # Delete
        result = low.delete_model("test_model", platform_info, "expect_error")
        assert result is True

        # Verify deleted
        assert not low.model_exists("test_model", platform_info, "expect_error")

    def test_list_models_returns_all_models(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """list_models should return all stored models."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Create multiple models
        for model_id in ["model_a", "model_b", "model_c"]:
            predictor = low.train_predictor(
                features_list=training_data,
                prediction_type="expect_error",
            )
            low.save_model(model_id, platform_info, "expect_error", predictor)

        # List all
        models = low.list_models()
        assert len(models) == 3

        model_ids = [m.model_id for m in models]
        assert "model_a" in model_ids
        assert "model_b" in model_ids
        assert "model_c" in model_ids

    def test_list_models_filter_by_prediction_type(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """list_models should filter by prediction_type."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Create models with different prediction types
        for pred_type in ["expect_error", "quantile"]:
            predictor = low.train_predictor(
                features_list=training_data,
                prediction_type=pred_type,
            )
            low.save_model(f"model_{pred_type}", platform_info, pred_type, predictor)

        # Filter by type
        expect_error_models = low.list_models(prediction_type="expect_error")
        assert len(expect_error_models) == 1
        assert expect_error_models[0].prediction_type == "expect_error"

        quantile_models = low.list_models(prediction_type="quantile")
        assert len(quantile_models) == 1
        assert quantile_models[0].prediction_type == "quantile"

    def test_model_exists_returns_true_for_existing(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """model_exists should return True for saved models."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("existing_model", platform_info, "expect_error", predictor)

        assert low.model_exists("existing_model", platform_info, "expect_error") is True

    def test_model_exists_returns_false_for_missing(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """model_exists should return False for non-existent models."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        assert low.model_exists("nonexistent", platform_info, "expect_error") is False

    # -------------------------------------------------------------------------
    # Prediction Tests
    # -------------------------------------------------------------------------

    def test_predict_with_predictor_returns_result(
        self,
        training_data: list[dict[str, Any]],
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """predict_with_predictor should return prediction dict."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        result = low.predict_with_predictor(predictor, sample_features)

        assert isinstance(result, dict)
        assert "expected_runtime_ms" in result
        assert "error_margin_ms" in result
        assert isinstance(result["expected_runtime_ms"], (int, float))
        assert isinstance(result["error_margin_ms"], (int, float))

    def test_predict_auto_filters_extra_features(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Extra features should be filtered out automatically."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        # Features with extra keys not in training data
        features_with_extra = {
            "batch_size": 32,
            "image_size": 224,
            "extra_feature": 999,  # Should be filtered out
            "another_extra": "ignored",
        }

        # Should not raise - extra features are ignored
        result = low.predict_with_predictor(predictor, features_with_extra)
        assert "expected_runtime_ms" in result

    def test_predict_raises_on_missing_features(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Should raise ValidationError if required features missing."""
        from src.api.core import PredictorLowLevel, ValidationError

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        # Missing "image_size"
        incomplete_features = {"batch_size": 32}

        with pytest.raises(ValidationError, match="Missing required features"):
            low.predict_with_predictor(predictor, incomplete_features)

    # -------------------------------------------------------------------------
    # Cache Tests
    # -------------------------------------------------------------------------

    def test_clear_cache_empties_cache(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """clear_cache should empty the model cache."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Train, save, and load to populate cache
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("cached_model", platform_info, "expect_error", predictor)
        low.load_model("cached_model", platform_info, "expect_error")

        # Verify cache has entries
        stats_before = low.get_cache_stats()
        assert stats_before["size"] > 0

        # Clear cache
        low.clear_cache()

        # Verify cache is empty
        stats_after = low.get_cache_stats()
        assert stats_after["size"] == 0

    def test_invalidate_cache_removes_specific_model(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """invalidate_cache should remove only the specified model."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Create and cache two models
        for model_id in ["model_1", "model_2"]:
            predictor = low.train_predictor(
                features_list=training_data,
                prediction_type="expect_error",
            )
            low.save_model(model_id, platform_info, "expect_error", predictor)
            low.load_model(model_id, platform_info, "expect_error")

        # Verify both are cached
        stats_before = low.get_cache_stats()
        assert stats_before["size"] == 2

        # Invalidate only model_1
        low.invalidate_cache("model_1", platform_info, "expect_error")

        # Verify only model_1 is removed
        stats_after = low.get_cache_stats()
        assert stats_after["size"] == 1

    def test_get_cache_stats_returns_statistics(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """get_cache_stats should return hit/miss statistics."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Initial stats
        stats = low.get_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats

        # Populate cache
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("test_model", platform_info, "expect_error", predictor)

        # First load - cache miss
        low.load_model("test_model", platform_info, "expect_error")

        # Second load - cache hit
        low.load_model("test_model", platform_info, "expect_error")

        stats_after = low.get_cache_stats()
        assert stats_after["hits"] >= 1
        assert stats_after["misses"] >= 1


# =============================================================================
# PredictorCore Tests
# =============================================================================


class TestPredictorCore:
    """Tests for high-level API."""

    # -------------------------------------------------------------------------
    # Accumulator Pattern Tests
    # -------------------------------------------------------------------------

    def test_collect_adds_sample_to_accumulator(
        self,
        platform_info: PlatformInfo,
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """collect should add sample to internal accumulator."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Initially empty
        assert core.get_collected_count("test", platform_info, "expect_error") == 0

        # Add sample
        core.collect(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features=sample_features,
            runtime_ms=100.0,
        )

        # Should have 1 sample
        assert core.get_collected_count("test", platform_info, "expect_error") == 1

    def test_get_collected_count_returns_correct_count(
        self,
        platform_info: PlatformInfo,
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """get_collected_count should return number of collected samples."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Add multiple samples
        for i in range(5):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        assert core.get_collected_count("test", platform_info, "expect_error") == 5

    def test_clear_collected_removes_samples(
        self,
        platform_info: PlatformInfo,
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """clear_collected should remove all accumulated samples."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Add samples
        for i in range(3):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        assert core.get_collected_count("test", platform_info, "expect_error") == 3

        # Clear
        core.clear_collected("test", platform_info, "expect_error")

        # Should be empty
        assert core.get_collected_count("test", platform_info, "expect_error") == 0

    def test_collect_stores_features_and_runtime(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """collect should store both features and runtime_ms."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        features = {"batch_size": 32, "image_size": 224}
        runtime = 150.5

        core.collect(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features=features,
            runtime_ms=runtime,
        )

        # Verify stored data (internal access for testing)
        key = core._make_accumulator_key("test", platform_info, "expect_error")
        samples = core._accumulated[key]

        assert len(samples) == 1
        assert samples[0].features == features
        assert samples[0].runtime_ms == runtime

    # -------------------------------------------------------------------------
    # Training Tests
    # -------------------------------------------------------------------------

    def test_train_uses_accumulated_data(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """train should use data from collect(), not accept features_list."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Collect 10 samples
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        # Train without features_list parameter
        result = core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        assert result.success is True
        assert result.samples_trained == 10

    def test_train_clears_accumulator_after_success(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """train should clear accumulated samples after training."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Collect samples
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        assert core.get_collected_count("test", platform_info, "expect_error") == 10

        # Train
        core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Should be cleared
        assert core.get_collected_count("test", platform_info, "expect_error") == 0

    def test_train_raises_if_no_collected_data(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """train should raise if no samples were collected."""
        from src.api.core import PredictorCore, ValidationError

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        with pytest.raises(ValidationError, match="No.*samples.*collected"):
            core.train(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
            )

    def test_train_raises_if_insufficient_samples(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """train should raise if < 10 samples collected."""
        from src.api.core import PredictorCore, ValidationError

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Only 5 samples
        for i in range(5):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        with pytest.raises(ValidationError, match="minimum.*10.*samples"):
            core.train(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
            )

    @pytest.mark.asyncio
    async def test_train_async_runs_in_background(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """train_async should be non-blocking."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Collect samples
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        # Should return awaitable
        result = await core.train_async(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        assert result.success is True

    # -------------------------------------------------------------------------
    # Feature Schema Tests
    # -------------------------------------------------------------------------

    def test_first_collect_defines_feature_schema(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """First collected sample defines expected features."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # First collect defines schema: batch_size, image_size
        core.collect(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
            runtime_ms=100.0,
        )

        # Check schema was recorded
        key = core._make_accumulator_key("test", platform_info, "expect_error")
        expected_features = core._feature_schemas.get(key)

        assert expected_features is not None
        assert set(expected_features) == {"batch_size", "image_size"}

    def test_collect_validates_feature_consistency(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """Subsequent collects should have same features as first."""
        from src.api.core import PredictorCore, ValidationError

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # First collect
        core.collect(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
            runtime_ms=100.0,
        )

        # Second collect with different features should raise
        with pytest.raises(ValidationError, match="inconsistent features"):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 32, "different_feature": 999},
                runtime_ms=150.0,
            )

    # -------------------------------------------------------------------------
    # Prediction Tests
    # -------------------------------------------------------------------------

    def test_predict_returns_prediction_result(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """predict should return PredictionResult."""
        from src.api.core import PredictorCore
        from src.models import PredictionResult

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Train a model first
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )
        core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Predict
        result = core.predict(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
        )

        assert isinstance(result, PredictionResult)
        assert result.model_id == "test"
        assert "expected_runtime_ms" in result.result

    def test_predict_auto_filters_extra_features(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """Extra features should be filtered automatically."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Train
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )
        core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Predict with extra features - should not raise
        result = core.predict(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={
                "batch_size": 32,
                "image_size": 224,
                "extra_feature": 999,  # Should be ignored
            },
        )

        assert result is not None

    def test_predict_raises_on_missing_features(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """Should raise ValidationError if features missing."""
        from src.api.core import PredictorCore, ValidationError

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Train
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )
        core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Predict with missing feature
        with pytest.raises(ValidationError, match="Missing required features"):
            core.predict(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 32},  # Missing image_size
            )

    def test_batch_predict_returns_multiple_results(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """batch_predict should return results for all inputs."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Train
        for i in range(10):
            core.collect(
                model_id="test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )
        core.train(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Batch predict
        features_list = [
            {"batch_size": 16, "image_size": 224},
            {"batch_size": 32, "image_size": 224},
            {"batch_size": 64, "image_size": 224},
        ]

        results = core.batch_predict(
            model_id="test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features_list=features_list,
        )

        assert len(results) == 3
        for result in results:
            assert "expected_runtime_ms" in result.result


# =============================================================================
# Integration Tests
# =============================================================================


class TestPredictorIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_collect_train_predict(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """Full workflow: collect -> train -> predict."""
        from src.api.core import PredictorCore

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        # Step 1: Collect samples
        for i in range(10):
            core.collect(
                model_id="workflow_test",
                platform_info=platform_info,
                prediction_type="quantile",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        # Step 2: Train
        train_result = core.train(
            model_id="workflow_test",
            platform_info=platform_info,
            prediction_type="quantile",
            config={"quantiles": [0.5, 0.9, 0.99]},
        )

        assert train_result.success is True

        # Step 3: Predict
        pred_result = core.predict(
            model_id="workflow_test",
            platform_info=platform_info,
            prediction_type="quantile",
            features={"batch_size": 32, "image_size": 224},
        )

        assert pred_result is not None
        assert "quantiles" in pred_result.result

    def test_low_level_direct_training_workflow(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Low-level: train_predictor -> save_model -> load_model -> predict."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Step 1: Train predictor directly
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        # Step 2: Save model
        low.save_model("direct_test", platform_info, "expect_error", predictor)

        # Step 3: Load model
        loaded = low.load_model("direct_test", platform_info, "expect_error")

        # Step 4: Predict
        result = low.predict_with_predictor(
            loaded,
            {"batch_size": 32, "image_size": 224},
        )

        assert "expected_runtime_ms" in result

    def test_high_level_wraps_low_level(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """PredictorCore should delegate to PredictorLowLevel."""
        from src.api.core import PredictorCore, PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        core = PredictorCore(low_level=low)

        # Use high-level to train
        for i in range(10):
            core.collect(
                model_id="wrapped_test",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 8 * (i + 1), "image_size": 224},
                runtime_ms=50.0 * (i + 1),
            )

        core.train(
            model_id="wrapped_test",
            platform_info=platform_info,
            prediction_type="expect_error",
        )

        # Verify model exists via low-level
        assert low.model_exists("wrapped_test", platform_info, "expect_error")

        # Load via low-level and predict
        loaded = low.load_model("wrapped_test", platform_info, "expect_error")
        result = low.predict_with_predictor(
            loaded,
            {"batch_size": 32, "image_size": 224},
        )

        assert result is not None
