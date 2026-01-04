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


# =============================================================================
# Inference Pipeline API Tests
# =============================================================================


class MockPreprocessor:
    """Mock preprocessor for testing pipeline API."""

    def __init__(
        self,
        output_key: str = "processed",
        remove_origin: bool = False,
        output_value_fn: callable = None,
    ):
        """Initialize mock preprocessor.

        Args:
            output_key: Key for output dictionary.
            remove_origin: Whether to remove original feature.
            output_value_fn: Function to transform input value. Defaults to identity.
        """
        self.output_key = output_key
        self.remove_origin = remove_origin
        self.output_value_fn = output_value_fn or (lambda x: x)
        self.call_count = 0

    def __call__(self, input_text: list[str]) -> tuple[dict[str, Any], bool]:
        """Process input and return output dict with remove flag.

        Args:
            input_text: List of input values (usually strings).

        Returns:
            Tuple of (output_dict, remove_origin_flag).
        """
        self.call_count += 1
        # Apply transformation to first input
        value = input_text[0] if input_text else None
        output = {self.output_key: self.output_value_fn(value)}
        return output, self.remove_origin


class TestInferencePipelineAPI:
    """Tests for inference pipeline API with per-feature preprocessing."""

    # -------------------------------------------------------------------------
    # apply_preprocess_pipeline Tests
    # -------------------------------------------------------------------------

    def test_apply_preprocess_pipeline_single_preprocessor(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Single preprocessor should be applied to specified feature."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Register mock preprocessor
        mock_prep = MockPreprocessor(output_key="text_encoded", remove_origin=False)
        low._preprocessors_registry._preprocessors["mock_encoder"] = mock_prep

        features = {"text": "hello world", "batch_size": 32}
        preprocess_config = {"text": ["mock_encoder"]}

        result = low.apply_preprocess_pipeline(features, preprocess_config)

        assert "text_encoded" in result
        assert result["text_encoded"] == "hello world"
        assert result["text"] == "hello world"  # Original preserved
        assert result["batch_size"] == 32  # Unprocessed feature preserved
        assert mock_prep.call_count == 1

    def test_apply_preprocess_pipeline_preserves_unprocessed_features(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Features not in config should be passed through unchanged."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        mock_prep = MockPreprocessor(output_key="processed_text")
        low._preprocessors_registry._preprocessors["text_processor"] = mock_prep

        features = {
            "text": "input",
            "batch_size": 32,
            "image_size": 224,
            "other_param": "value",
        }
        preprocess_config = {"text": ["text_processor"]}

        result = low.apply_preprocess_pipeline(features, preprocess_config)

        # All original features should be present
        assert result["batch_size"] == 32
        assert result["image_size"] == 224
        assert result["other_param"] == "value"
        # Plus processed output
        assert "processed_text" in result

    def test_apply_preprocess_pipeline_chain_multiple_preprocessors(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Multiple preprocessors should be applied in order."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Create chain: normalize -> tokenize -> encode
        # Each takes output of previous
        prep1 = MockPreprocessor(
            output_key="normalized",
            remove_origin=True,
            output_value_fn=lambda x: x.lower() if x else x,
        )
        prep2 = MockPreprocessor(
            output_key="tokenized",
            remove_origin=True,
            output_value_fn=lambda x: x.split() if x else x,
        )
        prep3 = MockPreprocessor(
            output_key="encoded",
            remove_origin=True,
            output_value_fn=lambda x: len(x) if isinstance(x, list) else 0,
        )

        low._preprocessors_registry._preprocessors["normalize"] = prep1
        low._preprocessors_registry._preprocessors["tokenize"] = prep2
        low._preprocessors_registry._preprocessors["encode"] = prep3

        features = {"text": "Hello World Test"}
        preprocess_config = {"text": ["normalize", "tokenize", "encode"]}

        result = low.apply_preprocess_pipeline(features, preprocess_config)

        # Final output should be the encoded result
        assert "encoded" in result
        assert result["encoded"] == 3  # "hello world test" -> ["hello", "world", "test"] -> 3

    def test_apply_preprocess_pipeline_remove_origin_flag(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """remove_origin=True should remove original feature."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        mock_prep = MockPreprocessor(output_key="processed", remove_origin=True)
        low._preprocessors_registry._preprocessors["remover"] = mock_prep

        features = {"text": "original", "batch_size": 32}
        preprocess_config = {"text": ["remover"]}

        result = low.apply_preprocess_pipeline(features, preprocess_config)

        assert "text" not in result  # Original removed
        assert "processed" in result  # Output present
        assert result["batch_size"] == 32  # Other features preserved

    def test_apply_preprocess_pipeline_skips_missing_features(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Features in config but not in input should be skipped."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        mock_prep = MockPreprocessor(output_key="processed")
        low._preprocessors_registry._preprocessors["processor"] = mock_prep

        features = {"batch_size": 32}  # No "text" feature
        preprocess_config = {"text": ["processor"]}  # Config expects "text"

        # Should not raise - just skip missing feature
        result = low.apply_preprocess_pipeline(features, preprocess_config)

        assert result["batch_size"] == 32
        assert mock_prep.call_count == 0  # Not called

    def test_apply_preprocess_pipeline_empty_config(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Empty config should return features unchanged."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        features = {"batch_size": 32, "image_size": 224}
        preprocess_config = {}

        result = low.apply_preprocess_pipeline(features, preprocess_config)

        assert result == features

    # -------------------------------------------------------------------------
    # train_predictor_with_pipeline Tests
    # -------------------------------------------------------------------------

    def test_train_predictor_with_pipeline_applies_preprocessing(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Training should apply preprocessing to each sample."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Mock preprocessor that doubles batch_size
        mock_prep = MockPreprocessor(
            output_key="batch_size_doubled",
            remove_origin=False,
            output_value_fn=lambda x: x * 2 if x else x,
        )
        low._preprocessors_registry._preprocessors["doubler"] = mock_prep

        preprocess_config = {"batch_size": ["doubler"]}

        predictor = low.train_predictor_with_pipeline(
            features_list=training_data,
            prediction_type="expect_error",
            preprocess_config=preprocess_config,
        )

        assert predictor is not None
        # Predictor should have new feature from preprocessing
        assert "batch_size_doubled" in predictor.feature_names

    def test_train_predictor_with_pipeline_no_config(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """No preprocess_config should behave like train_predictor."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor_with_pipeline(
            features_list=training_data,
            prediction_type="expect_error",
            preprocess_config=None,
        )

        assert predictor is not None
        assert set(predictor.feature_names) == {"batch_size", "image_size"}

    # -------------------------------------------------------------------------
    # predict_with_pipeline Tests
    # -------------------------------------------------------------------------

    def test_predict_with_pipeline_applies_preprocessing(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Prediction should apply preprocessing to features."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Train without preprocessing
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        # Mock preprocessor that passes through
        mock_prep = MockPreprocessor(output_key="batch_size", remove_origin=True)
        low._preprocessors_registry._preprocessors["passthrough"] = mock_prep

        features = {"raw_batch": 32, "image_size": 224}
        preprocess_config = {"raw_batch": ["passthrough"]}

        # Should preprocess "raw_batch" -> "batch_size" before prediction
        result = low.predict_with_pipeline(
            predictor=predictor,
            features=features,
            preprocess_config=preprocess_config,
        )

        assert "expected_runtime_ms" in result

    def test_predict_with_pipeline_no_config(
        self,
        training_data: list[dict[str, Any]],
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """No preprocess_config should behave like predict_with_predictor."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        result = low.predict_with_pipeline(
            predictor=predictor,
            features=sample_features,
            preprocess_config=None,
        )

        assert "expected_runtime_ms" in result

    # -------------------------------------------------------------------------
    # inference_pipeline Tests (PredictorCore high-level API)
    # -------------------------------------------------------------------------

    def test_inference_pipeline_full_workflow(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """inference_pipeline should load model, preprocess, and predict."""
        from src.api.core import PredictorCore, PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        core = PredictorCore(low_level=low)

        # Train and save model first
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("pipeline_test", platform_info, "expect_error", predictor)

        # Use inference pipeline
        result = core.inference_pipeline(
            model_id="pipeline_test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
            preprocess_config=None,  # No preprocessing
        )

        assert result is not None
        assert result.model_id == "pipeline_test"
        assert "expected_runtime_ms" in result.result

    def test_inference_pipeline_with_preprocessing(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """inference_pipeline should apply preprocessing before prediction."""
        from src.api.core import PredictorCore, PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        core = PredictorCore(low_level=low)

        # Register mock preprocessor
        mock_prep = MockPreprocessor(output_key="batch_size", remove_origin=True)
        low._preprocessors_registry._preprocessors["rename"] = mock_prep

        # Train and save model
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("preprocess_test", platform_info, "expect_error", predictor)

        # Use inference pipeline with preprocessing
        result = core.inference_pipeline(
            model_id="preprocess_test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"input_batch": 32, "image_size": 224},  # input_batch -> batch_size
            preprocess_config={"input_batch": ["rename"]},
        )

        assert result is not None
        assert mock_prep.call_count == 1

    def test_inference_pipeline_model_not_found(
        self,
        platform_info: PlatformInfo,
        temp_storage_dir: Path,
    ) -> None:
        """inference_pipeline should raise ModelNotFoundError for missing model."""
        from src.api.core import PredictorCore, ModelNotFoundError

        core = PredictorCore(storage_dir=str(temp_storage_dir))

        with pytest.raises(ModelNotFoundError):
            core.inference_pipeline(
                model_id="nonexistent",
                platform_info=platform_info,
                prediction_type="expect_error",
                features={"batch_size": 32, "image_size": 224},
            )


# =============================================================================
# V2 Preprocessing Pipeline Tests
# =============================================================================


class TestPreprocessorV2Integration:
    """Tests for V2 preprocessing integration with library API."""

    # -------------------------------------------------------------------------
    # apply_preprocess_pipeline_v2 Tests
    # -------------------------------------------------------------------------

    def test_apply_preprocess_pipeline_v2_with_chain(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Should apply V2 chain to features."""
        from src.api.core import PredictorLowLevel
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        chain = PreprocessorChainV2(name="test_chain").add(
            MultiplyPreprocessor("width", "height", "pixels")
        )

        features = {"width": 100, "height": 200, "extra": 999}
        result = low.apply_preprocess_pipeline_v2(features, chain)

        assert result["width"] == 100
        assert result["height"] == 200
        assert result["pixels"] == 20000
        assert result["extra"] == 999

    def test_apply_preprocess_pipeline_v2_none_chain(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """None chain should return features unchanged."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        features = {"width": 100, "height": 200}
        result = low.apply_preprocess_pipeline_v2(features, None)

        assert result == features

    def test_apply_preprocess_pipeline_v2_complex_chain(
        self,
        temp_storage_dir: Path,
    ) -> None:
        """Should handle complex chain with multiple preprocessors."""
        from src.api.core import PredictorLowLevel
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import (
            MultiplyPreprocessor,
            RemoveFeaturePreprocessor,
            TokenLengthPreprocessor,
        )

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        chain = (
            PreprocessorChainV2(name="complex")
            .add(MultiplyPreprocessor("w", "h", "pixels"))
            .add(RemoveFeaturePreprocessor(["w", "h"]))
            .add(TokenLengthPreprocessor("text", "text_len"))
        )

        features = {"w": 10, "h": 20, "text": "hello world", "keep": 42}
        result = low.apply_preprocess_pipeline_v2(features, chain)

        assert "w" not in result
        assert "h" not in result
        assert result["pixels"] == 200
        assert result["text"] == "hello world"  # Not removed
        assert result["text_len"] == 2
        assert result["keep"] == 42

    # -------------------------------------------------------------------------
    # train_predictor_with_pipeline_v2 Tests
    # -------------------------------------------------------------------------

    def test_train_predictor_with_pipeline_v2(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Should train predictor with V2 chain preprocessing."""
        from src.api.core import PredictorLowLevel
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Create chain that adds batch_size * image_size as new feature
        chain = PreprocessorChainV2(name="train_chain").add(
            MultiplyPreprocessor("batch_size", "image_size", "total_pixels")
        )

        predictor = low.train_predictor_with_pipeline_v2(
            features_list=training_data,
            prediction_type="expect_error",
            chain=chain,
        )

        assert predictor is not None
        # Should have original features plus new one
        assert "total_pixels" in predictor.feature_names
        assert "batch_size" in predictor.feature_names
        assert "image_size" in predictor.feature_names

    def test_train_predictor_with_pipeline_v2_none_chain(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """None chain should behave like train_predictor."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor_with_pipeline_v2(
            features_list=training_data,
            prediction_type="expect_error",
            chain=None,
        )

        assert predictor is not None
        assert set(predictor.feature_names) == {"batch_size", "image_size"}

    # -------------------------------------------------------------------------
    # predict_with_pipeline_v2 Tests
    # -------------------------------------------------------------------------

    def test_predict_with_pipeline_v2(
        self,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """Should preprocess features with V2 chain before prediction."""
        from src.api.core import PredictorLowLevel
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        # Train with preprocessed features
        chain = PreprocessorChainV2(name="pred_chain").add(
            MultiplyPreprocessor("batch_size", "image_size", "total_pixels")
        )

        predictor = low.train_predictor_with_pipeline_v2(
            features_list=training_data,
            prediction_type="expect_error",
            chain=chain,
        )

        # Predict with same chain
        features = {"batch_size": 32, "image_size": 224}
        result = low.predict_with_pipeline_v2(predictor, features, chain)

        assert "expected_runtime_ms" in result

    def test_predict_with_pipeline_v2_none_chain(
        self,
        training_data: list[dict[str, Any]],
        sample_features: dict[str, Any],
        temp_storage_dir: Path,
    ) -> None:
        """None chain should behave like predict_with_predictor."""
        from src.api.core import PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))

        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )

        result = low.predict_with_pipeline_v2(predictor, sample_features, None)

        assert "expected_runtime_ms" in result

    # -------------------------------------------------------------------------
    # inference_pipeline_v2 Tests
    # -------------------------------------------------------------------------

    def test_inference_pipeline_v2_full_workflow(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """inference_pipeline_v2 should use V2 chain for preprocessing."""
        from src.api.core import PredictorCore, PredictorLowLevel
        from src.preprocessor.chain_v2 import PreprocessorChainV2
        from src.preprocessor.preprocessors_v2 import MultiplyPreprocessor

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        core = PredictorCore(low_level=low)

        # Create chain
        chain = PreprocessorChainV2(name="v2_pipeline").add(
            MultiplyPreprocessor("batch_size", "image_size", "total_pixels")
        )

        # Train and save model with chain
        predictor = low.train_predictor_with_pipeline_v2(
            features_list=training_data,
            prediction_type="expect_error",
            chain=chain,
        )
        low.save_model("v2_test", platform_info, "expect_error", predictor)

        # Use V2 inference pipeline
        result = core.inference_pipeline_v2(
            model_id="v2_test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
            chain=chain,
        )

        assert result is not None
        assert result.model_id == "v2_test"
        assert "expected_runtime_ms" in result.result

    def test_inference_pipeline_v2_none_chain(
        self,
        platform_info: PlatformInfo,
        training_data: list[dict[str, Any]],
        temp_storage_dir: Path,
    ) -> None:
        """inference_pipeline_v2 with None chain should work without preprocessing."""
        from src.api.core import PredictorCore, PredictorLowLevel

        low = PredictorLowLevel(storage_dir=str(temp_storage_dir))
        core = PredictorCore(low_level=low)

        # Train without preprocessing
        predictor = low.train_predictor(
            features_list=training_data,
            prediction_type="expect_error",
        )
        low.save_model("no_chain_test", platform_info, "expect_error", predictor)

        result = core.inference_pipeline_v2(
            model_id="no_chain_test",
            platform_info=platform_info,
            prediction_type="expect_error",
            features={"batch_size": 32, "image_size": 224},
            chain=None,
        )

        assert result is not None
        assert "expected_runtime_ms" in result.result
