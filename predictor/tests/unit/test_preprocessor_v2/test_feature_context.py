"""Unit tests for FeatureContext - TDD tests written first.

These tests define the expected behavior of FeatureContext before implementation.
"""

from __future__ import annotations

import pytest


class TestFeatureContextInit:
    """Tests for FeatureContext initialization."""

    def test_init_stores_features(self) -> None:
        """FeatureContext should store features dictionary."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        features = {"a": 1, "b": 2}
        context = FeatureContext(features=features)

        assert context.features == {"a": 1, "b": 2}

    def test_init_copies_to_original(self) -> None:
        """FeatureContext should copy features to original_features."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        features = {"a": 1, "b": 2}
        context = FeatureContext(features=features)

        assert context.original_features == {"a": 1, "b": 2}
        # Verify it's a copy, not same reference
        context.features["c"] = 3
        assert "c" not in context.original_features

    def test_init_empty_tracking_sets(self) -> None:
        """FeatureContext should initialize empty tracking sets."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"x": 1})

        assert context.removed_features == set()
        assert context.added_features == set()
        assert context.modified_features == set()


class TestFeatureContextGet:
    """Tests for FeatureContext.get() method."""

    def test_get_returns_value(self) -> None:
        """get() should return feature value."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"x": 42})

        assert context.get("x") == 42

    def test_get_returns_default_for_missing(self) -> None:
        """get() should return default for missing feature."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})

        assert context.get("missing", default=-1) == -1

    def test_get_returns_none_for_missing_no_default(self) -> None:
        """get() should return None for missing feature when no default."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})

        assert context.get("missing") is None


class TestFeatureContextSet:
    """Tests for FeatureContext.set() method."""

    def test_set_adds_new_feature(self) -> None:
        """set() should add new feature and track in added_features."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})
        context.set("new_feature", 100)

        assert context.features["new_feature"] == 100
        assert "new_feature" in context.added_features

    def test_set_modifies_existing_feature(self) -> None:
        """set() should modify existing and track in modified_features."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"existing": 1})
        context.set("existing", 2)

        assert context.features["existing"] == 2
        assert "existing" in context.modified_features
        assert "existing" not in context.added_features

    def test_set_does_not_modify_original(self) -> None:
        """set() should not modify original_features."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"x": 1})
        context.set("x", 999)

        assert context.original_features["x"] == 1


class TestFeatureContextRemove:
    """Tests for FeatureContext.remove() method."""

    def test_remove_removes_feature(self) -> None:
        """remove() should remove feature and return its value."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"to_remove": 42})

        value = context.remove("to_remove")

        assert value == 42
        assert "to_remove" not in context.features
        assert "to_remove" in context.removed_features

    def test_remove_returns_none_for_missing(self) -> None:
        """remove() should return None for missing feature."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})

        value = context.remove("nonexistent")

        assert value is None

    def test_remove_does_not_add_missing_to_removed(self) -> None:
        """remove() should not track non-existent feature as removed."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})
        context.remove("nonexistent")

        assert "nonexistent" not in context.removed_features


class TestFeatureContextHas:
    """Tests for FeatureContext.has() method."""

    def test_has_returns_true_for_existing(self) -> None:
        """has() should return True for existing feature."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"exists": 1})

        assert context.has("exists") is True

    def test_has_returns_false_for_missing(self) -> None:
        """has() should return False for missing feature."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={})

        assert context.has("missing") is False

    def test_has_returns_false_after_remove(self) -> None:
        """has() should return False after feature is removed."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"temp": 1})
        context.remove("temp")

        assert context.has("temp") is False


class TestFeatureContextKeys:
    """Tests for FeatureContext.keys() method."""

    def test_keys_returns_current_feature_names(self) -> None:
        """keys() should return current feature names."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"a": 1, "b": 2, "c": 3})

        assert set(context.keys()) == {"a", "b", "c"}

    def test_keys_reflects_changes(self) -> None:
        """keys() should reflect added and removed features."""
        from src.preprocessor.base_preprocessor_v2 import FeatureContext

        context = FeatureContext(features={"a": 1, "b": 2})
        context.remove("a")
        context.set("c", 3)

        assert set(context.keys()) == {"b", "c"}
