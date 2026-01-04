"""
Shared pytest fixtures for all tests.
"""

import pytest
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from src.api import app


# Use a test-specific storage directory
TEST_STORAGE_DIR = "test_models"


@pytest.fixture(autouse=True, scope="function")
def setup_and_teardown():
    """Setup and teardown for each test."""
    # Setup: Create test storage directory
    Path(TEST_STORAGE_DIR).mkdir(exist_ok=True)

    # Reconfigure storage to use test directory
    # Need to update module-level refs AND internal API storage for full isolation
    from src import api
    from src.api import dependencies
    from src.storage.model_storage import ModelStorage

    test_storage = ModelStorage(storage_dir=TEST_STORAGE_DIR)

    # Update module-level references
    api.storage = test_storage
    dependencies.storage = test_storage

    # Update internal API storage (critical for proper isolation)
    dependencies.predictor_api._storage = test_storage
    dependencies.predictor_core._low_level._storage = test_storage

    # Also reset the model cache for each test
    api.model_cache.clear()
    dependencies.predictor_api._cache.clear()

    # Reset predictor_core accumulator state for V2 tests
    dependencies.predictor_core._accumulated.clear()
    dependencies.predictor_core._feature_schemas.clear()

    yield

    # Teardown: Clean up test storage
    if Path(TEST_STORAGE_DIR).exists():
        shutil.rmtree(TEST_STORAGE_DIR)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)
