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
    # Need to update both src.api and src.api.dependencies for compatibility
    from src import api
    from src.api import dependencies
    test_storage = api.ModelStorage(storage_dir=TEST_STORAGE_DIR)
    api.storage = test_storage
    dependencies.storage = test_storage

    # Also reset the model cache for each test
    api.model_cache.clear()

    yield

    # Teardown: Clean up test storage
    if Path(TEST_STORAGE_DIR).exists():
        shutil.rmtree(TEST_STORAGE_DIR)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)
