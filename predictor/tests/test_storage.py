"""
Unit tests for model storage layer with prediction_type support.

Tests the ModelStorage class functionality including:
- Model key generation with prediction_type
- Saving and loading models with different prediction types
- Listing models
- Model existence checks
- Model deletion
"""

import pytest
import tempfile
import shutil
from swarmpilot.predictor.storage.model_storage import ModelStorage


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage(temp_storage_dir):
    """Create a ModelStorage instance with temporary directory."""
    return ModelStorage(storage_dir=temp_storage_dir)


class TestModelKeyGeneration:
    """Tests for generate_model_key method."""

    def test_generate_key_with_prediction_type(self, storage):
        """Should generate key including prediction_type."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0.1',
            'hardware_name': 'nvidia-a100'
        }

        key = storage.generate_model_key(
            model_id='test-model',
            platform_info=platform_info,
            prediction_type='quantile'
        )

        assert key == 'test-model__pytorch-2.0.1__nvidia-a100__quantile'

    def test_generate_key_default_prediction_type(self, storage):
        """Should use default prediction_type when not specified."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0.1',
            'hardware_name': 'nvidia-a100'
        }

        key = storage.generate_model_key(
            model_id='test-model',
            platform_info=platform_info
        )

        # Default is 'expect_error'
        assert key == 'test-model__pytorch-2.0.1__nvidia-a100__expect_error'

    def test_generate_key_different_types_different_keys(self, storage):
        """Should generate different keys for different prediction types."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0.1',
            'hardware_name': 'nvidia-a100'
        }

        key1 = storage.generate_model_key(
            model_id='test-model',
            platform_info=platform_info,
            prediction_type='expect_error'
        )

        key2 = storage.generate_model_key(
            model_id='test-model',
            platform_info=platform_info,
            prediction_type='quantile'
        )

        assert key1 != key2
        assert key1 == 'test-model__pytorch-2.0.1__nvidia-a100__expect_error'
        assert key2 == 'test-model__pytorch-2.0.1__nvidia-a100__quantile'


class TestModelSaveAndLoad:
    """Tests for save_model and load_model methods."""

    def test_save_and_load_model_with_prediction_type(self, storage):
        """Should save and load model with prediction_type in metadata."""
        model_key = 'test__pytorch-2.0__cpu__quantile'
        predictor_state = {
            'model_weights': [1.0, 2.0, 3.0],
            'feature_names': ['f1', 'f2']
        }
        metadata = {
            'model_id': 'test',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'samples_count': 100
        }

        storage.save_model(model_key, predictor_state, metadata)
        loaded = storage.load_model(model_key)

        assert loaded is not None
        assert loaded['predictor_state'] == predictor_state
        assert loaded['metadata']['prediction_type'] == 'quantile'
        assert loaded['metadata']['samples_count'] == 100
        assert 'saved_at' in loaded

    def test_save_multiple_types_same_model(self, storage):
        """Should save multiple prediction types for same model independently."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'cpu'
        }

        # Save expect_error model
        key1 = storage.generate_model_key('model1', platform_info, 'expect_error')
        storage.save_model(
            key1,
            {'weights': [1, 2, 3]},
            {'model_id': 'model1', 'platform_info': platform_info, 'prediction_type': 'expect_error'}
        )

        # Save quantile model
        key2 = storage.generate_model_key('model1', platform_info, 'quantile')
        storage.save_model(
            key2,
            {'weights': [4, 5, 6]},
            {'model_id': 'model1', 'platform_info': platform_info, 'prediction_type': 'quantile'}
        )

        # Load both and verify they're different
        loaded1 = storage.load_model(key1)
        loaded2 = storage.load_model(key2)

        assert loaded1['predictor_state']['weights'] == [1, 2, 3]
        assert loaded2['predictor_state']['weights'] == [4, 5, 6]
        assert loaded1['metadata']['prediction_type'] == 'expect_error'
        assert loaded2['metadata']['prediction_type'] == 'quantile'

    def test_load_nonexistent_model(self, storage):
        """Should return None for nonexistent model."""
        result = storage.load_model('nonexistent-key')
        assert result is None


class TestModelExists:
    """Tests for model_exists method."""

    def test_model_exists_true(self, storage):
        """Should return True for existing model."""
        model_key = 'test__pytorch-2.0__cpu__quantile'
        storage.save_model(
            model_key,
            {'weights': [1, 2, 3]},
            {'model_id': 'test', 'prediction_type': 'quantile'}
        )

        assert storage.model_exists(model_key) is True

    def test_model_exists_false(self, storage):
        """Should return False for nonexistent model."""
        assert storage.model_exists('nonexistent-key') is False

    def test_model_exists_different_types(self, storage):
        """Should correctly check existence for different prediction types."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'cpu'
        }

        # Save only expect_error model
        key1 = storage.generate_model_key('model1', platform_info, 'expect_error')
        storage.save_model(
            key1,
            {'weights': [1, 2, 3]},
            {'model_id': 'model1', 'prediction_type': 'expect_error'}
        )

        # Check existence
        key2 = storage.generate_model_key('model1', platform_info, 'quantile')

        assert storage.model_exists(key1) is True
        assert storage.model_exists(key2) is False


class TestListModels:
    """Tests for list_models method."""

    def test_list_empty_storage(self, storage):
        """Should return empty list when no models saved."""
        models = storage.list_models()
        assert models == []

    def test_list_models_with_prediction_types(self, storage):
        """Should list all models including prediction_type in metadata."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'cpu'
        }

        # Save two models with different types
        key1 = storage.generate_model_key('model1', platform_info, 'expect_error')
        storage.save_model(
            key1,
            {'weights': [1, 2, 3]},
            {
                'model_id': 'model1',
                'platform_info': platform_info,
                'prediction_type': 'expect_error',
                'samples_count': 100
            }
        )

        key2 = storage.generate_model_key('model1', platform_info, 'quantile')
        storage.save_model(
            key2,
            {'weights': [4, 5, 6]},
            {
                'model_id': 'model1',
                'platform_info': platform_info,
                'prediction_type': 'quantile',
                'samples_count': 150
            }
        )

        models = storage.list_models()

        assert len(models) == 2

        # Find each model in the list
        expect_error_model = next(m for m in models if m['prediction_type'] == 'expect_error')
        quantile_model = next(m for m in models if m['prediction_type'] == 'quantile')

        assert expect_error_model['model_id'] == 'model1'
        assert expect_error_model['samples_count'] == 100
        assert quantile_model['model_id'] == 'model1'
        assert quantile_model['samples_count'] == 150


class TestDeleteModel:
    """Tests for delete_model method."""

    def test_delete_existing_model(self, storage):
        """Should delete existing model and return True."""
        model_key = 'test__pytorch-2.0__cpu__quantile'
        storage.save_model(
            model_key,
            {'weights': [1, 2, 3]},
            {'model_id': 'test', 'prediction_type': 'quantile'}
        )

        assert storage.model_exists(model_key) is True
        result = storage.delete_model(model_key)

        assert result is True
        assert storage.model_exists(model_key) is False

    def test_delete_nonexistent_model(self, storage):
        """Should return False when deleting nonexistent model."""
        result = storage.delete_model('nonexistent-key')
        assert result is False

    def test_delete_one_type_keeps_other(self, storage):
        """Should delete only specified prediction type."""
        platform_info = {
            'software_name': 'pytorch',
            'software_version': '2.0',
            'hardware_name': 'cpu'
        }

        # Save both types
        key1 = storage.generate_model_key('model1', platform_info, 'expect_error')
        key2 = storage.generate_model_key('model1', platform_info, 'quantile')

        storage.save_model(key1, {'weights': [1, 2, 3]}, {'prediction_type': 'expect_error'})
        storage.save_model(key2, {'weights': [4, 5, 6]}, {'prediction_type': 'quantile'})

        # Delete expect_error
        storage.delete_model(key1)

        # Verify
        assert storage.model_exists(key1) is False
        assert storage.model_exists(key2) is True


class TestStorageInfo:
    """Tests for get_storage_info method."""

    def test_storage_info(self, storage):
        """Should return storage statistics."""
        # Save some models
        storage.save_model(
            'test1__pytorch-2.0__cpu__quantile',
            {'weights': [1, 2, 3]},
            {'model_id': 'test1', 'prediction_type': 'quantile'}
        )
        storage.save_model(
            'test2__pytorch-2.0__cpu__expect_error',
            {'weights': [4, 5, 6]},
            {'model_id': 'test2', 'prediction_type': 'expect_error'}
        )

        info = storage.get_storage_info()

        assert 'storage_dir' in info
        assert 'model_count' in info
        assert 'total_size_bytes' in info
        assert 'is_accessible' in info

        assert info['model_count'] == 2
        assert info['total_size_bytes'] > 0
        assert info['is_accessible'] is True


class TestModelIds:
    """Tests for get model_ids method from get_storage_info."""

    def test_get_storage_info_includes_model_ids(self, storage):
        """Should return model_ids in storage info."""
        # Save some models
        storage.save_model(
            'model1__pytorch-2.0__cpu__quantile',
            {'weights': [1, 2, 3]},
            {'model_id': 'model1', 'prediction_type': 'quantile'}
        )

        info = storage.get_storage_info()
        assert 'model_count' in info
        assert info['model_count'] == 1

    def test_storage_overwrite_model(self, storage):
        """Should overwrite model when saving with same key."""
        model_key = 'test__pytorch-2.0__cpu__quantile'

        # Save first version
        storage.save_model(
            model_key,
            {'weights': [1, 2, 3]},
            {'model_id': 'test', 'samples_count': 100}
        )

        # Save second version (overwrite)
        storage.save_model(
            model_key,
            {'weights': [10, 20, 30]},
            {'model_id': 'test', 'samples_count': 200}
        )

        # Verify only one file exists
        info = storage.get_storage_info()
        assert info['model_count'] == 1

        # Verify content is updated
        loaded = storage.load_model(model_key)
        assert loaded['predictor_state']['weights'] == [10, 20, 30]
        assert loaded['metadata']['samples_count'] == 200


class TestStorageErrorHandling:
    """Tests for storage error handling paths."""

    def test_list_models_with_corrupted_file(self, temp_storage_dir):
        """Should skip corrupted model files and continue listing."""
        from swarmpilot.predictor.storage.model_storage import ModelStorage
        import os

        storage = ModelStorage(storage_dir=temp_storage_dir)

        # Save a valid model
        storage.save_model(
            'valid_model__pytorch__cpu__quantile',
            {'weights': [1, 2, 3]},
            {'model_id': 'valid', 'prediction_type': 'quantile'}
        )

        # Create a corrupted model file
        corrupted_path = os.path.join(temp_storage_dir, 'corrupted__pytorch__cpu__quantile.joblib')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid joblib file')

        # List should succeed and skip the corrupted file
        models = storage.list_models()

        # Should only have the valid model
        assert len(models) == 1
        assert models[0]['model_id'] == 'valid'

    def test_load_model_with_corrupted_file(self, temp_storage_dir):
        """Should raise exception when loading corrupted model."""
        from swarmpilot.predictor.storage.model_storage import ModelStorage
        import os

        storage = ModelStorage(storage_dir=temp_storage_dir)

        # Create a corrupted model file
        corrupted_path = os.path.join(temp_storage_dir, 'corrupted__pytorch__cpu__quantile.joblib')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid joblib file')

        # Loading should raise exception
        with pytest.raises(Exception):
            storage.load_model('corrupted__pytorch__cpu__quantile')

    def test_save_model_to_readonly_directory(self, temp_storage_dir):
        """Should handle permission errors gracefully."""
        from swarmpilot.predictor.storage.model_storage import ModelStorage
        import os
        import stat

        storage = ModelStorage(storage_dir=temp_storage_dir)

        # Make directory read-only
        os.chmod(temp_storage_dir, stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Try to save - should raise PermissionError
            with pytest.raises(PermissionError):
                storage.save_model(
                    'test_model__pytorch__cpu__quantile',
                    {'weights': [1, 2, 3]},
                    {'model_id': 'test', 'prediction_type': 'quantile'}
                )
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_storage_dir, stat.S_IRWXU)
