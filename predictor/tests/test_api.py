"""
Tests for API endpoints.
"""

import pytest


def generate_training_data(n_samples=20, include_sentence=False):
    """Generate training data for tests.

    Args:
        n_samples: Number of samples to generate
        include_sentence: Whether to include the 'sentence' field (for preprocessor tests)
    """
    data = []
    for i in range(n_samples):
        sample = {
            'batch_size': 16 + i,
            'sequence_length': 128,
            'runtime_ms': 100 + i * 2
        }
        if include_sentence:
            sample['sentence'] = f'Hello, world! {i}'
        data.append(sample)
    return data


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_returns_healthy(self, client):
        """Should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'healthy'


class TestListEndpoint:
    """Test /list endpoint."""

    def test_list_empty_models(self, client):
        """Should return empty list when no models exist."""
        response = client.get("/list")
        assert response.status_code == 200

        data = response.json()
        assert 'models' in data
        assert len(data['models']) == 0

    def test_list_models_after_training(self, client):
        """Should list models after training."""
        # Train a model first
        train_request = {
            'model_id': 'test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20)
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200

        # Now list models
        list_response = client.get("/list")
        assert list_response.status_code == 200

        data = list_response.json()
        assert len(data['models']) == 1

        model = data['models'][0]
        assert model['model_id'] == 'test-model'
        assert model['prediction_type'] == 'expect_error'
        assert model['samples_count'] == 20


class TestTrainEndpoint:
    """Test /train endpoint."""

    def test_train_expect_error_model(self, client):
        """Should successfully train expect_error model."""
        request = {
            'model_id': 'test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20),
            'training_config': {'epochs': 100}
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert data['samples_trained'] == 20
        assert 'model_key' in data
        assert 'test-model' in data['model_key']

    def test_train_quantile_model(self, client):
        """Should successfully train quantile model."""
        request = {
            'model_id': 'quantile-model',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.10',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'quantile',
            'features_list': generate_training_data(25),
            'training_config': {'epochs': 100, 'quantiles': [0.5, 0.9, 0.99]}
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert data['samples_trained'] == 25

    def test_train_with_insufficient_samples(self, client):
        """Should fail with less than 10 samples."""
        request = {
            'model_id': 'insufficient-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(5)
        }

        response = client.post("/train", json=request)
        assert response.status_code == 400

        data = response.json()
        assert 'detail' in data

    def test_train_with_invalid_prediction_type(self, client):
        """Should fail with invalid prediction_type."""
        request = {
            'model_id': 'invalid-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'invalid_type',
            'features_list': generate_training_data(20)
        }

        response = client.post("/train", json=request)
        assert response.status_code == 422  # Validation error from Pydantic

    def test_train_updates_existing_model(self, client):
        """Should update model when training with same model_id and platform."""
        request = {
            'model_id': 'update-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20)
        }

        # Train first time
        response1 = client.post("/train", json=request)
        assert response1.status_code == 200

        # Train again with more samples
        request['features_list'] = generate_training_data(30)
        response2 = client.post("/train", json=request)
        assert response2.status_code == 200

        data = response2.json()
        assert data['samples_trained'] == 30

        # Should still have only one model
        list_response = client.get("/list")
        assert len(list_response.json()['models']) == 1
    
    def test_train_with_preprocessor(self, client):
        """Should successfully train with preprocessor."""
        request = {
            'model_id': 'preprocessor-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20, include_sentence=True),
            'enable_preprocessors': ['semantic'],
            'preprocessor_mappings': {
                'semantic': ['sentence']
            }
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200, response.text

        data = response.json()
        assert data['status'] == 'success'
        assert data['samples_trained'] == 20


class TestPredictEndpoint:
    """Test /predict endpoint."""

    @pytest.fixture
    def trained_model(self, client):
        """Train a model for prediction tests."""
        request = {
            'model_id': 'pred-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20),
            'training_config': {'epochs': 200}
        }
        response = client.post("/train", json=request)
        assert response.status_code == 200
        return request

    def test_predict_with_trained_model(self, client, trained_model):
        """Should make predictions with trained model."""
        request = {
            'model_id': 'pred-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 25,
                'sequence_length': 128
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['model_id'] == 'pred-test-model'
        assert data['prediction_type'] == 'expect_error'
        assert 'result' in data

        result = data['result']
        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert result['expected_runtime_ms'] > 0
        assert result['error_margin_ms'] >= 0

    def test_predict_with_nonexistent_model(self, client):
        """Should return 404 for nonexistent model."""
        request = {
            'model_id': 'nonexistent-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 25,
                'sequence_length': 128
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 404

    def test_predict_with_missing_features(self, client, trained_model):
        """Should fail when features are missing."""
        request = {
            'model_id': 'pred-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 25
                # Missing sequence_length
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 400


class TestExperimentMode:
    """Test experiment mode in /predict endpoint."""

    def test_experiment_mode_with_exp_runtime(self, client):
        """Should use experiment mode when exp_runtime in features."""
        request = {
            'model_id': 'any-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'exp_runtime': 150.0,
                'batch_size': 32
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 200

        data = response.json()
        result = data['result']
        assert result['expected_runtime_ms'] == 150.0
        assert result['error_margin_ms'] == 7.5  # 5% of 150

    def test_experiment_mode_with_exp_platform(self, client):
        """Should use experiment mode when platform is all 'exp'."""
        request = {
            'model_id': 'any-model',
            'platform_info': {
                'software_name': 'exp',
                'software_version': 'exp',
                'hardware_name': 'exp'
            },
            'prediction_type': 'expect_error',
            'features': {
                'exp_runtime': 200.0,
                'batch_size': 32
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 200

        data = response.json()
        result = data['result']
        assert result['expected_runtime_ms'] == 200.0

    def test_experiment_mode_quantile(self, client):
        """Should generate quantile predictions in experiment mode."""
        request = {
            'model_id': 'any-model',
            'platform_info': {
                'software_name': 'exp',
                'software_version': 'exp',
                'hardware_name': 'exp'
            },
            'prediction_type': 'quantile',
            'features': {
                'exp_runtime': 100.0,
                'batch_size': 32
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 200

        data = response.json()
        result = data['result']
        assert 'quantiles' in result

        quantiles = result['quantiles']
        # Use larger tolerance due to random sampling (fixed seed but still has variance)
        assert quantiles['0.5'] == pytest.approx(100.0, rel=0.01)  # 1% tolerance
        assert quantiles['0.9'] == pytest.approx(105.0, rel=0.02)  # 2% tolerance
        assert quantiles['0.95'] == pytest.approx(107.5, rel=0.02)  # 2% tolerance
        assert quantiles['0.99'] == pytest.approx(112.0, rel=0.03)  # 3% tolerance


class TestQuantilePrediction:
    """Test quantile prediction type."""

    @pytest.fixture
    def trained_quantile_model(self, client):
        """Train a quantile model."""
        request = {
            'model_id': 'quantile-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features_list': generate_training_data(30),
            'training_config': {'epochs': 300, 'quantiles': [0.5, 0.9, 0.99]}
        }
        response = client.post("/train", json=request)
        assert response.status_code == 200
        return request

    def test_predict_quantiles(self, client, trained_quantile_model):
        """Should return quantile predictions."""
        request = {
            'model_id': 'quantile-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features': {
                'batch_size': 25,
                'sequence_length': 128
            }
        }

        response = client.post("/predict", json=request)
        assert response.status_code == 200

        data = response.json()
        result = data['result']
        assert 'quantiles' in result

        quantiles = result['quantiles']
        assert '0.5' in quantiles
        assert '0.9' in quantiles
        assert '0.99' in quantiles

    def test_prediction_type_mismatch(self, client, trained_quantile_model):
        """Should return 404 when requesting prediction_type that wasn't trained."""
        request = {
            'model_id': 'quantile-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',  # Model was trained with 'quantile' only
            'features': {
                'batch_size': 25,
                'sequence_length': 128
            }
        }

        response = client.post("/predict", json=request)
        # With prediction_type as part of the model key, this is now a 404 (model not found)
        # rather than 400 (type mismatch) since different types are stored separately
        assert response.status_code == 404
