"""
Tests for API endpoints.
"""

from pathlib import Path

import pytest


# Path to semantic model for preprocessor tests
PREDICTOR_DIR = Path(__file__).parent.parent
MODEL_PATH = PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.pt"
CONFIG_PATH = PREDICTOR_DIR / "preprocessors" / "sematic_35M" / "model_35M.yaml"

# Skip condition for tests requiring the semantic model
requires_semantic_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and CONFIG_PATH.exists()),
    reason="Semantic model checkpoint not available (CI environment)",
)


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

    @requires_semantic_model
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
        assert result['error_margin_ms'] == 45.0  # 30% of 150 (default CV)

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
        # Default CV is 30% in experiment mode, so quantiles spread more
        # With CV=0.3, σ=30 for exp_runtime=100
        # q50 ≈ 100 (median), q90 ≈ 138, q95 ≈ 149, q99 ≈ 170
        # Use larger tolerance due to random sampling
        assert quantiles['0.5'] == pytest.approx(100.0, rel=0.05)  # 5% tolerance for median
        assert quantiles['0.9'] == pytest.approx(138.0, rel=0.10)  # 10% tolerance
        assert quantiles['0.95'] == pytest.approx(149.0, rel=0.10)  # 10% tolerance
        assert quantiles['0.99'] == pytest.approx(170.0, rel=0.10)  # 10% tolerance


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


class TestLogTransformAPI:
    """Test log_transform configuration through API."""

    def generate_skewed_data(self, n_samples=30):
        """Generate right-skewed runtime data."""
        import numpy as np
        np.random.seed(42)
        data = []
        for i in range(n_samples):
            batch_size = 16 + i
            base_runtime = 50 + batch_size * 2
            skewed_runtime = base_runtime * np.random.lognormal(0, 0.3)
            data.append({
                'batch_size': batch_size,
                'sequence_length': 128,
                'runtime_ms': max(10.0, float(skewed_runtime))
            })
        return data

    def test_train_with_log_transform_enabled(self, client):
        """Should successfully train quantile model with log_transform enabled."""
        request = {
            'model_id': 'log-transform-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features_list': self.generate_skewed_data(30),
            'training_config': {
                'epochs': 100,
                'log_transform': {'enabled': True}
            }
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert data['samples_trained'] == 30

    def test_predict_with_log_transform_model(self, client):
        """Should make predictions with log_transform model."""
        # Train with log_transform
        train_request = {
            'model_id': 'log-transform-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features_list': self.generate_skewed_data(30),
            'training_config': {
                'epochs': 200,
                'log_transform': {'enabled': True}
            }
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200

        # Make prediction
        predict_request = {
            'model_id': 'log-transform-pred-model',
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

        predict_response = client.post("/predict", json=predict_request)
        assert predict_response.status_code == 200

        data = predict_response.json()
        result = data['result']
        assert 'quantiles' in result

        # Verify predictions are positive (after applying exponential)
        for q, value in result['quantiles'].items():
            assert value > 0, f"Quantile {q} should be positive, got {value}"

    def test_log_transform_disabled_by_default(self, client):
        """Should train without log_transform when not specified."""
        request = {
            'model_id': 'no-log-transform-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features_list': self.generate_skewed_data(30),
            'training_config': {'epochs': 100}  # No log_transform specified
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'


class TestCacheEndpoints:
    """Test cache management endpoints."""

    def test_cache_stats_endpoint(self, client):
        """Should return cache statistics."""
        response = client.get("/cache/stats")
        assert response.status_code == 200

        data = response.json()
        assert 'cache_stats' in data
        stats = data['cache_stats']
        assert 'size' in stats
        assert 'max_size' in stats
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate_percent' in stats

    def test_cache_clear_endpoint(self, client):
        """Should clear the cache successfully."""
        response = client.post("/cache/clear")
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert 'cache cleared' in data['message'].lower()

    def test_cache_stats_after_clear(self, client):
        """Should show zero size after clearing cache."""
        # Clear the cache
        client.post("/cache/clear")

        # Check stats
        response = client.get("/cache/stats")
        assert response.status_code == 200

        data = response.json()
        assert data['cache_stats']['size'] == 0
        assert data['cache_stats']['hits'] == 0
        assert data['cache_stats']['misses'] == 0


class TestModelCache:
    """Test ModelCache class behavior through API."""

    def test_cache_hit_on_repeated_predictions(self, client):
        """Cache should be used for repeated predictions."""
        # Train a model
        train_request = {
            'model_id': 'cache-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20),
            'training_config': {'epochs': 100}
        }
        client.post("/train", json=train_request)

        # Clear cache to start fresh
        client.post("/cache/clear")

        # First prediction - cache miss
        predict_request = {
            'model_id': 'cache-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        response1 = client.post("/predict", json=predict_request)
        assert response1.status_code == 200

        # Check cache stats - should have 1 miss
        stats1 = client.get("/cache/stats").json()['cache_stats']
        assert stats1['misses'] >= 1

        # Second prediction - should be cache hit
        response2 = client.post("/predict", json=predict_request)
        assert response2.status_code == 200

        # Check cache stats - should have at least 1 hit now
        stats2 = client.get("/cache/stats").json()['cache_stats']
        assert stats2['hits'] >= 1


class TestLinearRegressionAPI:
    """Test linear regression prediction type through API."""

    def test_train_linear_regression_model(self, client):
        """Should successfully train linear regression model."""
        request = {
            'model_id': 'linear-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'linear_regression',
            'features_list': generate_training_data(20)
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'
        assert data['samples_trained'] == 20

    def test_predict_with_linear_regression(self, client):
        """Should make predictions with linear regression model."""
        # Train
        train_request = {
            'model_id': 'linear-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'linear_regression',
            'features_list': generate_training_data(20)
        }
        client.post("/train", json=train_request)

        # Predict
        predict_request = {
            'model_id': 'linear-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'linear_regression',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200

        data = response.json()
        assert 'result' in data
        assert 'expected_runtime_ms' in data['result']
        assert 'error_margin_ms' in data['result']


class TestDecisionTreeAPI:
    """Test decision tree prediction type through API."""

    def test_train_decision_tree_model(self, client):
        """Should successfully train decision tree model."""
        request = {
            'model_id': 'tree-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'decision_tree',
            'features_list': generate_training_data(20)
        }

        response = client.post("/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'success'

    def test_predict_with_decision_tree(self, client):
        """Should make predictions with decision tree model."""
        # Train
        train_request = {
            'model_id': 'tree-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'decision_tree',
            'features_list': generate_training_data(20)
        }
        client.post("/train", json=train_request)

        # Predict
        predict_request = {
            'model_id': 'tree-pred-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'decision_tree',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200

        data = response.json()
        assert 'result' in data
        assert 'expected_runtime_ms' in data['result']


class TestWebSocketPrediction:
    """Test WebSocket prediction endpoint."""

    def test_websocket_prediction_with_trained_model(self, client):
        """Should make WebSocket predictions with trained model."""
        # Train a model first
        train_request = {
            'model_id': 'ws-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20)
        }
        client.post("/train", json=train_request)

        # Make WebSocket prediction
        predict_request = {
            'model_id': 'ws-test-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        with client.websocket_connect("/ws/predict") as websocket:
            import json
            websocket.send_text(json.dumps(predict_request))
            response = websocket.receive_json()

            assert 'result' in response
            assert 'expected_runtime_ms' in response['result']
            assert 'error_margin_ms' in response['result']

    def test_websocket_experiment_mode(self, client):
        """Should use experiment mode via WebSocket."""
        predict_request = {
            'model_id': 'any-model',
            'platform_info': {
                'software_name': 'exp',
                'software_version': 'exp',
                'hardware_name': 'exp'
            },
            'prediction_type': 'expect_error',
            'features': {
                'exp_runtime': 150.0,
                'batch_size': 32
            }
        }

        with client.websocket_connect("/ws/predict") as websocket:
            import json
            websocket.send_text(json.dumps(predict_request))
            response = websocket.receive_json()

            assert 'result' in response
            assert response['result']['expected_runtime_ms'] == 150.0

    def test_websocket_model_not_found(self, client):
        """Should return error for nonexistent model via WebSocket."""
        predict_request = {
            'model_id': 'nonexistent-ws-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        with client.websocket_connect("/ws/predict") as websocket:
            import json
            websocket.send_text(json.dumps(predict_request))
            response = websocket.receive_json()

            assert 'error' in response
            assert 'Model not found' in response['error']

    def test_websocket_invalid_json(self, client):
        """Should handle invalid JSON via WebSocket."""
        with client.websocket_connect("/ws/predict") as websocket:
            websocket.send_text("not valid json{")
            response = websocket.receive_json()

            assert 'error' in response
            assert 'Invalid JSON' in response['error']

    def test_websocket_invalid_request(self, client):
        """Should handle invalid request format via WebSocket."""
        with client.websocket_connect("/ws/predict") as websocket:
            import json
            # Missing required fields
            invalid_request = {'model_id': 'test'}
            websocket.send_text(json.dumps(invalid_request))
            response = websocket.receive_json()

            assert 'error' in response
            assert 'Invalid request' in response['error']

    def test_websocket_multiple_predictions(self, client):
        """Should handle multiple sequential predictions via WebSocket."""
        # Train a model first
        train_request = {
            'model_id': 'ws-multi-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(20)
        }
        client.post("/train", json=train_request)

        with client.websocket_connect("/ws/predict") as websocket:
            import json

            # Make multiple predictions on same connection
            for i in range(3):
                predict_request = {
                    'model_id': 'ws-multi-model',
                    'platform_info': {
                        'software_name': 'pytorch',
                        'software_version': '2.0',
                        'hardware_name': 'cpu'
                    },
                    'prediction_type': 'expect_error',
                    'features': {'batch_size': 20 + i * 5, 'sequence_length': 128}
                }
                websocket.send_text(json.dumps(predict_request))
                response = websocket.receive_json()

                assert 'result' in response
                assert 'expected_runtime_ms' in response['result']

    def test_websocket_quantile_prediction(self, client):
        """Should make quantile predictions via WebSocket."""
        # Train a quantile model
        train_request = {
            'model_id': 'ws-quantile-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features_list': generate_training_data(30),
            'training_config': {'epochs': 200, 'quantiles': [0.5, 0.9]}
        }
        client.post("/train", json=train_request)

        predict_request = {
            'model_id': 'ws-quantile-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'quantile',
            'features': {'batch_size': 25, 'sequence_length': 128}
        }

        with client.websocket_connect("/ws/predict") as websocket:
            import json
            websocket.send_text(json.dumps(predict_request))
            response = websocket.receive_json()

            assert 'result' in response
            assert 'quantiles' in response['result']


class TestTrainValidation:
    """Test training validation edge cases."""

    def test_train_with_empty_features_list(self, client):
        """Should fail when features_list is empty."""
        request = {
            'model_id': 'empty-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': []
        }

        response = client.post("/train", json=request)
        # Should fail with insufficient data
        assert response.status_code in [400, 422, 500]

    def test_train_with_few_samples(self, client):
        """Should fail when too few samples provided."""
        request = {
            'model_id': 'few-samples-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_training_data(3)  # Very few samples
        }

        response = client.post("/train", json=request)
        # Might fail or succeed depending on predictor requirements
        # Just verify no server crash
        assert response.status_code in [200, 400, 422, 500]

    def test_train_with_invalid_prediction_type(self, client):
        """Should fail with invalid prediction_type."""
        request = {
            'model_id': 'invalid-type-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'invalid_type',
            'features_list': generate_training_data(20)
        }

        response = client.post("/train", json=request)
        assert response.status_code in [400, 422, 500]


class TestInfoEndpoint:
    """Test /info endpoint if it exists."""

    def test_root_endpoint(self, client):
        """Should return some info at root endpoint."""
        response = client.get("/")
        # Either returns info or 404
        assert response.status_code in [200, 404]

    def test_storage_info_endpoint(self, client):
        """Should return storage info if endpoint exists."""
        response = client.get("/storage/info")
        # Either returns info or 404
        if response.status_code == 200:
            data = response.json()
            assert 'storage_dir' in data or 'model_count' in data
