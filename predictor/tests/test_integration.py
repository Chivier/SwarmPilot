"""
End-to-End Integration Tests.

Tests complete workflows from training to prediction,
simulating real-world usage scenarios.
"""

import pytest
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from src.api import app


# Use a test-specific storage directory
TEST_STORAGE_DIR = "test_models_integration"


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for each test."""
    # Setup: Create test storage directory
    Path(TEST_STORAGE_DIR).mkdir(exist_ok=True)

    # Reconfigure app's storage to use test directory
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

    # Clear model cache
    api.model_cache.clear()
    dependencies.predictor_api._cache.clear()

    # Reset predictor_core accumulator state
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


def generate_realistic_training_data(n_samples=30, base_runtime=100, noise_factor=0.1):
    """
    Generate realistic training data with correlation between features and runtime.

    Simulates batch processing where runtime scales with batch size and sequence length.
    """
    import random
    random.seed(42)

    data = []
    for i in range(n_samples):
        batch_size = random.randint(8, 64)
        sequence_length = random.choice([64, 128, 256, 512])
        hidden_size = random.choice([256, 512, 768, 1024])

        # Runtime scales with batch_size, sequence_length, and hidden_size
        # Add some noise to make it realistic
        base = (batch_size * 0.5 +
                sequence_length * 0.1 +
                hidden_size * 0.05)
        noise = random.uniform(-noise_factor, noise_factor) * base
        runtime = base_runtime + base + noise

        data.append({
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'hidden_size': hidden_size,
            'runtime_ms': round(runtime, 2)
        })

    return data


class TestCompleteWorkflow:
    """Test complete end-to-end workflow."""

    def test_expect_error_complete_workflow(self, client):
        """
        Complete workflow: train expect_error model, predict, verify results.

        This simulates a real user workflow:
        1. Service starts (health check)
        2. Train a model with real data
        3. Make predictions
        4. List models to verify persistence
        """
        # Step 1: Verify service is healthy
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()['status'] == 'healthy'

        # Step 2: Initially no models exist
        list_response = client.get("/list")
        assert list_response.status_code == 200
        assert len(list_response.json()['models']) == 0

        # Step 3: Train a model
        training_data = generate_realistic_training_data(n_samples=40)
        train_request = {
            'model_id': 'inference-optimizer-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0.1',
                'hardware_name': 'nvidia-a100'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data,
            'training_config': {
                'epochs': 300,
                'learning_rate': 0.01
            }
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200

        train_data = train_response.json()
        assert train_data['status'] == 'success'
        assert train_data['samples_trained'] == 40
        assert 'inference-optimizer-v1' in train_data['model_key']
        model_key = train_data['model_key']

        # Step 4: Verify model appears in list
        list_response = client.get("/list")
        assert list_response.status_code == 200

        models = list_response.json()['models']
        assert len(models) == 1
        assert models[0]['model_id'] == 'inference-optimizer-v1'
        assert models[0]['prediction_type'] == 'expect_error'
        assert models[0]['samples_count'] == 40
        assert models[0]['platform_info']['software_name'] == 'pytorch'

        # Step 5: Make predictions with the trained model
        predict_request = {
            'model_id': 'inference-optimizer-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0.1',
                'hardware_name': 'nvidia-a100'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }

        predict_response = client.post("/predict", json=predict_request)
        assert predict_response.status_code == 200

        pred_data = predict_response.json()
        assert pred_data['model_id'] == 'inference-optimizer-v1'
        assert pred_data['prediction_type'] == 'expect_error'

        result = pred_data['result']
        assert 'expected_runtime_ms' in result
        assert 'error_margin_ms' in result
        assert result['expected_runtime_ms'] > 0
        assert result['error_margin_ms'] >= 0

        # Step 6: Make multiple predictions to verify consistency
        predictions = []
        for _ in range(3):
            response = client.post("/predict", json=predict_request)
            assert response.status_code == 200
            predictions.append(response.json()['result']['expected_runtime_ms'])

        # All predictions should be identical (deterministic)
        assert len(set(predictions)) == 1

    def test_quantile_complete_workflow(self, client):
        """Complete workflow for quantile prediction type."""
        # Train quantile model
        training_data = generate_realistic_training_data(n_samples=50)
        train_request = {
            'model_id': 'sla-predictor-v1',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.12.0',
                'hardware_name': 'tpu-v4'
            },
            'prediction_type': 'quantile',
            'features_list': training_data,
            'training_config': {
                'epochs': 400,
                'learning_rate': 0.01,
                'quantiles': [0.5, 0.75, 0.9, 0.95, 0.99]
            }
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200
        assert train_response.json()['status'] == 'success'

        # Make prediction
        predict_request = {
            'model_id': 'sla-predictor-v1',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.12.0',
                'hardware_name': 'tpu-v4'
            },
            'prediction_type': 'quantile',
            'features': {
                'batch_size': 16,
                'sequence_length': 128,
                'hidden_size': 512
            }
        }

        predict_response = client.post("/predict", json=predict_request)
        assert predict_response.status_code == 200

        result = predict_response.json()['result']
        assert 'quantiles' in result

        quantiles = result['quantiles']
        assert '0.5' in quantiles
        assert '0.75' in quantiles
        assert '0.9' in quantiles
        assert '0.95' in quantiles
        assert '0.99' in quantiles

        # Verify quantiles are monotonically increasing
        values = [quantiles[str(q)] for q in [0.5, 0.75, 0.9, 0.95, 0.99]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] * 1.01  # Allow 1% tolerance


class TestMultiModelManagement:
    """Test managing multiple models simultaneously."""

    def test_multiple_models_same_prediction_type(self, client):
        """Train and manage multiple models with same prediction type but different platforms."""
        training_data = generate_realistic_training_data(n_samples=30)

        # Train model for platform 1
        train_request_1 = {
            'model_id': 'model-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }
        response_1 = client.post("/train", json=train_request_1)
        assert response_1.status_code == 200

        # Train same model for platform 2
        train_request_2 = {
            'model_id': 'model-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }
        response_2 = client.post("/train", json=train_request_2)
        assert response_2.status_code == 200

        # Train different model
        train_request_3 = {
            'model_id': 'model-v2',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.10',
                'hardware_name': 'tpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }
        response_3 = client.post("/train", json=train_request_3)
        assert response_3.status_code == 200

        # Verify all models are listed
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 3

        # Verify we can predict with each model
        for model_info in models:
            predict_request = {
                'model_id': model_info['model_id'],
                'platform_info': model_info['platform_info'],
                'prediction_type': 'expect_error',
                'features': {
                    'batch_size': 32,
                    'sequence_length': 256,
                    'hidden_size': 768
                }
            }
            predict_response = client.post("/predict", json=predict_request)
            assert predict_response.status_code == 200

    def test_multiple_prediction_types(self, client):
        """Train models with different prediction types."""
        training_data = generate_realistic_training_data(n_samples=30)

        # Train expect_error model
        train_expect = {
            'model_id': 'multi-type-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }
        response_expect = client.post("/train", json=train_expect)
        assert response_expect.status_code == 200

        # Train quantile model (same model_id, different platform or prediction_type creates different model)
        train_quantile = {
            'model_id': 'multi-type-model',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.10',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'quantile',
            'features_list': training_data,
            'training_config': {'quantiles': [0.5, 0.9, 0.99]}
        }
        response_quantile = client.post("/train", json=train_quantile)
        assert response_quantile.status_code == 200

        # List and verify both models
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 2

        prediction_types = [m['prediction_type'] for m in models]
        assert 'expect_error' in prediction_types
        assert 'quantile' in prediction_types


class TestModelUpdateAndRetrain:
    """Test model updating and retraining scenarios."""

    def test_incremental_training(self, client):
        """Test updating a model with new training data."""
        # Initial training with 20 samples
        initial_data = generate_realistic_training_data(n_samples=20)
        train_request = {
            'model_id': 'evolving-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': initial_data
        }

        initial_response = client.post("/train", json=train_request)
        assert initial_response.status_code == 200
        assert initial_response.json()['samples_trained'] == 20

        # Verify model exists with 20 samples
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 1
        assert models[0]['samples_count'] == 20

        # Retrain with more samples
        updated_data = generate_realistic_training_data(n_samples=50)
        train_request['features_list'] = updated_data

        update_response = client.post("/train", json=train_request)
        assert update_response.status_code == 200
        assert update_response.json()['samples_trained'] == 50

        # Verify model updated
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 1  # Still only one model
        assert models[0]['samples_count'] == 50  # Updated sample count

    def test_training_with_different_features(self, client):
        """Test that retraining with different features updates the model."""
        # Train with initial feature set
        initial_data = [
            {'feature_a': i, 'feature_b': i*2, 'runtime_ms': 100 + i}
            for i in range(20)
        ]

        train_request = {
            'model_id': 'feature-change-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': initial_data
        }

        response1 = client.post("/train", json=train_request)
        assert response1.status_code == 200

        # Predict with initial features
        predict_request_1 = {
            'model_id': 'feature-change-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {'feature_a': 10, 'feature_b': 20}
        }

        pred_response_1 = client.post("/predict", json=predict_request_1)
        assert pred_response_1.status_code == 200

        # Retrain with different features
        new_data = [
            {'feature_x': i, 'feature_y': i*3, 'feature_z': i*4, 'runtime_ms': 100 + i}
            for i in range(25)
        ]
        train_request['features_list'] = new_data

        response2 = client.post("/train", json=train_request)
        assert response2.status_code == 200

        # Old features should now fail
        pred_response_fail = client.post("/predict", json=predict_request_1)
        assert pred_response_fail.status_code == 400  # Missing features

        # New features should work
        predict_request_2 = {
            'model_id': 'feature-change-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {'feature_x': 10, 'feature_y': 30, 'feature_z': 40}
        }

        pred_response_2 = client.post("/predict", json=predict_request_2)
        assert pred_response_2.status_code == 200


class TestExperimentModeIntegration:
    """Test experiment mode integration with normal workflow."""

    def test_experiment_mode_without_trained_model(self, client):
        """Experiment mode should work even without any trained models."""
        # Verify no models exist
        list_response = client.get("/list")
        assert len(list_response.json()['models']) == 0

        # Experiment mode prediction should still work
        predict_request = {
            'model_id': 'nonexistent-model',
            'platform_info': {
                'software_name': 'exp',
                'software_version': 'exp',
                'hardware_name': 'exp'
            },
            'prediction_type': 'expect_error',
            'features': {
                'exp_runtime': 250.0,
                'batch_size': 32
            }
        }

        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200

        result = response.json()['result']
        assert result['expected_runtime_ms'] == 250.0
        assert result['error_margin_ms'] == pytest.approx(75.0)  # 30% of 250 (default CV)

    def test_switch_between_normal_and_experiment_mode(self, client):
        """Test switching between normal predictions and experiment mode."""
        # Train a model
        training_data = generate_realistic_training_data(n_samples=30)
        train_request = {
            'model_id': 'switchable-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200

        # Normal prediction
        normal_predict = {
            'model_id': 'switchable-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }

        normal_response = client.post("/predict", json=normal_predict)
        assert normal_response.status_code == 200
        normal_result = normal_response.json()['result']

        # Experiment mode prediction (with exp_runtime in features)
        exp_predict = {
            'model_id': 'switchable-model',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'exp_runtime': 200.0,
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }

        exp_response = client.post("/predict", json=exp_predict)
        assert exp_response.status_code == 200
        exp_result = exp_response.json()['result']

        # Results should be different
        assert exp_result['expected_runtime_ms'] == 200.0
        assert exp_result['expected_runtime_ms'] != normal_result['expected_runtime_ms']


class TestErrorRecovery:
    """Test error scenarios and recovery."""

    def test_invalid_training_then_valid_training(self, client):
        """Test that service recovers from failed training attempts."""
        # Attempt to train with insufficient samples
        invalid_request = {
            'model_id': 'recovery-test',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': generate_realistic_training_data(n_samples=5)  # Too few
        }

        invalid_response = client.post("/train", json=invalid_request)
        assert invalid_response.status_code == 400

        # Verify no model was created
        list_response = client.get("/list")
        assert len(list_response.json()['models']) == 0

        # Now train with valid data
        valid_request = invalid_request.copy()
        valid_request['features_list'] = generate_realistic_training_data(n_samples=30)

        valid_response = client.post("/train", json=valid_request)
        assert valid_response.status_code == 200

        # Verify model exists
        list_response = client.get("/list")
        assert len(list_response.json()['models']) == 1

    def test_prediction_with_wrong_platform_then_correct(self, client):
        """Test error recovery when using wrong platform for prediction."""
        # Train model
        training_data = generate_realistic_training_data(n_samples=30)
        train_request = {
            'model_id': 'platform-test',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }

        train_response = client.post("/train", json=train_request)
        assert train_response.status_code == 200

        # Try prediction with wrong platform
        wrong_predict = {
            'model_id': 'platform-test',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'  # Wrong hardware
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }

        wrong_response = client.post("/predict", json=wrong_predict)
        assert wrong_response.status_code == 404  # Model not found

        # Correct prediction
        correct_predict = wrong_predict.copy()
        correct_predict['platform_info']['hardware_name'] = 'gpu'

        correct_response = client.post("/predict", json=correct_predict)
        assert correct_response.status_code == 200


class TestRealWorldScenarios:
    """Test realistic real-world usage scenarios."""

    def test_model_versioning_workflow(self, client):
        """
        Simulate versioned model deployment:
        - Deploy v1
        - Use v1 for predictions
        - Deploy v2 (improved)
        - Both versions coexist
        - Eventually phase out v1
        """
        training_data_v1 = generate_realistic_training_data(n_samples=30, base_runtime=100)
        training_data_v2 = generate_realistic_training_data(n_samples=50, base_runtime=90)  # Improved

        # Deploy v1
        train_v1 = {
            'model_id': 'production-model-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'a100'
            },
            'prediction_type': 'quantile',
            'features_list': training_data_v1,
            'training_config': {'quantiles': [0.5, 0.9, 0.99]}
        }
        response_v1 = client.post("/train", json=train_v1)
        assert response_v1.status_code == 200

        # Use v1 for predictions
        predict_v1 = {
            'model_id': 'production-model-v1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'a100'
            },
            'prediction_type': 'quantile',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }
        pred_v1_response = client.post("/predict", json=predict_v1)
        assert pred_v1_response.status_code == 200
        v1_prediction = pred_v1_response.json()['result']

        # Deploy v2
        train_v2 = {
            'model_id': 'production-model-v2',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.1',  # Upgraded
                'hardware_name': 'h100'     # New hardware
            },
            'prediction_type': 'quantile',
            'features_list': training_data_v2,
            'training_config': {'quantiles': [0.5, 0.9, 0.99]}
        }
        response_v2 = client.post("/train", json=train_v2)
        assert response_v2.status_code == 200

        # Both versions exist
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 2

        model_ids = [m['model_id'] for m in models]
        assert 'production-model-v1' in model_ids
        assert 'production-model-v2' in model_ids

        # Use v2 for predictions
        predict_v2 = {
            'model_id': 'production-model-v2',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.1',
                'hardware_name': 'h100'
            },
            'prediction_type': 'quantile',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }
        pred_v2_response = client.post("/predict", json=predict_v2)
        assert pred_v2_response.status_code == 200
        v2_prediction = pred_v2_response.json()['result']

        # Both models should work independently
        assert v1_prediction != v2_prediction

    def test_multi_platform_deployment(self, client):
        """
        Simulate deployment across multiple platforms:
        - Same model deployed on different hardware
        - Each platform has its own performance characteristics
        """
        base_data = generate_realistic_training_data(n_samples=40)

        platforms = [
            ('cpu', 'intel-xeon', 1.5),
            ('gpu', 'nvidia-v100', 1.0),
            ('gpu', 'nvidia-a100', 0.8),
            ('tpu', 'google-tpu-v4', 0.6)
        ]

        # Train model for each platform
        for hw_type, hw_name, speed_multiplier in platforms:
            # Adjust runtime based on platform speed
            platform_data = [
                {**sample, 'runtime_ms': sample['runtime_ms'] * speed_multiplier}
                for sample in base_data
            ]

            train_request = {
                'model_id': 'multi-platform-model',
                'platform_info': {
                    'software_name': 'pytorch',
                    'software_version': '2.0',
                    'hardware_name': hw_name
                },
                'prediction_type': 'expect_error',
                'features_list': platform_data
            }

            response = client.post("/train", json=train_request)
            assert response.status_code == 200

        # Verify all platforms are available
        list_response = client.get("/list")
        models = list_response.json()['models']
        assert len(models) == 4

        # Make predictions for each platform
        predictions = {}
        for _, hw_name, _ in platforms:
            predict_request = {
                'model_id': 'multi-platform-model',
                'platform_info': {
                    'software_name': 'pytorch',
                    'software_version': '2.0',
                    'hardware_name': hw_name
                },
                'prediction_type': 'expect_error',
                'features': {
                    'batch_size': 32,
                    'sequence_length': 256,
                    'hidden_size': 768
                }
            }

            response = client.post("/predict", json=predict_request)
            assert response.status_code == 200
            predictions[hw_name] = response.json()['result']['expected_runtime_ms']

        # Verify predictions reflect platform performance
        # Faster hardware should predict shorter runtimes
        assert predictions['intel-xeon'] > predictions['nvidia-v100']
        assert predictions['nvidia-v100'] > predictions['nvidia-a100']
        assert predictions['nvidia-a100'] > predictions['google-tpu-v4']


class TestConcurrency:
    """Test concurrent operations."""

    def test_concurrent_training_and_prediction(self, client):
        """Test that training and prediction can happen concurrently."""
        # Train initial model
        training_data = generate_realistic_training_data(n_samples=30)
        train_request_1 = {
            'model_id': 'concurrent-model-1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }
        client.post("/train", json=train_request_1)

        # Train second model while predicting with first
        train_request_2 = {
            'model_id': 'concurrent-model-2',
            'platform_info': {
                'software_name': 'tensorflow',
                'software_version': '2.10',
                'hardware_name': 'gpu'
            },
            'prediction_type': 'expect_error',
            'features_list': training_data
        }

        predict_request_1 = {
            'model_id': 'concurrent-model-1',
            'platform_info': {
                'software_name': 'pytorch',
                'software_version': '2.0',
                'hardware_name': 'cpu'
            },
            'prediction_type': 'expect_error',
            'features': {
                'batch_size': 32,
                'sequence_length': 256,
                'hidden_size': 768
            }
        }

        # These operations should both succeed
        train_response = client.post("/train", json=train_request_2)
        predict_response = client.post("/predict", json=predict_request_1)

        assert train_response.status_code == 200
        assert predict_response.status_code == 200

        # Both models should exist
        list_response = client.get("/list")
        assert len(list_response.json()['models']) == 2
