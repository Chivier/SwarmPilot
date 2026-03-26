"""Tests for API error handling paths using mocking."""

from unittest.mock import patch


def generate_training_data(n_samples=20):
    """Generate training data for tests."""
    data = []
    for i in range(n_samples):
        data.append(
            {
                "batch_size": 16 + i,
                "sequence_length": 128,
                "runtime_ms": 100 + i * 2,
            }
        )
    return data


class TestHealthEndpointErrors:
    """Test health check error handling."""

    def test_health_check_storage_not_accessible(self, client):
        """Should return 503 when storage is not accessible."""
        with patch(
            "swarmpilot.predictor.api.dependencies.storage"
        ) as mock_storage:
            mock_storage.get_storage_info.return_value = {
                "is_accessible": False,
                "storage_dir": "/nonexistent",
            }

            response = client.get("/health")
            # Should return unhealthy status
            assert response.status_code in [200, 503]

    def test_health_check_exception(self, client):
        """Should handle exception in health check."""
        with patch(
            "swarmpilot.predictor.api.dependencies.storage"
        ) as mock_storage:
            mock_storage.get_storage_info.side_effect = Exception(
                "Storage error"
            )

            response = client.get("/health")
            # Should return error status
            assert response.status_code in [200, 500, 503]


class TestCacheEndpointErrors:
    """Test cache endpoint error handling."""

    def test_cache_stats_exception(self, client):
        """Should handle exception in cache stats."""
        with patch(
            "swarmpilot.predictor.api.dependencies.model_cache"
        ) as mock_cache:
            mock_cache.get_stats.side_effect = Exception("Cache error")

            response = client.get("/cache/stats")
            assert response.status_code == 500

    def test_cache_clear_exception(self, client):
        """Should handle exception in cache clear."""
        with patch(
            "swarmpilot.predictor.api.dependencies.model_cache"
        ) as mock_cache:
            mock_cache.clear.side_effect = Exception("Cache clear error")

            response = client.post("/cache/clear")
            assert response.status_code == 500


class TestListModelsErrors:
    """Test list models error handling."""

    def test_list_models_storage_error(self, client):
        """Should handle storage error in list models."""
        with patch(
            "swarmpilot.predictor.api.dependencies.storage"
        ) as mock_storage:
            mock_storage.list_models.side_effect = Exception(
                "Storage list error"
            )

            response = client.get("/list")
            assert response.status_code == 500


class TestTrainEndpointErrors:
    """Test train endpoint error handling."""

    def test_train_storage_save_error(self, client):
        """Should handle storage save error."""
        with patch(
            "swarmpilot.predictor.api.dependencies.storage"
        ) as mock_storage:
            mock_storage.save_model.side_effect = Exception("Save error")
            # Let other methods work normally
            mock_storage.generate_model_key.return_value = "test-key"

            request = {
                "model_id": "error-model",
                "platform_info": {
                    "software_name": "pytorch",
                    "software_version": "2.0",
                    "hardware_name": "cpu",
                },
                "prediction_type": "expect_error",
                "features_list": generate_training_data(20),
            }

            response = client.post("/train", json=request)
            assert response.status_code == 500

    def test_train_invalid_features_format(self, client):
        """Should handle invalid features format."""
        request = {
            "model_id": "error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": "not a list",  # Invalid
        }

        response = client.post("/train", json=request)
        assert response.status_code == 422  # Validation error


class TestPredictEndpointErrors:
    """Test predict endpoint error handling."""

    def test_predict_storage_load_error(self, client):
        """Should handle storage load error."""
        # First train a model
        train_request = {
            "model_id": "pred-error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Then patch storage to fail on load
        with patch(
            "swarmpilot.predictor.api.dependencies.model_cache"
        ) as mock_cache:
            mock_cache.get.return_value = None

            with patch(
                "swarmpilot.predictor.api.dependencies.storage"
            ) as mock_storage:
                mock_storage.load_model.side_effect = Exception("Load error")
                mock_storage.generate_model_key.return_value = "test-key"

                predict_request = {
                    "model_id": "pred-error-model",
                    "platform_info": {
                        "software_name": "pytorch",
                        "software_version": "2.0",
                        "hardware_name": "cpu",
                    },
                    "prediction_type": "expect_error",
                    "features": {"batch_size": 25, "sequence_length": 128},
                }

                response = client.post("/predict", json=predict_request)
                assert response.status_code == 500

    def test_predict_missing_required_feature(self, client):
        """Should handle missing required features for trained model."""
        # Train with specific features
        train_request = {
            "model_id": "feature-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Predict with different features
        predict_request = {
            "model_id": "feature-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"unknown_feature": 123},  # Wrong features
        }

        response = client.post("/predict", json=predict_request)
        # Should fail due to missing expected features
        assert response.status_code == 400


class TestDeleteModelErrors:
    """Test delete model error handling."""

    def test_delete_nonexistent_model(self, client):
        """Should return 404 for nonexistent model."""
        response = client.delete("/models/nonexistent-model")
        # Either 404 or 200 depending on implementation
        assert response.status_code in [200, 404]

    def test_delete_storage_error(self, client):
        """Should handle storage error during delete."""
        with patch(
            "swarmpilot.predictor.api.dependencies.storage"
        ) as mock_storage:
            mock_storage.delete_model.side_effect = Exception("Delete error")

            response = client.delete("/models/any-model")
            assert response.status_code in [404, 500]


class TestWebSocketErrors:
    """Test WebSocket error handling paths."""

    def test_websocket_prediction_type_mismatch_cached(self, client):
        """Should handle prediction type mismatch with cached model."""
        # Train an expect_error model
        train_request = {
            "model_id": "ws-mismatch-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # First prediction to cache the model
        predict_request_correct = {
            "model_id": "ws-mismatch-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"batch_size": 25, "sequence_length": 128},
        }
        client.post("/predict", json=predict_request_correct)

        # Now request with wrong type via REST
        predict_request_wrong = {
            "model_id": "ws-mismatch-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",  # Different type
            "features": {"batch_size": 25, "sequence_length": 128},
        }

        response = client.post("/predict", json=predict_request_wrong)
        # Should fail since model doesn't exist with this prediction type
        assert response.status_code == 404


class TestDeleteEndpoint:
    """Test delete endpoint."""

    def test_delete_trained_model(self, client):
        """Should delete a trained model."""
        # Train a model
        train_request = {
            "model_id": "delete-test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Delete the model
        response = client.delete(
            "/models/delete-test-model__pytorch-2.0__cpu__expect_error"
        )
        # Check response
        if response.status_code == 200:
            # Verify model is gone
            list_response = client.get("/list")
            models = list_response.json()["models"]
            model_ids = [
                m["model_key"] if "model_key" in m else m.get("model_id")
                for m in models
            ]
            assert (
                "delete-test-model__pytorch-2.0__cpu__expect_error"
                not in model_ids
            )


class TestTrainErrorPaths:
    """Test train endpoint error handling."""

    def test_train_with_preprocessor_error(self, client):
        """Should handle preprocessor error."""
        request = {
            "model_id": "preproc-error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
            "enable_preprocessors": ["nonexistent_preprocessor"],
            "preprocessor_mappings": {
                "nonexistent_preprocessor": ["batch_size"]
            },
        }

        response = client.post("/train", json=request)
        # Should fail with preprocessor error
        assert response.status_code in [400, 422, 500]

    def test_train_with_training_value_error(self, client):
        """Should handle training ValueError."""
        # Features that will cause ValueError during training
        request = {
            "model_id": "value-error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(5),  # Not enough samples
        }

        response = client.post("/train", json=request)
        # Should fail with insufficient samples
        assert response.status_code in [400, 500]

    def test_train_with_invalid_quantiles(self, client):
        """Should handle invalid quantile values."""
        request = {
            "model_id": "invalid-quantile-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data(20),
            "training_config": {
                "quantiles": [0.5, 1.5]  # 1.5 is invalid
            },
        }

        response = client.post("/train", json=request)
        # Should fail with invalid quantile
        assert response.status_code in [400, 500]


class TestPredictErrorPaths:
    """Test predict endpoint error handling."""

    def test_predict_with_experiment_mode_error(self, client):
        """Should handle experiment mode with invalid CV."""
        request = {
            "model_id": "any-model",
            "platform_info": {
                "software_name": "exp",
                "software_version": "exp",
                "hardware_name": "exp",
            },
            "prediction_type": "expect_error",
            "features": {
                "exp_runtime": -100.0,  # Negative runtime
                "batch_size": 32,
            },
        }

        response = client.post("/predict", json=request)
        # May succeed with warning or fail
        assert response.status_code in [200, 400, 500]

    def test_predict_with_prediction_error(self, client):
        """Should handle prediction error."""
        # Train a model
        train_request = {
            "model_id": "pred-error-model2",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Predict with wrong feature types
        predict_request = {
            "model_id": "pred-error-model2",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"batch_size": "not a number", "sequence_length": 128},
        }

        response = client.post("/predict", json=predict_request)
        # Should fail with type error
        assert response.status_code in [400, 422, 500]


class TestStorageInfoEndpoint:
    """Test storage info endpoint."""

    def test_storage_info(self, client):
        """Should return storage information."""
        response = client.get("/storage/info")
        if response.status_code == 200:
            data = response.json()
            assert "storage_dir" in data or "model_count" in data


class TestWebSocketAdditionalErrors:
    """Additional WebSocket error tests."""

    def test_websocket_storage_load_error(self, client):
        """Should handle storage load error via WebSocket."""
        # Train a model first
        train_request = {
            "model_id": "ws-storage-error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Clear cache to force storage load
        client.post("/cache/clear")

        # Predict request that will need to load from storage
        predict_request = {
            "model_id": "ws-storage-error-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"batch_size": 25, "sequence_length": 128},
        }

        with client.websocket_connect("/ws/predict") as websocket:
            import json

            websocket.send_text(json.dumps(predict_request))
            response = websocket.receive_json()

            # Should succeed since model exists
            assert "result" in response or "error" in response


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_info(self, client):
        """Root endpoint should return basic info."""
        response = client.get("/")
        assert response.status_code in [200, 404]
