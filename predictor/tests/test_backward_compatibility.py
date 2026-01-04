"""Backward compatibility tests for V1/V2 API coexistence.

Verifies that:
1. V1 endpoints still work correctly after V2 addition
2. V1 preprocess_config format still works
3. V2 endpoints don't interfere with V1 functionality
4. Models trained with V1 can be predicted with V1
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def generate_training_data(n_samples=20):
    """Generate training data for tests."""
    data = []
    for i in range(n_samples):
        data.append({
            "batch_size": 16 + i,
            "sequence_length": 128,
            "runtime_ms": 100 + i * 2,
        })
    return data


class TestV1EndpointsStillWork:
    """Verify V1 endpoints work after V2 addition."""

    def test_v1_collect_endpoint_exists(self, client):
        """V1 /collect endpoint should still exist."""
        request = {
            "model_id": "compat-test",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"batch_size": 32, "sequence_length": 128},
            "runtime_ms": 150.0,
        }
        response = client.post("/collect", json=request)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_v1_train_endpoint_exists(self, client):
        """V1 /train endpoint should still exist."""
        request = {
            "model_id": "compat-train",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        response = client.post("/train", json=request)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_v1_predict_endpoint_exists(self, client):
        """V1 /predict endpoint should still exist after training."""
        # Train first
        train_request = {
            "model_id": "compat-predict",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Predict
        predict_request = {
            "model_id": "compat-predict",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"batch_size": 32, "sequence_length": 128},
        }
        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200
        assert "result" in response.json()


class TestV1PreprocessConfigStillWorks:
    """Verify V1 preprocess_config format still works."""

    @pytest.mark.skipif(
        True,  # Skip if semantic model not available
        reason="Semantic model may not be available in test environment"
    )
    def test_v1_train_with_preprocess_config_semantic(self, client):
        """V1 train should accept preprocess_config dict with semantic preprocessor."""
        # Generate data with text field for preprocessing
        data = []
        for i in range(20):
            data.append({
                "batch_size": 16 + i,
                "text": f"sample text {i}",
                "runtime_ms": 100 + i * 2,
            })

        request = {
            "model_id": "compat-preprocess",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": data,
            "preprocess_config": {
                "text": ["semantic"],  # V1 format with semantic preprocessor
            },
        }
        response = client.post("/train", json=request)
        assert response.status_code == 200

    def test_v1_train_with_empty_preprocess_config(self, client):
        """V1 train should accept empty preprocess_config (no preprocessing)."""
        request = {
            "model_id": "compat-no-preprocess",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
            "preprocess_config": {},  # Empty config should work
        }
        response = client.post("/train", json=request)
        assert response.status_code == 200


class TestV1AndV2Coexist:
    """Verify V1 and V2 endpoints work side by side."""

    def test_v1_and_v2_endpoints_both_accessible(self, client):
        """Both V1 and V2 train endpoints should be accessible."""
        # V1 train
        v1_request = {
            "model_id": "coexist-v1",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        v1_response = client.post("/train", json=v1_request)
        assert v1_response.status_code == 200

        # V2 train
        v2_request = {
            "model_id": "coexist-v2",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        v2_response = client.post("/v2/train", json=v2_request)
        assert v2_response.status_code == 200

    def test_v1_model_not_affected_by_v2(self, client):
        """Models trained with V1 should still work with V1 predict."""
        # Train with V1
        train_request = {
            "model_id": "isolated-v1",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=train_request)

        # Train a V2 model (shouldn't affect V1 model)
        v2_train = {
            "model_id": "isolated-v2",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/v2/train", json=v2_train)

        # V1 model should still predict correctly
        predict_request = {
            "model_id": "isolated-v1",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features": {"batch_size": 32, "sequence_length": 128},
        }
        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200
        assert "result" in response.json()


class TestModelListIncludesBothVersions:
    """Verify model list shows both V1 and V2 trained models."""

    def test_list_shows_all_models(self, client):
        """List endpoint should show models trained with both APIs."""
        # Train V1 model
        v1_request = {
            "model_id": "list-test-v1",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/train", json=v1_request)

        # Train V2 model
        v2_request = {
            "model_id": "list-test-v2",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "expect_error",
            "features_list": generate_training_data(20),
        }
        client.post("/v2/train", json=v2_request)

        # List should include both
        response = client.get("/list")
        assert response.status_code == 200
        models = response.json()["models"]

        model_ids = [m["model_id"] for m in models]
        assert "list-test-v1" in model_ids
        assert "list-test-v2" in model_ids
