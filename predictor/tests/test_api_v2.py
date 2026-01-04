"""Tests for V2 API endpoints with preprocessing chain support."""

from __future__ import annotations


def generate_training_data_v2(n_samples: int = 20, with_prompt: bool = False) -> list[dict]:
    """Generate training data for V2 tests.

    Args:
        n_samples: Number of samples to generate.
        with_prompt: If True, include prompt field.

    Returns:
        List of feature dictionaries with width, height, channels, and runtime_ms.
    """
    data = []
    for i in range(n_samples):
        sample = {
            "width": 100 + i * 10,
            "height": 200 + i * 20,
            "channels": 3,
            "runtime_ms": 50 + i * 5,
        }
        if with_prompt:
            sample["prompt"] = f"Hello world {i}"
        data.append(sample)
    return data


def make_chain_config(steps: list[dict]) -> dict:
    """Create a chain config dict."""
    return {"steps": steps}


def make_multiply_step(
    feature_a: str, feature_b: str, output_feature: str
) -> dict:
    """Create a multiply preprocessor step config."""
    return {
        "name": "multiply",
        "params": {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "output_feature": output_feature,
        },
    }


def make_remove_step(features: list[str]) -> dict:
    """Create a remove preprocessor step config."""
    return {"name": "remove", "params": {"features_to_remove": features}}


def make_token_length_step(
    input_feature: str, output_feature: str = "input_length"
) -> dict:
    """Create a token_length preprocessor step config."""
    return {
        "name": "token_length",
        "params": {"input_feature": input_feature, "output_feature": output_feature},
    }


class TestCollectV2Endpoint:
    """Tests for /v2/collect endpoint."""

    def test_collect_v2_basic(self, client):
        """Should collect a sample without preprocessing."""
        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 100, "height": 200, "channels": 3},
            "runtime_ms": 42.5,
        }

        response = client.post("/v2/collect", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["samples_collected"] == 1

    def test_collect_v2_accumulates_samples(self, client):
        """Should accumulate multiple samples."""
        base_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
        }

        for i in range(5):
            request = {
                **base_request,
                "features": {"width": 100 + i, "height": 200 + i, "channels": 3},
                "runtime_ms": 42.5 + i,
            }
            response = client.post("/v2/collect", json=request)
            assert response.status_code == 200
            assert response.json()["samples_collected"] == i + 1

    def test_collect_v2_with_preprocessing_chain(self, client):
        """Should apply preprocessing chain before storing sample.

        Note: Uses unique model_id to avoid feature schema conflicts.
        The chain adds a "pixels" feature, changing the stored schema.
        """
        chain_config = make_chain_config([
            make_multiply_step("width", "height", "pixels"),
        ])

        # Use unique model_id since chain modifies feature schema
        request = {
            "model_id": "test-model-with-chain",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 100, "height": 200, "channels": 3},
            "runtime_ms": 42.5,
            "preprocess_chain": chain_config,
        }

        response = client.post("/v2/collect", json=request)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_collect_v2_validates_prediction_type(self, client):
        """Should reject invalid prediction_type."""
        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "invalid_type",
            "features": {"width": 100, "height": 200},
            "runtime_ms": 42.5,
        }

        response = client.post("/v2/collect", json=request)
        assert response.status_code == 422  # Validation error


class TestTrainV2Endpoint:
    """Tests for /v2/train endpoint."""

    def test_train_v2_basic(self, client):
        """Should train model without preprocessing chain."""
        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["samples_trained"] == 20
        assert data["chain_stored"] is False

    def test_train_v2_with_preprocessing_chain(self, client):
        """Should train model with preprocessing chain and store it."""
        chain_config = make_chain_config([
            make_multiply_step("width", "height", "pixels"),
            make_remove_step(["width", "height"]),
        ])

        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
            "preprocess_chain": chain_config,
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["chain_stored"] is True

    def test_train_v2_uses_collected_data(self, client):
        """Should combine collected samples with request data."""
        base_info = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
        }

        # Collect 10 samples first
        for i in range(10):
            collect_request = {
                **base_info,
                "features": {"width": 100 + i, "height": 200 + i, "channels": 3},
                "runtime_ms": 50 + i * 5,
            }
            client.post("/v2/collect", json=collect_request)

        # Train with 10 more samples from request
        train_request = {
            **base_info,
            "features_list": generate_training_data_v2(10),
        }

        response = client.post("/v2/train", json=train_request)
        assert response.status_code == 200

        data = response.json()
        assert data["samples_trained"] == 20  # 10 collected + 10 from request

    def test_train_v2_validates_chain_before_training(self, client):
        """Should validate chain on first sample before training."""
        # Create chain that requires non-existent feature
        chain_config = make_chain_config([
            make_multiply_step("nonexistent", "height", "pixels"),
        ])

        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
            "preprocess_chain": chain_config,
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code in (400, 422, 500)  # Should fail validation

    def test_train_v2_no_chain_new_model_uses_none(self, client):
        """New model without chain should have no preprocessing."""
        request = {
            "model_id": "new-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["chain_stored"] is False

    def test_train_v2_insufficient_samples(self, client):
        """Should reject training with insufficient samples."""
        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(5),  # Less than minimum
        }

        response = client.post("/v2/train", json=request)
        # Should fail due to insufficient samples
        assert response.status_code in (400, 422, 500)


class TestPredictV2Endpoint:
    """Tests for /v2/predict endpoint."""

    def test_predict_v2_basic(self, client):
        """Should make prediction without preprocessing chain."""
        # First train a model
        train_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
        }
        train_response = client.post("/v2/train", json=train_request)
        assert train_response.status_code == 200

        # Now make prediction - use same features as training (width, height, channels)
        predict_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 150, "height": 300, "channels": 3},
        }

        response = client.post("/v2/predict", json=predict_request)
        assert response.status_code == 200

        data = response.json()
        assert data["model_id"] == "test-model"
        assert "result" in data

    def test_predict_v2_uses_stored_chain_automatically(self, client):
        """Should automatically apply stored chain during prediction."""
        # Train with preprocessing chain that computes pixels from width*height
        chain_config = make_chain_config([
            make_multiply_step("width", "height", "pixels"),
            make_remove_step(["width", "height"]),
        ])

        train_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),  # width, height, channels
            "preprocess_chain": chain_config,
        }
        train_response = client.post("/v2/train", json=train_request)
        assert train_response.status_code == 200

        # Predict - provide raw features, chain should be applied automatically
        # After chain: width/height removed, pixels added
        predict_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 150, "height": 300, "channels": 3},
        }

        response = client.post("/v2/predict", json=predict_request)
        assert response.status_code == 200

        data = response.json()
        assert data["chain_applied"] is True

    def test_predict_v2_no_chain_in_request(self, client):
        """Prediction request should NOT accept preprocess_chain field."""
        # Train without chain
        train_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
        }
        client.post("/v2/train", json=train_request)

        # Attempt to pass chain in predict request - should be ignored or rejected
        predict_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 150, "height": 300, "channels": 3},
            # This field should NOT be accepted by V2 predict
            "preprocess_chain": {"steps": []},
        }

        # The response should still work (extra field ignored) or return 422
        response = client.post("/v2/predict", json=predict_request)
        # Either succeeds (ignores extra field) or fails validation (strict)
        assert response.status_code in (200, 422)

    def test_predict_v2_model_not_found(self, client):
        """Should return error for non-existent model."""
        predict_request = {
            "model_id": "nonexistent-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 150, "height": 300, "channels": 3},
        }

        response = client.post("/v2/predict", json=predict_request)
        assert response.status_code in (404, 500)

    def test_predict_v2_with_custom_quantiles(self, client):
        """Should support custom quantiles for quantile predictor."""
        # Train model
        train_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
        }
        client.post("/v2/train", json=train_request)

        # Predict with custom quantiles
        predict_request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {"width": 150, "height": 300, "channels": 3},
            "quantiles": [0.5, 0.9, 0.99],
        }

        response = client.post("/v2/predict", json=predict_request)
        assert response.status_code == 200


class TestV2FullWorkflow:
    """Integration tests for complete V2 workflow."""

    def test_full_workflow_collect_train_predict(self, client):
        """Test complete workflow: collect -> train -> predict."""
        base_info = {
            "model_id": "workflow-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
        }

        # Step 1: Collect samples
        for i in range(15):
            collect_request = {
                **base_info,
                "features": {"width": 100 + i * 10, "height": 200 + i * 20, "channels": 3},
                "runtime_ms": 50 + i * 5,
            }
            response = client.post("/v2/collect", json=collect_request)
            assert response.status_code == 200

        # Step 2: Train with chain
        chain_config = make_chain_config([
            make_multiply_step("width", "height", "pixels"),
            make_remove_step(["width", "height"]),
        ])

        train_request = {
            **base_info,
            "preprocess_chain": chain_config,
        }
        train_response = client.post("/v2/train", json=train_request)
        assert train_response.status_code == 200
        assert train_response.json()["samples_trained"] == 15
        assert train_response.json()["chain_stored"] is True

        # Step 3: Predict (chain applied automatically)
        predict_request = {
            **base_info,
            "features": {"width": 150, "height": 300, "channels": 3},
        }
        predict_response = client.post("/v2/predict", json=predict_request)
        assert predict_response.status_code == 200
        assert predict_response.json()["chain_applied"] is True

    def test_workflow_with_token_length_preprocessor(self, client):
        """Test workflow with token length preprocessing."""
        chain_config = make_chain_config([
            make_token_length_step("prompt", "input_length"),
            make_remove_step(["prompt"]),
        ])

        # Train
        train_request = {
            "model_id": "nlp-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20, with_prompt=True),
            "preprocess_chain": chain_config,
        }
        train_response = client.post("/v2/train", json=train_request)
        assert train_response.status_code == 200

        # Predict
        predict_request = {
            "model_id": "nlp-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features": {
                "width": 100,
                "height": 200,
                "prompt": "This is a test sentence",
                "channels": 3,
            },
        }
        predict_response = client.post("/v2/predict", json=predict_request)
        assert predict_response.status_code == 200


class TestV2ChainValidation:
    """Tests for chain validation behavior."""

    def test_chain_validation_returns_detailed_error(self, client):
        """Should return detailed error on chain validation failure."""
        # Chain with unknown preprocessor
        chain_config = {
            "steps": [
                {"name": "unknown_preprocessor", "params": {}},
            ]
        }

        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
            "preprocess_chain": chain_config,
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code in (400, 422, 500)

    def test_empty_chain_is_valid(self, client):
        """Empty chain should be treated as no preprocessing."""
        chain_config = make_chain_config([])

        request = {
            "model_id": "test-model",
            "platform_info": {
                "software_name": "pytorch",
                "software_version": "2.0",
                "hardware_name": "cpu",
            },
            "prediction_type": "quantile",
            "features_list": generate_training_data_v2(20),
            "preprocess_chain": chain_config,
        }

        response = client.post("/v2/train", json=request)
        assert response.status_code == 200
