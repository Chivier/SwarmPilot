"""Integration tests for custom quantiles functionality."""

import pytest


def test_custom_quantiles_experiment_mode(client):
    """Test custom quantiles are used in experiment mode."""
    custom_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 1000.0, "feature1": 10, "feature2": 20},
        "quantiles": custom_quantiles,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "result" in result
    assert "quantiles" in result["result"]

    # Check all custom quantiles are present in response
    result_quantiles = result["result"]["quantiles"]
    for q in custom_quantiles:
        assert str(q) in result_quantiles, f"Quantile {q} not found in response"

    # Verify only the requested quantiles are returned
    assert len(result_quantiles) == len(custom_quantiles)


def test_custom_quantiles_ignored_in_normal_mode(client):
    """Test custom quantiles are ignored in normal mode."""
    # First train a model with default quantiles
    train_request = {
        "model_id": "test_normal_mode",
        "platform_info": {
            "software_name": "python",
            "software_version": "3.9",
            "hardware_name": "cpu",
        },
        "prediction_type": "quantile",
        "features_list": [
            {"feature1": i, "feature2": i * 2, "runtime_ms": 100 + i * 5}
            for i in range(50)
        ],
    }

    train_response = client.post("/train", json=train_request)
    assert train_response.status_code == 200

    # Now predict with custom quantiles (should be ignored)
    custom_quantiles = [0.3, 0.7]
    predict_request = {
        "model_id": "test_normal_mode",
        "platform_info": {
            "software_name": "python",
            "software_version": "3.9",
            "hardware_name": "cpu",
        },
        "prediction_type": "quantile",
        "features": {"feature1": 25, "feature2": 50},
        "quantiles": custom_quantiles,  # These should be ignored
    }

    response = client.post("/predict", json=predict_request)
    assert response.status_code == 200

    result = response.json()
    result_quantiles = result["result"]["quantiles"]

    # Check that default quantiles are returned, not custom ones
    default_quantiles = ["0.5", "0.9", "0.95", "0.99"]
    assert set(result_quantiles.keys()) == set(default_quantiles)
    assert "0.3" not in result_quantiles
    assert "0.7" not in result_quantiles


def test_invalid_quantile_values(client):
    """Test validation of invalid quantile values."""
    invalid_test_cases = [
        ([0.5, 1.5], "above 1"),
        ([-0.1, 0.5], "below 0"),
        ([0, 0.5], "equal to 0"),
        ([0.5, 1], "equal to 1"),
        ([0.5, "invalid"], "non-numeric"),
    ]

    for quantiles, description in invalid_test_cases:
        request_data = {
            "model_id": "test_model",
            "platform_info": {
                "software_name": "exp",
                "software_version": "exp",
                "hardware_name": "exp",
            },
            "prediction_type": "quantile",
            "features": {"exp_runtime": 1000.0},
            "quantiles": quantiles,
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422, (
            f"Should reject quantiles {description}: {quantiles}"
        )


def test_empty_quantiles_list(client):
    """Test behavior with empty quantiles list."""
    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 1000.0},
        "quantiles": [],  # Empty list
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    result = response.json()
    # Should return empty quantiles dict
    assert result["result"]["quantiles"] == {}


def test_no_quantiles_field(client):
    """Test behavior when quantiles field is not provided (should use defaults)."""
    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 1000.0},
        # No quantiles field
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    result = response.json()
    result_quantiles = result["result"]["quantiles"]

    # Should use default quantiles
    default_quantiles = ["0.5", "0.9", "0.95", "0.99"]
    assert set(result_quantiles.keys()) == set(default_quantiles)


def test_many_quantiles(client):
    """Test with many quantiles to ensure performance is acceptable."""
    # Generate 20 quantiles
    many_quantiles = [
        i / 100 for i in range(5, 100, 5)
    ]  # 0.05, 0.10, ..., 0.95

    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 500.0},
        "quantiles": many_quantiles,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    result = response.json()
    result_quantiles = result["result"]["quantiles"]

    # All quantiles should be present
    assert len(result_quantiles) == len(many_quantiles)
    for q in many_quantiles:
        assert str(q) in result_quantiles


def test_quantiles_ordering(client):
    """Test that quantile values are properly ordered (monotonic increasing)."""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 1000.0},
        "quantiles": quantiles,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200

    result = response.json()
    result_quantiles = result["result"]["quantiles"]

    # Extract values in order of quantiles
    values = [result_quantiles[str(q)] for q in quantiles]

    # Check that values are monotonically increasing (allowing small tolerance for numerical issues)
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1] * 0.99, (
            f"Quantile values should be monotonic: {values}"
        )


@pytest.mark.asyncio
async def test_websocket_custom_quantiles(client):
    """Test custom quantiles through WebSocket endpoint."""
    import json

    custom_quantiles = [0.2, 0.4, 0.6, 0.8]

    request_data = {
        "model_id": "test_ws_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp",
        },
        "prediction_type": "quantile",
        "features": {"exp_runtime": 750.0, "feature1": 10},
        "quantiles": custom_quantiles,
    }

    with client.websocket_connect("/ws/predict") as websocket:
        websocket.send_text(json.dumps(request_data))
        data = websocket.receive_text()
        result = json.loads(data)

        assert "error" not in result
        assert "result" in result
        assert "quantiles" in result["result"]

        result_quantiles = result["result"]["quantiles"]
        for q in custom_quantiles:
            assert str(q) in result_quantiles

        assert len(result_quantiles) == len(custom_quantiles)
