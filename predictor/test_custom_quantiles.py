#!/usr/bin/env python3
"""
Test script for custom quantiles functionality.
Tests both REST API and WebSocket endpoints.
"""

import json
import asyncio
import aiohttp
import requests

# Base URL for the predictor service
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/predict"


def test_rest_api_experiment_mode_with_custom_quantiles():
    """Test REST API /predict endpoint with custom quantiles in experiment mode."""
    print("Testing REST API with custom quantiles in experiment mode...")

    # Test request with custom quantiles and exp_runtime (experiment mode)
    custom_quantiles = [0.25, 0.5, 0.75, 0.85, 0.95]
    request_data = {
        "model_id": "test_model",
        "platform_info": {
            "software_name": "exp",
            "software_version": "exp",
            "hardware_name": "exp"
        },
        "prediction_type": "quantile",
        "features": {
            "exp_runtime": 1000.0,
            "feature1": 10,
            "feature2": 20
        },
        "quantiles": custom_quantiles
    }

    response = requests.post(f"{BASE_URL}/predict", json=request_data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Custom quantiles test passed")
        print(f"  Request quantiles: {custom_quantiles}")
        print(f"  Response: {json.dumps(result, indent=2)}")

        # Check if response contains the custom quantiles
        result_quantiles = result['result']['quantiles']
        for q in custom_quantiles:
            if str(q) not in result_quantiles:
                print(f"✗ Missing quantile {q} in response")
                return False
        print(f"  ✓ All custom quantiles present in response")
        return True
    else:
        print(f"✗ Request failed with status {response.status_code}: {response.text}")
        return False


def test_rest_api_normal_mode_with_custom_quantiles():
    """Test REST API /predict endpoint with custom quantiles in normal mode (should be ignored)."""
    print("\nTesting REST API with custom quantiles in normal mode (should be ignored)...")

    # First train a model
    train_request = {
        "model_id": "test_model_normal",
        "platform_info": {
            "software_name": "python",
            "software_version": "3.9",
            "hardware_name": "cpu"
        },
        "prediction_type": "quantile",
        "features_list": [
            {"feature1": i, "feature2": i*2, "runtime_ms": 100 + i*10}
            for i in range(20)
        ]
    }

    train_response = requests.post(f"{BASE_URL}/train", json=train_request)
    if train_response.status_code != 200:
        print(f"✗ Training failed: {train_response.text}")
        return False
    print("  ✓ Model trained successfully")

    # Now test prediction with custom quantiles (should be ignored in normal mode)
    custom_quantiles = [0.25, 0.75]
    predict_request = {
        "model_id": "test_model_normal",
        "platform_info": {
            "software_name": "python",
            "software_version": "3.9",
            "hardware_name": "cpu"
        },
        "prediction_type": "quantile",
        "features": {
            "feature1": 5,
            "feature2": 10
        },
        "quantiles": custom_quantiles
    }

    predict_response = requests.post(f"{BASE_URL}/predict", json=predict_request)
    if predict_response.status_code == 200:
        result = predict_response.json()
        print(f"  ✓ Prediction succeeded (custom quantiles were ignored as expected)")
        print(f"  Response quantiles: {list(result['result']['quantiles'].keys())}")
        # Check that it returns the model's default quantiles, not the custom ones
        result_quantiles = list(result['result']['quantiles'].keys())
        if '0.25' in result_quantiles and '0.75' in result_quantiles and len(result_quantiles) != 2:
            print(f"  ✓ Used model's default quantiles, not custom ones")
        return True
    else:
        print(f"✗ Prediction failed: {predict_response.text}")
        return False


async def test_websocket_with_custom_quantiles():
    """Test WebSocket endpoint with custom quantiles."""
    print("\nTesting WebSocket with custom quantiles in experiment mode...")

    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect(WS_URL) as ws:
            # Test with custom quantiles in experiment mode
            custom_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
            request_data = {
                "model_id": "test_ws_model",
                "platform_info": {
                    "software_name": "exp",
                    "software_version": "exp",
                    "hardware_name": "exp"
                },
                "prediction_type": "quantile",
                "features": {
                    "exp_runtime": 500.0,
                    "feature1": 15,
                    "feature2": 30
                },
                "quantiles": custom_quantiles
            }

            await ws.send_str(json.dumps(request_data))

            response = await ws.receive_str()
            result = json.loads(response)

            if 'error' not in result:
                print(f"✓ WebSocket custom quantiles test passed")
                print(f"  Request quantiles: {custom_quantiles}")
                print(f"  Response: {json.dumps(result, indent=2)}")

                # Check if response contains the custom quantiles
                result_quantiles = result['result']['quantiles']
                for q in custom_quantiles:
                    if str(q) not in result_quantiles:
                        print(f"✗ Missing quantile {q} in response")
                        return False
                print(f"  ✓ All custom quantiles present in response")
                return True
            else:
                print(f"✗ WebSocket request failed: {result}")
                return False

    finally:
        await session.close()


def test_invalid_quantiles():
    """Test validation of invalid quantile values."""
    print("\nTesting invalid quantile values...")

    # Test with quantile values outside (0, 1)
    invalid_requests = [
        {
            "quantiles": [0.5, 1.5],  # 1.5 is invalid
            "error": "quantile > 1"
        },
        {
            "quantiles": [-0.1, 0.5],  # -0.1 is invalid
            "error": "quantile < 0"
        },
        {
            "quantiles": [0, 0.5],  # 0 is invalid
            "error": "quantile = 0"
        },
        {
            "quantiles": [0.5, 1],  # 1 is invalid
            "error": "quantile = 1"
        }
    ]

    for test_case in invalid_requests:
        request_data = {
            "model_id": "test_model",
            "platform_info": {
                "software_name": "exp",
                "software_version": "exp",
                "hardware_name": "exp"
            },
            "prediction_type": "quantile",
            "features": {
                "exp_runtime": 1000.0
            },
            "quantiles": test_case["quantiles"]
        }

        response = requests.post(f"{BASE_URL}/predict", json=request_data)

        if response.status_code == 422:  # Validation error
            print(f"  ✓ Correctly rejected {test_case['error']}: {test_case['quantiles']}")
        else:
            print(f"  ✗ Should have rejected {test_case['error']}: {test_case['quantiles']}")
            print(f"    Response: {response.status_code} - {response.text}")

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Custom Quantiles Functionality")
    print("=" * 60)

    # Check if service is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code != 200:
            print("✗ Service health check failed. Is the predictor service running?")
            print("  Start it with: cd predictor && uv run uvicorn src.api:app --reload")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to predictor service at", BASE_URL)
        print("  Start it with: cd predictor && uv run uvicorn src.api:app --reload")
        return

    print("✓ Service is running\n")

    # Run tests
    test_results = []

    # Test REST API
    test_results.append(test_rest_api_experiment_mode_with_custom_quantiles())
    test_results.append(test_rest_api_normal_mode_with_custom_quantiles())

    # Test WebSocket
    test_results.append(await test_websocket_with_custom_quantiles())

    # Test validation
    test_results.append(test_invalid_quantiles())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r)

    if passed_tests == total_tests:
        print(f"✓ All {total_tests} tests passed!")
    else:
        print(f"✗ {passed_tests}/{total_tests} tests passed")

    return passed_tests == total_tests


if __name__ == "__main__":
    asyncio.run(main())