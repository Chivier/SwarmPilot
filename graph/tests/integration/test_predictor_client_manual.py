"""
Simple test for PredictorClient implementation.
This is a basic verification test, not a comprehensive test suite.
"""

import asyncio

from src.clients.predictor_client import (
    PredictorClient,
    PlatformInfo,
    PredictionResult,
    TrainingResponse,
    ModelInfo,
)


async def test_client_initialization():
    """Test that client can be initialized properly."""
    client = PredictorClient(
        predictor_url="http://localhost:8000",
        timeout=10.0,
        max_retries=3,
        retry_delay=2.0,
    )

    assert client.predictor_url == "http://localhost:8000"
    assert client.timeout == 10.0
    assert client.max_retries == 3
    assert client.retry_delay == 2.0

    # Test URL normalization (trailing slash removal)
    client2 = PredictorClient(predictor_url="http://localhost:8000/")
    assert client2.predictor_url == "http://localhost:8000"

    await client.close()
    await client2.close()

    print("✓ Client initialization test passed")


async def test_platform_info():
    """Test PlatformInfo dataclass."""
    platform = PlatformInfo(
        software_name="pytorch",
        software_version="2.0.1",
        hardware_name="nvidia-a100",
    )

    assert platform.software_name == "pytorch"
    assert platform.software_version == "2.0.1"
    assert platform.hardware_name == "nvidia-a100"

    # Test to_dict conversion
    platform_dict = platform.to_dict()
    assert platform_dict == {
        "software_name": "pytorch",
        "software_version": "2.0.1",
        "hardware_name": "nvidia-a100",
    }

    print("✓ PlatformInfo test passed")


async def test_context_manager():
    """Test that client works as async context manager."""
    async with PredictorClient(predictor_url="http://localhost:8000") as client:
        assert client is not None
        assert client.predictor_url == "http://localhost:8000"

    # Client should be closed after exiting context
    print("✓ Context manager test passed")


async def test_prediction_type_validation():
    """Test that invalid prediction types are rejected."""
    client = PredictorClient()
    platform = PlatformInfo(
        software_name="pytorch",
        software_version="2.0.1",
        hardware_name="nvidia-a100",
    )

    try:
        # This should raise ValueError for invalid prediction_type
        await client.predict(
            model_id="test-model",
            platform_info=platform,
            features={"test": 1},
            prediction_type="invalid_type",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid prediction_type" in str(e)
        print("✓ Prediction type validation test passed")
    finally:
        await client.close()


async def main():
    """Run all tests."""
    print("Running PredictorClient tests...\n")

    try:
        await test_client_initialization()
        await test_platform_info()
        await test_context_manager()
        await test_prediction_type_validation()

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
