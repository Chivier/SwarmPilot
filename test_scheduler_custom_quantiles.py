#!/usr/bin/env python3
"""
Test script for custom quantiles integration between scheduler and predictor.
"""

import asyncio
import httpx
import json
from typing import List, Dict, Any

# Service URLs
SCHEDULER_URL = "http://localhost:8001"
PREDICTOR_URL = "http://localhost:8000"


async def test_custom_quantiles_integration():
    """Test the custom quantiles flow from scheduler to predictor."""

    print("=" * 60)
    print("Testing Custom Quantiles Integration")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Step 1: Check services are running
        print("\n1. Checking services health...")
        try:
            predictor_health = await client.get(f"{PREDICTOR_URL}/health")
            print(f"   ✓ Predictor service: {predictor_health.json()['status']}")
        except Exception as e:
            print(f"   ✗ Predictor service not running: {e}")
            print("   Start with: cd predictor && uv run uvicorn src.api:app --port 8000")
            return

        try:
            scheduler_health = await client.get(f"{SCHEDULER_URL}/health")
            print(f"   ✓ Scheduler service: {scheduler_health.json()['status']}")
        except Exception as e:
            print(f"   ✗ Scheduler service not running: {e}")
            print("   Start with: cd scheduler && uv run uvicorn src.api:app --port 8001")
            return

        # Step 2: Clean up any existing test instances first
        print("\n2. Cleaning up existing test instances...")
        for i in range(2):
            instance_id = f"test_instance_{i}"
            try:
                await client.post(
                    f"{SCHEDULER_URL}/instance/remove",
                    json={"instance_id": instance_id}
                )
                print(f"   ✓ Removed existing {instance_id}")
            except:
                pass  # Instance doesn't exist, that's fine

        # Step 3: Register test instances
        print("\n3. Registering test instances...")
        instances = [
            {
                "instance_id": f"test_instance_{i}",
                "model_id": "test_model",
                "endpoint": f"http://localhost:900{i}",
                "platform_info": {
                    "software_name": "exp",
                    "software_version": "exp",
                    "hardware_name": "exp"
                }
            }
            for i in range(2)
        ]

        for instance in instances:
            response = await client.post(
                f"{SCHEDULER_URL}/instance/register",
                json=instance
            )
            if response.status_code == 200:
                print(f"   ✓ Registered {instance['instance_id']}")
            else:
                print(f"   ✗ Failed to register {instance['instance_id']}: {response.text}")

        # Step 4: Test switching to probabilistic strategy with custom quantiles
        print("\n4. Testing probabilistic strategy with custom quantiles...")

        custom_quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

        # Switch to probabilistic strategy with custom quantiles
        switch_request = {
            "strategy_name": "probabilistic",
            "quantiles": custom_quantiles
        }

        response = await client.post(
            f"{SCHEDULER_URL}/strategy/set",
            json=switch_request
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Switched to probabilistic strategy")
            print(f"     Custom quantiles: {custom_quantiles}")
        else:
            print(f"   ✗ Failed to switch strategy: {response.text}")
            return

        # Step 5: Submit a task and observe the quantiles used
        print("\n5. Submitting test task with experiment mode...")

        task_request = {
            "task_id": "test_task_001",
            "model_id": "test_model",
            "task_input": {
                "input_data": "test"
            },
            "metadata": {
                "exp_runtime": 1000.0,  # This triggers experiment mode
                "feature1": 10,
                "feature2": 20
            }
        }

        # Enable debug logging to see the prediction request
        print("\n   Submitting task (check logs for quantiles being passed)...")

        response = await client.post(
            f"{SCHEDULER_URL}/task/submit",
            json=task_request
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Task submitted successfully")
            print(f"     Task ID: {result['task']['task_id']}")
            print(f"     Instance: {result['task']['assigned_instance']}")
            print(f"     Status: {result['task']['status']}")

            # Wait a moment for processing
            await asyncio.sleep(1)

            # Check task status
            status_response = await client.get(
                f"{SCHEDULER_URL}/task/{task_request['task_id']}/status"
            )

            if status_response.status_code == 200:
                status = status_response.json()
                print(f"\n   Task Status:")
                print(f"     Status: {status['status']}")
                if status.get('predicted_time_ms'):
                    print(f"     Predicted time: {status['predicted_time_ms']}ms")
                if status.get('predicted_quantiles'):
                    print(f"     Predicted quantiles: {status['predicted_quantiles']}")
        else:
            print(f"   ✗ Failed to submit task: {response.text}")

        # Step 6: Verify the quantiles in queue info
        print("\n6. Checking queue information...")

        for instance in instances:
            queue_response = await client.get(
                f"{SCHEDULER_URL}/queue/{instance['instance_id']}"
            )

            if queue_response.status_code == 200:
                queue_info = queue_response.json()
                print(f"\n   Queue for {instance['instance_id']}:")
                if 'quantiles' in queue_info:
                    print(f"     Quantiles: {queue_info['quantiles']}")
                    print(f"     Values: {queue_info['values']}")

                    # Verify quantiles match our custom ones
                    if queue_info['quantiles'] == custom_quantiles:
                        print(f"     ✓ Quantiles match custom configuration!")
                    else:
                        print(f"     ✗ Quantiles don't match: expected {custom_quantiles}")

        # Step 7: Test with different quantiles
        print("\n7. Testing with different quantiles...")

        new_quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

        switch_request = {
            "strategy_name": "probabilistic",
            "quantiles": new_quantiles
        }

        response = await client.post(
            f"{SCHEDULER_URL}/strategy/set",
            json=switch_request
        )

        if response.status_code == 200:
            print(f"   ✓ Switched to new quantiles: {new_quantiles}")

            # Submit another task
            task_request['task_id'] = "test_task_002"

            response = await client.post(
                f"{SCHEDULER_URL}/task/submit",
                json=task_request
            )

            if response.status_code == 200:
                print(f"   ✓ Task submitted with new quantiles")

        # Cleanup
        print("\n8. Cleaning up...")
        for instance in instances:
            await client.post(
                f"{SCHEDULER_URL}/instance/remove",
                json={"instance_id": instance['instance_id']}
            )
        print("   ✓ Removed test instances")

        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)


async def test_quantiles_validation():
    """Test that invalid quantiles are properly validated."""

    print("\n" + "=" * 60)
    print("Testing Quantiles Validation")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:

        # Test invalid quantiles
        invalid_cases = [
            ([0.5, 1.5], "quantile > 1"),
            ([-0.1, 0.5], "quantile < 0"),
            ([0, 0.5], "quantile = 0"),
            ([0.5, 1], "quantile = 1"),
        ]

        for quantiles, description in invalid_cases:
            print(f"\n   Testing {description}: {quantiles}")

            # Try submitting directly to predictor with invalid quantiles
            predict_request = {
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
                "quantiles": quantiles
            }

            response = await client.post(
                f"{PREDICTOR_URL}/predict",
                json=predict_request
            )

            if response.status_code == 422:
                print(f"     ✓ Correctly rejected invalid quantiles")
            else:
                print(f"     ✗ Should have rejected but got status {response.status_code}")


async def main():
    """Run all tests."""

    # Test the main integration
    await test_custom_quantiles_integration()

    # Test validation
    await test_quantiles_validation()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())