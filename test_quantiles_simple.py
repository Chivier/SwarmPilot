#!/usr/bin/env python3
"""
Simple test to demonstrate custom quantiles working end-to-end.
"""

import asyncio
import json
import httpx

async def test_custom_quantiles():
    """Test custom quantiles through scheduler and predictor."""

    async with httpx.AsyncClient() as client:

        # 1. Clean up any existing test instances
        print("Cleaning up...")
        for i in range(2):
            try:
                await client.post("http://localhost:8001/instance/remove",
                                json={"instance_id": f"exp_instance_{i}"})
            except:
                pass

        # 2. Register test instances with experiment mode platform_info
        print("\nRegistering instances...")
        for i in range(2):
            response = await client.post(
                "http://localhost:8001/instance/register",
                json={
                    "instance_id": f"exp_instance_{i}",
                    "model_id": "test_model",
                    "endpoint": f"http://localhost:900{i}",
                    "platform_info": {
                        "software_name": "exp",
                        "software_version": "exp",
                        "hardware_name": "exp"
                    }
                }
            )
            print(f"Instance {i}: {response.status_code}")

        # 3. Set probabilistic strategy with custom quantiles
        custom_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        print(f"\nSetting probabilistic strategy with quantiles: {custom_quantiles}")

        response = await client.post(
            "http://localhost:8001/strategy/set",
            json={
                "strategy_name": "probabilistic",
                "quantiles": custom_quantiles
            }
        )
        print(f"Strategy set: {response.status_code}")

        # 4. Submit a task that will trigger experiment mode
        print("\nSubmitting task with exp_runtime (triggers experiment mode)...")

        response = await client.post(
            "http://localhost:8001/task/submit",
            json={
                "task_id": "quantile_test_001",
                "model_id": "test_model",
                "task_input": {"data": "test"},
                "metadata": {
                    "exp_runtime": 1000.0,  # This triggers experiment mode
                    "feature1": 10
                }
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Task submitted: {result['task']['task_id']}")
            print(f"Status: {result['task']['status']}")

        # 5. Wait for task processing
        await asyncio.sleep(2)

        # 6. Check queue info to see quantiles
        print("\nChecking queue info...")
        for i in range(2):
            response = await client.get(f"http://localhost:8001/queue/exp_instance_{i}")
            if response.status_code == 200:
                queue = response.json()
                print(f"\nInstance {i} queue:")
                print(f"  Quantiles: {queue.get('quantiles', [])}")
                print(f"  Values: {[f'{v:.1f}' for v in queue.get('values', [])]}")

                # Verify custom quantiles are used
                if queue.get('quantiles') == custom_quantiles:
                    print("  ✓ Custom quantiles match!")
                else:
                    print(f"  ✗ Quantiles don't match. Expected: {custom_quantiles}")

        # 7. Directly test predictor with custom quantiles
        print("\n\nDirect predictor test with custom quantiles...")

        response = await client.post(
            "http://localhost:8000/predict",
            json={
                "model_id": "test_model",
                "platform_info": {
                    "software_name": "exp",
                    "software_version": "exp",
                    "hardware_name": "exp"
                },
                "prediction_type": "quantile",
                "features": {
                    "exp_runtime": 500.0
                },
                "quantiles": [0.2, 0.4, 0.6, 0.8]  # Different custom quantiles
            }
        )

        if response.status_code == 200:
            result = response.json()
            returned_quantiles = list(result['result']['quantiles'].keys())
            print(f"Requested quantiles: [0.2, 0.4, 0.6, 0.8]")
            print(f"Returned quantiles: {[float(q) for q in returned_quantiles]}")

            # Check values
            for q, v in result['result']['quantiles'].items():
                print(f"  {q}: {v:.2f}ms")

        # Cleanup
        print("\n\nCleaning up...")
        for i in range(2):
            await client.post("http://localhost:8001/instance/remove",
                            json={"instance_id": f"exp_instance_{i}"})
        print("Done!")

if __name__ == "__main__":
    asyncio.run(test_custom_quantiles())