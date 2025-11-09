"""
Example: Starting a model with InstanceClient using different scheduler_url formats.

This demonstrates the two ways to provide the scheduler URL to start_model():
1. As a string endpoint
2. As a SchedulerClient instance
"""

import asyncio
from src.clients.instance_client import InstanceClient
from src.clients.scheduler_client import SchedulerClient


async def example_start_model_with_string():
    """Example: Using string endpoint for scheduler_url."""
    print("=" * 60)
    print("Example 1: Using string endpoint")
    print("=" * 60)

    async with InstanceClient(base_url="http://localhost:5000") as instance:
        # Method 1: Full URL with scheme
        response = await instance.start_model(
            model_id="gpt-4-turbo",
            scheduler_url="http://localhost:8100",
            parameters={"max_tokens": 4096, "temperature": 0.7}
        )
        print(f"✓ Model started: {response.model_id}")
        print(f"  Scheduler: {response.scheduler_url}")
        print(f"  Started at: {response.started_at}")

        # Method 2: Endpoint without scheme (auto-adds http://)
        response = await instance.start_model(
            model_id="gpt-4-turbo",
            scheduler_url="localhost:8100",  # No http://
            parameters={"max_tokens": 4096}
        )
        print(f"\n✓ Model started with auto-scheme: {response.model_id}")
        print(f"  Scheduler: {response.scheduler_url}")


async def example_start_model_with_scheduler_client():
    """Example: Using SchedulerClient instance for scheduler_url."""
    print("\n" + "=" * 60)
    print("Example 2: Using SchedulerClient instance")
    print("=" * 60)

    # Create a SchedulerClient instance
    scheduler = SchedulerClient(base_url="http://localhost:8100")

    async with InstanceClient(base_url="http://localhost:5000") as instance:
        # Pass the SchedulerClient instance directly
        # The base_url will be automatically extracted
        response = await instance.start_model(
            model_id="gpt-4-turbo",
            scheduler_url=scheduler,  # Pass SchedulerClient instance
            parameters={"max_tokens": 4096}
        )
        print(f"✓ Model started: {response.model_id}")
        print(f"  Scheduler: {response.scheduler_url}")
        print(f"  URL extracted from SchedulerClient.base_url")


async def example_error_handling():
    """Example: Error handling for invalid scheduler_url types."""
    print("\n" + "=" * 60)
    print("Example 3: Error handling")
    print("=" * 60)

    async with InstanceClient(base_url="http://localhost:5000") as instance:
        try:
            # This will raise ValueError
            await instance.start_model(
                model_id="gpt-4-turbo",
                scheduler_url=12345,  # Invalid type!
                parameters={}
            )
        except ValueError as e:
            print(f"✗ Caught expected error: {e}")
            print("  scheduler_url must be str or SchedulerClient")


async def example_complete_workflow():
    """Example: Complete workflow with scheduler registration."""
    print("\n" + "=" * 60)
    print("Example 4: Complete workflow")
    print("=" * 60)

    # Initialize clients
    scheduler = SchedulerClient(base_url="http://localhost:8100")

    async with InstanceClient(base_url="http://localhost:5000") as instance:
        # Start model with scheduler registration
        print("1. Starting model...")
        response = await instance.start_model(
            model_id="gpt-4-turbo",
            scheduler_url=scheduler,
            parameters={
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        print(f"   ✓ Model {response.model_id} started")
        print(f"   ✓ Registered with scheduler: {response.scheduler_url}")

        # Check instance health
        print("\n2. Checking instance health...")
        health = await instance.health_check()
        print(f"   ✓ Status: {health.status}")
        print(f"   ✓ Model loaded: {health.model_loaded}")
        print(f"   ✓ Model ID: {health.model_id}")

        # Get instance info
        print("\n3. Getting instance info...")
        info = await instance.get_info()
        print(f"   ✓ Instance ID: {info.instance_id}")
        print(f"   ✓ Uptime: {info.uptime_seconds}s")
        print(f"   ✓ Tasks processed: {info.tasks_processed}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("InstanceClient start_model() Examples")
    print("=" * 60)

    try:
        # Example 1: String endpoints
        await example_start_model_with_string()

        # Example 2: SchedulerClient instance
        await example_start_model_with_scheduler_client()

        # Example 3: Error handling
        await example_error_handling()

        # Example 4: Complete workflow
        await example_complete_workflow()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"  Make sure the instance and scheduler services are running:")
        print(f"  - Instance: http://localhost:5000")
        print(f"  - Scheduler: http://localhost:8100")


if __name__ == "__main__":
    # Note: These examples assume running instance and scheduler services
    # For testing without services, mock the HTTP responses
    asyncio.run(main())
