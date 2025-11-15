"""
Test script for subprocess management functionality

This script tests the subprocess manager by:
1. Starting the sleep_model subprocess
2. Checking health
3. Submitting a task
4. Stopping the subprocess
"""

import asyncio
import sys

from src.subprocess_manager import get_docker_manager
from src.model_registry import get_registry
from src.task_queue import get_task_queue
from src.models import Task


    async def main():
    """Run subprocess management tests"""
    print("=" * 60)
    print("Subprocess Manager Test")
    print("=" * 60)
    print()

    # Get manager instances
    docker_manager = get_docker_manager()
    task_queue = get_task_queue()

    try:
        # Test 1: Load registry
        print("1. Loading model registry...")
        registry = get_registry()
        print(f"   ✓ Loaded {len(registry.models)} models")
        print()

        # Test 2: Start model
        print("2. Starting sleep_model subprocess...")
        model_info = await docker_manager.start_model(
            model_id="sleep_model",
            parameters={}
        )
        print(f"   ✓ Model started: {model_info.model_id}")
        print(f"   ✓ Container: {model_info.container_name}")
        print(f"   ✓ Started at: {model_info.started_at}")
        print()

        # Test 3: Check health
        print("3. Checking model health...")
        is_healthy = await docker_manager.check_model_health()
        print(f"   ✓ Model healthy: {is_healthy}")
        print()

        # Test 4: Submit test task
        print("4. Submitting test task (sleep 2 seconds)...")
        task = Task(
            task_id="test-001",
            model_id="sleep_model",
            task_input={"sleep_time": 2.0}
        )
        position = await task_queue.submit_task(task)
        print(f"   ✓ Task submitted: {task.task_id}")
        print(f"   ✓ Queue position: {position}")
        print()

        # Test 5: Wait for task completion
        print("5. Waiting for task completion...")
        for i in range(10):
            await asyncio.sleep(1)
            task = await task_queue.get_task("test-001")
            print(f"   Task status: {task.status.value}")
            if task.status.value in ["completed", "failed"]:
                break

        if task.status.value == "completed":
            print(f"   ✓ Task completed successfully")
            print(f"   ✓ Result: {task.result}")
        else:
            print(f"   ✗ Task failed: {task.error}")
        print()

        # Test 6: Get queue stats
        print("6. Getting queue statistics...")
        stats = await task_queue.get_queue_stats()
        print(f"   ✓ Total tasks: {stats['total']}")
        print(f"   ✓ Completed: {stats['completed']}")
        print(f"   ✓ Failed: {stats['failed']}")
        print()

        # Test 7: Stop model
        print("7. Stopping model subprocess...")
        stopped_model_id = await docker_manager.stop_model()
        print(f"   ✓ Model stopped: {stopped_model_id}")
        print()

        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

        # Try to clean up
        print("\nAttempting cleanup...")
        try:
            if await docker_manager.is_model_running():
                await docker_manager.stop_model()
                print("✓ Cleanup successful")
        except Exception:
            pass

        sys.exit(1)

    finally:
        # Close HTTP client
        await docker_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
