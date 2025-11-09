"""
Complete example of using the Node class.

This demonstrates the expected workflow:
1. Start Predictor (external)
2. Create and start Node (starts internal scheduler)
3. Register Instances
4. Execute tasks
5. Cleanup
"""

import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from node import Node
from clients.instance_client import InstanceClient
from clients.predictor_client import PredictorClient


async def example_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("Example 1: Basic Node Usage")
    print("=" * 70)

    # Create Node (predictor and instances assumed to be running)
    node = Node(
        model_id="gpt-4",
        predictor_url="http://localhost:8001",
        scheduler_port=8100,
    )

    try:
        # Start Node (starts internal scheduler)
        await node.start()

        # Create instance clients
        instance1 = InstanceClient(base_url="http://localhost:5001")
        instance2 = InstanceClient(base_url="http://localhost:5002")

        # Register instances
        await node.register_instance(instance1)
        await node.register_instance(instance2)

        # Execute task
        result = await node.exec({"prompt": "Hello, world!"})
        print(f"\nTask result: {result}")

    finally:
        # Always cleanup
        await node.stop()


async def example_with_context_manager():
    """Example using async context manager."""
    print("\n" + "=" * 70)
    print("Example 2: Using Context Manager")
    print("=" * 70)

    async with Node(
        model_id="gpt-4",
        predictor_url="http://localhost:8001",
        scheduler_port=8200,
    ) as node:
        # Register instances
        instance = InstanceClient(base_url="http://localhost:5001")
        await node.register_instance(instance)

        # Execute multiple tasks
        tasks = [
            {"prompt": "Task 1"},
            {"prompt": "Task 2"},
            {"prompt": "Task 3"},
        ]

        for task_input in tasks:
            result = await node.exec(task_input)
            print(f"Result: {result}")

    # Node automatically stops when exiting context


async def example_complete_workflow():
    """Complete workflow including predictor startup."""
    print("\n" + "=" * 70)
    print("Example 3: Complete Workflow (Mock)")
    print("=" * 70)

    print("""
This example shows the complete workflow:

1. Start Predictor (external service)
   $ predictor start --port 8001

2. Create Node with predictor URL
   node = Node(model_id="gpt-4", predictor_url="http://localhost:8001")

3. Start Node (internal scheduler starts automatically)
   await node.start()

4. Register Instances
   await node.register_instance(instance1)
   await node.register_instance(instance2)

5. Execute tasks
   result = await node.exec({"prompt": "Hello!"})

6. Cleanup
   await node.stop()

Note: Steps 1 and 4 require actual predictor and instance services running.
    """)


async def example_error_handling():
    """Example with error handling."""
    print("\n" + "=" * 70)
    print("Example 4: Error Handling")
    print("=" * 70)

    node = Node(
        model_id="gpt-4",
        predictor_url="http://localhost:8001",
        scheduler_port=8300,
    )

    try:
        # Try to execute without starting
        result = await node.exec({"prompt": "This will fail"})
    except RuntimeError as e:
        print(f"✓ Expected error: {e}")

    try:
        await node.start()

        # Try to execute without instances
        result = await node.exec({"prompt": "This will also fail"})
    except RuntimeError as e:
        print(f"✓ Expected error: {e}")

    finally:
        await node.stop()


async def example_multiple_nodes():
    """Example with multiple nodes."""
    print("\n" + "=" * 70)
    print("Example 5: Multiple Nodes (Different Models)")
    print("=" * 70)

    # Create nodes for different models
    node_gpt4 = Node(
        model_id="gpt-4",
        predictor_url="http://localhost:8001",
        scheduler_port=8400,
    )

    node_llama = Node(
        model_id="llama-2-70b",
        predictor_url="http://localhost:8002",
        scheduler_port=8500,
    )

    try:
        # Start both nodes
        await node_gpt4.start()
        await node_llama.start()

        print("\n✓ Both nodes started successfully")
        print(f"  - GPT-4 Node: scheduler on port 8400")
        print(f"  - Llama Node: scheduler on port 8500")

        # Each node manages its own instances and scheduler
        print("\nEach node:")
        print("  - Has its own scheduler process")
        print("  - Configured with its own predictor")
        print("  - Manages its own set of instances")

    finally:
        # Cleanup both
        await node_gpt4.stop()
        await node_llama.stop()


async def example_advanced_configuration():
    """Example with advanced configuration."""
    print("\n" + "=" * 70)
    print("Example 6: Advanced Configuration")
    print("=" * 70)

    # Custom scheduler module path
    node = Node(
        model_id="custom-model",
        predictor_url="http://localhost:8001",
        scheduler_host="0.0.0.0",  # Listen on all interfaces
        scheduler_port=9000,
        scheduler_module_path="/custom/path/to/scheduler",  # Custom path
    )

    print(f"Node configuration:")
    print(f"  Model ID: {node.model_id}")
    print(f"  Predictor: {node.predictor_url}")
    print(f"  Scheduler: {node.scheduler_host}:{node.scheduler_port}")
    print(f"  Scheduler module: {node.scheduler_module_path}")

    # Note: This node won't start if scheduler module doesn't exist
    # await node.start()


async def example_monitoring():
    """Example of monitoring node status."""
    print("\n" + "=" * 70)
    print("Example 7: Monitoring Node Status")
    print("=" * 70)

    node = Node(
        model_id="gpt-4",
        predictor_url="http://localhost:8001",
        scheduler_port=8600,
    )

    print(f"Before start - is_running(): {node.is_running()}")

    try:
        await node.start()
        print(f"After start - is_running(): {node.is_running()}")

        # Check registered instances
        print(f"Registered instances: {len(node.instance_list)}")

        # Simulate instance registration
        # instance = InstanceClient(base_url="http://localhost:5001")
        # await node.register_instance(instance)
        # print(f"After registration: {len(node.instance_list)} instances")

    finally:
        await node.stop()
        print(f"After stop - is_running(): {node.is_running()}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Node Usage Examples")
    print("=" * 70)

    print("""
IMPORTANT: These examples assume the following services are running:
- Predictor service on http://localhost:8001
- Instance service(s) on http://localhost:5001, 5002, etc.

For demonstration purposes, some examples will show expected behavior
without actually executing (to avoid connection errors).
    """)

    try:
        # Example 1: Basic usage
        # await example_basic_usage()

        # Example 2: Context manager
        # await example_with_context_manager()

        # Example 3: Complete workflow (documentation)
        await example_complete_workflow()

        # Example 4: Error handling
        await example_error_handling()

        # Example 5: Multiple nodes
        # await example_multiple_nodes()

        # Example 6: Advanced configuration
        await example_advanced_configuration()

        # Example 7: Monitoring
        # await example_monitoring()

        print("\n" + "=" * 70)
        print("Examples completed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Most examples require running predictor and instance services.")
        print("To run full examples, start the required services first.")


if __name__ == "__main__":
    asyncio.run(main())
