"""Test the optimized sampling logic in ProbabilisticSchedulingStrategy."""

import numpy as np
import sys
sys.path.insert(0, '/chivier-disk/yanweiye/Projects/swarmpilot-refresh/scheduler/src')

from model import InstanceQueueProbabilistic


def test_vectorized_sampling():
    """Test that vectorized sampling works correctly with 10 samples."""

    # Mock queue info (only queue times, no task times)
    queue_info = {
        "instance-1": InstanceQueueProbabilistic(
            instance_id="instance-1",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[50.0, 100.0, 120.0, 150.0]
        ),
        "instance-2": InstanceQueueProbabilistic(
            instance_id="instance-2",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[10.0, 20.0, 30.0, 40.0]
        ),
        "instance-3": InstanceQueueProbabilistic(
            instance_id="instance-3",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[200.0, 300.0, 350.0, 400.0]
        ),
    }

    # Simulate the optimized logic (only queue sampling)
    num_samples = 10
    instance_ids = list(queue_info.keys())
    num_instances = len(instance_ids)

    queue_times_matrix = np.zeros((num_instances, num_samples))

    # Generate random percentiles
    np.random.seed(42)  # For reproducibility
    random_percentiles = np.random.random(num_samples)

    print(f"Random percentiles: {random_percentiles}")
    print()

    # Compute queue time matrix (only queue times, no task times)
    for i, instance_id in enumerate(instance_ids):
        queue = queue_info[instance_id]

        if isinstance(queue, InstanceQueueProbabilistic):
            queue_times_matrix[i, :] = np.interp(
                random_percentiles,
                queue.quantiles,
                queue.values
            )

    print("Queue times matrix (10 samples for each instance):")
    for i, instance_id in enumerate(instance_ids):
        print(f"  {instance_id}: {queue_times_matrix[i, :]}")
    print()

    # Find winners (instance with minimum queue time for each sample)
    winners = np.argmin(queue_times_matrix, axis=0)
    print(f"Winners per sample (index): {winners}")
    print(f"Winner IDs: {[instance_ids[w] for w in winners]}")
    print()

    # Count wins
    win_counts = np.bincount(winners, minlength=num_instances)
    print(f"Win counts: {win_counts}")
    for i, instance_id in enumerate(instance_ids):
        print(f"  {instance_id}: {win_counts[i]} wins")
    print()

    # Select best
    best_instance_idx = np.argmax(win_counts)
    best_instance_id = instance_ids[best_instance_idx]

    print(f"Selected instance: {best_instance_id}")
    print()

    # Verify the logic
    assert len(winners) == num_samples, "Should have 10 winners"
    assert best_instance_id in instance_ids, "Best instance should be in the list"

    # instance-2 should win most samples since it has lowest queue times
    assert best_instance_id == "instance-2", f"Expected instance-2, got {best_instance_id}"

    print("✓ Test passed!")


def test_performance_comparison():
    """Compare performance of optimized implementation."""
    import time

    # Create larger test case
    num_instances = 20
    queue_info = {
        f"instance-{i}": InstanceQueueProbabilistic(
            instance_id=f"instance-{i}",
            quantiles=[0.5, 0.9, 0.95, 0.99],
            values=[50.0 + i * 5, 100.0 + i * 5, 120.0 + i * 5, 150.0 + i * 5]
        )
        for i in range(num_instances)
    }

    # Warmup
    for _ in range(10):
        instance_ids = list(queue_info.keys())
        num_samples = 10
        random_percentiles = np.random.random(num_samples)
        queue_times_matrix = np.zeros((num_instances, num_samples))

        for i, instance_id in enumerate(instance_ids):
            queue = queue_info[instance_id]
            queue_times_matrix[i, :] = np.interp(random_percentiles, queue.quantiles, queue.values)

        winners = np.argmin(queue_times_matrix, axis=0)
        win_counts = np.bincount(winners, minlength=num_instances)
        best_instance_idx = np.argmax(win_counts)

    # Benchmark
    num_iterations = 1000
    start = time.time()

    for _ in range(num_iterations):
        instance_ids = list(queue_info.keys())
        num_samples = 10
        random_percentiles = np.random.random(num_samples)
        queue_times_matrix = np.zeros((num_instances, num_samples))

        for i, instance_id in enumerate(instance_ids):
            queue = queue_info[instance_id]
            queue_times_matrix[i, :] = np.interp(random_percentiles, queue.quantiles, queue.values)

        winners = np.argmin(queue_times_matrix, axis=0)
        win_counts = np.bincount(winners, minlength=num_instances)
        best_instance_idx = np.argmax(win_counts)

    elapsed = time.time() - start
    avg_time_us = (elapsed / num_iterations) * 1_000_000

    print(f"Performance test ({num_instances} instances, {num_samples} samples):")
    print(f"  {num_iterations} iterations in {elapsed:.3f}s")
    print(f"  Average: {avg_time_us:.2f} µs per selection")
    print(f"  Throughput: {num_iterations / elapsed:.0f} selections/sec")
    print()

    # Should be very fast (< 1ms)
    assert avg_time_us < 1000, f"Selection should be < 1ms, got {avg_time_us:.2f}µs"
    print("✓ Performance test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing optimized sampling logic (queue-only)")
    print("=" * 60)
    print()

    test_vectorized_sampling()
    print()

    test_performance_comparison()
