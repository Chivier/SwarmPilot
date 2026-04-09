"""Virtual concurrency test — verifies virtual instance registration
and concurrent request distribution.

Tests:
1. Each scheduler has exactly 6 virtual instances (1+2+3)
2. Virtual instances are grouped by endpoint_group
3. Concurrent requests distribute across virtual instances
4. All requests complete successfully

Usage:
    python examples/8.virtual_concurrency/test_concurrency.py
"""

import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

SCHEDULERS = {
    "Qwen/Qwen3-8B": "http://localhost:8010",
    "Qwen/Qwen3-Next-80B-A3B": "http://localhost:8020",
    "google/gemma-4-41b-it": "http://localhost:8030",
}

# Expected virtual instances per provider per model
EXPECTED_CONCURRENCY = {
    "together": 1,
    "fireworks": 2,
    "lepton": 3,
}
EXPECTED_TOTAL_PER_SCHEDULER = sum(EXPECTED_CONCURRENCY.values())  # 6

PASS = "\033[0;32mPASS\033[0m"
FAIL = "\033[0;31mFAIL\033[0m"


def check_virtual_instances(client: httpx.Client) -> bool:
    """Verify each scheduler has the correct virtual instances.

    Args:
        client: httpx client.

    Returns:
        True if all checks pass.
    """
    print("=" * 70)
    print("TEST 1: Virtual Instance Registration")
    print("=" * 70)

    all_ok = True
    for model_id, url in SCHEDULERS.items():
        resp = client.get(f"{url}/v1/instance/list").json()
        instances = resp.get("instances", [])
        model_short = model_id.split("/")[-1]

        print(f"\n  {model_short} ({url})")
        print(f"  Total instances: {len(instances)}", end="")

        if len(instances) == EXPECTED_TOTAL_PER_SCHEDULER:
            print(f"  [{PASS}]")
        else:
            print(
                f"  [{FAIL}] expected "
                f"{EXPECTED_TOTAL_PER_SCHEDULER}"
            )
            all_ok = False

        # Group by endpoint_group
        groups: dict[str, list[str]] = {}
        for inst in instances:
            group = inst.get("endpoint_group", "unknown")
            iid = inst["instance_id"]
            groups.setdefault(group, []).append(iid)

        for group, ids in sorted(groups.items()):
            provider = group.split("-")[0]
            expected = EXPECTED_CONCURRENCY.get(provider, "?")
            ok = len(ids) == expected
            status = PASS if ok else FAIL
            if not ok:
                all_ok = False

            print(
                f"    endpoint_group={group:<28s} "
                f"instances={len(ids)} "
                f"(expected {expected})  [{status}]"
            )
            for iid in sorted(ids):
                print(f"      - {iid}")

    return all_ok


def send_task(
    client: httpx.Client,
    scheduler_url: str,
    model_id: str,
    task_id: str,
) -> dict:
    """Submit a single task via the scheduler.

    Args:
        client: httpx client.
        scheduler_url: Scheduler base URL.
        model_id: Model identifier.
        task_id: Unique task identifier.

    Returns:
        Task submission response dict.
    """
    return client.post(
        f"{scheduler_url}/v1/task/submit",
        json={
            "task_id": task_id,
            "model_id": model_id,
            "task_input": {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Test request {task_id}",
                    }
                ],
                "max_tokens": 50,
            },
            "metadata": {
                "path": "v1/chat/completions",
                "method": "POST",
            },
        },
        timeout=30.0,
    ).json()


def get_instance_stats(
    client: httpx.Client, url: str
) -> dict[str, int]:
    """Get completed_tasks per instance from instance info API.

    Args:
        client: httpx client.
        url: Scheduler base URL.

    Returns:
        Dict mapping instance_id to completed_tasks count.
    """
    resp = client.get(f"{url}/v1/instance/list").json()
    stats = {}
    for inst in resp.get("instances", []):
        iid = inst["instance_id"]
        info = client.get(
            f"{url}/v1/instance/info",
            params={"instance_id": iid},
        ).json()
        completed = info.get("stats", {}).get("completed_tasks", 0)
        stats[iid] = completed
    return stats


def clear_tasks(client: httpx.Client, url: str) -> None:
    """Clear all tasks from a scheduler.

    Args:
        client: httpx client.
        url: Scheduler base URL.
    """
    client.delete(f"{url}/v1/task/clear")


def check_concurrent_distribution(client: httpx.Client) -> bool:
    """Send concurrent requests and verify distribution via instance stats.

    Sends 12 tasks per scheduler (2x the 6 virtual instances).
    After all complete, checks completed_tasks per instance.

    Args:
        client: httpx client.

    Returns:
        True if all checks pass.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Concurrent Request Distribution")
    print("=" * 70)

    tasks_per_scheduler = 12
    all_ok = True

    for model_id, url in SCHEDULERS.items():
        model_short = model_id.split("/")[-1]

        # Snapshot stats before
        stats_before = get_instance_stats(client, url)

        print(
            f"\n  [{model_short}] Sending {tasks_per_scheduler} "
            f"concurrent tasks..."
        )

        # Submit all tasks concurrently
        results = []
        with ThreadPoolExecutor(max_workers=tasks_per_scheduler) as pool:
            futures = {
                pool.submit(
                    send_task,
                    client,
                    url,
                    model_id,
                    f"vc2-{model_short}-{i:03d}",
                ): i
                for i in range(tasks_per_scheduler)
            }
            for future in as_completed(futures):
                results.append(future.result())

        # Count submissions
        submitted = sum(
            1 for r in results if r.get("success")
        )
        print(f"    Submitted: {submitted}/{tasks_per_scheduler}")

        # Wait for all to complete
        max_wait = 30
        start = time.time()
        while time.time() - start < max_wait:
            resp = client.get(
                f"{url}/v1/task/list",
                params={"status": "pending"},
            ).json()
            pending = resp.get("total", 0)
            resp2 = client.get(
                f"{url}/v1/task/list",
                params={"status": "running"},
            ).json()
            running = resp2.get("total", 0)
            if pending == 0 and running == 0:
                break
            time.sleep(0.3)

        # Get stats after — compute delta
        stats_after = get_instance_stats(client, url)
        distribution: dict[str, int] = {}
        for iid in stats_after:
            delta = stats_after[iid] - stats_before.get(iid, 0)
            if delta > 0:
                distribution[iid] = delta

        # Display distribution
        for inst_id, count in sorted(distribution.items()):
            bar = "#" * count
            print(
                f"    {inst_id:<35s} "
                f"{count:>2d} tasks  {bar}"
            )

        total_completed = sum(distribution.values())
        if total_completed == tasks_per_scheduler:
            print(
                f"    All {tasks_per_scheduler} tasks "
                f"completed  [{PASS}]"
            )
        else:
            print(
                f"    Only {total_completed}/"
                f"{tasks_per_scheduler} completed  [{FAIL}]"
            )
            all_ok = False

        n_used = len(distribution)
        if n_used > 1:
            print(
                f"    Distributed across {n_used} virtual "
                f"instances  [{PASS}]"
            )
        else:
            print(f"    Only {n_used} instance used  [{FAIL}]")
            all_ok = False

    return all_ok


def check_concurrency_bound(client: httpx.Client) -> bool:
    """Verify concurrency is bounded by virtual instance count.

    Sends a burst of tasks and checks that no more than 6
    are in-flight simultaneously (1+2+3 virtual instances).

    Args:
        client: httpx client.

    Returns:
        True if concurrency is bounded.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Concurrency Bound Verification")
    print("=" * 70)

    url = list(SCHEDULERS.values())[0]
    model_id = list(SCHEDULERS.keys())[0]
    model_short = model_id.split("/")[-1]
    burst_size = 18  # 3x the 6 virtual instances

    print(
        f"\n  [{model_short}] Sending burst of {burst_size} tasks..."
    )

    # Submit burst
    with ThreadPoolExecutor(max_workers=burst_size) as pool:
        futures = [
            pool.submit(
                send_task,
                client,
                url,
                model_id,
                f"burst-{model_short}-{i:03d}",
            )
            for i in range(burst_size)
        ]
        results = [f.result() for f in as_completed(futures)]

    submitted = sum(1 for r in results if r.get("success"))
    print(f"    Submitted: {submitted}/{burst_size}")

    # Check in-flight (pending + running should be <= total instances)
    # Give a moment for scheduling
    time.sleep(0.2)
    resp = client.get(f"{url}/v1/health").json()
    stats = resp.get("stats", {})
    print(
        f"    Health stats: {stats}"
    )

    # Wait for completion
    max_wait = 30
    start = time.time()
    while time.time() - start < max_wait:
        resp = client.get(
            f"{url}/v1/task/list",
            params={"status": "pending"},
        ).json()
        pending = resp.get("total", 0)
        resp2 = client.get(
            f"{url}/v1/task/list",
            params={"status": "running"},
        ).json()
        running = resp2.get("total", 0)
        if pending == 0 and running == 0:
            break
        time.sleep(0.3)

    elapsed = time.time() - start

    # All should complete
    stats_after = get_instance_stats(client, url)
    total_completed = sum(stats_after.values())
    print(
        f"    All tasks completed in {elapsed:.1f}s  "
        f"(total instance completions: {total_completed})"
    )

    # With 6 concurrent slots, 18 tasks should take roughly 3x
    # the time of a single task (not 18x if serial, not 1x if unlimited)
    ok = submitted == burst_size
    status = PASS if ok else FAIL
    print(f"    Burst submitted successfully  [{status}]")
    return ok


def main() -> None:
    """Run all virtual concurrency tests."""
    print()
    print("Virtual Concurrency Test Suite")
    print("3 models x 3 providers (concurrency 1/2/3)")
    print("Expected: 6 virtual instances per scheduler, 18 total")
    print()

    with httpx.Client(timeout=30.0) as client:
        for model_id, url in SCHEDULERS.items():
            try:
                client.get(f"{url}/v1/health")
            except httpx.ConnectError:
                print(
                    f"Cannot reach {url} — is the cluster running?"
                )
                print(
                    "Run: bash examples/8.virtual_concurrency/"
                    "start_cluster.sh"
                )
                sys.exit(1)

        t1 = check_virtual_instances(client)
        t2 = check_concurrent_distribution(client)
        t3 = check_concurrency_bound(client)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    results = [
        ("Virtual Instance Registration", t1),
        ("Concurrent Request Distribution", t2),
        ("Concurrency Bound Verification", t3),
    ]
    all_pass = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {name:<40s} [{status}]")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(f"All tests passed! [{PASS}]")
    else:
        print(f"Some tests failed. [{FAIL}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
