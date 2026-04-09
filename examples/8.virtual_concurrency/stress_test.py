"""Stress test — submit 450 requests (150 per scheduler) and verify
all complete with correct distribution across virtual instances.

Usage:
    python examples/8.virtual_concurrency/stress_test.py
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

TASKS_PER_SCHEDULER = 150
TOTAL_TASKS = TASKS_PER_SCHEDULER * len(SCHEDULERS)  # 450
MAX_WORKERS = 50  # concurrent submission threads per scheduler
MAX_WAIT = 120  # seconds to wait for completion

PASS = "\033[0;32mPASS\033[0m"
FAIL = "\033[0;31mFAIL\033[0m"
BOLD = "\033[1m"
NC = "\033[0m"


def submit_task(
    client: httpx.Client,
    url: str,
    model_id: str,
    task_id: str,
) -> dict:
    """Submit one task.

    Args:
        client: httpx client.
        url: Scheduler base URL.
        model_id: Model identifier.
        task_id: Unique task ID.

    Returns:
        Response dict from scheduler.
    """
    return client.post(
        f"{url}/v1/task/submit",
        json={
            "task_id": task_id,
            "model_id": model_id,
            "task_input": {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Stress test request {task_id}",
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
) -> dict[str, dict]:
    """Get stats per instance.

    Args:
        client: httpx client.
        url: Scheduler base URL.

    Returns:
        Dict mapping instance_id to stats dict.
    """
    resp = client.get(f"{url}/v1/instance/list").json()
    result = {}
    for inst in resp.get("instances", []):
        iid = inst["instance_id"]
        info = client.get(
            f"{url}/v1/instance/info",
            params={"instance_id": iid},
        ).json()
        result[iid] = {
            "completed": info.get("stats", {}).get(
                "completed_tasks", 0
            ),
            "failed": info.get("stats", {}).get("failed_tasks", 0),
            "pending": info.get("stats", {}).get("pending_tasks", 0),
            "endpoint_group": inst.get("endpoint_group", "?"),
        }
    return result


def wait_for_completion(
    client: httpx.Client, url: str, expected: int
) -> tuple[float, int, int]:
    """Wait until all tasks finish.

    Args:
        client: httpx client.
        url: Scheduler base URL.
        expected: Expected total completed+failed.

    Returns:
        Tuple of (elapsed_seconds, completed, failed).
    """
    start = time.time()
    while time.time() - start < MAX_WAIT:
        health = client.get(f"{url}/v1/health").json()
        stats = health.get("stats", {})
        completed = stats.get("completed_tasks", 0)
        failed = stats.get("failed_tasks", 0)
        pending = stats.get("pending_tasks", 0)
        running = stats.get("running_tasks", 0)

        if pending == 0 and running == 0:
            return time.time() - start, completed, failed
        time.sleep(0.5)

    # Timed out — return what we have
    health = client.get(f"{url}/v1/health").json()
    stats = health.get("stats", {})
    return (
        time.time() - start,
        stats.get("completed_tasks", 0),
        stats.get("failed_tasks", 0),
    )


def main() -> None:
    """Run the stress test."""
    print()
    print(
        f"{BOLD}Stress Test: {TOTAL_TASKS} requests "
        f"({TASKS_PER_SCHEDULER} per scheduler){NC}"
    )
    print(
        f"3 models x 3 providers "
        f"(concurrency 1/2/3 = 6 virtual instances each)"
    )
    print()

    with httpx.Client(timeout=30.0) as client:
        # Connectivity check
        for model_id, url in SCHEDULERS.items():
            try:
                client.get(f"{url}/v1/health")
            except httpx.ConnectError:
                print(f"Cannot reach {url}")
                sys.exit(1)

        all_ok = True

        for model_id, url in SCHEDULERS.items():
            model_short = model_id.split("/")[-1]
            print("=" * 70)
            print(
                f"  {model_short} — submitting "
                f"{TASKS_PER_SCHEDULER} tasks"
            )
            print("=" * 70)

            # Snapshot before
            stats_before = get_instance_stats(client, url)

            # Submit all tasks concurrently
            t_submit_start = time.time()
            submit_results = []
            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as pool:
                futures = {
                    pool.submit(
                        submit_task,
                        client,
                        url,
                        model_id,
                        f"stress-{model_short}-{i:04d}",
                    ): i
                    for i in range(TASKS_PER_SCHEDULER)
                }
                for f in as_completed(futures):
                    submit_results.append(f.result())
            t_submit = time.time() - t_submit_start

            submitted = sum(
                1 for r in submit_results if r.get("success")
            )
            failed_submit = TASKS_PER_SCHEDULER - submitted
            print(
                f"  Submitted: {submitted}/{TASKS_PER_SCHEDULER} "
                f"in {t_submit:.1f}s "
                f"({submitted / max(t_submit, 0.01):.0f} req/s)"
            )
            if failed_submit:
                print(f"  Submit failures: {failed_submit}")
                all_ok = False

            # Wait for all to complete
            elapsed, completed, failed = wait_for_completion(
                client, url, TASKS_PER_SCHEDULER
            )

            # Get per-instance distribution
            stats_after = get_instance_stats(client, url)
            distribution: dict[str, int] = {}
            for iid in stats_after:
                delta = stats_after[iid]["completed"] - stats_before.get(
                    iid, {}
                ).get("completed", 0)
                if delta > 0:
                    distribution[iid] = delta

            # Display
            print(
                f"\n  Completed: {completed} | "
                f"Failed: {failed} | "
                f"Time: {elapsed:.1f}s | "
                f"Throughput: "
                f"{completed / max(elapsed, 0.01):.1f} req/s"
            )

            print(f"\n  {'Instance':<38s} {'Tasks':>5s}  Distribution")
            print(f"  {'─' * 38} {'─' * 5}  {'─' * 30}")

            # Group by endpoint_group for cleaner display
            groups: dict[str, list[tuple[str, int]]] = {}
            for iid, count in sorted(distribution.items()):
                group = stats_after[iid]["endpoint_group"]
                groups.setdefault(group, []).append((iid, count))

            for group, items in sorted(groups.items()):
                group_total = sum(c for _, c in items)
                for iid, count in items:
                    bar_len = int(count / max(1, TASKS_PER_SCHEDULER) * 40)
                    bar = "#" * bar_len
                    print(
                        f"  {iid:<38s} {count:>5d}  {bar}"
                    )
                print(
                    f"  {'':38s} {'─' * 5}  "
                    f"subtotal: {group_total}"
                )

            total_distributed = sum(distribution.values())
            n_instances_used = len(distribution)

            # Checks
            print()
            # All tasks completed?
            ok = completed >= TASKS_PER_SCHEDULER and failed == 0
            status = PASS if ok else FAIL
            if not ok:
                all_ok = False
            print(
                f"  All tasks completed: "
                f"{completed}/{TASKS_PER_SCHEDULER}, "
                f"{failed} failed  [{status}]"
            )

            # All 6 instances used?
            ok = n_instances_used == 6
            status = PASS if ok else FAIL
            if not ok:
                all_ok = False
            print(
                f"  All 6 virtual instances used: "
                f"{n_instances_used}/6  [{status}]"
            )

            # Distribution roughly proportional to concurrency?
            # Together(1): ~1/6, Fireworks(2): ~2/6, Lepton(3): ~3/6
            for group, items in sorted(groups.items()):
                provider = group.split("-")[0]
                expected_slots = {
                    "together": 1,
                    "fireworks": 2,
                    "lepton": 3,
                }.get(provider, 0)
                expected_frac = expected_slots / 6
                actual = sum(c for _, c in items)
                actual_frac = actual / max(1, total_distributed)
                # Allow 20% tolerance
                within = abs(actual_frac - expected_frac) < 0.20
                status = PASS if within else FAIL
                if not within:
                    all_ok = False
                print(
                    f"  {group}: {actual} tasks "
                    f"({actual_frac:.0%} actual vs "
                    f"{expected_frac:.0%} expected)  [{status}]"
                )

            print()

        # Final summary
        print("=" * 70)
        print(f"  {BOLD}FINAL RESULT{NC}")
        print("=" * 70)
        if all_ok:
            print(
                f"\n  All {TOTAL_TASKS} requests across "
                f"3 schedulers completed successfully.  [{PASS}]"
            )
        else:
            print(f"\n  Some checks failed.  [{FAIL}]")
            sys.exit(1)


if __name__ == "__main__":
    main()
