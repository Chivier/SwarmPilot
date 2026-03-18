# This example uses raw HTTP calls because SwarmPilotClient SDK
# requires a Planner. For single-model deployment without Planner,
# interact with the Scheduler directly via httpx.

import json
import sys

import httpx

SCHEDULER_URL = "http://localhost:8000"


def main() -> None:
    """List instances, submit a task, and check its status."""
    with httpx.Client(timeout=30.0) as client:
        print("=== 1. List registered instances ===")
        resp = client.get(f"{SCHEDULER_URL}/v1/instance/list")
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))

        print("\n=== 2. Submit a task ===")
        task_resp = client.post(
            f"{SCHEDULER_URL}/v1/task/submit",
            json={
                "task_id": "task-001",
                "model_id": "Qwen/Qwen3-8B-VL",
                "task_input": {
                    "prompt": "Hello!",
                    "max_tokens": 50,
                },
                "metadata": {
                    "path": "v1/completions",
                    "method": "POST",
                },
            },
        )
        task_resp.raise_for_status()
        print(json.dumps(task_resp.json(), indent=2))

        print("\n=== 3. Check task status ===")
        status_resp = client.get(f"{SCHEDULER_URL}/v1/task/status/task-001")
        status_resp.raise_for_status()
        print(json.dumps(status_resp.json(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print(
            "Error: Cannot connect to Scheduler. "
            "Run start_cluster.sh and deploy_model.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)
