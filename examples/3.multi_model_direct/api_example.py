"""Multi-model direct API usage with httpx (no SDK, no Planner).

Without a Planner, the client must know which scheduler handles which
model.  Each scheduler serves exactly one model.

    Scheduler A  http://localhost:8010  →  Qwen/Qwen3-8B-VL
    Scheduler B  http://localhost:8020  →  meta-llama/Llama-3.1-8B
"""

import httpx

SCHEDULER_QWEN = "http://localhost:8010"
SCHEDULER_LLAMA = "http://localhost:8020"


def main() -> None:
    """Query both per-model schedulers and submit one task to each."""
    with httpx.Client(timeout=30.0) as client:
        # (1) List Qwen instances from Scheduler A
        qwen_instances = client.get(f"{SCHEDULER_QWEN}/v1/instance/list").json()
        print(f"Qwen instances: {len(qwen_instances.get('instances', []))}")

        # (2) List Llama instances from Scheduler B
        llama_instances = client.get(
            f"{SCHEDULER_LLAMA}/v1/instance/list"
        ).json()
        print(f"Llama instances: {len(llama_instances.get('instances', []))}")

        # (3) Submit a task to Qwen via Scheduler A
        qwen_task = client.post(
            f"{SCHEDULER_QWEN}/v1/task/submit",
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
        ).json()
        print(f"Qwen task: {qwen_task}")

        # (4) Submit a task to Llama via Scheduler B
        llama_task = client.post(
            f"{SCHEDULER_LLAMA}/v1/task/submit",
            json={
                "task_id": "task-002",
                "model_id": "meta-llama/Llama-3.1-8B",
                "task_input": {
                    "prompt": "Summarize this.",
                    "max_tokens": 100,
                },
                "metadata": {
                    "path": "v1/completions",
                    "method": "POST",
                },
            },
        ).json()
        print(f"Llama task: {llama_task}")


if __name__ == "__main__":
    main()
