"""Online API multi-provider example — httpx client.

Demonstrates sending chat completion requests through SwarmPilot
Schedulers that load-balance across 3 cloud inference providers.
The Scheduler transparently picks a provider, injects auth headers,
and forwards the request — the client doesn't know or care which
provider actually served the request.

Architecture:
    Scheduler A  http://localhost:8010  →  Qwen3-8B           (Together / Fireworks / Lepton)
    Scheduler B  http://localhost:8020  →  Qwen3-Next-80B-A3B (Together / Fireworks / Lepton)
    Scheduler C  http://localhost:8030  →  Gemma4-41B         (Together / Fireworks / Lepton)

Usage:
    python examples/7.online_api/api_example.py
"""

import httpx

SCHEDULER_QWEN3_8B = "http://localhost:8010"
SCHEDULER_QWEN3_NEXT = "http://localhost:8020"
SCHEDULER_GEMMA4 = "http://localhost:8030"

SCHEDULERS = {
    "Qwen/Qwen3-8B": SCHEDULER_QWEN3_8B,
    "Qwen/Qwen3-Next-80B-A3B": SCHEDULER_QWEN3_NEXT,
    "google/gemma-4-41b-it": SCHEDULER_GEMMA4,
}

PROMPTS = [
    "Explain quantum computing in one paragraph.",
    "Write a Python function that checks if a number is prime.",
    "What are the main differences between TCP and UDP?",
]


def show_instances(client: httpx.Client) -> None:
    """List registered instances for each model/scheduler."""
    print("=" * 70)
    print("REGISTERED INSTANCES")
    print("=" * 70)

    for model_id, url in SCHEDULERS.items():
        resp = client.get(f"{url}/v1/instance/list").json()
        instances = resp.get("instances", [])
        print(f"\n  {model_id}  ({url})")
        for inst in instances:
            iid = inst.get("instance_id", "?")
            endpoint = inst.get("endpoint", "?")
            status = inst.get("status", "?")
            print(f"    ├── {iid:<28s} {endpoint:<45s} [{status}]")
        if not instances:
            print("    └── (no instances)")


def send_chat_request(
    client: httpx.Client,
    scheduler_url: str,
    model_id: str,
    prompt: str,
    task_id: str,
) -> dict:
    """Send a chat completion request via the Scheduler proxy.

    The request is routed to one of the 3 providers. The Scheduler
    picks the provider, injects the auth header, and forwards.

    Args:
        client: httpx client.
        scheduler_url: Scheduler base URL.
        model_id: Model identifier.
        prompt: User prompt text.
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
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            },
            "metadata": {
                "path": "v1/chat/completions",
                "method": "POST",
            },
        },
    ).json()


def main() -> None:
    """Send requests to all 3 models and display results."""
    with httpx.Client(timeout=30.0) as client:
        # ── Show registered instances ────────────────────────────
        show_instances(client)

        # ── Send requests to each model ─────────────────────────
        print("\n" + "=" * 70)
        print("SENDING REQUESTS (3 models x 3 prompts = 9 tasks)")
        print("=" * 70)

        task_num = 0
        for model_id, scheduler_url in SCHEDULERS.items():
            model_short = model_id.split("/")[-1]
            print(f"\n  [{model_short}]")

            for prompt in PROMPTS:
                task_num += 1
                task_id = f"task-{task_num:03d}"
                resp = send_chat_request(
                    client, scheduler_url, model_id, prompt, task_id
                )
                status = resp.get("status", "?")
                instance = resp.get("assigned_instance", "?")
                print(
                    f"    {task_id}  → {instance:<30s} "
                    f'"{prompt[:45]}..."  [{status}]'
                )

        # ── Show task list from each scheduler ──────────────────
        print("\n" + "=" * 70)
        print("TASK SUMMARY")
        print("=" * 70)

        for model_id, url in SCHEDULERS.items():
            resp = client.get(f"{url}/v1/task/list").json()
            tasks = resp.get("tasks", [])
            completed = sum(1 for t in tasks if t.get("status") == "completed")
            print(
                f"  {model_id.split('/')[-1]:<35s} "
                f"{completed}/{len(tasks)} completed"
            )


if __name__ == "__main__":
    main()
