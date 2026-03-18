"""SDK example for multi-model planner-managed deployment.

Requires running Planner with PyLet. For no-planner usage,
see examples/single_model/ and examples/multi_model_direct/.
"""

import asyncio

from swarmpilot.sdk import SwarmPilotClient


async def main() -> None:
    """Deploy two models via Planner SDK, scale, and terminate."""
    async with SwarmPilotClient("http://localhost:8002") as sp:
        schedulers = await sp.schedulers()
        print(f"Scheduler mapping: {schedulers}")

        qwen = await sp.serve("Qwen/Qwen3-8B-VL", gpu=1, replicas=2)
        print(f"Deployed {qwen.name}: {len(qwen.instances)} instances")

        llama = await sp.serve("meta-llama/Llama-3.1-8B", gpu=1, replicas=2)
        print(f"Deployed {llama.name}: {len(llama.instances)} instances")

        scaled = await sp.scale("Qwen/Qwen3-8B-VL", replicas=3)
        print(f"Scaled to {len(scaled.instances)} instances")

        await sp.terminate(all=True)
        print("All instances terminated")

    # Note: sp.instances() has a known API/SDK mismatch
    # and is not used here.


if __name__ == "__main__":
    asyncio.run(main())
