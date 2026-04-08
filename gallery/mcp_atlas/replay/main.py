"""Top-level orchestration: prepare (reshape) and run (replay) commands."""

from __future__ import annotations

import json
from pathlib import Path

from replay.models import ExperimentConfig, PlannerConfig, ReplayGroup


def prepare(
    output_path: str,
    limit: int | None = None,
) -> None:
    """Load MCP-Atlas dataset, reshape into ReplayGroups, and write to JSONL.

    Args:
        output_path: Path for the output JSONL file.
        limit: Maximum number of conversations to include.
    """
    from replay.reshapers.mcp_atlas import MCPAtlasReshaper

    groups = MCPAtlasReshaper().load_and_reshape(limit=limit)

    if not groups:
        print("No replay groups produced. Check dataset and limit.")
        return

    total_steps = sum(len(g.steps) for g in groups)
    print(f"Reshaped {len(groups)} groups, {total_steps} total steps")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for g in groups:
            f.write(json.dumps(g.model_dump(), ensure_ascii=False) + "\n")

    print(f"Prepared data written to {output_path}")


def _load_prepared_data(data_path: str) -> list[ReplayGroup]:
    """Load ReplayGroups from a prepared JSONL file (one group per line).

    Args:
        data_path: Path to the JSONL file produced by ``prepare()``.

    Returns:
        List of ReplayGroups.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    groups: list[ReplayGroup] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                groups.append(ReplayGroup.model_validate(json.loads(line)))
    return groups


async def _deploy_models(planner_cfg: PlannerConfig) -> None:
    """Deploy large and small models via the Planner SDK.

    Args:
        planner_cfg: Planner deployment configuration.
    """
    from swarmpilot.sdk import SwarmPilotClient

    async with SwarmPilotClient(planner_cfg.url) as sp:
        print(
            f"Deploying {planner_cfg.large_model_id}"
            f" (gpu={planner_cfg.gpu_per_instance_large},"
            f" replicas={planner_cfg.replicas_large})..."
        )
        group = await sp.serve(
            planner_cfg.large_model_id,
            gpu=planner_cfg.gpu_per_instance_large,
            replicas=planner_cfg.replicas_large,
        )
        print(f"  Large model: {len(group.instances)} instances")

        print(
            f"Deploying {planner_cfg.small_model_id}"
            f" (gpu={planner_cfg.gpu_per_instance_small},"
            f" replicas={planner_cfg.replicas_small})..."
        )
        group = await sp.serve(
            planner_cfg.small_model_id,
            gpu=planner_cfg.gpu_per_instance_small,
            replicas=planner_cfg.replicas_small,
        )
        print(f"  Small model: {len(group.instances)} instances")


async def _teardown_models(planner_cfg: PlannerConfig) -> None:
    """Terminate all model instances via the Planner SDK.

    Args:
        planner_cfg: Planner deployment configuration.
    """
    from swarmpilot.sdk import SwarmPilotClient

    async with SwarmPilotClient(planner_cfg.url) as sp:
        print("Terminating all model instances...")
        await sp.terminate(all=True)
        print("  Done")


async def run(
    data_path: str,
    config: ExperimentConfig,
    output_path: str = "./results.json",
    no_deploy: bool = False,
    no_teardown: bool = False,
) -> None:
    """Execute the replay experiment using prepared data and YAML-loaded config.

    When ``config.planner`` is set, automatically deploys models via
    the Planner SDK before the experiment and terminates them after.
    Use ``no_deploy`` / ``no_teardown`` to skip these steps.

    Args:
        data_path: Path to the prepared data JSON file.
        config: Experiment configuration (endpoints + timing).
        output_path: Path for the results JSON file.
        no_deploy: Skip automatic model deployment.
        no_teardown: Skip automatic model teardown.
    """
    from replay.client import ReplayClient
    from replay.metrics import MetricsCollector
    from replay.reporter import ReplayReporter
    from replay.scheduler import ReplayScheduler

    # Deploy models via Planner if configured.
    if config.planner and not no_deploy:
        await _deploy_models(config.planner)
        print()

    groups = _load_prepared_data(data_path)
    if not groups:
        print("No replay groups found in prepared data.")
        return

    total_steps = sum(len(g.steps) for g in groups)
    print(f"Replaying {len(groups)} groups, {total_steps} total requests")
    print(
        f"Poisson QPS={config.poisson_qps}, Global QPS={config.global_qps}, "
        f"Agent delay={config.agent_delay_ms}ms, User delay={config.user_delay_ms}ms"
    )

    reporter = ReplayReporter()
    collector = MetricsCollector(
        total_steps=total_steps,
        progress_callback=reporter.print_progress,
    )
    client = ReplayClient(
        large=config.large_model,
        small=config.small_model,
        timeout_s=config.timeout_s,
        max_tokens=config.max_tokens,
    )
    scheduler = ReplayScheduler(
        poisson_qps=config.poisson_qps,
        global_qps=config.global_qps,
        agent_delay_ms=config.agent_delay_ms,
        user_delay_ms=config.user_delay_ms,
        client=client,
        collector=collector,
    )

    try:
        group_metrics = await scheduler.run_all(groups)
    finally:
        await client.close()

    reporter.print_summary(group_metrics)
    reporter.write_json(group_metrics, output_path)
    print(f"\nResults written to {output_path}")

    # Teardown models via Planner if configured.
    if config.planner and not no_teardown:
        print()
        await _teardown_models(config.planner)
