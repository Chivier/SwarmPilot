#!/usr/bin/env python3
"""
Initial redeploy helper for Exp13 Workflow Benchmark - Type2 Deep Research (Simulation Mode).
Builds a planner migration payload and posts to /deploy/migration.

This script is designed to work with start_all_services.sh and uses
the same port configuration:
  - Scheduler A: 8100
  - Scheduler B: 8200
  - Planner: 8202
  - Group A instances: 8210, 8211, ...
  - Group B instances: 8300, 8301, ...

Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
  - Model A (sleep_model_a) -> Scheduler A (for A and Merge tasks)
  - Model B (sleep_model_b) -> Scheduler B (for B1/B2 tasks)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

import requests


# Default configuration (matches start_all_services.sh)
DEFAULT_HOST = "localhost"
DEFAULT_SCHEDULER_A_PORT = 8100
DEFAULT_SCHEDULER_B_PORT = 8200
DEFAULT_PLANNER_PORT = 8202
DEFAULT_INSTANCE_GROUP_A_START_PORT = 8210
DEFAULT_INSTANCE_GROUP_B_START_PORT = 8300
DEFAULT_MODEL_ID_A = "sleep_model_a"
DEFAULT_MODEL_ID_B = "sleep_model_b"
DEFAULT_N1 = 4
DEFAULT_N2 = 2


@dataclass
class InstanceInfo:
    endpoint: str
    current_model: str


def build_instances(
    n1: int,
    n2: int,
    host: str = DEFAULT_HOST,
    port_a_start: int = DEFAULT_INSTANCE_GROUP_A_START_PORT,
    port_b_start: int = DEFAULT_INSTANCE_GROUP_B_START_PORT,
    model_id_a: str = DEFAULT_MODEL_ID_A,
    model_id_b: str = DEFAULT_MODEL_ID_B,
    scheduler_a_url: str = None,
    scheduler_b_url: str = None,
) -> tuple[List[InstanceInfo], Dict[str, str], int, int]:
    """
    Build instance list matching start_all_services.sh configuration.

    Only includes instances registered to schedulers (even-indexed instances).
    In start_all_services.sh:
      - Even index (0, 2, 4...) -> registered to Scheduler
      - Odd index (1, 3, 5...)  -> registered to Planner (not included here)

    Group A: even-indexed instances from port_a_start (8210, 8212, ...)
    Group B: even-indexed instances from port_b_start (8300, 8302, ...)

    Returns:
        instances: List of InstanceInfo for scheduler-registered instances
        instance_scheduler_map: Mapping of endpoint -> scheduler URL
        actual_n1: Number of Group A instances included
        actual_n2: Number of Group B instances included
    """
    instances: List[InstanceInfo] = []
    instance_scheduler_map: Dict[str, str] = {}

    actual_n1 = 0
    actual_n2 = 0

    # Group A instances (Model A - for A and Merge tasks) - only even-indexed (registered to Scheduler A)
    for i in range(n1):
        if i % 2 == 0:  # Only even indices are registered to scheduler
            port = port_a_start + i
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model=model_id_a))
            instance_scheduler_map[endpoint] = scheduler_a_url
            actual_n1 += 1

    # Group B instances (Model B - for B1/B2 tasks) - only even-indexed (registered to Scheduler B)
    for i in range(n2):
        if i % 2 == 0:  # Only even indices are registered to scheduler
            port = port_b_start + i
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model=model_id_b))
            instance_scheduler_map[endpoint] = scheduler_b_url
            actual_n2 += 1

    return instances, instance_scheduler_map, actual_n1, actual_n2


def build_payload(
    instances: List[InstanceInfo],
    instance_scheduler_map: Dict[str, str],
    scheduler_a_url: str,
    scheduler_b_url: str,
    model_id_a: str,
    model_id_b: str,
    n1: int,
    n2: int,
) -> Dict:
    """Build the migration payload for the planner service."""
    total_inst = len(instances)

    # Capacity matrix: each instance can host either model
    # [capacity_for_model_a, capacity_for_model_b]
    B = [[1.0, 10] for _ in range(total_inst)]

    # Initial assignment: first n1 instances run model_a, rest run model_b
    initial = [0] * n1 + [1] * n2

    planner_input = {
        "M": total_inst,
        "N": 2,
        "B": B,
        "initial": initial,
        "a": 1.0,
        "target": [1.0, 10],
        "algorithm": "simulated_annealing",
        "objective_method": "ratio_difference",
    }

    scheduler_mapping = {
        model_id_a: scheduler_a_url,
        model_id_b: scheduler_b_url,
    }

    return {
        "instances": [asdict(inst) for inst in instances],
        "planner_input": planner_input,
        "scheduler_mapping": scheduler_mapping,
        "instance_scheduler_mapping": instance_scheduler_map,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Redeploy instances via planner /deploy/migration (Type2 Deep Research - simulation mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
  - Model A (sleep_model_a) -> Scheduler A (for A and Merge tasks)
  - Model B (sleep_model_b) -> Scheduler B (for B1/B2 tasks)

Examples:
  # Use defaults (N1=4, N2=2, localhost)
  python type2_redeploy_sim.py

  # Custom instance counts
  python type2_redeploy_sim.py --n1 8 --n2 4

  # Custom host (for remote deployment)
  python type2_redeploy_sim.py --host 192.168.1.100 --n1 10 --n2 6
        """,
    )

    # Instance configuration
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Host for all services (default: {DEFAULT_HOST})")
    parser.add_argument("--n1", type=int, default=DEFAULT_N1,
                        help=f"Number of Group A instances (default: {DEFAULT_N1})")
    parser.add_argument("--n2", type=int, default=DEFAULT_N2,
                        help=f"Number of Group B instances (default: {DEFAULT_N2})")

    # Port configuration
    parser.add_argument("--scheduler-a-port", type=int, default=DEFAULT_SCHEDULER_A_PORT,
                        help=f"Scheduler A port (default: {DEFAULT_SCHEDULER_A_PORT})")
    parser.add_argument("--scheduler-b-port", type=int, default=DEFAULT_SCHEDULER_B_PORT,
                        help=f"Scheduler B port (default: {DEFAULT_SCHEDULER_B_PORT})")
    parser.add_argument("--planner-port", type=int, default=DEFAULT_PLANNER_PORT,
                        help=f"Planner port (default: {DEFAULT_PLANNER_PORT})")
    parser.add_argument("--port-a-start", type=int, default=DEFAULT_INSTANCE_GROUP_A_START_PORT,
                        help=f"Group A instance start port (default: {DEFAULT_INSTANCE_GROUP_A_START_PORT})")
    parser.add_argument("--port-b-start", type=int, default=DEFAULT_INSTANCE_GROUP_B_START_PORT,
                        help=f"Group B instance start port (default: {DEFAULT_INSTANCE_GROUP_B_START_PORT})")

    # Model configuration
    parser.add_argument("--model-id-a", default=DEFAULT_MODEL_ID_A,
                        help=f"Model ID for Group A (default: {DEFAULT_MODEL_ID_A})")
    parser.add_argument("--model-id-b", default=DEFAULT_MODEL_ID_B,
                        help=f"Model ID for Group B (default: {DEFAULT_MODEL_ID_B})")

    # URL overrides (for non-standard setups)
    parser.add_argument("--scheduler-a-url", default=None,
                        help="Override Scheduler A URL (default: http://<host>:<scheduler-a-port>)")
    parser.add_argument("--scheduler-b-url", default=None,
                        help="Override Scheduler B URL (default: http://<host>:<scheduler-b-port>)")
    parser.add_argument("--planner-url", default=None,
                        help="Override Planner URL (default: http://<host>:<planner-port>)")

    # Dry run
    parser.add_argument("--dry-run", action="store_true",
                        help="Print payload without sending request")

    args = parser.parse_args()

    # Build URLs from host and ports if not explicitly provided
    scheduler_a_url = args.scheduler_a_url or f"http://{args.host}:{args.scheduler_a_port}"
    scheduler_b_url = args.scheduler_b_url or f"http://{args.host}:{args.scheduler_b_port}"
    planner_url = args.planner_url or f"http://{args.host}:{args.planner_port}"

    # Build instances (only scheduler-registered instances)
    instances, instance_scheduler_map, actual_n1, actual_n2 = build_instances(
        n1=args.n1,
        n2=args.n2,
        host=args.host,
        port_a_start=args.port_a_start,
        port_b_start=args.port_b_start,
        model_id_a=args.model_id_a,
        model_id_b=args.model_id_b,
        scheduler_a_url=scheduler_a_url,
        scheduler_b_url=scheduler_b_url,
    )

    if not instances:
        print("No instances provided for redeploy.", file=sys.stderr)
        sys.exit(1)

    # Build payload using actual counts of scheduler-registered instances
    payload = build_payload(
        instances,
        instance_scheduler_map,
        scheduler_a_url=scheduler_a_url,
        scheduler_b_url=scheduler_b_url,
        model_id_a=args.model_id_a,
        model_id_b=args.model_id_b,
        n1=actual_n1,
        n2=actual_n2,
    )

    # Print configuration summary
    print("=" * 60)
    print("Redeploy Configuration (Type2 Deep Research - Simulation Mode)")
    print("=" * 60)
    print(f"  Host:          {args.host}")
    print(f"  Total instances:  {args.n1 + args.n2} (N1={args.n1}, N2={args.n2})")
    print(f"  Scheduler-registered instances: {actual_n1 + actual_n2}")
    print(f"    Group A:     {actual_n1} of {args.n1} ({args.model_id_a} - for A/Merge tasks)")
    print(f"    Group B:     {actual_n2} of {args.n2} ({args.model_id_b} - for B1/B2 tasks)")
    print(f"  Scheduler A:   {scheduler_a_url}")
    print(f"  Scheduler B:   {scheduler_b_url}")
    print(f"  Planner:       {planner_url}")
    print("=" * 60)

    if args.dry_run:
        import json
        print("\n[DRY RUN] Payload that would be sent:")
        print(json.dumps(payload, indent=2))
        return

    # Send request
    url = f"{planner_url}/deploy/migration"
    print(f"\nPosting redeploy payload to {url} ...")

    try:
        resp = requests.post(url, json=payload, timeout=30)
        print(f"Planner responded: {resp.status_code}")
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to planner: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
