#!/usr/bin/env python3
"""
Initial redeploy helper for Type2 Deep Research Workflow (Real Mode).
Builds a basic planner migration payload and posts to /deploy/migration.

Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
  - Model A (llm_service_small_model) -> Scheduler A (for A and Merge tasks)
  - Model B (llm_service_large_model) -> Scheduler B (for B1/B2 tasks)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

import requests


# Group A Hosts (llm_service_small_model for A and Merge tasks)
GROUP_A_HOSTS = [
    "29.209.106.237",
    "29.209.114.56",
    "29.209.114.241",
    "29.209.112.177",
    "29.209.113.235",
    "29.209.105.60"
]

# Group B Hosts (llm_service_large_model for B1/B2 tasks)
GROUP_B_HOSTS = [
    "29.209.113.166",
    "29.209.113.176",
    "29.209.113.169",
    "29.209.112.74",
    "29.209.115.174",
    "29.209.113.156"
]

INSTANCE_PORT_LIST = [8200, 8201, 8202, 8203]

# Default scheduler URLs
DEFAULT_SCHEDULER_A_URL = "http://29.209.114.51:8100"
DEFAULT_SCHEDULER_B_URL = "http://29.209.113.228:8100"
DEFAULT_PLANNER_URL = "http://29.209.114.166:8100"


@dataclass
class InstanceInfo:
    endpoint: str
    current_model: str


def build_instances(
    n1: int,
    n2: int,
) -> tuple[List[InstanceInfo], Dict[str, str]]:
    """
    Build instance list for Type2 Deep Research workflow.

    Group A: llm_service_small_model (for A and Merge tasks) -> Scheduler A
    Group B: llm_service_large_model (for B1/B2 tasks) -> Scheduler B
    """
    instances: List[InstanceInfo] = []
    instance_scheduler_map: Dict[str, str] = {}

    for host in GROUP_A_HOSTS:
        for port in INSTANCE_PORT_LIST:
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model="llm_service_small_model"))
            instance_scheduler_map[endpoint] = DEFAULT_SCHEDULER_A_URL

    for host in GROUP_B_HOSTS:
        for port in INSTANCE_PORT_LIST:
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model="llm_service_large_model"))
            instance_scheduler_map[endpoint] = DEFAULT_SCHEDULER_B_URL

    return instances, instance_scheduler_map


def build_payload(
    instances: List[InstanceInfo],
    instance_scheduler_map: Dict[str, str],
    scheduler_a_url: str,
    scheduler_b_url: str,
) -> Dict:
    """Build the migration payload for the planner service."""
    total_inst = len(instances)
    instance_a_num = len(GROUP_A_HOSTS) * len(INSTANCE_PORT_LIST)
    instance_b_num = len(GROUP_B_HOSTS) * len(INSTANCE_PORT_LIST)

    # Capacity matrix: each instance can host either model
    # [capacity_for_model_a, capacity_for_model_b]
    B = [[5.0, 1.0] for _ in range(total_inst)]

    planner_input = {
        "M": instance_a_num + instance_b_num,
        "N": 2,
        "B": B,
        "initial": [0] * instance_a_num + [1] * instance_b_num,
        "a": 1.0,
        "target": [1.0, 1.0],
        "algorithm": "simulated_annealing",
        "objective_method": "ratio_difference",
    }

    scheduler_mapping = {
        "llm_service_small_model": scheduler_a_url,
        "llm_service_large_model": scheduler_b_url,
    }

    return {
        "instances": [asdict(inst) for inst in instances],
        "planner_input": planner_input,
        "scheduler_mapping": scheduler_mapping,
        "instance_scheduler_mapping": instance_scheduler_map,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Initial redeploy via planner /deploy/migration (Type2 Deep Research - Real Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Type2 Workflow: A -> n*B1 -> n*B2 -> Merge
  - Model A (llm_service_small_model) -> Scheduler A (for A and Merge tasks)
  - Model B (llm_service_large_model) -> Scheduler B (for B1/B2 tasks)

Examples:
  # Use defaults
  python type2_redeploy_real.py

  # Custom scheduler URLs
  python type2_redeploy_real.py --scheduler-a-url http://192.168.1.100:8100
        """,
    )
    parser.add_argument("--scheduler-a-url", default=DEFAULT_SCHEDULER_A_URL,
                        help=f"Scheduler A URL (default: {DEFAULT_SCHEDULER_A_URL})")
    parser.add_argument("--scheduler-b-url", default=DEFAULT_SCHEDULER_B_URL,
                        help=f"Scheduler B URL (default: {DEFAULT_SCHEDULER_B_URL})")
    parser.add_argument("--planner-url", default=DEFAULT_PLANNER_URL,
                        help=f"Planner URL (default: {DEFAULT_PLANNER_URL})")
    parser.add_argument("--n1", type=int, default=24,
                        help="Number of Group A instances (default: 24)")
    parser.add_argument("--n2", type=int, default=24,
                        help="Number of Group B instances (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print payload without sending request")

    args = parser.parse_args()

    instances, instance_scheduler_map = build_instances(
        n1=args.n1,
        n2=args.n2,
    )

    if not instances:
        print("No instances provided for redeploy.", file=sys.stderr)
        sys.exit(1)

    payload = build_payload(
        instances,
        instance_scheduler_map,
        scheduler_a_url=args.scheduler_a_url,
        scheduler_b_url=args.scheduler_b_url,
    )

    # Print configuration summary
    print("=" * 60)
    print("Redeploy Configuration (Type2 Deep Research - Real Mode)")
    print("=" * 60)
    print(f"  Group A hosts: {len(GROUP_A_HOSTS)} hosts x {len(INSTANCE_PORT_LIST)} ports = {len(GROUP_A_HOSTS) * len(INSTANCE_PORT_LIST)} instances")
    print(f"    Model: llm_service_small_model (for A and Merge tasks)")
    print(f"    Scheduler: {args.scheduler_a_url}")
    print(f"  Group B hosts: {len(GROUP_B_HOSTS)} hosts x {len(INSTANCE_PORT_LIST)} ports = {len(GROUP_B_HOSTS) * len(INSTANCE_PORT_LIST)} instances")
    print(f"    Model: llm_service_large_model (for B1/B2 tasks)")
    print(f"    Scheduler: {args.scheduler_b_url}")
    print(f"  Planner: {args.planner_url}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Payload that would be sent:")
        print(json.dumps(payload, indent=2))
        return

    url = f"{args.planner_url}/deploy/migration"
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
