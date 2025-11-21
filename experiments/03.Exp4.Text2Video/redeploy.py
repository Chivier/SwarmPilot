#!/usr/bin/env python3
"""
Initial redeploy helper for Exp03 Text2Video.
Builds a basic planner migration payload and posts to /deploy/migration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List

import requests


@dataclass
class InstanceInfo:
    endpoint: str
    current_model: str


def build_instances(
    n1: int,
    n2: int,
    port_a_start: int,
    port_b_start: int,
    model_a_id: str,
    model_b_id: str,
    scheduler_a_url: str,
    scheduler_b_url: str,
    planner_url: str,
) -> tuple[List[InstanceInfo], Dict[str, str]]:
    instances: List[InstanceInfo] = []
    instance_scheduler_map: Dict[str, str] = {}

    half_a = n1 // 2
    half_b = n2 // 2

    for i in range(half_a):
        port = port_a_start + i
        endpoint = f"http://localhost:{port}"
        instances.append(InstanceInfo(endpoint=endpoint, current_model=model_a_id))
        instance_scheduler_map[endpoint] = scheduler_a_url if i < half_a else planner_url

    for i in range(half_b):
        port = port_b_start + i
        endpoint = f"http://localhost:{port}"
        instances.append(InstanceInfo(endpoint=endpoint, current_model=model_b_id))
        instance_scheduler_map[endpoint] = scheduler_b_url if i < half_b else planner_url
    
    print(f"We find {len(instances)} in our system")
    print(f"{n1} in scheduler A")
    print(f"{n2} in scheduler B")

    return instances, instance_scheduler_map


def build_payload(
    instances: List[InstanceInfo],
    instance_scheduler_map: Dict[str, str],
    scheduler_a_url: str,
    scheduler_b_url: str,
) -> Dict:
    total_inst = len(instances)
    # Simple capacity matrix: each instance can host either model equally
    B = [[3.0, 1.0] for _ in range(total_inst)]
    planner_input = {
        "M": total_inst,
        "N": 2,
        "B": B,
        "initial": [0] * total_inst,
        "a": 1.0,
        "target": [1.0, 1.0],
        "algorithm": "simulated_annealing",
        "objective_method": "ratio_difference",
    }

    scheduler_mapping = {
        instances[0].current_model: scheduler_a_url if scheduler_a_url else "",
        instances[-1].current_model: scheduler_b_url if scheduler_b_url else "",
    }

    return {
        "instances": [asdict(inst) for inst in instances],
        "planner_input": planner_input,
        "scheduler_mapping": scheduler_mapping,
        "instance_scheduler_mapping": instance_scheduler_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Initial redeploy via planner /deploy/migration")
    parser.add_argument("--scheduler-a-url", default="http://localhost:8100")
    parser.add_argument("--scheduler-b-url", default="http://localhost:8200")
    parser.add_argument("--planner-url", default="http://localhost:8202")
    parser.add_argument("--model-id-a", default="sleep_model_a")
    parser.add_argument("--model-id-b", default="sleep_model_b")
    parser.add_argument("--n1", type=int, default=4)
    parser.add_argument("--n2", type=int, default=2)
    parser.add_argument("--port-a-start", type=int, default=8210)
    parser.add_argument("--port-b-start", type=int, default=8300)
    args = parser.parse_args()

    instances, instance_scheduler_map = build_instances(
        n1=args.n1,
        n2=args.n2,
        port_a_start=args.port_a_start,
        port_b_start=args.port_b_start,
        model_a_id=args.model_id_a,
        model_b_id=args.model_id_b,
        scheduler_a_url=args.scheduler_a_url,
        scheduler_b_url=args.scheduler_b_url,
        planner_url=args.planner_url,
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

    url = args.planner_url.rstrip("/") + "/deploy/migration"
    print(f"Posting redeploy payload to {url} ...")
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Planner responded: {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
