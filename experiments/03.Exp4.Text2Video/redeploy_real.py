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


# LLM (Model A) Hosts
LLM_MODEL_HOSTS=[
  "29.209.106.237",
  "29.209.114.56",
  "29.209.114.241",
  "29.209.112.177",
  "29.209.113.235",
  "29.209.105.60"
]

# T2Vid (Model B) Hosts
T2VID_MODEL_HOSTS=[
  "29.209.113.166",
  "29.209.113.176",
  "29.209.113.169",
  "29.209.112.74",
  "29.209.115.174",
  "29.209.113.156"
]

INSTANCE_PORT_LIST=[8200, 8201, 8202, 8203]

@dataclass
class InstanceInfo:
    endpoint: str
    current_model: str


def build_instances(
    n1: int,
    n2: int,
) -> tuple[List[InstanceInfo], Dict[str, str]]:
    instances: List[InstanceInfo] = []
    instance_scheduler_map: Dict[str, str] = {}

    for host in LLM_MODEL_HOSTS:
        for port in INSTANCE_PORT_LIST:
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model="llm_service_small_model"))
            instance_scheduler_map[endpoint] = "http://29.209.114.51:8100"

    for host in T2VID_MODEL_HOSTS:
        for port in INSTANCE_PORT_LIST:
            endpoint = f"http://{host}:{port}"
            instances.append(InstanceInfo(endpoint=endpoint, current_model="t2vid"))
            instance_scheduler_map[endpoint] = "http://29.209.113.228:8100"
    
    return instances, instance_scheduler_map


def build_payload(
    instances: List[InstanceInfo],
    instance_scheduler_map: Dict[str, str],
    scheduler_a_url: str,
    scheduler_b_url: str,
) -> Dict:
    total_inst = len(instances)
    instance_a_num = len(LLM_MODEL_HOSTS) * len(INSTANCE_PORT_LIST)
    instance_b_num = len(T2VID_MODEL_HOSTS) * len(INSTANCE_PORT_LIST)
    # Simple capacity matrix: each instance can host either model equally
    B = [[3.0, 1.0] for _ in range(total_inst)]
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
        "t2vid": scheduler_b_url,
    }

    return {
        "instances": [asdict(inst) for inst in instances],
        "planner_input": planner_input,
        "scheduler_mapping": scheduler_mapping,
        "instance_scheduler_mapping": instance_scheduler_map,
    }


def main():
    parser = argparse.ArgumentParser(description="Initial redeploy via planner /deploy/migration")
    parser.add_argument("--scheduler-a-url", default="http://29.209.114.51:8100")
    parser.add_argument("--scheduler-b-url", default="http://29.209.113.228:8100")
    parser.add_argument("--n1", type=int, default=24)
    parser.add_argument("--n2", type=int, default=24)
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

    url = "http://29.209.114.166:8100/deploy/migration"
    print(f"Posting redeploy payload to {url} ...")
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Planner responded: {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
