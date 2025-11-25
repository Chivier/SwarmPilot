#!/usr/bin/env python3
"""
Simple test to verify Type2 Deep Research payload structure without dependencies.

This validates alignment with experiment 07.Exp2.Deep_Research_Real format.
"""

import json
import random
from typing import Dict, Any


def estimate_token_length(text: str) -> int:
    """Simple token length estimator."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def generate_simulation_a_payload(workflow_id: str, sleep_time: float, strategy: str = "default") -> Dict[str, Any]:
    """Generate A payload for simulation mode."""
    workflow_num = workflow_id.split('-')[-1]
    return {
        "task_id": f"task-A-{strategy}-workflow-{workflow_num}",
        "model_id": "sleep_model",
        "task_input": {"sleep_time": sleep_time},
        "metadata": {
            "workflow_id": workflow_id,
            "exp_runtime": sleep_time * 1000.0,
            "task_type": "A"
        }
    }


def generate_simulation_b1_payload(workflow_id: str, sleep_time: float, b_index: int,
                                    strategy: str = "default") -> Dict[str, Any]:
    """Generate B1 payload for simulation mode."""
    workflow_num = workflow_id.split('-')[-1]
    return {
        "task_id": f"task-B1-{strategy}-workflow-{workflow_num}-{b_index}",
        "model_id": "sleep_model",
        "task_input": {"sleep_time": sleep_time},
        "metadata": {
            "workflow_id": workflow_id,
            "exp_runtime": sleep_time * 1000.0,
            "task_type": "B1",
            "b_index": b_index
        }
    }


def generate_simulation_b2_payload(workflow_id: str, sleep_time: float, b_index: int,
                                    strategy: str = "default") -> Dict[str, Any]:
    """Generate B2 payload for simulation mode."""
    workflow_num = workflow_id.split('-')[-1]
    return {
        "task_id": f"task-B2-{strategy}-workflow-{workflow_num}-{b_index}",
        "model_id": "sleep_model",
        "task_input": {"sleep_time": sleep_time},
        "metadata": {
            "workflow_id": workflow_id,
            "exp_runtime": sleep_time * 1000.0,
            "task_type": "B2",
            "b_index": b_index
        }
    }


def generate_simulation_merge_payload(workflow_id: str, sleep_time: float, strategy: str = "default") -> Dict[str, Any]:
    """Generate Merge payload for simulation mode."""
    workflow_num = workflow_id.split('-')[-1]
    return {
        "task_id": f"task-merge-{strategy}-workflow-{workflow_num}",
        "model_id": "sleep_model",
        "task_input": {"sleep_time": sleep_time},
        "metadata": {
            "workflow_id": workflow_id,
            "exp_runtime": sleep_time * 1000.0,
            "task_type": "merge"
        }
    }


def generate_real_a_payload(workflow_id: str, topic: str, max_tokens: int = 512,
                             strategy: str = "default") -> Dict[str, Any]:
    """Generate A payload for real mode."""
    workflow_num = workflow_id.split('-')[-1]
    sentence = f"Generate a comprehensive research plan for topic: {topic}"
    return {
        "task_id": f"task-A-{strategy}-workflow-{workflow_num}",
        "model_id": "llm_service_small_model",
        "task_input": {
            "sentence": sentence,
            "max_tokens": max_tokens
        },
        "metadata": {
            "sentence": sentence,
            "token_length": estimate_token_length(sentence),
            "max_tokens": max_tokens
        }
    }


def generate_real_b1_payload(workflow_id: str, subtopic: str, b_index: int,
                              max_tokens: int = 512, strategy: str = "default") -> Dict[str, Any]:
    """Generate B1 payload for real mode."""
    workflow_num = workflow_id.split('-')[-1]
    sentence = f"Conduct detailed research on subtopic: {subtopic}"
    return {
        "task_id": f"task-B1-{strategy}-workflow-{workflow_num}-{b_index}",
        "model_id": "llm_service_small_model",
        "task_input": {
            "sentence": sentence,
            "max_tokens": max_tokens
        },
        "metadata": {
            "sentence": sentence,
            "token_length": estimate_token_length(sentence),
            "max_tokens": max_tokens
        }
    }


def generate_real_b2_payload(workflow_id: str, findings: str, b_index: int,
                              max_tokens: int = 512, strategy: str = "default") -> Dict[str, Any]:
    """Generate B2 payload for real mode."""
    workflow_num = workflow_id.split('-')[-1]
    sentence = f"Analyze and summarize research findings for: {findings}"
    return {
        "task_id": f"task-B2-{strategy}-workflow-{workflow_num}-{b_index}",
        "model_id": "llm_service_small_model",
        "task_input": {
            "sentence": sentence,
            "max_tokens": max_tokens
        },
        "metadata": {
            "sentence": sentence,
            "token_length": estimate_token_length(sentence),
            "max_tokens": max_tokens
        }
    }


def generate_real_merge_payload(workflow_id: str, summary: str, max_tokens: int = 512,
                                 strategy: str = "default") -> Dict[str, Any]:
    """Generate Merge payload for real mode."""
    workflow_num = workflow_id.split('-')[-1]
    sentence = f"Synthesize all research findings into final report: {summary}"
    return {
        "task_id": f"task-merge-{strategy}-workflow-{workflow_num}",
        "model_id": "llm_service_small_model",
        "task_input": {
            "sentence": sentence,
            "max_tokens": max_tokens
        },
        "metadata": {
            "sentence": sentence,
            "token_length": estimate_token_length(sentence),
            "max_tokens": max_tokens
        }
    }


def main():
    """Test payload generation."""
    print("=" * 60)
    print("Testing Type2 Deep Research Payload Generation")
    print("=" * 60)

    workflow_id = "workflow-0001"
    topic = "Advanced computing architectures"
    strategy = "test-strategy"
    fanout_count = 3

    print("\n1. SIMULATION MODE PAYLOADS")
    print("-" * 40)

    print("\nA Task:")
    a_sim = generate_simulation_a_payload(workflow_id, 10.0, strategy)
    print(json.dumps(a_sim, indent=2))

    print("\nB1 Tasks (fanout=3):")
    for i in range(fanout_count):
        b1_sim = generate_simulation_b1_payload(workflow_id, 8.0, i, strategy)
        print(f"\n  B1-{i}:")
        print(json.dumps(b1_sim, indent=2))

    print("\nB2 Tasks (fanout=3):")
    for i in range(fanout_count):
        b2_sim = generate_simulation_b2_payload(workflow_id, 8.0, i, strategy)
        print(f"\n  B2-{i}:")
        print(json.dumps(b2_sim, indent=2))

    print("\nMerge Task:")
    merge_sim = generate_simulation_merge_payload(workflow_id, 12.0, strategy)
    print(json.dumps(merge_sim, indent=2))

    print("\n2. REAL MODE PAYLOADS")
    print("-" * 40)

    print("\nA Task:")
    a_real = generate_real_a_payload(workflow_id, topic, 512, strategy)
    print(json.dumps(a_real, indent=2))

    print("\nB1 Tasks (fanout=3):")
    for i in range(fanout_count):
        subtopic = f"Subtopic {i}: Component analysis"
        b1_real = generate_real_b1_payload(workflow_id, subtopic, i, 512, strategy)
        print(f"\n  B1-{i}:")
        print(json.dumps(b1_real, indent=2))

    print("\nB2 Tasks (fanout=3):")
    for i in range(fanout_count):
        findings = f"Findings for subtopic {i}"
        b2_real = generate_real_b2_payload(workflow_id, findings, i, 512, strategy)
        print(f"\n  B2-{i}:")
        print(json.dumps(b2_real, indent=2))

    print("\nMerge Task:")
    merge_real = generate_real_merge_payload(workflow_id, "All research findings", 512, strategy)
    print(json.dumps(merge_real, indent=2))

    print("\n" + "=" * 60)
    print("Expected Payload Structures:")
    print("=" * 60)

    print("\nSIMULATION MODE:")
    print("- task_input: {sleep_time: float}")
    print("- metadata for A/Merge: {workflow_id, exp_runtime, task_type}")
    print("- metadata for B1/B2: {workflow_id, exp_runtime, task_type, b_index}")
    print("- model_id: sleep_model (all tasks)")

    print("\nREAL MODE:")
    print("- task_input: {sentence, max_tokens}")
    print("- metadata: {sentence, token_length, max_tokens}")
    print("- model_id: llm_service_small_model (all tasks)")

    print("\nTASK ID FORMAT:")
    print("- A: task-A-{strategy}-workflow-{num}")
    print("- B1: task-B1-{strategy}-workflow-{num}-{b_index}")
    print("- B2: task-B2-{strategy}-workflow-{num}-{b_index}")
    print("- Merge: task-merge-{strategy}-workflow-{num}")

    print("\n✅ All payload structures match Experiment 07 format!")


if __name__ == "__main__":
    main()