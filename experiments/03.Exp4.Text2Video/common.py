#!/usr/bin/env python3
"""
Common utilities for continuous request mode in experiment 03.

This module provides shared functionality for continuous mode operations:
- Force clearing scheduler tasks
- Calculating makespan metrics
- Printing continuous mode summaries
"""

import logging
from typing import List, Dict


# ============================================================================
# Continuous Request Mode Functions
# ============================================================================

def force_clear_scheduler_tasks(scheduler_url: str):
    """
    Force clear all tasks from a scheduler (for continuous mode cleanup).

    Args:
        scheduler_url: Scheduler URL (e.g., http://localhost:8100)
    """
    import requests
    logger = logging.getLogger("ContinuousMode")
    try:
        response = requests.post(f"{scheduler_url}/task/clear")
        response.raise_for_status()
        logger.info(f"Force cleared tasks from {scheduler_url}")
    except Exception as e:
        logger.error(f"Failed to force clear tasks from {scheduler_url}: {e}")


def calculate_makespan(completed_workflows: List) -> Dict:
    """
    Calculate makespan and related metrics for continuous mode.

    Makespan = Time from first target workflow submission to last target workflow completion.

    Args:
        completed_workflows: List of all workflow completion events

    Returns:
        Dictionary with makespan metrics
    """
    # Filter to only target workflows (exclude warmup and non-target)
    target_workflows = [e for e in completed_workflows
                        if not e.is_warmup and e.is_target_for_stats]

    if not target_workflows:
        return {
            "makespan": 0.0,
            "first_target_submit_time": None,
            "last_target_complete_time": None,
            "num_target_workflows": 0,
            "num_total_workflows": len(completed_workflows),
            "num_warmup_workflows": sum(1 for e in completed_workflows if e.is_warmup),
            "num_overflow_workflows": sum(1 for e in completed_workflows
                                          if not e.is_warmup and not e.is_target_for_stats)
        }

    # Find first and last timestamps
    first_submit_time = min(e.a1_submit_time for e in target_workflows)
    last_complete_time = max(e.workflow_complete_time for e in target_workflows)

    makespan = last_complete_time - first_submit_time

    # Count workflows by category
    num_total = len(completed_workflows)
    num_warmup = sum(1 for e in completed_workflows if e.is_warmup)
    num_target = len(target_workflows)
    num_overflow = sum(1 for e in completed_workflows
                       if not e.is_warmup and not e.is_target_for_stats)

    return {
        "makespan": makespan,
        "first_target_submit_time": first_submit_time,
        "last_target_complete_time": last_complete_time,
        "num_target_workflows": num_target,
        "num_total_workflows": num_total,
        "num_warmup_workflows": num_warmup,
        "num_overflow_workflows": num_overflow
    }


def print_continuous_mode_summary(strategy: str, makespan_metrics: Dict,
                                  a1_metrics: Dict, a2_metrics: Dict, b_metrics: Dict,
                                  wf_metrics: Dict):
    """
    Print a summary of metrics for continuous request mode (adapted for experiment 03).

    Args:
        strategy: Strategy name
        makespan_metrics: Makespan and workflow count metrics
        a1_metrics: A1 task metrics
        a2_metrics: A2 task metrics
        b_metrics: B task metrics
        wf_metrics: Workflow metrics
    """
    print("\n" + "=" * 80)
    print(f"Continuous Request Mode Results: {strategy}")
    print("=" * 80)

    # Makespan section
    print("\nMakespan:")
    print(f"  Total time (first target → last target):  {makespan_metrics['makespan']:.2f}s")
    if makespan_metrics['first_target_submit_time']:
        from datetime import datetime
        first_time = datetime.fromtimestamp(makespan_metrics['first_target_submit_time'])
        last_time = datetime.fromtimestamp(makespan_metrics['last_target_complete_time'])
        print(f"  First target workflow submitted at:        {first_time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  Last target workflow completed at:         {last_time.strftime('%H:%M:%S.%f')[:-3]}")

    print(f"\nWorkflow Counts:")
    print(f"  Total workflows submitted:     {makespan_metrics['num_total_workflows']}")
    print(f"  Warmup workflows:              {makespan_metrics['num_warmup_workflows']}")
    print(f"  Target workflows (tracked):    {makespan_metrics['num_target_workflows']}")
    print(f"  Overflow workflows (extra):    {makespan_metrics['num_overflow_workflows']}")

    # A1 Model Tasks (Scheduler A)
    print("\nA1 Model Tasks (Scheduler A - Generate Positive Prompt):")
    print(f"  Completed:  {a1_metrics['num_completed']}")
    print(f"  Failed:     {a1_metrics['num_failed']}")
    if a1_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {a1_metrics['avg_completion_time']:.2f}s")
        print(f"  P95:        {a1_metrics['p95_completion_time']:.2f}s")
        print(f"  P99:        {a1_metrics['p99_completion_time']:.2f}s")

    # A2 Model Tasks (Scheduler A)
    print("\nA2 Model Tasks (Scheduler A - Generate Negative Prompt):")
    print(f"  Completed:  {a2_metrics['num_completed']}")
    print(f"  Failed:     {a2_metrics['num_failed']}")
    if a2_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {a2_metrics['avg_completion_time']:.2f}s")
        print(f"  P95:        {a2_metrics['p95_completion_time']:.2f}s")
        print(f"  P99:        {a2_metrics['p99_completion_time']:.2f}s")

    # B Model Tasks (Scheduler B)
    print("\nB Model Tasks (Scheduler B - Generate Video):")
    print(f"  Completed:  {b_metrics['num_completed']}")
    print(f"  Failed:     {b_metrics['num_failed']}")
    if b_metrics['avg_completion_time'] > 0:
        print(f"  Avg time:   {b_metrics['avg_completion_time']:.2f}s")
        print(f"  P95:        {b_metrics['p95_completion_time']:.2f}s")
        print(f"  P99:        {b_metrics['p99_completion_time']:.2f}s")

    # Target Workflows
    print(f"\nTarget Workflows (First {makespan_metrics['num_target_workflows']} non-warmup):")
    if wf_metrics['avg_workflow_time'] > 0:
        print(f"  Avg time:   {wf_metrics['avg_workflow_time']:.2f}s")
        print(f"  Median:     {wf_metrics['median_workflow_time']:.2f}s")
        print(f"  P95:        {wf_metrics['p95_workflow_time']:.2f}s")
        print(f"  P99:        {wf_metrics['p99_workflow_time']:.2f}s")

    # Overflow workflows note
    if makespan_metrics['num_overflow_workflows'] > 0:
        print(f"\nOverflow Workflows: {makespan_metrics['num_overflow_workflows']} workflows submitted but not tracked in statistics")

    print("=" * 80)
