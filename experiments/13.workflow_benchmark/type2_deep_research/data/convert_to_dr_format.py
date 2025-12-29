#!/usr/bin/env python3
"""
Convert training data JSON files to dr_* format.

Source files:
- training_data_llm_service_large_model.json -> dr_boot.json, dr_summary_dict.json
- training_data_llm_service_small_model.json -> dr_query.json, dr_criteria.json
"""

import json
from pathlib import Path


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data, filepath: Path) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def extract_runtime_by_task_type(samples: list, task_type: str) -> list:
    """Extract runtime_ms values for samples matching the given task_type."""
    return [s['runtime_ms'] for s in samples if s.get('task_type') == task_type]


def create_summary_dict(samples: list, n_groups: int = 11) -> dict:
    """
    Create summary dictionary by grouping samples evenly by token_length.

    Args:
        samples: List of sample dictionaries with 'token_length' and 'runtime_ms'
        n_groups: Number of groups (default 11, keys "5" to "15")

    Returns:
        Dictionary with string keys ("5" to "15") mapping to lists of runtime_ms values
    """
    # Filter summary samples
    summary_samples = [s for s in samples if s.get('task_type') == 'summary']

    if not summary_samples:
        return {}

    # Sort by token_length
    sorted_samples = sorted(summary_samples, key=lambda x: x['token_length'])

    # Calculate group sizes for even distribution
    group_size = len(sorted_samples) // n_groups
    remainder = len(sorted_samples) % n_groups

    result = {}
    idx = 0
    for i in range(n_groups):
        # First 'remainder' groups get one extra sample
        size = group_size + (1 if i < remainder else 0)
        group_samples = sorted_samples[idx:idx + size]
        key = str(i + 5)  # Keys from "5" to "15"
        result[key] = [s['runtime_ms'] for s in group_samples]
        idx += size

    return result


def main():
    # Define paths
    data_dir = Path(__file__).parent
    output_dir = data_dir / 'data_new'

    large_model_file = data_dir / 'training_data_llm_service_large_model.json'
    small_model_file = data_dir / 'training_data_llm_service_small_model.json'

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Process large_model file (boot, summary)
    print(f"Loading {large_model_file.name}...")
    large_data = load_json(large_model_file)
    large_samples = large_data['samples']
    print(f"  Found {len(large_samples)} samples")

    # Extract boot runtime values
    boot_runtimes = extract_runtime_by_task_type(large_samples, 'boot')
    print(f"  Boot samples: {len(boot_runtimes)}")
    save_json(boot_runtimes, output_dir / 'dr_boot.json')

    # Create summary dictionary (grouped by token_length)
    summary_dict = create_summary_dict(large_samples, n_groups=11)
    total_summary = sum(len(v) for v in summary_dict.values())
    print(f"  Summary samples: {total_summary} (in {len(summary_dict)} groups)")
    save_json(summary_dict, output_dir / 'dr_summary_dict.json')

    # Process small_model file (criteria, query)
    print(f"\nLoading {small_model_file.name}...")
    small_data = load_json(small_model_file)
    small_samples = small_data['samples']
    print(f"  Found {len(small_samples)} samples")

    # Extract query runtime values
    query_runtimes = extract_runtime_by_task_type(small_samples, 'query')
    print(f"  Query samples: {len(query_runtimes)}")
    save_json(query_runtimes, output_dir / 'dr_query.json')

    # Extract criteria runtime values
    criteria_runtimes = extract_runtime_by_task_type(small_samples, 'criteria')
    print(f"  Criteria samples: {len(criteria_runtimes)}")
    save_json(criteria_runtimes, output_dir / 'dr_criteria.json')

    print("\nConversion complete!")
    print(f"Output files in: {output_dir}")


if __name__ == '__main__':
    main()
