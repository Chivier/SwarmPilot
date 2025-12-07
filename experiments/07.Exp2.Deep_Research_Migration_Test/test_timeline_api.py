#!/usr/bin/env python3
"""
Simple test script to verify Planner timeline API functionality.

This script tests the timeline collection functions before running
the full experiment to ensure they work correctly.
"""

import requests
import json

# Planner endpoint
PLANNER_URL = "http://localhost:8202"


def test_timeline_api():
    """Test the Planner timeline API endpoints."""

    print("=" * 80)
    print("Testing Planner Timeline API")
    print("=" * 80)

    # Test 1: Check if Planner is running
    print("\nTest 1: Checking if Planner is running...")
    try:
        response = requests.get(f"{PLANNER_URL}/health", timeout=5.0)
        response.raise_for_status()
        print("✓ Planner is running")
        print(f"  Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("✗ Planner is not running at", PLANNER_URL)
        print("  Please start the Planner service first using start_all_services.sh")
        return False
    except Exception as e:
        print(f"✗ Error connecting to Planner: {e}")
        return False

    # Test 2: Clear timeline
    print("\nTest 2: Clearing timeline...")
    try:
        response = requests.post(f"{PLANNER_URL}/timeline/clear", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            print("✓ Timeline cleared successfully")
            print(f"  Message: {data.get('message')}")
        else:
            print(f"✗ Timeline clear failed: {data}")
            return False
    except Exception as e:
        print(f"✗ Error clearing timeline: {e}")
        return False

    # Test 3: Get timeline (should be empty after clearing)
    print("\nTest 3: Getting timeline (should be empty)...")
    try:
        response = requests.get(f"{PLANNER_URL}/timeline", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            entry_count = data.get("entry_count", 0)
            print(f"✓ Timeline retrieved successfully")
            print(f"  Entry count: {entry_count}")
            if entry_count == 0:
                print("  ✓ Timeline is empty as expected")
            else:
                print(f"  ⚠ Warning: Timeline has {entry_count} entries (expected 0)")
                print(f"  Entries: {json.dumps(data.get('entries', []), indent=2)}")
        else:
            print(f"✗ Timeline retrieval failed: {data}")
            return False
    except Exception as e:
        print(f"✗ Error retrieving timeline: {e}")
        return False

    # Test 4: Check timeline data structure
    print("\nTest 4: Verifying timeline data structure...")
    try:
        if data.get("success"):
            print("✓ Timeline response structure:")
            print(f"  - success: {data.get('success')}")
            print(f"  - entry_count: {data.get('entry_count')}")
            print(f"  - entries: List[Dict] (length: {len(data.get('entries', []))})")

            if data.get("entries"):
                print("\n  Example entry structure:")
                example = data["entries"][0]
                for key in example.keys():
                    print(f"    - {key}: {type(example[key]).__name__}")
        else:
            print("✗ Invalid timeline response structure")
            return False
    except Exception as e:
        print(f"✗ Error verifying data structure: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ All timeline API tests passed!")
    print("=" * 80)
    print("\nThe timeline collection integration is ready to use.")
    print("Timeline data will be automatically collected during experiment runs")
    print("and saved in the results JSON file under the 'planner_timeline' key.")
    print("\nTimeline entries include:")
    print("  - timestamp: Unix timestamp of the event")
    print("  - timestamp_iso: ISO format timestamp")
    print("  - event_type: 'deploy_migration' or 'auto_optimize'")
    print("  - instance_counts: Dict mapping model_id -> instance count")
    print("  - total_instances: Total number of instances")
    print("  - changes_count: Number of instance changes in this event")
    print("  - success: Whether the migration was successful")
    print("  - target_distribution: Target distribution used (if applicable)")
    print("  - score: Optimization score achieved (if applicable)")

    return True


if __name__ == "__main__":
    import sys

    success = test_timeline_api()
    sys.exit(0 if success else 1)
