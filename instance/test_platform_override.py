#!/usr/bin/env python3
"""
Test script for platform information override functionality.

This script tests:
1. Auto-detection (default behavior)
2. Environment variable overrides
3. CLI argument overrides
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path


def test_auto_detection():
    """Test default auto-detection behavior."""
    print("\n=== Test 1: Auto-detection (default) ===")

    # Start instance in background
    process = subprocess.Popen(
        ["uv", "run", "sinstance", "start", "--port", "5001"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ}
    )

    # Wait for service to start
    time.sleep(3)

    try:
        # Query the /info endpoint
        response = requests.get("http://localhost:5001/info")
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Platform Info:")
        print(f"  Software Name: {data['instance']['software_name']}")
        print(f"  Software Version: {data['instance']['software_version']}")
        print(f"  Hardware Name: {data['instance']['hardware_name']}")

        assert data['instance']['software_name'] is not None, "Software name should be auto-detected"
        print("✓ Auto-detection works")

    finally:
        process.terminate()
        process.wait(timeout=5)


def test_env_variable_override():
    """Test environment variable overrides."""
    print("\n=== Test 2: Environment variable overrides ===")

    # Set environment variables
    env = {
        **os.environ,
        "INSTANCE_PLATFORM_SOFTWARE_NAME": "TestOS",
        "INSTANCE_PLATFORM_SOFTWARE_VERSION": "1.0.0",
        "INSTANCE_PLATFORM_HARDWARE_NAME": "TestGPU",
    }

    # Start instance in background
    process = subprocess.Popen(
        ["uv", "run", "sinstance", "start", "--port", "5002"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    # Wait for service to start
    time.sleep(3)

    try:
        # Query the /info endpoint
        response = requests.get("http://localhost:5002/info")
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Platform Info:")
        print(f"  Software Name: {data['instance']['software_name']}")
        print(f"  Software Version: {data['instance']['software_version']}")
        print(f"  Hardware Name: {data['instance']['hardware_name']}")

        assert data['instance']['software_name'] == "TestOS", "Software name should be overridden"
        assert data['instance']['software_version'] == "1.0.0", "Software version should be overridden"
        assert data['instance']['hardware_name'] == "TestGPU", "Hardware name should be overridden"
        print("✓ Environment variable overrides work")

    finally:
        process.terminate()
        process.wait(timeout=5)


def test_cli_argument_override():
    """Test CLI argument overrides."""
    print("\n=== Test 3: CLI argument overrides ===")

    # Start instance with CLI arguments
    process = subprocess.Popen(
        [
            "uv", "run", "sinstance", "start",
            "--port", "5003",
            "--platform-software-name", "CLIOS",
            "--platform-software-version", "2.0.0",
            "--platform-hardware-name", "CLIGPU",
        ],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ}
    )

    # Wait for service to start
    time.sleep(3)

    try:
        # Query the /info endpoint
        response = requests.get("http://localhost:5003/info")
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Platform Info:")
        print(f"  Software Name: {data['instance']['software_name']}")
        print(f"  Software Version: {data['instance']['software_version']}")
        print(f"  Hardware Name: {data['instance']['hardware_name']}")

        assert data['instance']['software_name'] == "CLIOS", "Software name should be overridden by CLI"
        assert data['instance']['software_version'] == "2.0.0", "Software version should be overridden by CLI"
        assert data['instance']['hardware_name'] == "CLIGPU", "Hardware name should be overridden by CLI"
        print("✓ CLI argument overrides work")

    finally:
        process.terminate()
        process.wait(timeout=5)


def test_priority_order():
    """Test that CLI arguments take precedence over environment variables."""
    print("\n=== Test 4: Priority order (CLI > ENV) ===")

    # Set environment variables
    env = {
        **os.environ,
        "INSTANCE_PLATFORM_SOFTWARE_NAME": "EnvOS",
        "INSTANCE_PLATFORM_HARDWARE_NAME": "EnvGPU",
    }

    # Start instance with both env vars and CLI args
    process = subprocess.Popen(
        [
            "uv", "run", "sinstance", "start",
            "--port", "5004",
            "--platform-software-name", "CLIOS",  # This should win
            # No hardware override in CLI, should use env var
        ],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    # Wait for service to start
    time.sleep(3)

    try:
        # Query the /info endpoint
        response = requests.get("http://localhost:5004/info")
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Platform Info:")
        print(f"  Software Name: {data['instance']['software_name']}")
        print(f"  Hardware Name: {data['instance']['hardware_name']}")

        assert data['instance']['software_name'] == "CLIOS", "CLI should override ENV"
        assert data['instance']['hardware_name'] == "EnvGPU", "ENV should be used when CLI not provided"
        print("✓ Priority order works correctly")

    finally:
        process.terminate()
        process.wait(timeout=5)


def main():
    """Run all tests."""
    print("Platform Override Functionality Tests")
    print("=" * 50)

    try:
        test_auto_detection()
        test_env_variable_override()
        test_cli_argument_override()
        test_priority_order()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
