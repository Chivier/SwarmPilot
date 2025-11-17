#!/usr/bin/env python3
"""
Test script to validate configuration file loading and validation.
"""

import json
import sys
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path to import from collect_training_data
sys.path.insert(0, str(Path(__file__).parent))

from collect_training_data import load_config


def test_valid_config():
    """Test loading a valid configuration file."""
    config_data = {
        "dataset": "data/dataset.jsonl",
        "model_id": "llama-7b",
        "instances": [
            {
                "url": "http://localhost:8001",
                "hardware_name": "NVIDIA H20",
                "software_name": "sglang",
                "software_version": "1.0.0"
            }
        ],
        "predictor": {
            "url": "http://localhost:9000"
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        # Verify required fields
        assert config['dataset'] == "data/dataset.jsonl"
        assert config['model_id'] == "llama-7b"
        assert len(config['instances']) == 1
        assert config['predictor']['url'] == "http://localhost:9000"

        # Verify defaults
        assert config['prediction_types'] == ['expect_error', 'quantile']
        assert config['max_samples'] is None
        assert config['training_config']['quantiles'] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        assert config['execution']['timeout'] == 300.0
        assert config['execution']['max_concurrent_requests'] == 10

        print("✓ Valid config test passed")
    finally:
        Path(temp_path).unlink()


def test_missing_required_field():
    """Test that missing required field raises error."""
    config_data = {
        "dataset": "data/dataset.jsonl",
        "model_id": "llama-7b",
        # Missing 'instances' and 'predictor'
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Missing required field"):
            config = load_config(temp_path)
        print("✓ Missing required field test passed")
    finally:
        Path(temp_path).unlink()


def test_custom_config_overrides():
    """Test that custom config values override defaults."""
    config_data = {
        "dataset": "data/dataset.jsonl",
        "model_id": "llama-7b",
        "instances": [{"url": "http://localhost:8001", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "1.0.0"}],
        "predictor": {"url": "http://localhost:9000"},
        "prediction_types": ["quantile"],
        "max_samples": 100,
        "training_config": {
            "quantiles": [0.5, 0.9, 0.99],
            "epochs": 1000
        },
        "execution": {
            "timeout": 600.0,
            "max_concurrent_requests": 20
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        config = load_config(temp_path)

        # Verify custom values
        assert config['prediction_types'] == ['quantile']
        assert config['max_samples'] == 100
        assert config['training_config']['quantiles'] == [0.5, 0.9, 0.99]
        assert config['training_config']['epochs'] == 1000
        assert config['execution']['timeout'] == 600.0
        assert config['execution']['max_concurrent_requests'] == 20

        print("✓ Custom config overrides test passed")
    finally:
        Path(temp_path).unlink()


def test_example_config_file():
    """Test that config.example.json is valid."""
    example_path = Path(__file__).parent / "config.example.json"

    if not example_path.exists():
        pytest.skip("config.example.json not found")

    config = load_config(str(example_path))

    # Verify structure
    assert 'dataset' in config
    assert 'model_id' in config
    assert 'instances' in config
    assert len(config['instances']) >= 1
    assert 'predictor' in config

    print("✓ Example config file test passed")


if __name__ == "__main__":
    print("Running configuration validation tests...\n")

    tests = [
        test_valid_config,
        test_missing_required_field,
        test_custom_config_overrides,
        test_example_config_file,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()

    total = passed + failed

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {failed} test(s) failed")
        sys.exit(1)
