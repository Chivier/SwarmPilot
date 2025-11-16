# Platform Information Override Guide

## Overview

The Instance Service now supports overriding auto-detected platform information (OS name, OS version, hardware name). This is useful for:

- **Testing**: Simulate different environments without changing hardware
- **Multi-environment setups**: Standardize platform reporting across diverse infrastructure
- **Custom reporting**: Override detected values for organizational requirements
- **Scheduler integration**: Provide accurate platform info when auto-detection fails

## Implementation Details

### Architecture

Platform information is obtained with the following priority order (highest to lowest):

1. **CLI arguments** (`--platform-*` flags)
2. **Environment variables** (`INSTANCE_PLATFORM_*`)
3. **Model parameters** (from model metadata)
4. **Auto-detection** (default behavior using `platform` module and GPU detection)

### Modified Files

1. **[config.py](instance/src/config.py)**
   - Added three new optional configuration fields:
     - `platform_software_name`
     - `platform_software_version`
     - `platform_hardware_name`
   - Loaded from environment variables during initialization

2. **[scheduler_client.py](instance/src/scheduler_client.py)**
   - Updated `_get_platform_info()` method to check config overrides
   - Falls back to auto-detection if overrides not provided

3. **[api.py](instance/src/api.py)**
   - Updated `/info` endpoint to respect config overrides
   - Maintains backward compatibility with old `INSTANCE_SOFTWARE_*` env vars

4. **[cli.py](instance/src/cli.py)**
   - Added three new CLI options:
     - `--platform-software-name`
     - `--platform-software-version`
     - `--platform-hardware-name`
   - CLI options override environment variables
   - Displays active overrides on startup

## Usage Examples

### 1. Using Environment Variables

```bash
# Override all platform information
INSTANCE_PLATFORM_SOFTWARE_NAME="Linux" \
INSTANCE_PLATFORM_SOFTWARE_VERSION="5.15.0-151-generic" \
INSTANCE_PLATFORM_HARDWARE_NAME="NVIDIA GeForce RTX 4090" \
sinstance start

# Override only hardware (useful for testing CPU vs GPU)
INSTANCE_PLATFORM_HARDWARE_NAME="CPU" sinstance start

# Use .env file for persistent configuration
cat > .env << EOF
INSTANCE_PLATFORM_SOFTWARE_NAME=Ubuntu
INSTANCE_PLATFORM_SOFTWARE_VERSION=22.04
INSTANCE_PLATFORM_HARDWARE_NAME=NVIDIA A100
EOF
sinstance start
```

### 2. Using CLI Arguments

```bash
# Override all platform information via CLI
sinstance start \
  --platform-software-name "Linux" \
  --platform-software-version "5.15.0-151-generic" \
  --platform-hardware-name "NVIDIA GeForce RTX 4090"

# Override only specific values
sinstance start --platform-hardware-name "CPU"

# Combine with other CLI options
sinstance start \
  --port 8000 \
  --log-level DEBUG \
  --platform-hardware-name "NVIDIA A100"
```

### 3. Mixed Configuration (Priority Demonstration)

```bash
# Environment variable sets base values
export INSTANCE_PLATFORM_HARDWARE_NAME="CPU"

# CLI argument overrides environment variable
sinstance start --platform-hardware-name "NVIDIA RTX 4090"
# Result: Hardware will be "NVIDIA RTX 4090" (CLI wins)

# Partial override example
export INSTANCE_PLATFORM_SOFTWARE_NAME="CustomOS"
sinstance start --platform-hardware-name "GPU-Cluster"
# Result:
#   Software: CustomOS (from env var)
#   Hardware: GPU-Cluster (from CLI)
#   Version: <auto-detected> (no override)
```

## Verification

### Check Platform Information

Query the `/info` endpoint to see the effective platform information:

```bash
# Start instance with overrides
sinstance start \
  --port 5000 \
  --platform-software-name "TestOS" \
  --platform-hardware-name "TestGPU"

# Query info endpoint
curl http://localhost:5000/info | jq '.instance | {software_name, software_version, hardware_name}'
```

Expected output:
```json
{
  "software_name": "TestOS",
  "software_version": "5.15.0-151-generic",
  "hardware_name": "TestGPU"
}
```

### Run Automated Tests

```bash
# Run the comprehensive test suite
uv run python test_platform_override.py
```

Tests verify:
- Auto-detection (default behavior)
- Environment variable overrides
- CLI argument overrides
- Priority order (CLI > ENV > auto-detection)

## Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `INSTANCE_PLATFORM_SOFTWARE_NAME` | Override OS name | `Linux`, `Darwin`, `Windows` |
| `INSTANCE_PLATFORM_SOFTWARE_VERSION` | Override OS version | `5.15.0-151-generic`, `22.04` |
| `INSTANCE_PLATFORM_HARDWARE_NAME` | Override hardware name | `NVIDIA GeForce RTX 4090`, `CPU` |

### CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--platform-software-name` | Override OS name | `--platform-software-name "Linux"` |
| `--platform-software-version` | Override OS version | `--platform-software-version "22.04"` |
| `--platform-hardware-name` | Override hardware name | `--platform-hardware-name "NVIDIA A100"` |

## Use Cases

### Testing Multi-Platform Workflows

```bash
# Simulate different platforms for testing
sinstance start --port 5001 --platform-hardware-name "CPU"
sinstance start --port 5002 --platform-hardware-name "NVIDIA T4"
sinstance start --port 5003 --platform-hardware-name "NVIDIA A100"
```

### Scheduler Integration

When registering with a scheduler, override platform info to ensure accurate task routing:

```bash
# GPU instance with specific hardware info
SCHEDULER_URL=http://scheduler:5001 \
INSTANCE_PLATFORM_HARDWARE_NAME="NVIDIA GeForce RTX 4090" \
sinstance start

# CPU-only instance
SCHEDULER_URL=http://scheduler:5001 \
INSTANCE_PLATFORM_HARDWARE_NAME="CPU" \
sinstance start
```

### Container/Cloud Deployments

```dockerfile
# Dockerfile
ENV INSTANCE_PLATFORM_SOFTWARE_NAME="Linux"
ENV INSTANCE_PLATFORM_SOFTWARE_VERSION="Container"
ENV INSTANCE_PLATFORM_HARDWARE_NAME="Cloud GPU"

CMD ["sinstance", "start"]
```

## Backward Compatibility

The implementation maintains full backward compatibility:

- **Old behavior**: Auto-detection still works by default
- **Old env vars**: `INSTANCE_SOFTWARE_NAME` and `INSTANCE_SOFTWARE_VERSION` are still supported (deprecated)
- **Model parameters**: Software hints from model metadata are still respected
- **Priority order**: New overrides have highest priority, ensuring predictable behavior

## Troubleshooting

### Override Not Taking Effect

1. **Check startup logs**: Instance displays active overrides on startup
   ```
   Starting Instance Service on 0.0.0.0:5000
   Instance ID: instance-default
   Log Level: INFO
   Platform Overrides: Software: TestOS, Hardware: TestGPU
   ```

2. **Verify priority**: CLI options override environment variables
   ```bash
   # This will use CLI value, not env var
   INSTANCE_PLATFORM_HARDWARE_NAME="ENV" \
   sinstance start --platform-hardware-name "CLI"
   ```

3. **Query /info endpoint**: Verify effective platform info
   ```bash
   curl http://localhost:5000/info | jq '.instance.hardware_name'
   ```

### Auto-Detection Still Active

If overrides are ignored:

1. Check that environment variables are set in the correct format
2. Ensure CLI options use correct syntax (e.g., `--platform-hardware-name`, not `--hardware-name`)
3. Verify config loading by checking startup logs

## Future Enhancements

Potential improvements for future versions:

- Support for configuration file (YAML/JSON)
- Runtime platform info updates via API
- Platform capability detection (GPU memory, CPU cores, etc.)
- Platform-specific scheduling hints
- Historical platform info tracking

## Related Documentation

- [Instance Service README](README.md)
- [Configuration Guide](README.md#configuration)
- [Scheduler Integration](README.md#scheduler-integration)
- [Test Suite](test_platform_override.py)

---

**Version**: 1.0
**Last Updated**: 2025-11-16
**Author**: Instance Service Team
