# SwarmPilot

A distributed task scheduling and execution system with dynamic load balancing, intelligent task allocation, and runtime prediction.

## Architecture

SwarmPilot consists of four core microservices:

- **Scheduler**: Task orchestration and instance management
- **Instance**: Task execution nodes running containerized model services
- **Predictor**: MLP-based runtime prediction service
- **Planner**: Deployment optimization using linear programming

## Quick Start

### Prerequisites

- Python >= 3.11
- uv (recommended) or pip

### Installation

Install all services:

```bash
# Development mode
uv sync

# Or using pip (when published to PyPI)
pip install swarmpilot
```

### Usage

Start services using CLI commands:

```bash
# Start scheduler
sscheduler start

# Start instance service
sinstance start

# Start predictor service
spredictor start

# Start planner service
splanner start
```

For help with any command:

```bash
sscheduler --help
sinstance --help
spredictor --help
splanner --help
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific service
cd scheduler
uv run pytest
```

## Advanced Installation

### Install specific services

```bash
pip install swarmpilot[scheduler]
pip install swarmpilot[instance]
pip install swarmpilot[predictor]
pip install swarmpilot[planner]
```

### Development installation

```bash
uv sync --all-extras
```

### Standalone subpackage installation

```bash
cd scheduler
pip install .
```

## Project Structure

```
swarmpilot-refresh/
├── scheduler/          # Scheduling service
├── instance/           # Instance service
├── predictor/          # Prediction service
├── planner/            # Planning service
└── pyproject.toml      # Metapackage configuration
```

## License

TBD

## Contributing

TBD
