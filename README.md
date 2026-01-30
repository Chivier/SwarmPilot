# SwarmPilot

A distributed task scheduling and execution system with dynamic load balancing, intelligent task allocation, and runtime prediction.

## Architecture

SwarmPilot consists of three core microservices:

- **Scheduler**: Task orchestration and instance management
- **Predictor**: MLP-based runtime prediction service
- **Planner**: Deployment optimization using linear programming

Task execution nodes are managed via **PyLet** (see [Quick Start](docs/QUICK_START.md)).

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

# Start predictor service
spredictor start

# Start planner service
splanner start
```

For help with any command:

```bash
sscheduler --help
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
├── predictor/          # Prediction service
├── planner/            # Planning service (with PyLet integration)
└── pyproject.toml      # Metapackage configuration
```

## Experiments

### Basic Experiments: Verify Installation

Exp. 01, 02, 03

### Scheduler Experiments

Exp. 09 (Universial Entrance for 04~07)  

Cluster Config: static, A = m, B = n, m << n  
Example Config: m = 8, n = 120
```
Exp. 04:   -> B
         A -> B
           -> B


Exp. 05: A -> B -> B -> B


Exp. 06:   -> B -> 
         A -> B -> A
           -> B ->


Exp. 07:   -> B1 -> B2 -> 
         A -> B1 -> B2 -> A
           -> B1 -> B2 ->
```       


## License

TBD

## Contributing

TBD
