# Instance Service Documentation

## Overview

The Instance Service is the minimal execution and scheduling unit in the SwarmPilot project. Each instance serves a single model at a time and manages a task queue for processing inference requests.

## Documentation Structure

1. **[Architecture Overview](./1.ARCHITECTURE.md)** - System design, features, and components
2. **[Model Container Specification](./2.MODEL_CONTAINER_SPEC.md)** - Requirements for model containers
3. **[Model Registry Guide](./3.MODEL_REGISTRY.md)** - Model registry structure and configuration
4. **[Startup Procedures](./4.STARTUP_PROCEDURES.md)** - Detailed model container startup process
5. **[API Reference](./5.API_REFERENCE.md)** - Complete API endpoint documentation

## Quick Start

### Instance Features
- **WebSocket Communication**: Real-time bidirectional communication with Scheduler (5-8x lower latency)
- **Task Queue**: Sequential FIFO processing with automatic task execution
- **Dynamic Model Management**: Hot-swapping between different models via API
- **Docker-Based Containerization**: Isolated model execution environments
- **Health Monitoring**: Automated health checks and status reporting
- **Dual-Protocol Support**: WebSocket (default) with automatic HTTP fallback

### Core Concepts

**Instance**: A service unit that manages one model container and processes tasks sequentially.

**Task**: A work unit containing:
- `task_id`: Unique identifier
- `model_id`: Target model identifier
- `task_input`: Model-specific input data

**Model Container**: A Docker container that:
- Implements standardized HTTP endpoints
- Uses uv for Python dependency management
- Exposes inference capabilities via REST API

**Scheduler Connection**: The instance can optionally connect to a Scheduler service:
- WebSocket-based real-time task submission (default)
- HTTP-based fallback for task callbacks
- Automatic reconnection with exponential backoff
- Heartbeat mechanism for connection health

## Getting Started

1. Review the [Architecture Overview](./1.ARCHITECTURE.md) to understand the system design
2. Read the [Model Container Specification](./2.MODEL_CONTAINER_SPEC.md) to create compliant model containers
3. Configure your models in the [Model Registry](./3.MODEL_REGISTRY.md)
4. Consult the [API Reference](./5.API_REFERENCE.md) for integration

## Support

For issues and questions, please refer to the individual documentation files or contact the SwarmPilot development team.
