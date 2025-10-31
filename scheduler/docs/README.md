# Scheduler Service Documentation

Welcome to the Scheduler Service documentation. This guide will help you understand, deploy, and use the scheduler service.

## 📚 Documentation Index

### Getting Started
- [Project README](../README.md) - Overview, installation, and quick start

### Core Concepts
- [Data Models](./data-models.md) - Request/response models and data structures
- [Scheduling Strategies](./scheduling-strategies.md) - Available scheduling algorithms and their behavior

### API Reference
- [Instance Management API](./api-instances.md) - Register, remove, and query instances
- [Task Management API](./api-tasks.md) - Submit, track, and query tasks
- [WebSocket API](./api-websocket.md) - Real-time task result notifications

### Implementation
- [Implementation Details](./implementation.md) - Queue management and task lifecycle

## 🚀 Quick Links

### For Users
1. Start here: [Project README](../README.md)
2. Understand the data: [Data Models](./data-models.md)
3. Submit tasks: [Task Management API](./api-tasks.md)
4. Get real-time results: [WebSocket API](./api-websocket.md)

### For Developers
1. Architecture: [Implementation Details](./implementation.md)
2. Scheduling algorithms: [Scheduling Strategies](./scheduling-strategies.md)
3. API contracts: All API docs

### For Instance Providers
1. Register instances: [Instance Management API](./api-instances.md)
2. Understand queue updates: [Implementation Details](./implementation.md)

## 📖 Documentation Conventions

- **Request/Response Examples**: All API docs include JSON examples
- **Code References**: References to source code use format `file.py:line`
- **Cross-references**: Links between docs for related topics

## 🔗 External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework used
- [Pydantic Documentation](https://docs.pydantic.dev/) - Data validation library

## 📝 Contributing

When updating documentation:
1. Keep each doc focused on a single topic
2. Update cross-references when moving content
3. Include examples for all API endpoints
4. Reference source code for implementation details
