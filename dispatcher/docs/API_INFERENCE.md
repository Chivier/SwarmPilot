# Dispatcher - Architecture Overview

## Description

The Dispatcher provides a user interface to define workflow graphs, visualize workflows, and bind components through a Python API.

## Core Components

### Scheduler
Each model is bound to a dedicated scheduler that manages all requests for that specific model.

### Instance
The basic execution unit for models. Each instance runs a single model, though the model can be dynamically switched.

### Predictor
Provides runtime predictions for specific requests on given hardware configurations for a particular model.

### Planner
Analyzes profiling data and logs to generate deployment plans. Determines which instances should run which models and dispatches deployment requests to target instances.

## Component Relationships

- **System-Scheduler**: Each workflow/system contains multiple schedulers, one per model
- **Scheduler-Instance**: Each scheduler manages multiple instances
- **Scheduler-Planner**: Schedulers accept requests from users or other models and collect request statistics for the planner's future deployment decisions
- **Scheduler-Predictor**: Schedulers use runtime distribution predictions from the predictor to select optimal instances for request routing

## Data Structure Define

### Node



### Edge



### Graph



### Workflow
