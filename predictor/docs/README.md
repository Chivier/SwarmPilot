# Predictor Service - Documentation

## Overview

The Predictor is a prediction service component of swarmpilot that accepts task features from the scheduler and returns expected runtime or runtime distribution quantiles based on the input features and prediction type requested.

## Key Features

1. **Runtime Prediction**: Predict the expected runtime or distribution of runtime using a Multi-Layer Perceptron (MLP) model
2. **Model Training**: Support training MLP models from user-provided training data
3. **Experiment Mode**: Special mode for testing and development without trained models
4. **Multi-Model Support**: Maintains separate trained models for each unique combination of platform and model ID

## Documentation Index

### Getting Started
- **[Quick Start Guide](2.QUICK_START.md)** - Get up and running with the Predictor service in 5 minutes
- **[Usage Examples](3.USAGE_EXAMPLES.md)** - Complete examples and common workflows

### API Reference
- **[Data Models](4.DATA_MODELS.md)** - Complete reference for all request/response models
- **[API Endpoints](5.API_ENDPOINTS.md)** - Detailed endpoint specifications and examples

### Advanced Topics
- **[Prediction Types](6.PREDICTION_TYPES.md)** - Understanding expect_error vs quantile predictions
- **[Error Handling](7.ERROR_HANDLING.md)** - Troubleshooting and debugging guide
- **[Best Practices](8.BEST_PRACTICES.md)** - Tips for optimal usage and performance

### Legacy
- **[API Reference (Legacy)](1.API_REFERENCE.md)** - Original comprehensive reference document

## Quick Links

### Common Tasks
- [Train a model](3.USAGE_EXAMPLES.md#training-a-model)
- [Get predictions](3.USAGE_EXAMPLES.md#making-predictions)
- [List available models](5.API_ENDPOINTS.md#get-list)
- [Use experiment mode](3.USAGE_EXAMPLES.md#experiment-mode)

### API Endpoints
- `POST /predict` - Get runtime predictions
- `POST /train` - Train or update a model
- `GET /list` - List all available models
- `GET /health` - Check service health

## Service Information

**Default Port**: 8000
**Protocol**: HTTP/JSON
**Authentication**: None (internal service)

## Architecture

The Predictor service operates as follows:

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Scheduler  │────────>│  Predictor   │────────>│  MLP Model  │
│             │ features│   Service    │ predict │   Storage   │
└─────────────┘         └──────────────┘         └─────────────┘
                              │                          │
                              │ train                    │
                              └──────────────────────────┘
```

1. **Scheduler** sends task features to the Predictor
2. **Predictor** uses the appropriate trained model to generate predictions
3. **MLP Models** are stored persistently and can be retrained with new data

## Key Concepts

### Platform Info
Identifies the hardware and software environment:
- `software_name`: Framework (e.g., "pytorch", "tensorflow")
- `software_version`: Version (e.g., "2.0.1")
- `hardware_name`: Hardware identifier (e.g., "nvidia-a100")

### Prediction Types
- **expect_error**: Returns expected runtime with error margin
- **quantile**: Returns runtime distribution at specified percentiles

### Model Keys
Models are uniquely identified by the combination of:
- `model_id`
- `platform_info.software_name`
- `platform_info.software_version`
- `platform_info.hardware_name`

## Support and Feedback

For issues, questions, or contributions, please refer to the main swarmpilot repository.

## Version

This documentation is for Predictor service version 1.0.

Last updated: 2025-10-31
