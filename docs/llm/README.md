# LLM Documentation Index

This directory contains single-file documentation optimized for LLM consumption. Each file provides a complete reference for one module without requiring cross-file navigation.

## Available Documentation

| File | Module | Description |
|------|--------|-------------|
| [scheduler.md](scheduler.md) | Scheduler | Task scheduling service with ML-based predictions |
| [planner.md](planner.md) | Planner | Deployment optimization with PyLet integration |
| [predictor.md](predictor.md) | Predictor | Runtime prediction using MLP models |

## Usage

These documents are designed to be loaded as context for LLM-based coding assistants. Each file:

- Contains complete API specifications with correct endpoint paths
- Includes accurate file path references (under `swarmpilot/`)
- Provides request/response JSON schemas
- Documents component interactions

## Document Structure

Each module document follows this structure:

1. **Overview** - Purpose and key concepts
2. **File Structure** - Source file layout
3. **API Reference** - Complete endpoint specifications
4. **Key Schemas** - Request/response examples
5. **Configuration** - Environment variables
6. **Quick Reference** - Lookup tables

## Important Notes

- Scheduler and Planner endpoints use the `/v1/` prefix
- Predictor endpoints have **no** prefix
- All source files are under `swarmpilot/` (not `scheduler/src/`, `planner/src/`, or `predictor/src/`)

## Versioning

| Module | Version | Last Updated |
|--------|---------|--------------|
| Scheduler | 1.0.0 | 2026-01-30 |
| Planner | 1.0.0 | 2026-01-30 |
| Predictor | 0.1.0 | 2026-01-30 |

---

For detailed documentation, see [docs/](../).
