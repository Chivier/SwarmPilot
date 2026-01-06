# API Routes Reference

This document describes the implemented API routes for the SwarmX Instance API.

## Quick Reference

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/v1/system/health` | GET | Health check | Implemented |
| `/v1/system/info` | GET | System information | Implemented |
| `/v1/files/upload` | POST | Upload file | Implemented |
| `/v1/files` | GET | List files | Implemented |
| `/v1/files/{file_id}` | GET | Download file | Implemented |
| `/v1/files/{file_id}` | DELETE | Delete file | Implemented |

## Service Layer

The API uses a service layer for business logic:

### FileStorageService

Located at `src/services/file_storage.py`. Provides file persistence with:

- **save_file**: Save uploaded files with metadata
- **get_file**: Retrieve files by ID
- **list_files**: List files with filtering and pagination
- **delete_file**: Remove files
- **check_disk_space**: Verify available disk space

**Storage Layout:**
```
/data/files/
├── file_abc123/
│   ├── metadata.json  # FileInfo as JSON
│   └── context.jsonl  # Original file
└── file_def456/
    ├── metadata.json
    └── data.jsonl
```

**Exceptions:**
- `FileNotFoundError`: Raised when file doesn't exist
- `InsufficientDiskSpaceError`: Raised when disk space is insufficient

## System API

### GET /v1/system/health

Simple health check endpoint for load balancers and monitoring.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-05T12:00:00Z"
}
```

**Response Schema**: `HealthResponse`

### GET /v1/system/info

Returns detailed system information including resource usage.

**Response**
```json
{
  "instance_id": "inst_xyz789",
  "uptime_seconds": 3600,
  "supported_model_types": ["llm"],
  "inference_server": {
    "type": "vllm",
    "version": "0.6.0"
  },
  "resources": {
    "cpu": {
      "cores": 8,
      "usage_percent": 45.2
    },
    "memory": {
      "total_gb": 64.0,
      "used_gb": 32.5,
      "usage_percent": 50.8
    },
    "disk": {
      "total_gb": 500.0,
      "used_gb": 120.0,
      "available_gb": 380.0,
      "min_free_gb": 10.0,
      "usage_percent": 24.0
    },
    "gpu": [
      {
        "index": 0,
        "name": "NVIDIA A100",
        "memory_total_gb": 80.0,
        "memory_used_gb": 45.0,
        "utilization_percent": 78.5,
        "temperature_celsius": 65
      }
    ]
  }
}
```

**Response Schema**: `SystemInfo`

**Notes**:
- GPU information is collected via `nvidia-smi` and will be empty if no NVIDIA GPUs are available
- CPU/memory/disk metrics are collected via `psutil`
- The `uptime_seconds` reflects time since the instance started

## File API

### POST /v1/files/upload

Upload a file for inference workloads.

**Request**: Multipart form data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | Binary file content |
| purpose | string | Yes | Purpose (`inference` or `batch`) |
| tags | string | No | Comma-separated tags |
| ttl_hours | int | No | Auto-delete after N hours |

**Response** (201 Created)
```json
{
  "file_id": "file_abc123",
  "filename": "context.jsonl",
  "purpose": "inference",
  "size_bytes": 1048576,
  "created_at": "2026-01-05T12:00:00Z"
}
```

**Error Responses**:
- 422: Missing required fields
- 507: Insufficient disk space

### GET /v1/files

List files with optional filtering.

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| purpose | string | - | Filter by purpose |
| tag | string | - | Filter by tag |
| limit | int | 100 | Max results (1-1000) |
| offset | int | 0 | Pagination offset |

**Response**
```json
{
  "files": [
    {
      "file_id": "file_abc123",
      "filename": "context.jsonl",
      "purpose": "inference",
      "size_bytes": 1048576,
      "tags": ["training"],
      "created_at": "2026-01-05T12:00:00Z"
    }
  ],
  "total": 42
}
```

### GET /v1/files/{file_id}

Download a file by ID.

**Response**: Binary file stream with `Content-Disposition: attachment; filename="original_name.jsonl"`

**Error Responses**:
- 404: File not found

### DELETE /v1/files/{file_id}

Delete a file by ID.

**Response**
```json
{
  "deleted": true,
  "file_id": "file_abc123"
}
```

**Error Responses**:
- 404: File not found
