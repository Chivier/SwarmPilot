"""Instance management endpoints.

Provides endpoints for instance registration, removal, listing, and management.
Note: The actual endpoint implementations remain in api.py for now due to
their deep integration with helper functions. This module provides the router
structure for future refactoring.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/instance", tags=["instance"])

# Instance endpoints are currently defined in api.py
# They will be migrated here in a future refactoring phase
# when the helper functions are properly decoupled.
