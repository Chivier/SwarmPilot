"""Instance state verification service."""

import asyncio

import httpx
from loguru import logger


async def fetch_instance_model(
    client: httpx.AsyncClient, endpoint: str, timeout: float = 5.0
) -> tuple[str, str | None, str | None]:
    """Fetch the current model running on an instance via /info endpoint.

    Args:
        client: httpx AsyncClient for making requests
        endpoint: Instance endpoint URL (e.g., "http://localhost:8210")
        timeout: Request timeout in seconds

    Returns:
        Tuple of (endpoint, actual_model_id, error_message)
        - actual_model_id is None if request failed or no model running
        - error_message is None if request succeeded
    """
    try:
        response = await client.get(f"{endpoint}/info", timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("instance", {}).get(
                "current_model"
            ):
                model_id = data["instance"]["current_model"].get("model_id")
                return (endpoint, model_id, None)
            else:
                # No model running on this instance
                return (endpoint, None, "No model running")
        else:
            return (endpoint, None, f"HTTP {response.status_code}")
    except httpx.TimeoutException:
        return (endpoint, None, "Timeout")
    except Exception as e:
        return (endpoint, None, str(e))


async def verify_instance_states(
    endpoints: list, expected_models: list, timeout: float = 5.0
) -> tuple[bool, list, dict]:
    """Verify that all instances are running the expected models.

    Queries /info endpoints in parallel to check instance states.

    Args:
        endpoints: List of instance endpoint URLs
        expected_models: List of expected model IDs (same order as endpoints)
        timeout: Request timeout per instance

    Returns:
        Tuple of (all_match, mismatches, actual_states)
        - all_match: True if all instances match expected state
        - mismatches: List of dicts with endpoint, expected, actual for mismatched instances
        - actual_states: Dict mapping endpoint -> actual_model_id
    """
    if len(endpoints) != len(expected_models):
        raise ValueError(
            f"endpoints ({len(endpoints)}) and expected_models ({len(expected_models)}) must have same length"
        )

    async with httpx.AsyncClient() as client:
        # Fetch all instance states in parallel
        tasks = [
            fetch_instance_model(client, endpoint, timeout)
            for endpoint in endpoints
        ]
        results = await asyncio.gather(*tasks)

    actual_states = {}
    mismatches = []

    for (endpoint, actual_model, error), expected_model in zip(
        results, expected_models
    ):
        actual_states[endpoint] = actual_model

        if error:
            mismatches.append(
                {
                    "endpoint": endpoint,
                    "expected": expected_model,
                    "actual": None,
                    "error": error,
                }
            )
        elif actual_model != expected_model:
            mismatches.append(
                {
                    "endpoint": endpoint,
                    "expected": expected_model,
                    "actual": actual_model,
                    "error": None,
                }
            )

    all_match = len(mismatches) == 0
    return (all_match, mismatches, actual_states)
