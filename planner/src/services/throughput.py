"""Throughput data management service."""

from loguru import logger


def update_throughput_entry(
    throughput_data: dict[str, dict[str, float]],
    instance_url: str,
    model_id: str,
    new_capacity: float,
    alpha: float = 0.3,
) -> None:
    """Update throughput entry using exponential moving average.

    Args:
        throughput_data: The throughput data dictionary to update (mutated in place)
        instance_url: Instance endpoint URL
        model_id: Model ID for this instance
        new_capacity: New computed capacity (1/avg_execution_time)
        alpha: EMA smoothing factor (0.3 = 30% new, 70% old)
    """
    if instance_url not in throughput_data:
        throughput_data[instance_url] = {}

    if model_id in throughput_data[instance_url]:
        # EMA update: new = alpha * new_value + (1 - alpha) * old_value
        old_capacity = throughput_data[instance_url][model_id]
        updated_capacity = alpha * new_capacity + (1 - alpha) * old_capacity
        throughput_data[instance_url][model_id] = updated_capacity
        logger.debug(
            f"Throughput EMA update: {instance_url}/{model_id}: {old_capacity:.4f} -> {updated_capacity:.4f}"
        )
    else:
        # First submission: store exact value
        throughput_data[instance_url][model_id] = new_capacity
        logger.debug(
            f"Throughput first entry: {instance_url}/{model_id} = {new_capacity:.4f}"
        )


def apply_throughput_to_b_matrix(
    stored_deployment_input,
    stored_model_mapping: dict[str, int] | None,
    throughput_data: dict[str, dict[str, float]],
) -> int:
    """Apply collected throughput data to update the B matrix in stored deployment.

    For each (instance, model) pair with throughput data:
    - Find instance index i from stored_deployment_input.instances
    - Find model index j from stored_model_mapping
    - Update B[i][j] with the observed capacity

    Args:
        stored_deployment_input: The stored DeploymentInput to update (mutated in place)
        stored_model_mapping: Model name to ID mapping
        throughput_data: Collected throughput data

    Returns:
        Number of entries updated
    """
    if not stored_deployment_input or not stored_model_mapping:
        logger.debug(
            "Cannot apply throughput: no deployment input or model mapping"
        )
        return 0

    if not throughput_data:
        logger.debug("No throughput data to apply")
        return 0

    instances = stored_deployment_input.instances
    B = stored_deployment_input.planner_input.B
    update_count = 0

    for i, inst in enumerate(instances):
        instance_url = inst.endpoint
        if instance_url in throughput_data:
            for model_id, capacity in throughput_data[instance_url].items():
                if model_id in stored_model_mapping:
                    j = stored_model_mapping[model_id]
                    old_value = B[i][j]
                    B[i][j] = capacity
                    update_count += 1
                    logger.debug(
                        f"Updated B[{i}][{j}] for {instance_url}/{model_id}: {old_value:.4f} -> {capacity:.4f}"
                    )

    if update_count > 0:
        logger.info(f"Applied {update_count} throughput entries to B matrix")

    return update_count
