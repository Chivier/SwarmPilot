"""Utilities for optimizing migration operations by detecting and eliminating cycles.

This module provides functions to detect and eliminate redundant migration cycles
in deployment plans. There are two types of cycle detection:

1. **Model Swap Detection** (pre-fetch): Detects when instances need to swap models
   with each other. For example:
   - Instance A (running model_x) needs model_y
   - Instance B (running model_y) needs model_x
   This can be eliminated before fetching from store, as no actual migration is needed.

2. **Endpoint Cycle Detection** (post-fetch): Detects when migration targets form
   a closed loop with original endpoints. For example:
   - A -> B and B -> A (binary cycle)
   - A -> B -> C -> A (ternary cycle)

Example:
    >>> from migration_optimizer import detect_model_swap_pairs
    >>> endpoints = ["A", "B", "C"]
    >>> current_models = ["model_x", "model_y", "model_z"]
    >>> target_models = ["model_y", "model_x", "model_w"]
    >>> indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)
    >>> indices  # Indices 0 and 1 can be eliminated
    {0, 1}
"""

from loguru import logger


def detect_model_swap_pairs(
    endpoints: list[str], current_models: list[str], target_models: list[str]
) -> tuple[set[int], list[tuple[int, int]]]:
    """Detect model swap pairs BEFORE fetching from store.

    This function identifies instances that need to swap models with each other.
    These can be eliminated without any actual migration, as the net effect is zero.

    A swap pair occurs when:
    - Instance A (running model_x) needs model_y
    - Instance B (running model_y) needs model_x
    - Both A and B need to change (are in the migration list)

    This also supports multi-way swaps (cycles):
    - A (model_x) -> model_y
    - B (model_y) -> model_z
    - C (model_z) -> model_x
    All three can be eliminated as the models just rotate among themselves.

    Args:
        endpoints: List of all endpoint URLs (only those needing change)
        current_models: List of current model for each endpoint
        target_models: List of target model for each endpoint

    Returns:
        Tuple of (
            indices_to_eliminate: Set of indices that can be eliminated,
            swap_pairs: List of (idx1, idx2) pairs that form binary swaps
        )
    """
    n = len(endpoints)
    if n == 0:
        return set(), []

    indices_to_eliminate: set[int] = set()
    swap_pairs: list[tuple[int, int]] = []

    # Build mapping: current_model -> list of indices that have this model
    model_to_indices: dict[str, list[int]] = {}
    for idx, model in enumerate(current_models):
        if model not in model_to_indices:
            model_to_indices[model] = []
        model_to_indices[model].append(idx)

    # Build graph: each index points to the model it needs
    # We want to find cycles in the "model flow"
    # If A(model_x)->model_y and B(model_y)->model_x, they can swap

    # For each instance, find if there's another instance that:
    # 1. Currently runs the model this instance needs
    # 2. Needs the model this instance currently runs

    visited = set()

    for start_idx in range(n):
        if start_idx in visited:
            continue
        if start_idx in indices_to_eliminate:
            continue

        # Try to find a cycle starting from this index
        cycle = _find_model_swap_cycle(
            start_idx, current_models, target_models, model_to_indices, visited
        )

        if cycle and len(cycle) >= 2:
            # Found a swap cycle
            for idx in cycle:
                indices_to_eliminate.add(idx)
                visited.add(idx)

            # Record binary pairs for logging
            if len(cycle) == 2:
                swap_pairs.append((cycle[0], cycle[1]))

            cycle_desc = " -> ".join(
                f"{endpoints[i]}({current_models[i]}->{target_models[i]})"
                for i in cycle
            )
            logger.info(f"Detected model swap cycle: {cycle_desc}")

    if indices_to_eliminate:
        logger.info(
            f"Model swap detection: eliminated {len(indices_to_eliminate)} migrations "
            f"({len(swap_pairs)} binary swaps, {len(indices_to_eliminate) - 2 * len(swap_pairs)} in larger cycles)"
        )
    else:
        logger.debug("No model swap pairs detected")

    return indices_to_eliminate, swap_pairs


def _find_model_swap_cycle(
    start_idx: int,
    current_models: list[str],
    target_models: list[str],
    model_to_indices: dict[str, list[int]],
    global_visited: set[int],
) -> list[int]:
    """Find a cycle of model swaps starting from start_idx.

    A cycle exists when following the chain of "who has what I need" leads back to start.
    """
    path = [start_idx]
    visited_in_path = {start_idx}
    current_idx = start_idx

    while True:
        needed_model = target_models[current_idx]

        # Find an instance that currently runs the model we need
        # AND also needs to change (is in our list)
        candidates = model_to_indices.get(needed_model, [])

        next_idx = None
        for candidate in candidates:
            if candidate == current_idx:
                continue
            if candidate in global_visited:
                continue
            # Check if this candidate needs to change
            if current_models[candidate] != target_models[candidate]:
                next_idx = candidate
                break

        if next_idx is None:
            # No valid next step, no cycle
            return []

        if next_idx == start_idx:
            # Found cycle back to start!
            return path

        if next_idx in visited_in_path:
            # Hit a node already in path but not start, no valid cycle from start
            return []

        path.append(next_idx)
        visited_in_path.add(next_idx)
        current_idx = next_idx

        # Safety limit to prevent infinite loops
        if len(path) > len(current_models):
            return []

    return []


def eliminate_redundant_migrations(
    pending_change_original: list[str],
    pending_change_original_model: list[str],
    pending_change_target: list[str],
    pending_change_target_model: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[tuple[str, str]]]:
    """Eliminate redundant migration cycles (supports any cycle length: binary, ternary, n-ary).

    A migration cycle is a closed loop where migrations cancel each other out.
    For example:
    - Binary cycle: A->B and B->A (both can be eliminated)
    - Ternary cycle: A->B, B->C, C->A (all three can be eliminated)

    Args:
        pending_change_original: List of original endpoint URLs
        pending_change_original_model: List of models running on original endpoints
        pending_change_target: List of target endpoint URLs
        pending_change_target_model: List of models running on target endpoints

    Returns:
        Tuple of (
            filtered_original: Original endpoints after cycle removal
            filtered_original_model: Models for filtered original endpoints
            filtered_target: Target endpoints after cycle removal
            filtered_target_model: Models for filtered target endpoints
            cancelled_targets: List of (endpoint, model_id) tuples for endpoints
                              that need to be returned to the available instance store
        )
    """
    n = len(pending_change_original)

    # Handle empty input
    if n == 0:
        return [], [], [], [], []

    # Detect all cycle indices
    indices_to_remove = _detect_cycles(
        pending_change_original, pending_change_target
    )

    if not indices_to_remove:
        logger.info("No redundant migration cycles detected")
        return (
            pending_change_original.copy(),
            pending_change_original_model.copy(),
            pending_change_target.copy(),
            pending_change_target_model.copy(),
            [],
        )

    # Collect target endpoints that need to be returned to store
    cancelled_targets: list[tuple[str, str]] = []
    for idx in indices_to_remove:
        cancelled_targets.append(
            (pending_change_target[idx], pending_change_target_model[idx])
        )

    # Filter out redundant migrations
    filtered_original: list[str] = []
    filtered_original_model: list[str] = []
    filtered_target: list[str] = []
    filtered_target_model: list[str] = []

    for i in range(n):
        if i not in indices_to_remove:
            filtered_original.append(pending_change_original[i])
            filtered_original_model.append(pending_change_original_model[i])
            filtered_target.append(pending_change_target[i])
            filtered_target_model.append(pending_change_target_model[i])

    logger.info(
        f"Migration optimization: eliminated {len(indices_to_remove)} redundant migrations, "
        f"{len(filtered_original)} remaining"
    )

    return (
        filtered_original,
        filtered_original_model,
        filtered_target,
        filtered_target_model,
        cancelled_targets,
    )


def _detect_cycles(
    pending_change_original: list[str], pending_change_target: list[str]
) -> set[int]:
    """Detect all migration indices that are part of cycles using graph traversal.

    Algorithm:
    1. Build a directed graph: original[i] -> target[i]
    2. For each node, check if following the graph leads back to the start
    3. If a cycle is found, all edges in the cycle are marked as redundant

    A cycle is defined as: A->B->C->...->A, where each node is an original endpoint.

    Args:
        pending_change_original: List of original endpoint URLs
        pending_change_target: List of target endpoint URLs

    Returns:
        Set of indices representing migrations that are part of cycles
    """
    n = len(pending_change_original)
    indices_to_remove: set[int] = set()

    # Debug logging
    logger.debug(
        f"Cycle detection input - original endpoints: {pending_change_original}"
    )
    logger.debug(
        f"Cycle detection input - target endpoints: {pending_change_target}"
    )

    # Build mapping: original endpoint -> index
    original_to_index: dict[str, int] = {}
    for idx, ep in enumerate(pending_change_original):
        original_to_index[ep] = idx

    # Build directed graph: original -> target
    # Only include edges where target is also in original (potential cycle edges)
    graph: dict[str, str] = {}
    for i in range(n):
        if pending_change_target[i] in original_to_index:
            graph[pending_change_original[i]] = pending_change_target[i]
            logger.debug(
                f"Added edge to graph: {pending_change_original[i]} -> {pending_change_target[i]}"
            )
        else:
            logger.debug(
                f"Skipped edge (target not in original): {pending_change_original[i]} -> {pending_change_target[i]}"
            )

    logger.debug(f"Graph edges count: {len(graph)}")

    # Track globally visited nodes to avoid re-processing
    visited_globally: set[str] = set()

    # For each potential starting point, check if it forms a cycle
    for start_idx in range(n):
        start = pending_change_original[start_idx]

        if start in visited_globally:
            continue
        if start not in graph:
            continue

        # Try to find a cycle starting from this node
        cycle_path = _find_cycle_from(start, graph)

        if cycle_path:
            # Found a cycle, mark all migrations in it as redundant
            for node in cycle_path:
                idx = original_to_index[node]
                indices_to_remove.add(idx)
                visited_globally.add(node)

            logger.info(
                f"Detected cycle: {' -> '.join(cycle_path)} -> {cycle_path[0]}"
            )

    return indices_to_remove


def _find_cycle_from(start: str, graph: dict[str, str]) -> list[str]:
    """Starting from 'start', traverse the graph to detect if a cycle exists back to start.

    This function performs a simple path traversal, following edges in the graph
    until either:
    - We return to the start node (cycle found)
    - We reach a node not in the graph (no cycle)
    - We reach a previously visited node that isn't start (no cycle from this start)

    Args:
        start: The starting node for cycle detection
        graph: A dictionary mapping source nodes to target nodes

    Returns:
        List of nodes in the cycle (in order), or empty list if no cycle found.
        The returned list does NOT include the return edge back to start.
    """
    path: list[str] = [start]
    visited: set[str] = {start}
    current = start

    while current in graph:
        next_node = graph[current]

        if next_node == start:
            # Found cycle: returned to starting point
            return path

        if next_node in visited:
            # Reached a previously visited node that isn't start
            # This means we found a cycle but not one that includes start
            return []

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    # Reached a node not in graph (dead end), no cycle
    return []
