"""
Manager factory for dynamic manager selection.

This module provides a factory function that returns the appropriate manager
based on the configuration. It allows switching between DockerManager and
SubprocessManager at runtime.
"""

from typing import Union

from .config import config
from .docker_manager import DockerManager, get_docker_manager as get_docker_manager_impl
from .subprocess_manager import SubprocessManager, get_subprocess_manager


def get_docker_manager() -> Union[DockerManager, SubprocessManager]:
    """
    Get the appropriate manager based on configuration.

    This function returns either a DockerManager or SubprocessManager instance
    depending on the config.use_docker setting. The function name is kept as
    get_docker_manager() for backward compatibility with existing code.

    Returns:
        DockerManager: If config.use_docker is True (requires Docker)
        SubprocessManager: If config.use_docker is False (no Docker required)

    Notes:
        - The manager type can be controlled via INSTANCE_USE_DOCKER env var
        - The manager type can be controlled via --docker CLI parameter
        - CLI parameter takes precedence over environment variable
    """
    if config.use_docker:
        return get_docker_manager_impl()
    else:
        return get_subprocess_manager()
