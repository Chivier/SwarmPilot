"""Registry for V2 preprocessors and chains.

This module provides a registry for managing preprocessor instances,
factories, and chains. Built-in preprocessors are auto-registered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from swarmpilot.predictor.preprocessor.base_preprocessor_v2 import (
    BasePreprocessorV2,
)
from swarmpilot.predictor.preprocessor.chain_v2 import PreprocessorChainV2
from swarmpilot.predictor.preprocessor.preprocessors_v2 import (
    MultiplyPreprocessor,
    RemoveFeaturePreprocessor,
    TokenLengthPreprocessor,
)


class PreprocessorsRegistryV2:
    """Registry for V2 preprocessors, factories, and chains.

    Provides centralized management of preprocessors with support for:
    - Static preprocessor instances
    - Factory functions for dynamic creation
    - Named chains for reuse
    - Configuration-based chain creation

    Built-in preprocessors (multiply, remove, token_length) are
    auto-registered as factories on initialization.

    Example:
        >>> registry = PreprocessorsRegistryV2()
        >>> prep = registry.get("multiply", feature_a="a", feature_b="b",
        ...                     output_feature="c")
        >>> chain = registry.create_chain_from_config([
        ...     {"name": "multiply", "params": {"feature_a": "a", ...}}
        ... ])

    Attributes:
        _preprocessors: Static preprocessor instances by name.
        _factories: Factory functions by name.
        _chains: Named chains.
    """

    def __init__(self) -> None:
        """Initialize registry with built-in preprocessors."""
        self._preprocessors: dict[str, BasePreprocessorV2] = {}
        self._factories: dict[str, Callable[..., BasePreprocessorV2]] = {}
        self._chains: dict[str, PreprocessorChainV2] = {}

        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in preprocessor factories."""
        self.register_factory(
            "multiply",
            lambda **kwargs: MultiplyPreprocessor(**kwargs),
        )
        self.register_factory(
            "remove",
            lambda **kwargs: RemoveFeaturePreprocessor(**kwargs),
        )
        self.register_factory(
            "token_length",
            lambda **kwargs: TokenLengthPreprocessor(**kwargs),
        )

    def register(self, name: str, preprocessor: BasePreprocessorV2) -> None:
        """Register a static preprocessor instance.

        Args:
            name: Name to register under.
            preprocessor: The preprocessor instance.
        """
        self._preprocessors[name] = preprocessor

    def register_factory(
        self,
        name: str,
        factory: Callable[..., BasePreprocessorV2],
    ) -> None:
        """Register a factory function for creating preprocessors.

        Args:
            name: Name to register under.
            factory: Callable that creates a preprocessor instance.
        """
        self._factories[name] = factory

    def register_chain(self, name: str, chain: PreprocessorChainV2) -> None:
        """Register a named chain.

        Args:
            name: Name to register under.
            chain: The chain instance.
        """
        self._chains[name] = chain

    def get(self, name: str, **kwargs: Any) -> BasePreprocessorV2 | None:
        """Get a preprocessor by name.

        Static instances take precedence over factories. If kwargs are
        provided and a factory exists, creates a new instance.

        Args:
            name: Name of the preprocessor.
            **kwargs: Parameters to pass to factory.

        Returns:
            Preprocessor instance or None if not found.
        """
        if name in self._preprocessors:
            return self._preprocessors[name]

        if name in self._factories:
            return self._factories[name](**kwargs)

        return None

    def get_chain(self, name: str) -> PreprocessorChainV2 | None:
        """Get a registered chain by name.

        Args:
            name: Name of the chain.

        Returns:
            Chain instance or None if not found.
        """
        return self._chains.get(name)

    def create_chain_from_config(
        self,
        config: list[dict[str, Any]],
        chain_name: str = "config_chain",
    ) -> PreprocessorChainV2:
        """Create a chain from a configuration list.

        Each config item should have:
        - name: Registered preprocessor name
        - params: Dict of parameters for the preprocessor

        Args:
            config: List of preprocessor configurations.
            chain_name: Name for the created chain.

        Returns:
            Configured PreprocessorChainV2.

        Raises:
            ValueError: If a preprocessor name is not registered.

        Example:
            >>> config = [
            ...     {"name": "multiply", "params": {"feature_a": "a", ...}},
            ...     {"name": "remove", "params": {"features_to_remove": ["a"]}}
            ... ]
            >>> chain = registry.create_chain_from_config(config)
        """
        chain = PreprocessorChainV2(name=chain_name)

        for step in config:
            name = step["name"]
            params = step.get("params", {})

            prep = self.get(name, **params)
            if prep is None:
                raise ValueError(
                    f"Unknown preprocessor '{name}'. "
                    f"Available: {self.list_available()}"
                )

            chain.add(prep)

        return chain

    def list_available(self) -> list[str]:
        """List all available preprocessor names.

        Returns:
            List of registered preprocessor and factory names.
        """
        names = set(self._preprocessors.keys()) | set(self._factories.keys())
        return sorted(names)
