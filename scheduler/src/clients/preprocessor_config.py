"""Preprocessor chain builder for scheduler-side preprocessing.

Loads preprocessor rules from a JSON config file and builds V2
PreprocessorChainV2 instances based on model_id pattern matching.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.clients._predictor_lib import (
    PreprocessorChainV2,
    PreprocessorsRegistry,
    V1PreprocessorAdapter,
)


class PreprocessorChainBuilder:
    """Builds V2 preprocessor chains from JSON config rules.

    Loads rules from a JSON config file. Each rule matches model_ids
    containing certain substrings and defines a chain of preprocessor
    steps to apply.

    Args:
        config_file: Path to JSON config file (empty = no rules).
        registry_v1: V1 PreprocessorsRegistry for adapter lookups.
        strict: If True, raise RuntimeError when a configured
            preprocessor model is unavailable.

    Config file JSON format:
        {
          "rules": [
            {
              "model_id_contains": ["llm_service", "model"],
              "chain": [
                {
                  "type": "v1_adapter",
                  "name": "semantic",
                  "input_feature": "sentence"
                }
              ]
            }
          ]
        }
    """

    def __init__(
        self,
        config_file: str,
        registry_v1: PreprocessorsRegistry,
        strict: bool = True,
    ) -> None:
        self._rules: list[dict[str, Any]] = self._load_rules(config_file)
        self._registry_v1 = registry_v1
        self._strict = strict

        if self._rules:
            logger.info(
                f"Loaded {len(self._rules)} preprocessor rules " f"from {config_file}"
            )
            self._validate_all()
        else:
            logger.info("No preprocessor rules configured")

    def _load_rules(self, config_file: str) -> list[dict[str, Any]]:
        """Load preprocessor rules from JSON config file.

        Args:
            config_file: Path to JSON config file. Empty string
                means no rules.

        Returns:
            List of rule dictionaries.
        """
        if not config_file:
            return []

        try:
            with open(config_file) as f:
                data = json.load(f)
            return data.get("rules", [])
        except FileNotFoundError:
            logger.warning(f"Preprocessor config file not found: {config_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in preprocessor config: {config_file}: {e}")
            return []

    def _validate_all(self) -> None:
        """Validate all configured preprocessors are available.

        For each rule and each chain step, verifies the preprocessor
        can be instantiated. In strict mode, raises RuntimeError if
        any preprocessor model is missing.

        Raises:
            RuntimeError: If strict=True and a preprocessor is
                unavailable (e.g., missing model file).
        """
        for rule_idx, rule in enumerate(self._rules):
            chain_steps = rule.get("chain", [])
            for step_idx, step in enumerate(chain_steps):
                step_type = step.get("type", "")
                step_name = step.get("name", "")

                if step_type == "v1_adapter":
                    try:
                        self._registry_v1.get_preprocessor(step_name)
                    except Exception as e:
                        msg = (
                            f"Preprocessor '{step_name}' configured in "
                            f"rule {rule_idx}, step {step_idx} but "
                            f"unavailable: {e}. Check predictor settings "
                            f"and provide model files."
                        )
                        if self._strict:
                            raise RuntimeError(msg) from e
                        logger.warning(msg)

    def get_chain(self, model_id: str) -> PreprocessorChainV2 | None:
        """Build a V2 chain for the given model_id.

        Matches the first rule where ALL strings in model_id_contains
        appear in the model_id. Returns None if no rules match.

        Args:
            model_id: Model identifier to match against rules.

        Returns:
            A PreprocessorChainV2 if a rule matches, None otherwise.
        """
        for rule in self._rules:
            patterns = rule.get("model_id_contains", [])
            if all(p in model_id for p in patterns):
                return self._build_chain(rule)
        return None

    def _build_chain(self, rule: dict[str, Any]) -> PreprocessorChainV2:
        """Build a V2 chain from a rule definition.

        Args:
            rule: Rule dictionary with "chain" key.

        Returns:
            Configured PreprocessorChainV2.
        """
        chain_name = "_".join(rule.get("model_id_contains", ["default"]))
        chain = PreprocessorChainV2(name=chain_name)

        for step in rule.get("chain", []):
            step_type = step.get("type", "")
            step_name = step.get("name", "")

            if step_type == "v1_adapter":
                v1_preprocessor = self._registry_v1.get_preprocessor(step_name)
                input_feature = step.get("input_feature", "sentence")
                adapter = V1PreprocessorAdapter(
                    v1_preprocessor=v1_preprocessor,
                    input_feature=input_feature,
                )
                chain.add(adapter)
            else:
                logger.warning(f"Unknown preprocessor step type: {step_type}")

        return chain

    @property
    def has_rules(self) -> bool:
        """Whether any preprocessor rules are configured."""
        return len(self._rules) > 0
