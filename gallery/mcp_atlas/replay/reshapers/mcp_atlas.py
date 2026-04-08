"""Reshape ScaleAI/MCP-Atlas dataset into ReplayGroups.

MCP-Atlas conversations have one user PROMPT followed by a TRAJECTORY of
alternating assistant (large model) and tool (small model) messages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from replay.models import ReplayGroup, ReplayStep


class MCPAtlasReshaper:
    """Load MCP-Atlas from HuggingFace and reshape into replay groups."""

    def load_and_reshape(self, limit: int | None = None) -> list[ReplayGroup]:
        """Load the dataset and convert each record into a ReplayGroup.

        Args:
            limit: Maximum number of groups to produce.

        Returns:
            List of ReplayGroups ready for scheduling.
        """
        dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
        groups: list[ReplayGroup] = []
        for record in dataset:
            if limit is not None and len(groups) >= limit:
                break
            if not isinstance(record, Mapping):
                continue
            group = self._reshape_record(dict(record))
            if group is not None:
                groups.append(group)
        return groups

    def _reshape_record(self, record: dict[str, Any]) -> ReplayGroup | None:
        """Convert one MCP-Atlas row into a ReplayGroup.

        Args:
            record: Raw dataset row with TASK, PROMPT, TRAJECTORY fields.

        Returns:
            ReplayGroup, or None if the record has no usable trajectory.
        """
        prompt = str(record.get("PROMPT", ""))
        if not prompt:
            return None

        trajectory = self._parse_json_field(record.get("TRAJECTORY", "[]"), default=[])
        if not isinstance(trajectory, list):
            trajectory = []

        task_id = str(record.get("TASK", ""))
        group_id = task_id or f"mcp-atlas-{abs(hash(prompt))}"

        initial_messages = [{"role": "user", "content": prompt}]

        steps: list[ReplayStep] = []
        for i, msg in enumerate(trajectory):
            if not isinstance(msg, Mapping):
                continue
            role = msg.get("role")
            if role == "assistant":
                model_size = "large"
            elif role == "tool":
                model_size = "small"
            else:
                # Skip unexpected roles (e.g. "user" mid-trajectory)
                continue

            steps.append(
                ReplayStep(
                    step_index=i,
                    model_size=model_size,
                    sender_role="agent",
                    history_message=dict(msg),
                )
            )

        if not steps:
            return None

        return ReplayGroup(
            group_id=group_id,
            dataset_name="mcp-atlas",
            initial_messages=initial_messages,
            steps=steps,
        )

    @staticmethod
    def _parse_json_field(value: Any, default: Any) -> Any:
        """Parse a JSON string field, returning the value as-is if already native Python.

        Args:
            value: Candidate JSON string or Python object.
            default: Fallback for invalid JSON strings.

        Returns:
            Parsed value or fallback.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value
