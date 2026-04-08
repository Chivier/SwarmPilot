from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[reportMissingImports]

from swarmbench.models import Task


class MCPAtlasLoader:
    """Load and normalize MCP-Atlas tasks from Hugging Face."""

    def load(self, limit: int | None = None) -> list[Task]:
        """Load MCP-Atlas split and normalize each record.

        Args:
            limit: Optional max number of records to load.

        Returns:
            Normalized MCP-Atlas tasks.
        """
        dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
        tasks: list[Task] = []
        for index, record in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            if isinstance(record, Mapping):
                tasks.append(self._normalize(dict(record)))
        return tasks

    def _normalize(self, record: dict[str, Any]) -> Task:
        """Normalize a raw MCP-Atlas record into canonical Task.

        Args:
            record: Raw dataset row.

        Returns:
            Canonical task with ground-truth claims and metadata.
        """
        enabled_tools = self._parse_json_field(
            record.get("ENABLED_TOOLS", []), default=[]
        )
        claims = self._parse_json_field(record.get("GTFA_CLAIMS", []), default=[])
        trajectory = self._parse_json_field(record.get("TRAJECTORY", "[]"), default=[])

        return Task(
            task_id=str(record["TASK"]),
            dataset_name="mcp-atlas",
            prompt=str(record["PROMPT"]),
            ground_truth={"claims": claims},
            metadata={"enabled_tools": enabled_tools, "trajectory": trajectory},
        )

    def _parse_json_field(self, value: Any, default: Any) -> Any:
        """Parse a JSON field that may already be native Python.

        Args:
            value: Candidate JSON string or python value.
            default: Fallback value for invalid JSON strings.

        Returns:
            Parsed value when possible, otherwise provided fallback.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value
