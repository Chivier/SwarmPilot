from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[reportMissingImports]

from swarmbench.models import Task


class DataclawLoader:
    """Load and normalize Dataclaw tasks from Hugging Face."""

    def load(self, limit: int | None = None) -> list[Task]:
        """Load Dataclaw records and convert them into benchmark tasks.

        Args:
            limit: Optional maximum number of normalized tasks to return.

        Returns:
            Normalized Dataclaw tasks.
        """
        dataset = load_dataset("peteromallet/dataclaw-peteromallet", split="train")
        tasks: list[Task] = []
        for record in dataset:
            if limit is not None and len(tasks) >= limit:
                break
            if not isinstance(record, Mapping):
                continue
            normalized = self._normalize(dict(record))
            if normalized is not None:
                tasks.append(normalized)
        return tasks

    def _normalize(self, record: dict[str, Any]) -> Task | None:
        """Normalize one Dataclaw row into the canonical task model.

        Args:
            record: Raw Dataclaw dataset row.

        Returns:
            Canonical task, or None when the record has no user message.
        """
        messages = record.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        prompt = self._extract_first_user_prompt(messages)
        if prompt is None:
            return None

        trajectory = self._extract_tool_trajectory(messages)
        reference_response = self._extract_last_assistant_content(messages)

        task_id = record.get("task_id") or record.get("id") or ""
        task_id_str = str(task_id) if task_id else f"dataclaw-{abs(hash(prompt))}"

        return Task(
            task_id=task_id_str,
            dataset_name="dataclaw",
            prompt=prompt,
            ground_truth={
                "reference_response": reference_response,
                "reference_tool_trajectory": trajectory,
            },
            metadata={
                "model": record.get("model"),
                "project": record.get("project"),
                "stats": record.get("stats"),
            },
        )

    def _extract_first_user_prompt(self, messages: list[Any]) -> str | None:
        """Extract the first user message content.

        Args:
            messages: Raw session messages.

        Returns:
            First user content as text, or None when no user message exists.
        """
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if content is None:
                return ""
            return self._to_text(content)
        return None

    def _extract_tool_trajectory(self, messages: list[Any]) -> list[dict[str, str]]:
        """Build an ordered tool trajectory from assistant messages.

        Args:
            messages: Raw session messages.

        Returns:
            Ordered list of tool call objects with tool and input fields.
        """
        ordered_calls: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "assistant":
                continue

            tool_uses = message.get("tool_uses")
            if not isinstance(tool_uses, list):
                continue

            for tool_use in tool_uses:
                if not isinstance(tool_use, Mapping):
                    continue
                tool_name = tool_use.get("tool")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                ordered_calls.append(
                    {
                        "tool": tool_name,
                        "input": self._to_text(tool_use.get("input")),
                    }
                )
        return ordered_calls

    def _extract_last_assistant_content(self, messages: list[Any]) -> str:
        """Extract the last non-null assistant content string.

        Args:
            messages: Raw session messages.

        Returns:
            Last assistant textual content, or empty string when absent.
        """
        for message in reversed(messages):
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if content is None:
                continue
            return self._to_text(content)
        return ""

    def _to_text(self, value: Any) -> str:
        """Convert mixed content values into deterministic text.

        Args:
            value: Value to serialize as text.

        Returns:
            String representation suitable for task prompts and references.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value)
