"""Reshape Intelligent-Internet/swebench-pro-gpt-5-codex-ii-agent-trajectories into ReplayGroups.

SWE-bench Pro trajectories record ii-agent (GPT-5-Codex) solving software
engineering tasks.  Each trajectory is a flat sequence of message_lists:

- text_prompt (user)           → initial_messages
- thinking + tool_call (model) → large model request
- tool_output (environment)    → small model request
- thinking + text_result       → large model request (final)

This mirrors the MCP-Atlas convention: assistant → large, tool → small.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from replay.models import ReplayGroup, ReplayStep

_DATASET_ID = (
    "Intelligent-Internet/swebench-pro-gpt-5-codex-ii-agent-trajectories"
)


class SwebenchProReshaper:
    """Load SWE-bench Pro trajectories from HuggingFace and reshape into replay groups."""

    def load_and_reshape(self, limit: int | None = None) -> list[ReplayGroup]:
        """Load the dataset and convert each record into a ReplayGroup.

        Args:
            limit: Maximum number of groups to produce.

        Returns:
            List of ReplayGroups ready for scheduling.
        """
        dataset = load_dataset(_DATASET_ID, split="train")
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
        """Convert one SWE-bench Pro row into a ReplayGroup.

        Args:
            record: Raw dataset row with instance_id, traj, etc.

        Returns:
            ReplayGroup, or None if the record has no usable trajectory.
        """
        instance_id = str(record.get("instance_id", ""))
        traj_raw = record.get("traj", "[]")
        traj = self._parse_json_field(traj_raw, default=[])
        if not isinstance(traj, list) or not traj:
            return None

        step_data = traj[0]
        if not isinstance(step_data, Mapping):
            return None

        message_lists = step_data.get("message_lists", [])
        if not isinstance(message_lists, list) or len(message_lists) < 2:
            return None

        # message_lists[0] is always the text_prompt → initial_messages.
        initial_text = self._extract_text_prompt(message_lists[0])
        if not initial_text:
            return None

        group_id = instance_id or f"swebench-pro-{abs(hash(initial_text))}"
        initial_messages = [{"role": "user", "content": initial_text}]

        steps: list[ReplayStep] = []
        step_idx = 0

        for ml in message_lists[1:]:
            if not isinstance(ml, list) or not ml:
                continue

            classified = self._classify_message_list(ml)
            if classified is None:
                continue

            model_size, history_message = classified
            steps.append(
                ReplayStep(
                    step_index=step_idx,
                    model_size=model_size,
                    sender_role="agent",
                    history_message=history_message,
                )
            )
            step_idx += 1

        if not steps:
            return None

        return ReplayGroup(
            group_id=group_id,
            dataset_name="swebench-pro",
            initial_messages=initial_messages,
            steps=steps,
        )

    @staticmethod
    def _extract_text_prompt(ml: list[dict[str, Any]]) -> str:
        """Extract the user prompt text from the first message_list.

        Args:
            ml: First message_list (expected to contain a text_prompt message).

        Returns:
            Prompt text, or empty string if not found.
        """
        for msg in ml:
            if msg.get("type") == "text_prompt":
                return str(msg.get("text", ""))
        return ""

    @staticmethod
    def _classify_message_list(
        ml: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | None:
        """Classify a message_list and build the corresponding history_message.

        Returns:
            (model_size, history_message) tuple, or None if unrecognizable.
        """
        # Check for tool_output (small model).
        for msg in ml:
            if "tool_output" in msg:
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = msg.get("tool_name", "")
                tool_output = str(msg.get("tool_output", ""))
                return "small", {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_output,
                }

        # Check for model response: thinking + tool_call or thinking + text_result.
        thinking_text = ""
        tool_call_msg = None
        text_result_msg = None

        for msg in ml:
            if msg.get("type") == "thinking":
                thinking_text = str(msg.get("thinking", ""))
            elif "tool_name" in msg and "tool_output" not in msg:
                tool_call_msg = msg
            elif msg.get("type") == "text_result":
                text_result_msg = msg

        # Model response with tool call → large.
        if tool_call_msg is not None:
            tool_call_id = tool_call_msg.get("tool_call_id", "")
            tool_name = tool_call_msg.get("tool_name", "")
            tool_input = tool_call_msg.get("tool_input", {})
            arguments = (
                json.dumps(tool_input, ensure_ascii=False)
                if isinstance(tool_input, (dict, list))
                else str(tool_input)
            )
            history_message: dict[str, Any] = {
                "role": "assistant",
                "content": thinking_text,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                    }
                ],
            }
            return "large", history_message

        # Final text response → large.
        if text_result_msg is not None:
            text = str(text_result_msg.get("text", ""))
            content = text if text else thinking_text
            return "large", {"role": "assistant", "content": content}

        # Thinking-only block (no tool call, no text result) → large.
        if thinking_text:
            return "large", {"role": "assistant", "content": thinking_text}

        return None

    @staticmethod
    def _parse_json_field(value: Any, default: Any) -> Any:
        """Parse a JSON string field, returning as-is if already native Python.

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
