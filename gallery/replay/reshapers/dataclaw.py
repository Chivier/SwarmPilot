"""Reshape peteromallet/dataclaw-peteromallet dataset into ReplayGroups.

Dataclaw conversations contain user/assistant turns.  Mapping rules:
- After each user message, the first assistant response -> large model.
- Intermediate assistant responses -> small model.
- Last assistant response before next user message or conversation end -> large model.
- If content/thinking are both null but tool_uses exist, generate random filler (100-200 tokens).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from replay.models import ReplayGroup, ReplayStep

# Word list for generating filler prompts when content/thinking are null but tool_uses exist.
_FILLER_WORDS = (
    "the user requested analysis of code structure and dependencies "
    "examining file contents checking for patterns reviewing implementation "
    "details searching through project files reading configuration data "
    "processing tool output evaluating results verifying correctness "
    "updating documentation applying changes running validation checks "
    "inspecting build artifacts resolving module imports parsing syntax "
    "tracing execution paths collecting diagnostic information handling "
    "error conditions formatting output generating reports compiling "
    "summaries extracting metadata indexing references cross referencing "
    "specifications comparing versions merging modifications testing "
    "integration points validating schema definitions optimizing queries"
).split()


def _generate_filler(min_tokens: int = 100, max_tokens: int = 200) -> str:
    """Generate a random filler prompt of 100-200 tokens (words).

    Args:
        min_tokens: Minimum word count.
        max_tokens: Maximum word count.

    Returns:
        Filler text string.
    """
    length = random.randint(min_tokens, max_tokens)
    words = random.choices(_FILLER_WORDS, k=length)
    return " ".join(words)


class DataclawReshaper:
    """Load Dataclaw from HuggingFace and reshape into replay groups."""

    def load_and_reshape(self, limit: int | None = None) -> list[ReplayGroup]:
        """Load the dataset and convert each session into a ReplayGroup.

        Args:
            limit: Maximum number of groups to produce.

        Returns:
            List of ReplayGroups ready for scheduling.
        """
        dataset = load_dataset("peteromallet/dataclaw-peteromallet", split="train")
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
        """Convert one Dataclaw session into a ReplayGroup.

        Args:
            record: Raw dataset row with session_id, messages, etc.

        Returns:
            ReplayGroup, or None if the session has no usable messages.
        """
        messages = record.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return None

        session_id = str(record.get("session_id", ""))
        group_id = session_id or f"dataclaw-{abs(hash(str(messages[:3])))}"

        segments = self._segment_by_user_turns(messages)
        if not segments:
            return None

        # The first user message becomes the initial context.
        first_user_msg = segments[0][0]
        first_user_content = self._extract_user_content(first_user_msg)
        initial_messages = [{"role": "user", "content": first_user_content}]

        steps: list[ReplayStep] = []
        step_idx = 0

        for seg_idx, (user_msg, assistant_msgs) in enumerate(segments):
            # For segments after the first, the user message is a step with sender_role="user".
            if seg_idx > 0:
                user_content = self._extract_user_content(user_msg)
                steps.append(
                    ReplayStep(
                        step_index=step_idx,
                        model_size="large",
                        sender_role="user",
                        history_message={"role": "user", "content": user_content},
                    )
                )
                step_idx += 1

            # Map assistant messages to model sizes.
            n = len(assistant_msgs)
            for i, asst_msg in enumerate(assistant_msgs):
                content = self._extract_assistant_content(asst_msg)
                model_size = self._assign_model_size(i, n)

                steps.append(
                    ReplayStep(
                        step_index=step_idx,
                        model_size=model_size,
                        sender_role="agent",
                        history_message={"role": "assistant", "content": content},
                    )
                )
                step_idx += 1

        if not steps:
            return None

        return ReplayGroup(
            group_id=group_id,
            dataset_name="dataclaw",
            initial_messages=initial_messages,
            steps=steps,
        )

    @staticmethod
    def _segment_by_user_turns(
        messages: list[Any],
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Segment messages into (user_msg, [assistant_msgs...]) groups.

        Each user message starts a new segment.  Assistant messages following
        it are collected until the next user message.

        Args:
            messages: Raw session message list.

        Returns:
            List of (user_msg, assistant_msgs) tuples.
        """
        segments: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        current_user: dict[str, Any] | None = None
        current_assistants: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            role = msg.get("role")
            if role == "user":
                if current_user is not None:
                    segments.append((current_user, current_assistants))
                current_user = dict(msg)
                current_assistants = []
            elif role == "assistant" and current_user is not None:
                current_assistants.append(dict(msg))

        # Flush last segment.
        if current_user is not None:
            segments.append((current_user, current_assistants))

        return segments

    @staticmethod
    def _assign_model_size(index: int, total: int) -> str:
        """Determine model size for an assistant message within a user-turn segment.

        Rules:
        - Single assistant message: "large"
        - First assistant message: "large"
        - Last assistant message: "large"
        - Middle assistant messages: "small"

        Args:
            index: Zero-based position within assistant messages.
            total: Total number of assistant messages in this segment.

        Returns:
            "large" or "small".
        """
        if total <= 1:
            return "large"
        if index == 0 or index == total - 1:
            return "large"
        return "small"

    @staticmethod
    def _extract_user_content(msg: dict[str, Any]) -> str:
        """Extract text content from a user message.

        Args:
            msg: Raw user message dict.

        Returns:
            Text content string.
        """
        content = msg.get("content")
        if content is not None:
            return str(content)
        return ""

    @staticmethod
    def _extract_assistant_content(msg: dict[str, Any]) -> str:
        """Extract text content from an assistant message.

        Priority: content > thinking > filler (if tool_uses) > empty.

        Args:
            msg: Raw assistant message dict.

        Returns:
            Text content string.
        """
        content = msg.get("content")
        if content is not None and str(content).strip():
            return str(content)

        thinking = msg.get("thinking")
        if thinking is not None and str(thinking).strip():
            return str(thinking)

        tool_uses = msg.get("tool_uses")
        if isinstance(tool_uses, list) and tool_uses:
            return _generate_filler()

        return ""
