#!/usr/bin/env python3
"""Extract profiling prompts from MCP-Atlas and dataclaw datasets.

Loads the two HuggingFace datasets, flattens multi-turn conversations
into individual inference request prompts with cumulative history,
and saves them as a JSONL file for model profiling.

Each output line contains:
    - messages: list of message dicts (system + cumulative user/assistant history)
    - model_size: "large" or "small" (which model this step targets)
    - dataset: source dataset name
    - group_id: conversation/session identifier
    - step_index: position within the conversation

Usage:
    # Extract 400 prompts from each dataset (800 total)
    uv run python extract_prompts.py --limit 400 --output prompts.jsonl

    # Extract all available prompts
    uv run python extract_prompts.py --output prompts.jsonl

    # Extract only from MCP-Atlas
    uv run python extract_prompts.py --datasets mcp-atlas --output prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping
from typing import Any

from loguru import logger

# ── MCP-Atlas extraction ────────────────────────────────────────


def extract_mcp_atlas(limit: int | None = None) -> list[dict]:
    """Extract profiling prompts from ScaleAI/MCP-Atlas.

    Each step in the trajectory becomes a separate profiling prompt
    with cumulative message history, simulating real inference calls.

    Args:
        limit: Maximum number of prompts to extract.

    Returns:
        List of prompt dicts with messages, model_size, metadata.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading MCP-Atlas from HuggingFace...")
    dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
    prompts: list[dict] = []

    for record in dataset:
        if limit is not None and len(prompts) >= limit:
            break
        if not isinstance(record, Mapping):
            continue

        prompt_text = str(record.get("PROMPT", ""))
        if not prompt_text:
            continue

        trajectory_raw = record.get("TRAJECTORY", "[]")
        if isinstance(trajectory_raw, str):
            try:
                trajectory = json.loads(trajectory_raw)
            except json.JSONDecodeError:
                trajectory = []
        else:
            trajectory = trajectory_raw

        if not isinstance(trajectory, list):
            continue

        task_id = str(record.get("TASK", ""))
        group_id = task_id or f"mcp-atlas-{abs(hash(prompt_text))}"

        # Build cumulative message history step by step.
        cumulative: list[dict[str, str]] = [
            {"role": "user", "content": prompt_text},
        ]

        for i, msg in enumerate(trajectory):
            if not isinstance(msg, Mapping):
                continue
            role = msg.get("role")
            content = str(msg.get("content", ""))

            if role == "assistant":
                model_size = "large"
            elif role == "tool":
                model_size = "small"
            else:
                continue

            # Add this message to history before recording
            # the prompt (the model sees history up to this point).
            cumulative.append({"role": role, "content": content})

            prompts.append(
                {
                    "messages": list(cumulative),
                    "model_size": model_size,
                    "dataset": "mcp-atlas",
                    "group_id": group_id,
                    "step_index": i,
                }
            )

            if limit is not None and len(prompts) >= limit:
                break

    logger.info(f"Extracted {len(prompts)} prompts from MCP-Atlas")
    return prompts


# ── Dataclaw extraction ─────────────────────────────────────────


def _extract_assistant_content(msg: dict[str, Any]) -> str:
    """Extract text from an assistant message.

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
        # Generate short filler to maintain realistic prompt length.
        words = [
            "analyzing",
            "code",
            "structure",
            "checking",
            "dependencies",
            "reviewing",
            "implementation",
            "processing",
            "tool",
            "output",
            "evaluating",
        ]
        return " ".join(random.choices(words, k=random.randint(50, 100)))

    return ""


def extract_dataclaw(limit: int | None = None) -> list[dict]:
    """Extract profiling prompts from peteromallet/dataclaw-peteromallet.

    Each assistant turn becomes a profiling prompt with cumulative
    message history.

    Args:
        limit: Maximum number of prompts to extract.

    Returns:
        List of prompt dicts with messages, model_size, metadata.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading dataclaw from HuggingFace...")
    dataset = load_dataset(
        "peteromallet/dataclaw-peteromallet",
        split="train",
    )
    prompts: list[dict] = []

    for record in dataset:
        if limit is not None and len(prompts) >= limit:
            break
        if not isinstance(record, Mapping):
            continue

        messages = record.get("messages", [])
        if not isinstance(messages, list) or not messages:
            continue

        session_id = str(record.get("session_id", ""))
        group_id = session_id or f"dataclaw-{abs(hash(str(messages[:3])))}"

        # Segment by user turns: (user_msg, [assistant_msgs...])
        segments: list[tuple[dict, list[dict]]] = []
        current_user: dict | None = None
        current_assistants: list[dict] = []

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

        if current_user is not None:
            segments.append((current_user, current_assistants))

        if not segments:
            continue

        # Build cumulative history from segments.
        first_user = segments[0][0]
        first_content = str(first_user.get("content", ""))
        cumulative: list[dict[str, str]] = [
            {"role": "user", "content": first_content},
        ]

        step_idx = 0
        for seg_idx, (user_msg, assistant_msgs) in enumerate(segments):
            # After the first segment, add user message as context.
            if seg_idx > 0:
                user_content = str(user_msg.get("content", ""))
                cumulative.append(
                    {"role": "user", "content": user_content},
                )

            n = len(assistant_msgs)
            for i, asst_msg in enumerate(assistant_msgs):
                content = _extract_assistant_content(asst_msg)

                # Model size assignment (same as dataclaw reshaper).
                if n <= 1 or i == 0 or i == n - 1:
                    model_size = "large"
                else:
                    model_size = "small"

                cumulative.append(
                    {"role": "assistant", "content": content},
                )

                prompts.append(
                    {
                        "messages": list(cumulative),
                        "model_size": model_size,
                        "dataset": "dataclaw",
                        "group_id": group_id,
                        "step_index": step_idx,
                    }
                )
                step_idx += 1

                if limit is not None and len(prompts) >= limit:
                    break

            if limit is not None and len(prompts) >= limit:
                break

        if limit is not None and len(prompts) >= limit:
            break

    logger.info(f"Extracted {len(prompts)} prompts from dataclaw")
    return prompts


# ── Main ────────────────────────────────────────────────────────


def save_prompts(prompts: list[dict], output_path: str) -> None:
    """Save extracted prompts to a JSONL file.

    Args:
        prompts: List of prompt dicts.
        output_path: Output JSONL file path.
    """
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.success(f"Saved {len(prompts)} prompts to {output_path}")


def print_stats(prompts: list[dict]) -> None:
    """Print extraction statistics.

    Args:
        prompts: List of prompt dicts.
    """
    by_dataset: dict[str, int] = {}
    by_model_size: dict[str, int] = {}
    total_tokens_est = 0

    for p in prompts:
        ds = p["dataset"]
        ms = p["model_size"]
        by_dataset[ds] = by_dataset.get(ds, 0) + 1
        by_model_size[ms] = by_model_size.get(ms, 0) + 1
        # Rough token estimate: ~4 chars per token.
        msg_text = " ".join(m["content"] for m in p["messages"])
        total_tokens_est += max(1, len(msg_text) // 4)

    logger.info("─── Extraction Statistics ───")
    logger.info(f"  Total prompts: {len(prompts)}")
    for ds, count in sorted(by_dataset.items()):
        logger.info(f"  {ds}: {count}")
    for ms, count in sorted(by_model_size.items()):
        logger.info(f"  {ms} model: {count}")
    avg_tokens = total_tokens_est // max(1, len(prompts))
    logger.info(f"  Avg estimated tokens/prompt: {avg_tokens}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract profiling prompts from MCP-Atlas and dataclaw.",
    )
    parser.add_argument(
        "--output",
        default="profiling_prompts.jsonl",
        help="Output JSONL file (default: profiling_prompts.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max prompts per dataset (default: unlimited)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mcp-atlas", "dataclaw"],
        choices=["mcp-atlas", "dataclaw"],
        help="Which datasets to extract from",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle prompts before saving",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run prompt extraction pipeline."""
    args = parse_args()
    all_prompts: list[dict] = []

    if "mcp-atlas" in args.datasets:
        all_prompts.extend(extract_mcp_atlas(limit=args.limit))
    if "dataclaw" in args.datasets:
        all_prompts.extend(extract_dataclaw(limit=args.limit))

    if not all_prompts:
        logger.error("No prompts extracted. Check dataset availability.")
        sys.exit(1)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_prompts)

    print_stats(all_prompts)
    save_prompts(all_prompts, args.output)


if __name__ == "__main__":
    main()
