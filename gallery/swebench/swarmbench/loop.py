from __future__ import annotations

from swarmbench.datasets.base import ToolProvider
from swarmbench.llm import LLMClient
from swarmbench.models import ConversationLog, Message, RunConfig, Task


def truncate_output(text: str, max_chars: int) -> str:
    """Truncate tool output to a configured character budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated at {max_chars} chars]"


def run_task(
    task: Task,
    llm: LLMClient,
    tools: ToolProvider,
    config: RunConfig,
) -> ConversationLog:
    """Run one task through the multi-turn LLM+tool execution loop."""
    messages: list[Message] = [
        Message(role="system", content=tools.get_system_prompt(task)),
        Message(role="user", content=task.prompt),
    ]

    tools.setup()
    try:
        for _ in range(config.max_turns):
            tool_defs = tools.get_tools()
            response = llm.chat(messages, tool_defs if tool_defs else None)
            messages.append(response)

            if not response.tool_calls:
                break

            conversation = ConversationLog(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                messages=messages,
            )
            if tools.is_complete(task, conversation):
                break

            for tool_call in response.tool_calls:
                try:
                    tool_result = tools.execute(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    tool_result = truncate_output(
                        tool_result, config.tool_output_max_chars
                    )
                except Exception as exc:
                    tool_result = (
                        f"Error executing {tool_call.function.name}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                    )
                )
    finally:
        tools.teardown()

    return ConversationLog(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        messages=messages,
    )
