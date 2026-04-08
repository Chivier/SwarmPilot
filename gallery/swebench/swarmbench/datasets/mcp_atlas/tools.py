from __future__ import annotations

import json
from collections import defaultdict
from typing import Any
from urllib import error, request

from swarmbench.models import ConversationLog, Task, ToolDefinition, ToolFunctionDef

_MCP_ATLAS_SYSTEM_PROMPT = (
    "You are a factual, tool-aware assistant connected to a variety of tools. "
    "Use the available tools to answer the user query. Do not ask the user for "
    "clarification; fully complete the task using the information provided in the prompt."
)


class MockMCPTools:
    """Replay tool outputs from the MCP-Atlas gold trajectory."""

    def __init__(self, task: Task):
        """Initialize provider state for one task.

        Args:
            task: MCP-Atlas task with enabled tools and trajectory metadata.
        """
        self.task = task
        self._responses_by_tool: dict[str, list[str]] = {}

    def setup(self) -> None:
        """Build per-tool response queues from trajectory messages."""
        trajectory = self.task.metadata.get("trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []

        call_id_to_tool: dict[str, str] = {}
        queue: defaultdict[str, list[str]] = defaultdict(list)

        for message in trajectory:
            if not isinstance(message, dict):
                continue

            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function_def = tool_call.get("function")
                    if not isinstance(function_def, dict):
                        continue
                    call_id = tool_call.get("id")
                    tool_name = function_def.get("name")
                    if isinstance(call_id, str) and isinstance(tool_name, str):
                        call_id_to_tool[call_id] = tool_name

            if message.get("role") != "tool":
                continue

            tool_name = message.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    tool_name = call_id_to_tool.get(tool_call_id, "")

            if not tool_name:
                continue

            content = message.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content)

            queue[tool_name].append(content)

        self._responses_by_tool = dict(queue)

    def execute(self, tool_name: str, arguments_json: str) -> str:
        """Return the next replayed tool response for the tool.

        Args:
            tool_name: Requested tool name.
            arguments_json: Serialized tool arguments (unused in mock mode).

        Returns:
            Next queued tool output or deterministic error text when absent.
        """
        _ = arguments_json
        queue = self._responses_by_tool.get(tool_name, [])
        if not queue:
            return f"Error: no replay response available for tool '{tool_name}'."
        return queue.pop(0)

    def get_tools(self) -> list[ToolDefinition]:
        """Build OpenAI tool definitions from enabled tool names.

        Returns:
            Tool definitions with minimal object schemas.
        """
        enabled_tools = self.task.metadata.get("enabled_tools", [])
        if not isinstance(enabled_tools, list):
            enabled_tools = []

        tools: list[ToolDefinition] = []
        for entry in enabled_tools:
            tool_name = self._extract_tool_name(entry)
            if not tool_name:
                continue
            tools.append(
                ToolDefinition(
                    function=ToolFunctionDef(
                        name=tool_name,
                        description=f"MCP-Atlas tool: {tool_name}",
                        parameters={"type": "object", "properties": {}},
                    )
                )
            )
        return tools

    def get_system_prompt(self, task: Task) -> str:
        """Return the benchmark system prompt.

        Args:
            task: Current task (unused).

        Returns:
            Canonical MCP-Atlas instruction prompt.
        """
        _ = task
        return _MCP_ATLAS_SYSTEM_PROMPT

    def is_complete(self, task: Task, conversation: ConversationLog) -> bool:
        """Check completion condition from latest assistant message.

        Args:
            task: Current task (unused).
            conversation: Current conversation transcript.

        Returns:
            True when the last assistant message has no tool calls.
        """
        _ = task
        for message in reversed(conversation.messages):
            if message.role == "assistant":
                return not bool(message.tool_calls)
        return False

    def teardown(self) -> None:
        """Clear replay state after run completion."""
        self._responses_by_tool = {}

    def _extract_tool_name(self, entry: Any) -> str:
        """Extract a tool name from metadata entry.

        Args:
            entry: Enabled-tools list item.

        Returns:
            Tool name when present, otherwise empty string.
        """
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            candidate = entry.get("name")
            if isinstance(candidate, str):
                return candidate
            function_block = entry.get("function")
            if isinstance(function_block, dict):
                function_name = function_block.get("name")
                if isinstance(function_name, str):
                    return function_name
        return ""


class RealMCPTools(MockMCPTools):
    """Call a local MCP-Atlas REST tool endpoint."""

    def __init__(self, task: Task, port: int = 8000):
        """Initialize provider for online tool execution.

        Args:
            task: MCP-Atlas task definition.
            port: Local MCP service port.
        """
        super().__init__(task)
        self.port = port

    def execute(self, tool_name: str, arguments_json: str) -> str:
        """Execute a tool via HTTP POST against local MCP service.

        Args:
            tool_name: Tool name to invoke.
            arguments_json: JSON-encoded tool args from model tool call.

        Returns:
            Raw response body as text.

        Raises:
            RuntimeError: If request fails or endpoint returns an error status.
            ValueError: If arguments_json is invalid JSON.
        """
        try:
            tool_args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid tool arguments for '{tool_name}': {arguments_json}"
            ) from exc

        payload = json.dumps({"tool_name": tool_name, "tool_args": tool_args}).encode(
            "utf-8"
        )
        endpoint = f"http://localhost:{self.port}/call-tool"
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"MCP tool call failed for '{tool_name}' with status {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"MCP tool call failed for '{tool_name}' at {endpoint}: {exc.reason}"
            ) from exc
