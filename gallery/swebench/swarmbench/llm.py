from __future__ import annotations

import json
import uuid
from typing import Any

from openai import OpenAI  # type: ignore[reportMissingImports]

from swarmbench.models import (
    Message,
    RunConfig,
    ToolCallFunction,
    ToolCallMessage,
    ToolDefinition,
)


class LLMClient:
    """OpenAI-compatible chat client for non-streaming completions."""

    def __init__(self, config: RunConfig):
        """Initialize SDK client from run configuration.

        Args:
            config: Runtime configuration with model, base URL, and API key.
        """
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model
        self.temperature = config.temperature

    @staticmethod
    def _message_to_api(message: Message) -> dict[str, Any]:
        """Convert internal message model into OpenAI API payload shape.

        Args:
            message: Internal message object.

        Returns:
            Dict payload compatible with chat completions API.
        """
        payload = message.model_dump(exclude_none=True)
        if message.tool_calls:
            payload["tool_calls"] = [
                tool_call.model_dump(exclude_none=True)
                for tool_call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _tools_to_api(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert internal tool definitions into OpenAI API tool payloads.

        Args:
            tools: Internal tool definitions.

        Returns:
            Tool payload list for chat completions.
        """
        return [tool.model_dump(exclude_none=True) for tool in tools]

    @staticmethod
    def _coerce_tool_arguments(arguments: Any) -> str:
        """Normalize tool arguments from SDK response into JSON string.

        Args:
            arguments: Tool argument value from OpenAI SDK response.

        Returns:
            JSON string representation accepted by internal tool-call model.
        """
        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            return "{}"
        return json.dumps(arguments)

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: Any) -> list[ToolCallMessage] | None:
        """Parse SDK tool-call objects into internal canonical models.

        Args:
            raw_tool_calls: OpenAI SDK message.tool_calls value.

        Returns:
            Parsed tool calls, or None when absent.
        """
        if not raw_tool_calls:
            return None

        tool_calls: list[ToolCallMessage] = []
        for raw_call in raw_tool_calls:
            function_obj = getattr(raw_call, "function", None)
            if function_obj is None and isinstance(raw_call, dict):
                function_obj = raw_call.get("function")

            name = getattr(function_obj, "name", None)
            if name is None and isinstance(function_obj, dict):
                name = function_obj.get("name")

            arguments = getattr(function_obj, "arguments", None)
            if arguments is None and isinstance(function_obj, dict):
                arguments = function_obj.get("arguments")

            call_id = getattr(raw_call, "id", None)
            if call_id is None and isinstance(raw_call, dict):
                call_id = raw_call.get("id")

            call_type = getattr(raw_call, "type", None)
            if call_type is None and isinstance(raw_call, dict):
                call_type = raw_call.get("type")

            tool_calls.append(
                ToolCallMessage(
                    id=call_id or str(uuid.uuid4()),
                    type=call_type or "function",
                    function=ToolCallFunction(
                        name=name or "",
                        arguments=LLMClient._coerce_tool_arguments(arguments),
                    ),
                )
            )

        return tool_calls

    def chat(
        self, messages: list[Message], tools: list[ToolDefinition] | None = None
    ) -> Message:
        """Call OpenAI-compatible chat completions and return assistant message.

        Args:
            messages: Conversation history in canonical message model.
            tools: Optional tool definitions to expose during this call.

        Returns:
            Assistant message parsed into the internal message model.

        Raises:
            RuntimeError: If the OpenAI API call fails.
        """
        api_messages = [self._message_to_api(message) for message in messages]
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        if tools:
            request_kwargs["tools"] = self._tools_to_api(tools)

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise RuntimeError(f"LLM chat completion failed: {exc}") from exc

        api_message = response.choices[0].message
        content = getattr(api_message, "content", None)
        tool_calls = self._parse_tool_calls(getattr(api_message, "tool_calls", None))

        return Message(role="assistant", content=content, tool_calls=tool_calls)
