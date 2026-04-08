from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field  # type: ignore[reportMissingImports]


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCallMessage(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCallMessage] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolFunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionDef


class ConversationLog(BaseModel):
    task_id: str
    dataset_name: str
    messages: list[Message]
    metadata: dict = Field(default_factory=dict)


class Task(BaseModel):
    task_id: str
    dataset_name: Literal["mcp-atlas", "dataclaw", "swe-bench-pro"]
    prompt: str
    tools: list[ToolDefinition] | None = None
    ground_truth: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    task_id: str
    dataset_name: str
    score: float | None = None
    passed: bool = False
    details: dict = Field(default_factory=dict)
    error: str | None = None


class RunConfig(BaseModel):
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_turns: int = 30
    tool_output_max_chars: int = 16384
    temperature: float = 0.0
    dataset_name: str = ""
    mode: str = ""
    limit: int | None = None
    output_dir: str = "./output"
    workspace: str | None = None
    docker_timeout: int = 1800
