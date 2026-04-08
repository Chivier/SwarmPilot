from __future__ import annotations

from typing import Protocol

from swarmbench.models import ConversationLog, EvalResult, Task, ToolDefinition


class TaskLoader(Protocol):
    def load(self, limit: int | None = None) -> list[Task]: ...


class ToolProvider(Protocol):
    def get_tools(self) -> list[ToolDefinition]: ...

    def get_system_prompt(self, task: Task) -> str: ...

    def execute(self, tool_name: str, arguments_json: str) -> str: ...

    def is_complete(self, task: Task, conversation: ConversationLog) -> bool: ...

    def setup(self) -> None: ...

    def teardown(self) -> None: ...


class ResultEvaluator(Protocol):
    def evaluate(self, task: Task, conversation: ConversationLog) -> EvalResult: ...
