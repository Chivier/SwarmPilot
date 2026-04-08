from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from swarmbench.models import ConversationLog, Task, ToolDefinition, ToolFunctionDef

DATACLAW_TOOLS = [
    ToolDefinition(
        function=ToolFunctionDef(
            name="bash",
            description="Run a bash command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="edit_file",
            description="Replace text in a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="glob_files",
            description="Find files matching a pattern",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="grep_files",
            description="Search for a pattern in files",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        )
    ),
]

_DATACLAW_SYSTEM_PROMPT = (
    "You are a coding assistant. Use the provided tools to inspect, modify, and run "
    "code in the workspace. Complete the user's task fully and return a concise final "
    "answer once no more tool calls are needed."
)


class TrajectoryOnlyTools:
    """Tool provider for trajectory-only Dataclaw evaluation mode."""

    def setup(self) -> None:
        """Prepare provider state before task execution."""

    def teardown(self) -> None:
        """Clean up provider state after task execution."""

    def execute(self, tool_name: str, arguments_json: str) -> str:
        """Return a placeholder indicating no live execution occurs.

        Args:
            tool_name: Requested tool name.
            arguments_json: Serialized tool arguments.

        Returns:
            Deterministic placeholder output.
        """
        _ = (tool_name, arguments_json)
        return "[trajectory-only mode: tool execution skipped]"

    def get_tools(self) -> list[ToolDefinition]:
        """Return Dataclaw tool schemas exposed to the model.

        Returns:
            OpenAI-style function tool definitions.
        """
        return DATACLAW_TOOLS

    def get_system_prompt(self, task: Task) -> str:
        """Return the Dataclaw benchmark system prompt.

        Args:
            task: Current task context (unused).

        Returns:
            Coding-assistant instruction prompt.
        """
        _ = task
        return _DATACLAW_SYSTEM_PROMPT

    def is_complete(self, task: Task, conversation: ConversationLog) -> bool:
        """Determine completion from the latest assistant turn.

        Args:
            task: Current task context (unused).
            conversation: Conversation transcript.

        Returns:
            True when the latest assistant message has no tool calls.
        """
        _ = task
        for message in reversed(conversation.messages):
            if message.role == "assistant":
                return not bool(message.tool_calls)
        return False


class CodingTools(TrajectoryOnlyTools):
    """Tool provider for Dataclaw live execution within a workspace root."""

    def __init__(self, workspace: str):
        """Store and validate the workspace root path.

        Args:
            workspace: Root directory where all operations are constrained.
        """
        self.workspace_root = Path(workspace).resolve()
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise ValueError(
                f"Workspace does not exist or is not a directory: {workspace}"
            )

    def execute(self, tool_name: str, arguments_json: str) -> str:
        """Dispatch Dataclaw tool calls to concrete local implementations.

        Args:
            tool_name: Tool name to execute.
            arguments_json: JSON-encoded tool arguments.

        Returns:
            Tool execution output.

        Raises:
            ValueError: If arguments are invalid JSON or tool name is unknown.
        """
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid tool arguments for '{tool_name}': {arguments_json}"
            ) from exc

        if not isinstance(arguments, dict):
            raise ValueError(
                f"Invalid tool arguments for '{tool_name}': expected JSON object"
            )

        if tool_name == "bash":
            return self.bash(str(arguments.get("command", "")))
        if tool_name == "read_file":
            return self.read_file(str(arguments.get("path", "")))
        if tool_name == "write_file":
            return self.write_file(
                str(arguments.get("path", "")), str(arguments.get("content", ""))
            )
        if tool_name == "edit_file":
            return self.edit_file(
                str(arguments.get("path", "")),
                str(arguments.get("old_str", "")),
                str(arguments.get("new_str", "")),
            )
        if tool_name == "glob_files":
            return self.glob_files(str(arguments.get("pattern", "")))
        if tool_name == "grep_files":
            return self.grep_files(
                str(arguments.get("pattern", "")), str(arguments.get("path", "."))
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    def bash(self, command: str) -> str:
        """Execute a shell command inside the workspace root.

        Args:
            command: Bash command string to run.

        Returns:
            Combined stdout and stderr output.
        """
        completed = subprocess.run(  # noqa: S603
            ["bash", "-lc", command],  # noqa: S607
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        output = output.strip()
        if output:
            return output
        return f"[exit_code={completed.returncode}]"

    def read_file(self, path: str) -> str:
        """Read file contents constrained to workspace root.

        Args:
            path: Workspace-relative file path.

        Returns:
            File text content.
        """
        target = self._resolve_workspace_path(path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 content to a file within workspace root.

        Args:
            path: Workspace-relative file path.
            content: File contents to write.

        Returns:
            Status text describing the write operation.
        """
        target = self._resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Replace text in a file constrained to workspace root.

        Args:
            path: Workspace-relative file path.
            old_str: Text to replace.
            new_str: Replacement text.

        Returns:
            Status text describing replacement count.

        Raises:
            ValueError: If old_str does not exist in the file.
        """
        if not old_str:
            raise ValueError("old_str must not be empty")

        target = self._resolve_workspace_path(path)
        original = target.read_text(encoding="utf-8")
        occurrences = original.count(old_str)
        if occurrences == 0:
            raise ValueError(f"Text not found in {path}")

        updated = original.replace(old_str, new_str)
        target.write_text(updated, encoding="utf-8")
        return f"Replaced {occurrences} occurrence(s) in {path}"

    def glob_files(self, pattern: str) -> str:
        """Find paths matching a glob pattern under workspace root.

        Args:
            pattern: Glob pattern relative to workspace root.

        Returns:
            Newline-separated relative file paths.
        """
        matches = sorted(
            str(path.relative_to(self.workspace_root))
            for path in self.workspace_root.glob(pattern)
        )
        return "\n".join(matches)

    def grep_files(self, pattern: str, path: str = ".") -> str:
        """Search text files for a regex pattern under workspace root.

        Args:
            pattern: Regular expression pattern.
            path: Relative file or directory path to search.

        Returns:
            Newline-separated match rows as path:line:content.
        """
        search_root = self._resolve_workspace_path(path)
        regex = re.compile(pattern)

        if search_root.is_file():
            files = [search_root]
        else:
            files = [
                candidate for candidate in search_root.rglob("*") if candidate.is_file()
            ]

        rows: list[str] = []
        for candidate in files:
            try:
                content = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    relative = candidate.relative_to(self.workspace_root)
                    rows.append(f"{relative}:{line_number}:{line}")
        return "\n".join(rows)

    def _resolve_workspace_path(self, candidate_path: str) -> Path:
        """Resolve and validate a workspace-relative path.

        Args:
            candidate_path: Relative path supplied by tool arguments.

        Returns:
            Absolute resolved path within workspace root.

        Raises:
            ValueError: If the path is absolute or escapes workspace root.
        """
        raw_path = Path(candidate_path)
        if raw_path.is_absolute():
            raise ValueError("Absolute paths are not allowed")

        resolved = (self.workspace_root / raw_path).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace root") from exc
        return resolved
