from __future__ import annotations

import json
import subprocess
from pathlib import Path

from swarmbench.models import ConversationLog, Task, ToolDefinition, ToolFunctionDef

_SWE_SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks. "
    "Use bash, str_replace_editor, and submit to inspect files, edit code, and finalize your patch."
)

SWE_BENCH_PRO_TOOLS = [
    ToolDefinition(
        function=ToolFunctionDef(
            name="bash",
            description="Run a bash command in the working directory",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="str_replace_editor",
            description="View/create/edit files with commands: view, create, str_replace, insert, undo_edit",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "insert_line": {"type": "integer"},
                },
                "required": ["command", "path"],
            },
        )
    ),
    ToolDefinition(
        function=ToolFunctionDef(
            name="submit",
            description="Finalize and submit the current patch",
            parameters={"type": "object", "properties": {}},
        )
    ),
]


class SWEBenchProTools:
    def __init__(self, workspace: str, dry_run: bool = False):
        self.workspace_root = Path(workspace).resolve()
        self.dry_run = dry_run
        self.submitted_patch: str | None = None
        self._history: list[tuple[Path, str]] = []
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise ValueError(
                f"Workspace does not exist or is not a directory: {workspace}"
            )

    def setup(self) -> None:
        self.submitted_patch = None
        self._history = []

    def teardown(self) -> None:
        self._history = []

    def get_tools(self) -> list[ToolDefinition]:
        return SWE_BENCH_PRO_TOOLS

    def get_system_prompt(self, task: Task) -> str:
        _ = task
        return _SWE_SYSTEM_PROMPT

    def is_complete(self, task: Task, conversation: ConversationLog) -> bool:
        _ = task
        return self.submitted_patch is not None

    def execute(self, tool_name: str, arguments_json: str) -> str:
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
            return self._bash(str(arguments.get("command", "")))
        if tool_name == "str_replace_editor":
            return self._str_replace_editor(arguments)
        if tool_name == "submit":
            return self._submit()
        raise ValueError(f"Unknown tool: {tool_name}")

    def _bash(self, command: str) -> str:
        completed = subprocess.run(  # noqa: S603
            ["bash", "-lc", command],  # noqa: S607
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return output or f"[exit_code={completed.returncode}]"

    def _str_replace_editor(self, arguments: dict) -> str:
        command = str(arguments.get("command", ""))
        path = self._resolve_workspace_path(str(arguments.get("path", "")))

        if command == "view":
            return path.read_text(encoding="utf-8")
        if command == "create":
            file_text = str(arguments.get("file_text", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(file_text, encoding="utf-8")
            return f"Created {path.relative_to(self.workspace_root)}"
        if command == "str_replace":
            old_str = str(arguments.get("old_str", ""))
            new_str = str(arguments.get("new_str", ""))
            original = path.read_text(encoding="utf-8")
            self._history.append((path, original))
            if old_str not in original:
                raise ValueError(f"Text not found in {path.name}")
            updated = original.replace(old_str, new_str)
            path.write_text(updated, encoding="utf-8")
            return f"Updated {path.relative_to(self.workspace_root)}"
        if command == "insert":
            insert_line = int(arguments.get("insert_line", 0))
            new_str = str(arguments.get("new_str", ""))
            original = path.read_text(encoding="utf-8")
            self._history.append((path, original))
            lines = original.splitlines()
            lines.insert(max(0, min(insert_line, len(lines))), new_str)
            path.write_text(
                "\n".join(lines) + ("\n" if original.endswith("\n") else ""),
                encoding="utf-8",
            )
            return f"Inserted into {path.relative_to(self.workspace_root)}"
        if command == "undo_edit":
            for index in range(len(self._history) - 1, -1, -1):
                hist_path, hist_content = self._history[index]
                if hist_path == path:
                    path.write_text(hist_content, encoding="utf-8")
                    self._history.pop(index)
                    return f"Undid edit in {path.relative_to(self.workspace_root)}"
            raise ValueError(f"No edit history for {path.name}")

        raise ValueError(f"Unknown str_replace_editor command: {command}")

    def _submit(self) -> str:
        if self.dry_run:
            self.submitted_patch = "[dry-run submit]"
            return self.submitted_patch
        completed = subprocess.run(  # noqa: S603
            ["bash", "-lc", "git diff --no-ext-diff --binary"],  # noqa: S607
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        patch = ((completed.stdout or "") + (completed.stderr or "")).strip()
        self.submitted_patch = patch
        return patch or "[empty patch]"

    def _resolve_workspace_path(self, candidate_path: str) -> Path:
        raw_path = Path(candidate_path)
        if raw_path.is_absolute():
            raise ValueError("Absolute paths are not allowed")
        resolved = (self.workspace_root / raw_path).resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError("Path escapes workspace root") from exc
        return resolved
