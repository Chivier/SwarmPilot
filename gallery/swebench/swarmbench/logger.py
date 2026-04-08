from __future__ import annotations

import json
from pathlib import Path

from swarmbench.models import ConversationLog


class ConversationLogger:
    """Persist and retrieve conversation logs on disk."""

    def __init__(self, output_dir: str) -> None:
        """Initialize logger with a base output directory.

        Args:
            output_dir: Root directory where dataset/task logs are stored.

        Returns:
            None.
        """
        self.output_dir = Path(output_dir)

    def _file_path(self, dataset_name: str, task_id: str) -> Path:
        """Build log file path for a dataset/task pair.

        Args:
            dataset_name: Dataset identifier.
            task_id: Task identifier.

        Returns:
            Path to the JSONL log file.
        """
        return self.output_dir / dataset_name / f"{task_id}.jsonl"

    def save(self, log: ConversationLog) -> Path:
        """Write a conversation log as a single JSON line.

        Args:
            log: ConversationLog model to persist.

        Returns:
            Path where the log was written.
        """
        file_path = self._file_path(log.dataset_name, log.task_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(log.model_dump()))
            handle.write("\n")
        return file_path

    def load(self, dataset_name: str, task_id: str) -> ConversationLog:
        """Load one conversation log from disk.

        Args:
            dataset_name: Dataset identifier.
            task_id: Task identifier.

        Returns:
            Deserialized ConversationLog model.
        """
        file_path = self._file_path(dataset_name, task_id)
        with file_path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
        return ConversationLog.model_validate(json.loads(line))

    def load_all(self, dataset_name: str) -> list[ConversationLog]:
        """Load all conversation logs for a dataset directory.

        Args:
            dataset_name: Dataset identifier.

        Returns:
            List of deserialized ConversationLog models.
        """
        dataset_dir = self.output_dir / dataset_name
        if not dataset_dir.exists():
            return []

        logs: list[ConversationLog] = []
        for file_path in sorted(dataset_dir.glob("*.jsonl")):
            with file_path.open("r", encoding="utf-8") as handle:
                line = handle.readline().strip()
            logs.append(ConversationLog.model_validate(json.loads(line)))
        return logs
