from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset  # type: ignore[reportMissingImports]

from swarmbench.models import Task


class SWEBenchProLoader:
    def load(self, limit: int | None = None) -> list[Task]:
        dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
        tasks: list[Task] = []
        for index, record in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            if isinstance(record, Mapping):
                tasks.append(self._normalize(dict(record)))
        return tasks

    def _normalize(self, record: dict[str, Any]) -> Task:
        fail_to_pass = self._parse_json_field(
            record.get("fail_to_pass", []), default=[]
        )
        pass_to_pass = self._parse_json_field(
            record.get("pass_to_pass", []), default=[]
        )
        patch = record.get("patch", "")
        test_patch = record.get("test_patch", "")

        return Task(
            task_id=str(record.get("instance_id", "")),
            dataset_name="swe-bench-pro",
            prompt=str(record.get("problem_statement", "")),
            ground_truth={
                "patch": patch if isinstance(patch, str) else str(patch),
                "test_patch": test_patch
                if isinstance(test_patch, str)
                else str(test_patch),
                "fail_to_pass": fail_to_pass,
                "pass_to_pass": pass_to_pass,
            },
            metadata={
                "repo": record.get("repo"),
                "base_commit": record.get("base_commit"),
                "dockerhub_tag": record.get("dockerhub_tag"),
                "repo_language": record.get("repo_language"),
                "before_repo_set_cmd": record.get("before_repo_set_cmd"),
                "_swebench_instance": dict(record),
            },
        )

    def _parse_json_field(self, value: Any, default: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value
