from __future__ import annotations

from swarmbench.models import ConversationLog, EvalResult, Task


class SWEBenchProEvaluator:
    def evaluate(self, task: Task, conversation: ConversationLog) -> EvalResult:
        fail_to_pass = task.ground_truth.get("fail_to_pass", [])
        pass_to_pass = task.ground_truth.get("pass_to_pass", [])
        if not isinstance(fail_to_pass, list):
            fail_to_pass = []
        if not isinstance(pass_to_pass, list):
            pass_to_pass = []

        passed_tests = conversation.metadata.get("passed_tests", [])
        if not isinstance(passed_tests, list):
            passed_tests = []

        passed_set = {str(item) for item in passed_tests}
        required = {str(item) for item in fail_to_pass} | {
            str(item) for item in pass_to_pass
        }
        resolved = required <= passed_set

        return EvalResult(
            task_id=task.task_id,
            dataset_name="swe-bench-pro",
            score=1.0 if resolved else 0.0,
            passed=resolved,
            details={
                "required_tests": sorted(required),
                "passed_tests": sorted(passed_set),
                "missing_tests": sorted(required - passed_set),
            },
        )
