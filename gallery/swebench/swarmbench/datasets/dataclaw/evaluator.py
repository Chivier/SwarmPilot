from __future__ import annotations

import json
from typing import Any

from openai import OpenAI  # type: ignore[reportMissingImports]

from swarmbench.models import ConversationLog, EvalResult, Task


class DataclawEvaluator:
    """Evaluate Dataclaw runs using trajectory and response similarity."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        pass_threshold: float = 0.7,
    ):
        """Configure the LLM judge backend and pass threshold.

        Args:
            model: Judge model identifier.
            base_url: OpenAI-compatible API base URL.
            api_key: API key for judge model.
            pass_threshold: Minimum weighted score required to pass.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.pass_threshold = pass_threshold

    def evaluate(self, task: Task, conversation: ConversationLog) -> EvalResult:
        """Compute weighted score using tool trajectory and response quality.

        Args:
            task: Benchmark task with Dataclaw reference outputs.
            conversation: Candidate run conversation transcript.

        Returns:
            Evaluation result with weighted score and detailed breakdown.
        """
        reference_trajectory = task.ground_truth.get("reference_tool_trajectory", [])
        if not isinstance(reference_trajectory, list):
            reference_trajectory = []
        reference_response = task.ground_truth.get("reference_response")
        if not isinstance(reference_response, str):
            reference_response = ""

        candidate_trajectory = self._extract_candidate_trajectory(conversation)
        candidate_response = self._extract_candidate_response(conversation)

        try:
            trajectory_score, trajectory_reason = self._score_trajectory(
                candidate_trajectory=candidate_trajectory,
                reference_trajectory=reference_trajectory,
            )
            response_score, response_reason = self._score_response(
                candidate_response=candidate_response,
                reference_response=reference_response,
            )
        except Exception as exc:
            return EvalResult(
                task_id=task.task_id,
                dataset_name="dataclaw",
                score=0.0,
                passed=False,
                error=f"Evaluator error: {type(exc).__name__}: {exc}",
            )

        weighted_score = round((0.7 * trajectory_score) + (0.3 * response_score), 3)
        return EvalResult(
            task_id=task.task_id,
            dataset_name="dataclaw",
            score=weighted_score,
            passed=weighted_score >= self.pass_threshold,
            details={
                "trajectory_score": trajectory_score,
                "response_score": response_score,
                "trajectory_reason": trajectory_reason,
                "response_reason": response_reason,
                "candidate_tool_trajectory": candidate_trajectory,
                "reference_tool_trajectory": reference_trajectory,
                "candidate_response": candidate_response,
                "reference_response": reference_response,
            },
        )

    def _extract_candidate_trajectory(
        self, conversation: ConversationLog
    ) -> list[dict[str, str]]:
        """Extract ordered candidate tool calls from assistant messages.

        Args:
            conversation: Candidate run transcript.

        Returns:
            Ordered tool trajectory represented as tool/input pairs.
        """
        ordered_calls: list[dict[str, str]] = []
        for message in conversation.messages:
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                ordered_calls.append(
                    {
                        "tool": tool_call.function.name,
                        "input": tool_call.function.arguments,
                    }
                )
        return ordered_calls

    def _extract_candidate_response(self, conversation: ConversationLog) -> str:
        """Extract the latest assistant response text from conversation.

        Args:
            conversation: Candidate run transcript.

        Returns:
            Last assistant textual response, or empty string when absent.
        """
        for message in reversed(conversation.messages):
            if message.role == "assistant" and message.content:
                return message.content
        return ""

    def _score_trajectory(
        self,
        candidate_trajectory: list[dict[str, str]],
        reference_trajectory: list[dict[str, str]],
    ) -> tuple[float, str]:
        """Score candidate trajectory against reference trajectory.

        Args:
            candidate_trajectory: Model-produced ordered tool calls.
            reference_trajectory: Dataset reference tool calls.

        Returns:
            Judge score and reason for trajectory quality.
        """
        if not reference_trajectory and not candidate_trajectory:
            return 1.0, "Both trajectories are empty"
        if reference_trajectory and not candidate_trajectory:
            return 0.0, "Candidate has no tool trajectory"
        if candidate_trajectory and not reference_trajectory:
            return 0.0, "Reference has no tool trajectory"

        return self._judge_similarity(
            comparison_name="tool trajectory",
            reference=reference_trajectory,
            candidate=candidate_trajectory,
        )

    def _score_response(
        self, candidate_response: str, reference_response: str
    ) -> tuple[float, str]:
        """Score candidate final response against reference response.

        Args:
            candidate_response: Candidate final assistant response.
            reference_response: Dataset reference assistant response.

        Returns:
            Judge score and reason for response quality.
        """
        if not reference_response and not candidate_response:
            return 1.0, "Both responses are empty"
        if reference_response and not candidate_response:
            return 0.0, "Candidate has no final response"
        if candidate_response and not reference_response:
            return 0.0, "Reference has no final response"

        return self._judge_similarity(
            comparison_name="final response",
            reference=reference_response,
            candidate=candidate_response,
        )

    def _judge_similarity(
        self, comparison_name: str, reference: Any, candidate: Any
    ) -> tuple[float, str]:
        """Use LLM judge to produce similarity score in [0.0, 1.0].

        Args:
            comparison_name: Label describing what is being compared.
            reference: Ground-truth value.
            candidate: Candidate value.

        Returns:
            Tuple of bounded score and brief reason from judge.
        """
        prompt = (
            "You are an evaluator for agent benchmark runs. "
            f"Compare the candidate {comparison_name} to the reference {comparison_name}. "
            "Output strict JSON with keys: score (float between 0 and 1), reason (string). "
            "Higher score means higher semantic and procedural similarity.\n\n"
            f"Reference {comparison_name}:\n{json.dumps(reference, ensure_ascii=False, indent=2)}\n\n"
            f"Candidate {comparison_name}:\n{json.dumps(candidate, ensure_ascii=False, indent=2)}"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        payload = completion.choices[0].message.content
        if not payload:
            raise ValueError(f"Judge returned empty {comparison_name} evaluation")

        parsed = json.loads(payload)
        score = parsed.get("score", 0.0)
        reason = parsed.get("reason", "")
        bounded = max(0.0, min(1.0, float(score)))
        reason_text = str(reason) if reason else ""
        return bounded, reason_text
