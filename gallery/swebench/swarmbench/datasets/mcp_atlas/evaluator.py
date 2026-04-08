from __future__ import annotations

import json
import re

from openai import OpenAI  # type: ignore[reportMissingImports]

from swarmbench.models import ConversationLog, EvalResult, Task


class MCPAtlasEvaluator:
    """Claims-based evaluator for MCP-Atlas final responses."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
    ):
        """Create evaluator backed by OpenAI-compatible judge model.

        Args:
            model: Judge model identifier.
            base_url: OpenAI-compatible base URL.
            api_key: API key for judge service.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def evaluate(self, task: Task, conversation: ConversationLog) -> EvalResult:
        """Evaluate model output by averaging per-claim coverage scores.

        Args:
            task: Benchmark task containing target claims.
            conversation: Full run transcript.

        Returns:
            Evaluation result with score, pass/fail, and per-claim details.
        """
        claims = self._extract_claims(task.ground_truth.get("claims", []))
        if not claims:
            return EvalResult(
                task_id=task.task_id,
                dataset_name="mcp-atlas",
                score=None,
                passed=True,
                details={"reason": "no claims"},
            )

        response = self._get_final_response(conversation)
        if not response:
            return EvalResult(
                task_id=task.task_id,
                dataset_name="mcp-atlas",
                score=0.0,
                passed=False,
                error="No assistant response found",
            )

        per_claim: list[dict] = []
        for claim in claims:
            per_claim.append(self._judge_single_claim(claim, response))

        score_map = {
            "fulfilled": 1.0,
            "partially_fulfilled": 0.5,
            "not_fulfilled": 0.0,
        }
        scores = [
            score_map.get(result.get("coverage_outcome", ""), 0.0)
            for result in per_claim
        ]
        coverage = sum(scores) / len(scores) if scores else 0.0

        return EvalResult(
            task_id=task.task_id,
            dataset_name="mcp-atlas",
            score=round(coverage, 3),
            passed=coverage >= 0.75,
            details={"per_claim": per_claim, "total_claims": len(claims)},
        )

    def _judge_single_claim(self, claim: str, response: str) -> dict:
        """Judge a single claim using the external LLM judge.

        Args:
            claim: One claim to verify.
            response: Model final response text.

        Returns:
            JSON-compatible claim judgment payload.
        """
        prompt = f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed.
- partially_fulfilled: Claim is partially addressed.
- not_fulfilled: Claim is not addressed.
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
Return JSON: {{"claim_text": "...", "coverage_outcome": "fulfilled|partially_fulfilled|not_fulfilled", "justification": "...", "confidence_level": 0.0-1.0}}"""
        try:
            response_obj = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response_obj.choices[0].message.content
            if not content:
                raise ValueError("Empty judge response content")
            return json.loads(content)
        except Exception as exc:
            return {
                "claim_text": claim,
                "coverage_outcome": "not_fulfilled",
                "justification": f"Judge error: {exc}",
                "confidence_level": 0.0,
            }

    def _extract_claims(self, claims_raw: object) -> list[str]:
        """Extract clean claims from JSON, text, or object lists.

        Args:
            claims_raw: Ground-truth claims in mixed possible formats.

        Returns:
            Cleaned list of claim strings.
        """
        if not claims_raw:
            return []

        if isinstance(claims_raw, str):
            try:
                claims_raw = json.loads(claims_raw)
            except json.JSONDecodeError:
                return [line.strip() for line in claims_raw.split("\n") if line.strip()]

        if not isinstance(claims_raw, list):
            return []

        result: list[str] = []
        for item in claims_raw:
            if isinstance(item, dict) and "claim" in item:
                text = str(item["claim"]).strip()
            else:
                text = str(item).strip()
            cleaned = self._clean_claim(text)
            if cleaned and len(cleaned) > 3:
                result.append(cleaned)
        return result

    def _clean_claim(self, text: str) -> str:
        """Normalize a claim by removing list markers.

        Args:
            text: Raw claim text.

        Returns:
            Cleaned claim text.
        """
        text = re.sub(r"^[-*•]\s*", "", text)
        text = re.sub(r"^\d+[.)]\s*", "", text)
        return text.strip()

    def _get_final_response(self, conversation: ConversationLog) -> str | None:
        """Find the latest assistant message with content.

        Args:
            conversation: Conversation transcript.

        Returns:
            Assistant text if present, otherwise None.
        """
        for message in reversed(conversation.messages):
            if message.role == "assistant" and message.content:
                return message.content
        return None
