from __future__ import annotations

from src.core.judge import JudgeEvaluator
from src.models.config_schema import JudgeConfig


class _FakeClient:
    def __init__(self, response: str):
        self.response = response

    def call(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


def _judge_config() -> JudgeConfig:
    return JudgeConfig(
        enabled=True,
        base_url="https://api.openai.com/v1",
        api_key="k",
        model="gpt-4o",
        output_column="judge_score",
        system_prompt="judge",
        temperature=0,
        max_tokens=16,
        timeout_seconds=30,
    )


def test_judge_parse_success():
    evaluator = JudgeEvaluator(_judge_config(), max_retries=0)
    evaluator.client = _FakeClient('{"score": 1, "reason": "ok"}')
    out = evaluator.evaluate("q", "a", "e")
    assert out["score"] == 1


def test_judge_parse_error_fallback():
    evaluator = JudgeEvaluator(_judge_config(), max_retries=0)
    evaluator.client = _FakeClient("not json")
    out = evaluator.evaluate("q", "a", "e")
    assert out["score"] == -1
    assert out["reason"].startswith("PARSE_ERROR:")
