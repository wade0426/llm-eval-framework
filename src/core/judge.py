from __future__ import annotations

import json
import logging

from src.core.llm_client import LLMClient
from src.core.prompt_builder import build_judge_prompt
from src.models.config_schema import JudgeConfig


LOGGER = logging.getLogger("llm_eval")


class JudgeEvaluator:
    def __init__(self, judge_config: JudgeConfig, max_retries: int = 3, retry_delay_seconds: float = 2):
        self.judge_config = judge_config
        self.client = LLMClient(judge_config, max_retries=max_retries, retry_delay_seconds=retry_delay_seconds)

    def evaluate(self, question: str, llm_answer: str, expected_answer: str) -> dict:
        prompt = build_judge_prompt(question=question, llm_answer=llm_answer, expected_answer=expected_answer)
        raw = self.client.call(system_prompt=self.judge_config.system_prompt, user_prompt=prompt)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Judge result must be a JSON object")
            return parsed
        except Exception:
            LOGGER.warning("Judge response parse error: %s", raw)
            return {"score": -1, "reason": f"PARSE_ERROR: {raw}"}
