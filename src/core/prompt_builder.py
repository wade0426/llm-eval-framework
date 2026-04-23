from __future__ import annotations

import math
from string import Formatter

import pandas as pd

from src.models.config_schema import AppConfig


def _safe_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def build_user_prompt(row: pd.Series, config: AppConfig) -> str:
    answer_column = config.column_mapping.answer_column
    answer_value = _safe_to_text(row.get(answer_column, ""))
    input_cfg = config.column_mapping.input_columns

    if input_cfg.merge:
        replacements = {name: _safe_to_text(row.get(name, "")) for name in input_cfg.columns}
        prompt = input_cfg.merge_template.format(**replacements)
    else:
        prompt = _safe_to_text(row.get(input_cfg.columns[0], ""))

    if answer_value and answer_value in prompt:
        raise ValueError("Answer column value must not appear in user prompt")

    return prompt


def build_judge_prompt(question: str, llm_answer: str, expected_answer: str) -> str:
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[LLM Answer]\n"
        f"{llm_answer}\n\n"
        "[Expected Answer]\n"
        f"{expected_answer}\n"
    )


def extract_template_placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}
