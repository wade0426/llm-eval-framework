from __future__ import annotations

import pandas as pd
import pytest

from src.core.prompt_builder import build_judge_prompt, build_user_content, build_user_prompt
from src.models.config_schema import (
    AppConfig,
    ColumnMappingConfig,
    DatasetConfig,
    ExecutionConfig,
    InputColumnConfig,
    JudgeConfig,
    LLMConfig,
    OutputMode,
)


def _config(merge: bool = False) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(input_path="in.csv", output_path="out.csv", output_mode=OutputMode.NEW, encoding="utf-8"),
        column_mapping=ColumnMappingConfig(
            answer_column="expected_answer",
            input_columns=InputColumnConfig(
                merge=merge,
                columns=["question", "context"] if merge else ["question"],
                merge_template="{question}\n\nContext: {context}" if merge else "",
            ),
            output_column="llm_answer",
        ),
        primary_llm=LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model="gpt-4o",
            system_prompt="s",
            temperature=0,
            max_tokens=16,
            timeout_seconds=30,
        ),
        execution=ExecutionConfig(
            max_retries=3,
            retry_delay_seconds=2,
            rate_limit_rpm=0,
            checkpoint_path=".cp.json",
            batch_log_interval=10,
        ),
        judge=JudgeConfig(
            enabled=False,
            base_url="https://api.openai.com/v1",
            api_key="k",
            model="gpt-4o",
            output_column="judge_score",
            system_prompt="j",
            temperature=0,
            max_tokens=16,
            timeout_seconds=30,
        ),
    )


def test_build_user_prompt_single_column():
    row = pd.Series({"question": "What is 2+2?", "expected_answer": "4"})
    prompt = build_user_prompt(row, _config(False))
    assert prompt == "What is 2+2?"


def test_build_user_prompt_merge_mode():
    row = pd.Series({"question": "Q", "context": "C", "expected_answer": "A"})
    prompt = build_user_prompt(row, _config(True))
    assert prompt == "Q\n\nContext: C"


def test_build_user_prompt_answer_leak_detected():
    row = pd.Series({"question": "SECRET", "expected_answer": "SECRET"})
    with pytest.raises(ValueError):
        build_user_prompt(row, _config(False))


def test_build_judge_prompt():
    prompt = build_judge_prompt("Q", "A", "E")
    assert "Q" in prompt
    assert "A" in prompt
    assert "E" in prompt


def test_build_user_content_without_images_returns_string():
    content = build_user_content("What is in this image?", None)
    assert content == "What is in this image?"


def test_build_user_content_with_images_returns_parts_and_text_first():
    image_parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx", "detail": "auto"}}]
    content = build_user_content("Describe", image_parts)
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe"}
    assert content[1:] == image_parts
