from __future__ import annotations

import pandas as pd

from src.core.csv_handler import CsvHandler
from src.models.config_schema import (
    AppConfig,
    ColumnMappingConfig,
    DatasetConfig,
    ExecutionConfig,
    InputColumnConfig,
    JudgeConfig,
    LLMConfig,
)


def _config(tmp_path, input_path: str, output_path: str, mode: str = "new") -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(input_path=input_path, output_path=output_path, output_mode=mode, encoding="utf-8"),
        column_mapping=ColumnMappingConfig(
            answer_column="expected_answer",
            input_columns=InputColumnConfig(merge=False, columns=["question"], merge_template=""),
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
            checkpoint_path=str(tmp_path / ".cp.json"),
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


def test_load_and_save_new_mode(tmp_path):
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame([{"question": "q1", "expected_answer": "a1"}]).to_csv(input_path, index=False)

    handler = CsvHandler(_config(tmp_path, str(input_path), str(output_path), mode="new"))
    df = handler.load()
    assert "llm_answer" in df.columns
    handler.update_row(df, 0, "llm_answer", "ok")
    handler.save(df)

    saved = pd.read_csv(output_path)
    assert saved.loc[0, "llm_answer"] == "ok"


def test_get_pending_rows(tmp_path):
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.csv"
    pd.DataFrame(
        [
            {"question": "q1", "expected_answer": "a1"},
            {"question": "q2", "expected_answer": "a2"},
        ]
    ).to_csv(input_path, index=False)
    handler = CsvHandler(_config(tmp_path, str(input_path), str(output_path)))
    df = handler.load()
    pending = handler.get_pending_rows(df, {0})
    assert pending == [1]
