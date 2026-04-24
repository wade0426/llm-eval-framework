from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.runner import run
from src.models.config_schema import (
    AppConfig,
    ColumnMappingConfig,
    ConversationLogConfig,
    DatasetConfig,
    ExecutionConfig,
    ImageDetail,
    ImageConfig,
    InputColumnConfig,
    JudgeConfig,
    LLMConfig,
    OutputMode,
)


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(
            input_path=str(tmp_path / "in.csv"),
            output_path=str(tmp_path / "out.csv"),
            output_mode=OutputMode.NEW,
            encoding="utf-8",
        ),
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
            max_retries=0,
            retry_delay_seconds=0,
            rate_limit_rpm=0,
            checkpoint_path=str(tmp_path / ".cp.json"),
            batch_log_interval=1,
            max_workers=1,
        ),
        conversation_log=ConversationLogConfig(enabled=False),
        image=ImageConfig(enabled=False),
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


def test_run_v2_compat_image_disabled(tmp_path, monkeypatch):
    cfg = _config(tmp_path)

    class FakeCsvHandler:
        def __init__(self, config):
            self.config = config
            self.saved_df = None
            self.df = pd.DataFrame([{"question": "Q", "expected_answer": "E", "llm_answer": ""}])

        def load(self):
            return self.df

        def get_pending_rows(self, df, processed_indices):
            return [0]

        def update_row(self, df, index, column, value):
            df.at[index, column] = value
            return df

        def save(self, df):
            self.saved_df = df.copy()

    class FakeCheckpoint:
        def __init__(self, checkpoint_path, input_path):
            self.marked = []

        def load(self):
            return set()

        def mark_done(self, idx):
            self.marked.append(idx)

        def clear(self):
            return

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return

        def call(self, system_prompt, user_prompt):
            assert isinstance(user_prompt, str)
            return "ok"

    class FakeLimiter:
        def __init__(self, rpm):
            return

        def wait(self):
            return

    class FakeLogger:
        def __init__(self, config):
            self.entries = []

        def log(self, **kwargs):
            self.entries.append(kwargs)

        def close(self):
            return

    monkeypatch.setattr("src.core.runner.CsvHandler", FakeCsvHandler)
    monkeypatch.setattr("src.core.runner.CheckpointManager", FakeCheckpoint)
    monkeypatch.setattr("src.core.runner.LLMClient", FakeClient)
    monkeypatch.setattr("src.core.runner.RateLimiter", FakeLimiter)
    monkeypatch.setattr("src.core.runner.ConversationLogger", FakeLogger)

    run(cfg)


def test_run_image_error_marks_image_error_status(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.image = ImageConfig(
        enabled=True,
        image_column="image_path",
        base_dir=str(tmp_path),
        detail=ImageDetail.AUTO,
        separator=";",
    )

    class FakeCsvHandler:
        def __init__(self, config):
            self.config = config
            self.df = pd.DataFrame(
                [{"question": "Q", "expected_answer": "E", "image_path": "missing.png", "llm_answer": ""}]
            )
            self.updated = []

        def load(self):
            return self.df

        def get_pending_rows(self, df, processed_indices):
            return [0]

        def update_row(self, df, index, column, value):
            self.updated.append((index, column, value))
            df.at[index, column] = value
            return df

        def save(self, df):
            return

    class FakeCheckpoint:
        def __init__(self, checkpoint_path, input_path):
            self.marked = []

        def load(self):
            return set()

        def mark_done(self, idx):
            self.marked.append(idx)

        def clear(self):
            return

    class FakeClient:
        def __init__(self, *args, **kwargs):
            return

        def call(self, system_prompt, user_prompt):
            return "should_not_reach"

    class FakeLimiter:
        def __init__(self, rpm):
            return

        def wait(self):
            return

    entries: list[dict[str, Any]] = []

    class FakeLogger:
        def __init__(self, config):
            return

        def log(self, **kwargs):
            entries.append(kwargs)

        def close(self):
            return

    monkeypatch.setattr("src.core.runner.CsvHandler", FakeCsvHandler)
    monkeypatch.setattr("src.core.runner.CheckpointManager", FakeCheckpoint)
    monkeypatch.setattr("src.core.runner.LLMClient", FakeClient)
    monkeypatch.setattr("src.core.runner.RateLimiter", FakeLimiter)
    monkeypatch.setattr("src.core.runner.ConversationLogger", FakeLogger)

    run(cfg)

    assert entries
    assert entries[0]["status"] == "image_error"
    assert entries[0]["llm_response"] == "__IMAGE_ERROR__"
