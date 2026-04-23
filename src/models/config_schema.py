from __future__ import annotations

import re
from enum import Enum
from string import Formatter

from pydantic import BaseModel, Field, model_validator


class OutputMode(str, Enum):
    NEW = "new"
    OVERWRITE = "overwrite"


class DatasetConfig(BaseModel):
    input_path: str
    output_path: str
    output_mode: OutputMode = OutputMode.NEW
    encoding: str = "utf-8"


class InputColumnConfig(BaseModel):
    merge: bool = False
    columns: list[str] = Field(default_factory=list)
    merge_template: str = ""

    @model_validator(mode="after")
    def validate_columns(self) -> "InputColumnConfig":
        if not self.columns:
            raise ValueError("column_mapping.input_columns.columns must not be empty")
        if self.merge and not self.merge_template:
            raise ValueError("column_mapping.input_columns.merge_template is required when merge=true")
        return self


class ColumnMappingConfig(BaseModel):
    answer_column: str
    input_columns: InputColumnConfig
    output_column: str


class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 30


class JudgeConfig(LLMConfig):
    enabled: bool = False
    output_column: str = "judge_score"


class ExecutionConfig(BaseModel):
    max_retries: int = 3
    retry_delay_seconds: float = 2
    rate_limit_rpm: int = 60
    checkpoint_path: str
    batch_log_interval: int = 10


class AppConfig(BaseModel):
    dataset: DatasetConfig
    column_mapping: ColumnMappingConfig
    primary_llm: LLMConfig
    execution: ExecutionConfig
    judge: JudgeConfig = Field(default_factory=lambda: JudgeConfig(
        enabled=False,
        base_url="https://api.openai.com/v1",
        api_key="",
        model="",
        output_column="judge_score",
        system_prompt="",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=30,
    ))

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "AppConfig":
        answer_column = self.column_mapping.answer_column
        input_cols = self.column_mapping.input_columns.columns
        output_column = self.column_mapping.output_column

        if answer_column in input_cols:
            raise ValueError("column_mapping.answer_column must not appear in column_mapping.input_columns.columns")

        if output_column == answer_column:
            raise ValueError("column_mapping.output_column must not be equal to column_mapping.answer_column")

        if self.column_mapping.input_columns.merge:
            placeholders = {
                field_name
                for _, field_name, _, _ in Formatter().parse(self.column_mapping.input_columns.merge_template)
                if field_name
            }
            missing = placeholders - set(input_cols)
            if missing:
                raise ValueError(
                    "column_mapping.input_columns.merge_template contains unknown placeholders: "
                    f"{sorted(missing)}"
                )

        if self.judge.enabled:
            if not self.judge.base_url or not self.judge.api_key or not self.judge.model or not self.judge.output_column:
                raise ValueError("judge.enabled=true requires judge base_url, api_key, model, and output_column")
            if self.judge.output_column in {answer_column, output_column}:
                raise ValueError(
                    "judge.output_column must not be equal to column_mapping.answer_column or column_mapping.output_column"
                )

        return self


ENV_VAR_PATTERN = re.compile(r"^\$\{[A-Z0-9_]+\}$")
