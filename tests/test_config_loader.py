from __future__ import annotations

import os

import pytest

from src.core.config_loader import ConfigValidationError, load_config


def _write_config(tmp_path, content: str):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_config_success(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("JUDGE_API_KEY", "k2")
    config_path = _write_config(
        tmp_path,
        """
dataset:
  input_path: input.csv
  output_path: out.csv
  output_mode: new
  encoding: utf-8
column_mapping:
  answer_column: expected_answer
  input_columns:
    merge: false
    columns: [question]
    merge_template: ""
  output_column: llm_answer
primary_llm:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  system_prompt: test
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
execution:
  max_retries: 3
  retry_delay_seconds: 2
  rate_limit_rpm: 60
  checkpoint_path: .cp.json
  batch_log_interval: 10
judge:
  enabled: false
  base_url: https://api.openai.com/v1
  api_key: ${JUDGE_API_KEY}
  model: gpt-4o
  output_column: judge_score
  system_prompt: judge
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
""",
    )

    cfg = load_config(config_path)
    assert cfg.primary_llm.api_key == "k1"


def test_load_config_missing_env_raises(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
dataset:
  input_path: input.csv
  output_path: out.csv
  output_mode: new
  encoding: utf-8
column_mapping:
  answer_column: expected_answer
  input_columns:
    merge: false
    columns: [question]
    merge_template: ""
  output_column: llm_answer
primary_llm:
  base_url: https://api.openai.com/v1
  api_key: ${MISSING_KEY}
  model: gpt-4o
  system_prompt: test
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
execution:
  max_retries: 3
  retry_delay_seconds: 2
  rate_limit_rpm: 60
  checkpoint_path: .cp.json
  batch_log_interval: 10
judge:
  enabled: false
  base_url: https://api.openai.com/v1
  api_key: x
  model: gpt-4o
  output_column: judge_score
  system_prompt: judge
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
""",
    )
    with pytest.raises(EnvironmentError):
        load_config(config_path)


def test_invalid_answer_column_in_input_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    config_path = _write_config(
        tmp_path,
        """
dataset:
  input_path: input.csv
  output_path: out.csv
  output_mode: new
  encoding: utf-8
column_mapping:
  answer_column: expected_answer
  input_columns:
    merge: false
    columns: [expected_answer]
    merge_template: ""
  output_column: llm_answer
primary_llm:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  system_prompt: test
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
execution:
  max_retries: 3
  retry_delay_seconds: 2
  rate_limit_rpm: 60
  checkpoint_path: .cp.json
  batch_log_interval: 10
judge:
  enabled: false
  base_url: https://api.openai.com/v1
  api_key: x
  model: gpt-4o
  output_column: judge_score
  system_prompt: judge
  temperature: 0
  max_tokens: 16
  timeout_seconds: 30
""",
    )
    with pytest.raises(ConfigValidationError):
        load_config(config_path)
