from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.config_schema import AppConfig


class CsvHandler:
    def __init__(self, config: AppConfig):
        self.config = config

    def load(self) -> pd.DataFrame:
        input_path = self.config.dataset.input_path
        try:
            df = pd.read_csv(input_path, encoding=self.config.dataset.encoding)
        except Exception as exc:
            raise ValueError(f"Failed to read input CSV: {input_path}") from exc

        required_columns = set(self.config.column_mapping.input_columns.columns)
        required_columns.add(self.config.column_mapping.answer_column)

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        output_column = self.config.column_mapping.output_column
        if output_column not in df.columns:
            df[output_column] = ""

        if self.config.judge.enabled and self.config.judge.output_column not in df.columns:
            df[self.config.judge.output_column] = ""

        return df

    def get_pending_rows(self, df: pd.DataFrame, processed_indices: set[int]) -> list[int]:
        return [int(idx) for idx in df.index if int(idx) not in processed_indices]

    def save(self, df: pd.DataFrame) -> None:
        mode = self.config.dataset.output_mode.value
        destination = (
            self.config.dataset.output_path
            if mode == "new"
            else self.config.dataset.input_path
        )
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False, encoding=self.config.dataset.encoding)

    def update_row(self, df: pd.DataFrame, index: int, column: str, value: str) -> pd.DataFrame:
        df.at[index, column] = value
        return df
