from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.models.config_schema import AppConfig


ENV_PATTERN = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


def _resolve_env_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_value(item) for item in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            env_name = match.group(1)
            resolved = os.getenv(env_name)
            if resolved is None:
                raise EnvironmentError(f"Missing environment variable: {env_name}")
            return resolved
    return value


def _format_validation_error(error: ValidationError) -> str:
    lines: list[str] = ["Config validation failed:"]
    for item in error.errors():
        location = ".".join(str(p) for p in item.get("loc", ()))
        message = item.get("msg", "unknown validation error")
        lines.append(f"- {location}: {message}")
    return "\n".join(lines)


def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fp:
        raw_data = yaml.safe_load(fp)

    if raw_data is None:
        raise ConfigValidationError("Config file is empty")

    resolved_data = _resolve_env_value(raw_data)

    try:
        return AppConfig.model_validate(resolved_data)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error(exc)) from exc
