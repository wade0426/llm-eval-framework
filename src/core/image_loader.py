from __future__ import annotations

import base64
import logging
from pathlib import Path


LOGGER = logging.getLogger("llm_eval")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def get_mime_type(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {ext}")
    return _MIME_TYPES[ext]


def load_image_as_data_uri(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    mime_type = get_mime_type(file_path)
    image_bytes = file_path.read_bytes()
    LOGGER.info("Loaded image: %s (%d bytes)", file_path, len(image_bytes))

    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_image_paths(raw_value: str, base_dir: str, separator: str) -> list[Path]:
    if raw_value is None:
        return []

    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []

    base = Path(base_dir)
    parts = [item.strip() for item in text.split(separator)]
    paths: list[Path] = []
    for part in parts:
        if not part:
            continue
        candidate = Path(part)
        if candidate.is_absolute():
            paths.append(candidate)
        else:
            paths.append((base / candidate).resolve())
    return paths


def build_image_content_parts(image_paths: list[Path], detail: str) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    for image_path in image_paths:
        data_uri = load_image_as_data_uri(image_path)
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": detail},
            }
        )
    return parts
