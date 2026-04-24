from __future__ import annotations

from pathlib import Path

import pytest

from src.core.image_loader import (
    build_image_content_parts,
    get_mime_type,
    load_image_as_data_uri,
    parse_image_paths,
)


def test_get_mime_type_supported_and_unsupported():
    assert get_mime_type(Path("a.png")) == "image/png"
    assert get_mime_type(Path("a.jpg")) == "image/jpeg"
    with pytest.raises(ValueError):
        get_mime_type(Path("a.bmp"))


def test_load_image_as_data_uri_png_and_jpg(tmp_path):
    png_path = tmp_path / "a.png"
    jpg_path = tmp_path / "b.jpg"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    jpg_path.write_bytes(b"\xff\xd8\xff\xd9")

    png_uri = load_image_as_data_uri(png_path)
    jpg_uri = load_image_as_data_uri(jpg_path)

    assert png_uri.startswith("data:image/png;base64,")
    assert jpg_uri.startswith("data:image/jpeg;base64,")


def test_load_image_as_data_uri_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_as_data_uri(tmp_path / "missing.png")


def test_parse_image_paths_separator_strip_skip_empty_and_resolve(tmp_path):
    base_dir = tmp_path / "images"
    raw = "  a.png ; ; sub/b.jpg;  "

    paths = parse_image_paths(raw, str(base_dir), ";")

    assert len(paths) == 2
    assert paths[0] == (base_dir / "a.png").resolve()
    assert paths[1] == (base_dir / "sub" / "b.jpg").resolve()


def test_parse_image_paths_empty_or_nan_returns_empty(tmp_path):
    assert parse_image_paths("", str(tmp_path), ";") == []
    assert parse_image_paths("nan", str(tmp_path), ";") == []


def test_build_image_content_parts_multiple_images(tmp_path):
    image1 = tmp_path / "1.png"
    image2 = tmp_path / "2.webp"
    image1.write_bytes(b"\x89PNG\r\n\x1a\n")
    image2.write_bytes(b"RIFF....WEBP")

    parts = build_image_content_parts([image1, image2], detail="auto")

    assert len(parts) == 2
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["detail"] == "auto"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"]["url"].startswith("data:image/webp;base64,")
