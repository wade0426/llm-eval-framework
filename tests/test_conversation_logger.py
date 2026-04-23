from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from src.core.conversation_logger import ConversationLogger
from src.models.config_schema import ConversationLogConfig, ConversationLogMode


def test_conversation_logger_append_mode(tmp_path):
    path = tmp_path / "conversation.jsonl"
    logger = ConversationLogger(
        ConversationLogConfig(enabled=True, log_path=str(path), log_mode=ConversationLogMode.APPEND)
    )
    logger.log(0, "s", "u1", "r1", "success", 1.0)
    logger.log(1, "s", "u2", "r2", "api_error", 2.0)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["index"] == 0
    assert json.loads(lines[1])["index"] == 1


def test_conversation_logger_latest_only_mode(tmp_path):
    path = tmp_path / "conversation.jsonl"
    logger = ConversationLogger(
        ConversationLogConfig(enabled=True, log_path=str(path), log_mode=ConversationLogMode.LATEST_ONLY)
    )
    logger.log(0, "s", "u1", "r1", "success", 1.0)
    logger.log(1, "s", "u2", "r2", "success", 2.0)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["index"] == 1


def test_conversation_logger_disabled_is_noop(tmp_path):
    path = tmp_path / "conversation.jsonl"
    logger = ConversationLogger(
        ConversationLogConfig(enabled=False, log_path=str(path), log_mode=ConversationLogMode.APPEND)
    )
    logger.log(0, "s", "u", "r", "success", 1.0)
    assert not path.exists()


def test_conversation_logger_thread_safe_append(tmp_path):
    path = tmp_path / "conversation.jsonl"
    logger = ConversationLogger(
        ConversationLogConfig(enabled=True, log_path=str(path), log_mode=ConversationLogMode.APPEND)
    )

    def _write(i: int):
        logger.log(i, "s", f"u{i}", f"r{i}", "success", 0.1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(50):
            executor.submit(_write, i)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 50
    parsed = [json.loads(line) for line in lines]
    assert {item["index"] for item in parsed} == set(range(50))
