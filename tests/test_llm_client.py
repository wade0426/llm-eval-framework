from __future__ import annotations

from types import SimpleNamespace

import pytest
import httpx

from src.core.llm_client import LLMClient, RateLimiter
from src.models.config_schema import LLMConfig


class _FakeChatCompletions:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        value = self.responses[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))


def _config() -> LLMConfig:
    return LLMConfig(
        base_url="https://api.openai.com/v1",
        api_key="k",
        model="gpt-4o",
        system_prompt="s",
        temperature=0,
        max_tokens=16,
        timeout_seconds=30,
    )


def test_llm_client_call_success(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    client = LLMClient(_config(), max_retries=0)
    client.client = _FakeClient([response])
    out = client.call("sys", "user")
    assert out == "ok"


def test_llm_client_call_returns_api_error_on_failure(monkeypatch):
    client = LLMClient(_config(), max_retries=0)
    client.client = _FakeClient([httpx.TimeoutException("boom")])
    out = client.call("sys", "user")
    assert out == "__API_ERROR__"


def test_rate_limiter_noop_when_zero(monkeypatch):
    limiter = RateLimiter(0)
    limiter.wait()


def test_llm_client_call_accepts_content_parts_list(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok multimodal"))]
    )
    client = LLMClient(_config(), max_retries=0)
    fake = _FakeClient([response])
    client.client = fake

    user_content = [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx", "detail": "auto"}},
    ]
    out = client.call("sys", user_content)

    assert out == "ok multimodal"
    assert fake.chat.completions.calls == 1
