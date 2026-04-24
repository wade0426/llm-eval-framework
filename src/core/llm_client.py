from __future__ import annotations

import logging
import threading
import time
from typing import Any, cast

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.models.config_schema import LLMConfig


LOGGER = logging.getLogger("llm_eval")


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._last_called: float | None = None

    def wait(self) -> None:
        if self.rpm <= 0:
            return

        interval = 60.0 / self.rpm
        with self._lock:
            now = time.time()
            if self._last_called is None:
                self._last_called = now
                return

            elapsed = now - self._last_called
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._last_called = time.time()


class LLMClient:
    def __init__(self, llm_config: LLMConfig, max_retries: int = 3, retry_delay_seconds: float = 2):
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.client: Any = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url, timeout=llm_config.timeout_seconds)

    def call(self, system_prompt: str, user_prompt: str | list[dict[str, Any]]) -> str:
        retryable = (APIError, RateLimitError, httpx.TimeoutException, APIConnectionError, APITimeoutError)

        @retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential_jitter(initial=self.retry_delay_seconds, max=60),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        )
        def _call_once() -> str:
            response = self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=cast(
                    Any,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                ),
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            if not response.choices:
                return ""
            content = response.choices[0].message.content
            return "" if content is None else str(content)

        try:
            return _call_once()
        except retryable as exc:
            LOGGER.error("LLM call failed after retries: %s", exc)
            return "__API_ERROR__"
