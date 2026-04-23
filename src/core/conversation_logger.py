from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from src.models.config_schema import ConversationLogConfig, ConversationLogMode


class ConversationLogger:
    def __init__(self, config: ConversationLogConfig):
        self.config = config
        self._lock = threading.Lock()

        if not self.config.enabled:
            self._enabled = False
            return

        self._enabled = True
        self.path = Path(self.config.log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        index: int,
        system_prompt: str,
        user_prompt: str,
        llm_response: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        if not self._enabled:
            return

        payload = {
            "index": index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": llm_response,
            "status": status,
            "duration_seconds": float(duration_seconds),
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"

        with self._lock:
            if self.config.log_mode == ConversationLogMode.LATEST_ONLY:
                with self.path.open("w", encoding="utf-8") as fp:
                    fp.write(line)
                    fp.flush()
            else:
                with self.path.open("a", encoding="utf-8") as fp:
                    fp.write(line)
                    fp.flush()

    def close(self) -> None:
        return
