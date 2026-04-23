from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


LOGGER = logging.getLogger("llm_eval")


class CheckpointManager:
    def __init__(self, checkpoint_path: str, input_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.input_path = input_path
        self.processed_indices: set[int] = set()

    def load(self) -> set[int]:
        if not self.checkpoint_path.exists():
            self.processed_indices = set()
            return self.processed_indices

        try:
            with self.checkpoint_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception:
            LOGGER.warning("Checkpoint file is invalid, ignoring: %s", self.checkpoint_path)
            self.processed_indices = set()
            return self.processed_indices

        if payload.get("input_path") != self.input_path:
            self.processed_indices = set()
            return self.processed_indices

        indices = payload.get("processed_indices", [])
        self.processed_indices = {int(v) for v in indices}
        return self.processed_indices

    def mark_done(self, index: int) -> None:
        self.processed_indices.add(index)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "input_path": self.input_path,
            "processed_indices": sorted(self.processed_indices),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }

        tmp_path = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        tmp_path.replace(self.checkpoint_path)

    def clear(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
