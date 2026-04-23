from __future__ import annotations

import json

from src.core.checkpoint import CheckpointManager


def test_checkpoint_load_empty(tmp_path):
    cp = CheckpointManager(str(tmp_path / ".cp.json"), "in.csv")
    assert cp.load() == set()


def test_checkpoint_mark_done_and_reload(tmp_path):
    path = tmp_path / ".cp.json"
    cp = CheckpointManager(str(path), "in.csv")
    cp.mark_done(1)
    cp2 = CheckpointManager(str(path), "in.csv")
    assert cp2.load() == {1}


def test_checkpoint_input_path_mismatch(tmp_path):
    path = tmp_path / ".cp.json"
    path.write_text(
        json.dumps(
            {
                "input_path": "other.csv",
                "processed_indices": [1, 2],
                "last_updated": "2026-01-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    cp = CheckpointManager(str(path), "in.csv")
    assert cp.load() == set()
