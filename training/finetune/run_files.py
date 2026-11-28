"""A run directory's files as the trainers (`checkpoints.py`) and the recovery sync (`hf_sync.py`,
which loads no torch) both read them: its checkpoint pairs and its promoted checkpoint
(`best.json`), and the durable writes behind them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The mean chrF of the selecting sets (`eval_rows.selecting_sets`). Named apart from
# "free_running_chrf", the first set's alone, which a run of both directions selected on before
# 2026-11-11: its best.json is not comparable.
BEST_METRIC = "selecting_chrf"


def fsync_path(path: Path) -> None:
    """Flush a file, or a directory's entries, to disk: a rename alone can survive a host crash
    that the renamed file's data does not."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding="utf-8")
        fsync_path(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_pairs(out_dir: Path) -> list[tuple[int, Path, Path]]:
    models = {
        int(path.stem.removeprefix("model_step")): path
        for path in out_dir.glob("model_step*.safetensors")
        if path.stem.removeprefix("model_step").isdigit()
    }
    trainers = {
        int(path.stem.removeprefix("trainer_step")): path
        for path in out_dir.glob("trainer_step*.pt")
        if path.stem.removeprefix("trainer_step").isdigit()
    }
    return [
        (step, models[step], trainers[step])
        for step in sorted(models.keys() & trainers.keys())
    ]


def load_best_state(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "best.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if set(state) != {"step", "model", "metric", "chrf"}:
        raise RuntimeError("Invalid best-checkpoint metadata")
    step = int(state["step"])
    if state["metric"] != BEST_METRIC:
        raise RuntimeError("Best-checkpoint metric changed")
    model = out_dir / str(state["model"])
    if model.name != f"best_step{step:06d}.safetensors" or not model.is_file():
        raise RuntimeError("Best-checkpoint metadata references a missing model")
    float(state["chrf"])
    return state
