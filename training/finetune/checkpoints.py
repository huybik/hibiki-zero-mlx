"""A run directory, shared by the SFT trainer (`train.py`) and GRPO (`grpo.py`): checkpoint
pairs, resume and its contract (`run_config.json`), best-checkpoint promotion and the jsonl logs.

A pair is `model_stepNNNNNN.safetensors` plus `trainer_stepNNNNNN.pt` (the optimizer state); a run
resumes its newest complete pair. `hf_sync.py` uploads these files as they are.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import torch

from training.finetune import common
from training.paths import repo_display_path, require_file

BEST_METRIC = "free_running_chrf"


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding="utf-8")
        common.fsync_path(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        common.fsync_path(temporary)
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


def clean_incomplete_checkpoints(out_dir: Path) -> None:
    pairs = checkpoint_pairs(out_dir)
    paired_models = {model for _, model, _ in pairs}
    paired_trainers = {trainer for _, _, trainer in pairs}
    for path in out_dir.glob("model_step*.safetensors"):
        if path not in paired_models:
            path.unlink()
    for path in out_dir.glob("trainer_step*.pt"):
        if path not in paired_trainers:
            path.unlink()
    for pattern in (".model_step*.safetensors.tmp", ".trainer_step*.pt.tmp", ".best_step*.tmp"):
        for path in out_dir.glob(pattern):
            path.unlink()


def prune_checkpoint_pairs(out_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    pairs = checkpoint_pairs(out_dir)
    for _, model, trainer in pairs[: max(0, len(pairs) - keep)]:
        trainer.unlink()
        model.unlink()


def open_run(out_dir: Path) -> tuple[bool, dict[str, Any] | None]:
    """Prepare `out_dir` for a start or a resume: whether it holds a complete pair to resume, and
    its promoted checkpoint (`best.json`). Half-written pairs go either way. A start without a
    pair begins at step 0, so the logs, best.json and eval/ of a run cut short before its first
    save go too: they describe steps that will be trained again."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_incomplete_checkpoints(out_dir)
    resuming = bool(checkpoint_pairs(out_dir))
    if not resuming:
        for name in ("train_log.jsonl", "val_log.jsonl", "best.json"):
            (out_dir / name).unlink(missing_ok=True)
        shutil.rmtree(out_dir / "eval", ignore_errors=True)
    best_state = load_best_state(out_dir)
    best_model = None if best_state is None else out_dir / str(best_state["model"])
    for path in out_dir.glob("best_step*.safetensors"):
        if path != best_model:
            path.unlink()
    return resuming, best_state


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def claim_run_config(
    out_dir: Path, run_config: dict[str, Any], resume_keys: tuple[str, ...], resuming: bool
) -> None:
    """Write a new run's `run_config.json`, or hold a resumed run to the one it started with: a
    resume must match it on `resume_keys`."""
    path = out_dir / "run_config.json"
    if not resuming:
        atomic_write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", path)
        return
    if not path.is_file():
        raise RuntimeError("Resume requires the original run_config.json")
    previous = json.loads(path.read_text(encoding="utf-8"))
    for key in resume_keys:
        if previous.get(key) != run_config.get(key):
            raise RuntimeError(f"Same-run resume contract changed: {key}")


def append_log(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def build_metadata(args: argparse.Namespace) -> dict[str, str]:
    return {"target": "full", "base_model": repo_display_path(args.model_weight)}


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


def promote_best(
    step: int,
    chrf: float,
    out_dir: Path,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Promote the checkpoint of `step` if its chrF beats the promoted one's."""
    if previous is not None and chrf <= float(previous["chrf"]):
        return previous
    checkpoint_path = require_file(
        out_dir / f"model_step{step:06d}.safetensors",
        "validation-cadence model checkpoint",
    )
    model_path = out_dir / f"best_step{step:06d}.safetensors"
    # A resume can validate the promoted step again: its link is this checkpoint already, unless
    # a replacement pod restored that best model beside an older pair and retrained the step.
    # A new link replaces the old one in one rename, so best.json never names a missing file.
    if not (model_path.exists() and os.path.samefile(model_path, checkpoint_path)):
        temporary = model_path.with_name(f".{model_path.name}.tmp")
        temporary.unlink(missing_ok=True)
        os.link(checkpoint_path, temporary)
        temporary.replace(model_path)
    state = {
        "step": step,
        "model": model_path.name,
        "metric": BEST_METRIC,
        "chrf": chrf,
    }
    atomic_write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", out_dir / "best.json"
    )
    for path in out_dir.glob("best_step*.safetensors"):
        if path != model_path:
            path.unlink()
    print(
        f"Promoted best chrF={chrf:.2f} at step {step} "
        f"-> {repo_display_path(model_path)}"
    )
    return state


class CheckpointWriter:
    """Saves checkpoint pairs from a pinned host snapshot in a background thread.

    `save` enqueues the copies off the GPU on the current stream, so later optimizer steps
    cannot change the snapshot, and returns at once; the thread waits for the copies, writes the
    model then the trainer file (a pair is complete once both exist) and prunes old pairs. The
    pinned buffers are reused, so a save first waits for the previous write. The thread is not a
    daemon: if training raises, the interpreter finishes the pair before it exits."""

    def __init__(self, model: Any, optimizer: Any, args: argparse.Namespace, out_dir: Path):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.out_dir = out_dir
        self.buffers: dict[tuple[Any, ...], torch.Tensor] = {}
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def _host(self, value: Any, key: tuple[Any, ...]) -> Any:
        if torch.is_tensor(value):
            buffer = self.buffers.get(key)
            if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                buffer = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                self.buffers[key] = buffer
            return buffer.copy_(value.detach(), non_blocking=True)
        if isinstance(value, dict):
            return {name: self._host(item, (*key, name)) for name, item in value.items()}
        if isinstance(value, list):
            return [self._host(item, (*key, index)) for index, item in enumerate(value)]
        return value

    def save(self, step: int) -> None:
        self.wait()
        model_path = self.out_dir / f"model_step{step:06d}.safetensors"
        trainer_path = self.out_dir / f"trainer_step{step:06d}.pt"
        if model_path.is_file() and trainer_path.is_file():  # a resume at a save step
            return
        model_state = self._host(self.model.state_dict(), ("model",))
        optimizer_state = self._host(self.optimizer.state_dict(), ("optimizer",))
        copied = torch.cuda.Event()
        copied.record()
        trainer_state = {
            "step": step,
            "optimizer": optimizer_state,
            "model": model_path.name,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args).items()
            },
        }
        self.thread = threading.Thread(
            target=self._write,
            args=(copied, model_state, trainer_state, model_path, trainer_path),
        )
        self.thread.start()

    def _write(self, copied, model_state, trainer_state, model_path, trainer_path) -> None:
        try:
            copied.synchronize()
            clean_incomplete_checkpoints(self.out_dir)
            if self.args.keep_checkpoints > 0:  # room for the new pair before it is written
                prune_checkpoint_pairs(self.out_dir, max(1, self.args.keep_checkpoints - 1))
            common.save_state(model_state, model_path, build_metadata(self.args))
            atomic_torch_save(trainer_state, trainer_path)
            common.fsync_path(self.out_dir)
            prune_checkpoint_pairs(self.out_dir, self.args.keep_checkpoints)
        except BaseException as exc:  # raised by the next wait()
            model_path.unlink(missing_ok=True)
            self.error = exc

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            error, self.error = self.error, None
            raise RuntimeError("Checkpoint write failed") from error


def load_newest_pair(model: Any, optimizer: Any, out_dir: Path, device: torch.device) -> int:
    """Load the newest complete pair in `out_dir` into the fp32 `model` and its optimizer, and
    return its step, or 0 without one."""
    pairs = checkpoint_pairs(out_dir)
    if not pairs:
        return 0
    step, model_path, trainer_path = pairs[-1]
    checkpoint = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != step or checkpoint["model"] != model_path.name:
        raise RuntimeError(f"{trainer_path.name} does not match {model_path.name}")
    common.load_model(model, model_path, torch.float32)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    print(f"Resumed step {step} from {repo_display_path(trainer_path)}")
    return step


def keep_log_lines(path: Path, last_step: int) -> list[dict[str, Any]]:
    """Drop the lines of a jsonl log past `last_step` (written after the pair a run resumes), and
    a last line cut short by a crash."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line]
    items = []
    for index, line in enumerate(lines):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    kept = [item for item in items if int(item["step"]) <= last_step]
    # A line cut just before its newline parses, but the next append would join it.
    if len(kept) != len(lines) or (text and not text.endswith("\n")):
        atomic_write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in kept), path)
    return kept
