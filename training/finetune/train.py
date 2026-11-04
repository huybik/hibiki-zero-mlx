#!/usr/bin/env python
"""Full-model Hibiki-Zero Vietnamese-to-English SFT on multi-sentence streams.

Trains every parameter from the upstream weight with fp32 masters, bf16 autocast and fused
AdamW, under a linear warmup then cosine decay. Streams are planned once from the grounded cache
rows with --seed (`streams.py`) and batched up to --batch-frames padded frames; every epoch
replays that order. Each validation reports teacher-forced losses on a fixed, seeded stream plan
and batch order of the validation cache (so --val-batches takes a random subset) and translates
the rows of every --eval-pairs file free-running in one pass, sampled as `eval.py` samples, and
scores each file as its own set. By default these are FLEURS dev (real Vietnamese speech) and
held-out Common Voice and Libriheavy rows of the training data's own domain. Of the checkpoints
whose every set passes the generation health gates, the one with the highest chrF on the first
set is promoted to best_stepNNNNNN.safetensors.

Checkpoint pairs (model + optimizer) are saved every --save-every steps from a pinned-memory
snapshot written by a background thread, so a save stalls training only for the copy off the
GPU. A start in an --out-dir that holds complete pairs resumes the newest one, so a preempted run
continues by running the same command again.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.finetune import common, streams
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

BEST_METRIC = "free_running_chrf"
# A resumed run must match these run_config.json entries.
RESUME_KEYS = (
    "cache_dir",
    "val_cache_dir",
    "config_path",
    "model_weight",
    "mimi_weight",
    "tokenizer",
    "epochs",
    "max_steps",
    "batch_frames",
    "stream_seconds",
    "max_frames",
    "lr",
    "warmup_fraction",
    "adam_beta1",
    "adam_beta2",
    "weight_decay",
    "grad_clip",
    "audio_loss_weight",
    "text_loss_weight",
    "first_codebook_weight",
    "text_pad_weight",
    "val_batches",
    "eval_pairs",
    "eval_batch_size",
    "val_every",
    "save_every",
    "keep_checkpoints",
    "seed",
    "train_rows",
    "train_streams",
    "val_rows",
    "val_streams",
    "eval_rows",
    "total_steps",
)
LOG_LOSSES = (
    "loss",
    "audio_loss",
    "semantic_loss",
    "acoustic_loss",
    "text_loss",
    "content_text_loss",
    "pad_text_loss",
)


def parse_args() -> argparse.Namespace:
    # No abbreviations: h100.sh must be able to reject --out-dir.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 trains all epochs.")
    parser.add_argument(
        "--batch-frames", type=int, default=4608, help="Padded frames per batch (streams x frames)."
    )
    parser.add_argument(
        "--stream-seconds",
        type=float,
        nargs=2,
        default=list(streams.STREAM_SECONDS),
        metavar=("MIN", "MAX"),
        help="Each stream's length is drawn from this range.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=280, help="Cache rows longer than this are dropped."
    )
    parser.add_argument("--lr", type=float, default=1e-6, help="Peak learning rate.")
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.05, help="Share of steps warming up linearly."
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--first-codebook-weight",
        type=float,
        default=100.0,
        help="Weight of audio codebook 0 against each acoustic codebook.",
    )
    parser.add_argument(
        "--text-pad-weight", type=float, default=0.5, help="Weight of a text PAD against a word."
    )
    parser.add_argument("--val-every", type=int, default=9000)
    parser.add_argument("--val-batches", type=int, default=0, help="0 validates every batch.")
    parser.add_argument(
        "--eval-pairs",
        type=Path,
        nargs="+",
        default=[
            Path("data/eval/fleurs-vi/dev.jsonl"),
            Path("data/eval/en-vi-val/cv27.jsonl"),
            Path("data/eval/en-vi-val/libriheavy.jsonl"),
        ],
        help="Pairs files (training.prep.eval_sets) translated free-running at each validation, "
        "each scored as its own set; the first selects the checkpoint.",
    )
    # bf16 at batch 64 decodes FLEURS dev with an 8 s tail in ~1,750 steps of ~22 ms on the H100.
    parser.add_argument("--eval-batch-size", type=int, default=64)
    # Frequent saves bound the work a preemption loses; the snapshot stalls training ~0.6 s.
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4, help="Stream-assembly workers.")
    return parser.parse_args()


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


def load_newest_pair(
    model: Any, optimizer: Any, out_dir: Path, device: torch.device, dtype: torch.dtype
) -> int:
    """Load the newest complete pair in `out_dir` and return its step, or 0 without one."""
    pairs = checkpoint_pairs(out_dir)
    if not pairs:
        return 0
    step, model_path, trainer_path = pairs[-1]
    checkpoint = torch.load(trainer_path, map_location="cpu", weights_only=False)
    if int(checkpoint["step"]) != step or checkpoint["model"] != model_path.name:
        raise RuntimeError(f"{trainer_path.name} does not match {model_path.name}")
    common.load_model(model, model_path, dtype)
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


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_frames", "max_frames", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_steps", "grad_clip", "val_every", "val_batches", "save_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    shortest, longest = args.stream_seconds
    if not 0 < shortest <= longest:
        raise ValueError("--stream-seconds needs 0 < MIN <= MAX")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    # Best-model promotion hard-links the checkpoint saved at the validation step.
    if args.val_every and (not args.save_every or args.val_every % args.save_every):
        raise ValueError("--val-every must be a multiple of a positive --save-every")
    if os.environ.get("NO_TORCH_COMPILE"):
        raise ValueError("Full-model training requires torch compile; unset NO_TORCH_COMPILE")


def stream_frames(stream_seconds: list[float]) -> tuple[int, int]:
    shortest, longest = stream_seconds
    return round(shortest * streams.FRAME_RATE), round(longest * streams.FRAME_RATE)


def learning_rate(step: int, peak: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to `peak`, then cosine decay towards 0 at `total_steps` (moshi-finetune's
    OneCycle shape, without its momentum cycling). `step` counts updates from 0."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    # Validation builds and frees a bf16 model and generation caches between training steps;
    # expandable segments keep the freed memory reusable for the training batches.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    validate_args(args)
    common.seed_all(args.seed)

    device = common.check_device("cuda")
    dtype = torch.float32
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    val_cache_dir = require_dir(args.val_cache_dir, "validation code cache directory")
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "upstream model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    args.eval_pairs = [require_file(path, "evaluation pairs") for path in args.eval_pairs]

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_incomplete_checkpoints(out_dir)
    resuming = bool(checkpoint_pairs(out_dir))
    if not resuming:
        # A start without a pair begins at step 0: logs of a run cut short before its first save
        # describe steps that will be trained again.
        for name in ("train_log.jsonl", "val_log.jsonl", "best.json"):
            (out_dir / name).unlink(missing_ok=True)
        shutil.rmtree(out_dir / "eval", ignore_errors=True)
    best_state = load_best_state(out_dir)
    best_model = None if best_state is None else out_dir / str(best_state["model"])
    for path in out_dir.glob("best_step*.safetensors"):
        if path != best_model:
            path.unlink()

    checkpoint_info = common.load_checkpoint_info(args)
    frames = stream_frames(args.stream_seconds)
    # On the CPU in float32, so every machine and resume encodes the same silence (up to a rare
    # near-tie in a residual code).
    silence = streams.silence_track(checkpoint_info.get_mimi(device="cpu"), frames[1])

    rows, layout = streams.load_rows(cache_dirs, args.max_frames)
    dataset = streams.StreamDataset(rows, layout, silence, frames, args.seed)
    batches = dataset.batches(args.batch_frames, args.seed)
    exposure = dataset.exposure()
    print(
        f"Planned {exposure['streams']} streams from {exposure['rows']} cache rows; "
        f"stream_frames={exposure['stream_frames']:,} source_hours={exposure['source_hours']:.2f} "
        f"batches={len(batches)}"
    )
    val_rows, val_layout = streams.load_rows([val_cache_dir], args.max_frames)
    if val_layout != layout:
        raise RuntimeError(f"Validation layout {val_layout} differs from training {layout}")
    val_dataset = streams.StreamDataset(
        val_rows, layout, silence, frames, streams.VALIDATION_SEED
    )
    # In the main process: forking workers at a validation would fork the checkpoint writer's
    # thread too, and its 300 batches assemble in ~3 s.
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_dataset.batches(args.batch_frames, streams.VALIDATION_SEED),
        collate_fn=streams.collate,
    )
    print(f"Planned {len(val_dataset)} validation streams from {len(val_rows)} rows")
    # Checkpoints are selected on free-running translation, sampled as eval.py samples.
    generation = common.GenerationConfig(text_only=True)
    rows_by_set = common.read_eval_sets(args.eval_pairs)
    eval_sets = list(rows_by_set)
    eval_rows = [row for rows in rows_by_set.values() for row in rows]
    common.validate_eval_rows(
        eval_rows,
        generation.source_column,
        generation.reference_column,
        generation.id_column,
        duration_column="duration_s",
    )
    for row in eval_rows:
        require_file(row[generation.source_column], "evaluation audio")
    # The sets share batches, so one pass translates them all.
    print(
        f"Validation translates {len(eval_rows)} rows of {', '.join(eval_sets)}; "
        f"{eval_sets[0]} selects the checkpoint"
    )

    batches_per_epoch = len(batches)
    total_steps = args.max_steps or batches_per_epoch * args.epochs
    # A finished run stops before loading the model, so a boot hook that retries the run until it
    # succeeds ends here.
    pairs = checkpoint_pairs(out_dir)
    if pairs and pairs[-1][0] == total_steps:
        logged = keep_log_lines(out_dir / "val_log.jsonl", total_steps)
        if total_steps in {int(item["step"]) for item in logged}:
            print(f"The run finished at step {total_steps}")
            return
    warmup_steps = round(total_steps * args.warmup_fraction)
    weights = common.LossWeights(
        audio=args.audio_loss_weight,
        text=args.text_loss_weight,
        first_codebook=args.first_codebook_weight,
        text_pad=args.text_pad_weight,
    )

    print(f"Loading upstream Hibiki-Zero from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(
        device=device,
        dtype=dtype,
        lm_kwargs_overrides={"gradient_checkpointing": False},
    )
    lm.train()
    # The training forward attends causally over the whole stream (lm_training): the longest
    # padded stream must fit the attention context, checked here rather than at its first batch.
    context = lm.transformer.layers[0].self_attn.context
    longest = math.ceil(frames[1] / streams.FRAME_BUCKET) * streams.FRAME_BUCKET
    if context is not None and longest > context:
        raise ValueError(f"--stream-seconds allows {longest} frames, over the context of {context}")
    common.enable_full_finetune(lm)
    params = common.trainable_parameters(lm)
    if not params:
        raise RuntimeError("No trainable parameters")
    print(
        f"Trainable params: {sum(param.numel() for param in params):,} / "
        f"{sum(param.numel() for param in lm.parameters()):,}"
    )
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    val_log_path = out_dir / "val_log.jsonl"
    log_path = out_dir / "train_log.jsonl"
    global_step = load_newest_pair(lm, optimizer, out_dir, device, dtype) if resuming else 0
    # Steps already validated. A pair is saved before its validation, so a run cut short there
    # validates on resume; a replacement pod without val_log validates again.
    validated = {int(item["step"]) for item in keep_log_lines(val_log_path, global_step)}
    keep_log_lines(log_path, global_step)
    if global_step > total_steps:
        raise RuntimeError("The run is beyond the requested stop")

    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    run_config = {key: jsonable(value) for key, value in vars(args).items()}
    run_config.update(
        {
            "train_rows": exposure["rows"],
            "train_streams": exposure["streams"],
            "total_steps": total_steps,
            "batches_per_epoch": batches_per_epoch,
            "warmup_steps": warmup_steps,
            "stream_frames": list(frames),
            "frame_bucket": streams.FRAME_BUCKET,
            "best_metric": BEST_METRIC,
            "val_rows": len(val_rows),
            "val_streams": len(val_dataset),
            "eval_rows": len(eval_rows),
        }
    )
    run_config_path = out_dir / "run_config.json"
    if not resuming:
        atomic_write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", run_config_path
        )
    else:
        if not run_config_path.is_file():
            raise RuntimeError("Resume requires the original run_config.json")
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key in RESUME_KEYS:
            if previous.get(key) != run_config.get(key):
                raise RuntimeError(f"Same-run resume contract changed: {key}")

    writer = CheckpointWriter(lm, optimizer, args, out_dir)

    def training_batches(start_step: int):
        """The planned batches from `start_step` on, epoch after epoch."""
        offset = start_step % batches_per_epoch
        while True:
            writer.wait()  # the workers fork; the writer's thread must not be running
            yield from DataLoader(
                dataset,
                batch_sampler=batches[offset:],
                collate_fn=streams.collate,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            offset = 0

    data_iter = training_batches(global_step)
    condition_cache: dict[int, Any | None] = {}

    def empty_log() -> dict[str, Any]:
        return {
            "losses": {key: 0.0 for key in LOG_LOSSES},
            "steps": 0,
            "audio_tokens": 0,
            "text_tokens": 0,
            "streams": 0,
            "stream_frames": 0,
            "padded_frames": 0,
            "source_frames": 0,
            "min_batch": math.inf,
            "max_batch": 0,
            "max_frames": 0,
            "grad_norm": 0.0,
            "max_grad_norm": None,
        }

    def check_gradients(step: int) -> None:
        """Raise on a non-finite gradient norm since the last log. The losses cannot show the last
        step's update, so a save checks first: a NaN update saved at a save step would be resumed
        by every restart."""
        if log["max_grad_norm"] is not None and not math.isfinite(float(log["max_grad_norm"])):
            raise RuntimeError(f"Non-finite gradient norm by step {step}")

    log = empty_log()
    last_log_time = time.time()

    def run_validation(step: int) -> None:
        nonlocal best_state, last_log_time
        started = time.time()
        # The next step zeroes the gradients anyway; freed now, they leave room for generation.
        optimizer.zero_grad(set_to_none=True)
        teacher_forced = common.evaluate_teacher_forced(
            lm,
            val_dataloader,
            device,
            checkpoint_info.model_type,
            weights,
            args.val_batches,
        )
        eval_dir = out_dir / "eval" / f"step{step:06d}"
        # Generation runs on a bf16 copy, as eval.py loads a checkpoint: 1.8x faster than the
        # fp32 masters, and the room it takes is the gradients' freed above. Built without
        # loading --model-weight: the copy overwrites every tensor anyway.
        generator = checkpoint_info.get_moshi(
            device=device, dtype=torch.bfloat16, load_weight=False
        )
        generator.load_state_dict(lm.state_dict())
        generator.eval()
        records = common.run_greedy_eval(
            eval_rows,
            generation,
            args.eval_batch_size,
            mimi,
            generator,
            text_tokenizer,
            checkpoint_info,
            eval_dir,
            args.seed,
        )
        # moshi's streaming state holds the generator in reference cycles: without the collect,
        # its 5.8 GiB outlive `del` and the next training step runs out of an 80 GB card.
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        common.write_predictions(eval_dir / "predictions.csv", records)
        free_running = common.score_sets(records, eval_sets)
        healthy = all(metrics["health_eligible"] for metrics in free_running.values())
        print(
            f"val step={step} loss={teacher_forced['loss']:.4f} "
            f"semantic={teacher_forced['semantic_loss']:.4f} "
            f"content={teacher_forced['content_text_loss']:.4f} "
            f"acc={teacher_forced['content_acc']:.3f} pad_acc={teacher_forced['pad_acc']:.3f} "
            f"health={'pass' if healthy else 'fail'}"
        )
        for name, metrics in free_running.items():
            print(f"val step={step} {name} {common.score_summary(metrics)}")
        # chrF weighs recall, so overlong output barely lowers it: the gates come first. Promotion
        # links the step's model file, so its write must be done, and precedes the log line,
        # which marks the step done for a resume.
        if healthy:
            writer.wait()
            chrf = float(free_running[eval_sets[0]]["chrf"])
            best_state = promote_best(step, chrf, out_dir, best_state)
        seconds = time.time() - started
        item = {
            "step": step,
            "seconds": seconds,
            "teacher_forced": teacher_forced,
            "free_running": free_running,
        }
        with val_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        validated.add(step)
        print(f"val step={step} took {seconds:.0f} s")
        last_log_time += seconds  # the next train-log line times training steps only

    resumed_at_validation = args.val_every and global_step % args.val_every == 0
    if global_step and resumed_at_validation and global_step not in validated:
        run_validation(global_step)

    while global_step < total_steps:
        lr_value = learning_rate(global_step, args.lr, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_value
        optimizer.zero_grad(set_to_none=True)
        batch = next(data_iter)
        codes = batch["codes"].to(device=device, non_blocking=True)
        batch_size = int(codes.shape[0])
        if batch_size not in condition_cache:
            condition_cache[batch_size] = common.batch_condition_tensors(
                lm, checkpoint_info.model_type, batch_size
            )
        with autocast:
            losses = common.compute_batch_losses(lm, codes, condition_cache[batch_size], weights)
        loss = losses["loss"]
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        else:
            grad_norm = torch.nn.utils.get_total_norm(
                [param.grad for param in params if param.grad is not None]
            )
        optimizer.step()
        global_step += 1

        # Summed on the GPU, as the losses are: checked and logged without a sync per step.
        grad_norm = grad_norm.detach()
        log["grad_norm"] += grad_norm
        previous_max = log["max_grad_norm"]
        log["max_grad_norm"] = (
            grad_norm if previous_max is None else torch.maximum(previous_max, grad_norm)
        )
        for key in LOG_LOSSES:
            log["losses"][key] += losses[key].detach()
        log["steps"] += 1
        log["audio_tokens"] += losses["semantic_n"] + losses["acoustic_n"]
        log["text_tokens"] += losses["content_n"] + losses["pad_n"]
        log["streams"] += batch_size
        log["stream_frames"] += int(batch["frames"].sum())
        log["padded_frames"] += batch_size * int(codes.shape[-1])
        log["source_frames"] += int(batch["source_frames"].sum())
        log["min_batch"] = min(log["min_batch"], batch_size)
        log["max_batch"] = max(log["max_batch"], batch_size)
        log["max_frames"] = max(log["max_frames"], int(codes.shape[-1]))

        if args.log_every and global_step % args.log_every == 0:
            steps = log["steps"]
            means = {key: float(value) / steps for key, value in log["losses"].items()}
            if not math.isfinite(means["loss"]):
                raise RuntimeError(f"Non-finite loss by step {global_step}")
            check_gradients(global_step)
            if writer.error is not None:
                writer.wait()  # raises the failed write now, not at the next save
            now = time.time()
            item = {
                "epoch": (global_step - 1) // batches_per_epoch + 1,
                "step": global_step,
                **means,
                "audio_tokens": int(log["audio_tokens"]),
                "text_tokens": int(log["text_tokens"]),
                "streams": log["streams"],
                "stream_frames": log["stream_frames"],
                "padded_frames": log["padded_frames"],
                "source_hours": log["source_frames"] / streams.FRAME_RATE / 3600,
                "min_batch_size": int(log["min_batch"]),
                "max_batch_size": log["max_batch"],
                "max_padded_frames": log["max_frames"],
                "sec_per_step": (now - last_log_time) / steps,
                "log_steps": steps,
                "lr": lr_value,
                "grad_norm": float(log["grad_norm"]) / steps,
                "max_grad_norm": float(log["max_grad_norm"]),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(
                f"step={global_step} loss={item['loss']:.4f} "
                f"audio={item['audio_loss']:.4f} text={item['text_loss']:.4f} "
                f"B={item['streams'] / steps:.1f} T<={item['max_padded_frames']} "
                f"lr={item['lr']:.1e} grad_norm={item['grad_norm']:.3f} "
                f"s/step={item['sec_per_step']:.3f}"
            )
            log = empty_log()
            last_log_time = now

        if args.save_every and global_step % args.save_every == 0:
            check_gradients(global_step)
            writer.save(global_step)
        if args.val_every and global_step % args.val_every == 0:
            run_validation(global_step)

    check_gradients(global_step)
    writer.save(global_step)
    if global_step not in validated:
        run_validation(global_step)
    writer.wait()
    print(
        f"Saved final full-model checkpoint at step {global_step} "
        f"in {repo_display_path(out_dir)}"
    )


if __name__ == "__main__":
    main()
