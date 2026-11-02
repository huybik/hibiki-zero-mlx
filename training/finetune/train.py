#!/usr/bin/env python
"""Full-model Hibiki-Zero Vietnamese-to-English SFT on multi-sentence streams.

Trains every parameter from the upstream weight with fp32 masters, bf16 autocast and fused
AdamW, under a linear warmup then cosine decay. Streams are planned once from the grounded cache
rows with --seed (`streams.py`) and batched up to --batch-frames padded frames; every epoch
replays that order. Each validation reports teacher-forced losses on a fixed, seeded stream plan
and batch order of the validation cache (so --val-batches takes a random subset) and translates
the --eval-pairs rows of real Vietnamese speech free-running (FLEURS dev by default, as
`eval.py` does). Of the checkpoints that pass the generation health gates, the one with the
highest chrF is promoted to best_stepNNNNNN.safetensors. Resume only from the newest complete
model/trainer pair of the same run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
    # No abbreviations: h100.sh must be able to reject --out-dir/--resume-checkpoint.
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
        default=Path("data/eval/fleurs-vi/dev.jsonl"),
        help="Pairs (training.prep.eval_sets) translated free-running at each validation.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=9000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
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
    for pattern in (".model_step*.safetensors.tmp", ".trainer_step*.pt.tmp"):
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
    healthy: bool,
    out_dir: Path,
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Promote the checkpoint of `step` if it passed the generation health gates with a higher
    chrF than the promoted one."""
    if not healthy or (previous is not None and chrf <= float(previous["chrf"])):
        return previous
    checkpoint_path = require_file(
        out_dir / f"model_step{step:06d}.safetensors",
        "validation-cadence model checkpoint",
    )
    model_path = out_dir / f"best_step{step:06d}.safetensors"
    model_path.unlink(missing_ok=True)
    os.link(checkpoint_path, model_path)
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


def save_checkpoint(
    model: Any,
    optimizer: Any,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
    best_state: dict[str, Any] | None,
) -> Path:
    """Save the model and trainer pair of `step`, recording the checkpoint promoted so far."""
    model_path = out_dir / f"model_step{step:06d}.safetensors"
    trainer_path = out_dir / f"trainer_step{step:06d}.pt"
    clean_incomplete_checkpoints(out_dir)
    if model_path.is_file() and trainer_path.is_file():
        prune_checkpoint_pairs(out_dir, args.keep_checkpoints)
        return model_path
    if args.keep_checkpoints > 0:
        prune_checkpoint_pairs(out_dir, max(1, args.keep_checkpoints - 1))
    common.save_model(model, model_path, build_metadata(args))
    try:
        atomic_torch_save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "model": model_path.name,
                "best_step": None if best_state is None else int(best_state["step"]),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            trainer_path,
        )
    except BaseException:
        model_path.unlink(missing_ok=True)
        raise
    prune_checkpoint_pairs(out_dir, args.keep_checkpoints)
    return model_path


def load_resume_checkpoint(
    model: Any,
    optimizer: Any,
    resume_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[int, int | None]:
    """Load the pair; returns its step and the step promoted when it was saved."""
    pairs = checkpoint_pairs(resume_path.parent)
    if not pairs or pairs[-1][2].resolve() != resume_path.resolve():
        raise RuntimeError("--resume-checkpoint must be the newest complete pair")
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model_path = require_file(resume_path.parent / checkpoint["model"], "resume model")
    common.load_model(model, model_path, dtype)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    step = int(checkpoint["step"])
    print(f"Resumed step {step} from {repo_display_path(resume_path)}")
    return step, checkpoint["best_step"]


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
    args.eval_pairs = require_file(args.eval_pairs, "evaluation pairs")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_incomplete_checkpoints(out_dir)
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
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=val_dataset.batches(args.batch_frames, streams.VALIDATION_SEED),
        collate_fn=streams.collate,
        num_workers=args.num_workers,
    )
    print(f"Planned {len(val_dataset)} validation streams from {len(val_rows)} rows")
    # Checkpoints are selected on free-running translation, sampled as eval.py samples.
    generation = common.GenerationConfig(text_only=True)
    eval_rows = common.read_eval_rows(args.eval_pairs)
    if not eval_rows:
        raise RuntimeError(f"No rows in {args.eval_pairs}")
    common.validate_eval_rows(
        eval_rows, generation.source_column, generation.reference_column, generation.id_column
    )
    for row in eval_rows:
        require_file(row[generation.source_column], "evaluation audio")
    print(f"Validation translates {len(eval_rows)} rows of {repo_display_path(args.eval_pairs)}")

    batches_per_epoch = len(batches)
    total_steps = args.max_steps or batches_per_epoch * args.epochs
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

    global_step = 0
    if args.resume_checkpoint is not None:
        global_step, resumed_best = load_resume_checkpoint(
            lm, optimizer, args.resume_checkpoint, device, dtype
        )
        # A replacement pod restores the promoted checkpoint from the recovery repo's best/.
        if resumed_best is not None and (
            best_state is None or int(best_state["step"]) < resumed_best
        ):
            raise RuntimeError(
                f"Resume requires the checkpoint promoted at step {resumed_best} or a later one"
            )
    if global_step >= total_steps:
        raise RuntimeError("Resume checkpoint is already at or beyond the requested stop")

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
    if args.resume_checkpoint is None:
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

    def training_batches(start_step: int):
        """The planned batches from `start_step` on, epoch after epoch."""
        offset = start_step % batches_per_epoch
        while True:
            yield from DataLoader(
                dataset,
                batch_sampler=batches[offset:],
                collate_fn=streams.collate,
                num_workers=args.num_workers,
            )
            offset = 0

    data_iter = training_batches(global_step)
    condition_cache: dict[int, Any | None] = {}
    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"

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
        }

    log = empty_log()
    last_log_time = time.time()

    def run_validation(step: int) -> None:
        nonlocal best_state
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
        lm.eval()
        records, free_running = common.run_greedy_eval(
            eval_rows,
            generation,
            args.eval_batch_size,
            mimi,
            lm,
            text_tokenizer,
            checkpoint_info,
            eval_dir,
            args.seed,
        )
        lm.train()
        common.write_predictions(eval_dir / "predictions.csv", records)
        healthy, gates = common.generation_health(free_running)
        free_running.update(health_eligible=healthy, health_gates=gates)
        item = {"step": step, "teacher_forced": teacher_forced, "free_running": free_running}
        with val_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={teacher_forced['loss']:.4f} "
            f"semantic={teacher_forced['semantic_loss']:.4f} "
            f"content={teacher_forced['content_text_loss']:.4f} "
            f"acc={teacher_forced['content_acc']:.3f} pad_acc={teacher_forced['pad_acc']:.3f} "
            f"chrf={free_running['chrf']:.2f} bleu={free_running['bleu']:.2f} "
            f"length_ratio={free_running['mean_length_ratio']:.2f} "
            f"eos={free_running['eos_found']}/{free_running['num_predictions']} "
            f"health={'pass' if healthy else 'fail'}"
        )
        best_state = promote_best(step, float(free_running["chrf"]), healthy, out_dir, best_state)

    while global_step < total_steps:
        lr_value = learning_rate(global_step, args.lr, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_value
        optimizer.zero_grad(set_to_none=True)
        batch = next(data_iter)
        codes = batch["codes"].to(device=device, dtype=torch.long)
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
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        global_step += 1

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
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(
                f"step={global_step} loss={item['loss']:.4f} "
                f"audio={item['audio_loss']:.4f} text={item['text_loss']:.4f} "
                f"B={item['streams'] / steps:.1f} T<={item['max_padded_frames']} "
                f"lr={item['lr']:.1e} s/step={item['sec_per_step']:.3f}"
            )
            log = empty_log()
            last_log_time = now

        if args.save_every and global_step % args.save_every == 0:
            save_checkpoint(lm, optimizer, args, global_step, out_dir, best_state)
        if args.val_every and global_step % args.val_every == 0:
            run_validation(global_step)

    save_checkpoint(lm, optimizer, args, global_step, out_dir, best_state)
    if not args.val_every or global_step % args.val_every:
        run_validation(global_step)
    print(
        f"Saved final full-model checkpoint at step {global_step} "
        f"in {repo_display_path(out_dir)}"
    )


if __name__ == "__main__":
    main()
