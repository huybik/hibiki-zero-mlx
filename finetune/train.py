#!/usr/bin/env python
"""Direct full-model Hibiki-Zero Vietnamese-to-English trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from finetune import common  # noqa: E402
from finetune.freeze_full_data_receipt import (  # noqa: E402
    BATCH_SIZE,
    BEST_METRIC,
    CACHE_COUNTS,
    CADENCE_STEPS,
    EPOCHS,
    EXPECTED_ARTIFACTS,
    EXPECTED_CACHE,
    MAX_FRAMES,
    ROWS,
    SEED,
    STEPS_PER_EPOCH,
    TOTAL_STEPS,
    VALIDATION_BATCH_SIZE,
    VALIDATION_MAX_FRAMES,
    VALIDATION_OBSERVED_MAX_FRAMES,
    VALIDATION_ROWS,
)
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_RUN_DIR,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_CACHE_ROOT / "train"],
    )
    parser.add_argument("--val-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--full-data-receipt", type=Path, required=True)
    parser.add_argument("--input-sample-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-pad-loss-weight", type=float, default=0.05)
    parser.add_argument("--val-max-frames", type=int, default=VALIDATION_MAX_FRAMES)
    parser.add_argument("--val-batch-size", type=int, default=VALIDATION_BATCH_SIZE)
    parser.add_argument("--val-every", type=int, default=CADENCE_STEPS)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=CADENCE_STEPS)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-longest-first", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def cache_counts(dataset: common.CachedCodeDataset) -> list[int]:
    return [
        sum(sample["cache_index"] == cache_index for sample in dataset.samples)
        for cache_index in range(dataset.cache_count)
    ]


def load_receipt(path: Path) -> tuple[dict[str, Any], bytes]:
    encoded = path.read_bytes()
    document = json.loads(encoded)
    if set(document) != {"sha256", "full_data_receipt"}:
        raise RuntimeError("Invalid full-data receipt wrapper")
    receipt = document["full_data_receipt"]
    digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != document["sha256"]:
        raise RuntimeError("Full-data receipt SHA-256 mismatch")
    expected = {
        "version": 3,
        "strategy": "direct_voice_preserving_simultaneous_translation",
        "artifacts": EXPECTED_ARTIFACTS,
        "cache": EXPECTED_CACHE,
        "streams": {
            "target_audio": {
                "content": "cached_english_mimi",
                "termination": "native_minus_one_mask",
            },
            "target_text": {
                "content": "cached_ctc_timed_english",
                "termination": "tokenizer_eos",
            },
            "source_audio": {
                "content": "cached_vietnamese_mimi",
                "termination": "explicit_card_eos",
            },
            "transform": None,
        },
        "rows": ROWS,
        "cache_counts": CACHE_COUNTS,
        "cache_weights": [0.95, 0.05],
        "selection_seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_frames": MAX_FRAMES,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps": TOTAL_STEPS,
        "validation_every_steps": CADENCE_STEPS,
        "checkpoint_every_steps": CADENCE_STEPS,
        "best_metric": BEST_METRIC,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"Direct full-data receipt changed: {key}")
    validation = receipt.get("validation", {})
    if (
        validation.get("rows") != VALIDATION_ROWS
        or validation.get("batch_size") != VALIDATION_BATCH_SIZE
        or validation.get("max_frames") != VALIDATION_MAX_FRAMES
        or validation.get("observed_max_frames")
        != VALIDATION_OBSERVED_MAX_FRAMES
        or validation.get("shuffle") is not False
    ):
        raise RuntimeError("Direct validation receipt changed")
    return receipt, encoded


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
    return {
        "target": "full",
        "recipe": "direct_voice_preserving_simultaneous_translation",
        "base_model": repo_display_path(args.model_weight),
    }


def load_best_state(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "best.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if set(state) != {"step", "model", "metric", "validation_loss"}:
        raise RuntimeError("Invalid best-checkpoint metadata")
    step = int(state["step"])
    if state["metric"] != BEST_METRIC:
        raise RuntimeError("Best-checkpoint metric changed")
    model = out_dir / str(state["model"])
    if model.name != f"best_step{step:06d}.safetensors" or not model.is_file():
        raise RuntimeError("Best-checkpoint metadata references a missing model")
    float(state["validation_loss"])
    return state


def promote_best(
    step: int,
    validation_loss: float,
    out_dir: Path,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if previous is not None and validation_loss >= float(previous["validation_loss"]):
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
        "validation_loss": validation_loss,
    }
    atomic_write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", out_dir / "best.json"
    )
    if previous is not None:
        (out_dir / str(previous["model"])).unlink(missing_ok=True)
    for path in out_dir.glob("best_step*.safetensors"):
        if path != model_path:
            path.unlink()
    print(
        f"Promoted best validation loss={validation_loss:.4f} at step {step} "
        f"-> {repo_display_path(model_path)}"
    )
    return state


def save_checkpoint(
    model: Any, optimizer: Any, args: argparse.Namespace, step: int, out_dir: Path
) -> Path:
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
) -> int:
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
    return step


def validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": 1,
        "max_frames": MAX_FRAMES,
        "lr": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "audio_loss_weight": 1.0,
        "text_loss_weight": 1.0,
        "text_pad_loss_weight": 0.05,
        "val_max_frames": VALIDATION_MAX_FRAMES,
        "val_batch_size": VALIDATION_BATCH_SIZE,
        "seed": SEED,
        "gradient_checkpointing": False,
    }
    for key, expected in fixed.items():
        if getattr(args, key) != expected:
            raise ValueError(f"Direct full-data recipe requires {key}={expected}")
    if args.max_steps and (not args.smoke_longest_first or args.max_steps > 11):
        raise ValueError("--max-steps is reserved for the 11-step smoke")
    if args.smoke_longest_first and not args.max_steps:
        raise ValueError("--smoke-longest-first requires --max-steps")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    if not args.smoke_longest_first and (
        args.val_every != CADENCE_STEPS or args.save_every != CADENCE_STEPS
    ):
        raise ValueError(
            f"Direct production requires validation/checkpoints every {CADENCE_STEPS} steps"
        )
    if common.FRAME_BUCKET != 16 or os.environ.get("NO_TORCH_COMPILE") == "1":
        raise ValueError("Direct full-data training requires compile and 16-frame buckets")


def main() -> None:
    args = parse_args()
    validate_args(args)
    common.seed_all(args.seed)

    device = common.check_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full-model training requires CUDA")
    dtype = torch.float32
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    if len(cache_dirs) != len(CACHE_COUNTS):
        raise RuntimeError("Direct receipt requires PhoMT and FLEURS train caches")
    val_cache_dir = require_dir(args.val_cache_dir, "validation code cache directory")
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "upstream model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    args.full_data_receipt = require_file(args.full_data_receipt, "full-data receipt")
    args.input_sample_manifest = require_file(
        args.input_sample_manifest, "input sample manifest"
    )
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")

    receipt, receipt_encoded = load_receipt(args.full_data_receipt)
    artifact_hashes = {
        "config": sha256_file(args.config_path),
        "model": sha256_file(args.model_weight),
        "mimi": sha256_file(args.mimi_weight),
        "tokenizer": sha256_file(args.tokenizer),
    }
    if artifact_hashes != receipt["artifacts"]:
        raise RuntimeError("Training artifacts differ from the direct full-data receipt")
    manifest_sha256 = sha256_file(args.input_sample_manifest)
    if manifest_sha256 != receipt["sample_manifest_sha256"]:
        raise RuntimeError("Training manifest differs from the direct full-data receipt")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_receipt = out_dir / "full_data_receipt.json"
    if run_receipt.is_file() and run_receipt.read_bytes() != receipt_encoded:
        raise RuntimeError("Run receipt differs from the supplied full-data receipt")
    if not run_receipt.is_file():
        atomic_write_text(receipt_encoded.decode(), run_receipt)
    clean_incomplete_checkpoints(out_dir)
    best_state = load_best_state(out_dir)
    best_model = None if best_state is None else out_dir / str(best_state["model"])
    for path in out_dir.glob("best_step*.safetensors"):
        if path != best_model:
            path.unlink()

    dataset = common.CachedCodeDataset(
        cache_dirs,
        False,
        0,
        args.max_frames,
        sample_manifest=args.input_sample_manifest,
        sample_manifest_sha256=manifest_sha256,
    )
    if (
        len(dataset) != ROWS
        or cache_counts(dataset) != CACHE_COUNTS
        or max(sample["frames"] for sample in dataset.samples)
        != receipt["observed_max_frames"]
    ):
        raise RuntimeError("Loaded training data differs from the direct receipt")
    exposure = dataset.exposure()
    print(
        f"Loaded {len(dataset)} direct cached samples; "
        f"assembled_frames={exposure['assembled_frames']:,} "
        f"source_hours={exposure['source_hours']:.2f}"
    )

    val_dataset = common.CachedCodeDataset(
        val_cache_dir, True, 0, args.val_max_frames
    )
    if (
        len(val_dataset) != receipt["validation"]["rows"]
        or max(sample["frames"] for sample in val_dataset.samples)
        != receipt["validation"]["observed_max_frames"]
    ):
        raise RuntimeError("Loaded validation data differs from the direct receipt")
    if args.smoke_longest_first:
        val_dataset.samples.reverse()
    val_dataloader = common.make_cached_dataloader(
        val_dataset,
        args.val_batch_size,
        args.num_workers,
        True,
        seed=args.seed,
        shuffle=False,
    )
    print(f"Loaded {len(val_dataset)} non-shuffled validation samples")

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = batches_per_epoch // args.grad_accum_steps
    if steps_per_epoch != STEPS_PER_EPOCH:
        raise RuntimeError(f"Direct full-data steps per epoch changed: {steps_per_epoch}")
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    if not args.max_steps and total_steps != TOTAL_STEPS:
        raise RuntimeError(f"Direct full-data total steps changed: {total_steps}")

    checkpoint_info = common.load_checkpoint_info(args)
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
        global_step = load_resume_checkpoint(
            lm, optimizer, args.resume_checkpoint, device, dtype
        )
    if global_step >= total_steps:
        raise RuntimeError("Resume checkpoint is already at or beyond the requested stop")
    if args.resume_checkpoint is not None and global_step >= CADENCE_STEPS and best_state is None:
        raise RuntimeError("Production resume requires the previously promoted best checkpoint")

    manifest_path = out_dir / "sample_manifest.jsonl"
    manifest_encoded = args.input_sample_manifest.read_bytes()
    if manifest_path.is_file() and manifest_path.read_bytes() != manifest_encoded:
        raise RuntimeError("Run sample manifest changed")
    if not manifest_path.is_file():
        atomic_write_text(manifest_encoded.decode(), manifest_path)

    def jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [jsonable(item) for item in value]
        return value

    run_config = {key: jsonable(value) for key, value in vars(args).items()}
    run_config.update(
        {
            "total_steps": total_steps,
            "batches_per_epoch": batches_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "frame_bucket": common.FRAME_BUCKET,
            "torch_compile_enabled": True,
            "sample_manifest_sha256": manifest_sha256,
            "sample_manifest_rows": len(dataset),
            "sample_manifest_cache_counts": cache_counts(dataset),
            "receipt_sha256": hashlib.sha256(receipt_encoded).hexdigest(),
            "initialization": "upstream_hibiki_zero",
            "target_audio_teacher_forcing": True,
            "audio_ce": True,
            "post_source_transform": None,
            "validation_shuffle": False,
            "best_metric": BEST_METRIC,
            "observed_train_max_frames": max(
                sample["frames"] for sample in dataset.samples
            ),
            "observed_val_max_frames": max(
                sample["frames"] for sample in val_dataset.samples
            ),
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
        resume_keys = (
            "cache_dir",
            "val_cache_dir",
            "config_path",
            "model_weight",
            "mimi_weight",
            "tokenizer",
            "full_data_receipt",
            "input_sample_manifest",
            "epochs",
            "batch_size",
            "grad_accum_steps",
            "max_frames",
            "lr",
            "adam_beta1",
            "adam_beta2",
            "weight_decay",
            "grad_clip",
            "audio_loss_weight",
            "text_loss_weight",
            "text_pad_loss_weight",
            "val_max_frames",
            "val_batch_size",
            "seed",
            "sample_manifest_sha256",
            "receipt_sha256",
            "initialization",
            "target_audio_teacher_forcing",
            "audio_ce",
            "post_source_transform",
            "validation_shuffle",
            "best_metric",
            "val_every",
            "save_every",
            "keep_checkpoints",
        )
        for key in resume_keys:
            if previous.get(key) != run_config.get(key):
                raise RuntimeError(f"Direct same-run resume contract changed: {key}")
        if not args.smoke_longest_first and previous.get("total_steps") != total_steps:
            raise RuntimeError("Direct same-run stop step changed")

    sample_order = None
    if args.smoke_longest_first:
        sample_order = sorted(
            range(len(dataset)),
            key=lambda index: dataset.samples[index]["frames"],
            reverse=True,
        )
    dataloader = common.make_cached_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        False,
        seed=args.seed,
        shuffle=False,
        sample_order=sample_order,
    )
    data_iter = iter(dataloader)
    resume_skip_batches = global_step % batches_per_epoch
    while resume_skip_batches:
        next(data_iter)
        resume_skip_batches -= 1

    condition_cache: dict[int, Any | None] = {}
    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    log_sums = {
        "loss": 0.0,
        "audio_loss": 0.0,
        "text_loss": 0.0,
        "content_text_loss": 0.0,
        "pad_text_loss": 0.0,
    }
    log_steps = 0
    log_audio_tokens = 0
    log_text_tokens = 0
    log_samples = 0
    log_assembled_frames = 0
    log_padded_frames = 0
    log_source_frames = 0
    log_min_batch_size = math.inf
    log_max_batch_size = 0
    log_max_frames = 0
    log_max_padded_frames = 0
    last_log_time = time.time()

    def run_teacher_forced_val(step: int) -> None:
        nonlocal best_state
        metrics = common.evaluate_teacher_forced(
            lm,
            val_dataloader,
            device,
            checkpoint_info.model_type,
            args.audio_loss_weight,
            args.text_loss_weight,
            args.val_batches,
            "prefix",
            args.text_pad_loss_weight,
        )
        keys = (
            "loss",
            "audio_loss",
            "text_loss",
            "audio_tokens",
            "text_tokens",
            "samples",
            "content_text_loss",
            "content_acc",
            "content_tokens",
            "pad_text_loss",
            "pad_acc",
            "pad_tokens",
            "max_frames",
            "max_padded_frames",
        )
        item = {"step": step, **{key: metrics[key] for key in keys}}
        item.update(common.mps_memory_stats(device))
        with val_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={metrics['loss']:.4f} "
            f"audio={metrics['audio_loss']:.4f} text={metrics['text_loss']:.4f} "
            f"content={metrics['content_text_loss']:.4f} "
            f"acc={metrics['content_acc']:.3f} pad_acc={metrics['pad_acc']:.3f}"
        )
        best_state = promote_best(
            step, float(metrics["loss"]), out_dir, best_state
        )

    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        lr_value = args.lr
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        codes = batch["codes"].to(device=device, dtype=torch.long)
        batch_size = int(codes.shape[0])
        if batch_size not in condition_cache:
            condition_cache[batch_size] = common.batch_condition_tensors(
                lm, checkpoint_info.model_type, batch_size
            )
        with autocast:
            losses = common.compute_batch_losses(
                lm,
                codes,
                condition_cache[batch_size],
                args.audio_loss_weight,
                args.text_loss_weight,
                text_pad_loss_weight=args.text_pad_loss_weight,
                text_pad_mode="prefix",
            )
        loss = losses["loss"]
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        global_step += 1

        log_sums["loss"] += loss.detach()
        log_sums["audio_loss"] += losses["audio_loss"].detach()
        log_sums["text_loss"] += losses["text_loss"].detach()
        log_sums["content_text_loss"] += losses["content_text_loss"].detach()
        log_sums["pad_text_loss"] += losses["pad_text_loss"].detach()
        log_steps += 1
        log_audio_tokens += losses["audio_tokens"]
        log_text_tokens += losses["text_tokens"]
        log_samples += batch_size
        log_assembled_frames += int(batch["frames"].sum())
        log_padded_frames += batch_size * int(codes.shape[-1])
        log_source_frames += int(batch["source_frames"].sum())
        log_min_batch_size = min(log_min_batch_size, batch_size)
        log_max_batch_size = max(log_max_batch_size, batch_size)
        log_max_frames = max(log_max_frames, int(batch["frames"].max()))
        log_max_padded_frames = max(log_max_padded_frames, int(codes.shape[-1]))

        if args.log_every and global_step % args.log_every == 0:
            loss_average = float(log_sums["loss"]) / log_steps
            if not math.isfinite(loss_average):
                raise RuntimeError(f"Non-finite loss by step {global_step}")
            now = time.time()
            item = {
                "epoch": (global_step - 1) // steps_per_epoch + 1,
                "step": global_step,
                "loss": loss_average,
                "audio_loss": float(log_sums["audio_loss"]) / log_steps,
                "text_loss": float(log_sums["text_loss"]) / log_steps,
                "content_text_loss": float(log_sums["content_text_loss"]) / log_steps,
                "pad_text_loss": float(log_sums["pad_text_loss"]) / log_steps,
                "audio_tokens": int(log_audio_tokens),
                "text_tokens": int(log_text_tokens),
                "samples": log_samples,
                "microbatches": log_steps,
                "assembled_frames": log_assembled_frames,
                "padded_frames": log_padded_frames,
                "source_hours": log_source_frames / float(dataset.frame_rate) / 3600,
                "min_batch_size": int(log_min_batch_size),
                "max_batch_size": log_max_batch_size,
                "max_frames": log_max_frames,
                "max_padded_frames": log_max_padded_frames,
                "sec_per_step": (now - last_log_time) / log_steps,
                "log_steps": log_steps,
                "lr": lr_value,
                "audio_weight": args.audio_loss_weight,
                "text_weight": args.text_loss_weight,
                "target_audio_teacher_forcing": True,
            }
            item.update(common.mps_memory_stats(device))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(
                f"step={global_step} loss={item['loss']:.4f} "
                f"audio={item['audio_loss']:.4f} text={item['text_loss']:.4f} "
                f"B={item['samples'] / item['microbatches']:.1f} "
                f"T<={item['max_frames']} lr={item['lr']:.1e} "
                f"s/step={item['sec_per_step']:.3f}"
            )
            log_sums = {key: 0.0 for key in log_sums}
            log_steps = 0
            log_audio_tokens = 0
            log_text_tokens = 0
            log_samples = 0
            log_assembled_frames = 0
            log_padded_frames = 0
            log_source_frames = 0
            log_min_batch_size = math.inf
            log_max_batch_size = 0
            log_max_frames = 0
            log_max_padded_frames = 0
            last_log_time = now

        if args.save_every and global_step % args.save_every == 0:
            save_checkpoint(lm, optimizer, args, global_step, out_dir)
        if args.val_every and global_step % args.val_every == 0:
            run_teacher_forced_val(global_step)

    save_checkpoint(lm, optimizer, args, global_step, out_dir)
    if not args.val_every or global_step % args.val_every:
        run_teacher_forced_val(global_step)
    print(
        f"Saved final direct full-model checkpoint at step {global_step} "
        f"in {repo_display_path(out_dir)}"
    )


if __name__ == "__main__":
    main()
