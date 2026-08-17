#!/usr/bin/env python
"""Full-model Hibiki-Zero SFT trainer for cached Vietnamese-to-English codes."""
from __future__ import annotations

import argparse
import contextlib
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
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_PAIRS_DIR,
    DEFAULT_RUN_DIR,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

POST_SOURCE_EOS_EXTENSION_START_STEP = 1_000
POST_SOURCE_EOS_EXTENSION_END_STEP = 3_125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all Hibiki-Zero parameters on cached vi->en codes."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_CACHE_ROOT / "train"],
        help="One or more cache dirs; shards from all are pooled (e.g. FLEURS + PhoMT).",
    )
    parser.add_argument(
        "--cache-weights",
        type=float,
        nargs="+",
        help="Target sampling proportions, one per --cache-dir.",
    )
    parser.add_argument("--val-cache-dir", type=Path, help="Cached val split for teacher-forced CE.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional full-model initialization; optimizer always starts fresh.",
    )
    parser.add_argument("--init-checkpoint-sha256")
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda", help="CUDA device for full-model training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--max-samples", type=int, default=0, help="First N cached samples; 0=all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Drop cached samples longer than N Mimi frames; 0=off.")
    parser.add_argument("--expected-target-delay-min-ratio", type=float)
    parser.add_argument("--expected-target-delay-max-ratio", type=float)
    parser.add_argument("--expected-target-delay-seed", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", default="", help='Learning-rate schedule "lr@frac,...".')
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps; 0=off.")
    parser.add_argument("--cosine-lr-end", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--audio-weight-schedule", default="", help="Audio loss-weight schedule.")
    parser.add_argument(
        "--mask-target-audio-input",
        action="store_true",
        help="Replace teacher-forced target-audio inputs with zero tokens; requires zero audio weight.",
    )
    parser.add_argument("--contrastive-source-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-source-margin", type=float, default=0.5)
    parser.add_argument(
        "--source-asr-pretrain",
        action="store_true",
        help="Emit Vietnamese text after source EOS with no target-audio supervision.",
    )
    parser.add_argument(
        "--source-asr-ascii",
        action="store_true",
        help="Strip Vietnamese diacritics from source-ASR targets and eval references.",
    )
    parser.add_argument(
        "--post-source-eos-translation",
        action="store_true",
        help="Emit cached English text only after Vietnamese source EOS.",
    )
    parser.add_argument(
        "--post-source-eos-extension",
        action="store_true",
        help="Resume the exact 1,000-step post-source-EOS run through one 50k pass.",
    )
    parser.add_argument("--source-asr-replay-weight", type=float, default=0.0)
    parser.add_argument("--source-asr-replay-batch-size", type=int, default=4)
    parser.add_argument("--source-asr-replay-max-frames", type=int, default=0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-weight-schedule", default="", help='Text loss-weight schedule "5@0,2@0.6".')
    parser.add_argument(
        "--text-pad-loss-weight",
        type=float,
        default=0.05,
        help="Aggregate PAD-loss weight after PAD and content are reduced independently.",
    )
    parser.add_argument(
        "--first-content-loss-weight",
        type=float,
        default=0.0,
        help="Extra aggregate weight for the first non-PAD target token in each row.",
    )
    parser.add_argument(
        "--text-pad-mode",
        choices=("prefix", "all"),
        default="prefix",
        help="Supervise prefix PAD only, or all valid PAD timing positions.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Optimizer steps, 0 means all.")
    # Teacher-forced CE validation (cheap, cached).
    parser.add_argument("--val-every", type=int, default=0, help="Steps between teacher-forced val; 0=final.")
    parser.add_argument("--val-max-samples", type=int, default=0, help="First N val samples; 0=all.")
    parser.add_argument("--val-batches", type=int, default=0, help="First N val batches; 0=all.")
    parser.add_argument(
        "--val-max-frames",
        type=int,
        default=0,
        help="Hard maximum validation length in Mimi frames; 0=off.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        help="Teacher-forced validation batch size; defaults to --batch-size.",
    )
    # Paired free-running val eval + best-checkpoint selection.
    parser.add_argument("--eval-every", type=int, default=0, help="Steps between paired val eval; 0=off.")
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="Run a non-promotable step-0 greedy baseline before training.",
    )
    parser.add_argument("--eval-pairs", type=Path, default=DEFAULT_PAIRS_DIR / "validation.jsonl")
    parser.add_argument("--eval-limit", type=int, default=128, help="Paired val rows (val128 gate).")
    parser.add_argument("--eval-ids-file", type=Path, help="Optional id file for the greedy eval set.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-text-temp", type=float, default=0.4)
    parser.add_argument("--eval-audio-temp", type=float, default=0.8)
    parser.add_argument("--eval-top-k", type=int, default=250)
    parser.add_argument("--eval-top-k-text", type=int, default=250)
    parser.add_argument("--eval-tail-s", type=float, default=8.0)
    parser.add_argument("--eval-gen-duration", type=float, default=0.0)
    parser.add_argument("--eval-source-column", default="vi_audio")
    parser.add_argument("--eval-duration-column", default="vi_duration_s")
    parser.add_argument("--eval-reference-column", default="text_en")
    parser.add_argument("--eval-id-column", default="id")
    parser.add_argument(
        "--eval-derangement",
        type=Path,
        help="Frozen duration-matched derangement; default <out-dir>/eval_derangement.json.",
    )
    parser.add_argument("--min-source-bleu-gap", type=float, default=None)
    parser.add_argument("--min-source-chrf-gap", type=float, default=None)
    parser.add_argument("--min-correct-chrf", type=float)
    parser.add_argument("--max-correct-wer", type=float)
    # Bookkeeping.
    parser.add_argument("--save-every", type=int, default=50, help="Steps between saves.")
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Keep only the newest N complete step pairs (versioned best untouched); 0=keep all. "
        "Full-finetune saves are ~35 GB each — rotation is essential on rented disks.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="Steps between logs.")
    parser.add_argument("--resume-checkpoint", type=Path, help="trainer_step*.pt to resume from.")
    parser.add_argument("--seed", type=int, default=1234, help="Training and data-order RNG seed.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--persist-sample-manifest",
        action="store_true",
        help="Freeze ordered cache_index/id training membership and verify it on resume.",
    )
    parser.add_argument(
        "--input-sample-manifest",
        type=Path,
        help="Authoritative ordered cache_index/id membership, including repeats.",
    )
    parser.add_argument("--input-sample-manifest-sha256")
    parser.add_argument(
        "--smoke-longest-first",
        action="store_true",
        help="Smoke only: feed longest manifest rows first without changing persisted membership.",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", help="Pass gradient_checkpointing=True."
    )
    parser.add_argument(
        "--sort-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort cached samples by frame length to reduce padding.",
    )
    return parser.parse_args()


def build_metadata(args: argparse.Namespace) -> dict[str, str]:
    metadata = {
        "target": "full",
        "base_model": repo_display_path(args.model_weight),
    }
    if args.init_checkpoint is not None:
        metadata["init_checkpoint_sha256"] = args.init_checkpoint_sha256
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return [(step, models[step], trainers[step]) for step in sorted(models.keys() & trainers.keys())]


def load_best_state(out_dir: Path) -> dict[str, Any] | None:
    marker = out_dir / "best.json"
    if not marker.is_file():
        return None
    state = json.loads(marker.read_text(encoding="utf-8"))
    step = int(state["step"])
    float(state["correct"]["bleu"])
    float(state["correct"]["chrf"])
    model_name = str(state["model"])
    if model_name != f"best_step{step:06d}.safetensors":
        raise RuntimeError("best.json model does not match its step")
    model = out_dir / model_name
    if not model.is_file():
        raise RuntimeError(f"best.json references a missing model: {model}")
    if not state.get("promotion_eligible"):
        raise RuntimeError("best.json references an ineligible checkpoint")
    return state


def clean_incomplete_checkpoints(out_dir: Path) -> None:
    pairs = checkpoint_pairs(out_dir)
    paired_models = {model for _, model, _ in pairs}
    paired_trainers = {trainer for _, _, trainer in pairs}
    for trainer in out_dir.glob("trainer_step*.pt"):
        if trainer not in paired_trainers:
            trainer.unlink()
    for model in out_dir.glob("model_step*.safetensors"):
        if model not in paired_models:
            model.unlink()
    for pattern in (
        ".model_step*.safetensors.tmp",
        ".trainer_step*.pt.tmp",
        ".best_step*.safetensors.tmp",
        ".best.json.tmp",
    ):
        for temp_path in out_dir.glob(pattern):
            temp_path.unlink()
    best_state = load_best_state(out_dir)
    best_model = out_dir / best_state["model"] if best_state is not None else None
    for model in out_dir.glob("best_step*.safetensors"):
        if model != best_model:
            model.unlink()


def prune_checkpoint_pairs(out_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    pairs = checkpoint_pairs(out_dir)
    for _, model, trainer in pairs[: max(0, len(pairs) - keep)]:
        trainer.unlink()
        model.unlink()


def atomic_torch_save(payload: Any, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def freeze_sample_manifest(
    dataset: common.CachedCodeDataset, out_dir: Path, resume: bool
) -> str:
    content = "".join(
        json.dumps(
            {"cache_index": sample["cache_index"], "id": sample["id"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for sample in dataset.samples
    )
    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = out_dir / "sample_manifest.jsonl"
    if resume:
        if not path.is_file():
            raise RuntimeError("Resume requires the original sample_manifest.jsonl")
        if path.read_bytes() != encoded:
            raise RuntimeError("Selected training membership differs from sample_manifest.jsonl")
    else:
        atomic_write_text(content, path)
    return digest


def save_checkpoint(
    model: Any, optimizer: Any, args: argparse.Namespace, step: int, out_dir: Path
) -> Path:
    model_path = out_dir / f"model_step{step:06d}.safetensors"
    trainer_path = out_dir / f"trainer_step{step:06d}.pt"
    clean_incomplete_checkpoints(out_dir)
    if model_path.is_file() and trainer_path.is_file():
        prune_checkpoint_pairs(out_dir, args.keep_checkpoints)
        print(f"Checkpoint step {step} already complete; skipping duplicate save.")
        return model_path
    if args.keep_checkpoints > 0:
        # Keep one known-good recovery pair while writing the next. For keep>=2,
        # pre-pruning avoids the old N+1-pair transient disk spike.
        prune_checkpoint_pairs(out_dir, max(1, args.keep_checkpoints - 1))
    common.save_model(model, model_path, build_metadata(args))
    try:
        atomic_torch_save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "model": model_path.name,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            trainer_path,
        )
    except BaseException:
        model_path.unlink(missing_ok=True)
        raise
    prune_checkpoint_pairs(out_dir, args.keep_checkpoints)
    return model_path


def load_resume_checkpoint(
    model: Any, optimizer: Any, resume_path: Path, device: torch.device, dtype: torch.dtype
) -> int:
    pairs = checkpoint_pairs(resume_path.parent)
    if not pairs or pairs[-1][2].resolve() != resume_path.resolve():
        raise RuntimeError("--resume-checkpoint must be the newest complete pair in its directory")
    # weights_only=False: our own trusted checkpoint; its `args` dict holds Path
    # objects that PyTorch 2.6's default weights_only=True refuses to unpickle.
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model_path = require_file(resume_path.parent / checkpoint["model"], "resume model")
    common.load_model(model, model_path, dtype)
    # load_state_dict restores the saved run's param-group hyperparams, including
    # our custom "points" schedule — reassert this run's --lr-schedule/names.
    fresh_groups = [{key: group[key] for key in ("name", "points")} for group in optimizer.param_groups]
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group, fresh in zip(optimizer.param_groups, fresh_groups, strict=True):
        group.update(fresh)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    step = int(checkpoint["step"])
    print(f"Resumed step {step} from {repo_display_path(resume_path)}")
    return step


def freeze_post_source_eos_extension(
    path: Path, payload: dict[str, Any], global_step: int
) -> None:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected = {"sha256": digest, "post_source_eos_extension": payload}
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Post-source-EOS extension contract changed")
    if not path.is_file() and global_step != POST_SOURCE_EOS_EXTENSION_START_STEP:
        raise RuntimeError(
            "First post-source-EOS extension resume requires the exact step-1000 pair"
        )
    if not path.is_file():
        atomic_write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", path)


def order_validation_samples(
    dataset: common.CachedCodeDataset, sort_by_length: bool, longest_first: bool
) -> int:
    if sort_by_length:
        dataset.samples.sort(key=lambda sample: sample["frames"])
    observed_max_frames = max(int(sample["frames"]) for sample in dataset.samples)
    if longest_first:
        dataset.samples.reverse()
    return observed_max_frames


def main() -> None:
    args = parse_args()
    torch_compile_enabled = os.environ.get("NO_TORCH_COMPILE") != "1"
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.val_batch_size <= 0:
        raise ValueError("--val-batch-size must be positive")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.val_max_frames < 0:
        raise ValueError("--val-max-frames must be non-negative")
    delay_values = (
        args.expected_target_delay_min_ratio,
        args.expected_target_delay_max_ratio,
        args.expected_target_delay_seed,
    )
    if any(value is not None for value in delay_values) and any(
        value is None for value in delay_values
    ):
        raise ValueError("Expected target-delay min, max, and seed must be set together")
    expected_target_delay = None
    if all(value is not None for value in delay_values):
        if not (
            math.isfinite(args.expected_target_delay_min_ratio)
            and math.isfinite(args.expected_target_delay_max_ratio)
            and 0
            <= args.expected_target_delay_min_ratio
            <= args.expected_target_delay_max_ratio
        ):
            raise ValueError("Expected target-delay ratios must satisfy finite 0 <= min <= max")
        expected_target_delay = {
            "min_ratio": args.expected_target_delay_min_ratio,
            "max_ratio": args.expected_target_delay_max_ratio,
            "seed": args.expected_target_delay_seed,
        }
    if (args.input_sample_manifest is None) != (args.input_sample_manifest_sha256 is None):
        raise ValueError("--input-sample-manifest and its SHA-256 must be set together")
    if (args.init_checkpoint is None) != (args.init_checkpoint_sha256 is None):
        raise ValueError("--init-checkpoint and its SHA-256 must be set together")
    if args.input_sample_manifest is not None and not args.persist_sample_manifest:
        raise ValueError("Authoritative input membership requires --persist-sample-manifest")
    if args.post_source_eos_extension and (
        not args.post_source_eos_translation
        or args.resume_checkpoint is None
        or args.smoke_longest_first
        or args.max_steps != POST_SOURCE_EOS_EXTENSION_END_STEP
        or args.val_every != 500
        or args.eval_every != 500
        or args.save_every != 500
        or args.keep_checkpoints != 2
    ):
        raise ValueError(
            "--post-source-eos-extension requires a production post-source-EOS resume "
            "through step 3125 with val/eval/save cadence 500 and keep-checkpoints 2"
        )
    if args.smoke_longest_first and (
        args.input_sample_manifest is None or not args.max_steps or args.max_steps > 11
    ):
        raise ValueError("--smoke-longest-first requires manifest input and at most 11 steps")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.text_pad_loss_weight < 0:
        raise ValueError("--text-pad-loss-weight must be non-negative")
    if args.first_content_loss_weight < 0:
        raise ValueError("--first-content-loss-weight must be non-negative")
    if args.contrastive_source_weight < 0 or args.contrastive_source_margin < 0:
        raise ValueError("Contrastive source weight and margin must be non-negative")
    if args.contrastive_source_weight and args.input_sample_manifest is None:
        raise ValueError("Contrastive source loss requires authoritative input membership")
    if args.source_asr_pretrain and args.input_sample_manifest is None:
        raise ValueError("Source-ASR preadaptation requires authoritative input membership")
    if args.source_asr_pretrain and args.contrastive_source_weight:
        raise ValueError("Source-ASR preadaptation and contrastive translation are exclusive")
    if args.post_source_eos_translation and (
        args.source_asr_pretrain
        or args.contrastive_source_weight
        or args.source_asr_replay_weight
    ):
        raise ValueError(
            "Post-source-EOS translation is exclusive with ASR and contrastive objectives"
        )
    if args.post_source_eos_translation and args.init_checkpoint is None:
        raise ValueError("Post-source-EOS translation requires exact initialization")
    if args.post_source_eos_translation and args.input_sample_manifest is None and (
        not args.persist_sample_manifest
        or args.cache_weights is None
        or args.max_samples <= 0
    ):
        raise ValueError(
            "Post-source-EOS translation without input membership requires an exact "
            "weighted sample count and persisted manifest"
        )
    if args.source_asr_replay_weight < 0:
        raise ValueError("--source-asr-replay-weight must be non-negative")
    if args.source_asr_replay_weight and (
        args.source_asr_pretrain or args.contrastive_source_weight
    ):
        raise ValueError("Source-ASR replay is exclusive with ASR preadaptation and contrastive loss")
    if args.source_asr_replay_weight and (
        args.input_sample_manifest is None or args.init_checkpoint is None
    ):
        raise ValueError("Source-ASR replay requires authoritative membership and initialization")
    if args.source_asr_replay_weight and (
        args.source_asr_replay_batch_size <= 0 or args.source_asr_replay_max_frames <= 0
    ):
        raise ValueError("Source-ASR replay requires a positive batch size and frame cap")
    if args.source_asr_ascii and not (
        args.source_asr_pretrain or args.source_asr_replay_weight
    ):
        raise ValueError("--source-asr-ascii requires source-ASR preadaptation or replay")
    if args.source_asr_replay_weight and not args.source_asr_ascii:
        raise ValueError("Source-ASR replay requires --source-asr-ascii")
    if args.source_asr_pretrain and args.eval_every and args.eval_reference_column != "text_vi":
        raise ValueError("Source-ASR evaluation requires --eval-reference-column text_vi")
    if args.cache_weights is not None and len(args.cache_weights) != len(args.cache_dir):
        raise ValueError("--cache-weights must match --cache-dir")
    if args.eval_every and (
        args.min_source_bleu_gap is None or args.min_source_chrf_gap is None
    ):
        raise ValueError(
            "Best promotion requires explicit --min-source-bleu-gap and --min-source-chrf-gap"
        )
    if args.eval_every and args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if any(
        value is not None and not math.isfinite(value)
        for value in (
            args.min_source_bleu_gap,
            args.min_source_chrf_gap,
            args.min_correct_chrf,
            args.max_correct_wer,
        )
    ):
        raise ValueError("Evaluation thresholds must be finite")
    if args.min_correct_chrf is not None and args.min_correct_chrf < 0:
        raise ValueError("--min-correct-chrf must be non-negative")
    if args.max_correct_wer is not None and args.max_correct_wer < 0:
        raise ValueError("--max-correct-wer must be non-negative")
    if not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in (0, 1)")

    common.seed_all(args.seed)

    device = common.check_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full-model training requires CUDA.")
    dtype = torch.float32
    # TF32 matmuls: fp32 master weights without paying full-fp32 matmul speed.
    torch.set_float32_matmul_precision("high")

    cache_dir = [require_dir(d, "code cache directory") for d in args.cache_dir]
    val_cache_dir = (
        require_dir(args.val_cache_dir, "validation code cache directory")
        if args.val_cache_dir is not None
        else None
    )
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    if args.init_checkpoint is not None:
        args.init_checkpoint = require_file(args.init_checkpoint, "initialization checkpoint")
        digest = sha256_file(args.init_checkpoint)
        if digest != args.init_checkpoint_sha256:
            raise RuntimeError(
                f"Initialization checkpoint SHA-256 mismatch: {digest} != "
                f"{args.init_checkpoint_sha256}"
            )
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")
    if args.input_sample_manifest is not None:
        args.input_sample_manifest = require_file(
            args.input_sample_manifest, "input sample manifest"
        )
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extension_path = out_dir / "post_source_eos_extension.json"
    if extension_path.is_file() and not args.post_source_eos_extension:
        raise RuntimeError("This run requires explicit --post-source-eos-extension")
    clean_incomplete_checkpoints(out_dir)

    retained_text_column = None
    if args.source_asr_pretrain:
        retained_text_column = "text_vi"
    elif args.post_source_eos_translation:
        retained_text_column = "text_en"
    transforms_cached_shape = args.source_asr_pretrain or args.post_source_eos_translation
    dataset = common.CachedCodeDataset(
        cache_dir,
        args.sort_by_length and not transforms_cached_shape,
        args.max_samples,
        0 if transforms_cached_shape else args.max_frames,
        args.cache_weights,
        args.seed,
        expected_target_delay,
        args.input_sample_manifest,
        args.input_sample_manifest_sha256,
        retained_text_column,
    )
    source_asr_replay_dataset = None
    source_asr_replay_sha256 = None
    if args.source_asr_replay_weight:
        source_asr_replay_dataset = common.CachedCodeDataset(
            cache_dir,
            False,
            0,
            args.source_asr_replay_max_frames,
            seed=args.seed,
            expected_target_delay=expected_target_delay,
            sample_manifest=args.input_sample_manifest,
            sample_manifest_sha256=args.input_sample_manifest_sha256,
            retained_text_column="text_vi",
        )
        source_asr_replay_sha256 = common.prepare_source_asr(
            source_asr_replay_dataset,
            args.tokenizer,
            out_dir / "source_asr_replay.json",
            ascii_target=True,
        )
        source_asr_replay_dataset.require_max_frames(
            args.source_asr_replay_max_frames, "Source-ASR replay data"
        )
    source_asr_sha256 = None
    if args.source_asr_pretrain:
        source_asr_sha256 = common.prepare_source_asr(
            dataset,
            args.tokenizer,
            out_dir / "source_asr.json",
            ascii_target=args.source_asr_ascii,
        )
        dataset.require_max_frames(args.max_frames, "Source-ASR training data")
    post_source_eos_translation_sha256 = None
    if args.post_source_eos_translation:
        post_source_eos_translation_sha256 = (
            common.prepare_post_source_eos_translation(
                dataset,
                args.tokenizer,
                out_dir / "post_source_eos_translation.json",
            )
        )
        dataset.require_max_frames(args.max_frames, "Post-source-EOS translation data")
    if transforms_cached_shape and args.sort_by_length:
        dataset.samples.sort(key=lambda sample: sample["frames"])
    if args.sort_by_length:
        dataset.shuffle_batch_order(args.batch_size, args.seed)
    exposure = dataset.exposure()
    print(
        f"Loaded {len(dataset)} cached samples from "
        f"{', '.join(repo_display_path(d) for d in cache_dir)}; "
        f"assembled_frames={exposure['assembled_frames']:,} "
        f"source_hours={exposure['source_hours']:.2f}"
    )
    # Schedules (a static flag becomes a single-point schedule).
    text_points = common.parse_schedule(args.text_weight_schedule or args.text_loss_weight)
    audio_points = common.parse_schedule(args.audio_weight_schedule or args.audio_loss_weight)
    lr_points = common.parse_schedule(args.lr_schedule or args.lr)
    if args.mask_target_audio_input and any(weight != 0.0 for _, weight in audio_points):
        raise ValueError("--mask-target-audio-input requires every audio weight to be zero")
    if args.contrastive_source_weight and (
        not args.mask_target_audio_input or any(weight != 0.0 for _, weight in audio_points)
    ):
        raise ValueError("Contrastive source loss requires masked target audio and zero audio loss")
    if args.source_asr_pretrain and (
        not args.mask_target_audio_input or any(weight != 0.0 for _, weight in audio_points)
    ):
        raise ValueError("Source-ASR preadaptation requires masked target audio and zero audio loss")
    if args.source_asr_replay_weight and (
        not args.mask_target_audio_input or any(weight != 0.0 for _, weight in audio_points)
    ):
        raise ValueError("Source-ASR replay requires masked target audio and zero audio loss")
    if args.post_source_eos_translation and (
        not args.mask_target_audio_input or any(weight != 0.0 for _, weight in audio_points)
    ):
        raise ValueError(
            "Post-source-EOS translation requires masked target audio and zero audio loss"
        )
    source_derangement_sha256 = None
    if args.contrastive_source_weight:
        source_derangement_sha256 = common.attach_source_derangement(
            dataset, out_dir / "source_derangement.json"
        )

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = max(1, batches_per_epoch // args.grad_accum_steps)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
    lr_horizon_steps = total_steps

    val_dataloader = None
    validation_sort_by_length = None
    validation_shuffle = None
    observed_val_max_frames = None
    checkpoint_info = common.load_checkpoint_info(args)
    if val_cache_dir is not None:
        val_sort_by_length = args.sort_by_length or args.input_sample_manifest is not None
        validation_sort_by_length = val_sort_by_length
        validation_shuffle = False if args.input_sample_manifest is not None else None
        val_dataset = common.CachedCodeDataset(
            val_cache_dir,
            val_sort_by_length,
            args.val_max_samples,
            expected_target_delay=expected_target_delay,
            retained_text_column=retained_text_column,
        )
        if args.source_asr_pretrain:
            common.prepare_source_asr(
                val_dataset, args.tokenizer, ascii_target=args.source_asr_ascii
            )
        if args.post_source_eos_translation:
            common.prepare_post_source_eos_translation(val_dataset, args.tokenizer)
        val_dataset.require_max_frames(args.val_max_frames, "Validation cache")
        observed_val_max_frames = order_validation_samples(
            val_dataset, val_sort_by_length, args.smoke_longest_first
        )
        val_dataloader = common.make_cached_dataloader(
            val_dataset,
            args.val_batch_size,
            args.num_workers,
            val_sort_by_length,
            seed=args.seed,
            shuffle=validation_shuffle,
        )
        print(f"Loaded {len(val_dataset)} val cached samples from {repo_display_path(val_cache_dir)}")

    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(
        device=device,
        dtype=dtype,
        lm_kwargs_overrides={"gradient_checkpointing": args.gradient_checkpointing},
    )
    if args.init_checkpoint is not None and args.resume_checkpoint is None:
        common.load_model(lm, args.init_checkpoint, dtype)
        print(f"Initialized full model from {repo_display_path(args.init_checkpoint)}")
    lm.train()
    common.enable_full_finetune(lm)

    params = common.trainable_parameters(lm)
    if not params:
        raise RuntimeError("No trainable parameters after freeze map.")
    print(f"Trainable params: {sum(p.numel() for p in params):,} / {sum(p.numel() for p in lm.parameters()):,}")

    groups = common.param_groups(lm, lr_points)
    optimizer = torch.optim.AdamW(
        groups,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    # fp32 master weights + bf16 autocast forward = standard mixed precision:
    # bf16-speed matmuls, fp32 grads/Adam updates. CE already upcasts logits.
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and dtype == torch.float32
        else contextlib.nullcontext()
    )

    # Optional greedy-eval resources (Mimi is loaded only when selection is on).
    mimi = None
    text_tokenizer = None
    eval_rows: list[dict[str, str]] = []
    eval_cfg = argparse.Namespace(
        source_column=args.eval_source_column,
        duration_column=args.eval_duration_column,
        reference_column=args.eval_reference_column,
        id_column=args.eval_id_column,
        gen_duration=args.eval_gen_duration,
        tail_s=args.eval_tail_s,
        stop_on_eos=True,
        text_only=True,
        tag="val",
    )
    if args.eval_every:
        checkpoint_info.lm_gen_config["temp"] = args.eval_audio_temp
        checkpoint_info.lm_gen_config["temp_text"] = args.eval_text_temp
        checkpoint_info.lm_gen_config["top_k"] = args.eval_top_k
        checkpoint_info.lm_gen_config["top_k_text"] = args.eval_top_k_text
        mimi = checkpoint_info.get_mimi(device=device)
        text_tokenizer = checkpoint_info.get_text_tokenizer()
        eval_ids = common.ids_from_args("", args.eval_ids_file)
        eval_rows = common.select_eval_rows(
            common.read_eval_rows(args.eval_pairs), eval_ids, args.eval_id_column, args.eval_limit
        )
        common.validate_eval_rows(
            eval_rows,
            args.eval_source_column,
            args.eval_reference_column,
            args.eval_id_column,
            args.eval_duration_column,
        )
        if args.source_asr_pretrain and args.source_asr_ascii:
            for row in eval_rows:
                row[args.eval_reference_column] = common.ascii_text(
                    row[args.eval_reference_column]
                )
        print(f"Paired val eval every {args.eval_every} steps on {len(eval_rows)} rows.")

    global_step = 0
    micro_step = 0
    resume_skip_batches = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(lm, optimizer, args.resume_checkpoint, device, dtype)
        micro_step = global_step * args.grad_accum_steps
        if args.sort_by_length or args.input_sample_manifest is not None:
            resume_skip_batches = (global_step * args.grad_accum_steps) % batches_per_epoch
            if resume_skip_batches:
                print(f"Skipping {resume_skip_batches} ordered batches already covered by resume.")
        else:
            print("Resume with shuffled data starts a fresh sampler order.")

    def _jsonable(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return v

    sample_manifest_sha256 = None
    sample_manifest_rows = None
    sample_manifest_cache_counts = None
    if args.persist_sample_manifest:
        sample_manifest_sha256 = freeze_sample_manifest(
            dataset, out_dir, args.resume_checkpoint is not None
        )
        sample_manifest_rows = len(dataset)
        sample_manifest_cache_counts = [
            sum(sample["cache_index"] == cache_index for sample in dataset.samples)
            for cache_index in range(len(cache_dir))
        ]
        if (
            args.input_sample_manifest_sha256 is not None
            and sample_manifest_sha256 != args.input_sample_manifest_sha256
        ):
            raise RuntimeError("Persisted sample manifest differs from authoritative input")
        if args.resume_checkpoint is not None:
            previous_config_path = out_dir / "run_config.json"
            if not previous_config_path.is_file():
                raise RuntimeError("Resume requires the original run_config.json")
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
            ordered_resume_contract = {
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "sort_by_length": args.sort_by_length,
                "smoke_longest_first": args.smoke_longest_first,
                "seed": args.seed,
                "max_frames": args.max_frames,
            }
            for key, value in ordered_resume_contract.items():
                if previous_config.get(key) != value:
                    raise RuntimeError(f"Ordered-data resume contract changed: {key}")
            if previous_config.get("sample_manifest_sha256") != sample_manifest_sha256:
                raise RuntimeError("sample_manifest.jsonl SHA-256 differs from run_config.json")
            if previous_config.get("sample_manifest_rows") != sample_manifest_rows:
                raise RuntimeError("Sample manifest row count differs from run_config.json")
            if (
                previous_config.get("sample_manifest_cache_counts")
                != sample_manifest_cache_counts
            ):
                raise RuntimeError("Sample manifest cache counts differ from run_config.json")
            if previous_config.get("source_derangement_sha256") != source_derangement_sha256:
                raise RuntimeError("Source derangement differs from run_config.json")
            if previous_config.get("source_asr_sha256") != source_asr_sha256:
                raise RuntimeError("Source-ASR policy differs from run_config.json")
            if (
                previous_config.get("post_source_eos_translation_sha256")
                != post_source_eos_translation_sha256
            ):
                raise RuntimeError(
                    "Post-source-EOS translation policy differs from run_config.json"
                )
            if (
                previous_config.get("source_asr_replay_sha256")
                != source_asr_replay_sha256
            ):
                raise RuntimeError("Source-ASR replay policy differs from run_config.json")
            if args.post_source_eos_translation or previous_config.get(
                "post_source_eos_translation", False
            ):
                translation_resume_contract = {
                    "post_source_eos_translation": args.post_source_eos_translation,
                    "init_checkpoint_sha256": args.init_checkpoint_sha256,
                    "mask_target_audio_input": args.mask_target_audio_input,
                    "audio_loss_weight": args.audio_loss_weight,
                    "audio_weight_schedule": args.audio_weight_schedule,
                    "text_loss_weight": args.text_loss_weight,
                    "text_weight_schedule": args.text_weight_schedule,
                    "text_pad_mode": args.text_pad_mode,
                    "text_pad_loss_weight": args.text_pad_loss_weight,
                    "first_content_loss_weight": args.first_content_loss_weight,
                    "lr": args.lr,
                    "lr_schedule": args.lr_schedule,
                    "cosine_lr_end": args.cosine_lr_end,
                    "warmup_steps": args.warmup_steps,
                    "adam_beta1": args.adam_beta1,
                    "adam_beta2": args.adam_beta2,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                }
                if not args.smoke_longest_first and not args.post_source_eos_extension:
                    translation_resume_contract["max_steps"] = args.max_steps
                for key, value in translation_resume_contract.items():
                    if previous_config.get(key) != value:
                        raise RuntimeError(
                            f"Post-source-EOS translation resume contract changed: {key}"
                        )
                if args.post_source_eos_extension:
                    effective_batch = args.batch_size * args.grad_accum_steps
                    if not (
                        previous_config.get("max_steps")
                        == previous_config.get("total_steps")
                        == POST_SOURCE_EOS_EXTENSION_START_STEP
                        and len(dataset) == 50_000
                        and effective_batch == 16
                        and total_steps * effective_batch == len(dataset)
                        and global_step
                        in range(
                            POST_SOURCE_EOS_EXTENSION_START_STEP,
                            POST_SOURCE_EOS_EXTENSION_END_STEP,
                        )
                    ):
                        raise RuntimeError(
                            "Run does not match the exact step-1000 to step-3125 extension"
                        )
                    lr_horizon_steps = POST_SOURCE_EOS_EXTENSION_START_STEP
                    extension_commit = os.environ.get("HIBIKI_EXTENSION_COMMIT", "")
                    if len(extension_commit) != 40 or any(
                        character not in "0123456789abcdef"
                        for character in extension_commit
                    ):
                        raise RuntimeError("Missing exact extension Git commit")
                    extension_receipt = {
                        "version": 1,
                        "strategy": "continue_exact_post_source_eos_run_to_one_pass",
                        "code_commit": extension_commit,
                        "start_step": POST_SOURCE_EOS_EXTENSION_START_STEP,
                        "end_step": POST_SOURCE_EOS_EXTENSION_END_STEP,
                        "lr_horizon_steps": lr_horizon_steps,
                        "end_lr": args.cosine_lr_end,
                        "val_every": args.val_every,
                        "eval_every": args.eval_every,
                        "save_every": args.save_every,
                        "keep_checkpoints": args.keep_checkpoints,
                        "dataset_rows": len(dataset),
                        "effective_batch_size": effective_batch,
                        "sample_manifest_sha256": sample_manifest_sha256,
                        "post_source_eos_translation_sha256": (
                            post_source_eos_translation_sha256
                        ),
                        "init_checkpoint_sha256": args.init_checkpoint_sha256,
                        "original_run_config_sha256": sha256_file(previous_config_path),
                    }
                    freeze_post_source_eos_extension(
                        extension_path, extension_receipt, global_step
                    )
            if args.source_asr_replay_weight or previous_config.get(
                "source_asr_replay_weight", 0.0
            ):
                replay_resume_contract = {
                    "source_asr_replay_weight": args.source_asr_replay_weight,
                    "source_asr_replay_batch_size": args.source_asr_replay_batch_size,
                    "source_asr_replay_max_frames": args.source_asr_replay_max_frames,
                    "source_asr_ascii": args.source_asr_ascii,
                    "init_checkpoint_sha256": args.init_checkpoint_sha256,
                    "mask_target_audio_input": args.mask_target_audio_input,
                    "text_pad_mode": args.text_pad_mode,
                    "text_pad_loss_weight": args.text_pad_loss_weight,
                    "first_content_loss_weight": args.first_content_loss_weight,
                    "torch_compile_enabled": torch_compile_enabled,
                }
                for key, value in replay_resume_contract.items():
                    if previous_config.get(key) != value:
                        raise RuntimeError(f"Source-ASR replay resume contract changed: {key}")
    run_config = {k: _jsonable(v) for k, v in vars(args).items()}
    run_config["total_steps"] = total_steps
    run_config["batches_per_epoch"] = batches_per_epoch
    run_config["steps_per_epoch"] = steps_per_epoch
    run_config["torch_compile_enabled"] = torch_compile_enabled
    run_config["validation_sort_by_length"] = validation_sort_by_length
    run_config["validation_shuffle"] = validation_shuffle
    run_config["observed_val_max_frames"] = observed_val_max_frames
    run_config["observed_train_max_frames"] = max(
        int(sample["frames"]) for sample in dataset.samples
    )
    if sample_manifest_sha256 is not None:
        run_config["sample_manifest_sha256"] = sample_manifest_sha256
        run_config["sample_manifest_rows"] = sample_manifest_rows
        run_config["sample_manifest_cache_counts"] = sample_manifest_cache_counts
    if source_derangement_sha256 is not None:
        run_config["source_derangement_sha256"] = source_derangement_sha256
    if source_asr_sha256 is not None:
        run_config["source_asr_sha256"] = source_asr_sha256
    if post_source_eos_translation_sha256 is not None:
        run_config["post_source_eos_translation_sha256"] = (
            post_source_eos_translation_sha256
        )
    if source_asr_replay_sha256 is not None:
        run_config["source_asr_replay_sha256"] = source_asr_replay_sha256
        run_config["observed_source_asr_replay_max_frames"] = max(
            int(sample["frames"]) for sample in source_asr_replay_dataset.samples
        )
    run_config_path = out_dir / "run_config.json"
    if args.resume_checkpoint is None:
        atomic_write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", run_config_path
        )
    elif not run_config_path.is_file():
        raise RuntimeError("Resume requires the original run_config.json")
    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    greedy_log_path = out_dir / "greedy_eval_log.jsonl"
    condition_cache: dict[int, Any | None] = {}
    log_sums = {
        "loss": 0.0,
        "audio_loss": 0.0,
        "text_loss": 0.0,
        "content_text_loss": 0.0,
        "pad_text_loss": 0.0,
        "first_content_loss": 0.0,
        "contrastive_source_loss": 0.0,
        "source_text_nll_gap": 0.0,
        "source_asr_replay_loss": 0.0,
    }
    log_steps = 0
    log_text_tokens = 0
    log_microbatches = 0
    log_samples = 0
    log_assembled_frames = 0
    log_padded_frames = 0
    log_source_frames = 0
    log_min_batch_size = math.inf
    log_max_batch_size = 0
    log_max_frames = 0
    log_contrastive_active = 0.0
    log_source_asr_replay_tokens = 0
    log_source_asr_replay_samples = 0
    log_source_asr_replay_max_frames = 0
    best_key = (-1.0, -1.0)
    best_state = load_best_state(out_dir)
    if args.resume_checkpoint is not None and best_state is not None:
        best_step = int(best_state["step"])
        if best_step > global_step:
            raise RuntimeError("best.json is newer than the resume checkpoint")
        best_key = (
            float(best_state["correct"]["bleu"]),
            float(best_state["correct"]["chrf"]),
        )
        print(
            f"Restored best paired val BLEU={best_key[0]:.3f} "
            f"chrF={best_key[1]:.3f} from step {best_step}."
        )

    def run_teacher_forced_val(step: int) -> None:
        if val_dataloader is None:
            return
        text_w = common.schedule_value(text_points, step, total_steps)
        audio_w = common.schedule_value(audio_points, step, total_steps)
        metrics = common.evaluate_teacher_forced(
            lm,
            val_dataloader,
            device,
            checkpoint_info.model_type,
            audio_w,
            text_w,
            args.val_batches,
            args.text_pad_mode,
            args.text_pad_loss_weight,
            args.first_content_loss_weight,
            args.mask_target_audio_input,
        )
        item = {"step": step, **{k: metrics[k] for k in (
            "loss", "audio_loss", "text_loss", "audio_tokens", "text_tokens", "samples",
            "content_text_loss", "content_acc", "content_tokens", "pad_text_loss", "pad_acc",
            "pad_tokens", "first_content_loss", "first_content_tokens",
            "first_content_margin", "silence_score",
        )}}
        item.update(common.mps_memory_stats(device))
        with val_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={metrics['loss']:.4f} audio={metrics['audio_loss']:.4f} "
            f"text={metrics['text_loss']:.4f} content={metrics['content_text_loss']:.4f} "
            f"first={metrics['first_content_loss']:.4f} pad={metrics['pad_text_loss']:.4f} "
            f"acc={metrics['content_acc']:.3f} pad_acc={metrics['pad_acc']:.3f} "
            f"margin={metrics['first_content_margin']:.3f} silence={metrics['silence_score']:.3f}"
        )

    def run_greedy_val(step: int, allow_promotion: bool = True) -> None:
        nonlocal best_key
        if not args.eval_every:
            return
        lm.eval()
        eval_out = out_dir / f"greedy_step{step:06d}"
        _, metrics = common.run_paired_eval(
            eval_rows,
            eval_cfg,
            args.eval_batch_size,
            mimi,
            lm,
            text_tokenizer,
            checkpoint_info,
            eval_out,
            resolve_repo_path(args.eval_derangement)
            if args.eval_derangement is not None
            else out_dir / "eval_derangement.json",
            args.seed,
        )
        lm.train()
        correct = metrics["correct"]
        score_key = (float(correct["bleu"]), float(correct["chrf"]))
        promotion_eligible = (
            bool(metrics["health_eligible"])
            and float(metrics["source_bleu_gap"]) >= args.min_source_bleu_gap
            and float(metrics["source_chrf_gap"]) >= args.min_source_chrf_gap
            and (
                args.min_correct_chrf is None
                or float(correct["chrf"]) >= args.min_correct_chrf
            )
            and (
                args.max_correct_wer is None
                or float(correct["wer"]) <= args.max_correct_wer
            )
        )
        item = {
            "step": step,
            **metrics,
            "min_source_bleu_gap": args.min_source_bleu_gap,
            "min_source_chrf_gap": args.min_source_chrf_gap,
            "min_correct_chrf": args.min_correct_chrf,
            "max_correct_wer": args.max_correct_wer,
            "promotion_eligible": promotion_eligible,
        }
        with greedy_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        marker = ""
        if allow_promotion and promotion_eligible and score_key > best_key:
            previous_best = load_best_state(out_dir)
            best_key = score_key
            best_model = out_dir / f"best_step{step:06d}.safetensors"
            common.save_model(lm, best_model, build_metadata(args))
            best_metadata = {**item, "model": best_model.name}
            atomic_write_text(
                json.dumps(best_metadata, indent=2, sort_keys=True) + "\n",
                out_dir / "best.json",
            )
            if previous_best is not None and previous_best["model"] != best_model.name:
                (out_dir / previous_best["model"]).unlink()
            marker = " *best*"
        print(
            f"paired step={step} bleu={score_key[0]:.3f} chrf={score_key[1]:.3f} "
            f"source_gap={metrics['source_bleu_gap']:.3f}/{metrics['source_chrf_gap']:.3f} "
            f"nonempty={correct['nonempty_predictions']}/{correct['num_predictions']}{marker}"
        )
        common.empty_device_cache(device)

    if args.eval_at_start and global_step == 0:
        run_greedy_val(0, allow_promotion=False)

    sample_order = None
    if args.smoke_longest_first:
        sample_order = sorted(
            range(len(dataset)), key=lambda index: dataset.samples[index]["frames"], reverse=True
        )
    dataloader = common.make_cached_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        args.sort_by_length,
        seed=args.seed,
        shuffle=False if args.input_sample_manifest is not None else None,
        sample_order=sample_order,
    )
    data_iter = iter(dataloader)
    while resume_skip_batches > 0:
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)
        resume_skip_batches -= 1
    source_asr_replay_iter = None
    if source_asr_replay_dataset is not None:
        replay_order = None
        if args.smoke_longest_first:
            replay_order = sorted(
                range(len(source_asr_replay_dataset)),
                key=lambda index: source_asr_replay_dataset.samples[index]["frames"],
                reverse=True,
            )
        source_asr_replay_dataloader = common.make_cached_dataloader(
            source_asr_replay_dataset,
            args.source_asr_replay_batch_size,
            args.num_workers,
            False,
            seed=args.seed,
            shuffle=False,
            sample_order=replay_order,
        )
        source_asr_replay_iter = iter(source_asr_replay_dataloader)
        replay_skip_batches = global_step % math.ceil(
            len(source_asr_replay_dataset) / args.source_asr_replay_batch_size
        )
        while replay_skip_batches > 0:
            try:
                next(source_asr_replay_iter)
            except StopIteration:
                source_asr_replay_iter = iter(source_asr_replay_dataloader)
                next(source_asr_replay_iter)
            replay_skip_batches -= 1
    last_log_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        text_w = common.schedule_value(text_points, global_step, total_steps)
        audio_w = common.schedule_value(audio_points, global_step, total_steps)
        if args.cosine_lr_end:
            lr_value = common.apply_cosine_lr_schedule(
                optimizer,
                global_step,
                lr_horizon_steps,
                args.warmup_steps,
                args.cosine_lr_end,
            )
        else:
            lr_value = common.apply_lr_schedule(
                optimizer, global_step, total_steps, args.warmup_steps
            )

        optimizer.zero_grad(set_to_none=True)
        step_loss = step_audio = step_text = 0.0
        step_content = step_pad = step_first = 0.0
        step_contrastive = step_source_gap = step_contrastive_active = 0.0
        step_source_asr_replay = 0.0
        step_source_asr_replay_tokens = 0
        step_text_tokens = 0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            actual_max_frames = int(batch["frames"].max())
            log_microbatches += 1
            log_samples += batch_size
            log_assembled_frames += int(batch["frames"].sum())
            log_padded_frames += batch_size * int(codes.shape[-1])
            log_source_frames += int(batch["source_frames"].sum())
            log_min_batch_size = min(log_min_batch_size, batch_size)
            log_max_batch_size = max(log_max_batch_size, batch_size)
            log_max_frames = max(log_max_frames, actual_max_frames)
            if batch_size not in condition_cache:
                condition_cache[batch_size] = common.batch_condition_tensors(
                    lm, checkpoint_info.model_type, batch_size
                )
            with autocast:
                losses = common.compute_batch_losses(
                    lm,
                    codes,
                    condition_cache[batch_size],
                    audio_w,
                    text_w,
                    text_pad_loss_weight=args.text_pad_loss_weight,
                    first_content_loss_weight=args.first_content_loss_weight,
                    text_pad_mode=args.text_pad_mode,
                    mask_target_audio_input=args.mask_target_audio_input,
                    return_per_sample_content_loss=bool(args.contrastive_source_weight),
                )
            loss = losses["loss"]
            (loss / args.grad_accum_steps).backward()
            contrastive_loss = loss.detach().new_zeros(())
            source_text_nll_gap = loss.detach().new_zeros(())
            contrastive_active = loss.detach().new_zeros(())
            if args.contrastive_source_weight:
                if "shuffled_codes" not in batch:
                    raise RuntimeError("Contrastive source batch is missing donor codes")
                correct_content_losses = losses.pop("per_sample_content_text_loss").detach()
                shuffled_codes = batch["shuffled_codes"].to(device=device, dtype=torch.long)
                with autocast:
                    shuffled_losses = common.compute_batch_losses(
                        lm,
                        shuffled_codes,
                        condition_cache[batch_size],
                        0.0,
                        1.0,
                        text_pad_loss_weight=args.text_pad_loss_weight,
                        first_content_loss_weight=args.first_content_loss_weight,
                        text_pad_mode=args.text_pad_mode,
                        mask_target_audio_input=True,
                        return_per_sample_content_loss=True,
                    )
                shuffled_content_losses = shuffled_losses["per_sample_content_text_loss"]
                hinge = torch.relu(
                    args.contrastive_source_margin
                    + correct_content_losses
                    - shuffled_content_losses
                )
                contrastive_loss = hinge.mean()
                (
                    args.contrastive_source_weight
                    * contrastive_loss
                    / args.grad_accum_steps
                ).backward()
                source_text_nll_gap = (
                    shuffled_content_losses.detach() - correct_content_losses
                ).mean()
                contrastive_active = (hinge.detach() > 0).sum()
            micro_step += 1
            # Accumulate on-device; host sync (and the non-finite check) happens
            # only at --log-every boundaries. Per-micro-step .cpu() reads stalled
            # the CUDA pipeline every step.
            step_loss = (
                step_loss
                + loss.detach()
                + args.contrastive_source_weight * contrastive_loss.detach()
            )
            step_audio = step_audio + losses["audio_loss"].detach()
            step_text = step_text + losses["text_loss"].detach()
            step_content = step_content + losses["content_text_loss"].detach()
            step_pad = step_pad + losses["pad_text_loss"].detach()
            step_first = step_first + losses["first_content_loss"].detach()
            step_contrastive = step_contrastive + contrastive_loss.detach()
            step_source_gap = step_source_gap + source_text_nll_gap.detach()
            step_contrastive_active = step_contrastive_active + contrastive_active
            step_text_tokens = step_text_tokens + losses["text_tokens"]

        if source_asr_replay_iter is not None:
            try:
                replay_batch = next(source_asr_replay_iter)
            except StopIteration:
                source_asr_replay_iter = iter(source_asr_replay_dataloader)
                replay_batch = next(source_asr_replay_iter)
            replay_codes = replay_batch["codes"].to(device=device, dtype=torch.long)
            replay_batch_size = int(replay_codes.shape[0])
            if replay_batch_size not in condition_cache:
                condition_cache[replay_batch_size] = common.batch_condition_tensors(
                    lm, checkpoint_info.model_type, replay_batch_size
                )
            with autocast:
                replay_losses = common.compute_batch_losses(
                    lm,
                    replay_codes,
                    condition_cache[replay_batch_size],
                    0.0,
                    1.0,
                    text_pad_loss_weight=0.0,
                    first_content_loss_weight=args.first_content_loss_weight,
                    text_pad_mode=args.text_pad_mode,
                    mask_target_audio_input=True,
                )
            replay_loss = replay_losses["loss"]
            (args.source_asr_replay_weight * replay_loss).backward()
            step_source_asr_replay = replay_loss.detach()
            step_source_asr_replay_tokens = replay_losses["text_tokens"]
            log_source_asr_replay_samples += replay_batch_size
            log_source_asr_replay_max_frames = max(
                log_source_asr_replay_max_frames, int(replay_batch["frames"].max())
            )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        global_step += 1

        log_sums["loss"] += (
            step_loss / args.grad_accum_steps
            + args.source_asr_replay_weight * step_source_asr_replay
        )
        log_sums["audio_loss"] += step_audio / args.grad_accum_steps
        log_sums["text_loss"] += step_text / args.grad_accum_steps
        log_sums["content_text_loss"] += step_content / args.grad_accum_steps
        log_sums["pad_text_loss"] += step_pad / args.grad_accum_steps
        log_sums["first_content_loss"] += step_first / args.grad_accum_steps
        log_sums["contrastive_source_loss"] += step_contrastive / args.grad_accum_steps
        log_sums["source_text_nll_gap"] += step_source_gap / args.grad_accum_steps
        log_sums["source_asr_replay_loss"] += step_source_asr_replay
        log_contrastive_active += step_contrastive_active
        log_steps += 1
        log_text_tokens += step_text_tokens
        log_source_asr_replay_tokens += step_source_asr_replay_tokens

        if args.log_every and global_step % args.log_every == 0:
            loss_avg = float(log_sums["loss"]) / log_steps
            if not math.isfinite(loss_avg):
                raise RuntimeError(
                    f"Non-finite loss by step {global_step}. "
                    "Lower --lr or inspect the last finite checkpoint."
                )
            now = time.time()
            item = {
                "epoch": (global_step - 1) // steps_per_epoch + 1,
                "step": global_step,
                "loss": loss_avg,
                "audio_loss": float(log_sums["audio_loss"]) / log_steps,
                "text_loss": float(log_sums["text_loss"]) / log_steps,
                "content_text_loss": float(log_sums["content_text_loss"]) / log_steps,
                "pad_text_loss": float(log_sums["pad_text_loss"]) / log_steps,
                "first_content_loss": float(log_sums["first_content_loss"]) / log_steps,
                "contrastive_source_loss": float(log_sums["contrastive_source_loss"])
                / log_steps,
                "source_text_nll_gap": float(log_sums["source_text_nll_gap"]) / log_steps,
                "source_asr_replay_loss": float(log_sums["source_asr_replay_loss"])
                / log_steps,
                "source_asr_replay_weight": args.source_asr_replay_weight,
                "source_asr_replay_tokens": int(log_source_asr_replay_tokens),
                "source_asr_replay_samples": log_source_asr_replay_samples,
                "source_asr_replay_max_frames": log_source_asr_replay_max_frames,
                "contrastive_active_fraction": (
                    float(log_contrastive_active) / log_samples if log_samples else 0.0
                ),
                "text_tokens": int(log_text_tokens),
                "microbatches": log_microbatches,
                "samples": log_samples,
                "assembled_frames": log_assembled_frames,
                "padded_frames": log_padded_frames,
                "source_hours": log_source_frames / float(dataset.frame_rate) / 3600,
                "min_batch_size": int(log_min_batch_size),
                "max_batch_size": log_max_batch_size,
                "max_frames": log_max_frames,
                "sec_per_step": (now - last_log_time) / log_steps,
                "log_steps": log_steps,
                "lr": lr_value,
                "text_weight": text_w,
                "audio_weight": audio_w,
                "pad_loss_weight": args.text_pad_loss_weight,
                "first_content_loss_weight": args.first_content_loss_weight,
                "contrastive_source_weight": args.contrastive_source_weight,
                "contrastive_source_margin": args.contrastive_source_margin,
            }
            item.update(common.mps_memory_stats(device))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
            memory_msg = ""
            if "mps_driver_gb" in item:
                memory_msg = f" mps={item['mps_allocated_gb']:.1f}/{item['mps_driver_gb']:.1f}GB"
            print(
                f"step={global_step} loss={item['loss']:.4f} audio={item['audio_loss']:.4f} "
                f"text={item['text_loss']:.4f} content={item['content_text_loss']:.4f} "
                f"first={item['first_content_loss']:.4f} pad={item['pad_text_loss']:.4f} tw={text_w:g} "
                f"contrast={item['contrastive_source_loss']:.4f} "
                f"source_gap={item['source_text_nll_gap']:.4f} "
                f"asr_replay={item['source_asr_replay_loss']:.4f} "
                f"active={item['contrastive_active_fraction']:.3f} "
                f"B={item['samples'] / item['microbatches']:.1f} "
                f"[{item['min_batch_size']}-{item['max_batch_size']}] T<={item['max_frames']} "
                f"s/step={item['sec_per_step']:.3f}{memory_msg}"
            )
            log_sums = {
                "loss": 0.0,
                "audio_loss": 0.0,
                "text_loss": 0.0,
                "content_text_loss": 0.0,
                "pad_text_loss": 0.0,
                "first_content_loss": 0.0,
                "contrastive_source_loss": 0.0,
                "source_text_nll_gap": 0.0,
                "source_asr_replay_loss": 0.0,
            }
            log_steps = 0
            log_text_tokens = 0
            log_microbatches = 0
            log_samples = 0
            log_assembled_frames = 0
            log_padded_frames = 0
            log_source_frames = 0
            log_min_batch_size = math.inf
            log_max_batch_size = 0
            log_max_frames = 0
            log_contrastive_active = 0.0
            log_source_asr_replay_tokens = 0
            log_source_asr_replay_samples = 0
            log_source_asr_replay_max_frames = 0
            last_log_time = now

        if args.save_every and global_step % args.save_every == 0:
            save_checkpoint(lm, optimizer, args, global_step, out_dir)
        if val_dataloader is not None and args.val_every and global_step % args.val_every == 0:
            run_teacher_forced_val(global_step)
        if args.eval_every and global_step % args.eval_every == 0:
            run_greedy_val(global_step)

    if global_step == 0:
        raise RuntimeError("Training ended before any optimizer step. Check cache size and grad accum.")
    if val_dataloader is not None and (not args.val_every or global_step % args.val_every != 0):
        run_teacher_forced_val(global_step)
    if args.eval_every and global_step % args.eval_every != 0:
        run_greedy_val(global_step)
    save_checkpoint(lm, optimizer, args, global_step, out_dir)
    if best_key[0] >= 0.0:
        final_best = load_best_state(out_dir)
        assert final_best is not None
        best_path = out_dir / final_best["model"]
        print(
            f"Best paired val BLEU={best_key[0]:.3f} chrF={best_key[1]:.3f} "
            f"-> {repo_display_path(best_path)}"
        )
    print(f"Saved final full-model checkpoint at step {global_step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()
