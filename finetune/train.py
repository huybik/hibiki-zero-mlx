#!/usr/bin/env python
"""Full-model Hibiki-Zero SFT trainer for cached Vietnamese-to-English codes."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
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
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda", help="CUDA device for full-model training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--max-samples", type=int, default=0, help="First N cached samples; 0=all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Drop cached samples longer than N Mimi frames; 0=off.")
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
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-weight-schedule", default="", help='Text loss-weight schedule "5@0,2@0.6".')
    parser.add_argument(
        "--text-prefix-pad-weight",
        type=float,
        default=0.5,
        help="Per-token CE weight for supervised prefix PAD; content/EOS stay at 1.0.",
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
        "--val-batch-size",
        type=int,
        help="Teacher-forced validation batch size; defaults to --batch-size.",
    )
    # Autoregressive greedy val eval + best-checkpoint selection.
    parser.add_argument("--eval-every", type=int, default=0, help="Steps between greedy val eval; 0=off.")
    parser.add_argument("--eval-pairs", type=Path, default=DEFAULT_PAIRS_DIR / "validation.jsonl")
    parser.add_argument("--eval-limit", type=int, default=128, help="Greedy val rows (val128 gate).")
    parser.add_argument("--eval-ids-file", type=Path, help="Optional id file for the greedy eval set.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-text-temp", type=float, default=0.0, help="Greedy text temp for selection.")
    parser.add_argument("--eval-audio-temp", type=float, default=0.8)
    parser.add_argument("--eval-top-k", type=int, default=250)
    parser.add_argument("--eval-top-k-text", type=int, default=250)
    parser.add_argument("--eval-tail-s", type=float, default=8.0)
    parser.add_argument("--eval-gen-duration", type=float, default=0.0)
    parser.add_argument("--eval-source-column", default="vi_audio")
    parser.add_argument("--eval-reference-column", default="text_en")
    parser.add_argument("--eval-id-column", default="id")
    parser.add_argument(
        "--eval-shuffled-source",
        action="store_true",
        help="Also evaluate a cyclic source permutation to measure source dependence.",
    )
    parser.add_argument(
        "--best-requires-gates",
        action="store_true",
        help="Save best only after nonempty/EOS/loop/length eligibility gates pass.",
    )
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
    return {
        "target": "full",
        "base_model": repo_display_path(args.model_weight),
    }


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
    chrf = float(state["chrf"])
    model_name = str(state["model"])
    if model_name != f"best_step{step:06d}.safetensors":
        raise RuntimeError("best.json model does not match its step")
    model = out_dir / model_name
    if not model.is_file():
        raise RuntimeError(f"best.json references a missing model: {model}")
    return {"step": step, "chrf": chrf, "model": model_name}


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


def main() -> None:
    args = parse_args()
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
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.text_prefix_pad_weight < 0:
        raise ValueError("--text-prefix-pad-weight must be non-negative")
    if args.cache_weights is not None and len(args.cache_weights) != len(args.cache_dir):
        raise ValueError("--cache-weights must match --cache-dir")
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
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_incomplete_checkpoints(out_dir)

    dataset = common.CachedCodeDataset(
        cache_dir,
        args.sort_by_length,
        args.max_samples,
        args.max_frames,
        args.cache_weights,
        args.seed,
    )
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

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    steps_per_epoch = max(1, batches_per_epoch // args.grad_accum_steps)
    total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch

    val_dataloader = None
    checkpoint_info = common.load_checkpoint_info(args)
    if val_cache_dir is not None:
        val_dataset = common.CachedCodeDataset(val_cache_dir, args.sort_by_length, args.val_max_samples)
        val_dataloader = common.make_cached_dataloader(
            val_dataset, args.val_batch_size, args.num_workers, args.sort_by_length, seed=args.seed
        )
        print(f"Loaded {len(val_dataset)} val cached samples from {repo_display_path(val_cache_dir)}")

    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(
        device=device,
        dtype=dtype,
        lm_kwargs_overrides={"gradient_checkpointing": args.gradient_checkpointing},
    )
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
            eval_rows, args.eval_source_column, args.eval_reference_column, args.eval_id_column
        )
        print(f"Greedy val eval every {args.eval_every} steps on {len(eval_rows)} rows.")

    global_step = 0
    micro_step = 0
    resume_skip_batches = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(lm, optimizer, args.resume_checkpoint, device, dtype)
        micro_step = global_step * args.grad_accum_steps
        if args.sort_by_length:
            resume_skip_batches = (global_step * args.grad_accum_steps) % batches_per_epoch
            if resume_skip_batches:
                print(f"Skipping {resume_skip_batches} sorted batches already covered by resume.")
        else:
            print("Resume with shuffled data starts a fresh sampler order.")

    def _jsonable(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [_jsonable(x) for x in v]
        return v

    run_config = {k: _jsonable(v) for k, v in vars(args).items()}
    run_config["total_steps"] = total_steps
    run_config["batches_per_epoch"] = batches_per_epoch
    run_config["steps_per_epoch"] = steps_per_epoch
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), "utf-8")
    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    greedy_log_path = out_dir / "greedy_eval_log.jsonl"
    condition_cache: dict[int, Any | None] = {}
    log_sums = {"loss": 0.0, "audio_loss": 0.0, "text_loss": 0.0}
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
    best_chrf = -1.0
    best_state = load_best_state(out_dir)
    if args.resume_checkpoint is not None and best_state is not None:
        best_step = int(best_state["step"])
        if best_step > global_step:
            raise RuntimeError("best.json is newer than the resume checkpoint")
        best_chrf = float(best_state["chrf"])
        print(f"Restored best greedy val chrF={best_chrf:.3f} from step {best_step}.")

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
            args.text_prefix_pad_weight,
        )
        item = {"step": step, **{k: metrics[k] for k in (
            "loss", "audio_loss", "text_loss", "audio_tokens", "text_tokens", "samples",
            "content_text_loss", "content_acc", "content_tokens", "silence_score",
        )}}
        item.update(common.mps_memory_stats(device))
        with val_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={metrics['loss']:.4f} audio={metrics['audio_loss']:.4f} "
            f"text={metrics['text_loss']:.4f} content={metrics['content_text_loss']:.4f} "
            f"acc={metrics['content_acc']:.3f} silence={metrics['silence_score']:.3f}"
        )

    def run_greedy_val(step: int) -> None:
        nonlocal best_chrf
        if not args.eval_every:
            return
        lm.eval()
        eval_out = out_dir / f"greedy_step{step:06d}"
        records, metrics = common.run_greedy_eval(
            eval_rows, eval_cfg, args.eval_batch_size, mimi, lm, text_tokenizer, checkpoint_info, eval_out
        )
        chrf = float(metrics.get("chrf", 0.0))
        eligible = (
            metrics["nonempty_predictions"] >= 122
            and metrics["eos_found"] >= 116
            and metrics["repeated_4gram_predictions"] <= 12
            and metrics["mean_length_ratio"] <= 2.0
        )
        item = {
            "step": step,
            "chrf": chrf,
            "nonempty_chrf": metrics.get("nonempty_chrf"),
            "bleu": metrics.get("bleu"),
            "nonempty": metrics["nonempty_predictions"],
            "num": metrics["num_predictions"],
            "eos": metrics["eos_found"],
            "overlong": metrics["overlong_predictions"],
            "repeat4": metrics["repeated_4gram_predictions"],
            "mean_length_ratio": metrics["mean_length_ratio"],
            "eligible": eligible,
        }
        if args.eval_shuffled_source:
            shuffled_rows = [dict(row) for row in eval_rows]
            shuffled_sources = [row[args.eval_source_column] for row in eval_rows[1:] + eval_rows[:1]]
            for row, source in zip(shuffled_rows, shuffled_sources, strict=True):
                row[args.eval_source_column] = source
            _, shuffled = common.run_greedy_eval(
                shuffled_rows,
                eval_cfg,
                args.eval_batch_size,
                mimi,
                lm,
                text_tokenizer,
                checkpoint_info,
                out_dir / f"greedy_step{step:06d}_source_shuffled",
            )
            item["shuffled_chrf"] = shuffled["chrf"]
            item["source_chrf_gap"] = chrf - float(shuffled["chrf"])
            item["shuffled_nonempty"] = shuffled["nonempty_predictions"]
        lm.train()
        with greedy_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        marker = ""
        if chrf > best_chrf and (eligible or not args.best_requires_gates):
            previous_best = load_best_state(out_dir)
            best_chrf = chrf
            best_model = out_dir / f"best_step{step:06d}.safetensors"
            common.save_model(lm, best_model, build_metadata(args))
            atomic_write_text(
                json.dumps(
                    {"step": step, "chrf": chrf, "model": best_model.name}, sort_keys=True
                )
                + "\n",
                out_dir / "best.json",
            )
            if previous_best is not None and previous_best["model"] != best_model.name:
                (out_dir / previous_best["model"]).unlink()
            marker = " *best*"
        print(
            f"greedy step={step} chrf={chrf:.3f} nonempty_chrf={item['nonempty_chrf']:.3f} "
            f"nonempty={item['nonempty']}/{item['num']}{marker}"
        )
        common.empty_device_cache(device)

    dataloader = common.make_cached_dataloader(
        dataset,
        args.batch_size,
        args.num_workers,
        args.sort_by_length,
        seed=args.seed,
    )
    data_iter = iter(dataloader)
    while resume_skip_batches > 0:
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)
        resume_skip_batches -= 1
    last_log_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        text_w = common.schedule_value(text_points, global_step, total_steps)
        audio_w = common.schedule_value(audio_points, global_step, total_steps)
        if args.cosine_lr_end:
            lr_value = common.apply_cosine_lr_schedule(
                optimizer,
                global_step,
                total_steps,
                args.warmup_steps,
                args.cosine_lr_end,
            )
        else:
            lr_value = common.apply_lr_schedule(
                optimizer, global_step, total_steps, args.warmup_steps
            )

        optimizer.zero_grad(set_to_none=True)
        step_loss = step_audio = step_text = 0.0
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
                    text_prefix_pad_weight=args.text_prefix_pad_weight,
                    text_pad_mode=args.text_pad_mode,
                )
            loss = losses["loss"]
            (loss / args.grad_accum_steps).backward()
            micro_step += 1
            # Accumulate on-device; host sync (and the non-finite check) happens
            # only at --log-every boundaries. Per-micro-step .cpu() reads stalled
            # the CUDA pipeline every step.
            step_loss = step_loss + loss.detach()
            step_audio = step_audio + losses["audio_loss"].detach()
            step_text = step_text + losses["text_loss"].detach()
            step_text_tokens = step_text_tokens + losses["text_tokens"]

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        global_step += 1

        log_sums["loss"] += step_loss / args.grad_accum_steps
        log_sums["audio_loss"] += step_audio / args.grad_accum_steps
        log_sums["text_loss"] += step_text / args.grad_accum_steps
        log_steps += 1
        log_text_tokens += step_text_tokens

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
            }
            item.update(common.mps_memory_stats(device))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
            memory_msg = ""
            if "mps_driver_gb" in item:
                memory_msg = f" mps={item['mps_allocated_gb']:.1f}/{item['mps_driver_gb']:.1f}GB"
            print(
                f"step={global_step} loss={item['loss']:.4f} audio={item['audio_loss']:.4f} "
                f"text={item['text_loss']:.4f} tw={text_w:g} "
                f"B={item['samples'] / item['microbatches']:.1f} "
                f"[{item['min_batch_size']}-{item['max_batch_size']}] T<={item['max_frames']} "
                f"s/step={item['sec_per_step']:.3f}{memory_msg}"
            )
            log_sums = {"loss": 0.0, "audio_loss": 0.0, "text_loss": 0.0}
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
    if best_chrf >= 0.0:
        final_best = load_best_state(out_dir)
        assert final_best is not None
        best_path = out_dir / final_best["model"]
        print(f"Best greedy val chrF={best_chrf:.3f} -> {repo_display_path(best_path)}")
    print(f"Saved final full-model checkpoint at step {global_step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()
