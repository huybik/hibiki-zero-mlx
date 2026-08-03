#!/usr/bin/env python
"""Main-transformer LoRA trainer for cached vi->en codes.

Supports piecewise-constant schedules for loss weights, replay weight, and
per-group learning rate, plus periodic greedy val eval with best-on-chrF
checkpoint selection. All schedules degrade to the old static flags when their
`--*-schedule` variant is unset, so existing commands run unchanged.
"""
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
        description="Minimal Hibiki-Zero main-transformer LoRA trainer for cached vi->en codes."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_CACHE_ROOT / "train"],
        help="One or more cache dirs; shards from all are pooled (e.g. FLEURS + PhoMT).",
    )
    parser.add_argument("--val-cache-dir", type=Path, help="Cached val split for teacher-forced CE.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="mps", help="Torch device. No automatic fallback.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model/LoRA dtype. bfloat16 is the stable smoke-test default on MPS.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size.")
    parser.add_argument("--max-samples", type=int, default=0, help="First N cached samples; 0=all.")
    parser.add_argument("--max-frames", type=int, default=0, help="Drop cached samples longer than N Mimi frames; 0=off.")
    parser.add_argument(
        "--frame-batch-schedule",
        default="",
        help='Cumulative length buckets "MAX_FRAMES:BATCH_SIZE,...", e.g. "288:10,384:8,512:5". '
        "Requires matching --max-frames and replaces --batch-size; sizes must be benchmarked.",
    )
    parser.add_argument("--replay-ids", default="", help="Comma-separated ids to upweight.")
    parser.add_argument(
        "--replay-weight", type=float, default=1.0, help="Static replay weight; 1 disables replay."
    )
    parser.add_argument(
        "--replay-weight-schedule",
        default="",
        help='Piecewise "weight@fraction" replay schedule, e.g. "300@0,100@0.5". '
        "Overrides --replay-weight when set. Requires --replay-ids.",
    )
    parser.add_argument("--replay-seed", type=int, default=0, help="Deterministic replay seed.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", default="", help='Transformer LoRA LR schedule "lr@frac,...".')
    parser.add_argument("--text-head-lr", type=float, default=0.0, help="Static text_linear LR; 0=lr.")
    parser.add_argument("--text-head-lr-schedule", default="", help="text_linear LR schedule.")
    parser.add_argument("--audio-head-lr", type=float, default=0.0, help="Static audio-head LR; 0=lr.")
    parser.add_argument("--audio-head-lr-schedule", default="", help="audio-head LoRA LR schedule.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps; 0=off.")
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full-model SFT (paper §4.6): train every LM param, no LoRA. The scaled CUDA run; "
        "uses --lr/--lr-schedule for all params (per-group head LR flags ignored).",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-scaling", type=float, default=2.0)
    parser.add_argument("--train-text-head", action="store_true", help="Also train LMModel.text_linear.")
    parser.add_argument(
        "--train-audio-heads", action="store_true", help="Also LoRA depformer_in + audio linears."
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--audio-weight-schedule", default="", help="Audio loss-weight schedule.")
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-weight-schedule", default="", help='Text loss-weight schedule "5@0,2@0.6".')
    parser.add_argument(
        "--text-prefix-pad-weight",
        type=float,
        default=1.0,
        help="Per-token CE weight for supervised prefix PAD; content/EOS stay at 1.0.",
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
    # Bookkeeping.
    parser.add_argument("--save-every", type=int, default=50, help="Steps between saves.")
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Keep only the newest N step checkpoints (best/final untouched); 0=keep all. "
        "Full-finetune saves are ~35 GB each — rotation is essential on rented disks.",
    )
    parser.add_argument("--log-every", type=int, default=1, help="Steps between logs.")
    parser.add_argument(
        "--mps-empty-cache-every",
        type=int,
        default=10,
        help="Steps between MPS synchronize+empty_cache calls (no-op off MPS), 0 disables.",
    )
    parser.add_argument("--resume-checkpoint", type=Path, help="trainer_step*.pt to resume from.")
    parser.add_argument("--init-adapter", type=Path, help="Load adapter weights without optimizer resume.")
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
    target = "full" if args.full_finetune else common.adapter_target(
        args.train_text_head, args.train_audio_heads
    )
    return {
        "lora_rank": str(args.lora_rank),
        "lora_scaling": str(args.lora_scaling),
        "target": target,
        "train_text_head": str(args.train_text_head),
        "train_audio_heads": str(args.train_audio_heads),
        "base_model": repo_display_path(args.model_weight),
    }


def checkpoint_prefix(args: argparse.Namespace) -> str:
    return "model" if args.full_finetune else "adapter"


def save_checkpoint(
    model: Any, optimizer: Any, args: argparse.Namespace, step: int, out_dir: Path
) -> Path:
    adapter_path = out_dir / f"{checkpoint_prefix(args)}_step{step:06d}.safetensors"
    common.save_adapter(model, adapter_path, build_metadata(args))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "adapter": str(adapter_path),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        out_dir / f"trainer_step{step:06d}.pt",
    )
    if args.keep_checkpoints > 0:
        for pattern in (f"{checkpoint_prefix(args)}_step*.safetensors", "trainer_step*.pt"):
            for stale in sorted(out_dir.glob(pattern))[: -args.keep_checkpoints]:
                stale.unlink()
    return adapter_path


def load_resume_checkpoint(
    model: Any, optimizer: Any, resume_path: Path, device: torch.device, dtype: torch.dtype
) -> int:
    # weights_only=False: our own trusted checkpoint; its `args` dict holds Path
    # objects that PyTorch 2.6's default weights_only=True refuses to unpickle.
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    adapter_path = require_file(checkpoint["adapter"], "resume adapter")
    common.load_adapter_state(model, adapter_path, dtype)
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


def lr_schedule_specs(args: argparse.Namespace) -> dict[str, list[tuple[float, float]]]:
    base = args.lr_schedule or str(args.lr)
    text = args.text_head_lr_schedule or (str(args.text_head_lr) if args.text_head_lr else base)
    audio = args.audio_head_lr_schedule or (str(args.audio_head_lr) if args.audio_head_lr else base)
    return {
        "transformer": common.parse_schedule(base),
        "text_linear": common.parse_schedule(text),
        "audio_heads": common.parse_schedule(audio),
    }


def main() -> None:
    args = parse_args()
    frame_batch_schedule = (
        common.parse_frame_batch_schedule(args.frame_batch_schedule)
        if args.frame_batch_schedule
        else []
    )
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
    if args.replay_weight <= 0:
        raise ValueError("--replay-weight must be positive")
    if args.text_head_lr < 0 or args.audio_head_lr < 0:
        raise ValueError("Custom LR values must be non-negative")
    if args.text_prefix_pad_weight < 0:
        raise ValueError("--text-prefix-pad-weight must be non-negative")
    if args.resume_checkpoint is not None and args.init_adapter is not None:
        raise ValueError("--resume-checkpoint and --init-adapter are mutually exclusive")
    if frame_batch_schedule:
        if args.max_frames <= 0 or frame_batch_schedule[-1][0] != args.max_frames:
            raise ValueError("--frame-batch-schedule final limit must equal positive --max-frames")
        if args.batch_size != 1:
            raise ValueError("Omit --batch-size when --frame-batch-schedule is set")
        if not args.sort_by_length:
            raise ValueError("--frame-batch-schedule requires --sort-by-length")
        if args.replay_ids.strip() or args.replay_weight_schedule or args.replay_weight != 1.0:
            raise ValueError("--frame-batch-schedule does not support replay sampling")

    common.seed_all(args.seed)

    device = common.check_device(args.device)
    dtype = common.dtype_from_name(args.dtype)
    if device.type == "cuda":
        # TF32 matmuls: fp32 master weights (needed for full-finetune Adam updates —
        # bf16 weights round away lr~1e-5 updates) without paying full-fp32 speed.
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
    if args.init_adapter is not None:
        args.init_adapter = require_file(args.init_adapter, "initial adapter")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = common.CachedCodeDataset(cache_dir, args.sort_by_length, args.max_samples, args.max_frames)
    frame_batch_sampler = None
    if frame_batch_schedule:
        frame_batch_sampler = common.FrameBudgetBatchSampler(dataset, frame_batch_schedule, args.seed)
    elif args.sort_by_length:
        dataset.shuffle_batch_order(args.batch_size, args.seed)
    exposure = dataset.exposure()
    print(
        f"Loaded {len(dataset)} cached samples from "
        f"{', '.join(repo_display_path(d) for d in cache_dir)}; "
        f"assembled_frames={exposure['assembled_frames']:,} "
        f"source_hours={exposure['source_hours']:.2f}"
    )
    if frame_batch_sampler is not None:
        for bucket in frame_batch_sampler.bucket_exposure:
            print(
                f"[frame batch] {bucket['min_frames']}-{bucket['max_frames']} frames: "
                f"batch={bucket['batch_size']} samples={bucket['samples']:,} "
                f"batches={bucket['batches']:,} assembled_frames={bucket['assembled_frames']:,} "
                f"source_hours={bucket['source_hours']:.2f}"
            )
    replay_ids = common.parse_ids(args.replay_ids)

    # Schedules (static flag becomes a single-point schedule).
    text_points = common.parse_schedule(args.text_weight_schedule or args.text_loss_weight)
    audio_points = common.parse_schedule(args.audio_weight_schedule or args.audio_loss_weight)
    replay_points = common.parse_schedule(args.replay_weight_schedule or args.replay_weight)
    lr_specs = lr_schedule_specs(args)
    if args.replay_weight_schedule and not replay_ids:
        raise ValueError("--replay-weight-schedule requires --replay-ids")

    batches_per_epoch = (
        len(frame_batch_sampler)
        if frame_batch_sampler is not None
        else math.ceil(len(dataset) / args.batch_size)
    )
    if frame_batch_sampler is not None and batches_per_epoch % args.grad_accum_steps:
        raise ValueError(
            f"--grad-accum-steps {args.grad_accum_steps} must divide the "
            f"{batches_per_epoch} frame-budget batches per epoch"
        )
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
    if args.full_finetune:
        common.apply_full_finetune(lm)
    else:
        common.apply_lora_targets(
            lm, args.lora_rank, args.lora_scaling, args.train_text_head, args.train_audio_heads
        )
    if args.init_adapter is not None:
        common.load_adapter_state(lm, args.init_adapter, dtype)
        print(f"Loaded init adapter {repo_display_path(args.init_adapter)}")

    params = common.trainable_parameters(lm)
    if not params:
        raise RuntimeError("No trainable parameters after freeze map.")
    print(f"Trainable params: {sum(p.numel() for p in params):,} / {sum(p.numel() for p in lm.parameters()):,}")

    groups = (
        common.full_param_groups(lm, lr_specs["transformer"])
        if args.full_finetune
        else common.build_param_groups(lm, lr_specs)
    )
    optimizer = torch.optim.AdamW(groups, lr=args.lr, fused=device.type == "cuda")
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
    data_epoch = 0
    resume_skip_batches = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(lm, optimizer, args.resume_checkpoint, device, dtype)
        micro_step = global_step * args.grad_accum_steps
        if frame_batch_sampler is not None:
            data_epoch, resume_skip_batches = divmod(micro_step, batches_per_epoch)
            if resume_skip_batches:
                print(
                    f"Resuming frame-batch epoch {data_epoch}; skipping "
                    f"{resume_skip_batches} covered batches."
                )
        elif args.sort_by_length and not replay_ids:
            resume_skip_batches = (global_step * args.grad_accum_steps) % batches_per_epoch
            if resume_skip_batches:
                print(f"Skipping {resume_skip_batches} sorted batches already covered by resume.")
        else:
            print("Resume with shuffled/replay data starts a fresh sampler order.")

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

    def run_teacher_forced_val(step: int) -> None:
        if val_dataloader is None:
            return
        text_w = common.schedule_value(text_points, step, total_steps)
        audio_w = common.schedule_value(audio_points, step, total_steps)
        metrics = common.evaluate_teacher_forced(
            lm, val_dataloader, device, checkpoint_info.model_type, audio_w, text_w, args.val_batches
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
        lm.train()
        chrf = float(metrics.get("chrf", 0.0))
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
        }
        with greedy_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
        marker = ""
        if chrf > best_chrf:
            best_chrf = chrf
            common.save_adapter(lm, out_dir / f"{checkpoint_prefix(args)}_best.safetensors", build_metadata(args))
            (out_dir / "best.json").write_text(
                json.dumps({"step": step, "chrf": chrf}, sort_keys=True) + "\n", "utf-8"
            )
            marker = " *best*"
        print(
            f"greedy step={step} chrf={chrf:.3f} nonempty_chrf={item['nonempty_chrf']:.3f} "
            f"nonempty={item['nonempty']}/{item['num']}{marker}"
        )
        common.empty_device_cache(device)

    dataloader = None
    data_iter = None
    current_replay_weight = None
    rebuild_count = 0
    last_log_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        replay_w = common.schedule_value(replay_points, global_step, total_steps)
        if dataloader is None or replay_w != current_replay_weight:
            current_replay_weight = replay_w
            loader_seed = args.seed + rebuild_count
            sampler = (
                common.make_replay_sampler(dataset, replay_ids, replay_w, args.replay_seed + rebuild_count)
                if replay_ids
                else None
            )
            rebuild_count += 1
            if frame_batch_sampler is not None:
                frame_batch_sampler.set_epoch(data_epoch)
            dataloader = common.make_cached_dataloader(
                dataset,
                args.batch_size,
                args.num_workers,
                args.sort_by_length,
                sampler,
                frame_batch_sampler,
                loader_seed,
            )
            data_iter = iter(dataloader)
            while resume_skip_batches > 0:
                try:
                    next(data_iter)
                except StopIteration:
                    if frame_batch_sampler is not None:
                        data_epoch += 1
                        frame_batch_sampler.set_epoch(data_epoch)
                    data_iter = iter(dataloader)
                    next(data_iter)
                resume_skip_batches -= 1

        text_w = common.schedule_value(text_points, global_step, total_steps)
        audio_w = common.schedule_value(audio_points, global_step, total_steps)
        lr_value = common.apply_lr_schedule(optimizer, global_step, total_steps, args.warmup_steps)

        optimizer.zero_grad(set_to_none=True)
        step_loss = step_audio = step_text = 0.0
        step_text_tokens = 0
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                if frame_batch_sampler is not None:
                    data_epoch += 1
                    frame_batch_sampler.set_epoch(data_epoch)
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

        if (
            args.mps_empty_cache_every
            and common.is_mps(device)
            and global_step % args.mps_empty_cache_every == 0
        ):
            common.empty_device_cache(device)

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
                    "On MPS, use --dtype bfloat16 or lower --lr."
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
                "replay_weight": replay_w,
            }
            item.update(common.mps_memory_stats(device))
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
            memory_msg = ""
            if "mps_driver_gb" in item:
                memory_msg = f" mps={item['mps_allocated_gb']:.1f}/{item['mps_driver_gb']:.1f}GB"
            print(
                f"step={global_step} loss={item['loss']:.4f} audio={item['audio_loss']:.4f} "
                f"text={item['text_loss']:.4f} tw={text_w:g} rw={replay_w:g} "
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
        best_path = out_dir / f"{checkpoint_prefix(args)}_best.safetensors"
        print(f"Best greedy val chrF={best_chrf:.3f} -> {repo_display_path(best_path)}")
    kind = "full-model checkpoint" if args.full_finetune else "LoRA adapter"
    print(f"Saved final {kind} at step {global_step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()
