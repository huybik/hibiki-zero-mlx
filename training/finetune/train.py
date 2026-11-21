#!/usr/bin/env python
"""Full-model Hibiki-Zero SFT on multi-sentence streams, of one direction or both.

Trains every parameter from the upstream weight with fp32 masters, bf16 autocast and fused
AdamW, under a linear warmup then --schedule: cosine decay to 0 by default; `constant` holds the
peak, the stable phase of a warmup-stable-decay run, and `decay` falls from the peak to 0 as
1 - sqrt of the progress, its decay, run as a new run from one of the constant run's pairs (its
model as --model-weight, its AdamW moments as --optimizer-state). Streams are planned once from the grounded cache
rows with --seed (`streams.py`) and batched up to --batch-frames padded frames, one update a batch;
every epoch replays that order. Each validation reports teacher-forced losses on a fixed, seeded stream plan
and batch order of each validation cache (so --val-batches takes a random subset of each) and translates
the rows of every --eval-pairs file free-running in one pass, sampled as `eval.py` samples, and
scores each file as its own set. By default these are FLEURS dev (real Vietnamese speech) and
held-out Common Voice and Libriheavy rows of the training data's own domain. Of the checkpoints
whose every set passes the generation health gates, the one with the highest mean chrF over the
first set of each direction (`common.selecting_sets`: the first set, for a run of one) is
promoted to best_stepNNNNNN.safetensors.
With --teacher-weight (a model reading the cache's codebooks, on the same tokenizer), each token's
training loss is its KL from that frozen bf16 teacher's distribution on the same stream
(`common.token_kl`) instead of its cross-entropy, with the same weights, where both models score
(their delays may differ, as a heads student's do): a phone student distils the model it was cut
from ([phone-student](../../docs/agent-memory/phone-student.md)). The teacher reads every codebook
of the stream, the model its own cut. Validation is unchanged.

With --palettize-bits (a parallel-heads student), each step trains against the Core ML export's
palettization (`training.student.palettize`): the forward and backward run on the weights snapped
to their k-means tables, the optimizer updates the full-precision ones, and validation scores the
palettized model; saved checkpoints hold the full-precision weights, which `palettize` snaps for
the export.

With --sampled-text P (parallel scheduled sampling of the words, against exposure bias), each
step first runs the main transformer teacher-forced without gradients and samples every text
token as generation samples it (temperature 0.8, top-k 250); each word piece of the text then
becomes the sampled word piece there with probability P in the history the main transformer
reads, while the PADs (the timing), each step's own tokens (the depformer's inputs), the audio
and the targets stay the dataset's. Generation conditions on the model's own words, teacher
forcing never does: this shows the model its own word choices in training at the cost of one more
main-transformer forward a step. Validation is unchanged
([exposure-bias](../../docs/agent-memory/exposure-bias.md)).

Checkpoint pairs (model + optimizer) are saved every --save-every steps from a pinned-memory
snapshot written by a background thread, so a save stalls training only for the copy off the
GPU. A start in an --out-dir that holds complete pairs resumes the newest one, so a preempted run
continues by running the same command again.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from moshi.utils.sampling import sample_token
from torch.utils.data import DataLoader

from training.finetune import common, lm_training, streams
from training.student import heads, palettize
from training.finetune.checkpoints import (
    BEST_METRIC,
    CheckpointWriter,
    append_log,
    checkpoint_pairs,
    claim_run_config,
    jsonable,
    keep_log_lines,
    load_newest_pair,
    load_optimizer_state,
    open_run,
    promote_best,
)
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

# A resumed run must match these run_config.json entries.
RESUME_KEYS = (
    "cache_dir",
    "val_cache_dir",
    "config_path",
    "model_weight",
    "teacher_config",
    "teacher_weight",
    "optimizer_state",
    "palettize_bits",
    "palettize_refit",
    "sampled_text",
    "mimi_weight",
    "tokenizer",
    "epochs",
    "max_steps",
    "batch_frames",
    "stream_seconds",
    "max_frames",
    "lr",
    "schedule",
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
SCHEDULES = ("cosine", "constant", "decay")  # what follows the warmup (learning_rate)


def parse_args() -> argparse.Namespace:
    # No abbreviations: h100.sh must be able to reject --out-dir.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument(
        "--val-cache-dir", type=Path, nargs="+", action="extend", required=True,
        help="Teacher-forced validation caches; a two-way run passes each direction's.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument(
        "--teacher-weight", type=Path,
        help="Distil this frozen model's per-token distributions instead of training on the targets alone.",
    )
    parser.add_argument("--teacher-config", type=Path, help="The teacher's config; default the 3B's.")
    parser.add_argument(
        "--palettize-bits", type=int, choices=(4, 6, 8),
        help="Train against the Core ML export's palettization at this many bits (a heads student).",
    )
    parser.add_argument(
        "--palettize-refit", type=int, default=100,
        help="Refit the palettization tables every this many steps; must divide --save-every.",
    )
    parser.add_argument(
        "--sampled-text", type=float, default=0.0,
        help="Probability that a word piece the main transformer reads is the model's own sampled "
        "word piece there (scheduled sampling of the words); the timing and the targets stay "
        "the dataset's.",
    )
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, help="Default: the one the model weight is on (its text_tokenizer metadata), in weights/.")
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 trains all epochs.")
    parser.add_argument(
        "--batch-frames", type=int, default=4608,
        help="Padded frames per batch (streams x frames), one update each; an 80 GB card fits 4,608 at once.",
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
    parser.add_argument(
        "--lr", type=float, default=2.25e-6,
        help="Peak learning rate. 2.25e-6 is round 4's 3e-6 at 8,192-frame batches times "
        "sqrt(4,608 / 8,192), Adam's square-root rule for the smaller batch.",
    )
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.05, help="Share of steps warming up linearly."
    )
    parser.add_argument(
        "--schedule", choices=SCHEDULES, default="cosine",
        help="After the warmup: cosine to 0; constant at the peak (warmup-stable-decay's stable "
        "phase); decay from the peak to 0 as 1 - sqrt(progress) (its decay, from a pair of the stable run).",
    )
    parser.add_argument(
        "--optimizer-state", type=Path,
        help="A pair's trainer_stepNNNNNN.pt whose model is --model-weight, beside it under its "
        "own name: a new run continues its AdamW moments, as a decay branched from a constant run does.",
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
        "each scored as its own set; the first of each direction selects the checkpoint (their mean chrF).",
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


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_frames", "max_frames", "eval_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_steps", "grad_clip", "val_every", "val_batches", "save_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("--keep-checkpoints must be non-negative")
    if len({resolve_repo_path(path).resolve() for path in args.val_cache_dir}) != len(args.val_cache_dir):
        raise ValueError("--val-cache-dir names a cache twice")
    shortest, longest = args.stream_seconds
    if not 0 < shortest <= longest:
        raise ValueError("--stream-seconds needs 0 < MIN <= MAX")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    if args.teacher_config is not None and args.teacher_weight is None:
        raise ValueError("--teacher-config needs --teacher-weight")
    if not 0 <= args.sampled_text < 1:
        raise ValueError("--sampled-text must be in [0, 1)")
    if args.sampled_text and args.teacher_weight is not None:
        raise ValueError("--sampled-text trains on the dataset's targets; a student learns from "
                         "its own samples on-policy, with grpo.py --teacher-weight")
    if args.schedule == "decay" and (args.optimizer_state is None or args.warmup_fraction):
        raise ValueError("--schedule decay continues a constant run's pair: it needs "
                         "--optimizer-state and --warmup-fraction 0")
    # A resumed pair must sit on a refit step, so it gets the tables its run had.
    if args.palettize_bits and (args.palettize_refit <= 0 or not args.save_every
                                or args.save_every % args.palettize_refit):
        raise ValueError("--palettize-refit must be positive and divide a positive --save-every")
    # Best-model promotion hard-links the checkpoint saved at the validation step.
    if args.val_every and (not args.save_every or args.val_every % args.save_every):
        raise ValueError("--val-every must be a multiple of a positive --save-every")
    common.require_torch_compile()


def sample_text(
    lm: Any, codes: torch.Tensor, condition_tensors: Any, rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel scheduled sampling of the words (Duckworth et al. 2019, arXiv 1906.04331): a history
    for `common.compute_batch_losses`, `codes` with each word piece of the text replaced, with
    probability `rate`, by the word piece the main transformer's teacher-forced distribution
    samples there as validation samples text (`common.TRAINER_GENERATION`), and how many replaced
    pieces differ from the dataset's. Runs without gradients, under the caller's autocast.
    Only a word piece the sample keeps a word piece is replaced: the targets stay the dataset's,
    so a word sampled a frame early (where the dataset has PAD) would teach the model to write it
    again a frame later, and one sampled late (PAD) to drop it; the timing stays the dataset's.
    The audio stays the dataset's too: generation draws a frame's codebooks one after another,
    and tokens sampled each on its own would pair a sampled codebook with the dataset's next."""
    cfg = common.TRAINER_GENERATION
    with torch.no_grad():
        logits, scored = lm_training.text_logits(lm, codes, condition_tensors)
        # Unscored positions hold NaN logits; they are never replaced.
        sampled = sample_token(logits.float().nan_to_num(), True, cfg.text_temp, cfg.top_k_text)
    text, sampled = codes[:, :1], sampled.to(codes.dtype)
    pad = lm.text_padding_token_id  # the special tokens (PAD, EOS) sit at or below it
    words = (text > pad) & (sampled > pad)
    replace = scored & words & (torch.rand(scored.shape, device=codes.device) < rate)
    history = codes.clone()
    history[:, :1] = torch.where(replace, sampled, text)
    return history, (replace & (sampled != text)).sum()


def learning_rate(
    step: int, peak: float, warmup_steps: int, total_steps: int, schedule: str = "cosine"
) -> float:
    """Linear warmup to `peak`, then `schedule` towards `total_steps`: `cosine` decays to 0
    (moshi-finetune's OneCycle shape, without its momentum cycling); `constant` holds the peak;
    `decay` falls to 0 as 1 - sqrt(progress), the cooldown that matched cosine in Hägele et al.
    2024 (arXiv 2405.18392). `step` counts updates from 0."""
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if schedule == "constant":
        return peak
    if schedule == "decay":
        return peak * (1 - math.sqrt(progress))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    # Validation builds and frees a bf16 model and generation caches between training steps;
    # expandable segments keep the freed memory reusable for the training batches.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    validate_args(args)
    common.seed_all(args.seed)

    device = common.check_device("cuda")
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    val_cache_dirs = [require_dir(path, "validation code cache directory") for path in args.val_cache_dir]
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "upstream model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = common.run_tokenizer(args.tokenizer, args.model_weight)
    args.eval_pairs = [require_file(path, "evaluation pairs") for path in args.eval_pairs]

    out_dir = resolve_repo_path(args.out_dir)
    resuming, best_state = open_run(out_dir)
    if args.optimizer_state is not None:  # read only at the start: a resume has its own pairs
        args.optimizer_state = (resolve_repo_path(args.optimizer_state) if resuming
                                else require_file(args.optimizer_state, "optimizer state"))

    checkpoint_info = common.load_checkpoint_info(args)
    if args.palettize_bits and checkpoint_info.audio_heads != heads.KIND:
        raise ValueError("--palettize-bits trains for the Core ML export, which runs only a parallel-heads student")
    codebooks = int(checkpoint_info.lm_config["dep_q"])  # a stream's, fewer for a phone student
    teacher_info = None if args.teacher_weight is None else common.teacher_checkpoint_info(args)
    frames = streams.frame_range(args.stream_seconds)
    rows, layout = streams.load_rows(cache_dirs, args.max_frames, args.tokenizer)
    # On the CPU in float32, so every machine and resume encodes the same silence (up to a rare
    # near-tie in a residual code); at the cache's codebooks, which streams are assembled at.
    silence = streams.silence_track(checkpoint_info.get_mimi(device="cpu"), frames[1], layout.dep_q)
    # A teacher reads the whole stream; the model's rows are cut from each batch on the GPU.
    student_rows = None
    if teacher_info is not None:
        if int(teacher_info.lm_config["dep_q"]) != layout.dep_q:
            raise ValueError(f"the teacher must read the cache's {layout.dep_q} codebooks a stream")
        student_rows = torch.tensor(streams.model_rows(layout, codebooks), device=device)
    dataset = streams.StreamDataset(
        rows, layout, silence, frames, args.seed, layout.dep_q if teacher_info else codebooks
    )
    batches = dataset.batches(args.batch_frames, args.seed)
    exposure = dataset.exposure()
    print(
        f"Planned {exposure['streams']} streams from {exposure['rows']} cache rows; "
        f"stream_frames={exposure['stream_frames']:,} source_hours={exposure['source_hours']:.2f} "
        f"batches={len(batches)}"
    )
    # Each validation cache is planned and scored apart, as `validate.py` scores one: a direction's
    # losses stay visible beside the other's, and a cache's streams do not change when another
    # joins the run. In the main process: forking workers at a validation would fork the checkpoint
    # writer's thread too, and 300 batches assemble in ~3 s.
    names = [str(path) for path in args.val_cache_dir]  # as given: h100.sh replays the arguments
    val_loaders: dict[str, DataLoader] = {}
    val_row_counts: dict[str, int] = {}
    val_stream_counts: dict[str, int] = {}
    for name, val_cache_dir in zip(names, val_cache_dirs, strict=True):
        val_rows, val_layout = streams.load_rows([val_cache_dir], args.max_frames, args.tokenizer)
        if val_layout != layout:
            raise RuntimeError(f"Validation layout {val_layout} of {name} differs from training {layout}")
        val_dataset = streams.StreamDataset(val_rows, layout, silence, frames, streams.VALIDATION_SEED, codebooks)
        val_loaders[name] = DataLoader(
            val_dataset,
            batch_sampler=val_dataset.batches(args.batch_frames, streams.VALIDATION_SEED),
            collate_fn=streams.collate,
        )
        val_row_counts[name], val_stream_counts[name] = len(val_rows), len(val_dataset)
        print(f"Planned {len(val_dataset)} validation streams from {len(val_rows)} rows of {name}")
    # Checkpoints are selected on free-running translation, sampled as eval.py samples. The sets
    # share batches, so one pass translates them all.
    eval_sets, eval_rows = common.load_eval_rows(args.eval_pairs)
    selecting = common.selecting_sets(eval_sets, eval_rows)
    print(
        f"Validation translates {len(eval_rows)} rows of {', '.join(eval_sets)}; "
        f"{', '.join(selecting)} select the checkpoint"
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
            "val_rows": val_row_counts,
            "val_streams": val_stream_counts,
            "eval_rows": len(eval_rows),
        }
    )
    # Before the model loads: a resume with other arguments or data stops here, in seconds.
    claim_run_config(out_dir, run_config, RESUME_KEYS, resuming)

    print(f"Loading the start model from {repo_display_path(args.model_weight)}")
    lm = common.load_policy(checkpoint_info, device)
    # The training forward attends causally over the whole stream (lm_training): the longest
    # padded stream must fit the attention context, checked here rather than at its first batch.
    context = lm.transformer.layers[0].self_attn.context
    longest = math.ceil(frames[1] / streams.FRAME_BUCKET) * streams.FRAME_BUCKET
    if context is not None and longest > context:
        raise ValueError(f"--stream-seconds allows {longest} frames, over the context of {context}")
    params = list(lm.parameters())
    print(f"Trainable params: {sum(param.numel() for param in params):,}")
    teacher = None
    if teacher_info is not None:
        print(f"Loading the teacher from {repo_display_path(args.teacher_weight)}")
        teacher = common.load_teacher(teacher_info, args.teacher_weight, device, codebooks)
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    optimizer = common.fused_adamw(params, args)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    val_log_path = out_dir / "val_log.jsonl"
    log_path = out_dir / "train_log.jsonl"
    global_step = load_newest_pair(lm, optimizer, out_dir, device) if resuming else 0
    if not global_step and args.optimizer_state is not None:
        branched = load_optimizer_state(optimizer, args.optimizer_state, args.model_weight, device)
        print(f"AdamW continues from step {branched} of {repo_display_path(args.optimizer_state)}")
    palettizer = None
    if args.palettize_bits:
        palettizer = palettize.Palettizer(lm, args.palettize_bits)
        palettizer.refit()
        print(f"Palettizing {len(palettizer.params)} weights at {args.palettize_bits} bits")

    def palettized():
        """The palettized weights in place for a forward and its backward, or a validation."""
        return palettizer.applied() if palettizer is not None else contextlib.nullcontext()
    # Steps already validated. A pair is saved before its validation, so a run cut short there
    # validates on resume; a replacement pod without val_log validates again.
    validated = {int(item["step"]) for item in keep_log_lines(val_log_path, global_step)}
    keep_log_lines(log_path, global_step)
    if global_step > total_steps:
        raise RuntimeError("The run is beyond the requested stop")

    writer = CheckpointWriter(lm, optimizer, args, out_dir)

    def training_batches(start_step: int):
        """The planned batches from `start_step` on, epoch after epoch."""
        offset = start_step % batches_per_epoch
        while True:
            writer.wait()  # the workers fork; the writer's thread must not be running
            # Forked, not Python 3.14's default forkserver, which would pickle the dataset into each
            # worker: every row's codes, ~133 GiB for the two-way run, four times.
            yield from DataLoader(
                dataset,
                batch_sampler=batches[offset:],
                collate_fn=streams.collate,
                num_workers=args.num_workers,
                pin_memory=True,
                multiprocessing_context="fork" if args.num_workers else None,
            )
            offset = 0

    data_iter = training_batches(global_step)
    condition_cache: dict[int, Any | None] = {}
    teacher_conditions: dict[int, Any | None] = {}

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
            "data_seconds": 0.0,
            "text_changed": 0,
            "word_tokens": 0,
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
        with palettized():  # a palettizing run scores the model it would export
            teacher_forced = {
                name: common.evaluate_teacher_forced(
                    lm, loader, device, checkpoint_info.model_type, weights, args.val_batches
                )
                for name, loader in val_loaders.items()
            }
            # On a bf16 copy of the model, in the room the gradients freed above.
            free_running = common.validate_free_running(
                lm, checkpoint_info, mimi, text_tokenizer, eval_rows, eval_sets,
                args.eval_batch_size, out_dir / "eval" / f"step{step:06d}", args.seed,
            )
        healthy = all(metrics["health_eligible"] for metrics in free_running.values())
        for name, losses in teacher_forced.items():
            print(
                f"val step={step} {name} loss={losses['loss']:.4f} "
                f"semantic={losses['semantic_loss']:.4f} "
                f"content={losses['content_text_loss']:.4f} "
                f"acc={losses['content_acc']:.3f} pad_acc={losses['pad_acc']:.3f}"
            )
        print(f"val step={step} health={'pass' if healthy else 'fail'}")
        for name, metrics in free_running.items():
            print(f"val step={step} {name} {common.score_summary(metrics)}")
        # chrF weighs recall, so overlong output barely lowers it: the gates come first. Promotion
        # links the step's model file, so its write must be done, and precedes the log line,
        # which marks the step done for a resume.
        if healthy:
            writer.wait()
            chrf = sum(float(free_running[name]["chrf"]) for name in selecting) / len(selecting)
            best_state = promote_best(step, chrf, out_dir, best_state)
        seconds = time.time() - started
        item = {
            "step": step,
            "seconds": seconds,
            "teacher_forced": teacher_forced,
            "free_running": free_running,
        }
        append_log(val_log_path, item)
        validated.add(step)
        print(f"val step={step} took {seconds:.0f} s")
        last_log_time += seconds  # the next train-log line times training steps only

    resumed_at_validation = args.val_every and global_step % args.val_every == 0
    if global_step and resumed_at_validation and global_step not in validated:
        run_validation(global_step)

    while global_step < total_steps:
        lr_value = learning_rate(global_step, args.lr, warmup_steps, total_steps, args.schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr_value
        optimizer.zero_grad(set_to_none=True)
        fetch_started = time.time()
        batch = next(data_iter)
        log["data_seconds"] += time.time() - fetch_started  # the wait for the stream workers
        codes = batch["codes"].to(device=device, non_blocking=True)
        batch_size, frames_padded = int(codes.shape[0]), int(codes.shape[-1])
        if batch_size not in condition_cache:
            condition_cache[batch_size] = common.batch_condition_tensors(
                lm, checkpoint_info.model_type, batch_size
            )
        teacher_output = None
        if teacher is not None:
            if batch_size not in teacher_conditions:
                teacher_conditions[batch_size] = common.batch_condition_tensors(
                    teacher, teacher_info.model_type, batch_size
                )
            with torch.no_grad(), autocast:
                teacher_output = teacher(codes, condition_tensors=teacher_conditions[batch_size])
            codes = codes.index_select(1, student_rows)
        with palettized():  # the straight-through estimator: gradients at the palettized weights
            with autocast:
                history = None
                if args.sampled_text:
                    history, changed = sample_text(
                        lm, codes, condition_cache[batch_size], args.sampled_text
                    )
                    log["text_changed"] += changed
                losses = common.compute_batch_losses(
                    lm, codes, condition_cache[batch_size], weights,
                    teacher=teacher_output, history=history,
                )
            history = teacher_output = None  # the loss's graph keeps what its backward needs
            losses["loss"].backward()
        for key in LOG_LOSSES:
            log["losses"][key] += losses[key].detach()
        log["audio_tokens"] += losses["semantic_n"] + losses["acoustic_n"]
        log["text_tokens"] += losses["content_n"] + losses["pad_n"]
        log["word_tokens"] += losses["content_n"]
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        else:
            grad_norm = torch.nn.utils.get_total_norm(
                [param.grad for param in params if param.grad is not None]
            )
        optimizer.step()
        global_step += 1
        # On the last step too: its validation and the snapped export then share one fit.
        if palettizer is not None and (global_step % args.palettize_refit == 0
                                       or global_step == total_steps):
            palettizer.refit()

        # Summed on the GPU, as the losses are: checked and logged without a sync per step.
        grad_norm = grad_norm.detach()
        log["grad_norm"] += grad_norm
        previous_max = log["max_grad_norm"]
        log["max_grad_norm"] = (
            grad_norm if previous_max is None else torch.maximum(previous_max, grad_norm)
        )
        log["steps"] += 1
        log["streams"] += batch_size
        log["stream_frames"] += int(batch["frames"].sum())
        log["padded_frames"] += batch_size * frames_padded
        log["source_frames"] += int(batch["source_frames"].sum())
        log["min_batch"] = min(log["min_batch"], batch_size)
        log["max_batch"] = max(log["max_batch"], batch_size)
        log["max_frames"] = max(log["max_frames"], frames_padded)

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
                "data_sec_per_step": log["data_seconds"] / steps,
                "log_steps": steps,
                "lr": lr_value,
                "grad_norm": float(log["grad_norm"]) / steps,
                "max_grad_norm": float(log["max_grad_norm"]),
            }
            if args.sampled_text:  # the share of the content tokens (words, EOS) it read as other words
                item["text_changed"] = float(log["text_changed"]) / max(1, int(log["word_tokens"]))
            append_log(log_path, item)
            print(
                f"step={global_step} loss={item['loss']:.4f} "
                f"audio={item['audio_loss']:.4f} text={item['text_loss']:.4f} "
                f"B={item['streams'] / steps:.1f} T<={item['max_padded_frames']} "
                f"lr={item['lr']:.1e} grad_norm={item['grad_norm']:.3f} "
                f"s/step={item['sec_per_step']:.3f} data={item['data_sec_per_step']:.3f}"
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
