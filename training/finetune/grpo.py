#!/usr/bin/env python
"""GRPO for the full model (Hibiki-Zero, arXiv 2602.11072v2, §3.3; plan in docs/grpo_plan.md).

Every --tau updates, π_old becomes the current policy:
1. a bf16 copy of it samples --group rollouts for each stream of the window's updates (the
   planned streams in a seeded order), each as long as its stream plus --tail-seconds, the tail
   evaluation generates past a source, and at most --frames. The window's streams are generated
   longest first in batches of --gen-frames (rollouts times the batch's longest), so a batch
   stops at its own longest rollout and shorter ones batch more rows. Meanwhile the fp32 policy
   and AdamW's states wait on the CPU, leaving the GPU's memory to the batches;
2. the rollouts get BLEU process rewards and group-normalized advantages (`rewards.py`) in CPU
   workers, while the GPU keeps generating;
3. the policy itself scores them, giving π_old's logprobs. It runs the update's own forward on
   the update's own micro-batches, gradients enabled and the result detached: a no-grad forward,
   or another batch shape, picks other bf16 kernels.
   The first update's ratio is then exactly 1 (MPS: 1.000000 against a no-grad pass's 1.0012,
   with 6 tokens clipped).
Each update then takes --streams streams' rollouts, in micro-batches, each forward as long as
its last kept token needs. Its loss is the clipped ratio π_θ/π_old of every text and target-audio
token times its frame's advantage, weighted --text-weight for text against 1 for each audio
codebook, summed over frames and averaged over rollouts.

Every --eval-every updates, and at the start, the run translates --eval-pairs free-running, as SFT
validation does. The first set of each direction selects (`eval_rows.selecting_sets`: FLEURS dev
both ways by default): the run promotes the checkpoint with the best mean chrF over them among
those that pass every set's health gates and end within LATENCY_LIMIT_S of the source on each (its
median end offset); val_log.jsonl records each evaluation's verdict.

With --teacher-weight, the update also distils a frozen teacher on-policy: the reverse
KL(policy || teacher) of every kept token's next-token distribution on the policy's own rollouts
(the teacher scores the same rollout), weighted as the GRPO term is, times --kl-coef. That term
alone (--reward-coef 0) is on-policy distillation (Agarwal et al. 2024, arXiv 2306.13649): no π_old
pass, the rewards only logged. Both together keep a GRPO run near its teacher. A teacher that
delays a codebook less than the policy reads, at a text token, frames of it the policy sampled after
that token (a heads student's codebooks 3-15), so its distribution may lean towards the policy's
own choice; grpo_profile.py bounds that.

A --room-share of streams hold a room's noise in their source wherever nobody speaks, instead of
digital silence (`rooms.room_codes`, from the noise bank in --bank-dir): text written there
costs BLEU like any other extra word, so the policy learns to stay quiet in a room, as a phone's
microphone hears one between sentences.

Rollouts replay a cached stream's Vietnamese through `LMGen` and keep every sampled text and
audio token in the training frame layout, so the training forward scores them in exactly the
contexts they were sampled in. In that layout, `LMGen` reads the input given at step s as source
frame s - 1, samples text frame s at step s, and samples audio codebook k's frame s - delay_k.

Pairs, resume and logs are `checkpoints.py`'s, so `h100.sh grpo` and `hf_sync.py` run it as they
run SFT. Saves land on window boundaries: a resume starts a window and draws its rollouts again.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.finetune import grpo_objective, grpo_rollout, rewards
from training.finetune.checkpoints import (
    CheckpointWriter,
    append_log,
    claim_run_config,
    jsonable,
    keep_log_lines,
    load_newest_pair,
    open_run,
    promote_best,
)
from training.finetune.eval_rows import selecting_sets
from training.finetune.generation import GenerationConfig, load_eval_rows, validate_free_running
from training.finetune.losses import batch_condition_tensors
from training.finetune.metrics import score_summary
from training.finetune.model import (
    HostOffload,
    adam_states,
    bf16_copy,
    check_device,
    fused_adamw,
    load_checkpoint_info,
    load_policy,
    load_teacher,
    require_torch_compile,
    run_tokenizer,
)
from training.finetune.packs import FRAME_RATE
from training.finetune.rooms import add_room_args
from training.finetune.rows import load_rows
from training.finetune.run_files import BEST_METRIC
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

# A resumed run must match these run_config.json entries.
RESUME_KEYS = (
    "cache_dir", "pair_cache_dir", "config_path", "model_weight", "mimi_weight", "tokenizer", "max_frames",
    "stream_seconds", "frames", "tail_seconds", "streams", "group", "updates", "tau", "lr", "adam_beta1",
    "adam_beta2", "weight_decay", "grad_clip", "epsilon", "alpha", "text_weight", "temperature",
    "top_k", "eval_pairs", "eval_batch_size", "eval_every", "save_every", "keep_checkpoints",
    "seed", "train_rows", "train_streams", "eval_rows", "room_share", "teacher_weight",
    "teacher_config", "reward_coef", "kl_coef",
)
EVAL_PAIRS = [  # the two-way model's (user, 2026-11-10: one GRPO, on both directions)
    Path("data/eval/fleurs-vi/dev.jsonl"),
    Path("data/eval/fleurs-en/dev.jsonl"),
    Path("data/eval/en-vi-val/cv27.jsonl"),
    Path("data/eval/en-vi-val/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-noisy/cv27.jsonl"),
    Path("data/eval/en-vi-val-noisy/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-en/cv27.jsonl"),
    Path("data/eval/en-vi-val-en/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-en-noisy/cv27.jsonl"),  # the English sources the training rows now take noisy
    Path("data/eval/en-vi-val-en-noisy/libriheavy.jsonl"),
]
LATENCY_LIMIT_S = 3.0  # the latest median end offset, on each selecting set, at which a checkpoint may promote
# The defaults the profiler measures with too.
TEMPERATURE = 0.8  # rollouts sample, and logprobs are taken, at evaluation's temperature
TOP_K = 250
EPSILON = 0.2  # the ratio clip
TEXT_WEIGHT = 100.0  # c_0: the text stream against 1 for each audio codebook
# The allocators of a run and its profile. Expandable segments keep freed GPU memory reusable
# between generation batches and updates. Pinned host blocks over 1 MB keep their size, where torch
# would round each up to a power of two: `HostOffload`'s and the checkpoint writer's buffers, one a
# tensor, then pin 34.4 GiB on the heads student under the 3B instead of 39.6, 70.0 on the 3B
# instead of 77.4 (`h100.sh sizes`).
ALLOC_CONF = "expandable_segments:True,pinned_max_round_threshold_mb:1"


def set_alloc_conf() -> None:
    """ALLOC_CONF for this process, before CUDA starts. A different preset value is refused, not kept:
    `h100.sh sizes` checks the RAM for exact pinned blocks, so a run pinning rounded ones (5 GiB more on
    the heads student, 7 on the 3B) would pass preflight and be killed at its first save."""
    if (value := os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)) != ALLOC_CONF:
        raise RuntimeError(f"PYTORCH_CUDA_ALLOC_CONF={value!r} is set; GRPO runs on {ALLOC_CONF!r}: unset it")


def trainer_tensors(lm: Any, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """The fp32 policy's parameters and AdamW's state tensors."""
    return [*lm.parameters(), *adam_states(optimizer)]


def park_states(offload: HostOffload, optimizer: torch.optim.Optimizer) -> None:
    """AdamW's states to pinned host memory, once, until `offload.to_device()` (`--park-states`): only
    `optimizer.step` reads them, so the window's old-logprob pass and an update's micro-batches get
    their memory (10 GB on the 1.28B student, 25 on the 3B). None before the first step."""
    states = adam_states(optimizer)
    if states and not offload.out:
        offload.to_host(states)


def parse_args() -> argparse.Namespace:
    # No abbreviations: h100.sh must be able to reject --out-dir.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--pair-cache-dir", type=Path, nargs="+", default=[], help="The other direction's caches of the same rows, read only for their text, which times each row's source (rows.load_rows); not needed for caches given with --cache-dir.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True, help="The SFT checkpoint.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, help="Default: the one the model weight is on (its text_tokenizer metadata), in weights/.")
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument(
        "--stream-seconds", type=float, nargs=2, default=[30.0, 75.0], metavar=("MIN", "MAX"),
        help="The paper's RL inputs are 30-75 s utterances.",
    )
    parser.add_argument("--frames", type=int, default=1200, help="The longest rollout.")
    parser.add_argument(
        "--tail-seconds", type=float, default=GenerationConfig.tail_s,
        help="A rollout runs this long past its stream, as evaluation generates past a source.",
    )
    parser.add_argument("--streams", type=int, default=32, help="Streams per update.")
    parser.add_argument("--group", type=int, default=4, help="Rollouts per stream (G).")
    parser.add_argument(
        "--updates", type=int, default=2000,
        help="The paper's 2,000; every evaluation promotes, so a plateaued run can just be stopped.",
    )
    parser.add_argument("--tau", type=int, default=20, help="Updates per π_old.")
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="Ratio clip.")
    parser.add_argument("--alpha", type=float, default=rewards.ALPHA, help="Final BLEU's share.")
    grpo_objective.add_objective_args(parser)
    parser.add_argument("--text-weight", type=float, default=TEXT_WEIGHT, help="c_0 against 1 per codebook.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gen-frames", type=int, default=230_400,
        help="A generation batch's rollouts times its longest rollout's frames (profiled: grpo.md).",
    )
    parser.add_argument(
        "--park-states", action="store_true",
        help="AdamW's states wait in pinned host memory while the old-logprob pass and an update's "
             "micro-batches run (`park_states`), for a card where the update runs out of memory. It "
             "costs a copy of them out and back an update, and the results don't change.",
    )
    parser.add_argument("--micro-batch", type=int, default=2, help="Rollouts per forward.")
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--eval-pairs", type=Path, nargs="+", default=EVAL_PAIRS)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    add_room_args(parser, grpo_rollout.ROOM_SHARE)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("frames", "streams", "group", "updates", "tau", "gen_frames", "micro_batch",
                 "reward_workers", "eval_batch_size", "eval_every", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.group < 2:
        raise ValueError("--group needs at least 2 rollouts to normalize")
    grpo_objective.validate_objective_args(args)
    if args.save_every % args.tau or args.eval_every % args.save_every or args.updates % args.tau:
        raise ValueError(
            "--save-every and --updates must be multiples of --tau, --eval-every of --save-every"
        )
    if args.gen_frames < args.group * args.frames:
        raise ValueError("--gen-frames must hold a group of the longest rollouts")
    longest = round(args.stream_seconds[1] * FRAME_RATE)
    if longest + round(args.tail_seconds * FRAME_RATE) > args.frames:
        raise ValueError("--frames must hold the longest stream and the tail after it")
    if args.tail_seconds < 0:
        raise ValueError("--tail-seconds must not be negative")
    require_torch_compile()


def main() -> None:
    set_alloc_conf()
    args = parse_args()
    validate_args(args)
    args.model_weight = require_file(args.model_weight, "start checkpoint")
    args.tokenizer = run_tokenizer(args.tokenizer, args.model_weight)  # before the workers load it
    pool = grpo_rollout.reward_pool(args.reward_workers, args.tokenizer)
    device = check_device("cuda")
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    args.config_path = require_file(args.config_path, "config")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.eval_pairs = [require_file(path, "evaluation pairs") for path in args.eval_pairs]
    out_dir = resolve_repo_path(args.out_dir)
    resuming, best_state = open_run(out_dir)

    info = load_checkpoint_info(args)
    tokenizer = info.get_text_tokenizer()
    codebooks = int(info.lm_config["dep_q"])  # a stream's, fewer for a phone student
    teach = grpo_objective.teacher_info(args, codebooks)
    pair_dirs = [require_dir(path, "pair code cache directory") for path in args.pair_cache_dir]
    rows, layout = load_rows(cache_dirs, args.max_frames, args.tokenizer, pair_dirs)
    mimi = info.get_mimi(device=device)
    dataset = grpo_rollout.stream_dataset(args, info, rows, layout, codebooks, mimi)
    if len(dataset) < args.updates * args.streams:
        raise ValueError(f"{len(dataset)} streams, fewer than {args.updates} updates need")
    print(f"Planned {len(dataset)} streams from {len(rows)} rows; {args.updates} updates use "
          f"{args.updates * args.streams}")
    # Planned streams follow their documents (a long recording fills ~75 in a row), so updates
    # take them in a seeded order: the same on a resume.
    order = grpo_rollout.stream_order(len(dataset), args.seed)
    eval_sets, eval_rows = load_eval_rows(args.eval_pairs)
    selecting = selecting_sets(eval_sets, eval_rows)
    print(f"Evaluation translates {len(eval_rows)} rows of {', '.join(eval_sets)}; {', '.join(selecting)} select")

    run_config = {key: jsonable(value) for key, value in vars(args).items()}
    run_config.update(
        train_rows=len(rows), train_streams=len(dataset), eval_rows=len(eval_rows),
        total_steps=args.updates, val_every=args.eval_every, best_metric=BEST_METRIC,
    )
    # Before the model loads: a resume with other arguments or data stops here, in seconds.
    claim_run_config(out_dir, run_config, RESUME_KEYS, resuming)

    lm = load_policy(info, device)
    params = list(lm.parameters())
    optimizer = fused_adamw(params, args)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    stream_weights = torch.ones(1 + lm.dep_q, device=device)
    stream_weights[0] = args.text_weight
    teacher = None
    if teach is not None:
        print(f"Loading the teacher from {repo_display_path(args.teacher_weight)}")
        teacher = load_teacher(teach, args.teacher_weight, device, codebooks)
    objective = grpo_objective.Objective(stream_weights, args.epsilon, args.temperature,
                                         args.reward_coef, args.kl_coef or 0.0)
    # Parked on the host beside the policy while a window generates.
    teacher_params = [] if teacher is None else list(teacher.parameters())
    silence_frame = dataset.silence[:codebooks, -1].long().to(device)
    tail = round(args.tail_seconds * FRAME_RATE)
    delays = grpo_objective.scored_delays(lm, teacher)

    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    step = load_newest_pair(lm, optimizer, out_dir, device) if resuming else 0
    validated = {int(item["step"]) for item in keep_log_lines(val_log_path, step)}
    keep_log_lines(log_path, step)
    if step % args.tau:
        raise RuntimeError(f"Resumed step {step} is not on a --tau boundary")
    writer = CheckpointWriter(lm, optimizer, args, out_dir)
    # The rows and streams (~4 million objects for the tranche) never become garbage: frozen, they
    # stay out of the collection after every generation batch (0.6 s each otherwise).
    gc.collect()
    gc.freeze()

    offload = HostOffload()

    def evaluate(at: int) -> None:
        nonlocal best_state
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        # It reads neither, and at batch 64 the heads student's ran out of memory beside them on 32 GB.
        offload.to_host(adam_states(optimizer) + teacher_params)
        free_running = validate_free_running(
            lm, info, mimi, tokenizer, eval_rows, eval_sets, args.eval_batch_size,
            out_dir / "eval" / f"step{at:06d}", args.seed,
        )
        offload.to_device()
        healthy = all(metrics["health_eligible"] for metrics in free_running.values())
        for name, metrics in free_running.items():
            print(f"val step={at} {name} {score_summary(metrics)}")
        # A fixed limit, the same for both directions and every run: quality comes first.
        # A blocked checkpoint is not kept.
        on_time = all(free_running[name]["median_end_offset_s"] <= LATENCY_LIMIT_S for name in selecting)
        if at > 0 and healthy and on_time:
            writer.wait()
            chrf = sum(float(free_running[name]["chrf"]) for name in selecting) / len(selecting)
            best_state = promote_best(at, chrf, out_dir, best_state)
        promoted = best_state is not None and best_state["step"] == at
        seconds = time.time() - started
        item = {"step": at, "seconds": seconds, "free_running": free_running, "healthy": healthy,
                "on_time": on_time, "promoted": promoted}
        append_log(val_log_path, item)
        validated.add(at)
        print(f"val step={at} took {seconds:.0f} s")

    if step % args.eval_every == 0 and step not in validated:
        evaluate(step)

    per_update = args.streams * args.group
    while step < args.updates:
        window = min(args.tau, args.updates - step)
        window_started = time.time()
        # 1. π_old's rollouts for the window, rewards in the workers as each batch lands.
        optimizer.zero_grad(set_to_none=True)
        stream_ids = [int(order[index]) for update in range(step, step + window)
                      for index in range(update * args.streams, (update + 1) * args.streams)]
        sources = torch.stack([grpo_rollout.stream_source(dataset, index, codebooks)
                               for index in stream_ids])
        sources = sources.repeat_interleave(args.group, dim=0)
        lengths = torch.tensor(
            [grpo_rollout.rollout_frames(dataset.streams[index].frames, tail, args.frames)
             for index in stream_ids]
        ).repeat_interleave(args.group)
        jobs: dict[int, Future] = {}

        def submit(batch_rows: torch.Tensor, rollouts: torch.Tensor) -> None:
            for offset in range(0, len(batch_rows), args.group):
                first_row = int(batch_rows[offset])
                group = first_row // args.group
                stream = dataset.streams[stream_ids[group]]
                sentences = grpo_rollout.stream_sentences(stream, rows, tokenizer)
                tokens = rollouts[offset : offset + args.group, 0, : int(lengths[first_row])]
                jobs[group] = pool.submit(grpo_rollout._group_rewards, tokens.numpy().copy(), sentences,
                                          args.alpha)

        generator = bf16_copy(info, lm, device)
        moving = time.time()
        offload.to_host(trainer_tensors(lm, optimizer) + teacher_params)
        moved = time.time() - moving
        batches = grpo_rollout.generation_batches(lengths, args.gen_frames, args.group)
        rollouts = grpo_rollout.sample_window(generator, info, sources, lengths, batches,
                                              silence_frame, (args.seed, step), args.temperature,
                                              args.top_k, submit)
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        moving = time.time()
        offload.to_device()
        moved += time.time() - moving
        generated = time.time() - window_started
        results = [jobs[group].result() for group in range(len(stream_ids))]
        advantages = torch.zeros((len(rollouts), args.frames))
        for group, result in enumerate(results):  # [G, the group's length]; 0 past it
            rows_of = slice(group * args.group, (group + 1) * args.group)
            advantages[rows_of, : result[2].shape[1]] = torch.from_numpy(result[2]).float()
        final_bleu = np.concatenate([result[1][:, -1] for result in results])
        eos_rate = (rollouts[:, 0] == tokenizer.eos_id()).any(dim=1).float()
        keep = grpo_objective.kept_tokens(rollouts, lengths, delays, int(tokenizer.eos_id()))
        spans = grpo_objective.forward_frames(keep, delays)
        codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
        rewarded = time.time() - window_started

        # 2. π_old's logprobs, which only the GRPO term reads: the policy before this window's
        # first update, through the update's own forward (gradients enabled, never
        # backpropagated). Micro-batches cut as each update cuts them: the same rows in the same
        # batch shapes.
        if args.park_states:
            park_states(offload, optimizer)
        old_logprobs = None
        if objective.reward_coef:
            old_logprobs = torch.zeros((len(codes), 1 + lm.dep_q, args.frames), dtype=torch.float32)
            with autocast:
                for start, end in grpo_objective.micro_batches(window, per_update, args.micro_batch):
                    frames = grpo_objective.micro_frames(spans, start, end, args.frames)
                    batch = codes[start:end, :, :frames].to(device=device, dtype=torch.long)
                    condition = batch_condition_tensors(lm, info.model_type, end - start)
                    logprobs, _ = grpo_objective.token_logprobs(lm, batch, condition, args.temperature)
                    old_logprobs[start:end, :, :frames] = logprobs.detach().cpu()
                    del logprobs
        scored = time.time() - window_started

        # 3. The window's updates.
        for update in range(window):
            update_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            if args.park_states:
                park_states(offload, optimizer)
            first = update * per_update
            totals: dict[str, Any] = {"loss": 0.0}
            forward_total = 0
            for start, end in grpo_objective.micro_batches(window, per_update, args.micro_batch, update):
                frames = grpo_objective.micro_frames(spans, start, end, args.frames)
                forward_total += frames * (end - start)
                batch = codes[start:end, :, :frames].to(device=device, dtype=torch.long)
                condition = batch_condition_tensors(lm, info.model_type, end - start)
                teacher_output = (None if teacher is None
                                  else grpo_objective.teacher_forward(teacher, teach, batch))
                with autocast:
                    loss, stats = grpo_objective.update_loss(
                        lm, batch, condition, keep[start:end, :, :frames].to(device), objective,
                        None if old_logprobs is None else old_logprobs[start:end, :, :frames].to(device),
                        advantages[start:end, :frames].to(device), teacher_output)
                teacher_output = None  # the loss's graph keeps what its backward needs
                (loss / per_update).backward()
                totals["loss"] += loss.detach()
                for key, value in stats.items():
                    if key == "max_deviation":
                        totals[key] = value if key not in totals else torch.maximum(totals[key], value)
                    else:
                        totals[key] = totals.get(key, 0) + value
            offload.to_device()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip or math.inf)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"Non-finite gradient norm at update {step + 1}")
            optimizer.step()
            step += 1
            item = {
                "step": step,
                "loss": float(totals["loss"]) / per_update,
                "grad_norm": float(grad_norm),
                "final_bleu": float(final_bleu[first : first + per_update].mean()),
                "eos_rate": float(eos_rate[first : first + per_update].mean()),
                "rollout_frames": float(lengths[first : first + per_update].float().mean()),
                "forward_frames": forward_total / per_update,
                "lr": args.lr,
                "sec_per_update": time.time() - update_started,
                "window_generate_s": generated,
                "window_gen_batches": len(batches),
                "window_move_s": moved,
                "window_reward_wait_s": rewarded - generated,
                "window_old_logprob_s": scored - rewarded,
            }
            terms = []
            if old_logprobs is not None:
                tokens = float(totals["tokens"])
                item["ratio_mean"] = float(totals["ratio_sum"]) / tokens
                item["clip_fraction"] = float(totals["clipped"]) / tokens
                item["ratio_max_deviation"] = float(totals["max_deviation"])
                terms.append(f"ratio={item['ratio_mean']:.4f} clip={item['clip_fraction']:.4f}")
            if teacher is not None:  # mean KL a token, unweighted
                item["kl_text"] = float(totals["kl_text_sum"]) / max(1.0, float(totals["kl_text_tokens"]))
                item["kl_audio"] = float(totals["kl_audio_sum"]) / max(1.0, float(totals["kl_audio_tokens"]))
                terms.append(f"kl_text={item['kl_text']:.4f} kl_audio={item['kl_audio']:.4f}")
            append_log(log_path, item)
            print(f"update={step} loss={item['loss']:.1f} bleu={item['final_bleu']:.1f} "
                  f"{' '.join(terms)} grad_norm={item['grad_norm']:.2f} "
                  f"s={item['sec_per_update']:.1f}", flush=True)
            if step % args.save_every == 0:
                writer.save(step)
            if step % args.eval_every == 0:
                evaluate(step)

    writer.save(step)
    if step not in validated:
        evaluate(step)
    writer.wait()
    pool.shutdown()
    print(f"Saved the final GRPO checkpoint at update {step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()
