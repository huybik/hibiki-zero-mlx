#!/usr/bin/env python
"""GRPO for the full model (Hibiki-Zero, arXiv 2602.11072v2, §3.3; plan in docs/grpo_plan.md).

Every --tau updates, π_old becomes the current policy:
1. a bf16 copy of it samples --group rollouts of --frames frames for each stream of the window's
   updates;
2. the rollouts get BLEU process rewards and group-normalized advantages (`rewards.py`) in CPU
   workers, while the GPU keeps generating;
3. the policy itself scores them, giving π_old's logprobs. It runs the update's own forward,
   gradients enabled and the result detached, because a no-grad forward picks other bf16 kernels.
   The first update's ratio is then exactly 1 (MPS: 1.000000 against a no-grad pass's 1.0012,
   with 6 tokens clipped).
Each update then takes --streams streams' rollouts, in micro-batches. Its loss is the clipped
ratio π_θ/π_old of every text and target-audio token times its frame's advantage, weighted
--text-weight for text against 1 for each audio codebook, summed over frames and averaged over
rollouts.

Every --eval-every updates, and at the start, the run translates --eval-pairs free-running, as SFT
validation does. It promotes the checkpoint with the best chrF on the first set among those that
pass the health gates and end no later than the start (that set's median end offset).

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
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.finetune import common, rewards, streams
from training.finetune.checkpoints import (
    BEST_METRIC,
    CheckpointWriter,
    atomic_write_text,
    checkpoint_pairs,
    clean_incomplete_checkpoints,
    keep_log_lines,
    load_best_state,
    load_newest_pair,
    promote_best,
)
from training.finetune.hibiki_helpers import get_lmgen
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

# A resumed run must match these run_config.json entries.
RESUME_KEYS = (
    "cache_dir", "config_path", "model_weight", "mimi_weight", "tokenizer", "max_frames",
    "stream_seconds", "frames", "streams", "group", "updates", "tau", "lr", "adam_beta1",
    "adam_beta2", "weight_decay", "grad_clip", "epsilon", "alpha", "text_weight", "temperature",
    "top_k", "eval_pairs", "eval_batch_size", "eval_every", "save_every", "keep_checkpoints",
    "seed", "train_rows", "train_streams", "eval_rows",
)
EVAL_PAIRS = [
    Path("data/eval/fleurs-vi/dev.jsonl"),
    Path("data/eval/en-vi-val/cv27.jsonl"),
    Path("data/eval/en-vi-val/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-noisy/cv27.jsonl"),
    Path("data/eval/en-vi-val-noisy/libriheavy.jsonl"),
]


def rollout(lm_gen: Any, sources: torch.Tensor, silence: torch.Tensor) -> torch.Tensor:
    """Sample one rollout per row of `sources` (source codes [B, Q, T], training frames):
    text and audio tokens [B, 1 + dep_q, T] in the same frames."""
    lm = lm_gen.lm_model
    batch, _, frames = sources.shape
    audio_delays = lm.delays[1 : 1 + lm.dep_q]
    steps = frames + max(audio_delays)
    # Step s reads source frame s - 1 (step 0's input is never read); past the stream, silence.
    tail = silence[None, :, None].expand(batch, -1, steps - frames - 1)
    inputs = torch.cat([sources[:, :, :1], sources, tail], dim=-1)
    texts: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    # CUDA graphs reuse their output buffers, so each step's tokens are copied out.
    lm_gen.on_text_hook = lambda tokens: texts.append(tokens.clone())
    lm_gen.on_audio_hook = lambda tokens: audios.append(tokens.clone())
    with torch.no_grad(), lm_gen.streaming(batch):
        for step in range(steps):
            lm_gen.step(inputs[:, :, step : step + 1])
    text = torch.stack(texts[:frames], dim=-1)  # [B, T]: text frame s is sampled at step s
    audio = torch.stack(audios, dim=-1)  # [B, dep_q, steps]: codebook k's frame s - delay_k
    audio = torch.stack(
        [audio[:, k, delay : delay + frames] for k, delay in enumerate(audio_delays)], dim=1
    )
    return torch.cat([text[:, None], audio], dim=1)


def token_logprobs(
    lm: Any, codes: torch.Tensor, condition_tensors: Any | None, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logprobs at `temperature` of the text and target audio tokens of `codes` (the cache
    layout [B, 1 + 2 * dep_q, T]), [B, 1 + dep_q, T] float32, and where they are defined."""
    output = lm(codes, condition_tensors=condition_tensors)
    targets = codes[:, : 1 + lm.dep_q]
    logprobs = torch.zeros(targets.shape, dtype=torch.float32, device=codes.device)
    for rows, logits, mask in (
        (slice(0, 1), output.text_logits, output.text_mask),
        (slice(1, 1 + lm.dep_q), output.logits, output.mask),
    ):
        scores = torch.log_softmax(logits[mask].float() / temperature, dim=-1)
        logprobs[:, rows][mask] = scores.gather(-1, targets[:, rows][mask][:, None])[:, 0]
    return logprobs, torch.cat([output.text_mask, output.mask], dim=1)


def grpo_loss(
    lm: Any,
    codes: torch.Tensor,
    condition_tensors: Any | None,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    stream_weights: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The negated clipped objective summed over a micro-batch's tokens (Eq. 7-8), with the sums
    behind its ratio statistics. `advantages` is [B, T], `stream_weights` [1 + dep_q]."""
    logprobs, mask = token_logprobs(lm, codes, condition_tensors, temperature)
    ratio = torch.exp(logprobs - old_logprobs)
    advantage = advantages[:, None, :]
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    objective = (objective * stream_weights[None, :, None] * mask).sum()
    deviation = (ratio.detach()[mask] - 1).abs()
    stats = {
        "ratio_sum": ratio.detach()[mask].sum(),
        "clipped": (deviation > epsilon).sum(),
        "tokens": mask.sum(),
        "max_deviation": deviation.max(),
    }
    return -objective, stats


def stream_sentences(
    stream: streams.Stream, rows: list[streams.Row], tokenizer: Any, pad_id: int
) -> list[rewards.Sentence]:
    """Each row of `stream` as a sentence: its Vietnamese span and its English reference, the
    decoded text stream of its cache row."""
    eos_id = int(tokenizer.eos_id())
    sentences = []
    for piece in stream.pieces:
        row = rows[piece.row]
        tokens = row.codes[0, row.delay_frames : row.en_frames].tolist()
        words = [token for token in tokens if token not in (pad_id, eos_id)]
        sentences.append(
            rewards.Sentence(piece.vi_start, piece.vi_start + row.vi_frames, tokenizer.decode(words))
        )
    return sentences


def stream_source(
    dataset: streams.StreamDataset, index: int, silence: torch.Tensor
) -> torch.Tensor:
    """A stream's Vietnamese codes [Q, T], silence past its end."""
    codes = dataset[index]["codes"]
    source = silence.long().clone()
    source[:, : codes.shape[1]] = codes[1 + dataset.layout.dep_q :]
    return source


_TOKENIZER: Any = None


def _start_reward_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    import sentencepiece

    _TOKENIZER = sentencepiece.SentencePieceProcessor(tokenizer_path)


def _group_rewards(
    tokens: np.ndarray, sentences: list[rewards.Sentence], skip: set[int], alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rewards.group_advantages(tokens, sentences, skip, _TOKENIZER.decode, alpha)


def reward_pool(workers: int, tokenizer: Path) -> ProcessPoolExecutor:
    """Reward workers, forked now, before CUDA or any thread starts: a fork pool starts its
    workers at the first submit."""
    pool = ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_start_reward_worker,
        initargs=(str(tokenizer),),
    )
    pool.submit(int).result()
    return pool


def bf16_copy(info: Any, lm: Any, device: torch.device) -> Any:
    """The policy's weights in a bf16 generation model, as SFT validation builds it."""
    generator = info.get_moshi(device=device, dtype=torch.bfloat16, load_weight=False)
    generator.load_state_dict(lm.state_dict())
    generator.eval()
    return generator


def sample_window(
    generator: Any,
    info: Any,
    sources: torch.Tensor,
    gen_batch: int,
    silence_frame: torch.Tensor,
    seed: int,
    temperature: float,
    top_k: int,
    on_batch: Any,
) -> torch.Tensor:
    """Rollouts [N, 1 + dep_q, T] int32 on the CPU for `sources` [N, Q, T], in batches of
    `gen_batch`, each seeded from `seed`; `on_batch(start, end, rollouts)` after each."""
    count, _, frames = sources.shape
    outputs = torch.empty((count, 1 + generator.dep_q, frames), dtype=torch.int32)
    steps = frames + max(generator.delays)
    for index, start in enumerate(range(0, count, gen_batch)):
        common.seed_all(seed + index)
        batch = sources[start : start + gen_batch].to(silence_frame.device)
        lm_gen = get_lmgen(
            generator, info, batch_size=len(batch), temp=temperature, temp_text=temperature,
            top_k=top_k, top_k_text=top_k,
        )
        common.set_attention_context(generator.transformer, min(steps, generator.context))
        try:
            sampled = rollout(lm_gen, batch, silence_frame)
        finally:
            common.set_attention_context(generator.transformer, generator.context)
        outputs[start : start + len(batch)] = sampled.to(device="cpu", dtype=torch.int32)
        on_batch(start, start + len(batch), outputs[start : start + len(batch)])
        del lm_gen, sampled
    return outputs


def parse_args() -> argparse.Namespace:
    # No abbreviations: h100.sh must be able to reject --out-dir.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True, help="The SFT checkpoint.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument(
        "--stream-seconds", type=float, nargs=2, default=[30.0, 75.0], metavar=("MIN", "MAX"),
        help="The paper's RL inputs are 30-75 s utterances.",
    )
    parser.add_argument("--frames", type=int, default=1200, help="Rollout length T.")
    parser.add_argument("--streams", type=int, default=32, help="Streams per update.")
    parser.add_argument("--group", type=int, default=4, help="Rollouts per stream (G).")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--tau", type=int, default=20, help="Updates per π_old.")
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.2, help="Ratio clip.")
    parser.add_argument("--alpha", type=float, default=rewards.ALPHA, help="Final BLEU's share.")
    parser.add_argument("--text-weight", type=float, default=100.0, help="c_0 against 1 per codebook.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--gen-batch", type=int, default=128, help="Rollouts generated at once.")
    parser.add_argument("--micro-batch", type=int, default=3, help="Rollouts per forward.")
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--eval-pairs", type=Path, nargs="+", default=EVAL_PAIRS)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("frames", "streams", "group", "updates", "tau", "gen_batch", "micro_batch",
                 "reward_workers", "eval_batch_size", "eval_every", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.group < 2:
        raise ValueError("--group needs at least 2 rollouts to normalize")
    if args.save_every % args.tau or args.eval_every % args.save_every:
        raise ValueError("--save-every must be a multiple of --tau, --eval-every of --save-every")
    if args.gen_batch % args.group:
        raise ValueError("--gen-batch must hold whole groups")
    if round(args.stream_seconds[1] * streams.FRAME_RATE) >= args.frames:
        raise ValueError("--frames must exceed the longest stream")
    if os.environ.get("NO_TORCH_COMPILE"):
        raise ValueError("Full-model training requires torch compile; unset NO_TORCH_COMPILE")


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    validate_args(args)
    pool = reward_pool(args.reward_workers, args.tokenizer)
    device = common.check_device("cuda")
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "start checkpoint")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    args.eval_pairs = [require_file(path, "evaluation pairs") for path in args.eval_pairs]
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_incomplete_checkpoints(out_dir)
    resuming = bool(checkpoint_pairs(out_dir))
    if not resuming:
        for name in ("train_log.jsonl", "val_log.jsonl", "best.json"):
            (out_dir / name).unlink(missing_ok=True)
        shutil.rmtree(out_dir / "eval", ignore_errors=True)
    best_state = load_best_state(out_dir)

    info = common.load_checkpoint_info(args)
    tokenizer = info.get_text_tokenizer()
    stream_frames = tuple(round(seconds * streams.FRAME_RATE) for seconds in args.stream_seconds)
    silence = streams.silence_track(info.get_mimi(device="cpu"), args.frames)
    rows, layout = streams.load_rows(cache_dirs, args.max_frames)
    dataset = streams.StreamDataset(rows, layout, silence, stream_frames, args.seed)
    if len(dataset) < args.updates * args.streams:
        raise ValueError(f"{len(dataset)} streams, fewer than {args.updates} updates need")
    skip = {layout.pad_id, int(tokenizer.eos_id())}
    print(f"Planned {len(dataset)} streams from {len(rows)} rows; {args.updates} updates use "
          f"{args.updates * args.streams}")

    generation = common.GenerationConfig(text_only=True)
    rows_by_set = common.read_eval_sets(args.eval_pairs)
    eval_sets = list(rows_by_set)
    eval_rows = [row for set_rows in rows_by_set.values() for row in set_rows]
    common.validate_eval_rows(eval_rows, generation.source_column, generation.reference_column,
                              generation.id_column, duration_column="duration_s")

    lm = info.get_moshi(device=device, dtype=torch.float32, load_weight=False,
                        lm_kwargs_overrides={"gradient_checkpointing": False})
    common.load_model(lm, args.model_weight, torch.float32)
    lm.train()
    common.enable_full_finetune(lm)
    params = common.trainable_parameters(lm)
    mimi = info.get_mimi(device=device)
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay, fused=True,
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    stream_weights = torch.ones(1 + lm.dep_q, device=device)
    stream_weights[0] = args.text_weight
    silence_frame = silence[:, -1].long().to(device)

    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    step = load_newest_pair(lm, optimizer, out_dir, device, torch.float32) if resuming else 0
    logged = keep_log_lines(val_log_path, step)
    validated = {int(item["step"]) for item in logged}
    keep_log_lines(log_path, step)
    if step % args.tau:
        raise RuntimeError(f"Resumed step {step} is not on a --tau boundary")

    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(
        cache_dir=[str(path) for path in args.cache_dir],
        eval_pairs=[str(path) for path in args.eval_pairs],
        train_rows=len(rows), train_streams=len(dataset), eval_rows=len(eval_rows),
        total_steps=args.updates, val_every=args.eval_every, best_metric=BEST_METRIC,
    )
    config_path = out_dir / "run_config.json"
    if not resuming:
        atomic_write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", config_path)
    else:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in RESUME_KEYS:
            if previous.get(key) != run_config.get(key):
                raise RuntimeError(f"Same-run resume contract changed: {key}")
    writer = CheckpointWriter(lm, optimizer, args, out_dir)
    # The start's end offset on the first set: a promoted checkpoint may not end later.
    start_offset = next(
        (item["free_running"][eval_sets[0]]["median_end_offset_s"] for item in logged
         if int(item["step"]) == 0), None,
    )

    def evaluate(at: int) -> None:
        nonlocal best_state, start_offset
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        eval_dir = out_dir / "eval" / f"step{at:06d}"
        generator = bf16_copy(info, lm, device)
        records = common.run_greedy_eval(eval_rows, generation, args.eval_batch_size, mimi,
                                         generator, tokenizer, info, eval_dir, args.seed)
        del generator
        gc.collect()  # moshi's streaming state holds the generator in reference cycles
        torch.cuda.empty_cache()
        common.write_predictions(eval_dir / "predictions.csv", records)
        free_running = common.score_sets(records, eval_sets)
        healthy = all(metrics["health_eligible"] for metrics in free_running.values())
        first = free_running[eval_sets[0]]
        for name, metrics in free_running.items():
            print(f"val step={at} {name} {common.score_summary(metrics)}")
        if at == 0:
            start_offset = first["median_end_offset_s"]
        elif healthy and first["median_end_offset_s"] <= start_offset:
            writer.wait()
            best_state = promote_best(at, float(first["chrf"]), out_dir, best_state)
        seconds = time.time() - started
        item = {"step": at, "seconds": seconds, "free_running": free_running}
        with val_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        validated.add(at)
        print(f"val step={at} took {seconds:.0f} s")

    if step % args.eval_every == 0 and step not in validated:
        evaluate(step)
    if start_offset is None:
        raise RuntimeError("The run's step-0 evaluation is missing from val_log.jsonl")

    per_update = args.streams * args.group
    while step < args.updates:
        window = min(args.tau, args.updates - step)
        window_started = time.time()
        # 1. π_old's rollouts for the window, rewards in the workers as each batch lands.
        optimizer.zero_grad(set_to_none=True)
        stream_ids = [index for update in range(step, step + window)
                      for index in range(update * args.streams, (update + 1) * args.streams)]
        sources = torch.stack([stream_source(dataset, index, silence)
                               for index in stream_ids for _ in range(args.group)])
        jobs: dict[int, Future] = {}

        def submit(start: int, end: int, rollouts: torch.Tensor) -> None:
            for offset in range(0, end - start, args.group):
                group = (start + offset) // args.group
                stream = dataset.streams[stream_ids[group]]
                sentences = stream_sentences(stream, rows, tokenizer, layout.pad_id)
                tokens = rollouts[offset : offset + args.group, 0].numpy().copy()
                jobs[group] = pool.submit(_group_rewards, tokens, sentences, skip, args.alpha)

        generator = bf16_copy(info, lm, device)
        rollouts = sample_window(generator, info, sources, args.gen_batch, silence_frame,
                                 args.seed * 1_000_003 + step, args.temperature, args.top_k, submit)
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        generated = time.time() - window_started
        results = [jobs[group].result() for group in range(len(stream_ids))]
        advantages = torch.from_numpy(np.concatenate([result[2] for result in results])).float()
        final_bleu = np.concatenate([result[1][:, -1] for result in results])
        codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
        rewarded = time.time() - window_started

        # 2. π_old's logprobs: the policy before this window's first update, through the update's
        # own forward (gradients enabled, never backpropagated).
        old_logprobs = torch.empty((len(codes), 1 + lm.dep_q, args.frames), dtype=torch.float32)
        with autocast:
            for start in range(0, len(codes), args.micro_batch):
                batch = codes[start : start + args.micro_batch].to(device=device, dtype=torch.long)
                condition = common.batch_condition_tensors(lm, info.model_type, len(batch))
                logprobs, _ = token_logprobs(lm, batch, condition, args.temperature)
                old_logprobs[start : start + len(batch)] = logprobs.detach().cpu()
                del logprobs
        scored = time.time() - window_started

        # 3. The window's updates.
        for update in range(window):
            update_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            first = update * per_update
            totals = {"loss": 0.0, "ratio_sum": 0, "clipped": 0, "tokens": 0, "max_deviation": 0}
            for start in range(first, first + per_update, args.micro_batch):
                end = min(start + args.micro_batch, first + per_update)
                batch = codes[start:end].to(device=device, dtype=torch.long)
                condition = common.batch_condition_tensors(lm, info.model_type, end - start)
                with autocast:
                    loss, stats = grpo_loss(
                        lm, batch, condition, old_logprobs[start:end].to(device),
                        advantages[start:end].to(device), stream_weights, args.epsilon,
                        args.temperature)
                (loss / per_update).backward()
                totals["loss"] += loss.detach()
                totals["max_deviation"] = torch.maximum(
                    torch.as_tensor(totals["max_deviation"], device=device), stats.pop("max_deviation"))
                for key, value in stats.items():
                    totals[key] += value
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip or math.inf)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"Non-finite gradient norm at update {step + 1}")
            optimizer.step()
            step += 1
            tokens = float(totals["tokens"])
            item = {
                "step": step,
                "loss": float(totals["loss"]) / per_update,
                "ratio_mean": float(totals["ratio_sum"]) / tokens,
                "clip_fraction": float(totals["clipped"]) / tokens,
                "ratio_max_deviation": float(totals["max_deviation"]),
                "grad_norm": float(grad_norm),
                "final_bleu": float(final_bleu[first : first + per_update].mean()),
                "lr": args.lr,
                "sec_per_update": time.time() - update_started,
                "window_generate_s": generated,
                "window_reward_wait_s": rewarded - generated,
                "window_old_logprob_s": scored - rewarded,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print(f"update={step} loss={item['loss']:.1f} bleu={item['final_bleu']:.1f} "
                  f"ratio={item['ratio_mean']:.4f} clip={item['clip_fraction']:.4f} "
                  f"grad_norm={item['grad_norm']:.2f} s={item['sec_per_update']:.1f}", flush=True)
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
