#!/usr/bin/env python
"""Profile GRPO's stages on the GPU before a run (docs/grpo_plan.md, phases 1 and 2).

The policy's fp32 weights and AdamW's two states stay resident as in training, and each stage
runs on real streams from the start checkpoint:
- rollout generation at each --gen-batch: time per step and peak memory;
- the rewards of the rollouts, serially and in --reward-workers worker processes;
- the no-grad logprob pass and the loss's forward and backward at each --micro-batch.
Results go to <out-dir>/profile.json, with a few streams' references, rollouts and rewards to
read. Sample GPU utilization beside it (nvidia-smi every 0.5 s).
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.finetune import common, grpo, rewards, streams
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    require_dir,
    require_file,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model-weight", type=Path, required=True, help="Start checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument("--stream-seconds", type=float, nargs=2, default=[30.0, 75.0])
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--gen-batch", type=int, nargs="+", default=[64, 128, 192])
    parser.add_argument("--micro-batch", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--reward-workers", type=int, default=8)
    return parser.parse_args()


def timed(function, repeats: int = 1) -> float:
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats


def main() -> None:
    args = parse_args()
    pool = grpo.reward_pool(args.reward_workers, args.tokenizer)
    device = common.check_device("cuda")
    torch.set_float32_matmul_precision("high")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    args.model_weight = require_file(args.model_weight, "start checkpoint")
    info = common.load_checkpoint_info(args)
    tokenizer = info.get_text_tokenizer()
    silence = streams.silence_track(info.get_mimi(device="cpu"), args.frames)
    rows, layout = streams.load_rows(cache_dirs, args.max_frames)
    stream_frames = tuple(round(seconds * streams.FRAME_RATE) for seconds in args.stream_seconds)
    dataset = streams.StreamDataset(rows, layout, silence, stream_frames, args.seed)
    count = max(args.gen_batch)
    stream_ids = list(range(count // args.group))
    sources = torch.stack([grpo.stream_source(dataset, index, silence)
                           for index in stream_ids for _ in range(args.group)])
    report: dict = {"frames": args.frames, "group": args.group,
                    "stream_frames": [dataset.streams[i].frames for i in stream_ids]}

    lm = info.get_moshi(device=device, dtype=torch.float32, load_weight=False,
                        lm_kwargs_overrides={"gradient_checkpointing": False})
    common.load_model(lm, args.model_weight, torch.float32)
    lm.train()
    common.enable_full_finetune(lm)
    adam_states = [torch.zeros_like(param) for param in lm.parameters() for _ in range(2)]
    report["resident_gib"] = torch.cuda.memory_allocated() / 2**30
    silence_frame = silence[:, -1].long().to(device)

    # Phase 1: generation.
    generator = grpo.bf16_copy(info, lm, device)
    fitting = []
    for gen_batch in args.gen_batch:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            seconds = timed(functools.partial(
                grpo.sample_window, generator, info, sources[:gen_batch], gen_batch,
                silence_frame, args.seed, 0.8, 250, lambda *_: None))
        except torch.cuda.OutOfMemoryError:
            report[f"generate_batch_{gen_batch}"] = "out of memory"
            print(f"generate batch {gen_batch}: out of memory", flush=True)
            continue
        fitting.append(gen_batch)
        steps = args.frames + max(generator.delays)
        peak = torch.cuda.max_memory_allocated() / 2**30
        report[f"generate_batch_{gen_batch}"] = {
            "seconds": seconds, "ms_per_step": 1000 * seconds / steps,
            "rollouts_per_s": gen_batch / seconds, "peak_gib": peak}
        print(f"generate batch {gen_batch}: {1000 * seconds / steps:.1f} ms/step, "
              f"{gen_batch / seconds:.2f} rollouts/s, peak {peak:.1f} GiB", flush=True)
    if not fitting:
        raise RuntimeError("No --gen-batch fits")
    gc.collect()
    torch.cuda.empty_cache()
    rollouts = grpo.sample_window(generator, info, sources, max(fitting), silence_frame,
                                  args.seed, 0.8, 250, lambda *_: None)
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 1: rewards.
    skip = {layout.pad_id, int(tokenizer.eos_id())}
    jobs = []
    for group, index in enumerate(stream_ids):
        sentences = grpo.stream_sentences(dataset.streams[index], rows, tokenizer, layout.pad_id)
        tokens = rollouts[group * args.group : (group + 1) * args.group, 0].numpy().copy()
        jobs.append((tokens, sentences))
    start = time.time()
    results = [rewards.group_advantages(tokens, sentences, skip, tokenizer.decode)
               for tokens, sentences in jobs]
    serial = time.time() - start
    start = time.time()
    [future.result() for future in [pool.submit(grpo._group_rewards, tokens, sentences, skip,
                                                rewards.ALPHA) for tokens, sentences in jobs]]
    parallel = time.time() - start
    report["rewards"] = {"groups": len(jobs), "serial_s": serial, "parallel_s": parallel}
    print(f"rewards of {len(jobs)} groups: {serial:.1f} s serial, {parallel:.1f} s in "
          f"{args.reward_workers} workers", flush=True)

    # Phase 2: the no-grad logprob pass and the update's forward and backward.
    codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
    advantages = torch.from_numpy(np.concatenate([r[2] for r in results])).float()
    weights = torch.ones(1 + lm.dep_q, device=device)
    weights[0] = 100.0
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    for micro in args.micro_batch:
        batch = codes[:micro].to(device=device, dtype=torch.long)
        condition = common.batch_condition_tensors(lm, info.model_type, micro)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def score():
            with autocast:  # the update's forward, gradients enabled, as grpo.py scores π_old
                logprobs, mask = grpo.token_logprobs(lm, batch, condition, 0.8)
            return logprobs.detach(), mask

        old, mask = score()  # compiles
        no_grad_s = timed(score, 3)

        def update():
            with autocast:
                loss, _ = grpo.grpo_loss(lm, batch, condition, old, advantages[:micro].to(device),
                                         weights, 0.2, 0.8)
            loss.backward()

        try:
            update()  # compiles
            update_s = timed(update, 3)
        except torch.cuda.OutOfMemoryError:
            lm.zero_grad(set_to_none=True)
            report[f"micro_batch_{micro}"] = "out of memory"
            print(f"micro batch {micro}: out of memory", flush=True)
            continue
        peak = torch.cuda.max_memory_allocated() / 2**30
        lm.zero_grad(set_to_none=True)
        frames = micro * args.frames
        with autocast:
            _, stats = grpo.grpo_loss(lm, batch, condition, old, advantages[:micro].to(device),
                                      weights, 0.2, 0.8)
        lm.zero_grad(set_to_none=True)
        report[f"micro_batch_{micro}"] = {
            "first_update_ratio_max_deviation": float(stats["max_deviation"]),
            "logprob_frames_per_s": frames / no_grad_s,
            "update_frames_per_s": frames / update_s, "peak_gib": peak,
            "sampled_logprob_mean": float(old[mask].mean())}
        print(f"micro batch {micro}: logprobs {frames / no_grad_s:,.0f} frames/s, update "
              f"{frames / update_s:,.0f} frames/s, peak {peak:.1f} GiB", flush=True)

    report["samples"] = []
    for (tokens, sentences), (frames, group_rewards, _) in list(zip(jobs, results))[:4]:
        report["samples"].append({
            "reference": " ".join(sentence.reference for sentence in sentences),
            "rollouts": [rewards.emitted_texts(row, frames[-1:], skip, tokenizer.decode)[0]
                         for row in tokens],
            "final_bleu": group_rewards[:, -1].tolist(),
        })
    report["final_bleu_mean"] = float(np.mean([r[1][:, -1].mean() for r in results]))
    (out_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "stream_frames")},
                     indent=2))
    del adam_states
    pool.shutdown()


if __name__ == "__main__":
    main()
