#!/usr/bin/env python
"""Profile GRPO's stages on the GPU before a run (docs/grpo_plan.md, phases 1 and 2).

The policy's fp32 weights and AdamW's two states are held as in training (in pinned host memory
while a window generates, `grpo.HostOffload`), and each stage runs on real streams from the start
checkpoint:
- rollout generation at each --gen-batch rows of --frames: time per step and peak memory, timed
  on a second call so moshi's compilation stays out (each batch's CUDA graph capture stays in,
  as in a run);
- the rewards of the rollouts, serially and in --reward-workers worker processes;
- π_old's logprob pass (the update's forward, gradients on) and the update's forward and backward
  at each --micro-batch, and the first update's largest ratio deviation, which should be 0.
A size that runs out of memory is recorded and skipped. Every stage runs at --frames, a run's
longest rollout and so its peak memory; `window` then gives what a run's first window costs at
its own lengths (`grpo.rollout_frames`, `grpo.micro_frames`) for each --gen-batch's
`--gen-frames` (rows times --frames, `grpo.generation_batches`), from a step's time fitted as a
line in its rows and the measured time per update frame. <out-dir>/profile.json is rewritten after each stage,
with a few streams' references, rollouts and rewards to read. Sample GPU utilization beside it
(nvidia-smi every 0.5 s).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from moshi.modules.streaming import StreamingModule

from training.finetune import common, grpo, rewards, streams
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
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
    parser.add_argument("--tokenizer", type=Path, help="Default: the one the model weight is on (its text_tokenizer metadata), in weights/.")
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument("--stream-seconds", type=float, nargs=2, default=[30.0, 75.0])
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--tail-seconds", type=float, default=common.GenerationConfig.tail_s)
    parser.add_argument("--streams", type=int, default=32, help="Streams per update.")
    parser.add_argument("--tau", type=int, default=20, help="Updates per window.")
    parser.add_argument("--gen-batch", type=int, nargs="+", default=[192, 384, 512])
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


def profile_generation(
    args, info, lm, optimizer, sources, silence_frame, report: dict, write
) -> torch.Tensor:
    """Time rollout generation at each --gen-batch on a bf16 copy of `lm`, the policy and its
    optimizer states in host memory as a run parks them, and return the rollouts of every source
    at the largest size that fits, with them back on the GPU."""
    generator = common.bf16_copy(info, lm, lm.device)
    offload = grpo.HostOffload()
    offload.to_host(grpo.trainer_tensors(lm, optimizer))
    report["resident_while_generating_gib"] = torch.cuda.memory_allocated() / 2**30

    def generate(count: int, gen_batch: int) -> torch.Tensor:
        lengths = torch.full((count,), args.frames)
        batches = grpo.generation_batches(lengths, gen_batch * args.frames, args.group)
        return grpo.sample_window(
            generator, info, sources[:count], lengths, batches, silence_frame, (args.seed,),
            grpo.TEMPERATURE, grpo.TOP_K, lambda *_: None)

    fitting = []
    for gen_batch in args.gen_batch:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            generate(gen_batch, gen_batch)  # compiles and captures
            seconds = timed(lambda: generate(gen_batch, gen_batch))
        except torch.cuda.OutOfMemoryError:
            # Running out while it builds its streaming state leaves some modules streaming, the
            # detached depformer among them, which moshi's own _stop_streaming skips.
            for module in generator.modules():
                if isinstance(module, StreamingModule):
                    module._streaming_state = None
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
        write(report)
    if not fitting:
        raise RuntimeError("No --gen-batch fits")
    gc.collect()
    torch.cuda.empty_cache()
    rollouts = generate(len(sources), max(fitting))
    del generator
    gc.collect()  # the generator's 6 GiB must not count in phase 2's peaks
    torch.cuda.empty_cache()
    offload.to_device()
    return rollouts


def window_estimate(args, dataset, rollouts, stream_ids, delays, eos_id, report: dict) -> dict:
    """What a run's first window (--tau updates of --streams streams) costs at its own lengths.
    Generation: its rollouts in `grpo.generation_batches` of each --gen-batch's --gen-frames, each
    as many steps as its longest rollout, a step's time a line in its rows fitted to the measured
    sizes. Updates: the forward frames of the profiled rollouts cut to their streams' lengths and
    kept tokens, scaled to the window. Times use this profile's frames per second."""
    tail = round(args.tail_seconds * streams.FRAME_RATE)

    def length(index: int) -> int:
        return grpo.rollout_frames(dataset.streams[index].frames, tail, args.frames)

    order = grpo.stream_order(len(dataset), args.seed)[: args.tau * args.streams]
    window = torch.tensor([length(int(index)) for index in order]).repeat_interleave(args.group)
    estimate: dict = {"streams": len(order), "rollout_frames_mean": float(window.float().mean())}
    extra = max(delays.tolist())
    measured = {gen_batch: report[f"generate_batch_{gen_batch}"]["ms_per_step"]
                for gen_batch in args.gen_batch
                if isinstance(report.get(f"generate_batch_{gen_batch}"), dict)}
    if len(measured) >= 2:
        slope, intercept = np.polyfit(list(measured), list(measured.values()), 1)
        estimate["ms_per_step_fit"] = {"intercept": float(intercept), "per_row": float(slope)}
        for gen_batch in measured:
            batches = grpo.generation_batches(window, gen_batch * args.frames, args.group)
            steps = [int(window[rows].max()) + extra for rows in batches]
            ms = sum(count * (intercept + slope * len(rows)) for count, rows in zip(steps, batches))
            estimate[f"generate_batch_{gen_batch}"] = {
                "gen_frames": gen_batch * args.frames, "batches": len(batches),
                "steps": sum(steps), "seconds_per_update": ms / 1000 / args.tau}
    # The profiled rollouts ran to --frames; cut each to its stream's length, as a run does.
    lengths = torch.tensor([length(index) for index in stream_ids]).repeat_interleave(args.group)
    cut = rollouts.clone()
    cut.masked_fill_((torch.arange(args.frames)[None] >= lengths[:, None])[:, None], -1)
    spans = grpo.forward_frames(grpo.kept_tokens(cut, lengths, delays, eos_id), delays)
    estimate["eos_rate"] = float((cut[:, 0] == eos_id).any(dim=1).float().mean())
    for micro in args.micro_batch:
        measured = report.get(f"micro_batch_{micro}")
        if not isinstance(measured, dict):
            continue
        frames = [grpo.micro_frames(spans, start, min(start + micro, len(spans)), args.frames)
                  * (min(start + micro, len(spans)) - start)
                  for start in range(0, len(spans), micro)]
        per_rollout = sum(frames) / len(spans)
        per_update = per_rollout * args.streams * args.group
        estimate[f"micro_batch_{micro}"] = {
            "forward_frames_mean": per_rollout,
            "logprob_seconds_per_update": per_update / measured["logprob_frames_per_s"],
            "update_seconds_per_update": per_update / measured["update_frames_per_s"]}
    estimate["rollouts"] = len(window)
    print(json.dumps({"window": estimate}, indent=2), flush=True)
    return estimate


def main() -> None:
    args = parse_args()
    args.model_weight = require_file(args.model_weight, "start checkpoint")
    args.tokenizer = common.run_tokenizer(args.tokenizer, args.model_weight)
    pool = grpo.reward_pool(args.reward_workers, args.tokenizer)
    device = common.check_device("cuda")
    torch.set_float32_matmul_precision("high")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    info = common.load_checkpoint_info(args)
    tokenizer = info.get_text_tokenizer()
    silence = streams.silence_track(info.get_mimi(device="cpu"), args.frames)
    rows, layout = streams.load_rows(cache_dirs, args.max_frames, args.tokenizer)
    dataset = streams.StreamDataset(
        rows, layout, silence, streams.frame_range(args.stream_seconds), args.seed
    )
    count = max(args.gen_batch)
    stream_ids = [int(i) for i in grpo.stream_order(len(dataset), args.seed)[: count // args.group]]
    sources = torch.stack([grpo.stream_source(dataset, index, silence)
                           for index in stream_ids for _ in range(args.group)])
    report: dict = {"frames": args.frames, "group": args.group,
                    "stream_frames": [dataset.streams[i].frames for i in stream_ids]}

    def write(report: dict) -> None:
        (out_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n")

    lm = common.load_policy(info, device)
    optimizer = torch.optim.AdamW(lm.parameters(), fused=True)  # its states, never stepped
    for param in lm.parameters():
        optimizer.state[param] = {"step": torch.zeros((), device=device),
                                  "exp_avg": torch.zeros_like(param),
                                  "exp_avg_sq": torch.zeros_like(param)}
    report["resident_gib"] = torch.cuda.memory_allocated() / 2**30
    silence_frame = silence[:, -1].long().to(device)

    # Phase 1: generation.
    rollouts = profile_generation(args, info, lm, optimizer, sources, silence_frame, report, write)

    # Phase 1: rewards.
    eos_id, pad_id = int(tokenizer.eos_id()), int(tokenizer.pad_id())
    jobs = []
    for group, index in enumerate(stream_ids):
        sentences = grpo.stream_sentences(dataset.streams[index], rows, tokenizer)
        tokens = rollouts[group * args.group : (group + 1) * args.group, 0].numpy().copy()
        jobs.append((tokens, sentences))
    start = time.time()
    results = [rewards.group_advantages(tokens, sentences, eos_id, pad_id, tokenizer.decode)
               for tokens, sentences in jobs]
    serial = time.time() - start
    start = time.time()
    [future.result() for future in [pool.submit(grpo._group_rewards, tokens, sentences,
                                                rewards.ALPHA) for tokens, sentences in jobs]]
    parallel = time.time() - start
    report["rewards"] = {"groups": len(jobs), "serial_s": serial, "parallel_s": parallel}
    print(f"rewards of {len(jobs)} groups: {serial:.1f} s serial, {parallel:.1f} s in "
          f"{args.reward_workers} workers", flush=True)
    write(report)

    # Phase 2: π_old's logprob pass and the update's forward and backward.
    codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
    delays = torch.tensor(lm.delays[: 1 + lm.dep_q])
    keep = grpo.kept_tokens(rollouts, torch.full((len(rollouts),), args.frames), delays, eos_id)
    advantages = torch.from_numpy(np.concatenate([r[2] for r in results])).float()
    weights = torch.ones(1 + lm.dep_q, device=device)
    weights[0] = grpo.TEXT_WEIGHT
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    for micro in args.micro_batch:
        batch = codes[:micro].to(device=device, dtype=torch.long)
        condition = common.batch_condition_tensors(lm, info.model_type, micro)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def score():
            with autocast:  # the update's forward, gradients enabled, as grpo.py scores π_old
                logprobs, mask = grpo.token_logprobs(lm, batch, condition, grpo.TEMPERATURE)
            return logprobs.detach(), mask

        def update():
            with autocast:
                loss, _ = grpo.grpo_loss(lm, batch, condition, old, advantages[:micro].to(device),
                                         keep[:micro].to(device), weights, grpo.EPSILON,
                                         grpo.TEMPERATURE)
            loss.backward()

        try:
            old, mask = score()  # compiles
            score_s = timed(score, 3)
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
                                      keep[:micro].to(device), weights, grpo.EPSILON,
                                      grpo.TEMPERATURE)
        lm.zero_grad(set_to_none=True)
        report[f"micro_batch_{micro}"] = {
            "first_update_ratio_max_deviation": float(stats["max_deviation"]),
            "logprob_frames_per_s": frames / score_s,
            "update_frames_per_s": frames / update_s, "peak_gib": peak,
            "sampled_logprob_mean": float(old[mask].mean())}
        print(f"micro batch {micro}: logprobs {frames / score_s:,.0f} frames/s, update "
              f"{frames / update_s:,.0f} frames/s, peak {peak:.1f} GiB", flush=True)
        write(report)

    report["window"] = window_estimate(args, dataset, rollouts, stream_ids, delays, eos_id, report)
    write(report)

    report["samples"] = []
    for (tokens, sentences), (frames, group_rewards, _) in list(zip(jobs, results))[:4]:
        report["samples"].append({
            "reference": " ".join(sentence.reference for sentence in sentences),
            "rollouts": [rewards.emitted_texts(row, frames[-1:], eos_id, pad_id,
                                               tokenizer.decode)[0] for row in tokens],
            "final_bleu": group_rewards[:, -1].tolist(),
        })
    report["final_bleu_mean"] = float(np.mean([r[1][:, -1].mean() for r in results]))
    write(report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "stream_frames")},
                     indent=2))
    del optimizer
    pool.shutdown()


if __name__ == "__main__":
    main()
