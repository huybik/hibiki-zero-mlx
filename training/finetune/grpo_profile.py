#!/usr/bin/env python
"""Profile GRPO's stages on the GPU before a run (docs/grpo_plan.md, phases 1 and 2).

The policy's fp32 weights and AdamW's two states are held as in training (in pinned host memory
while a window generates, `grpo.HostOffload`; with --park-states AdamW's also while the old-logprob
pass and the update's micro-batches run, the round trip untimed and the optimizer's step, which
holds them, left out of the peaks), and each stage runs on real streams from the start checkpoint:
- rollout generation at each --gen-batch rows of --frames: time per step and peak memory, timed
  on a second call so moshi's compilation stays out (each batch's CUDA graph capture stays in,
  as in a run);
- the rewards of the rollouts, serially and in --reward-workers worker processes;
- π_old's logprob pass (the update's forward, gradients on) and the update's forward and backward
  at each --micro-batch, and the first update's largest ratio deviation, which should be 0. With
  --teacher-weight the update includes the teacher's forward and the distillation term (the
  teacher parked on the host while generating, as in a run); with --reward-coef 0 a run skips
  the logprob pass, so `window` counts none;
- with a teacher, `teacher_leak`: an upper bound on how much the teacher's distributions follow
  what the policy sampled after the scored token, for text and each codebook, against the start's
  `kl_text` and `kl_audio`.
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
    parser.add_argument("--pair-cache-dir", type=Path, nargs="+", default=[], help="The other direction's caches of the same rows, read only for their text, which times each row's source (streams.load_rows); not needed for caches given with --cache-dir.")
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
    parser.add_argument("--park-states", action="store_true", help="As a run's --park-states.")
    parser.add_argument("--reward-workers", type=int, default=8)
    grpo.add_objective_args(parser)
    grpo.add_room_args(parser)
    return parser.parse_args()


def timed(function, repeats: int = 1) -> float:
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats


def profile_generation(
    args, info, lm, optimizer, parked, sources, silence_frame, report: dict, write
) -> torch.Tensor:
    """Time rollout generation at each --gen-batch on a bf16 copy of `lm`, the policy, its
    optimizer states and `parked` (a teacher's parameters) in host memory as a run parks them, and
    return the rollouts of every source at the largest size that fits, with them back on the GPU."""
    generator = common.bf16_copy(info, lm, lm.device)
    offload = grpo.HostOffload()
    offload.to_host(grpo.trainer_tensors(lm, optimizer) + parked)
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


def window_estimate(args, dataset, rollouts, stream_ids, delays, extra, eos_id, report: dict) -> dict:
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
            "logprob_seconds_per_update": (per_update / measured["logprob_frames_per_s"]
                                           if args.reward_coef else 0.0),
            "update_seconds_per_update": per_update / measured["update_frames_per_s"]}
    estimate["rollouts"] = len(window)
    print(json.dumps({"window": estimate}, indent=2), flush=True)
    return estimate


def teacher_leak(teacher, teach, codes, keep, shifts: dict[int, int], device, rows: int = 8) -> dict:
    """An upper bound on how much the teacher's distributions follow what the policy sampled after
    the token they score. Where the teacher delays codebook j less than the policy, at a token it
    reads frames of j the policy sampled later, conditioned on that token. Shifting each such
    codebook `shifts[j]` (the policy's delay minus the teacher's) frames later in the teacher's
    input leaves it only frames the policy had sampled by then. Over the first `rows` rollouts, the
    teacher's mean logprob of the policy's kept tokens with the rollout as sampled, minus with the
    shifted one, is that bias plus what the misaligned codebooks cost the teacher (taken as never
    negative): an upper bound, for text and each codebook. Not covered: the 3B's depformer scores
    codebook 1 of frame f after the text and codebook 0 of frame f + 2, which a heads student
    samples later."""
    own = codes[:rows]
    shifted = own.clone()
    for j, shift in shifts.items():
        shifted[:, 1 + j, shift:] = own[:, 1 + j, :-shift]
        shifted[:, 1 + j, :shift] = own[:, 1 + j, :1]  # the stream's first frame stands in before it
    codebooks = codes.shape[1] // 2
    means = []
    for batch in (own, shifted):
        sums, counts = torch.zeros(1 + codebooks), torch.zeros(1 + codebooks)
        for start in range(0, len(batch), 2):  # two rollouts a forward, as the smallest micro-batch
            part = batch[start : start + 2].to(device=device, dtype=torch.long)
            kept = keep[start : start + 2].to(device)
            output = grpo.teacher_forward(teacher, teach, part)
            text_valid = output.text_mask & kept[:, :1]
            audio_valid = output.mask & kept[:, 1:]
            text = common.token_ce(output.text_logits, part[:, :1], text_valid)
            audio = common.token_ce(output.logits, part[:, 1 : 1 + codebooks], audio_valid)
            sums -= torch.cat([text, audio], dim=1).sum(dim=(0, 2)).cpu()
            counts += torch.cat([text_valid, audio_valid], dim=1).sum(dim=(0, 2)).cpu()
        means.append(sums / counts.clamp(min=1))
    bound = (means[0] - means[1]).tolist()
    return {"rollouts": len(own), "shifted_codebooks": shifts, "text_bound_nats": bound[0],
            "audio_bound_nats": bound[1:]}


def main() -> None:
    grpo.set_alloc_conf()  # a run's allocators
    args = parse_args()
    grpo.validate_objective_args(args)
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
    pair_dirs = [require_dir(path, "pair code cache directory") for path in args.pair_cache_dir]
    rows, layout = streams.load_rows(cache_dirs, args.max_frames, args.tokenizer, pair_dirs)
    codebooks = int(info.lm_config["dep_q"])  # a stream's, fewer for a phone student
    teach = grpo.teacher_info(args, codebooks)
    dataset = grpo.stream_dataset(args, info, rows, layout, codebooks, info.get_mimi(device=device))
    count = max(args.gen_batch)
    stream_ids = [int(i) for i in grpo.stream_order(len(dataset), args.seed)[: count // args.group]]
    sources = torch.stack([grpo.stream_source(dataset, index, codebooks)
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
    teacher = None
    if teach is not None:
        teacher = common.load_teacher(teach, args.teacher_weight, device, codebooks)
    report["resident_gib"] = torch.cuda.memory_allocated() / 2**30
    silence_frame = dataset.silence[:codebooks, -1].long().to(device)

    # Phase 1: generation.
    parked = [] if teacher is None else list(teacher.parameters())
    rollouts = profile_generation(args, info, lm, optimizer, parked, sources, silence_frame,
                                  report, write)

    # Phase 1: rewards.
    eos_id, pad_id = int(tokenizer.eos_id()), int(tokenizer.pad_id())
    jobs = []
    for group, index in enumerate(stream_ids):
        sentences = grpo.stream_sentences(dataset.streams[index], rows, tokenizer)
        tokens = rollouts[group * args.group : (group + 1) * args.group, 0].numpy().copy()
        jobs.append((tokens, sentences))
    start = time.time()
    scale = args.advantage == "zscore"
    results = [rewards.group_advantages(tokens, sentences, eos_id, pad_id, tokenizer.decode,
                                        scale=scale)
               for tokens, sentences in jobs]
    serial = time.time() - start
    start = time.time()
    [future.result() for future in [pool.submit(grpo._group_rewards, tokens, sentences,
                                                rewards.ALPHA, scale)
                                    for tokens, sentences in jobs]]
    parallel = time.time() - start
    report["rewards"] = {"groups": len(jobs), "serial_s": serial, "parallel_s": parallel}
    print(f"rewards of {len(jobs)} groups: {serial:.1f} s serial, {parallel:.1f} s in "
          f"{args.reward_workers} workers", flush=True)
    write(report)

    # Phase 2: π_old's logprob pass and the update's forward and backward.
    codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
    delays = grpo.scored_delays(lm, teacher)
    keep = grpo.kept_tokens(rollouts, torch.full((len(rollouts),), args.frames), delays, eos_id)
    advantages = torch.from_numpy(np.concatenate([r[2] for r in results])).float()
    weights = torch.ones(1 + lm.dep_q, device=device)
    weights[0] = grpo.TEXT_WEIGHT
    objective = grpo.Objective(weights, grpo.EPSILON, grpo.TEMPERATURE, args.reward_coef,
                               args.kl_coef or 0.0)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    if args.park_states:
        grpo.park_states(grpo.HostOffload(), optimizer)  # as a run's old-logprob pass and micro-batches find them
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

        def update_loss():  # as an update of grpo.py runs it, the teacher's forward included
            teacher_output = None if teacher is None else grpo.teacher_forward(teacher, teach, batch)
            with autocast:
                return grpo.update_loss(lm, batch, condition, keep[:micro].to(device), objective,
                                        old, advantages[:micro].to(device), teacher_output)

        def update():
            loss, _ = update_loss()
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
        stats = update_loss()[1]  # its loss's graph would count in the next size's peak
        lm.zero_grad(set_to_none=True)
        report[f"micro_batch_{micro}"] = {
            "logprob_frames_per_s": frames / score_s,
            "update_frames_per_s": frames / update_s, "peak_gib": peak,
            "sampled_logprob_mean": float(old[mask].mean())}
        if "max_deviation" in stats:
            report[f"micro_batch_{micro}"]["first_update_ratio_max_deviation"] = float(stats["max_deviation"])
        if teacher is not None:  # the start's mean KL a token, unweighted
            for part in ("text", "audio"):
                report[f"micro_batch_{micro}"][f"kl_{part}"] = float(
                    stats[f"kl_{part}_sum"] / stats[f"kl_{part}_tokens"].clamp(min=1))
        print(f"micro batch {micro}: logprobs {frames / score_s:,.0f} frames/s, update "
              f"{frames / update_s:,.0f} frames/s, peak {peak:.1f} GiB", flush=True)
        write(report)

    gc.collect()
    torch.cuda.empty_cache()
    if teacher is not None:
        policy, teaching = lm.delays[1 : 1 + lm.dep_q], teacher.delays[1 : 1 + lm.dep_q]
        shifts = {j: policy[j] - teaching[j] for j in range(lm.dep_q) if policy[j] > teaching[j]}
        if shifts:
            report["teacher_leak"] = teacher_leak(teacher, teach, codes, keep, shifts, device)
            print(f"teacher leak: {json.dumps(report['teacher_leak'])}", flush=True)
            write(report)

    generation_extra = max(lm.delays[: 1 + lm.dep_q])  # what generation steps past a rollout
    report["window"] = window_estimate(args, dataset, rollouts, stream_ids, delays,
                                       generation_extra, eos_id, report)
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
