#!/usr/bin/env python
"""Unified per-stage benchmark for the MLX hibiki runtime.

  python scripts/bench.py --model 3b                 # per-stage ms table + projections
  python scripts/bench.py --model 3b --silence       # silence-in gate (rms/peak)

Splits each frame into mimi encode / LM main transformer / configured audio head /
mimi decode using mx.eval barriers, then reports RT factor, artifact size and a
projected iPhone frame time. The projection is a documented assumption, not a
measurement: --scale is the assumed A18 Pro / M4 Pro GPU throughput ratio for
these kernels (default 0.5, plausible range 0.4-0.6).
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import sphn

from hibiki_mlx import pipeline as pl
from moshi_mlx import models, utils

ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="3b", help="3b or a q4/bf16 Hibiki-Zero model directory")
    p.add_argument("--frames", type=int, default=150, help="timed frames (default 150)")
    p.add_argument("--warmup", type=int, default=15, help="untimed warmup frames (default 15)")
    p.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="assumed iPhone/M4 GPU throughput ratio (default 0.5)",
    )
    p.add_argument(
        "--silence",
        action="store_true",
        help="feed zeros instead of speech and gate rms<0.10, peak<1.1",
    )
    args = p.parse_args()

    mx.random.seed(299792458)
    weights_dir = pl.resolve_weights_dir(args.model)
    model, lm_config, _tok, mimi_enc, mimi_dec = pl.load(weights_dir)
    ct = None
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    other_cb = lm_config.other_codebooks
    n_slices = lm_config.generated_codebooks

    if args.silence:
        in_pcms = np.zeros((1, args.frames * 1920), dtype=np.float32)
        args.warmup = 0
    else:
        in_pcms, _ = sphn.read(str(ROOT / "assets" / "samples" / "leon.wav"), sample_rate=24000)
    total = args.warmup + args.frames
    assert in_pcms.shape[-1] >= total * 1920, "sample too short for --frames"

    gen = models.LmGen(
        model=model,
        max_steps=total + 8,
        text_sampler=utils.Sampler(top_k=25, temp=0.4),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0,
        check=False,
    )

    # Time the selected audio head inside the LM step with eval barriers.
    if model.depformer is not None:
        head_label = f"LM depformer ({n_slices}x)"
        audio_head = model.depformer
    else:
        head_label = f"LM parallel_v1 ({lm_config.parallel_head.passes} pass)"
        audio_head = model.parallel_head
    head_t = [0.0]
    original_sample = audio_head.sample

    def timed_head(*a, **k):
        mx.eval(a[0])  # barrier: main transformer out is ready
        t0 = time.perf_counter()
        r = original_sample(*a, **k)
        mx.eval(r)  # barrier: all codebooks sampled
        head_t[0] += time.perf_counter() - t0
        return r

    audio_head.sample = timed_head

    model.warmup(ct)
    mx.eval(model.parameters())

    enc_t = dec_t = step_t = 0.0
    out_pcm: list[np.ndarray] = []
    t_all0 = time.perf_counter()
    for idx in range(total):
        if idx == args.warmup:  # discard warmup frames from every accumulator
            enc_t = dec_t = step_t = head_t[0] = 0.0
            t_all0 = time.perf_counter()
        pcm = in_pcms[:, idx * 1920 : (idx + 1) * 1920]
        t0 = time.perf_counter()
        oat = mimi_enc.encode_step(pcm[None, 0:1])
        enc_t += time.perf_counter() - t0
        oat = np.transpose(oat, (0, 2, 1))[0, :, :other_cb]
        t0 = time.perf_counter()
        tt = gen.step(mx.array(oat), ct)
        tt[0].item()  # barrier: full LM step (main + dep) forced
        step_t += time.perf_counter() - t0
        at = gen.last_audio_tokens()
        if at is not None:
            t0 = time.perf_counter()
            at = np.array(at[:, :, None]).astype(np.uint32)
            out_pcm.append(mimi_dec.decode_step(at)[0, 0])
            dec_t += time.perf_counter() - t0
    wall = time.perf_counter() - t_all0

    n = args.frames
    main_t = step_t - head_t[0]
    lm_ms = 1000 * step_t / n
    cfg = json.loads((weights_dir / "config.json").read_text())
    model_name = pl._model_name(cfg)
    size_gb = (weights_dir / model_name).stat().st_size / 1e9
    weight_dtype = cfg.get("weight_dtype", "q4")

    print(
        f"\n=== {args.model} {weight_dtype} | {n} frames, wall {wall:.2f}s "
        f"-> {n / wall:.1f} frames/s ({n / wall / 12.5:.2f}x RT sequential) ==="
    )
    print(f"{'stage':<26}{'ms/frame':>10}{'% of wall':>10}")
    for name, t in [
        ("mimi encode", enc_t),
        ("LM main transformer", main_t),
        (head_label, head_t[0]),
        ("mimi decode", dec_t),
    ]:
        print(f"{name:<26}{1000 * t / n:>10.2f}{100 * t / wall:>9.1f}%")
    print(f"{'LM total':<26}{lm_ms:>10.2f}")
    if model.depformer is not None:
        print(f"depformer per slice: {1000 * head_t[0] / n / max(n_slices, 1):.2f} ms")
    print(f"artifact (LM weights): {size_gb:.2f} GB")
    print(
        f"pipelined critical path = LM total {lm_ms:.1f} ms/frame "
        f"-> {80 / lm_ms:.1f}x RT live ({1000 / lm_ms:.0f} frames/s)"
    )
    print(
        f"projected iPhone LM step @ scale {args.scale:.2f} (A18 Pro ~= {args.scale:.2f}x M4 Pro "
        f"assumption): {lm_ms / args.scale:.1f} ms vs 80 ms budget "
        f"-> {'FITS' if lm_ms / args.scale < 80 else 'OVER'}"
    )

    if args.silence:
        if out_pcm:
            pcm = np.concatenate(out_pcm)
            rms = float(np.sqrt((pcm**2).mean()))
            peak = float(np.abs(pcm).max())
        else:
            rms = peak = 0.0
        ok = rms < 0.10 and peak < 1.1
        print(f"silence-in: rms={rms:.4f} peak={peak:.3f} -> {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
