#!/usr/bin/env python
"""Per-stage profiler for the MLX hibiki-zero decode loop.

Splits each frame into: main transformer (forward_text + text sample) vs
depformer (16 codebook steps) vs mimi codec, using mx.eval barriers so the
lazy graph is actually forced at each boundary. Run before/after changes.
"""
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece

from moshi_mlx import models, utils

HERE = Path(__file__).parent
W = HERE / "weights"
N_FRAMES = 150

mx.random.seed(299792458)
cfg = json.loads((W / "config.json").read_text())
lm_config = models.LmConfig.from_config_dict(cfg)
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)
nn.quantize(model, bits=4, group_size=32)
model.load_weights(str(W / "hibiki.q4.safetensors"), strict=True)

other_cb = lm_config.other_codebooks
mimi = rustymimi.Tokenizer(str(W / "mimi-pytorch-e351c8d8@125.safetensors"), num_codebooks=other_cb)
in_pcms, _ = sphn = __import__("sphn").read(str(HERE / "hibiki_zero" / "samples" / "leon.wav"), sample_rate=24000)

gen = models.LmGen(model=model, max_steps=N_FRAMES + 8,
                   text_sampler=utils.Sampler(top_k=25, temp=0.8),
                   audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                   cfg_coef=1.0, check=False)

# Instrument depformer.sample to time it inside the step.
dep_t = [0.0]
_orig_dep = model.depformer.sample
def timed_dep(*a, **k):
    mx.eval(a[0])           # barrier: main transformer out is ready
    t0 = time.perf_counter()
    r = _orig_dep(*a, **k)
    mx.eval(r)              # barrier: all 16 codebooks sampled
    dep_t[0] += time.perf_counter() - t0
    return r
model.depformer.sample = timed_dep

model.warmup()
mx.eval(model.parameters())

enc_t = dec_t = step_t = 0.0
t_all0 = time.perf_counter()
for idx in range(N_FRAMES):
    pcm = in_pcms[:, idx * 1920:(idx + 1) * 1920]
    if pcm.shape[-1] < 1920:
        break
    t0 = time.perf_counter()
    oat = mimi.encode_step(pcm[None, 0:1])
    enc_t += time.perf_counter() - t0
    oat = mx.array(oat).transpose(0, 2, 1)[:, :, :other_cb]
    t0 = time.perf_counter()
    tt = gen.step(oat[0])
    tt[0].item()                 # barrier: full LM step (main + dep) forced
    step_t += time.perf_counter() - t0
    at = gen.last_audio_tokens()
    if at is not None:
        t0 = time.perf_counter()
        at = np.array(at[:, :, None]).astype(np.uint32)
        mimi.decode_step(at)
        dec_t += time.perf_counter() - t0
wall = time.perf_counter() - t_all0

n = N_FRAMES
main_t = step_t - dep_t[0]      # LM step minus depformer portion
print(f"\n=== {n} frames, wall {wall:.2f}s -> {n/wall:.1f} frames/s ({n/wall/12.5:.2f}x RT) ===")
print(f"{'stage':<22}{'ms/frame':>10}{'% of wall':>10}")
for name, t in [("mimi encode", enc_t), ("LM main transformer", main_t),
                ("LM depformer (16x)", dep_t[0]), ("mimi decode", dec_t)]:
    print(f"{name:<22}{1000*t/n:>10.2f}{100*t/wall:>9.1f}%")
