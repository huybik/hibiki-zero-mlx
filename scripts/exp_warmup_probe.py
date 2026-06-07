#!/usr/bin/env python
"""Probe: log per-frame text emission to locate the leading spurious phrase.

Runs a clip through the q4 fast LM and prints (frame_idx, time_s, piece) for every
non-pad text token, so we can see whether the spurious lead-in sits in a clean
warmup window (droppable) or is interleaved with the real translation.
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import sphn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import infer_mlx_fast as f
from moshi_mlx import models, utils

mx.random.seed(299792458)


def probe(infile: str, preloaded, tail_s: float = 8.0):
    model, lm_config, text_tok, mimi_enc, mimi_dec = preloaded
    other_cb = lm_config.other_codebooks
    in_pcms, _ = sphn.read(f._resolve_audio_path(infile), sample_rate=24000)
    steps = in_pcms.shape[-1] // 1920
    tail = int(round(tail_s * 12.5))
    gen = models.LmGen(
        model=model, max_steps=steps + tail + 8,
        text_sampler=utils.Sampler(top_k=25, temp=0.8),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0, check=False,
    )
    model.warmup()
    silence = np.zeros((1, 1, 1920), dtype=in_pcms.dtype)

    def enc(frame):
        codes = mimi_enc.encode_step(frame)
        return mx.array(codes).transpose(0, 2, 1)[:, :, :other_cb][0]

    print(f"\n=== {Path(infile).stem}  ({steps} audio frames = {steps/12.5:.2f}s, +{tail} tail) ===")
    for idx in range(steps + tail):
        frame = in_pcms[None, 0:1, idx * 1920:(idx + 1) * 1920] if idx < steps else silence
        oat = enc(frame)
        tt = gen.step(oat)[0].item()
        if tt not in (0, 3):
            piece = text_tok.id_to_piece(tt).replace("▁", " ")
            marker = "AUDIO" if idx < steps else "TAIL "
            print(f"  f{idx:3d} t={idx/12.5:5.2f}s [{marker}] {tt:6d} '{piece}'")


if __name__ == "__main__":
    pre = f.load(f.W)
    model, lm_config, text_tok, _, _ = pre
    for stem in (sys.argv[1:] or ["fr_0022", "fr_0023", "fr_0000", "fr_0014"]):
        wav = f"remote_dataset/covost2_fr_en_test/{stem}.wav"
        mimi_enc, mimi_dec = f.make_mimi(f.W, lm_config)
        probe(wav, (model, lm_config, text_tok, mimi_enc, mimi_dec))
