#!/usr/bin/env python
"""Pipelined MLX hibiki-zero inference.

Overlaps the CPU Mimi codec (rustymimi, GIL-released) with the GPU LM:
  - encoder thread streams encode_step over the whole file, running ahead
  - main thread runs the autoregressive LM step on the GPU
  - decoder thread streams decode_step on the audio tokens
FIFO queues preserve the streaming order, so output is bit-identical to the
sequential loop; we just stop letting the CPU and GPU idle on each other.

Usage: python infer_mlx_fast.py <in.wav> <out.wav>
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn

import mlx_hibiki_patch  # noqa: F401  (patches moshi_mlx for hibiki-zero)
from moshi_mlx import models, utils

HERE = Path(__file__).parent
W = HERE / "weights"
SENTINEL = object()


def load(weights_dir: Path):
    cfg = json.loads((weights_dir / "config.json").read_text())
    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32)
    model.load_weights(str(weights_dir / "hibiki.q4.safetensors"), strict=True)
    mx.eval(model.parameters())
    tok = sentencepiece.SentencePieceProcessor(str(weights_dir / "tokenizer_spm_48k_multi6_2.model"))
    # Separate codec instances per thread: a single rustymimi.Tokenizer can't be
    # borrowed by the encoder and decoder threads at once ("Already borrowed").
    mimi_path = str(weights_dir / "mimi-pytorch-e351c8d8@125.safetensors")
    nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
    mimi_enc = rustymimi.Tokenizer(mimi_path, num_codebooks=nq)
    mimi_dec = rustymimi.Tokenizer(mimi_path, num_codebooks=nq)
    return model, lm_config, tok, mimi_enc, mimi_dec


def run(infile: str, outfile: str, weights_dir: Path = W):
    model, lm_config, text_tok, mimi_enc, mimi_dec = load(weights_dir)
    other_cb = lm_config.other_codebooks
    gen_cb = lm_config.generated_codebooks

    in_pcms, _ = sphn.read(infile, sample_rate=24000)
    steps = in_pcms.shape[-1] // 1920

    gen = models.LmGen(
        model=model, max_steps=steps + 8,
        text_sampler=utils.Sampler(top_k=25, temp=0.8),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0, check=False,
    )
    model.warmup()

    enc_q: queue.Queue = queue.Queue(maxsize=64)   # encoder -> main
    dec_q: queue.Queue = queue.Queue(maxsize=64)   # main -> decoder
    out_pcm: list = []

    def encoder():
        # Separate streaming state from the model; runs ahead of the LM.
        for idx in range(steps):
            pcm = in_pcms[:, idx * 1920:(idx + 1) * 1920]
            codes = mimi_enc.encode_step(pcm[None, 0:1])      # CPU, GIL released
            codes = mx.array(codes).transpose(0, 2, 1)[:, :, :other_cb]
            enc_q.put(codes[0])
        enc_q.put(SENTINEL)

    def decoder():
        while True:
            item = dec_q.get()
            if item is SENTINEL:
                break
            out_pcm.append(mimi_dec.decode_step(item))        # CPU, GIL released

    enc_t = threading.Thread(target=encoder, daemon=True)
    dec_t = threading.Thread(target=decoder, daemon=True)
    enc_t.start(); dec_t.start()

    text_pieces: list[str] = []
    t0 = time.perf_counter()
    while True:
        oat = enc_q.get()
        if oat is SENTINEL:
            break
        text_token = gen.step(oat)
        tt = text_token[0].item()                             # sync this frame's LM
        if tt not in (0, 3):
            piece = text_tok.id_to_piece(tt).replace("▁", " ")
            text_pieces.append(piece)
        audio = gen.last_audio_tokens()
        if audio is not None and gen_cb > 0:
            dec_q.put(np.array(audio[:, :, None]).astype(np.uint32))
    dec_q.put(SENTINEL)
    enc_t.join(); dec_t.join()
    wall = time.perf_counter() - t0

    if out_pcm:
        pcm = np.concatenate(out_pcm, axis=-1)[0, 0]
        sphn.write_wav(outfile, pcm, 24000)
    print("".join(text_pieces).strip())
    print(f"\n[{steps} frames in {wall:.2f}s -> {steps/wall:.1f} frames/s "
          f"({steps/wall/12.5:.2f}x RT), out: {outfile}]")


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "hibiki_zero" / "samples" / "leon.wav")
    outfile = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "translations" / "leon_mlx_fast.wav")
    mx.random.seed(299792458)
    run(infile, outfile)
