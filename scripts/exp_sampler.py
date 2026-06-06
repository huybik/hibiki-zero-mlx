#!/usr/bin/env python
"""Does a tighter audio sampler suppress q4 crackle? Same published q4 weights,
sweep audio (top_k, temp). One model load; re-run with fresh LmGen each time."""
import json, time
from pathlib import Path
import mlx.core as mx, mlx.nn as nn, numpy as np
import rustymimi, sentencepiece, sphn
from moshi_mlx import models, utils

HERE = Path(__file__).resolve().parent.parent; W = HERE / "weights"  # scripts/ -> ..
SAMPLE = str(HERE / "hibiki_zero" / "samples" / "leon.wav")

cfg = json.loads((W/"config.json").read_text())
lm_config = models.LmConfig.from_config_dict(cfg)
model = models.Lm(lm_config); model.set_dtype(mx.bfloat16)
nn.quantize(model, bits=4, group_size=32)
model.load_weights(str(W/"hibiki.q4.safetensors"), strict=True)
mx.eval(model.parameters())
tok = sentencepiece.SentencePieceProcessor(str(W/"tokenizer_spm_48k_multi6_2.model"))
mp = str(W/"mimi-pytorch-e351c8d8@125.safetensors")
nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
other_cb = lm_config.other_codebooks; gen_cb = lm_config.generated_codebooks

in_pcms, _ = sphn.read(SAMPLE, sample_rate=24000)
steps = in_pcms.shape[-1] // 1920

def analyze(x):
    fb=1920; n=len(x)//fb
    jumps=np.array([abs(x[i*fb]-x[i*fb-1]) for i in range(1,n)])
    dd=np.abs(np.diff(x))
    return f"rms={np.sqrt((x**2).mean()):.4f} clip={(np.abs(x)>=0.999).sum()} maxStep={dd.max():.3f} step99.9={np.percentile(dd,99.9):.3f}"

def run(atopk, atemp):
    mx.random.seed(299792458)
    enc = rustymimi.Tokenizer(mp, num_codebooks=nq); dec = rustymimi.Tokenizer(mp, num_codebooks=nq)
    gen = models.LmGen(model=model, max_steps=steps+8,
        text_sampler=utils.Sampler(top_k=25, temp=0.8),
        audio_sampler=utils.Sampler(top_k=atopk, temp=atemp), cfg_coef=1.0, check=False)
    model.warmup()
    out=[]
    for idx in range(steps):
        pcm = in_pcms[:, idx*1920:(idx+1)*1920]
        codes = mx.array(enc.encode_step(pcm[None,0:1])).transpose(0,2,1)[:,:,:other_cb]
        gen.step(codes[0])
        a = gen.last_audio_tokens()
        if a is not None and gen_cb>0:
            out.append(dec.decode_step(np.array(a[:,:,None]).astype(np.uint32)))
    x = np.concatenate(out, axis=-1)[0,0]
    return x

for atopk, atemp in [(250,0.8),(250,0.6),(100,0.6),(50,0.4),(250,0.0)]:
    x = run(atopk, atemp)
    sphn.write_wav(str(HERE/"translations"/f"exp_smp_k{atopk}_t{atemp}.wav"), x, 24000)
    print(f"audio top_k={atopk:3d} temp={atemp}: {analyze(x)}")
