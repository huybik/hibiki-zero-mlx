#!/usr/bin/env python
"""Experiment: which quant scheme removes the q4 audio crackle, at what size.
Loads PyTorch LM once, then for each scheme quantizes a fresh-from-pth copy,
runs leon, and reports clipping / sample-step stats + estimated weight size."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sphn
from mlx.utils import tree_flatten

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)
W = HERE / "weights"
VENDORED_MOSHI_MLX = HERE / "moshi-mlx"
if VENDORED_MOSHI_MLX.exists():
    sys.path.insert(0, str(VENDORED_MOSHI_MLX))
sys.path.insert(0, str(HERE / "src"))

from moshi_mlx import models
import infer_mlx_fast as f
PTH = str(W / "hibiki-pytorch-77f82164@110.safetensors")
CFG = json.loads((W / "config.json").read_text())
SAMPLE = str(HERE / "hibiki_zero" / "samples" / "leon.wav")

import rustymimi, sentencepiece


def fresh_pth_model():
    lm_config = models.LmConfig.from_config_dict(CFG)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_pytorch_weights(PTH, lm_config, strict=True)
    return model, lm_config


def est_mb(model):
    # 4-bit packs 8 weights/uint32 + fp16 scales/biases per group_size; 8-bit packs 4.
    mb = 0.0
    for k, v in tree_flatten(model.parameters()):
        if "scales" in k or "biases" in k:
            mb += v.size * 2
        elif v.dtype == mx.uint32:
            mb += v.size * 4
        else:
            mb += v.size * 2
    return mb / 1e6


def make_loader(predicate):
    def _load(weights_dir):
        model, lm_config = fresh_pth_model()
        if predicate is not None:
            nn.quantize(model, bits=4, group_size=32, class_predicate=predicate)
        mx.eval(model.parameters())
        print(f"   est size: {est_mb(model):.0f} MB")
        tok = sentencepiece.SentencePieceProcessor(str(weights_dir / "tokenizer_spm_48k_multi6_2.model"))
        mp = str(weights_dir / "mimi-pytorch-e351c8d8@125.safetensors")
        nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
        return model, lm_config, tok, rustymimi.Tokenizer(mp, num_codebooks=nq), rustymimi.Tokenizer(mp, num_codebooks=nq)
    return _load


def analyze(path):
    x = sphn.read(path)[0][0]
    fb = 1920; n = len(x) // fb
    jumps = np.array([abs(x[i*fb] - x[i*fb-1]) for i in range(1, n)])
    dd = np.abs(np.diff(x))
    return (f"peak={np.abs(x).max():.3f} rms={np.sqrt((x**2).mean()):.4f} "
            f"clip={(np.abs(x)>=0.999).sum()} maxStep={dd.max():.3f} "
            f"step99.9={np.percentile(dd,99.9):.3f}")


def has_quant(m):  # only real Linears (skip the LayerNorms we added)
    return hasattr(m, "to_quantized")

SCHEMES = {
    "q4_all":          lambda p, m: has_quant(m),                                  # current published
    "q4_dep_bf16":     lambda p, m: has_quant(m) and not p.startswith("depformer"),
    "q4_depout_bf16":  lambda p, m: has_quant(m) and not (p.startswith("depformer") and ".linear_out" in p),
    "q4_depout_q8":    lambda p, m: ({"bits":8,"group_size":32} if (p.startswith("depformer") and ".linear_out" in p) else True) if has_quant(m) else False,
    "q8_all":          lambda p, m: {"bits":8,"group_size":32} if has_quant(m) else False,
}

if __name__ == "__main__":
    for name, pred in SCHEMES.items():
        print(f"\n=== {name} ===")
        mx.random.seed(299792458)
        f.load = make_loader(pred)
        out = str(HERE / "translations" / f"exp_{name}.wav")
        f.run(SAMPLE, out)
        print("  ", analyze(out))
