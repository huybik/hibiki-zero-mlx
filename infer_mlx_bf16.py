#!/usr/bin/env python
"""Isolation run: identical pipeline/codec/sampler to infer_mlx_fast, but the LM
is loaded at full bf16 precision (no q4 quant). A/B this against *_mlx_fast.wav
to tell whether any audio degradation comes from the q4 quant or not."""
import json
from pathlib import Path

import mlx.core as mx
import rustymimi
import sentencepiece

import mlx_hibiki_patch  # noqa: F401
from moshi_mlx import models
import infer_mlx_fast as f

HERE = Path(__file__).parent
W = HERE / "weights"


def load_bf16(weights_dir: Path = W):
    cfg = json.loads((weights_dir / "config.json").read_text())
    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_pytorch_weights(
        str(weights_dir / "hibiki-pytorch-77f82164@110.safetensors"), lm_config, strict=True)
    mx.eval(model.parameters())
    tok = sentencepiece.SentencePieceProcessor(
        str(weights_dir / "tokenizer_spm_48k_multi6_2.model"))
    mimi_path = str(weights_dir / "mimi-pytorch-e351c8d8@125.safetensors")
    nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
    return (model, lm_config, tok,
            rustymimi.Tokenizer(mimi_path, num_codebooks=nq),
            rustymimi.Tokenizer(mimi_path, num_codebooks=nq))


if __name__ == "__main__":
    f.load = load_bf16  # swap only the LM precision; pipeline/codec/sampler unchanged
    mx.random.seed(299792458)
    f.run(str(HERE / "hibiki_zero" / "samples" / "leon.wav"),
          str(HERE / "translations" / "leon_mlx_bf16.wav"))
    f.run(str(HERE / "hibiki_zero" / "samples" / "crepes.mp3"),
          str(HERE / "translations" / "crepes_mlx_bf16.wav"))
