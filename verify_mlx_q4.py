#!/usr/bin/env python
"""Verify the 4-bit MLX hibiki-zero weights by translating a sample clip.

Applies the hibiki-zero patches (mlx_hibiki_patch) then reuses
moshi_mlx.run_inference on the bundled leon.wav (FR -> EN).
"""
import sys
from pathlib import Path

import mlx_hibiki_patch  # noqa: F401  (patches moshi_mlx for hibiki-zero)
from moshi_mlx import run_inference

HERE = Path(__file__).parent
WEIGHTS = HERE / "weights"

sys.argv = [
    "run_inference",
    "--lm-config", str(WEIGHTS / "config.json"),
    "--moshi-weights", str(WEIGHTS / "hibiki.q4.safetensors"),
    "--mimi-weights", str(WEIGHTS / "mimi-pytorch-e351c8d8@125.safetensors"),
    "--tokenizer", str(WEIGHTS / "tokenizer_spm_48k_multi6_2.model"),
    str(HERE / "hibiki_zero" / "samples" / "leon.wav"),
    str(HERE / "translations" / "leon_mlx_q4.wav"),
]
run_inference.main()
