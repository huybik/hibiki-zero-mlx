#!/usr/bin/env python
"""Verify the 4-bit MLX hibiki-zero weights by translating a sample clip.

Uses the pipelined inference path (hibiki_mlx.pipeline), which overlaps the CPU
Mimi codec with the GPU LM (~3x real-time vs ~1.3x for the sequential
run_inference loop). Output is identical; this is just the fast MLX entry point.
"""
from pathlib import Path

import mlx.core as mx

from hibiki_mlx import run

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)

if __name__ == "__main__":
    mx.random.seed(299792458)
    run(
        str(HERE / "assets" / "samples" / "leon.wav"),
        str(HERE / "translations" / "leon_mlx_q4.wav"),
    )
