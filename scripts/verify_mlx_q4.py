#!/usr/bin/env python
"""Verify the 4-bit MLX hibiki-zero weights by translating a sample clip.

Uses the pipelined inference path (infer_mlx_fast), which overlaps the CPU Mimi
codec with the GPU LM (~3x real-time vs ~1.3x for the sequential run_inference
loop). Output is identical; this is just the fast entry point for the MLX path.
"""
import sys
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)
sys.path.insert(0, str(HERE / "src"))
from infer_mlx_fast import run

if __name__ == "__main__":
    mx.random.seed(299792458)
    run(
        str(HERE / "hibiki_zero" / "samples" / "leon.wav"),
        str(HERE / "translations" / "leon_mlx_q4.wav"),
    )
