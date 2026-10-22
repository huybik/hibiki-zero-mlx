#!/usr/bin/env python
"""Verify the 4-bit MLX hibiki-zero weights by translating a sample clip.

Uses the pipelined inference path (hibiki_mlx.pipeline), which overlaps the CPU
Mimi codec with the GPU LM. This is the canonical MLX entry point.
"""
import mlx.core as mx

from hibiki_mlx.pipeline import ROOT, SAMPLE, run

if __name__ == "__main__":
    mx.random.seed(299792458)
    run(str(SAMPLE), str(ROOT / "translations" / "leon_mlx_q4.wav"))
