#!/usr/bin/env python
"""Verify upstream's staged q4 model (weights/hibiki-zero-3b-q4) by translating a sample clip.

Uses the pipelined inference path (hibiki_mlx.pipeline), which overlaps the CPU
Mimi codec with the GPU LM. This is the canonical MLX entry point.
"""
import mlx.core as mx

from hibiki_mlx.pipeline import ROOT, SAMPLE, UPSTREAM, load, run

if __name__ == "__main__":
    model = load(UPSTREAM)
    mx.random.seed(299792458)
    run(str(SAMPLE), str(ROOT / "translations" / "leon_mlx_q4.wav"), model)
