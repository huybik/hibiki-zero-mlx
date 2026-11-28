"""The cache build's MLX models (`build.MODELS["mlx"]`, Apple Silicon): Mimi (`mimi_mlx`) and the
target aligner's wav2vec2 CTC (`wav2vec2_mlx`) on the GPU, the CTC's transformer optionally on the
Neural Engine (`wav2vec2_coreml`) ([cache-build](../../docs/agent-memory/cache-build.md))."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.cache.rows import Aligned, AlignJob, Row
from training.cache.timing import Aligner, vocab_columns

# Freed MLX buffers kept for reuse across the ~36 bucketed shapes of Mimi and wav2vec2; on
# libriheavy rows 4, 8 and 16 GB ran 195x, 209x and 225x.
MLX_CACHE_BYTES = 16 << 30
# With --ane-ctc, the share of a window's audio whose CTC stays on the GPU (MLXModels). On 500
# libriheavy and 1,000 cv27 rows with omniASR: 0 ran 261x and 316x (the GPU waiting on the Neural
# Engine on long rows), 0.1 278x and 301x, 0.2 293x and 309x (the GPU 97-99% busy), 0.35 slower. With
# wav2vec2-base-vietnamese-250h (2026-11-22, interleaved twice): 0.1, 0.2 and 0.3 alike (256-299x), 0
# 187-225x.
ANE_GPU_SHARE = 0.2


class MLXModels:
    """Mimi (`mimi_mlx`) and the aligner's wav2vec2 CTC (`wav2vec2_mlx`) on Apple Silicon: one clip per forward
    (batching clips gained nothing in MLX), padded to length buckets, so each compiled forward
    sees few shapes. Mimi runs in float32; wav2vec2 runs in float32 by default or float16 when
    `half_ctc=True` (log-probs cast back to float32 before the Viterbi). With `ane_ctc=True` the
    aligner's transformer runs on the Neural Engine (`wav2vec2_coreml`, its own process) beside
    Mimi on the GPU, for all of a window's rows but the longest, which hold ANE_GPU_SHARE of its
    audio and keep the whole CTC on the GPU so that both finish together."""

    window = 64  # rows queued on the GPU at once

    def __init__(self, mimi_weight: Path, num_codebooks: int, aligner: Aligner, files: Path, half_ctc: bool = False,
                 ane_ctc: bool = False):
        import mlx.core as mx

        from training.cache.mimi_mlx import MimiEncoder
        from training.cache.wav2vec2_mlx import Wav2Vec2CTC

        mx.set_cache_limit(MLX_CACHE_BYTES)
        self.mimi = MimiEncoder(mimi_weight, num_codebooks)
        self.ctc = Wav2Vec2CTC(files, half=half_ctc)
        self.columns = mx.array(vocab_columns(aligner, files))
        self.cardinality = self.mimi.cardinality
        self.ane = None
        if ane_ctc:
            from training.cache.wav2vec2_coreml import Wav2Vec2ANE

            if half_ctc:  # its transformer input would come in float16: unmeasured
                raise ValueError("ane_ctc takes the float32 CTC's transformer input: not with half_ctc")

            self.ane = Wav2Vec2ANE(aligner, files)

    def _split(self, rows: list[Row]) -> tuple[list[int], list[int]]:
        """The rows whose transformer runs on the Neural Engine, shortest first (a shape switch
        costs it ~40%), and the rows whose CTC runs on the GPU: the longest, holding ANE_GPU_SHARE
        of the window's audio, and any longer than the Neural Engine's shapes; all without it."""
        if self.ane is None:
            return [], list(range(len(rows)))
        order = sorted(range(len(rows)), key=lambda index: len(rows[index].target16))
        budget = ANE_GPU_SHARE * sum(len(row.target16) for row in rows)
        gpu: list[int] = []
        while order:
            samples = len(rows[order[-1]].target16)
            if samples <= budget or samples > self.ane.max_samples:
                budget -= samples
                gpu.append(order.pop())
            else:
                break
        return order, gpu

    def encode(self, rows: list[Row], align: AlignJob) -> list[tuple[Aligned, Any, Any]]:
        """Queue CTC and Mimi on the GPU (with the Neural Engine, its rows' transformer inputs first,
        handed to it while the GPU runs the rest); block until CTC is done and workers are submitted
        (a job a row, for the Mac's two workers); return lazy Mimi codes so the caller can overlap
        planning the next window with Mimi."""
        import mlx.core as mx

        ane, gpu = self._split(rows)
        inputs = [self.ctc.transformer_input(rows[index].target16) for index in ane]
        mx.async_eval([hs for hs, _ in inputs])
        on_ane = self.ane.submit([(rows[index].data["id"], np.array(hs), frames)
                                  for index, (hs, frames) in zip(ane, inputs, strict=True)]) if ane else None
        log_probs = {index: mx.take(self.ctc.log_probs(rows[index].target16), self.columns, axis=1) for index in gpu}
        mx.async_eval(list(log_probs.values()))
        codes = [
            (self.mimi.encode(row.source24), self.mimi.encode(row.target24, delay_samples=row.delay_samples))
            for row in rows
        ]
        mx.async_eval(codes)
        aligned = {index: (align([rows[index]], [lp]), 0)
                   for index, lp in zip(ane, on_ane.result() if ane else [], strict=True)}
        # np.array(lp) blocks until CTC is done; Mimi runs on its own streams (mimi_mlx.STREAMS) beside it
        aligned |= {index: (align([rows[index]], [np.array(lp)]), 0) for index, lp in log_probs.items()}
        return [(aligned[index], source, target) for index, (source, target) in enumerate(codes)]
