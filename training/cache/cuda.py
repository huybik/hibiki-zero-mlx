"""The cache build's CUDA models (`build.MODELS["cuda"]`, and both directions at once for
`build.DualBuilder`): Mimi (`mimi_cuda`) and the target aligner's wav2vec2 CTC (`wav2vec2_cuda`)
over padded batches, with pinned staging and output buffers
([cache-speed-cuda](../../docs/agent-memory/cache-speed-cuda.md))."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from training.cache.rows import CTC_RATE, FRAME_SAMPLES, SAMPLE_RATE, Aligned, AlignJob, Row
from training.cache.timing import Aligner, vocab_columns

# Padded audio per batch, Mimi and CTC. On a 4,094-row GigaSpeech 2 window (H200, TF32): CTC + Mimi took
# 13.0, 11.0, 10.7, 10.7 and 11.3 s at 80, 160, 320, 640 and 1,280 s (peak 2.4 to 27 GiB): bigger
# batches pad more. 1,280 s also overflows int32 indexing under torch.compile (64 channels x 30.7M).
CUDA_BATCH_S = 320.0  # omniASR's CTC (the former Vietnamese aligner) alone ran alike at 1,280 s (2026-11-11)
# TF32 matmuls and convolutions in CUDAModels (2026-11-09, H200, a 4,094-row GigaSpeech 2 window): CTC
# 11.7 → 4.1 s, Mimi 13.8 → 7.9 s. 0.021% of text token frames move (at most 4 frames, no row changes
# its accept/reject); Mimi's first codebook is 99.92% equal and the residual ones 96–99%, near-ties:
# the fp32 decode of the TF32 codes is as close to the source as the fp32 codes' (9.043 against
# 9.040 dB SNR). bf16 Mimi rewrote half the residual codes for ~10% more; bf16 CTC was slower.
CUDA_TF32 = True
DEVICE = "cuda"  # the torch device of CUDAModels


def cuda_mimi(mimi_weight: Path, num_codebooks: int) -> Any:
    """The encode-only CUDA Mimi (`mimi_cuda.FastMimiEncoder`: FP16 SEANet and transformer linears,
    FP32 norms and RVQ). Sets `CUDA_TF32` process-wide."""
    import torch

    from training.cache.mimi_cuda import FastMimiEncoder

    torch.backends.cuda.matmul.allow_tf32 = CUDA_TF32
    torch.backends.cudnn.allow_tf32 = CUDA_TF32
    return FastMimiEncoder.from_checkpoint(mimi_weight, num_codebooks=num_codebooks, device=DEVICE)


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices shortest first, cut where the batch padded to its longest would pass `budget`."""
    out: list[list[int]] = []
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        if out and (len(out[-1]) + 1) * lengths[index] <= budget:
            out[-1].append(index)
        else:
            out.append([index])
    return out


class _PinnedOutputPool:
    """Double-buffered reusable pinned host buffers (eliminates per-batch `cudaHostAlloc`)."""

    def __init__(self) -> None:
        self._bufs: list[Any] = [None, None]
        self._turn = 0

    def copy_from(self, tensor: Any) -> tuple[Any, Any]:
        import torch

        k = self._turn
        self._turn ^= 1
        need = tensor.numel()
        buf = self._bufs[k]
        if buf is None or buf.dtype != tensor.dtype or buf.numel() < need:
            self._bufs[k] = torch.empty(max(need, 1), dtype=tensor.dtype, pin_memory=True)
            buf = self._bufs[k]
        host = buf[:need].view(tensor.shape)
        host.copy_(tensor, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return host, done


class CUDAModels:
    """Blackwell-optimized Mimi (`FastMimiEncoder`) and wav2vec2 CTC (`FastWav2Vec2CTC`) on CUDA,
    over padded batches of up to `CUDA_BATCH_S` of audio, shortest first. Uses FP16 compute with
    FP32 GroupNorm/lm_head/RVQ/accumulation (10-bit mantissa = TF32) on the workers' 16 kHz CTC
    clips, zero-copy delay staging, and preallocated double-buffered pinned H2D/D2H pools."""

    window = 512
    threaded = True  # encode runs in the build's GPU thread, a window ahead of the main thread

    def __init__(self, mimi_weight: Path, num_codebooks: int, aligner: Aligner, files: Path):
        from training.cache.wav2vec2_cuda import FastWav2Vec2CTC

        self.mimi = cuda_mimi(mimi_weight, num_codebooks)
        self.cardinality = self.mimi.cardinality
        self.ctc = FastWav2Vec2CTC(files, vocab_columns(aligner, files), device=DEVICE)
        self._mimi_out_pool = _PinnedOutputPool()
        self._ctc_out_pool = _PinnedOutputPool()

    def encode(self, rows: list[Row], align: AlignJob) -> list[tuple[Aligned, np.ndarray, np.ndarray]]:
        """Each row's alignment (`align` given a CTC batch's rows and log-probabilities, as the batch
        lands) and Mimi codes: the CTC first, so the workers align while Mimi encodes; both clips of
        every row in one sort, so Mimi's batches come out fuller."""
        aligned: list[Any] = [None] * len(rows)

        def landed(group: list[int], log_probs: list[np.ndarray]) -> None:
            job = align([rows[i] for i in group], log_probs)
            for place, i in enumerate(group):
                aligned[i] = (job, place)

        self.ctc_log_probs(self.ctc, [row.target16 for row in rows], landed)
        clips = [row.source24 for row in rows] + [row.target24 for row in rows]
        delays = [0] * len(rows) + [row.delay_samples for row in rows]
        codes = self.mimi_codes(clips, delays=delays)
        return list(zip(aligned, codes[: len(rows)], codes[len(rows):], strict=True))

    def _pipelined(self, groups: list[list[int]], launch: Callable, collect: Callable) -> None:
        prev = None
        for group in groups:
            handle = launch(group)
            if prev is not None:
                collect(*prev)
            prev = (group, handle)
        if prev is not None:
            collect(*prev)

    def _stage(
        self,
        clips: list[np.ndarray],
        group: list[int],
        length: int,
        delays: list[int] | None = None,
    ) -> Any:
        """`group`'s clips (optionally left-padded by `delays[i]`) right-padded to `length` in the
        next pinned buffer, copied to the GPU without blocking."""
        import torch

        if not hasattr(self, "_pinned"):
            self._pinned, self._copied, self._turn = [torch.empty(0)] * 2, [None, None], 0
        k = self._turn
        self._turn ^= 1
        if self._copied[k] is not None:
            self._copied[k].synchronize()  # the buffer's last copy has left it
        need = len(group) * length
        if self._pinned[k].numel() < need:
            self._pinned[k] = torch.empty(max(need, int(CUDA_BATCH_S * SAMPLE_RATE)), pin_memory=True)
        host = self._pinned[k][:need].view(len(group), 1, length)
        view = host.numpy()
        for row, i in enumerate(group):
            pad = delays[i] if delays is not None else 0
            n = len(clips[i])
            if pad:
                view[row, 0, :pad] = 0.0
            view[row, 0, pad : pad + n] = clips[i]
            if pad + n < length:
                view[row, 0, pad + n :] = 0.0
        batch = host.to(DEVICE, non_blocking=True)
        self._copied[k] = torch.cuda.Event()
        self._copied[k].record()
        return batch

    def mimi_codes(self, clips: list[np.ndarray], delays: list[int] | None = None) -> list[np.ndarray]:
        """Each 24 kHz clip's codes, [codebooks, ceil((samples + delay) / 1920)] int32."""
        import torch

        out: list[np.ndarray] = [np.empty(0)] * len(clips)
        eff_lens = [len(c) + (delays[i] if delays is not None else 0) for i, c in enumerate(clips)]

        def launch(group: list[int]) -> tuple[Any, Any, list[int]]:
            frames = [math.ceil(eff_lens[i] / FRAME_SAMPLES) for i in group]
            batch = self._stage(clips, group, max(frames) * FRAME_SAMPLES, delays=delays)
            with torch.no_grad():
                codes = self.mimi.encode(batch).to(torch.int32)
            host, done = self._mimi_out_pool.copy_from(codes)
            return (host, done, frames)

        def collect(group: list[int], handle: tuple[Any, Any, list[int]]) -> None:
            host, done, frames = handle
            done.synchronize()
            codes = host.numpy()
            for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                # Copy out of the reusable pinned output buffer before the next collect overwrites it!
                out[i] = codes[row, :, :count].copy()

        self._pipelined(batches(eff_lens, int(CUDA_BATCH_S * SAMPLE_RATE)), launch, collect)
        return out

    def ctc_log_probs(
        self, ctc: Any, clips: list[np.ndarray], landed: Callable[[list[int], list[np.ndarray]], None]
    ) -> None:
        """`ctc` over normalized 16 kHz clips in batches; each batch's clip indices and
        log-probabilities go to `landed` as the batch lands."""
        import torch

        def launch(group: list[int]) -> tuple[Any, Any, list[int]]:
            lengths = torch.tensor([len(clips[i]) for i in group])
            batch = self._stage(clips, group, int(lengths.max()))
            with torch.no_grad():
                log_probs, frames = ctc(batch, lengths)
            host, done = self._ctc_out_pool.copy_from(log_probs)
            return (host, done, frames)

        def collect(group: list[int], handle: tuple[Any, Any, list[int]]) -> None:
            host, done, frames = handle
            done.synchronize()
            arr = host.numpy()
            landed(group, [arr[row, :count].copy() for row, count in enumerate(frames)])

        self._pipelined(batches([len(c) for c in clips], int(CUDA_BATCH_S * CTC_RATE)), launch, collect)


class CUDAModelsDual(CUDAModels):
    """Dual-direction CUDA models: ONE shared Mimi + BOTH English (`vi-en`) and Vietnamese (`en-vi`)
    `FastWav2Vec2CTC` aligners, encoding both directions of a shard in a single pass."""

    def __init__(
        self,
        mimi_weight: Path,
        num_codebooks: int,
        aligner_ve: Aligner,
        files_ve: Path,
        aligner_ev: Aligner,
        files_ev: Path,
    ):
        from training.cache.wav2vec2_cuda import FastWav2Vec2CTC

        self.mimi = cuda_mimi(mimi_weight, num_codebooks)
        self.cardinality = self.mimi.cardinality
        self.ctc_ve = FastWav2Vec2CTC(files_ve, vocab_columns(aligner_ve, files_ve), device=DEVICE)
        self.ctc_ev = FastWav2Vec2CTC(files_ev, vocab_columns(aligner_ev, files_ev), device=DEVICE)
        self._mimi_out_pool = _PinnedOutputPool()
        self._ctc_out_pool = _PinnedOutputPool()

    def encode_dual(
        self,
        rows_ve: list[Row],
        rows_ev: list[Row],
        align_ve: AlignJob,
        align_ev: AlignJob,
    ) -> tuple[list[tuple[Aligned, np.ndarray, np.ndarray]], list[tuple[Aligned, np.ndarray, np.ndarray]]]:
        """Run EN CTC (`vi-en`) and VI CTC (`en-vi`), dispatching both Viterbi queues to CPU
        workers, then run ONE unified sorted Mimi pass over all 4 clip streams."""
        aligned_ve: list[Any] = [None] * len(rows_ve)
        aligned_ev: list[Any] = [None] * len(rows_ev)

        if rows_ve:
            def landed_ve(group: list[int], lps: list[np.ndarray]) -> None:
                job = align_ve([rows_ve[i] for i in group], lps)
                for place, i in enumerate(group):
                    aligned_ve[i] = (job, place)

            self.ctc_log_probs(self.ctc_ve, [r.target16 for r in rows_ve], landed_ve)

        if rows_ev:
            def landed_ev(group: list[int], lps: list[np.ndarray]) -> None:
                job = align_ev([rows_ev[i] for i in group], lps)
                for place, i in enumerate(group):
                    aligned_ev[i] = (job, place)

            self.ctc_log_probs(self.ctc_ev, [r.target16 for r in rows_ev], landed_ev)

        n_ve, n_ev = len(rows_ve), len(rows_ev)
        clips = (
            [r.source24 for r in rows_ve]
            + [r.target24 for r in rows_ve]
            + [r.source24 for r in rows_ev]
            + [r.target24 for r in rows_ev]
        )
        delays = (
            [0] * n_ve
            + [r.delay_samples for r in rows_ve]
            + [0] * n_ev
            + [r.delay_samples for r in rows_ev]
        )
        codes = self.mimi_codes(clips, delays=delays) if clips else []
        off = 2 * n_ve
        res_ve = list(zip(aligned_ve, codes[:n_ve], codes[n_ve:off], strict=True))
        res_ev = list(zip(aligned_ev, codes[off : off + n_ev], codes[off + n_ev :], strict=True))
        return res_ve, res_ev
