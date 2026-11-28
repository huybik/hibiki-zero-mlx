"""The build of grounded caches, on MLX (Apple Silicon) or CUDA: Mimi and the target language's
CTC aligner (`text_timing.ALIGNERS`) on each device, and the stream of source shards through
them, in either direction (`Settings.direction`). Every
builder runs on it (`cache_codes`, the EN-VI dataset; `training.tranche.cache`, the
real-Vietnamese tranche); a builder only reads and gates its own source rows and decides where
the shards go ([cache-build](../../docs/agent-memory/cache-build.md)).

  keys ─► read (the builder's), in a thread one shard ahead: rows and the builder's own gates
       ─► decode workers (PyAV, soxr) and cache_rows.plan (decode gates, delay draw, text groups),
          one window of rows ahead of the device; the clips come back through /dev/shm
       ─► windows, across shard ends
       ─► device: CTC ─► the Viterbi in the workers while the device runs Mimi
       ─► samples back to their shard ─► the shard, in order, once its last row is in
"""

from __future__ import annotations

import fcntl
import math
import os
import re
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import cpu_budget, read_json, require_file
from training.prep import cache_rows
from training.prep.audio import Rejected
from training.prep.cache_rows import CTC_RATE, FRAME_SAMPLES, SAMPLE_RATE, Planned, Row, Settings, Shard
from training.prep.text_timing import Aligner, AlignmentError, aligner_files, vocab_columns

if sys.platform == "darwin":  # before the builders' readers import pyarrow, whose mimalloc pool keeps freed buffers on macOS
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Freed MLX buffers kept for reuse across the ~36 bucketed shapes of Mimi and wav2vec2; on
# libriheavy rows 4, 8 and 16 GB ran 195x, 209x and 225x.
MLX_CACHE_BYTES = 16 << 30
MLX_WORKERS = 2  # 2, 4 and 8 ran alike; the Mac's CPU and GPU share one power budget
# With --ane-ctc, the share of a window's audio whose CTC stays on the GPU (MLXModels). On 500
# libriheavy and 1,000 cv27 rows with omniASR: 0 ran 261x and 316x (the GPU waiting on the Neural
# Engine on long rows), 0.1 278x and 301x, 0.2 293x and 309x (the GPU 97-99% busy), 0.35 slower. With
# wav2vec2-base-vietnamese-250h (2026-11-22, interleaved twice): 0.1, 0.2 and 0.3 alike (256-299x), 0
# 187-225x.
ANE_GPU_SHARE = 0.2
# Padded audio per batch, Mimi and CTC. On a 4,094-row GigaSpeech 2 window (H200, TF32): CTC + Mimi took
# 13.0, 11.0, 10.7, 10.7 and 11.3 s at 80, 160, 320, 640 and 1,280 s (peak 2.4 to 27 GiB): bigger
# batches pad more. 1,280 s also overflows int32 indexing under torch.compile (64 channels x 30.7M).
CUDA_BATCH_S = 320.0  # omniASR's CTC (the former Vietnamese aligner) alone ran alike at 1,280 s (2026-11-11)
# Rows per decode job (`cache_rows.prepare_rows`); a CTC batch's rows are one alignment job. One job a
# row cost the main process each row's pool round trip under its one GIL, beside the GPU thread.
DECODE_JOB_ROWS = 32
# TF32 matmuls and convolutions in CUDAModels (2026-11-09, H200, a 4,094-row GigaSpeech 2 window): CTC
# 11.7 → 4.1 s, Mimi 13.8 → 7.9 s. 0.021% of text token frames move (at most 4 frames, no row changes
# its accept/reject); Mimi's first codebook is 99.92% equal and the residual ones 96–99%, near-ties:
# the fp32 decode of the TF32 codes is as close to the source as the fp32 codes' (9.043 against
# 9.040 dB SNR). bf16 Mimi rewrote half the residual codes for ~10% more; bf16 CTC was slower.
CUDA_TF32 = True
DEVICE = "cuda"  # the torch device of CUDAModels
# A row's alignment: its job (`cache_rows.align_rows`, a CTC batch's rows) and its place in the job's results.
Aligned = tuple[Future, int]
AlignJob = Callable[[list[Row], list[np.ndarray]], Future]  # Builder._encode_start's: submits one


PCM_PREFIX = "cache_build_pcm_"  # each build's directory of clips in /dev/shm
PCM_LOCK = ".owner"  # in it, locked (flock) by its build while it runs


@contextmanager
def pcm_dir() -> Iterator[Path | None]:
    """Where the workers leave decoded clips for the main process to map (`cache_rows.Planned`): a
    fresh directory in /dev/shm, removed with whatever a failed run left; None where there is none
    (macOS), and the clips come back in the replies. A build killed outright (SIGKILL, the
    out-of-memory killer) cannot remove its own, and /dev/shm holds RAM until deleted (~11 GB after
    four killed builds on a 31 GB box, 2026-11-22). So each build holds a lock in its directory,
    which the kernel drops however the build dies, and first removes every directory whose lock it
    can take. A directory is locked before it takes its name, so no build finds one unlocked while
    in use; one from before the lock has none and is left alone. Its lock file goes last
    (`_remove_pcm`), so a removal that fails leaves the directory to the next build."""
    shm = Path("/dev/shm")
    if not shm.is_dir():
        yield None
        return
    for stale in shm.glob(f"{PCM_PREFIX}*"):
        try:
            fd = os.open(stale / PCM_LOCK, os.O_RDWR)
        except OSError:  # no lock file (from before the lock), or not ours to open
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:  # its build is running
            continue
        else:
            _remove_pcm(stale)
        finally:
            os.close(fd)
    staging = Path(tempfile.mkdtemp(prefix=f".{PCM_PREFIX}", dir=shm))  # outside the glob until locked
    lock = os.open(staging / PCM_LOCK, os.O_CREAT | os.O_RDWR, 0o600)  # not inherited by the workers
    fcntl.flock(lock, fcntl.LOCK_EX)
    path = staging.rename(shm / staging.name.removeprefix("."))
    try:
        yield path
    finally:
        _remove_pcm(path)
        os.close(lock)


def _remove_pcm(path: Path) -> None:
    """Remove a build's directory of clips, its lock file last: a failure (a dead build's worker
    still writing) leaves the lock file, and the next build tries again."""
    try:
        for entry in path.iterdir():
            if entry.name != PCM_LOCK:
                entry.unlink()
        (path / PCM_LOCK).unlink()
        path.rmdir()
    except FileNotFoundError:  # another build removed it first
        pass
    except OSError as error:
        print(f"could not remove {path}, left to the next build: {error}", file=sys.stderr, flush=True)


def default_workers(device: str) -> int:
    """Decode and alignment processes: few on the Mac, the CPU budget less the main process on the box."""
    return MLX_WORKERS if device == "mlx" else max(1, math.floor(cpu_budget()) - 1)


# --------------------------------------------------------------------------- MLX


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

        from training.prep.mimi_mlx import MimiEncoder
        from training.prep.wav2vec2_mlx import Wav2Vec2CTC

        mx.set_cache_limit(MLX_CACHE_BYTES)
        self.mimi = MimiEncoder(mimi_weight, num_codebooks)
        self.ctc = Wav2Vec2CTC(files, half=half_ctc)
        self.columns = mx.array(vocab_columns(aligner, files))
        self.cardinality = self.mimi.cardinality
        self.ane = None
        if ane_ctc:
            from training.prep.wav2vec2_coreml import Wav2Vec2ANE

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


# --------------------------------------------------------------------------- CUDA


def cuda_mimi(
    mimi_weight: Path,
    num_codebooks: int,
    fp16_seanet: bool = True,
    fp8_transformer: bool = False,
) -> Any:
    """Blackwell-optimized encode-only Mimi (`mimi_cuda.FastMimiEncoder`) on CUDA, or legacy
    moshi Mimi when `HIBIKI_LEGACY_MIMI=1`. Sets `CUDA_TF32` process-wide."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = CUDA_TF32
    torch.backends.cudnn.allow_tf32 = CUDA_TF32
    if os.environ.get("HIBIKI_LEGACY_MIMI"):
        from moshi.models import loaders

        return loaders.get_mimi(mimi_weight, device=torch.device(DEVICE), num_codebooks=num_codebooks).eval()
    from training.prep.mimi_cuda import FastMimiEncoder

    return FastMimiEncoder.from_checkpoint(
        mimi_weight,
        num_codebooks=num_codebooks,
        device=DEVICE,
        fp16_seanet=fp16_seanet and not os.environ.get("HIBIKI_FP32_MIMI"),
        fp16_transformer=fp16_seanet and not os.environ.get("HIBIKI_FP32_MIMI"),
        fp8_transformer=fp8_transformer,
    )


def mimi_batch(mimi: Any, clips: list[Any], device: Any) -> list[Any]:
    """The codes of 24 kHz clips (1-D tensors, on the CPU or `device`) encoded on `device` as one
    right-padded batch: each [codebooks, ceil(samples / 1920)] on the CPU as int32. Padding changes
    nothing before it: Mimi's convolutions and attention are causal."""
    import torch

    frames = [math.ceil(clip.shape[-1] / FRAME_SAMPLES) for clip in clips]
    # Filled where the clips are, then one copy: a copy per clip costs a sync each.
    batch = torch.zeros(len(clips), 1, max(frames) * FRAME_SAMPLES, device=clips[0].device)
    for row, clip in enumerate(clips):
        batch[row, 0, : clip.shape[-1]] = clip
    batch = batch.to(device, non_blocking=True)
    with torch.no_grad():
        codes = mimi.encode(batch).to("cpu", torch.int32)
    return [codes[row, :, :count] for row, count in enumerate(frames)]


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
    FP32 GroupNorm/lm_head/RVQ/accumulation (10-bit mantissa = TF32), GPU 24k->16k resampling for
    CTC, zero-copy delay staging, and preallocated double-buffered pinned H2D/D2H pools."""

    window = 512
    threaded = True  # encode runs in the build's GPU thread, a window ahead of the main thread

    def __init__(
        self,
        mimi_weight: Path,
        num_codebooks: int,
        aligner: Aligner,
        files: Path,
        half_ctc: bool = False,
        fp8_ctc: bool = False,
        fp8_mimi: bool = False,
        gpu_resample_ctc: bool = False,
    ):
        from training.prep.wav2vec2_cuda import FastWav2Vec2CTC, Resample24to16

        self.mimi = cuda_mimi(mimi_weight, num_codebooks, fp8_transformer=fp8_mimi)
        self.cardinality = self.mimi.cardinality
        use_fp16_ctc = (not os.environ.get("HIBIKI_FP32_CTC")) and (DEVICE == "cuda")
        self.ctc = FastWav2Vec2CTC(
            files,
            vocab_columns(aligner, files),
            device=DEVICE,
            fp16=use_fp16_ctc or half_ctc,
            use_fp8=fp8_ctc,
        )
        self.gpu_resample_ctc = bool(gpu_resample_ctc or os.environ.get("HIBIKI_GPU_RESAMPLE_CTC"))
        self.resampler = Resample24to16(DEVICE) if self.gpu_resample_ctc else None
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

        if self.gpu_resample_ctc:
            self.ctc_log_probs_24k([row.target24 for row in rows], landed)
        else:
            self.ctc_log_probs([row.target16 for row in rows], landed)
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
        if DEVICE == "cuda":
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
            if DEVICE == "cuda":
                host, done = self._mimi_out_pool.copy_from(codes)
            else:
                host, done = codes.cpu(), None
            return (host, done, frames)

        def collect(group: list[int], handle: tuple[Any, Any, list[int]]) -> None:
            host, done, frames = handle
            if done is not None:
                done.synchronize()
            codes = host.numpy()
            for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                # Copy out of the reusable pinned output buffer before the next collect overwrites it!
                out[i] = codes[row, :, :count].copy()

        self._pipelined(batches(eff_lens, int(CUDA_BATCH_S * SAMPLE_RATE)), launch, collect)
        return out

    def _run_ctc_batches(
        self,
        ctc_model: Any,
        clips: list[np.ndarray],
        rate: int,
        resample_24k: bool,
        landed: Callable[[list[int], list[np.ndarray]], None] | None,
    ) -> list[np.ndarray]:
        import torch

        out: list[np.ndarray] = [np.empty(0)] * len(clips)

        def launch(group: list[int]) -> tuple[Any, Any, list[int], bool]:
            lengths = torch.tensor([len(clips[i]) for i in group])
            batch = self._stage(clips, group, int(lengths.max()))
            with torch.no_grad():
                if resample_24k:
                    assert self.resampler is not None
                    batch, lengths = self.resampler(batch, lengths)
                log_probs, frames, is_packed = ctc_model(batch, lengths)
            if DEVICE == "cuda":
                host, done = self._ctc_out_pool.copy_from(log_probs)
            else:
                host, done = log_probs.cpu(), None
            return (host, done, frames, is_packed)

        def collect(group: list[int], handle: tuple[Any, Any, list[int], bool]) -> None:
            host, done, frames, is_packed = handle
            if done is not None:
                done.synchronize()
            arr = host.numpy()
            if is_packed:
                pos = 0
                for i, count in zip(group, frames, strict=True):
                    out[i] = arr[pos : pos + count].copy()
                    pos += count
            else:
                for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                    out[i] = arr[row, :count].copy()
            if landed is not None:
                landed(group, [out[i] for i in group])

        self._pipelined(batches([len(c) for c in clips], int(CUDA_BATCH_S * rate)), launch, collect)
        return out

    def ctc_log_probs(
        self,
        clips: list[np.ndarray],
        landed: Callable[[list[int], list[np.ndarray]], None] | None = None,
    ) -> list[np.ndarray]:
        """CTC log-probabilities on pre-normalized 16 kHz clips."""
        return self._run_ctc_batches(self.ctc, clips, CTC_RATE, resample_24k=False, landed=landed)

    def ctc_log_probs_24k(
        self,
        clips: list[np.ndarray],
        landed: Callable[[list[int], list[np.ndarray]], None] | None = None,
    ) -> list[np.ndarray]:
        """CTC log-probabilities on raw 24 kHz clips (resampled & normalized on GPU)."""
        return self._run_ctc_batches(self.ctc, clips, SAMPLE_RATE, resample_24k=True, landed=landed)


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
        fp8_ctc: bool = False,
        fp8_mimi: bool = False,
        gpu_resample_ctc: bool = False,
    ):
        from training.prep.wav2vec2_cuda import FastWav2Vec2CTC, Resample24to16

        self.mimi = cuda_mimi(mimi_weight, num_codebooks, fp8_transformer=fp8_mimi)
        self.cardinality = self.mimi.cardinality
        use_fp16_ctc = (not os.environ.get("HIBIKI_FP32_CTC")) and (DEVICE == "cuda")
        self.ctc_ve = FastWav2Vec2CTC(
            files_ve, vocab_columns(aligner_ve, files_ve), device=DEVICE, fp16=use_fp16_ctc, use_fp8=fp8_ctc
        )
        self.ctc_ev = FastWav2Vec2CTC(
            files_ev, vocab_columns(aligner_ev, files_ev), device=DEVICE, fp16=use_fp16_ctc, use_fp8=fp8_ctc
        )
        self.gpu_resample_ctc = bool(gpu_resample_ctc or os.environ.get("HIBIKI_GPU_RESAMPLE_CTC"))
        self.resampler = Resample24to16(DEVICE) if self.gpu_resample_ctc else None
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

            if self.gpu_resample_ctc:
                self._run_ctc_batches(self.ctc_ve, [r.target24 for r in rows_ve], SAMPLE_RATE, True, landed_ve)
            else:
                self._run_ctc_batches(self.ctc_ve, [r.target16 for r in rows_ve], CTC_RATE, False, landed_ve)

        if rows_ev:
            def landed_ev(group: list[int], lps: list[np.ndarray]) -> None:
                job = align_ev([rows_ev[i] for i in group], lps)
                for place, i in enumerate(group):
                    aligned_ev[i] = (job, place)

            if self.gpu_resample_ctc:
                self._run_ctc_batches(self.ctc_ev, [r.target24 for r in rows_ev], SAMPLE_RATE, True, landed_ev)
            else:
                self._run_ctc_batches(self.ctc_ev, [r.target16 for r in rows_ev], CTC_RATE, False, landed_ev)

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


MODELS = {"mlx": MLXModels, "cuda": CUDAModels}  # --device


# --------------------------------------------------------------------------- the build


class Builder:
    """A cache build on one device; `run` turns a builder's source shards into cache shards."""

    def __init__(
        self,
        device: str,
        settings: Settings,
        config_path: Path,
        mimi_weight: Path,
        tokenizer: Path,
        half_ctc: bool = False,
        ane_ctc: bool = False,
        fp8_ctc: bool = False,
        fp8_mimi: bool = False,
    ):
        import sentencepiece

        self.settings = settings
        self.cfg = read_json(config_path)
        self.tokenizer_path = require_file(tokenizer, "text tokenizer")
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(self.tokenizer_path))
        self.eos_id = int(self.tokenizer.eos_id())
        num_codebooks = max(int(self.cfg["dep_q"]), int(self.cfg["n_q"]) - int(self.cfg["dep_q"]))
        if ane_ctc and device != "mlx":
            raise ValueError("the Neural Engine CTC (ane_ctc) runs beside MLX: device mlx only")
        if ane_ctc and settings.target != "vi":
            # The English aligner there kept 99.35% of MLX float32's argmaxes (the Vietnamese 99.94%).
            raise ValueError("the Neural Engine CTC (ane_ctc) times Vietnamese text only: direction en-vi")
        extra_kwargs: dict[str, Any] = {}
        if device == "mlx" and ane_ctc:
            extra_kwargs["ane_ctc"] = True
        elif device == "cuda":
            extra_kwargs["fp8_ctc"] = fp8_ctc
            extra_kwargs["fp8_mimi"] = fp8_mimi
        self.models = MODELS[device](
            require_file(mimi_weight, "Mimi weight"),
            num_codebooks,
            settings.aligner,
            aligner_files(settings.aligner),
            half_ctc=half_ctc,
            **extra_kwargs,
        )
        if self.models.cardinality != int(self.cfg["card"]):
            raise RuntimeError(f"config card {self.cfg['card']} != Mimi cardinality {self.models.cardinality}")

    def payload(self, shard: Shard, source: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return cache_rows.payload(shard, self.settings, self.cfg, source, tokenizer=self.tokenizer_path.name, **extra)

    def run(self, keys: Iterable[Any], read: Callable[[Any], Shard], workers: int) -> Iterator[Shard]:
        """Each key's shard, read by `read`, with its samples and rejections, in order, once its last
        row is in. Only data faults reject a row (`Rejected`, `AlignmentError`); anything else (a
        dead worker's BrokenProcessPool, a device error) stops the build before the shard is
        yielded, so a rerun rebuilds it."""
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # the spawned workers' thread pools, one thread each
        make_target16 = not getattr(self.models, "gpu_resample_ctc", False)
        with (
            ThreadPoolExecutor(1) as reads,
            pcm_dir() as pcm,
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=cache_rows.init_worker,
                initargs=(self.settings, self.tokenizer_path, pcm, make_target16),
            ) as pool,
        ):
            try:
                yield from self._stream(self._read_ahead(keys, read, reads), pool)
            except BaseException:
                # Drop the queued decodes and the next shard's read rather than wait for them. The
                # decodes already running write clips into pcm: wait for them, so its removal follows.
                pool.shutdown(wait=True, cancel_futures=True)
                reads.shutdown(wait=False, cancel_futures=True)
                raise

    @staticmethod
    def _read_ahead(keys: Iterable[Any], read: Callable[[Any], Shard], reads: ThreadPoolExecutor) -> Iterator[Shard]:
        """The shards, each read in the thread while the one before it streams."""
        ahead: Future | None = None
        for key in keys:
            current, ahead = ahead, reads.submit(read, key)
            if current is not None:
                yield current.result()
        if ahead is not None:
            yield ahead.result()

    @staticmethod
    def _consume_job(job: Future) -> tuple[np.ndarray | None, list]:
        slab_path, items = job.result()
        if slab_path is None:
            return None, items
        slab = np.fromfile(slab_path, np.float32)
        os.unlink(slab_path)
        return slab, items

    def _stream(self, shards: Iterator[Shard], pool: ProcessPoolExecutor) -> Iterator[Shard]:
        """The shards filled, their rows planned and encoded in windows that run across shard ends.
        Each window's GPU work is queued before the previous window is collected, so the GPU runs
        continuously: CTC_N → Mimi_N → CTC_N+1 → Mimi_N+1 → … (on MLX Mimi has its own streams, so
        Mimi_N runs beside CTC_N+1). On MLX `encode` returns lazy codes;
        on CUDA (`threaded`) it blocks, so it runs in a GPU thread with the next window queued
        behind it while the main thread collects one window and plans the next."""
        started: deque[Shard] = deque()
        window: list[Row] = []
        inflight: deque[Future] = deque()  # _encode_start results, oldest first
        threaded = getattr(self.models, "threaded", False)
        with ThreadPoolExecutor(1, thread_name_prefix="gpu") as gpu:

            def submit(rows: list[Row]) -> None:
                if threaded:
                    inflight.append(gpu.submit(self._encode_start, rows, pool))
                else:
                    done: Future = Future()
                    done.set_result(self._encode_start(rows, pool))
                    inflight.append(done)

            for shard, batch_rows, job in self._decoded(shards, pool, started):
                slab, items = self._consume_job(job)
                for (index, row), planned in zip(batch_rows, items, strict=True):
                    if isinstance(planned, (Rejected, AlignmentError)):
                        shard.reject(index, row["id"], planned)
                    else:
                        window.append(planned.row(shard, index, row, slab))
                    if len(window) == self.models.window:
                        submit(window)
                        window = []
                        while len(inflight) > 1:
                            self._encode_finish(inflight.popleft().result())
                    while started and not started[0].left:
                        yield started.popleft().done()
            submit(window)
            while inflight:
                self._encode_finish(inflight.popleft().result())
        if any(shard.left for shard in started):
            raise RuntimeError("the build ended with rows neither sampled nor rejected")
        for shard in started:
            yield shard.done()

    def _decoded(
        self, shards: Iterator[Shard], pool: ProcessPoolExecutor, started: deque[Shard]
    ) -> Iterator[tuple[Shard, list[tuple[int, dict[str, Any]]], Future]]:
        """Each 32-row decode job (`cache_rows.prepare_rows`), submitted a window ahead."""
        queued: deque[tuple[Shard, list[tuple[int, dict[str, Any]]], Future]] = deque()
        queued_rows = 0
        for shard in shards:
            started.append(shard)
            for start in range(0, len(shard.rows), DECODE_JOB_ROWS):
                rows = shard.rows[start : start + DECODE_JOB_ROWS]
                job = pool.submit(cache_rows.prepare_rows, [self._decode_args(row) for _, row in rows], shard.split)
                queued.append((shard, rows, job))
                queued_rows += len(rows)
                while queued_rows > self.models.window:
                    item = queued.popleft()
                    queued_rows -= len(item[1])
                    yield item
        while queued:
            yield queued.popleft()

    def _decode_args(self, row: dict[str, Any]) -> tuple[dict[str, Any], bytes, bytes]:
        """A row's part of a decode job: its fields, then the source's and target's audio, taken from it."""
        audio = {side: row.pop(f"{side}_audio") for side in ("vi", "en")}
        fields = {key: row[key] for key in ("id", "text_en", "text_vi", "vi_duration_s", "en_duration_s")}
        return fields, audio[self.settings.source], audio[self.settings.target]

    def _encode_start(self, window: list[Row], pool: ProcessPoolExecutor) -> list:
        """Queue GPU work for `window` and submit alignment workers; return lazy Mimi codes."""
        if not window:
            return []

        def align(rows: list[Row], log_probs: list[np.ndarray]) -> Future:
            return pool.submit(cache_rows.align_rows, log_probs,
                               [(r.lead, r.groups, r.target_frames - r.delay_frames, r.delay_frames) for r in rows],
                               self.eos_id)

        return list(zip(window, self.models.encode(window, align), strict=True))

    def _encode_finish(self, pending: list) -> None:
        """Collect Mimi codes and alignment results from `_encode_start`, finalizing each row."""
        for row, ((job, place), source, target) in pending:
            aligned = job.result()[place]
            if isinstance(aligned, AlignmentError):
                row.shard.reject(row.index, row.data["id"], aligned)
                continue
            tokens, text_frames, score = aligned
            row.shard.samples.append(cache_rows.sample(row, tokens, text_frames, score, np.array(source),
                                                        np.array(target), self.cfg, self.settings))
            row.shard.left -= 1


class DualBuilder:
    """Single-pass dual-direction CUDA cache builder (`vi-en` and `en-vi` simultaneously): reads
    each shard once, decodes & resamples each clip to 24 kHz once, and encodes both directions with
    one shared Mimi + both CTC aligners."""

    def __init__(
        self,
        settings_ve: Settings,
        settings_ev: Settings,
        config_path: Path,
        mimi_weight: Path,
        tokenizer_ve: Path,
        tokenizer_ev: Path,
        fp8_ctc: bool = False,
        fp8_mimi: bool = False,
    ):
        import sentencepiece

        self.settings_ve = settings_ve
        self.settings_ev = settings_ev
        self.cfg = read_json(config_path)
        self.tok_ve_path = require_file(tokenizer_ve, "vi-en text tokenizer")
        self.tok_ev_path = require_file(tokenizer_ev, "en-vi text tokenizer")
        self.eos_ve = int(sentencepiece.SentencePieceProcessor(str(self.tok_ve_path)).eos_id())
        self.eos_ev = int(sentencepiece.SentencePieceProcessor(str(self.tok_ev_path)).eos_id())
        num_codebooks = max(int(self.cfg["dep_q"]), int(self.cfg["n_q"]) - int(self.cfg["dep_q"]))
        self.models = CUDAModelsDual(
            require_file(mimi_weight, "Mimi weight"),
            num_codebooks,
            settings_ve.aligner,
            aligner_files(settings_ve.aligner),
            settings_ev.aligner,
            aligner_files(settings_ev.aligner),
            fp8_ctc=fp8_ctc,
            fp8_mimi=fp8_mimi,
        )
        if self.models.cardinality != int(self.cfg["card"]):
            raise RuntimeError(f"config card {self.cfg['card']} != Mimi cardinality {self.models.cardinality}")

    def payload(self, direction: str, shard: Shard, source: dict[str, Any], **extra: Any) -> dict[str, Any]:
        settings = self.settings_ve if direction == "vi-en" else self.settings_ev
        tok_name = self.tok_ve_path.name if direction == "vi-en" else self.tok_ev_path.name
        return cache_rows.payload(shard, settings, self.cfg, source, tokenizer=tok_name, **extra)

    def run(self, keys: Iterable[Any], read: Callable[[Any], Shard], workers: int) -> Iterator[tuple[Shard, Shard]]:
        """Yield `(shard_vi_en, shard_en_vi)` in key order."""
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        make_target16 = not self.models.gpu_resample_ctc
        with (
            ThreadPoolExecutor(1) as reads,
            pcm_dir() as pcm,
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=cache_rows.init_dual_worker,
                initargs=(self.settings_ve, self.tok_ve_path, self.settings_ev, self.tok_ev_path, pcm, make_target16),
            ) as pool,
        ):
            try:
                yield from self._stream(Builder._read_ahead(keys, read, reads), pool)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                reads.shutdown(wait=False, cancel_futures=True)
                raise

    def _stream(self, shards: Iterator[Shard], pool: ProcessPoolExecutor) -> Iterator[tuple[Shard, Shard]]:
        started: deque[tuple[Shard, Shard]] = deque()
        win_ve: list[Row] = []
        win_ev: list[Row] = []
        inflight: deque[Future] = deque()
        # Half the single-direction row count since each row carries 4 Mimi clips (2 directions).
        win_cap = max(1, self.models.window // 2)
        with ThreadPoolExecutor(1, thread_name_prefix="gpu") as gpu:

            def submit(r_ve: list[Row], r_ev: list[Row]) -> None:
                if r_ve or r_ev:
                    inflight.append(gpu.submit(self._encode_start, r_ve, r_ev, pool))

            for sh_ve, sh_ev, batch_rows, job in self._decoded(shards, pool, started, win_cap):
                slab, items = Builder._consume_job(job)
                for (index, row), dp in zip(batch_rows, items, strict=True):
                    if isinstance(dp, Rejected):
                        sh_ve.reject(index, row["id"], dp)
                        sh_ev.reject(index, row["id"], dp)
                        continue
                    vi16: np.ndarray | None = None
                    en16: np.ndarray | None = None
                    if dp.pcm is not None:
                        vi24, en24, vi16, en16 = dp.pcm
                    elif slab is not None and dp.lengths[0]:
                        o = dp.offset
                        a = o + dp.lengths[0]
                        b = a + dp.lengths[1]
                        c = b + dp.lengths[2]
                        d = c + dp.lengths[3]
                        vi24 = slab[o:a]
                        en24 = slab[a:b]
                        vi16 = slab[b:c] if dp.lengths[2] else None
                        en16 = slab[c:d] if dp.lengths[3] else None
                    else:
                        vi24 = en24 = np.empty(0, np.float32)

                    if isinstance(dp.vi_en, (Rejected, AlignmentError)):
                        sh_ve.reject(index, row["id"], dp.vi_en)
                    else:
                        p = dp.vi_en
                        win_ve.append(Row(
                            sh_ve, index, row, vi24, en24, en16, p.lead, p.groups, p.delay_s,
                            p.delay_frames, p.source_frames, p.target_frames, p.delay_samples,
                        ))

                    if isinstance(dp.en_vi, (Rejected, AlignmentError)):
                        sh_ev.reject(index, row["id"], dp.en_vi)
                    else:
                        p = dp.en_vi
                        win_ev.append(Row(
                            sh_ev, index, row, en24, vi24, vi16, p.lead, p.groups, p.delay_s,
                            p.delay_frames, p.source_frames, p.target_frames, p.delay_samples,
                        ))

                    if max(len(win_ve), len(win_ev)) >= win_cap:
                        submit(win_ve, win_ev)
                        win_ve, win_ev = [], []
                        while len(inflight) > 1:
                            self._encode_finish(inflight.popleft().result())
                    while started and not started[0][0].left and not started[0][1].left:
                        s_ve, s_ev = started.popleft()
                        yield s_ve.done(), s_ev.done()
            submit(win_ve, win_ev)
            while inflight:
                self._encode_finish(inflight.popleft().result())
        if any(s_ve.left or s_ev.left for s_ve, s_ev in started):
            raise RuntimeError("dual build ended with rows neither sampled nor rejected")
        for s_ve, s_ev in started:
            yield s_ve.done(), s_ev.done()

    def _decoded(
        self,
        shards: Iterator[Shard],
        pool: ProcessPoolExecutor,
        started: deque[tuple[Shard, Shard]],
        win_cap: int,
    ) -> Iterator[tuple[Shard, Shard, list[tuple[int, dict[str, Any]]], Future]]:
        queued: deque[tuple[Shard, Shard, list[tuple[int, dict[str, Any]]], Future]] = deque()
        queued_rows = 0
        for shard in shards:
            sh_ve = Shard(shard.key, shard.file, shard.split, shard.size, [], list(shard.rejected))
            sh_ev = Shard(shard.key, shard.file, shard.split, shard.size, [], list(shard.rejected))
            sh_ve.left = sh_ev.left = len(shard.rows)
            started.append((sh_ve, sh_ev))
            for start in range(0, len(shard.rows), DECODE_JOB_ROWS):
                rows = shard.rows[start : start + DECODE_JOB_ROWS]
                args = []
                for _, row in rows:
                    vi_audio = row.pop("vi_audio")
                    en_audio = row.pop("en_audio")
                    fields = {k: row[k] for k in ("id", "text_en", "text_vi", "vi_duration_s", "en_duration_s")}
                    args.append((fields, vi_audio, en_audio))
                job = pool.submit(cache_rows.prepare_dual_rows, args, shard.split)
                queued.append((sh_ve, sh_ev, rows, job))
                queued_rows += len(rows)
                while queued_rows > win_cap:
                    item = queued.popleft()
                    queued_rows -= len(item[2])
                    yield item
        while queued:
            yield queued.popleft()

    def _encode_start(
        self, rows_ve: list[Row], rows_ev: list[Row], pool: ProcessPoolExecutor
    ) -> tuple[list, list]:
        def align_ve(rows: list[Row], lps: list[np.ndarray]) -> Future:
            return pool.submit(
                cache_rows.align_rows,
                lps,
                [(r.lead, r.groups, r.target_frames - r.delay_frames, r.delay_frames) for r in rows],
                self.eos_ve,
                "vi-en",
            )

        def align_ev(rows: list[Row], lps: list[np.ndarray]) -> Future:
            return pool.submit(
                cache_rows.align_rows,
                lps,
                [(r.lead, r.groups, r.target_frames - r.delay_frames, r.delay_frames) for r in rows],
                self.eos_ev,
                "en-vi",
            )

        res_ve, res_ev = self.models.encode_dual(rows_ve, rows_ev, align_ve, align_ev)
        return list(zip(rows_ve, res_ve, strict=True)), list(zip(rows_ev, res_ev, strict=True))

    def _encode_finish(self, pending: tuple[list, list]) -> None:
        pend_ve, pend_ev = pending
        for items, settings in ((pend_ve, self.settings_ve), (pend_ev, self.settings_ev)):
            for row, ((job, place), source, target) in items:
                aligned = job.result()[place]
                if isinstance(aligned, AlignmentError):
                    row.shard.reject(row.index, row.data["id"], aligned)
                    continue
                tokens, text_frames, score = aligned
                row.shard.samples.append(
                    cache_rows.sample(row, tokens, text_frames, score, np.array(source), np.array(target), self.cfg, settings)
                )
                row.shard.left -= 1


def summary(shard: Shard, hours: float, wall: float) -> str:
    """A built shard's log line: rows kept, rejections by reason, its Mimi audio, then the build's
    speed so far, `hours` of Mimi audio in `wall` seconds (windows run across shard ends, so a shard
    has no time of its own)."""
    reasons: Counter[str] = Counter()
    for item in shard.rejected:
        head, _, rest = re.sub(r"[\d.]+", "N", item["reason"]).partition(": ")
        reasons[f"{head}: {rest}"[:60] if head in ("Rejected", "AlignmentError") else head] += 1
    return (f"{len(shard.samples)} of {shard.size} rows kept, rejected {dict(reasons)}; "
            f"{cache_rows.mimi_hours(shard.samples):.2f} h of Mimi audio; so far {hours:.2f} h in "
            f"{wall:.0f} s ({hours * 3600 / max(wall, 1e-9):.0f}x RT)")
