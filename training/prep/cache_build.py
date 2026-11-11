"""The build of grounded caches, on MLX (Apple Silicon) or CUDA: Mimi and the target language's
CTC aligner (`text_timing.ALIGNERS`) on each device, and the stream of source shards through
them, in either direction (`Settings.direction`; MLX runs only the English aligner). Every
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

import math
import os
import re
import shutil
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
from training.prep.cache_rows import CTC_RATE, FRAME_SAMPLES, SAMPLE_RATE, Planned, Row, Settings, Shard
from training.prep.text_timing import Aligner, AlignmentError, aligner_files, vocab_columns

if sys.platform == "darwin":  # before the builders' readers import pyarrow, whose mimalloc pool keeps freed buffers on macOS
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Freed MLX buffers kept for reuse across the ~36 bucketed shapes of Mimi and wav2vec2; on
# libriheavy rows 4, 8 and 16 GB ran 195x, 209x and 225x.
MLX_CACHE_BYTES = 16 << 30
MLX_WORKERS = 2  # 2, 4 and 8 ran alike; the Mac's CPU and GPU share one power budget
# Padded audio per batch, Mimi and CTC. On a 4,094-row GigaSpeech 2 window (H200, TF32): CTC + Mimi took
# 13.0, 11.0, 10.7, 10.7 and 11.3 s at 80, 160, 320, 640 and 1,280 s (peak 2.4 to 27 GiB): bigger
# batches pad more. 1,280 s also overflows int32 indexing under torch.compile (64 channels x 30.7M).
CUDA_BATCH_S = 320.0
CTC_BATCH_S = CUDA_BATCH_S  # the CTC's own budget (omniASR's measured below)
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


@contextmanager
def pcm_dir() -> Iterator[Path | None]:
    """Where the workers leave decoded clips for the main process to map (`cache_rows.Planned`): a
    fresh directory in /dev/shm, removed with whatever a failed run left; None where there is none
    (macOS), and the clips come back in the replies."""
    if not Path("/dev/shm").is_dir():
        yield None
        return
    path = Path(tempfile.mkdtemp(prefix="cache_build_pcm_", dir="/dev/shm"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def default_workers(device: str) -> int:
    """Decode and alignment processes: few on the Mac, the CPU budget less the main process on the box."""
    return MLX_WORKERS if device == "mlx" else max(1, math.floor(cpu_budget()) - 1)


# --------------------------------------------------------------------------- MLX


class MLXModels:
    """Mimi (`mimi_mlx`) and wav2vec2 (`wav2vec2_mlx`) on Apple Silicon: one clip per forward
    (batching clips gained nothing in MLX), padded to length buckets, so each compiled forward
    sees few shapes. Mimi runs in float32; wav2vec2 runs in float32 by default or float16 when
    `half_ctc=True` (log-probs cast back to float32 before the Viterbi)."""

    window = 64  # rows queued on the GPU at once

    def __init__(self, mimi_weight: Path, num_codebooks: int, aligner: Aligner, files: Path, half_ctc: bool = False):
        import mlx.core as mx

        from training.prep.mimi_mlx import MimiEncoder
        from training.prep.wav2vec2_mlx import Wav2Vec2CTC

        if aligner.repo != "facebook/wav2vec2-base-960h":
            raise SystemExit(f"wav2vec2_mlx runs only wav2vec2-base-960h, not {aligner.repo}: build on --device cuda")
        mx.set_cache_limit(MLX_CACHE_BYTES)
        self.mimi = MimiEncoder(mimi_weight, num_codebooks)
        self.ctc = Wav2Vec2CTC(files / "model.safetensors", half=half_ctc)
        self.columns = mx.array(vocab_columns(aligner, files))
        self.cardinality = self.mimi.cardinality

    def encode(self, rows: list[Row], align: AlignJob) -> list[tuple[Aligned, Any, Any]]:
        """Queue CTC and Mimi on the GPU; block until CTC is done and workers are submitted (a job a
        row, for the Mac's two workers); return lazy Mimi codes so the caller can overlap planning
        the next window with Mimi."""
        import mlx.core as mx

        log_probs = [mx.take(self.ctc.log_probs(row.target16), self.columns, axis=1) for row in rows]
        codes = [(self.mimi.encode(row.source24), self.mimi.encode(row.target24)) for row in rows]
        mx.async_eval(log_probs)
        mx.async_eval(codes)
        # np.array(lp) blocks until CTC is done; Mimi is queued and runs asynchronously after it
        aligned = [(align([row], [np.array(lp)]), 0) for row, lp in zip(rows, log_probs, strict=True)]
        return [(a, source, target) for a, (source, target) in zip(aligned, codes, strict=True)]


# --------------------------------------------------------------------------- CUDA


def cuda_mimi(mimi_weight: Path, num_codebooks: int, tf32: bool = False) -> Any:
    """moshi's Mimi on CUDA in float32, with TF32 matmuls and convolutions only when `tf32` (a
    process-wide switch; it moves ~3% of residual codes to near-ties, see CUDA_TF32)."""
    import torch
    from moshi.models import loaders

    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    return loaders.get_mimi(mimi_weight, device=torch.device(DEVICE), num_codebooks=num_codebooks).eval()


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


class CUDAModels:
    """Mimi (moshi, float32) and the aligner's wav2vec2 CTC (transformers) on CUDA, over padded
    batches of up to CUDA_BATCH_S of audio, shortest first. CTC runs in float32 by default; pass
    `half_ctc=True` to run its transformer in bfloat16 (feature extractor and lm_head stay
    float32, so GroupNorm stats are safe and log-probs match float32 within CTC rounding)."""

    window = 4096  # rows per batched pass
    threaded = True  # encode runs in the build's GPU thread, a window ahead of the main thread

    def __init__(self, mimi_weight: Path, num_codebooks: int, aligner: Aligner, files: Path, half_ctc: bool = False):
        import torch
        from transformers import Wav2Vec2ForCTC

        self.mimi = cuda_mimi(mimi_weight, num_codebooks, tf32=CUDA_TF32)
        self.cardinality = self.mimi.cardinality
        # cuda_mimi's TF32 switch covers the CTC too (CUDA_TF32).
        self.ctc = Wav2Vec2ForCTC.from_pretrained(files, dtype=torch.float32).to(DEVICE).eval()
        # A base model (wav2vec2-base-960h) normalizes its first convolution over the whole input
        # and takes no attention mask; a large one (omniASR) normalizes each frame and takes one.
        self._group_norm = self.ctc.config.feat_extract_norm == "group"
        if self.ctc.wav2vec2.adapter is not None:  # ctc_log_probs runs the encoder without it
            raise ValueError(f"{aligner.repo} has an adapter, which ctc_log_probs does not apply")
        self._half_ctc = half_ctc
        self._columns = torch.tensor(vocab_columns(aligner, files), device=DEVICE)

    def encode(self, rows: list[Row], align: AlignJob) -> list[tuple[Aligned, np.ndarray, np.ndarray]]:
        """Each row's alignment (`align` given a CTC batch's rows and log-probabilities, as the batch
        lands) and Mimi codes: the CTC first, so the workers align while Mimi encodes; both clips of
        every row in one sort, so Mimi's batches come out fuller."""
        aligned: list[Any] = [None] * len(rows)

        def landed(group: list[int], log_probs: list[np.ndarray]) -> None:
            job = align([rows[i] for i in group], log_probs)
            for place, i in enumerate(group):
                aligned[i] = (job, place)

        self.ctc_log_probs([row.target16 for row in rows], landed)
        codes = self.mimi_codes([row.source24 for row in rows] + [row.target24 for row in rows])
        return list(zip(aligned, codes[: len(rows)], codes[len(rows):], strict=True))

    # The GPU thread's host work (padding a batch into pinned memory, the copies back) runs while
    # the GPU computes the batch before it: `_pipelined` launches batch k+1 before collecting k, and
    # `_stage` fills one of two pinned buffers while the other's copy runs. Done in turn, it held
    # each GigaSpeech 2 window ~16.5 s against ~12 s of kernels (2026-11-09).

    def _pipelined(self, groups: list[list[int]], launch: Callable, collect: Callable) -> None:
        prev = None
        for group in groups:
            handle = launch(group)
            if prev is not None:
                collect(*prev)
            prev = (group, handle)
        if prev is not None:
            collect(*prev)

    def _stage(self, clips: list[np.ndarray], group: list[int], length: int) -> Any:
        """`group`'s clips right-padded to `length` in the next pinned buffer, copied to the GPU
        without blocking."""
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
            n = len(clips[i])
            view[row, 0, :n] = clips[i]
            view[row, 0, n:] = 0.0
        batch = host.to(DEVICE, non_blocking=True)
        self._copied[k] = torch.cuda.Event()
        self._copied[k].record()
        return batch

    @staticmethod
    def _to_host(tensor: Any) -> tuple[Any, Any]:
        """A non-blocking copy to pinned host memory and the event that marks it done."""
        import torch

        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return host, done

    def mimi_codes(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """Each 24 kHz clip's codes, [codebooks, ceil(samples / 1920)] int32, as `mimi_batch`."""
        import torch

        out: list[np.ndarray] = [np.empty(0)] * len(clips)

        def launch(group: list[int]) -> tuple[Any, Any, list[int]]:
            frames = [math.ceil(len(clips[i]) / FRAME_SAMPLES) for i in group]
            batch = self._stage(clips, group, max(frames) * FRAME_SAMPLES)
            with torch.no_grad():
                codes = self.mimi.encode(batch).to(torch.int32)
            return (*self._to_host(codes), frames)

        def collect(group: list[int], handle: tuple[Any, Any, list[int]]) -> None:
            host, done, frames = handle
            done.synchronize()
            codes = host.numpy()
            for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                out[i] = codes[row, :, :count]

        self._pipelined(batches([len(c) for c in clips], int(CUDA_BATCH_S * SAMPLE_RATE)), launch, collect)
        return out

    def ctc_log_probs(self, clips: list[np.ndarray],
                      landed: Callable[[list[int], list[np.ndarray]], None] | None = None) -> list[np.ndarray]:
        """Each normalized 16 kHz clip's CTC log-probabilities [frames, `vocab_columns`], as
        computed alone: the padding is masked out of the positional convolution's input and the
        attention keys, and, for a base model, out of the statistics of its first convolution's
        GroupNorm, which normalizes over the whole input (as wav2vec2_mlx does). `landed(indices,
        log_probs)` hears of each batch's clips as it reaches the host."""
        import torch

        w2v = self.ctc.wav2vec2
        first, rest = w2v.feature_extractor.conv_layers[0], w2v.feature_extractor.conv_layers[1:]
        out: list[np.ndarray] = [np.empty(0)] * len(clips)

        def launch(group: list[int]) -> tuple[Any, Any, list[int]]:
            lengths = torch.tensor([len(clips[i]) for i in group])
            batch = self._stage(clips, group, int(lengths.max()))
            with torch.no_grad():
                if self._group_norm:
                    hs = first.conv(batch)
                    valid = (lengths - first.conv.kernel_size[0]) // first.conv.stride[0] + 1
                    own = (torch.arange(hs.shape[-1])[None, :] < valid[:, None])[:, None, :].to(DEVICE, torch.float32)
                    count = own.sum(-1, keepdim=True)
                    mean = (hs * own).sum(-1, keepdim=True) / count
                    var = ((hs - mean).square() * own).sum(-1, keepdim=True) / count
                    norm = first.layer_norm
                    hs = (hs - mean) / torch.sqrt(var + norm.eps) * norm.weight[:, None] + norm.bias[:, None]
                    hs = first.activation(hs)
                    for layer in rest:
                        hs = layer(hs)
                    hs = hs.transpose(1, 2)
                else:  # each frame normalized alone: the padding's frames never reach a clip's own
                    hs = w2v.feature_extractor(batch[:, 0]).transpose(1, 2)
                hs, _ = w2v.feature_projection(hs)
                frames = self.ctc._get_feat_extract_output_lengths(lengths)
                mask = (torch.arange(hs.shape[1]) < frames[:, None]).to(DEVICE)
                if self._half_ctc:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hs = w2v.encoder(hs, attention_mask=mask).last_hidden_state
                    hs = hs.float()
                else:
                    hs = w2v.encoder(hs, attention_mask=mask).last_hidden_state
                log_probs = torch.log_softmax(self.ctc.lm_head(hs), dim=-1)[..., self._columns]
            return (*self._to_host(log_probs), frames.tolist())

        def collect(group: list[int], handle: tuple[Any, Any, list[int]]) -> None:
            host, done, frames = handle
            done.synchronize()
            log_probs = host.numpy()
            for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                out[i] = log_probs[row, :count]
            if landed is not None:
                landed(group, [out[i] for i in group])

        self._pipelined(batches([len(c) for c in clips], int(CTC_BATCH_S * CTC_RATE)), launch, collect)
        return out


MODELS = {"mlx": MLXModels, "cuda": CUDAModels}  # --device


# --------------------------------------------------------------------------- the build


class Builder:
    """A cache build on one device; `run` turns a builder's source shards into cache shards."""

    def __init__(self, device: str, settings: Settings, config_path: Path, mimi_weight: Path, tokenizer: Path,
                 half_ctc: bool = False):
        import sentencepiece

        self.settings = settings
        self.cfg = read_json(config_path)
        self.tokenizer_path = require_file(tokenizer, "text tokenizer")
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(self.tokenizer_path))
        self.eos_id = int(self.tokenizer.eos_id())
        num_codebooks = max(int(self.cfg["dep_q"]), int(self.cfg["n_q"]) - int(self.cfg["dep_q"]))
        self.models = MODELS[device](require_file(mimi_weight, "Mimi weight"), num_codebooks, settings.aligner,
                                     aligner_files(settings.aligner), half_ctc=half_ctc)
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
        with (
            ThreadPoolExecutor(1) as reads,
            pcm_dir() as pcm,
            ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=cache_rows.init_worker,
                                initargs=(self.settings, self.tokenizer_path, pcm)) as pool,
        ):
            try:
                yield from self._stream(self._read_ahead(keys, read, reads), pool)
            except BaseException:
                # Drop the queued decodes and the next shard's read rather than wait for them.
                pool.shutdown(wait=False, cancel_futures=True)
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

    def _stream(self, shards: Iterator[Shard], pool: ProcessPoolExecutor) -> Iterator[Shard]:
        """The shards filled, their rows planned and encoded in windows that run across shard ends.
        Each window's GPU work is queued before the previous window is collected, so the GPU runs
        continuously: CTC_N → Mimi_N → CTC_N+1 → Mimi_N+1 → … On MLX `encode` returns lazy codes;
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

            for shard, index, row, job, place in self._decoded(shards, pool, started):
                planned = job.result()[place]
                if isinstance(planned, Planned):
                    window.append(planned.row(shard, index, row))
                else:
                    shard.reject(index, row["id"], planned)
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
    ) -> Iterator[tuple[Shard, int, dict[str, Any], Future, int]]:
        """Each kept row with its decode job (`cache_rows.prepare_rows`, DECODE_JOB_ROWS rows of its
        shard) and its place in the job's results, submitted a window ahead (the pool runs jobs in
        order, so more would only queue a window's alignments behind more decodes); a shard joins
        `started` as its first job is submitted."""
        queued: deque[tuple[Shard, int, dict[str, Any], Future, int]] = deque()
        for shard in shards:
            started.append(shard)
            for start in range(0, len(shard.rows), DECODE_JOB_ROWS):
                rows = shard.rows[start: start + DECODE_JOB_ROWS]
                job = pool.submit(cache_rows.prepare_rows, [self._decode_args(row) for _, row in rows], shard.split)
                queued.extend((shard, index, row, job, place) for place, (index, row) in enumerate(rows))
                while len(queued) > self.models.window:
                    yield queued.popleft()
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
