"""The build of grounded caches, on MLX (Apple Silicon) or CUDA: Mimi and wav2vec2-base-960h CTC on
each device, and the stream of source shards through them. Every builder runs on it
(`cache_codes`, the EN-VI dataset; `training.tranche.cache`, the real-Vietnamese tranche); a
builder only reads and gates its own source rows and decides where the shards go
([cache-build](../../docs/agent-memory/cache-build.md)).

  keys ─► read (the builder's), in a thread one shard ahead: rows and the builder's own gates
       ─► decode workers (PyAV, soxr), one window of rows ahead of the device
       ─► cache_rows.plan: decode gates, delay draw, text groups ─► windows, across shard ends
       ─► device: CTC ─► the Viterbi in the workers while the device runs Mimi
       ─► samples back to their shard ─► the shard, in order, once its last row is in
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import cpu_budget, read_json, require_file
from training.prep import cache_rows
from training.prep.audio import Rejected
from training.prep.cache_rows import CTC_RATE, FRAME_SAMPLES, SAMPLE_RATE, Row, Settings, Shard
from training.prep.text_timing import ALIGNER_REPO, ALIGNER_REVISION, AlignmentError, aligner_files

if sys.platform == "darwin":  # before the builders' readers import pyarrow, whose mimalloc pool keeps freed buffers on macOS
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Freed MLX buffers kept for reuse across the ~36 bucketed shapes of Mimi and wav2vec2; on
# libriheavy rows 4, 8 and 16 GB ran 195x, 209x and 225x.
MLX_CACHE_BYTES = 16 << 30
MLX_WORKERS = 2  # 2, 4 and 8 ran alike; the Mac's CPU and GPU share one power budget
CUDA_BATCH_S = 640.0  # padded audio per batch, Mimi and CTC
DEVICE = "cuda"  # the torch device of CUDAModels


def default_workers(device: str) -> int:
    """Decode and alignment processes: few on the Mac, the CPU budget less the main process on the box."""
    return MLX_WORKERS if device == "mlx" else max(1, math.floor(cpu_budget()) - 1)


# --------------------------------------------------------------------------- MLX


class MLXModels:
    """Mimi (`mimi_mlx`) and wav2vec2 (`wav2vec2_mlx`) in float32 on Apple Silicon: clips grouped
    by bucket length and sub-batched up to MAX_BUCKET_BATCH, so each compiled forward sees few
    shapes and peak GPU activation stays bounded."""

    window = 64  # rows queued on the GPU at once

    def __init__(self, mimi_weight: Path, num_codebooks: int, files: Path):
        import mlx.core as mx

        from training.prep.mimi_mlx import MimiEncoder
        from training.prep.wav2vec2_mlx import Wav2Vec2CTC

        mx.set_cache_limit(MLX_CACHE_BYTES)
        self.mimi = MimiEncoder(mimi_weight, num_codebooks)
        self.ctc = Wav2Vec2CTC(files / "model.safetensors")
        self.cardinality = self.mimi.cardinality

    def encode(
        self, rows: list[Row], align: Callable[[Row, np.ndarray], Future]
    ) -> list[tuple[Future, np.ndarray, np.ndarray]]:
        """Each row's alignment (`align` given its CTC log-probabilities) and Mimi codes. CTC runs
        first as one batched pass so the workers align while Mimi encodes both sides."""
        import mlx.core as mx

        log_probs = self.ctc.log_probs_batch([row.en16 for row in rows])
        vi_lazy = self.mimi.encode_batch([row.vi24 for row in rows])
        en_lazy = self.mimi.encode_batch([row.en24 for row in rows])
        mx.eval(*log_probs, *vi_lazy, *en_lazy)
        aligned = [align(row, np.array(lp)) for row, lp in zip(rows, log_probs, strict=True)]
        return [(a, np.array(vi), np.array(en)) for a, vi, en in zip(aligned, vi_lazy, en_lazy, strict=True)]


# --------------------------------------------------------------------------- CUDA


def cuda_mimi(mimi_weight: Path, num_codebooks: int) -> Any:
    """moshi's Mimi on CUDA in full float32: TF32 changes its residual codes as bf16 does."""
    import torch
    from moshi.models import loaders

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return loaders.get_mimi(mimi_weight, device=torch.device(DEVICE), num_codebooks=num_codebooks).eval()


def mimi_batch(mimi: Any, clips: list[Any], device: Any) -> list[Any]:
    """The codes of 24 kHz clips (1-D tensors, on the CPU or `device`) encoded on `device` as one
    right-padded batch: each [codebooks, ceil(samples / 1920)] on the CPU as int32. Padding changes
    nothing before it: Mimi's convolutions and attention are causal."""
    import torch

    frames = [math.ceil(clip.shape[-1] / FRAME_SAMPLES) for clip in clips]
    batch = torch.zeros(len(clips), 1, max(frames) * FRAME_SAMPLES, device=device)
    for row, clip in enumerate(clips):
        batch[row, 0, : clip.shape[-1]] = clip
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
    """Mimi (moshi) and wav2vec2-base-960h CTC (transformers) in full float32 on CUDA, over padded
    batches of up to CUDA_BATCH_S of audio, shortest first."""

    window = 4096  # rows per batched pass

    def __init__(self, mimi_weight: Path, num_codebooks: int, files: Path):
        import torch
        from transformers import Wav2Vec2ForCTC

        self.mimi = cuda_mimi(mimi_weight, num_codebooks)
        self.cardinality = self.mimi.cardinality
        # TF32, off since cuda_mimi, would also move CTC near-ties.
        self.ctc = Wav2Vec2ForCTC.from_pretrained(ALIGNER_REPO, revision=ALIGNER_REVISION,
                                                  dtype=torch.float32).to(DEVICE).eval()
        if self.ctc.config.feat_extract_norm != "group":
            raise RuntimeError("the CTC forward masks the first GroupNorm of a group-norm wav2vec2")

    def encode(
        self, rows: list[Row], align: Callable[[Row, np.ndarray], Future]
    ) -> list[tuple[Future, np.ndarray, np.ndarray]]:
        """Each row's alignment (`align` given its CTC log-probabilities) and Mimi codes: the CTC
        first, so the workers align while Mimi encodes."""
        log_probs = self.ctc_log_probs([row.en16 for row in rows])
        aligned = [align(row, lp) for row, lp in zip(rows, log_probs, strict=True)]
        vi_codes = self.mimi_codes([row.vi24 for row in rows])
        en_codes = self.mimi_codes([row.en24 for row in rows])
        return list(zip(aligned, vi_codes, en_codes, strict=True))

    def mimi_codes(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        out: list[np.ndarray] = [np.empty(0)] * len(clips)
        for group in batches([len(c) for c in clips], int(CUDA_BATCH_S * SAMPLE_RATE)):
            codes = mimi_batch(self.mimi, [torch.from_numpy(clips[i]) for i in group], DEVICE)
            for i, row in zip(group, codes, strict=True):
                out[i] = row.numpy()
        return out

    def ctc_log_probs(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """Each normalized 16 kHz clip's CTC log-probabilities [frames, vocab], as computed alone:
        the base model normalizes its first convolution over the whole input and has no attention
        mask, so the padding is masked out of those statistics, the positional convolution's input
        and the attention keys (as wav2vec2_mlx does)."""
        import torch

        w2v = self.ctc.wav2vec2
        first, rest = w2v.feature_extractor.conv_layers[0], w2v.feature_extractor.conv_layers[1:]
        out: list[np.ndarray] = [np.empty(0)] * len(clips)
        with torch.no_grad():
            for group in batches([len(c) for c in clips], int(CUDA_BATCH_S * CTC_RATE)):
                lengths = torch.tensor([len(clips[i]) for i in group])
                batch = torch.zeros(len(group), 1, int(lengths.max()), device=DEVICE)
                for row, i in enumerate(group):
                    batch[row, 0, : len(clips[i])] = torch.from_numpy(clips[i])
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
                hs, _ = w2v.feature_projection(hs.transpose(1, 2))
                frames = self.ctc._get_feat_extract_output_lengths(lengths)
                mask = (torch.arange(hs.shape[1]) < frames[:, None]).to(DEVICE)
                hs = w2v.encoder(hs, attention_mask=mask).last_hidden_state
                log_probs = torch.log_softmax(self.ctc.lm_head(hs), dim=-1).cpu().numpy()
                for row, (i, count) in enumerate(zip(group, frames.tolist(), strict=True)):
                    out[i] = log_probs[row, :count]
        return out


MODELS = {"mlx": MLXModels, "cuda": CUDAModels}  # --device


# --------------------------------------------------------------------------- the build


class Builder:
    """A cache build on one device; `run` turns a builder's source shards into cache shards."""

    def __init__(self, device: str, settings: Settings, config_path: Path, mimi_weight: Path, tokenizer: Path):
        import sentencepiece

        self.settings = settings
        self.cfg = read_json(config_path)
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(require_file(tokenizer, "text tokenizer")))
        self.eos_id = int(self.tokenizer.eos_id())
        self.files = aligner_files()  # resolved here, so the workers never reach the Hub
        num_codebooks = max(int(self.cfg["dep_q"]), int(self.cfg["n_q"]) - int(self.cfg["dep_q"]))
        self.models = MODELS[device](require_file(mimi_weight, "Mimi weight"), num_codebooks, self.files)
        if self.models.cardinality != int(self.cfg["card"]):
            raise RuntimeError(f"config card {self.cfg['card']} != Mimi cardinality {self.models.cardinality}")

    def payload(self, shard: Shard, source: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return cache_rows.payload(shard, self.settings, self.cfg, source, **extra)

    def run(self, keys: Iterable[Any], read: Callable[[Any], Shard], workers: int) -> Iterator[Shard]:
        """Each key's shard, read by `read`, with its samples and rejections, in order, once its last
        row is in. Only data faults reject a row (`Rejected`, `AlignmentError`); anything else (a
        dead worker's BrokenProcessPool, a device error) stops the build before the shard is
        yielded, so a rerun rebuilds it."""
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"  # the spawned workers' thread pools, one thread each
        with (
            ThreadPoolExecutor(1) as reads,
            ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=cache_rows.init_worker,
                                initargs=(self.files, self.settings.min_alignment_score)) as pool,
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
        """The shards filled, their rows planned and encoded in windows that run across shard ends."""
        started: deque[Shard] = deque()
        window: list[Row] = []
        for shard, index, row, clips in self._decoded(shards, pool, started):
            try:
                window.append(cache_rows.plan(shard, index, row, clips.result(), self.settings, self.tokenizer))
            except (Rejected, AlignmentError) as exc:
                shard.reject(index, row["id"], exc)
            if len(window) == self.models.window:
                self._encode(window, pool)
                window = []
            while started and not started[0].left:
                yield started.popleft().done()
        self._encode(window, pool)
        if any(shard.left for shard in started):
            raise RuntimeError("the build ended with rows neither sampled nor rejected")
        for shard in started:
            yield shard.done()

    def _decoded(
        self, shards: Iterator[Shard], pool: ProcessPoolExecutor, started: deque[Shard]
    ) -> Iterator[tuple[Shard, int, dict[str, Any], Future]]:
        """Each kept row with its decode, submitted a window ahead (the pool runs jobs in order, so
        more would only queue a window's alignments behind more decodes); a shard joins `started`
        as its first decode is submitted."""
        queued: deque[tuple[Shard, int, dict[str, Any], Future]] = deque()
        for shard in shards:
            started.append(shard)
            for index, row in shard.rows:
                clips = pool.submit(cache_rows.prepare, row.pop("vi_audio"), row.pop("en_audio"))
                queued.append((shard, index, row, clips))
                if len(queued) > self.models.window:
                    yield queued.popleft()
        while queued:
            yield queued.popleft()

    def _encode(self, window: list[Row], pool: ProcessPoolExecutor) -> None:
        """The window's codes and text timing, each row's sample or rejection added to its shard."""
        if not window:
            return

        def align(row: Row, log_probs: np.ndarray) -> Future:
            return pool.submit(cache_rows.align, log_probs, row.lead, row.groups,
                               row.en_frames - row.delay_frames, row.delay_frames, self.eos_id)

        for row, (aligned, vi, en) in zip(window, self.models.encode(window, align), strict=True):
            try:
                tokens, text_frames, score = aligned.result()
            except AlignmentError as exc:
                row.shard.reject(row.index, row.source["id"], exc)
                continue
            row.shard.samples.append(cache_rows.sample(row, tokens, text_frames, score, vi, en, self.cfg))
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
