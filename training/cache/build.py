"""The build of grounded caches, on MLX (Apple Silicon) or CUDA: Mimi and the target language's
CTC aligner (`timing.ALIGNERS`) on each device (`mlx`, `cuda`), and the stream of source shards through
them, in either direction (`Settings.direction`). Every
builder runs on it (`en_speech`, the EN-VI dataset; `training.tranche.cache`, the
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
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from training.audio.decode import Rejected
from training.cache import rows as cache_rows  # `rows` names the rows in the code below
from training.cache.cuda import CUDAModels, CUDAModelsDual
from training.cache.mlx import MLXModels
from training.cache.rows import Row, Settings, Shard
from training.cache.timing import AlignmentError, aligner_files
from training.paths import cpu_budget, read_json, require_file

if sys.platform == "darwin":  # before the builders' readers import pyarrow, whose mimalloc pool keeps freed buffers on macOS
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

MLX_WORKERS = 2  # 2, 4 and 8 ran alike; the Mac's CPU and GPU share one power budget
# Rows per decode job (`cache_rows.prepare_rows`); a CTC batch's rows are one alignment job. One job a
# row cost the main process each row's pool round trip under its one GIL, beside the GPU thread.
DECODE_JOB_ROWS = 32


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


MODELS = {"mlx": MLXModels, "cuda": CUDAModels}  # --device


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
    ):
        import sentencepiece

        self.settings = settings
        self.cfg = read_json(config_path)
        self.tokenizer_path = require_file(tokenizer, "text tokenizer")
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(self.tokenizer_path))
        self.eos_id = int(self.tokenizer.eos_id())
        num_codebooks = max(int(self.cfg["dep_q"]), int(self.cfg["n_q"]) - int(self.cfg["dep_q"]))
        if half_ctc and device != "mlx":
            raise ValueError("half_ctc is MLX's float16 wav2vec2: device mlx only (CUDA's CTC always runs in FP16)")
        if ane_ctc and device != "mlx":
            raise ValueError("the Neural Engine CTC (ane_ctc) runs beside MLX: device mlx only")
        if ane_ctc and settings.target != "vi":
            # The English aligner there kept 99.35% of MLX float32's argmaxes (the Vietnamese 99.94%).
            raise ValueError("the Neural Engine CTC (ane_ctc) times Vietnamese text only: direction en-vi")
        mlx_options = {"half_ctc": half_ctc, "ane_ctc": ane_ctc} if device == "mlx" else {}
        self.models = MODELS[device](
            require_file(mimi_weight, "Mimi weight"),
            num_codebooks,
            settings.aligner,
            aligner_files(settings.aligner),
            **mlx_options,
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
        with (
            ThreadPoolExecutor(1) as reads,
            pcm_dir() as pcm,
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=cache_rows.init_worker,
                initargs=(self.settings, self.tokenizer_path, pcm),
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
        with (
            ThreadPoolExecutor(1) as reads,
            pcm_dir() as pcm,
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=cache_rows.init_dual_worker,
                initargs=(self.settings_ve, self.tok_ve_path, self.settings_ev, self.tok_ev_path, pcm),
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
