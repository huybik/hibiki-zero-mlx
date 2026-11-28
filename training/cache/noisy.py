#!/usr/bin/env python
"""Noisy copies of the grounded caches: the source of a cache split (either language) re-encoded
under simulated recording conditions, with Mimi on CUDA or MLX (--device).

Each `shard_<name>.pt` of --cache-dir becomes <out-dir>/shard_<name>.pt with the same rows in the
same order. A seeded --fraction of the rows get new source codes: their clip (the cache's source
language), read from the dataset file the shard's own `source` names (`clip_file`), is decoded and
resampled by the cache builder's own code (PyAV, soxr), put under `augment`'s drawn conditions and
Mimi-encoded as the cache build encodes (`cuda.cuda_mimi` on CUDA, `mimi_mlx` in float32 on
MLX). The file is an EN-VI dataset parquet (`data/<split>/<name>.parquet`,
`vi_audio` or `en_audio`) for `en_speech`'s caches, and for the tranche's a part of its repo
(`<source>/en_audio/<name>.parquet` beside the segments an `en-vi` shard was built from); a Vietnamese source
built with `en_speech --vi-check` takes the re-voiced clips of the part its `source` names, as that build did. Text,
target codes, frame counts and both EOS positions stay as they were. Each row records its
conditions under `source_conditions` (None for a row kept clean). The noisy second epoch is
--seed 0, --fraction 0.6; another epoch's conditions take another --seed and --out-dir.

Two steps, so the download runs while the GPU still trains:

    fetch  (network, CPU) HF parquet or --dataset-dir's copy ─► .sources/<name>.parquet:
                          id and the source clip's bytes of the selected rows only (26-33% of a
                          parquet's bytes is Vietnamese audio, 67-74% English; 60% of that is kept)
    encode (CUDA or MLX)  .sources + clean shard ─► workers: PyAV decode, soxr to 24 kHz
                          ─► augment (on the GPU on CUDA, on the CPU on MLX), Mimi
                          ─► <out-dir>/shard_<name>.pt; the source is deleted.
                          The next shard's clips decode while the GPU encodes this one.

The first fetch writes <out-dir>/manifest.json (seed, fraction, the cache's direction, the dataset,
bank SHA-256); later fetches require the same, and every encode the same bank and each shard the
manifest's direction. `encode` waits for a source that a running `fetch` has not written yet, and
stops when none is running. Both skip the shards already written, so a rerun continues.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import math
import multiprocessing
import os
import random
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soxr
import torch
from huggingface_hub import hf_hub_download

from training.audio import augment
from training.audio import decode as audio  # `decode` is this module's own
from training.cache import build, cuda
from training.cache.en_speech import vi_checked
from training.cache.mlx import MLX_CACHE_BYTES
from training.cache.rows import FRAME_SAMPLES, Settings
from training.paths import (
    CACHE_FORMAT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DIRECTION,
    DEFAULT_MIMI_WEIGHT,
    EN_VI_REPO,
    EN_VI_REVISION,
    durable,
    exclusive,
    exit_with_parent,
    read_json,
    require_dir,
    require_file,
)
from training.tranche.repo import REPO as TRANCHE_REPO

SOURCES = ".sources"
FETCH_LOCK = ".fetch.lock"
ENCODE_LOCK = ".encode.lock"
MANIFEST = "manifest.json"
META_KEY = b"noisy_cache"  # in each source's parquet metadata, as fetched
WAIT_S = 30
DOWNLOAD_ATTEMPTS = 4  # one failed download would leave its shard unwritten overnight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    fetch = steps.add_parser("fetch", help="Download the selected rows' source audio.")
    encode = steps.add_parser("encode", help="Write the noisy shards on the GPU.")
    for step in (fetch, encode):
        step.add_argument("--cache-dir", type=Path, required=True, help="A clean cache split.")
        step.add_argument("--out-dir", type=Path, required=True)
    fetch.add_argument("--fraction", type=float, default=0.6, help="Rows given new conditions.")
    fetch.add_argument("--seed", type=int, default=0)
    fetch.add_argument("--jobs", type=int, default=None, help="Parallel downloads (default: 16 Hub, 4 local --dataset-dir).")
    fetch.add_argument(
        "--dataset-dir",
        type=Path,
        help="Read the parquets from this local copy of the dataset the cache was built from "
        "(each `clip_file` under it: data/<split>/<name>.parquet, or the tranche's "
        "<source>/en_audio/<name>.parquet; a vi_check part at en_speech/vi_check/<name>.parquet), "
        "instead of downloading them.",
    )
    # A trainer on the same disk needs a checkpoint pair (37.5 GB) beyond its steady state while
    # the recovery sync still uploads the pair it pruned.
    fetch.add_argument(
        "--min-free-gb", type=float, default=50.0, help="Pause while the disk has less free."
    )
    encode.add_argument("--device", choices=list(build.MODELS), default="cuda")
    encode.add_argument("--bank-dir", type=Path, default=Path("data/rirs_noises"))
    encode.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    encode.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    encode.add_argument(
        "--workers",
        type=int,
        help="Decode processes (default: the device's, `build.default_workers`; lower it "
        "when another CPU-bound job shares the machine).",
    )
    encode.add_argument(
        "--batch-seconds", type=float, default=640.0, help="Padded audio per Mimi batch."
    )
    args = parser.parse_args()
    if args.step == "fetch":
        if not 0 < args.fraction <= 1:
            parser.error("--fraction must be in (0, 1]")
        if args.jobs is None:
            args.jobs = 4 if args.dataset_dir is not None else 16
    return args


def selected(seed: int, row_id: str, fraction: float) -> bool:
    return random.Random(f"{seed}:select:{row_id}").random() < fraction


def shard_name(shard_path: Path) -> str:
    return shard_path.stem.removeprefix("shard_")


def todo(cache_dir: Path, out_dir: Path) -> list[Path]:
    """The clean shards whose noisy shard is not written yet, in path order."""
    return [
        path for path in sorted(cache_dir.glob("shard_*.pt")) if not (out_dir / path.name).exists()
    ]


def cache_direction(cache_dir: Path) -> str:
    """The direction of a clean cache split, read from its first shard."""
    first = next(iter(sorted(cache_dir.glob("shard_*.pt"))), None)
    if first is None:
        raise RuntimeError(f"{cache_dir} holds no shard_*.pt")
    return torch.load(first, map_location="cpu", mmap=True).get("direction", DEFAULT_DIRECTION)


def shard_source(shard_path: Path) -> str:
    """The dataset file a clean shard was built from (its payload's `source`)."""
    return torch.load(shard_path, map_location="cpu", mmap=True)["source"]["file"]


def clip_file(source_file: str, direction: str) -> tuple[str, str]:
    """The dataset file holding the source clips of a shard built from `source_file`, and their
    column: an EN-VI dataset parquet (`data/<split>/<name>.parquet`) holds both languages'
    (`vi_audio`, `en_audio`); a tranche shard's (`<source>/segments/<name>.parquet`) English
    voice is its `en_audio` part, rows in the segments' order. A tranche `vi-en` cache's source is
    real speech, which these copies do not re-encode."""
    language = Settings(direction=direction).source
    if source_file.startswith("data/"):
        return source_file, f"{language}_audio"
    source, part, name = source_file.split("/")
    if part != "segments" or language != "en":
        raise RuntimeError(f"no noisy source for a {direction} shard built from {source_file}")
    return f"{source}/en_audio/{name}", "en_audio"


def hub_dataset(source_file: str) -> tuple[str, str | None]:
    """The Hub dataset repo and revision holding a shard's `clip_file`: the EN-VI dataset at its
    pinned revision, or the tranche's repo."""
    return (EN_VI_REPO, EN_VI_REVISION) if source_file.startswith("data/") else (TRANCHE_REPO, None)


def dataset(args: argparse.Namespace, cache_dir: Path) -> str:
    """Where a fetch reads the clips: a local copy, or the Hub repo (at its pinned revision, if any)."""
    if args.dataset_dir is not None:
        return str(args.dataset_dir)
    first = next(iter(sorted(cache_dir.glob("shard_*.pt"))))
    repo, revision = hub_dataset(shard_source(first))
    return repo if revision is None else f"{repo}@{revision}"


def manifest(args: argparse.Namespace, cache_dir: Path) -> dict[str, Any]:
    """The out-dir's settings, written by its first fetch. Every run needs the same bank, and a
    fetch the same seed, fraction, cache direction and dataset."""
    path = args.out_dir / MANIFEST
    needed: dict[str, Any] = {"bank_sha256": augment.BANK_SHA256}
    if args.step == "fetch":
        needed |= {
            "seed": args.seed,
            "fraction": args.fraction,
            "direction": cache_direction(cache_dir),
            "dataset": dataset(args, cache_dir),
        }
        if not path.exists():
            partial = path.with_name(f".{path.name}.tmp")
            partial.write_text(json.dumps(needed, indent=2) + "\n")
            durable(partial)
            partial.replace(path)
    found = read_json(path)
    if missing := {"seed", "fraction", "direction", "dataset"} - found.keys():
        raise RuntimeError(f"{path} lacks {sorted(missing)}: fetch into a new --out-dir")
    if any(found.get(key) != value for key, value in needed.items()):
        raise RuntimeError(f"{path} holds {found}, this run needs {needed}")
    return found


def held(lock: Path) -> bool:
    """Whether a running process holds the `exclusive` lock at `lock`."""
    if not lock.exists():
        return False
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


# --------------------------------------------------------------------------- fetch


def fetch_one(args: argparse.Namespace, settings: dict[str, Any], shard_path: Path) -> str:
    """Write the selected rows' source audio of one shard's `clip_file` to .sources."""
    name = shard_name(shard_path)
    target = args.out_dir / SOURCES / f"{name}.parquet"
    if target.exists():
        return f"{name}: fetched before"
    if shutil.disk_usage(args.out_dir).free < args.min_free_gb * 1e9:
        print(f"{name}: waiting for {args.min_free_gb:.0f} GB of free disk", flush=True)
        while shutil.disk_usage(args.out_dir).free < args.min_free_gb * 1e9:
            time.sleep(WAIT_S)
    built = torch.load(shard_path, map_location="cpu", mmap=True)["source"]
    built_from = built["file"]
    source, column = clip_file(built_from, settings["direction"])
    vi_check = built.get("vi_check") if column == "vi_audio" else None

    def local(file: str, repo: str, revision: str | None, tmp: str) -> Path:
        if args.dataset_dir is not None:
            return require_file(args.dataset_dir / file, "dataset parquet")
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                return Path(hf_hub_download(repo, file, repo_type="dataset", revision=revision, local_dir=tmp))
            except Exception as exc:  # network and Hub errors alike; the last one raises
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                print(f"{name}: download attempt {attempt} failed ({exc}); retrying", flush=True)
                time.sleep(60 * attempt)
        raise AssertionError("unreachable: the last attempt raises")

    with tempfile.TemporaryDirectory(dir=args.out_dir / SOURCES, prefix=".download-") as tmp:
        path = local(source, *hub_dataset(built_from), tmp)
        table = pq.read_table(path, columns=["id", column])
        ids = table.column("id").to_pylist()
        clips = table.column(column).combine_chunks().field("bytes")
        if vi_check is not None:  # as the clean build read them: its rejected rows are in no sample
            check = local(vi_check["file"], TRANCHE_REPO, None, tmp)
            if check.stat().st_size != vi_check["bytes"]:
                raise RuntimeError(f"{vi_check['file']} is {check.stat().st_size} bytes, not the build's {vi_check['bytes']}")
            rows = vi_checked(table.to_pylist(), check)
            clips = pa.array([r and r["vi_audio"]["bytes"] for r in rows], clips.type)
        keep = pa.array([selected(args.seed, row_id, args.fraction) for row_id in ids])
        slim = pa.table(
            {
                "id": table.column("id"),
                # Named as in the dataset, so encode reads only its own direction's clips; null for
                # the rows that stay clean.
                column: pc.if_else(keep, clips, pa.scalar(None, clips.type)),
            }
        )
        meta = {"file": built_from, "clips": source, "bytes": path.stat().st_size, "vi_check": vi_check}
        slim = slim.replace_schema_metadata({META_KEY: json.dumps(meta)})
        partial = target.with_name(f".{target.name}.tmp")
        pq.write_table(slim, partial)
        durable(partial)
        partial.replace(target)
    return f"{name}: {sum(keep.to_pylist())} of {len(ids)} rows"


def fetch(args: argparse.Namespace) -> None:
    cache_dir = require_dir(args.cache_dir, "clean cache split")
    if args.dataset_dir is not None:
        # Resolved, so the manifest names the copy one way however it is typed.
        args.dataset_dir = require_dir(args.dataset_dir, "dataset copy").resolve()
    (args.out_dir / SOURCES).mkdir(parents=True, exist_ok=True)
    shards = todo(cache_dir, args.out_dir)
    print(f"{len(shards)} shards to fetch for {args.out_dir}", flush=True)
    busy = f"another noisy_cache fetch uses {args.out_dir}"
    with (
        exclusive(args.out_dir / FETCH_LOCK, busy),
        ThreadPoolExecutor(args.jobs) as pool,
    ):
        settings = manifest(args, cache_dir)
        # What a killed fetch left: downloads and unrenamed sources.
        for stale in (args.out_dir / SOURCES).glob(".*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        jobs = [pool.submit(fetch_one, args, settings, path) for path in shards]
        for number, job in enumerate(jobs, 1):
            print(f"{job.result()} [{number}/{len(jobs)}]", flush=True)


# --------------------------------------------------------------------------- encode

_noisy_pcm_dir: Path | None = None
DECODE_JOB_ROWS = build.DECODE_JOB_ROWS


def _init_decode_worker(pcm_dir: Path | None) -> None:
    exit_with_parent()
    global _noisy_pcm_dir
    _noisy_pcm_dir = pcm_dir


def decode(data: bytes) -> np.ndarray:
    """Worker: the clip at 24 kHz, decoded and resampled by the cache builder's code, so it has
    the samples and frame count the cache was built from."""
    pcm, rate = audio.decode(data)
    return soxr.resample(pcm, rate, augment.SAMPLE_RATE).astype(np.float32, copy=False)


def decode_batch(
    items: list[tuple[int, bytes]]
) -> tuple[str | None, list[tuple[int, int, int] | tuple[int, np.ndarray]]]:
    """Worker: decode a 32-clip job into one `/dev/shm` slab file (or inline arrays when no
    `/dev/shm` exists), avoiding per-clip IPC and pickle pipe overhead."""
    if _noisy_pcm_dir is None:
        return None, [(idx, decode(data)) for idx, data in items]
    chunks: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    offset = 0
    for idx, data in items:
        pcm = np.ascontiguousarray(decode(data), np.float32)
        chunks.append(pcm)
        meta.append((idx, offset, len(pcm)))
        offset += len(pcm)
    slab_path = str(_noisy_pcm_dir / f"{uuid.uuid4().hex}.f32")
    with open(slab_path, "wb") as f:
        for c in chunks:
            c.tofile(f)
    return slab_path, meta


class Encoder:
    """Conditions, Mimi (FastMimiEncoder on CUDA, mimi_mlx on MLX) and the new source codes
    written into the rows. On CUDA uses double-buffered pinned H2D/D2H, batched GPU
    `Bank.apply_batch`, and pipelined batch k+1 launches."""

    def __init__(self, args: argparse.Namespace):
        cfg = read_json(args.config_path)
        self.dep_q = int(cfg["dep_q"])
        self.source_q = int(cfg["n_q"]) - self.dep_q
        self.card = int(cfg["card"])
        weight = require_file(args.mimi_weight, "Mimi weight")
        self.mlx = args.device == "mlx"
        if self.mlx:
            import mlx.core as mx

            from training.cache.mimi_mlx import MimiEncoder

            mx.set_cache_limit(MLX_CACHE_BYTES)
            self.device = torch.device("cpu")
            self.mimi = MimiEncoder(weight, self.source_q)
        else:
            self.device = torch.device(cuda.DEVICE)
            self.mimi = cuda.cuda_mimi(weight, self.source_q)
        if self.mimi.cardinality != self.card:
            raise RuntimeError(
                f"config card {self.card} != Mimi cardinality {self.mimi.cardinality}"
            )
        self.bank = augment.Bank(augment.fetch_bank(args.bank_dir), self.device)
        self.batch_samples = int(args.batch_seconds * augment.SAMPLE_RATE)
        self._pinned = [torch.empty(0), torch.empty(0)]
        self._copied: list[Any] = [None, None]
        self._turn = 0
        self._out_pool = cuda._PinnedOutputPool()

    def _stage_cuda(self, pcms: list[np.ndarray], padded_len: int) -> torch.Tensor:
        k = self._turn
        self._turn ^= 1
        if self._copied[k] is not None:
            self._copied[k].synchronize()
        need = len(pcms) * padded_len
        if self._pinned[k].numel() < need:
            self._pinned[k] = torch.empty(max(need, self.batch_samples), dtype=torch.float32, pin_memory=True)
        host = self._pinned[k][:need].view(len(pcms), padded_len)
        view = host.numpy()
        for row, pcm in enumerate(pcms):
            n = len(pcm)
            view[row, :n] = pcm
            if n < padded_len:
                view[row, n:] = 0.0
        batch = host.to(self.device, non_blocking=True)
        self._copied[k] = torch.cuda.Event()
        self._copied[k].record()
        return batch

    def shard(self, samples: list[dict[str, Any]], jobs: list[Future], frames: str) -> float:
        """Replace the source codes of all decoded clips from `jobs` (submitted shortest-first in
        32-clip batches). Return the seconds of audio encoded."""
        pending: list[tuple[dict[str, Any], np.ndarray]] = []
        launched: Any = None
        seconds = 0.0

        def flush_cuda(
            batch_items: list[tuple[dict[str, Any], np.ndarray]], prev: Any
        ) -> tuple[list[dict[str, Any]], list[int], Any, Any]:
            b_samples = [s for s, _ in batch_items]
            b_pcms = [p for _, p in batch_items]
            b_lens = [len(p) for p in b_pcms]
            b_frames = [s[frames] for s in b_samples]
            padded_len = max(b_frames) * FRAME_SAMPLES
            conds = [augment.Conditions(**s["source_conditions"]) for s in b_samples]
            wav = self._stage_cuda(b_pcms, padded_len)
            with torch.no_grad():
                aug = self.bank.apply_batch(wav, b_lens, conds)
                codes = self.mimi.encode(aug.unsqueeze(1)).to(torch.int32)
            host, done = self._out_pool.copy_from(codes)
            if prev is not None:
                self._collect_cuda(*prev)
            return b_samples, b_frames, host, done

        def flush_mlx(
            batch_items: list[tuple[dict[str, Any], np.ndarray]], prev: Any
        ) -> tuple[list[dict[str, Any]], list[Any]]:
            import mlx.core as mx

            b_samples = [s for s, _ in batch_items]
            codes = []
            for s, pcm in batch_items:
                cond = augment.Conditions(**s["source_conditions"])
                aug = self.bank.apply(torch.from_numpy(pcm), cond)
                codes.append(self.mimi.encode(aug.numpy()))
            mx.async_eval(codes)
            if prev is not None:
                self._collect_mlx(*prev, frames)
            return b_samples, codes

        for job in jobs:
            try:
                slab_path, entries = job.result()
            except audio.Rejected as exc:
                raise RuntimeError(str(exc)) from exc
            slab = None
            if slab_path is not None:
                slab = np.fromfile(slab_path, np.float32)
                os.unlink(slab_path)
            for entry in entries:
                if slab is not None:
                    index, off, length = entry  # type: ignore[misc]
                    pcm = slab[off : off + length]
                else:
                    index, pcm = entry  # type: ignore[misc]
                sample = samples[index]
                if math.ceil(len(pcm) / FRAME_SAMPLES) != sample[frames]:
                    raise RuntimeError(
                        f"{sample['id']}: {len(pcm)} samples, the cache has "
                        f"{sample[frames]} source frames"
                    )
                if (
                    pending
                    and (len(pending) + 1) * sample[frames] * FRAME_SAMPLES > self.batch_samples
                ):
                    launched = flush_mlx(pending, launched) if self.mlx else flush_cuda(pending, launched)
                    pending = []
                pending.append((sample, pcm))
                seconds += len(pcm) / augment.SAMPLE_RATE
        if pending:
            launched = flush_mlx(pending, launched) if self.mlx else flush_cuda(pending, launched)
        if launched is not None:
            if self.mlx:
                self._collect_mlx(*launched, frames)
            else:
                self._collect_cuda(*launched)
        return seconds

    def _collect_cuda(
        self, b_samples: list[dict[str, Any]], b_frames: list[int], host: torch.Tensor, done: Any
    ) -> None:
        done.synchronize()
        arr = host.numpy()
        for row, (sample, count) in enumerate(zip(b_samples, b_frames, strict=True)):
            source = sample["codes"][1 + self.dep_q : 1 + self.dep_q + self.source_q]
            if int(source[0, count]) != self.card:
                raise RuntimeError(f"{sample['id']}: no codec EOS at frame {count}")
            source[:, :count] = torch.from_numpy(arr[row, :, :count].copy())

    def _collect_mlx(self, b_samples: list[dict[str, Any]], codes: list[Any], frames: str) -> None:
        for sample, new in zip(b_samples, codes, strict=True):
            count = sample[frames]
            source = sample["codes"][1 + self.dep_q : 1 + self.dep_q + self.source_q]
            if int(source[0, count]) != self.card:
                raise RuntimeError(f"{sample['id']}: no codec EOS at frame {count}")
            source[:, :count] = torch.from_numpy(np.array(new))


def wait_for_source(out_dir: Path, name: str) -> Path | None:
    """The fetched source of shard `name`, waiting while a fetch runs; None when it is missing
    and no fetch is running."""
    path = out_dir / SOURCES / f"{name}.parquet"
    while not path.exists():
        # Checked again after the lock: the fetch may have written it and exited in between.
        if not held(out_dir / FETCH_LOCK) and not path.exists():
            return None
        time.sleep(WAIT_S)
    return path


def encode(args: argparse.Namespace) -> None:
    cache_dir = require_dir(args.cache_dir, "clean cache split")
    split = cache_dir.name
    busy = f"another noisy_cache encode uses {args.out_dir}"
    with exclusive(args.out_dir / ENCODE_LOCK, busy):
        settings = manifest(args, cache_dir)
        # Sources of shards already written: a stop between the shard's rename and the unlink.
        for written in sorted(args.out_dir.glob("shard_*.pt")):
            (args.out_dir / SOURCES / f"{shard_name(written)}.parquet").unlink(missing_ok=True)
        shards = todo(cache_dir, args.out_dir)
        print(f"{len(shards)} shards to write under {args.out_dir}", flush=True)
        missing = encode_shards(args, settings, split, shards) if shards else []
        if missing:
            raise SystemExit(
                f"{len(missing)} shards lack a fetched source and were not written "
                f"({', '.join(missing[:5])}...): run fetch, then encode again"
            )
        print(f"{args.out_dir}: every shard written", flush=True)


@dataclasses.dataclass
class Loaded:
    """A clean shard with its rows' conditions drawn and their clips queued in 32-clip decode jobs."""

    source_path: Path
    payload: dict[str, Any]
    jobs: list[Future]
    count: int
    frames: str  # the samples' source frame count key


def load(
    args: argparse.Namespace,
    settings: dict[str, Any],
    bank: augment.Bank,
    pool: ProcessPoolExecutor,
    shard_path: Path,
) -> Loaded | None:
    """The shard, its source checked against it, and its selected rows submitted in 32-clip jobs
    shortest-first to the decode workers; None when its source is missing and no fetch runs."""
    source_path = wait_for_source(args.out_dir, shard_name(shard_path))
    if source_path is None:
        return None
    table = pq.read_table(source_path)
    source = json.loads(table.schema.metadata[META_KEY])
    payload = torch.load(shard_path, map_location="cpu")
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"{shard_path} is not a {CACHE_FORMAT} shard")
    direction = payload.get("direction", DEFAULT_DIRECTION)
    if direction != settings["direction"]:
        raise RuntimeError(f"{shard_path} is {direction}, the fetch {settings['direction']}")
    built_from = payload["source"]
    clips_file, column = clip_file(built_from["file"], direction)
    vi_check = built_from.get("vi_check") if column == "vi_audio" else None
    if column not in table.column_names or source.get("clips") != clips_file or source.get("vi_check") != vi_check:
        raise RuntimeError(
            f"{source_path} holds {table.column_names} of {source.get('clips')}, not {column} of "
            f"{clips_file}: delete it, then fetch again"
        )
    ids = table.column("id").to_pylist()
    # An EN-VI dataset shard names its parquet's size, a tranche shard its segments' row count.
    if (
        source["file"] != built_from["file"]
        or built_from.get("bytes", source["bytes"]) != source["bytes"]
        or built_from.get("rows", len(ids)) != len(ids)
    ):
        raise RuntimeError(f"{shard_path} was built from {built_from}, not {source}")
    audio_bytes = table.column(column).to_pylist()
    samples = payload["samples"]
    frames = f"{Settings(direction=direction).source}_frames"
    clip_rows: dict[int, bytes] = {}
    for index, sample in enumerate(samples):
        file, row = sample["source_row"].split("#r")
        row = int(row)
        if file != source["file"] or ids[row] != sample["id"]:
            raise RuntimeError(f"{sample['source_row']} holds {ids[row]}, not {sample['id']}")
        sample["source_conditions"] = None
        if audio_bytes[row] is None:
            continue
        conditions = bank.draw(f"{settings['seed']}:{sample['id']}")
        sample["source_conditions"] = dataclasses.asdict(conditions)
        clip_rows[index] = audio_bytes[row]
    # Shortest first, in 32-clip decode jobs, so Encoder.shard consumes jobs in sorted order.
    order = sorted(clip_rows, key=lambda index: samples[index][frames])
    jobs = [
        pool.submit(decode_batch, [(idx, clip_rows[idx]) for idx in order[start : start + DECODE_JOB_ROWS]])
        for start in range(0, len(order), DECODE_JOB_ROWS)
    ]
    return Loaded(source_path, payload, jobs, len(order), frames)


def _save_noisy_shard(payload: dict[str, Any], out_path: Path, source_path: Path) -> None:
    partial = out_path.with_name(f".{out_path.name}.tmp")
    torch.save(payload, partial)
    durable(partial)
    partial.replace(out_path)
    durable(out_path.parent)
    source_path.unlink()


def encode_shards(
    args: argparse.Namespace, settings: dict[str, Any], split: str, shards: list[Path]
) -> list[str]:
    """Write the noisy shards; return the names of those whose source is missing. Reads the next
    shard in a background loader thread and writes the finished shard in a background writer
    thread so neither `pq.read_table`/`torch.load` nor `torch.save`/`durable` stalls the GPU."""
    encoder = Encoder(args)
    workers = args.workers if args.workers is not None else build.default_workers(args.device)
    print(f"{workers} decode workers", flush=True)
    spawn = multiprocessing.get_context("spawn")
    missing: list[str] = []
    with (
        build.pcm_dir() as pcm,
        ProcessPoolExecutor(
            workers, mp_context=spawn, initializer=_init_decode_worker, initargs=(pcm,)
        ) as pool,
        ThreadPoolExecutor(1, thread_name_prefix="noisy-load") as loader,
        ThreadPoolExecutor(1, thread_name_prefix="noisy-save") as saver,
    ):
        try:
            ahead: Future | None = loader.submit(load, args, settings, encoder.bank, pool, shards[0])
            saved: Future | None = None
            last = time.perf_counter()
            for number, shard_path in enumerate(shards, 1):
                shard = ahead.result()
                ahead = (
                    loader.submit(load, args, settings, encoder.bank, pool, shards[number])
                    if number < len(shards)
                    else None
                )
                if shard is None:
                    print(f"{shard_path.name}: no fetched source, skipped", flush=True)
                    missing.append(shard_name(shard_path))
                    last = time.perf_counter()
                    continue
                payload = shard.payload
                count = shard.count
                seconds = encoder.shard(payload["samples"], shard.jobs, shard.frames)
                payload["source_augment"] = {"clean_cache": f"{split}/{shard_path.name}"} | settings
                if saved is not None:
                    saved.result()
                saved = saver.submit(_save_noisy_shard, payload, args.out_dir / shard_path.name, shard.source_path)
                now = time.perf_counter()
                wall, last = now - last, now
                print(
                    f"{shard_path.name}: {count} of {len(payload['samples'])} rows re-encoded, "
                    f"{seconds / 3600:.2f} h in {wall:.0f} s ({seconds / wall:.0f}x RT) "
                    f"[{number}/{len(shards)}]",
                    flush=True,
                )
            if saved is not None:
                saved.result()
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            loader.shutdown(wait=False, cancel_futures=True)
            saver.shutdown(wait=False, cancel_futures=True)
            raise
    return missing


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.step == "fetch":
        fetch(args)
    else:
        encode(args)


if __name__ == "__main__":
    main()
