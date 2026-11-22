#!/usr/bin/env python
"""Noisy copies of the grounded caches: the source of a cache split (either language) re-encoded
under simulated recording conditions, with Mimi on CUDA or MLX (--device).

Each `shard_<name>.pt` of --cache-dir becomes <out-dir>/shard_<name>.pt with the same rows in the
same order. A seeded --fraction of the rows get new source codes: their clip (the cache's source
language), read from the dataset file the shard's own `source` names (`clip_file`), is decoded and
resampled by the cache builder's own code (PyAV, soxr), put under `augment`'s drawn conditions and
Mimi-encoded in float32. The file is an EN-VI dataset parquet (`data/<split>/<name>.parquet`,
`vi_audio` or `en_audio`) for `cache_codes`' caches, and for the tranche's a part of its repo
(`<source>/en_audio/<name>.parquet` beside the segments an `en-vi` shard was built from). Text,
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
import random
import shutil
import tempfile
import time
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

from training.paths import (
    CACHE_FORMAT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DIRECTION,
    DEFAULT_MIMI_WEIGHT,
    EN_VI_REPO,
    EN_VI_REVISION,
    durable,
    exit_with_parent,
    exclusive,
    read_json,
    require_dir,
    require_file,
)
from training.prep import audio, augment, cache_build
from training.prep.cache_rows import FRAME_SAMPLES, Settings
from training.tranche.ingest import REPO as TRANCHE_REPO

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
    fetch.add_argument("--jobs", type=int, default=16, help="Parallel downloads.")
    fetch.add_argument(
        "--dataset-dir",
        type=Path,
        help="Read the parquets from this local copy of the dataset the cache was built from "
        "(each `clip_file` under it: data/<split>/<name>.parquet, or the tranche's "
        "<source>/en_audio/<name>.parquet), instead of downloading them.",
    )
    # A trainer on the same disk needs a checkpoint pair (37.5 GB) beyond its steady state while
    # the recovery sync still uploads the pair it pruned.
    fetch.add_argument(
        "--min-free-gb", type=float, default=50.0, help="Pause while the disk has less free."
    )
    encode.add_argument("--device", choices=list(cache_build.MODELS), default="cuda")
    encode.add_argument("--bank-dir", type=Path, default=Path("data/rirs_noises"))
    encode.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    encode.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    encode.add_argument(
        "--workers",
        type=int,
        help="Decode processes (default: the device's, `cache_build.default_workers`; lower it "
        "when another CPU-bound job shares the machine).",
    )
    encode.add_argument(
        "--batch-seconds", type=float, default=640.0, help="Padded audio per Mimi batch."
    )
    args = parser.parse_args()
    if args.step == "fetch" and not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
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
    built_from = shard_source(shard_path)
    source, column = clip_file(built_from, settings["direction"])
    with tempfile.TemporaryDirectory(dir=args.out_dir / SOURCES, prefix=".download-") as tmp:
        if args.dataset_dir is not None:
            path = require_file(args.dataset_dir / source, "dataset parquet")
        else:
            repo, revision = hub_dataset(built_from)
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    path = Path(
                        hf_hub_download(
                            repo, source, repo_type="dataset", revision=revision, local_dir=tmp
                        )
                    )
                    break
                except Exception as exc:  # network and Hub errors alike; the last one raises
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    print(
                        f"{name}: download attempt {attempt} failed ({exc}); retrying", flush=True
                    )
                    time.sleep(60 * attempt)
        table = pq.read_table(path, columns=["id", column])
        ids = table.column("id").to_pylist()
        clips = table.column(column).combine_chunks().field("bytes")
        keep = pa.array([selected(args.seed, row_id, args.fraction) for row_id in ids])
        slim = pa.table(
            {
                "id": table.column("id"),
                # Named as in the dataset, so encode reads only its own direction's clips; null for
                # the rows that stay clean.
                column: pc.if_else(keep, clips, pa.scalar(None, clips.type)),
            }
        )
        meta = {"file": built_from, "clips": source, "bytes": path.stat().st_size}
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


def decode(data: bytes) -> np.ndarray:
    """Worker: the clip at 24 kHz, decoded and resampled by the cache builder's code, so it has
    the samples and frame count the cache was built from."""
    pcm, rate = audio.decode(data)
    return soxr.resample(pcm, rate, augment.SAMPLE_RATE)


class Encoder:
    """Conditions, Mimi in float32 and the new source codes written into the rows. On CUDA moshi's
    Mimi encodes right-padded batches and the conditions run on the GPU; on MLX `mimi_mlx` encodes
    a batch's clips one per forward (as the cache build does) while the CPU conditions the next
    batch's."""

    def __init__(self, args: argparse.Namespace):
        cfg = read_json(args.config_path)
        self.dep_q = int(cfg["dep_q"])
        self.source_q = int(cfg["n_q"]) - self.dep_q
        self.card = int(cfg["card"])
        weight = require_file(args.mimi_weight, "Mimi weight")
        self.mlx = args.device == "mlx"
        if self.mlx:
            import mlx.core as mx

            from training.prep.mimi_mlx import MimiEncoder

            mx.set_cache_limit(cache_build.MLX_CACHE_BYTES)
            self.device = torch.device("cpu")
            self.mimi = MimiEncoder(weight, self.source_q)
        else:
            self.device = torch.device(cache_build.DEVICE)
            self.mimi = cache_build.cuda_mimi(weight, self.source_q)
        if self.mimi.cardinality != self.card:
            raise RuntimeError(
                f"config card {self.card} != Mimi cardinality {self.mimi.cardinality}"
            )
        self.bank = augment.Bank(augment.fetch_bank(args.bank_dir), self.device)
        self.batch_samples = int(args.batch_seconds * augment.SAMPLE_RATE)

    def shard(self, samples: list[dict[str, Any]], clips: dict[int, Future], frames: str) -> float:
        """Replace the source codes of the rows in `clips` (sample index -> decoded clip), shortest
        first in padded batches; `frames` is the samples' source frame count key (`vi_frames` or
        `en_frames`). Return the seconds of audio encoded. Empties `clips`, so each clip's PCM is
        freed once it is encoded."""
        pending: list[tuple[dict[str, Any], torch.Tensor]] = []
        launched: tuple[list, list] | None = None
        seconds = 0.0
        for index in sorted(clips, key=lambda index: samples[index][frames]):
            sample = samples[index]
            # Shortest first, so this row sets the padded length of the batch it joins.
            if (
                pending
                and (len(pending) + 1) * sample[frames] * FRAME_SAMPLES > self.batch_samples
            ):
                launched = self._launch(pending, launched, frames)
                pending = []
            try:
                pcm = clips.pop(index).result()
            except audio.Rejected as exc:
                raise RuntimeError(f"{sample['id']}: {exc}") from exc
            clip = torch.from_numpy(pcm).to(self.device)
            if math.ceil(clip.shape[-1] / FRAME_SAMPLES) != sample[frames]:
                raise RuntimeError(
                    f"{sample['id']}: {clip.shape[-1]} samples, the cache has "
                    f"{sample[frames]} source frames"
                )
            conditions = augment.Conditions(**sample["source_conditions"])
            pending.append((sample, self.bank.apply(clip, conditions)))
            seconds += clip.shape[-1] / augment.SAMPLE_RATE
        if pending:
            launched = self._launch(pending, launched, frames)
        if launched is not None:
            self._write(*launched, frames)
        return seconds

    def _launch(
        self, pending: list[tuple[dict[str, Any], torch.Tensor]], launched: tuple[list, list] | None, frames: str
    ) -> tuple[list, list]:
        """Start `pending`'s Mimi, then write the batch started before it: on MLX the GPU encodes
        one batch while the CPU conditions the next. Return `pending` with its codes to come."""
        clips = [clip for _, clip in pending]
        if self.mlx:
            import mlx.core as mx

            codes = [self.mimi.encode(clip.numpy()) for clip in clips]
            mx.async_eval(codes)
        else:
            codes = cache_build.mimi_batch(self.mimi, clips, self.device)
        if launched is not None:
            self._write(*launched, frames)
        return pending, codes

    def _write(self, pending: list[tuple[dict[str, Any], torch.Tensor]], codes: list, frames: str) -> None:
        for (sample, _), new in zip(pending, codes, strict=True):
            if self.mlx:
                new = torch.from_numpy(np.array(new))
            source = sample["codes"][1 + self.dep_q : 1 + self.dep_q + self.source_q]
            if int(source[0, sample[frames]]) != self.card:
                raise RuntimeError(f"{sample['id']}: no codec EOS at frame {sample[frames]}")
            source[:, : sample[frames]] = new


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
    """A clean shard with its rows' conditions drawn and their clips queued for decoding."""

    source_path: Path
    payload: dict[str, Any]
    clips: dict[int, Future]  # sample index -> the decoded clip
    frames: str  # the samples' source frame count key


def load(
    args: argparse.Namespace,
    settings: dict[str, Any],
    bank: augment.Bank,
    pool: ProcessPoolExecutor,
    shard_path: Path,
) -> Loaded | None:
    """The shard, its source checked against it, and its selected rows submitted to the decode
    workers; None when its source is missing and no fetch runs."""
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
    if column not in table.column_names or source.get("clips") != clips_file:
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
    # Shortest first, the order Encoder.shard takes them in, so its first batch is ready first.
    order = sorted(clip_rows, key=lambda index: samples[index][frames])
    clips = {index: pool.submit(decode, clip_rows[index]) for index in order}
    return Loaded(source_path, payload, clips, frames)


def encode_shards(
    args: argparse.Namespace, settings: dict[str, Any], split: str, shards: list[Path]
) -> list[str]:
    """Write the noisy shards; return the names of those whose source is missing. Each shard's
    clips are queued before the GPU encodes the shard before it, so the workers decode them
    meanwhile."""
    encoder = Encoder(args)
    # One CPU stays for this process, which feeds the GPU: workers past the budget exhaust a
    # container's quota, and the throttle stalls this process with them.
    workers = args.workers if args.workers is not None else cache_build.default_workers(args.device)
    print(f"{workers} decode workers", flush=True)
    spawn = multiprocessing.get_context("spawn")
    missing: list[str] = []
    with ProcessPoolExecutor(workers, mp_context=spawn, initializer=exit_with_parent) as pool:
        try:
            ahead = load(args, settings, encoder.bank, pool, shards[0])
            last = time.perf_counter()
            for number, shard_path in enumerate(shards, 1):
                shard = ahead
                if number < len(shards):
                    ahead = load(args, settings, encoder.bank, pool, shards[number])
                if shard is None:
                    print(f"{shard_path.name}: no fetched source, skipped", flush=True)
                    missing.append(shard_name(shard_path))
                    last = time.perf_counter()
                    continue
                payload = shard.payload
                count = len(shard.clips)
                seconds = encoder.shard(payload["samples"], shard.clips, shard.frames)
                payload["source_augment"] = {"clean_cache": f"{split}/{shard_path.name}"} | settings
                out_path = args.out_dir / shard_path.name
                partial = out_path.with_name(f".{out_path.name}.tmp")
                torch.save(payload, partial)
                durable(partial)
                partial.replace(out_path)
                durable(args.out_dir)
                shard.source_path.unlink()
                # Since the previous shard: with the next one decoding meanwhile, the pace.
                now = time.perf_counter()
                wall, last = now - last, now
                print(
                    f"{shard_path.name}: {count} of {len(payload['samples'])} rows re-encoded, "
                    f"{seconds / 3600:.2f} h in {wall:.0f} s ({seconds / wall:.0f}x RT) "
                    f"[{number}/{len(shards)}]",
                    flush=True,
                )
        except BaseException:
            # Without this, leaving the pool decodes every queued clip before the error shows.
            pool.shutdown(cancel_futures=True)
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
