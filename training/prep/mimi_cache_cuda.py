#!/usr/bin/env python
"""Mimi on CUDA for the grounded caches: re-encode the Vietnamese source of a cache split.

Each `shard_<name>.pt` of --cache-dir becomes <out-dir>/shard_<name>.pt with the same rows in the
same order. A seeded --fraction of the rows get new Vietnamese source codes: their clip, taken
from the EN-VI dataset's parquet shard that the cache shard was built from, is decoded and
resampled by the cache builder's own code (PyAV, soxr), put under `augment`'s drawn conditions
and Mimi-encoded in float32. Text, English codes, frame counts and both EOS positions stay as
they were. Each row records its conditions under `source_conditions` (None for a row kept
clean). The noisy second epoch is --seed 0, --fraction 0.6; another epoch's conditions take
another --seed and --out-dir.

Two steps, so the download runs while the GPU still trains:

    fetch  (network, CPU) HF parquet ─► <out-dir>/.sources/<name>.parquet: id and the Opus bytes
                          of the selected rows only (30% of a parquet's bytes is Vietnamese
                          audio; 60% of that is kept)
    encode (CUDA)         .sources + clean shard ─► workers: PyAV decode, soxr to 24 kHz
                          ─► GPU: augment, Mimi ─► <out-dir>/shard_<name>.pt; the source is deleted.
                          The next shard's clips decode while the GPU encodes this one.

The first fetch writes <out-dir>/manifest.json (seed, fraction, dataset revision, bank SHA-256);
later fetches and every encode require the same. `encode` waits for a source that a running
`fetch` has not written yet, and stops when none is running. Both skip the shards already written,
so a rerun continues.
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
    DEFAULT_MIMI_WEIGHT,
    cpu_budget,
    durable,
    exclusive,
    read_json,
    require_dir,
    require_file,
)
from training.prep import audio, augment, cache_build
from training.prep.cache_rows import FRAME_SAMPLES
from training.prep.eval_sets import EN_VI_REPO, EN_VI_REVISION

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
    fetch = steps.add_parser("fetch", help="Download the selected rows' Vietnamese audio.")
    encode = steps.add_parser("encode", help="Write the noisy shards on CUDA.")
    for step in (fetch, encode):
        step.add_argument("--cache-dir", type=Path, required=True, help="A clean cache split.")
        step.add_argument("--out-dir", type=Path, required=True)
    fetch.add_argument("--fraction", type=float, default=0.6, help="Rows given new conditions.")
    fetch.add_argument("--seed", type=int, default=0)
    fetch.add_argument("--jobs", type=int, default=16, help="Parallel downloads.")
    # A trainer on the same disk needs a checkpoint pair (37.5 GB) beyond its steady state while
    # the recovery sync still uploads the pair it pruned.
    fetch.add_argument(
        "--min-free-gb", type=float, default=50.0, help="Pause while the disk has less free."
    )
    encode.add_argument("--bank-dir", type=Path, default=Path("data/rirs_noises"))
    encode.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    encode.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    encode.add_argument(
        "--workers",
        type=int,
        help="Decode processes (default: the CPU budget less one; lower it when another CPU-bound "
        "job shares the machine).",
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


def manifest(args: argparse.Namespace) -> dict[str, Any]:
    """The out-dir's settings, written by its first fetch. Every run needs the same dataset
    revision and bank, and a fetch the same seed and fraction."""
    path = args.out_dir / MANIFEST
    needed = {"dataset_revision": EN_VI_REVISION, "bank_sha256": augment.BANK_SHA256}
    if args.step == "fetch":
        needed |= {"seed": args.seed, "fraction": args.fraction}
        if not path.exists():
            partial = path.with_name(f".{path.name}.tmp")
            partial.write_text(json.dumps(needed, indent=2) + "\n")
            durable(partial)
            partial.replace(path)
    found = read_json(path)
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


def fetch_one(args: argparse.Namespace, split: str, shard_path: Path) -> str:
    """Write the selected rows' audio of one shard's parquet to .sources."""
    name = shard_name(shard_path)
    target = args.out_dir / SOURCES / f"{name}.parquet"
    if target.exists():
        return f"{name}: fetched before"
    if shutil.disk_usage(args.out_dir).free < args.min_free_gb * 1e9:
        print(f"{name}: waiting for {args.min_free_gb:.0f} GB of free disk", flush=True)
        while shutil.disk_usage(args.out_dir).free < args.min_free_gb * 1e9:
            time.sleep(WAIT_S)
    source = f"data/{split}/{name}.parquet"
    with tempfile.TemporaryDirectory(dir=args.out_dir / SOURCES, prefix=".download-") as tmp:
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                path = Path(
                    hf_hub_download(
                        EN_VI_REPO,
                        source,
                        repo_type="dataset",
                        revision=EN_VI_REVISION,
                        local_dir=tmp,
                    )
                )
                break
            except Exception as exc:  # network and Hub errors alike; the last one raises
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                print(f"{name}: download attempt {attempt} failed ({exc}); retrying", flush=True)
                time.sleep(60 * attempt)
        table = pq.read_table(path, columns=["id", "vi_audio"])
        ids = table.column("id").to_pylist()
        opus = table.column("vi_audio").combine_chunks().field("bytes")
        keep = pa.array([selected(args.seed, row_id, args.fraction) for row_id in ids])
        slim = pa.table(
            {
                "id": table.column("id"),
                # Null for the rows that stay clean.
                "vi_opus": pc.if_else(keep, opus, pa.scalar(None, opus.type)),
            }
        )
        meta = {"file": source, "bytes": path.stat().st_size}
        slim = slim.replace_schema_metadata({META_KEY: json.dumps(meta)})
        partial = target.with_name(f".{target.name}.tmp")
        pq.write_table(slim, partial)
        durable(partial)
        partial.replace(target)
    return f"{name}: {sum(keep.to_pylist())} of {len(ids)} rows"


def fetch(args: argparse.Namespace) -> None:
    cache_dir = require_dir(args.cache_dir, "clean cache split")
    (args.out_dir / SOURCES).mkdir(parents=True, exist_ok=True)
    shards = todo(cache_dir, args.out_dir)
    print(f"{len(shards)} shards to fetch for {args.out_dir}", flush=True)
    busy = f"another mimi_cache_cuda fetch uses {args.out_dir}"
    with (
        exclusive(args.out_dir / FETCH_LOCK, busy),
        ThreadPoolExecutor(args.jobs) as pool,
    ):
        manifest(args)
        # What a killed fetch left: downloads and unrenamed sources.
        for stale in (args.out_dir / SOURCES).glob(".*"):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        jobs = [pool.submit(fetch_one, args, cache_dir.name, path) for path in shards]
        for number, job in enumerate(jobs, 1):
            print(f"{job.result()} [{number}/{len(jobs)}]", flush=True)


# --------------------------------------------------------------------------- encode


def decode(data: bytes) -> np.ndarray:
    """Worker: the clip at 24 kHz, decoded and resampled by the cache builder's code, so it has
    the samples and frame count the cache was built from."""
    pcm, rate = audio.decode(data)
    return soxr.resample(pcm, rate, augment.SAMPLE_RATE)


class Encoder:
    """The GPU side: conditions, Mimi in float32, new source codes written into the rows."""

    def __init__(self, args: argparse.Namespace):
        cfg = read_json(args.config_path)
        self.dep_q = int(cfg["dep_q"])
        self.source_q = int(cfg["n_q"]) - self.dep_q
        self.card = int(cfg["card"])
        self.device = torch.device(cache_build.DEVICE)
        self.mimi = cache_build.cuda_mimi(require_file(args.mimi_weight, "Mimi weight"), self.source_q)
        if self.mimi.cardinality != self.card:
            raise RuntimeError(
                f"config card {self.card} != Mimi cardinality {self.mimi.cardinality}"
            )
        self.bank = augment.Bank(augment.fetch_bank(args.bank_dir), self.device)
        self.batch_samples = int(args.batch_seconds * augment.SAMPLE_RATE)

    def shard(self, samples: list[dict[str, Any]], clips: dict[int, Future]) -> float:
        """Replace the source codes of the rows in `clips` (sample index -> decoded clip), shortest
        first in padded batches; return the seconds of audio encoded. Empties `clips`, so each
        clip's PCM is freed once it is encoded."""
        pending: list[tuple[dict[str, Any], torch.Tensor]] = []
        seconds = 0.0
        for index in sorted(clips, key=lambda index: samples[index]["vi_frames"]):
            sample = samples[index]
            # Shortest first, so this row sets the padded length of the batch it joins.
            if (
                pending
                and (len(pending) + 1) * sample["vi_frames"] * FRAME_SAMPLES > self.batch_samples
            ):
                self._write(pending)
                pending = []
            try:
                pcm = clips.pop(index).result()
            except audio.Rejected as exc:
                raise RuntimeError(f"{sample['id']}: {exc}") from exc
            clip = torch.from_numpy(pcm).to(self.device)
            if math.ceil(clip.shape[-1] / FRAME_SAMPLES) != sample["vi_frames"]:
                raise RuntimeError(
                    f"{sample['id']}: {clip.shape[-1]} samples, the cache has "
                    f"{sample['vi_frames']} source frames"
                )
            conditions = augment.Conditions(**sample["source_conditions"])
            pending.append((sample, self.bank.apply(clip, conditions)))
            seconds += clip.shape[-1] / augment.SAMPLE_RATE
        if pending:
            self._write(pending)
        return seconds

    def _write(self, pending: list[tuple[dict[str, Any], torch.Tensor]]) -> None:
        codes = cache_build.mimi_batch(self.mimi, [clip for _, clip in pending], self.device)
        for (sample, _), new in zip(pending, codes, strict=True):
            source = sample["codes"][1 + self.dep_q : 1 + self.dep_q + self.source_q]
            if int(source[0, sample["vi_frames"]]) != self.card:
                raise RuntimeError(f"{sample['id']}: no codec EOS at frame {sample['vi_frames']}")
            source[:, : sample["vi_frames"]] = new


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
    busy = f"another mimi_cache_cuda encode uses {args.out_dir}"
    with exclusive(args.out_dir / ENCODE_LOCK, busy):
        settings = manifest(args)
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


def load(
    args: argparse.Namespace,
    settings: dict[str, Any],
    split: str,
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
    if (
        payload["source"] != source
        or source["file"] != f"data/{split}/{shard_name(shard_path)}.parquet"
    ):
        raise RuntimeError(f"{shard_path} was built from {payload['source']}, not {source}")
    ids = table.column("id").to_pylist()
    opus = table.column("vi_opus").to_pylist()
    samples = payload["samples"]
    opus_rows: dict[int, bytes] = {}
    for index, sample in enumerate(samples):
        file, row = sample["source_row"].split("#r")
        row = int(row)
        if file != source["file"] or ids[row] != sample["id"]:
            raise RuntimeError(f"{sample['source_row']} holds {ids[row]}, not {sample['id']}")
        sample["source_conditions"] = None
        if opus[row] is None:
            continue
        conditions = bank.draw(f"{settings['seed']}:{sample['id']}")
        sample["source_conditions"] = dataclasses.asdict(conditions)
        opus_rows[index] = opus[row]
    # Shortest first, the order Encoder.shard takes them in, so its first batch is ready first.
    order = sorted(opus_rows, key=lambda index: samples[index]["vi_frames"])
    clips = {index: pool.submit(decode, opus_rows[index]) for index in order}
    return Loaded(source_path, payload, clips)


def encode_shards(
    args: argparse.Namespace, settings: dict[str, Any], split: str, shards: list[Path]
) -> list[str]:
    """Write the noisy shards; return the names of those whose source is missing. Each shard's
    clips are queued before the GPU encodes the shard before it, so the workers decode them
    meanwhile."""
    encoder = Encoder(args)
    # One CPU stays for this process, which feeds the GPU: workers past the budget exhaust a
    # container's quota, and the throttle stalls this process with them.
    workers = args.workers if args.workers is not None else max(1, math.floor(cpu_budget()) - 1)
    print(f"{workers} decode workers", flush=True)
    spawn = multiprocessing.get_context("spawn")
    missing: list[str] = []
    with ProcessPoolExecutor(workers, mp_context=spawn) as pool:
        try:
            ahead = load(args, settings, split, encoder.bank, pool, shards[0])
            last = time.perf_counter()
            for number, shard_path in enumerate(shards, 1):
                shard = ahead
                if number < len(shards):
                    ahead = load(args, settings, split, encoder.bank, pool, shards[number])
                if shard is None:
                    print(f"{shard_path.name}: no fetched source, skipped", flush=True)
                    missing.append(shard_name(shard_path))
                    last = time.perf_counter()
                    continue
                payload = shard.payload
                count = len(shard.clips)
                seconds = encoder.shard(payload["samples"], shard.clips)
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
