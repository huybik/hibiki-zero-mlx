#!/usr/bin/env python
"""Re-encode the Vietnamese source of a grounded cache under simulated recording conditions, on CUDA.

Each `shard_<name>.pt` of --cache-dir becomes <out-dir>/shard_<name>.pt with the same rows in the
same order. A seeded --fraction of the rows get new Vietnamese source codes: their clip, decoded
from the EN-VI dataset's parquet shard that the cache shard names, resampled with soxr as the cache
builder does, put under `augment`'s drawn conditions and Mimi-encoded in float32. Text, English
codes, frame counts and both EOS positions stay as they were. Each row records its conditions
under `source_conditions` (None for a row kept clean).

    HF parquet (one shard ahead) ─► workers: Opus decode, soxr to 24 kHz ─► GPU: augment, Mimi
    cache shard ──────────────────────────────────────────────────────────► new source codes ─► shard

The parquet shards download from the Hub at the revision `eval_sets` pins and are deleted once
used. A rerun skips the shards already written.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import multiprocessing
import random
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soxr
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from training.paths import (
    CACHE_FORMAT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    durable,
    exclusive,
    read_json,
    require_dir,
    require_file,
)
from training.prep import augment
from training.prep.eval_sets import EN_VI_REPO, EN_VI_REVISION, OPUS_RATE, decode_opus

FRAME_SAMPLES = 1920  # 24 kHz / 12.5 Hz
DOWNLOAD_AHEAD = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True, help="Folder of clean shard_*.pt.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=Path("data/rirs_noises"))
    parser.add_argument("--fraction", type=float, default=0.6, help="Rows given new conditions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--workers", type=int, default=24, help="Decode processes.")
    parser.add_argument(
        "--batch-seconds", type=float, default=640.0, help="Padded audio per Mimi batch."
    )
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    return args


def decode(data: bytes, duration_s: float, name: str) -> np.ndarray:
    """Worker: the clip at 24 kHz, decoded and cut as the cache builder decoded it."""
    wav = decode_opus(data, duration_s, name)[0]
    return soxr.resample(wav, OPUS_RATE, augment.SAMPLE_RATE)


def fetch(source: str, folder: Path) -> Path:
    """Download one parquet shard of the dataset (`data/<split>/<name>.parquet`) into `folder`."""
    return Path(
        hf_hub_download(
            EN_VI_REPO, source, repo_type="dataset", revision=EN_VI_REVISION, local_dir=folder
        )
    )


class Encoder:
    """The GPU side: conditions, Mimi in float32, new source codes written into the rows."""

    def __init__(self, args: argparse.Namespace):
        cfg = read_json(args.config_path)
        self.source_q = int(cfg["n_q"]) - int(cfg["dep_q"])
        self.card = int(cfg["card"])
        # TF32 changes Mimi's residual codes as bf16 does, so every matmul and convolution stays
        # full float32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda")
        self.mimi = loaders.get_mimi(
            require_file(args.mimi_weight, "Mimi weight"),
            device=self.device,
            num_codebooks=self.source_q,
        ).eval()
        if self.mimi.cardinality != self.card:
            raise RuntimeError(
                f"config card {self.card} != Mimi cardinality {self.mimi.cardinality}"
            )
        self.bank = augment.Bank(augment.fetch_bank(args.bank_dir), self.device)
        self.batch_samples = int(args.batch_seconds * augment.SAMPLE_RATE)

    def encode(self, clips: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each clip's codes [source_q, ceil(samples / 1920)]. Right padding changes nothing
        before it: Mimi's convolutions and attention are causal."""
        frames = [math.ceil(clip.shape[-1] / FRAME_SAMPLES) for clip in clips]
        batch = torch.zeros(len(clips), 1, max(frames) * FRAME_SAMPLES, device=self.device)
        for row, clip in enumerate(clips):
            batch[row, 0, : clip.shape[-1]] = clip
        with torch.no_grad():
            codes = self.mimi.encode(batch)
        return [codes[row, :, :count].to("cpu", torch.int32) for row, count in enumerate(frames)]

    def shard(self, samples: list[dict[str, Any]], clips: dict[int, Future]) -> float:
        """Replace the source codes of the rows in `clips` (sample index -> decoded clip), shortest
        first in padded batches; return the seconds of audio encoded."""
        pending: list[tuple[dict[str, Any], torch.Tensor]] = []
        seconds = 0.0
        for index in sorted(clips, key=lambda index: samples[index]["vi_frames"]):
            sample = samples[index]
            # Shortest first, so this row sets the padded length of the batch it joins.
            if (len(pending) + 1) * sample["vi_frames"] * FRAME_SAMPLES > self.batch_samples:
                self._write(pending)
                pending = []
            clip = torch.from_numpy(clips[index].result()).to(self.device)
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
        codes = self.encode([clip for _, clip in pending])
        for (sample, _), new in zip(pending, codes, strict=True):
            source = sample["codes"][1 + self.source_q :]
            if int(source[0, sample["vi_frames"]]) != self.card:
                raise RuntimeError(f"{sample['id']}: no codec EOS at frame {sample['vi_frames']}")
            source[:, : sample["vi_frames"]] = new


def selected(seed: int, row_id: str, fraction: float) -> bool:
    return random.Random(f"{seed}:select:{row_id}").random() < fraction


def main() -> None:
    args = parse_args()
    cache_dir = require_dir(args.cache_dir, "clean cache split")
    shards = [
        path
        for path in sorted(cache_dir.glob("shard_*.pt"))
        if not (args.out_dir / path.name).exists()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(shards)} shards to write under {args.out_dir}", flush=True)
    if not shards:
        return
    encoder = Encoder(args)
    split = cache_dir.name
    spawn = multiprocessing.get_context("spawn")
    with (
        exclusive(
            args.out_dir / ".noisy_cache.lock", f"another noisy_cache run uses {args.out_dir}"
        ),
        tempfile.TemporaryDirectory(dir=args.out_dir.parent, prefix=".parquet-") as tmp,
        ThreadPoolExecutor(DOWNLOAD_AHEAD) as downloads,
        ProcessPoolExecutor(args.workers, mp_context=spawn) as pool,
    ):
        sources = [f"data/{split}/{path.stem.removeprefix('shard_')}.parquet" for path in shards]
        fetched = [
            downloads.submit(fetch, source, Path(tmp)) for source in sources[:DOWNLOAD_AHEAD]
        ]
        for number, (shard_path, source) in enumerate(zip(shards, sources)):
            started = time.perf_counter()
            parquet_path = fetched[number].result()
            if number + DOWNLOAD_AHEAD < len(sources):
                fetched.append(downloads.submit(fetch, sources[number + DOWNLOAD_AHEAD], Path(tmp)))
            payload = torch.load(shard_path, map_location="cpu")
            if payload.get("format") != CACHE_FORMAT:
                raise RuntimeError(f"{shard_path} is not a {CACHE_FORMAT} shard")
            if payload["source"] != {"file": source, "bytes": parquet_path.stat().st_size}:
                raise RuntimeError(f"{shard_path} was built from {payload['source']}, not {source}")
            table = pq.read_table(parquet_path, columns=["id", "vi_audio", "vi_duration_s"])
            clips: dict[int, Future] = {}
            for index, sample in enumerate(payload["samples"]):
                sample["source_conditions"] = None
                if not selected(args.seed, sample["id"], args.fraction):
                    continue
                file, row_index = sample["source_row"].split("#r")
                row = table.slice(int(row_index), 1).to_pylist()[0]
                if file != source or row["id"] != sample["id"]:
                    raise RuntimeError(
                        f"{sample['source_row']} holds {row['id']}, not {sample['id']}"
                    )
                conditions = encoder.bank.draw(f"{args.seed}:{sample['id']}")
                sample["source_conditions"] = dataclasses.asdict(conditions)
                clips[index] = pool.submit(
                    decode, row["vi_audio"]["bytes"], row["vi_duration_s"], row["id"]
                )
            del table
            seconds = encoder.shard(payload["samples"], clips)
            parquet_path.unlink()
            payload["source_augment"] = {
                "clean_cache": f"{split}/{shard_path.name}",
                "fraction": args.fraction,
                "seed": args.seed,
                "bank_sha256": augment.BANK_SHA256,
                "dataset_revision": EN_VI_REVISION,
            }
            out_path = args.out_dir / shard_path.name
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            torch.save(payload, tmp_path)
            durable(tmp_path)
            tmp_path.replace(out_path)
            durable(args.out_dir)
            wall = time.perf_counter() - started
            print(
                f"{shard_path.name}: {len(clips)} of {len(payload['samples'])} rows re-encoded, "
                f"{seconds / 3600:.2f} h in {wall:.0f} s ({seconds / wall:.0f}x RT) "
                f"[{number + 1}/{len(shards)}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
