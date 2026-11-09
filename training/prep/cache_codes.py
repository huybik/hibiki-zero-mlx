#!/usr/bin/env python
"""Cache the EN-VI speech dataset as grounded Mimi code shards for SFT.

Each source shard `data/<split>/<name>.parquet` becomes `<out-dir>/<split>/shard_<name>.pt`.
A row is
[33, T]: CTC-timed English text, 16 English target Mimi codebooks (delayed by a seeded draw that
fits --max-frames), then 16 Vietnamese source Mimi codebooks ending in codec EOS. Rows whose clips
exceed --max-frames, or whose English text fails CTC alignment, are rejected and listed with their
reason. The build itself (decode, gates, wav2vec2 CTC and Mimi on --device, the Viterbi) is
`cache_build`'s, shared with the tranche's builder.

Source shards go top down in path order, or bottom up with --bottom-up on a second machine;
`publish_caches --follow` beside each run commits the shards to the cache repo and brings in those
the other machine built, which the run then skips, so the two meet in between.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    exclusive,
    require_dir,
)
from training.prep.cache_build import MODELS, Builder, default_workers, summary
from training.prep.cache_rows import Settings, Shard, mimi_hours, write

DATASET_DIR = Path("/Volumes/data/datasets/en_vi_speech")
COLUMNS = [
    "id",
    "en_audio",
    "vi_audio",
    "text_en",
    "text_vi",
    "en_duration_s",
    "vi_duration_s",
    "cross_lingual_timbre_matched",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Cache root; one folder per split."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--shards",
        nargs="+",
        default=["train/*.parquet", "validation/*.parquet"],
        help="Source shard globs under <dataset-dir>/data.",
    )
    parser.add_argument(
        "--bottom-up",
        action="store_true",
        help="Build the source shards in reverse path order, as the second machine does.",
    )
    parser.add_argument("--device", choices=list(MODELS), default="mlx")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-frames", type=int, default=280, help="The trainer's frame cap.")
    parser.add_argument(
        "--target-delay-ratio",
        type=float,
        default=0.5,
        help="Maximum target delay as a ratio of Vietnamese source duration, within the frame cap.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic delays.")
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        default=0.5,
        help="Reject rows below this mean forced-alignment posterior.",
    )
    parser.add_argument(
        "--workers", type=int, help="Decode and alignment processes (default: the device's)."
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing shards.")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not (math.isfinite(args.target_delay_ratio) and args.target_delay_ratio >= 0):
        parser.error("--target-delay-ratio must be finite and non-negative")
    if not 0 <= args.min_alignment_score <= 1:
        parser.error("--min-alignment-score must be in [0, 1]")
    return args


def read(source: Path) -> Shard:
    """The source shard's rows, all kept: this dataset has no gates of its own."""
    import pyarrow.parquet as pq

    split = source.parent.name
    rows = pq.read_table(source, columns=COLUMNS).to_pylist()
    kept: list[tuple[int, dict[str, Any]]] = [
        (
            index,
            {
                "id": row["id"],
                "text_en": row["text_en"],
                "vi_duration_s": row["vi_duration_s"],
                "en_duration_s": row["en_duration_s"],
                "vi_audio": row["vi_audio"]["bytes"],
                "en_audio": row["en_audio"]["bytes"],
                "fields": {
                    "text_vi": row["text_vi"],
                    "cross_lingual_timbre_matched": row["cross_lingual_timbre_matched"],
                },
            },
        )
        for index, row in enumerate(rows)
    ]
    return Shard(source, f"data/{split}/{source.name}", split, len(rows), kept, [])


def source_shards(dataset_dir: Path, patterns: list[str]) -> list[Path]:
    """The .parquet source shards matching `patterns` under <dataset_dir>/data, in path order."""
    data = require_dir(dataset_dir / "data", "dataset data directory")
    sources = set()
    for pattern in patterns:
        matched = {path for path in data.glob(pattern) if path.suffix == ".parquet"}
        if not matched:
            raise FileNotFoundError(f"No .parquet source shards match {pattern!r} under {data}")
        sources |= matched
    return sorted(sources)


def cache_path(out_dir: Path, source: Path) -> Path:
    return out_dir / source.parent.name / f"shard_{source.stem}.pt"


def todo(args: argparse.Namespace, sources: list[Path]) -> Iterator[Path]:
    """The sources without a cache shard, checked as each is read (a shard and a window ahead of
    the build), so a shard publish_caches brought in meanwhile is skipped."""
    for source in sources[::-1] if args.bottom_up else sources:
        if cache_path(args.out_dir, source).exists() and not args.overwrite:
            print(f"Skipping existing {cache_path(args.out_dir, source)}", flush=True)
            continue
        yield source


def main() -> None:
    args = parse_args()
    sources = source_shards(args.dataset_dir, args.shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(args.max_frames, args.min_alignment_score, args.target_delay_ratio, args.seed)
    with exclusive(args.out_dir / ".cache_codes.lock", f"another cache_codes run is using {args.out_dir}"):
        builder = Builder(args.device, settings, args.config_path, args.mimi_weight, args.tokenizer)
        began, hours = time.perf_counter(), 0.0
        workers = args.workers or default_workers(args.device)
        for shard in builder.run(todo(args, sources), read, workers):
            out_path = cache_path(args.out_dir, shard.key)
            write(builder.payload(shard, {"file": shard.file, "bytes": shard.key.stat().st_size}), out_path)
            hours += mimi_hours(shard.samples)
            line = summary(shard, hours, time.perf_counter() - began)
            print(f"{shard.split}/{out_path.name}: {line}", flush=True)


if __name__ == "__main__":
    main()
