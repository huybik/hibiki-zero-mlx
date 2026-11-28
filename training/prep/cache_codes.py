#!/usr/bin/env python
"""Cache the EN-VI speech dataset as grounded Mimi code shards for SFT.

Each source shard `data/<split>/<name>.parquet` becomes `<out-dir>/<split>/shard_<name>.pt`.
A row is
[33, T]: CTC-timed English text, 16 English target Mimi codebooks (delayed by a seeded draw that
fits --max-frames), then 16 Vietnamese source Mimi codebooks ending in codec EOS. With
--direction en-vi the languages trade places: Vietnamese text and target, English source, for
two-way training; that build tokenizes with the Vietnamese swap (the direction's default --tokenizer)
and takes its own --out-dir. Rows whose clips exceed --max-frames, or whose target text fails CTC alignment, are
rejected and listed with their reason. With --vi-check, each train shard's Vietnamese is as its part of
`training.tranche.voice run en_speech` says (`en_speech/vi_check/<name>.parquet` of the tranche repo): a row it
re-voiced takes the new audio, a row still broken or never heard there is rejected, and the shard's `source`
records the part (noisy_cache reads it too). The build itself (decode, gates, the target language's
CTC aligner and Mimi on --device, the Viterbi) is `cache_build`'s, shared with the tranche's
builder.

Source shards go top down in path order, or bottom up with --bottom-up on a second machine;
`publish_caches --follow` beside each run commits the shards to the cache repo and brings in those
the other machine built, which the run then skips, so the two meet in between.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DIRECTION_TOKENIZERS,
    exclusive,
    require_dir,
    require_file,
)
from training.prep.cache_build import MODELS, Builder, default_workers, summary
from training.prep.cache_rows import DEFAULT_DIRECTION, DIRECTIONS, Settings, Shard, mimi_hours, write

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
    parser.add_argument("--vi-check", type=Path,
                        help="A copy of the tranche repo's en_speech/vi_check/: every train shard needs its part.")
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
    parser.add_argument("--ane-ctc", action="store_true",
                        help="MLX, en-vi: the Vietnamese aligner's transformer on the Neural Engine beside Mimi on "
                             "the GPU (wav2vec2_coreml; text timings move by near-ties).")
    parser.add_argument("--direction", choices=DIRECTIONS, default=DEFAULT_DIRECTION,
                        help="<source>-<target>: the language heard, then the one spoken and written.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, help="Default: the direction's (paths.DIRECTION_TOKENIZERS).")
    parser.add_argument("--max-frames", type=int, default=280, help="The trainer's frame cap.")
    parser.add_argument(
        "--target-delay-ratio",
        type=float,
        default=0.5,
        help="Maximum target delay as a ratio of the source clip's duration, within the frame cap.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for deterministic delays.")
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        help="Reject rows below this mean forced-alignment posterior (default: the target aligner's).",
    )
    parser.add_argument(
        "--workers", type=int, help="Decode and alignment processes (default: the device's)."
    )
    parser.add_argument("--fp8-ctc", action="store_true", help="CUDA Blackwell FP8 on wav2vec2 transformer linears.")
    parser.add_argument("--fp8-mimi", action="store_true", help="CUDA Blackwell FP8 on Mimi encoder transformer linears.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing shards.")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not (math.isfinite(args.target_delay_ratio) and args.target_delay_ratio >= 0):
        parser.error("--target-delay-ratio must be finite and non-negative")
    if args.min_alignment_score is not None and not 0 <= args.min_alignment_score <= 1:
        parser.error("--min-alignment-score must be in [0, 1]")
    return args


VI_CHECK = "en_speech/vi_check"  # the tranche repo's folder of voice.py's en_speech parts


def vi_checked(rows: list[dict[str, Any]], check_file: Path) -> list[dict[str, Any] | None]:
    """A source shard's rows (id, vi_audio, vi_duration_s) as its vi_check part says: a row re-voiced there with
    its new Vietnamese, None for a row still broken or never heard. Shared with noisy_cache."""
    import pyarrow.parquet as pq

    check = pq.read_table(check_file, columns=["id", "vi_audio", "vi_duration_s", "vi_broken"]).to_pylist()
    if [c["id"] for c in check] != [r["id"] for r in rows]:
        raise RuntimeError(f"{check_file} does not hold its source shard's rows in their order")
    return [
        None if c["vi_broken"] is not False
        else row | ({"vi_audio": c["vi_audio"], "vi_duration_s": c["vi_duration_s"]} if c["vi_audio"] else {})
        for row, c in zip(rows, check)
    ]


def source_record(source: Path, vi_check: Path | None) -> dict[str, Any]:
    """A cache shard's `source`: the dataset file and its size, and the vi_check part read for a train shard
    (`publish_caches` checks a pulled shard against it)."""
    record: dict[str, Any] = {"file": f"data/{source.parent.name}/{source.name}", "bytes": source.stat().st_size}
    if vi_check is not None and source.parent.name == "train":
        check = vi_check / source.name
        record["vi_check"] = {"file": f"{VI_CHECK}/{check.name}", "bytes": check.stat().st_size}
    return record


def read(source: Path, vi_check: Path | None) -> Shard:
    """The source shard's rows, all kept (this dataset has no gates of its own) but, with `vi_check` on a train
    shard, those whose Vietnamese is still broken or was never heard (`vi_checked`)."""
    import pyarrow.parquet as pq

    split = source.parent.name
    rows = pq.read_table(source, columns=COLUMNS).to_pylist()
    checked: list[dict[str, Any] | None] = rows
    if vi_check is not None and split == "train":
        checked = vi_checked(rows, require_file(vi_check / source.name, "vi_check part"))
    kept: list[tuple[int, dict[str, Any]]] = [
        (
            index,
            {
                "id": row["id"],
                "text_en": row["text_en"],
                "text_vi": row["text_vi"],
                "vi_duration_s": row["vi_duration_s"],
                "en_duration_s": row["en_duration_s"],
                "vi_audio": row["vi_audio"]["bytes"],
                "en_audio": row["en_audio"]["bytes"],
                "fields": {"cross_lingual_timbre_matched": row["cross_lingual_timbre_matched"]},
            },
        )
        for index, row in enumerate(checked)
        if row is not None
    ]
    rejected = [{"id": rows[i]["id"], "reason": "Vietnamese broken or not heard (vi_check)"}
                for i, row in enumerate(checked) if row is None]
    return Shard(source, f"data/{split}/{source.name}", split, len(rows), kept, rejected)


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
    settings = Settings(args.max_frames, args.min_alignment_score, args.direction, args.target_delay_ratio, args.seed)
    with exclusive(args.out_dir / ".cache_codes.lock", f"another cache_codes run is using {args.out_dir}"):
        builder = Builder(
            args.device,
            settings,
            args.config_path,
            args.mimi_weight,
            args.tokenizer or DIRECTION_TOKENIZERS[args.direction],
            ane_ctc=args.ane_ctc,
            fp8_ctc=args.fp8_ctc,
            fp8_mimi=args.fp8_mimi,
        )
        began, hours = time.perf_counter(), 0.0
        workers = args.workers or default_workers(args.device)
        with ThreadPoolExecutor(1) as writes:
            written: Future | None = None
            for shard in builder.run(todo(args, sources), lambda source: read(source, args.vi_check), workers):
                out_path = cache_path(args.out_dir, shard.key)
                payload = builder.payload(shard, source_record(shard.key, args.vi_check))
                if written is not None:
                    written.result()
                written = writes.submit(write, payload, out_path)
                hours += mimi_hours(shard.samples)
                line = summary(shard, hours, time.perf_counter() - began)
                print(f"{shard.split}/{out_path.name}: {line}", flush=True)
            if written is not None:
                written.result()


if __name__ == "__main__":
    main()
