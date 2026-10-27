#!/usr/bin/env python
"""Cache the EN-VI speech dataset as grounded Mimi code shards for SFT, on Apple Silicon.

Each source shard `data/<split>/<name>.parquet` becomes `<out-dir>/<split>/shard_<name>.pt`.
A row is
[33, T]: CTC-timed English text, 16 English target Mimi codebooks (delayed by a seeded draw that
fits --max-frames), then 16 Vietnamese source Mimi codebooks ending in codec EOS. Rows whose clips
exceed --max-frames, or whose English text fails CTC alignment, are rejected before Mimi and listed
with their reason.

Worker processes decode the Opus/MP3 bytes (PyAV) and resample them (soxr); the main process runs
wav2vec2 CTC and the Mimi encoder in MLX on the GPU.
"""

from __future__ import annotations

import argparse
import collections
import io
import math
import os
import random
import re
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "ARROW_DEFAULT_MEMORY_POOL", "system"
)  # mimalloc keeps freed buffers on macOS
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import av  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soxr  # noqa: E402
import torch  # noqa: E402

from training.paths import (  # noqa: E402
    CACHE_FORMAT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    durable,
    exclusive,
    read_json,
    require_dir,
    require_file,
)
from training.prep.mimi_mlx import FRAME_SAMPLES, SAMPLE_RATE, MimiEncoder  # noqa: E402
from training.prep.text_timing import (  # noqa: E402
    ALIGNMENT_NAME,
    AlignmentError,
    EnglishCTCAligner,
    sentencepiece_groups,
    timed_sentencepiece_tokens,
)

FRAME_RATE = 12.5
CTC_RATE = 16_000
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
PREFETCH = 64  # rows decoded ahead of the GPU (~5 MB each)
MLX_CACHE_BYTES = 1 << 30  # MLX keeps freed buffers; shapes vary per row
DURATION_TOLERANCE_S = 0.02  # healthy clips decode exactly to their stored duration
CONFIG_KEYS = ("n_q", "dep_q", "card", "text_card", "existing_text_padding_id")


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
    parser.add_argument("--workers", type=int, default=2, help="Audio decode processes.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing shards.")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not (math.isfinite(args.target_delay_ratio) and args.target_delay_ratio >= 0):
        parser.error("--target-delay-ratio must be finite and non-negative")
    return args


class Rejected(Exception):
    """A data fault in one row (bad audio, too long); any other error stops the run."""


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Opus or MP3 bytes -> mono float32 PCM at its own rate (Opus 48 kHz, MP3 32/48 kHz)."""
    try:
        with av.open(io.BytesIO(data)) as container:
            chunks, rate = [], 0
            for frame in container.decode(audio=0):
                if frame.format.name != "fltp" or frame.layout.nb_channels != 1:
                    raise Rejected(
                        f"expected mono fltp audio, got {frame.format.name} {frame.layout.name}"
                    )
                chunks.append(frame.to_ndarray()[0])
                rate = frame.sample_rate
    except av.error.MemoryError:
        raise
    except av.FFmpegError as exc:
        raise Rejected(f"undecodable audio: {exc}") from None
    if not chunks:
        raise Rejected("no audio frames")
    return np.concatenate(chunks), rate


def prepare(vi_audio: bytes, en_audio: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: both clips at 24 kHz for Mimi, and the English clip at 16 kHz for CTC."""
    vi, vi_rate = decode(vi_audio)
    en, en_rate = decode(en_audio)
    return (
        soxr.resample(vi, vi_rate, SAMPLE_RATE),
        soxr.resample(en, en_rate, SAMPLE_RATE),
        soxr.resample(en, en_rate, CTC_RATE),
    )


def target_delay_s(row_id: str, split: str, max_delay_s: float, seed: int) -> float:
    """Draw the English left-silence delay from U[0, max_delay_s], seeded per row."""
    return random.Random(f"{seed}:{split}:{row_id}").uniform(0.0, max_delay_s)


def assemble_codes(
    row_id: str,
    vi_codes: np.ndarray,
    en_codes: np.ndarray,
    tokens: list[int],
    text_frames: list[int],
    cfg: dict[str, Any],
) -> np.ndarray:
    n_q = int(cfg["n_q"])
    dep_q = int(cfg["dep_q"])
    source_q = n_q - dep_q
    card = int(cfg["card"])
    text_card = int(cfg["text_card"])
    text_pad_id = int(cfg["existing_text_padding_id"])

    if vi_codes.shape[0] < source_q:
        raise RuntimeError(
            f"Vietnamese Mimi codes have {vi_codes.shape[0]} codebooks, need {source_q}"
        )
    if en_codes.shape[0] < dep_q:
        raise RuntimeError(f"English Mimi codes have {en_codes.shape[0]} codebooks, need {dep_q}")
    if any(token < 0 or token >= text_card for token in tokens):
        raise RuntimeError(f"Text token out of range for id={row_id}")
    if len(text_frames) != len(tokens) or len(set(text_frames)) != len(tokens):
        raise ValueError(f"Text frames must be unique and match tokens for id={row_id}")
    if text_frames != sorted(text_frames) or text_frames[0] < 0:
        raise ValueError(f"Text frames must be sorted and non-negative for id={row_id}")

    target_len = int(en_codes.shape[1])
    source_len = int(vi_codes.shape[1])
    text_end = text_frames[-1] + 1
    codes = np.full((1 + n_q, max(text_end, target_len, source_len + 1)), -1, dtype=np.int32)
    codes[0, :text_end] = text_pad_id
    codes[0, text_frames] = tokens
    codes[1 : 1 + dep_q, :target_len] = en_codes[:dep_q]
    source = slice(1 + dep_q, 1 + n_q)
    codes[source, :source_len] = vi_codes[:source_q]
    codes[source, source_len] = card
    return codes


class Cacher:
    """GPU side of the build: gates, CTC timing and Mimi codes for one source shard at a time."""

    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any]):
        import sentencepiece

        self.args = args
        self.cfg = cfg
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            str(require_file(args.tokenizer, "text tokenizer"))
        )
        self.eos_id = int(self.tokenizer.eos_id())
        self.aligner = EnglishCTCAligner(args.min_alignment_score)
        num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
        self.mimi = MimiEncoder(require_file(args.mimi_weight, "Mimi weight"), num_codebooks)
        if self.mimi.cardinality != int(cfg["card"]):
            raise RuntimeError(
                f"Config card={cfg['card']} != Mimi cardinality {self.mimi.cardinality}"
            )

    def row(
        self, split: str, row: dict[str, Any], vi24: np.ndarray, en24: np.ndarray, en16: np.ndarray
    ) -> dict[str, Any]:
        """One cache sample; raises Rejected or AlignmentError to reject the row."""
        args = self.args
        for side, pcm, stored in (
            ("vi", vi24, row["vi_duration_s"]),
            ("en", en24, row["en_duration_s"]),
        ):
            # PyAV decodes a truncated or damaged clip without error, only shorter.
            seconds = len(pcm) / SAMPLE_RATE
            if abs(seconds - stored) > DURATION_TOLERANCE_S:
                raise Rejected(
                    f"{side} audio decodes to {seconds:.2f} s, the row says {stored:.2f} s"
                )
        vi_frames = math.ceil(len(vi24) / FRAME_SAMPLES)
        # Text EOS sits at en_frames, codec EOS at vi_frames.
        frames = max(math.ceil(len(en24) / FRAME_SAMPLES), vi_frames) + 1
        if frames > args.max_frames:
            raise Rejected(f"{frames} frames exceed --max-frames {args.max_frames}")
        # The delay is drawn within the room the English clip leaves under the cap, so a long clip
        # gets a shorter delay instead of being rejected. vi_duration_s is float32 holding a
        # multiple of 0.01; round to the decimal it stores.
        room_s = ((args.max_frames - 1) * FRAME_SAMPLES - len(en24)) / SAMPLE_RATE
        max_delay_s = min(args.target_delay_ratio * round(row["vi_duration_s"], 2), room_s)
        delay_s = target_delay_s(row["id"], split, max_delay_s, args.seed)
        pad = int(round(delay_s * SAMPLE_RATE))
        delay_frames = int(round(delay_s * FRAME_RATE))
        en_frames = math.ceil((pad + len(en24)) / FRAME_SAMPLES)
        lead, groups = sentencepiece_groups(row["text_en"], self.tokenizer)
        alignment = self.aligner.align(en16, groups)
        tokens, text_frames = timed_sentencepiece_tokens(
            lead, groups, alignment, en_frames - delay_frames, delay_frames, self.eos_id
        )
        vi_codes = self.mimi.encode(vi24)
        en_codes = self.mimi.encode(np.concatenate([np.zeros(pad, np.float32), en24]))
        codes = assemble_codes(row["id"], vi_codes, en_codes, tokens, text_frames, self.cfg)
        return {
            "id": row["id"],
            "split": split,
            "codes": codes,
            "frames": int(codes.shape[1]),
            "vi_frames": vi_frames,
            "en_frames": en_frames,
            "text_tokens": len(tokens),
            "target_delay_s": delay_s,
            "target_delay_frames": delay_frames,
            "text_en": row["text_en"],
            "text_vi": row["text_vi"],
            "cross_lingual_timbre_matched": row["cross_lingual_timbre_matched"],
            "text_timing": ALIGNMENT_NAME,
            "alignment_score": alignment.score,
            "alignment_text": " ".join(spoken for _, spoken in groups),
        }

    def shard(self, pool: ProcessPoolExecutor, source: Path, out_path: Path) -> None:
        split = source.parent.name
        source_bytes = source.stat().st_size
        parquet = pq.ParquetFile(source)
        rows = (
            row
            for group in range(parquet.num_row_groups)
            for row in parquet.read_row_group(group, columns=COLUMNS).to_pylist()
        )
        pending: collections.deque[tuple[int, dict[str, Any], Future]] = collections.deque()
        samples: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        audio_s = 0.0
        started = time.perf_counter()
        for index, row in enumerate(rows):
            job = pool.submit(prepare, row["vi_audio"]["bytes"], row["en_audio"]["bytes"])
            pending.append((index, row, job))
            row["vi_audio"] = row["en_audio"] = None
            if len(pending) < PREFETCH:
                continue
            audio_s += self._take(pending.popleft(), split, source, samples, rejected)
        while pending:
            audio_s += self._take(pending.popleft(), split, source, samples, rejected)

        payload = {
            "format": CACHE_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "alignment_min_score": self.args.min_alignment_score,
            "max_frames": self.args.max_frames,
            "config": {key: int(self.cfg[key]) for key in CONFIG_KEYS},
            "source": {"file": f"data/{split}/{source.name}", "bytes": source_bytes},
            "samples": samples,
            "rejected": rejected,
        }
        tmp = out_path.with_name(out_path.name + ".tmp")
        torch.save(payload, tmp)
        durable(tmp)
        tmp.replace(out_path)
        durable(out_path.parent)
        wall = time.perf_counter() - started
        reasons = collections.Counter(
            re.sub(r"[\d.]+", "N", item["reason"])[:60] for item in rejected
        )
        print(
            f"{split}/{out_path.name}: {len(samples)} kept, "
            f"{len(rejected)} rejected {dict(reasons)}; "
            f"{audio_s / 3600:.2f} h encoded in {wall:.0f} s ({audio_s / wall:.0f}x RT)",
            flush=True,
        )

    def _take(self, item, split, source, samples, rejected) -> float:
        index, row, future = item
        try:
            vi24, en24, en16 = future.result()
            sample = self.row(split, row, vi24, en24, en16)
        except (Rejected, AlignmentError) as exc:
            # Only data faults reject a row. Anything else (a dead worker's BrokenProcessPool, an
            # MLX error) stops the run before the shard is written, so a rerun rebuilds it.
            rejected.append({"id": row["id"], "reason": f"{type(exc).__name__}: {exc}"})
            return 0.0
        sample["codes"] = torch.from_numpy(sample["codes"])
        sample["source_row"] = f"data/{split}/{source.name}#r{index}"
        samples.append(sample)
        return (sample["vi_frames"] + sample["en_frames"]) * FRAME_SAMPLES / SAMPLE_RATE


def main() -> None:
    args = parse_args()
    data = require_dir(args.dataset_dir / "data", "dataset data directory")
    sources = set()
    for pattern in args.shards:
        matched = {path for path in data.glob(pattern) if path.suffix == ".parquet"}
        if not matched:
            raise FileNotFoundError(f"No .parquet source shards match {pattern!r} under {data}")
        sources |= matched
    args.out_dir.mkdir(parents=True, exist_ok=True)
    busy = f"another cache_codes run is using {args.out_dir}"
    mx.set_cache_limit(MLX_CACHE_BYTES)
    cacher = Cacher(args, read_json(args.config_path))
    with (
        exclusive(args.out_dir / ".cache_codes.lock", busy),
        ProcessPoolExecutor(args.workers) as pool,
    ):
        for source in sorted(sources):
            out_path = args.out_dir / source.parent.name / f"shard_{source.stem}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing {out_path}", flush=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cacher.shard(pool, source, out_path)


if __name__ == "__main__":
    main()
