"""Multi-sentence training streams assembled from grounded cache rows.

A stream joins cached rows (`[33, T]`: text, 16 English target and 16 Vietnamese source
codebooks) the way a speaker talks on. Consecutive Libriheavy segments of one recording and
consecutive PhoMT rows of one document stay together in order; other rows (Common Voice) follow at
random. Each row's Vietnamese follows the previous one after a gap. Its English starts the row's
cached delay after its Vietnamese, never before the previous English ends, and a gap is stretched
so the previous English overhangs the next Vietnamese by at most MAX_OVERHANG.

Wherever nobody speaks, both audio streams hold encoded silence and the text holds PAD, all
supervised. The last row keeps the source EOS and the text EOS; the source after its EOS stays
silence (as upstream inference feeds it), while the target and text after the text EOS are -1.

    VI   lead [row 1]  gap  [row 2]  gap ... [row n] EOS  silence ...
    EN   silence  δ1 [row 1 en] [row 2 en]  ...   [row n en] EOS -1 ...
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from training.paths import CACHE_FORMAT

FRAME_RATE = 12.5
# Batches pad to a multiple of this many frames, so the model sees few shapes.
FRAME_BUCKET = 16
MAX_LEAD = 25  # frames of silence before the first Vietnamese, drawn from [0, 2 s]
SHORT_GAP = (4, 19)  # a pause between sentences, 0.3-1.5 s
LONG_GAP = (38, 125)  # a 3-10 s silence, to practise staying quiet
LONG_GAP_PROBABILITY = 0.1
MAX_OVERHANG = 25  # the previous English may run at most 2 s into the next Vietnamese
SILENCE_WARMUP = 16  # frames of the silence encoder's start transient, dropped
# Teacher-forced validation streams are planned with one fixed seed, so every run and
# validate.py score the same streams.
VALIDATION_SEED = 0
# Consecutive rows: Libriheavy `<recording>_<segment>` and PhoMT `phomt_s<document>r<row>`.
GROUPED_ID = re.compile(r"^(libriheavy/.+)_(\d+)$|^(phomt_s\d+)r(\d+)$")


@dataclass(frozen=True)
class Layout:
    """The cache's code layout, shared by every shard of a run."""

    dep_q: int  # target codebooks, rows 1..dep_q; the source follows
    card: int  # audio cardinality, also the source EOS
    pad_id: int  # text PAD

    @property
    def rows(self) -> int:
        return 1 + 2 * self.dep_q


@dataclass(frozen=True)
class Row:
    id: str
    codes: torch.Tensor  # int32 [33, T]
    vi_frames: int  # source length; the cached source EOS sits here
    en_frames: int  # target length including the delay; the cached text EOS sits here
    delay_frames: int


@dataclass(frozen=True)
class Piece:
    row: int  # index into the rows
    vi_start: int
    en_start: int


@dataclass(frozen=True)
class Stream:
    pieces: tuple[Piece, ...]
    frames: int
    silence_offset: int


def load_rows(cache_dirs: list[Path], max_frames: int) -> tuple[list[Row], Layout]:
    """Every row of the grounded cache shards in `cache_dirs` that fits `max_frames`."""
    rows: list[Row] = []
    layout: Layout | None = None
    dropped = 0
    for shard_path in [path for d in cache_dirs for path in sorted(d.glob("shard_*.pt"))]:
        payload = torch.load(shard_path, map_location="cpu")
        if payload.get("format") != CACHE_FORMAT:
            raise RuntimeError(f"Unsupported cache format in {shard_path}")
        if float(payload["frame_rate"]) != FRAME_RATE:
            raise RuntimeError(f"{shard_path} has frame rate {payload['frame_rate']}")
        config = payload["config"]
        shard_layout = Layout(
            dep_q=int(config["dep_q"]),
            card=int(config["card"]),
            pad_id=int(config["existing_text_padding_id"]),
        )
        if 2 * shard_layout.dep_q != int(config["n_q"]):
            raise RuntimeError(f"{shard_path} has unequal source and target codebooks")
        if layout is None:
            layout = shard_layout
        elif shard_layout != layout:
            raise RuntimeError(f"{shard_path} has layout {shard_layout}, others {layout}")
        for sample in payload["samples"]:
            codes = sample["codes"]
            if codes.ndim != 2 or codes.shape[0] != layout.rows:
                raise RuntimeError(f"{shard_path} id={sample['id']} codes are {tuple(codes.shape)}")
            if codes.shape[1] > max_frames:
                dropped += 1
                continue
            rows.append(
                Row(
                    id=str(sample["id"]),
                    codes=codes.to(torch.int32),
                    vi_frames=int(sample["vi_frames"]),
                    en_frames=int(sample["en_frames"]),
                    delay_frames=int(sample["target_delay_frames"]),
                )
            )
    if layout is None or not rows:
        raise RuntimeError(f"No usable shard_*.pt cache rows in {cache_dirs}")
    if dropped:
        print(f"[streams] dropped {dropped} rows over {max_frames} frames; kept {len(rows)}")
    return rows, layout


def _gap(rng: random.Random) -> int:
    low, high = LONG_GAP if rng.random() < LONG_GAP_PROBABILITY else SHORT_GAP
    return rng.randint(low, high)


def plan_streams(
    rows: list[Row], min_frames: int, max_frames: int, silence_frames: int, seed: int
) -> list[Stream]:
    """Cut every row, once, into streams of min_frames..max_frames frames."""
    longest = max(max(row.vi_frames, row.en_frames) for row in rows)
    if min_frames <= longest + MAX_LEAD + 1:
        raise ValueError(f"Streams of {min_frames} frames cannot hold a {longest}-frame row")
    if silence_frames < max_frames:
        raise ValueError("The silence track must cover the longest stream")
    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    units: list[list[int]] = []
    for index, row in enumerate(rows):
        match = GROUPED_ID.match(row.id)
        if match is None:
            units.append([index])
        else:
            key, order = (match.group(1), match.group(2)) if match.group(1) else match.group(3, 4)
            groups[key].append((int(order), index))
    units.extend([index for _, index in sorted(group)] for group in groups.values())
    rng.shuffle(units)

    streams: list[Stream] = []
    pieces: list[Piece] = []
    budget = vi_end = en_end = frames = 0

    def close() -> None:
        streams.append(Stream(tuple(pieces), frames, rng.randint(0, silence_frames - frames)))

    for index in (index for unit in units for index in unit):
        row = rows[index]
        for _ in range(2):  # a row that overflows the stream starts the next one
            if pieces:
                vi_start = max(vi_end + _gap(rng), en_end - MAX_OVERHANG)
            else:
                budget = rng.randint(min_frames, max_frames)
                vi_start = rng.randint(0, MAX_LEAD)
            en_start = max(vi_start + row.delay_frames, en_end)
            next_vi_end = vi_start + row.vi_frames
            next_en_end = en_start + row.en_frames - row.delay_frames
            next_frames = max(next_vi_end, next_en_end) + 1  # the EOS frames
            if pieces and next_frames > budget:
                close()
                pieces, vi_end, en_end = [], 0, 0
                continue
            pieces.append(Piece(index, vi_start, en_start))
            vi_end, en_end, frames = next_vi_end, next_en_end, next_frames
            break
    if pieces:
        close()
    return streams


def assemble(
    stream: Stream, rows: list[Row], silence: torch.Tensor, layout: Layout
) -> torch.Tensor:
    """The stream's int32 codes [33, frames]; `silence` is encoded silence [dep_q, >= frames]."""
    q = layout.dep_q
    frames = stream.frames
    quiet = silence[:, stream.silence_offset : stream.silence_offset + frames]
    last_piece = stream.pieces[-1]
    last = rows[last_piece.row]
    text_eos = last_piece.en_start + last.en_frames - last.delay_frames
    source_eos = last_piece.vi_start + last.vi_frames
    codes = torch.full((layout.rows, frames), -1, dtype=torch.int32)
    codes[0, :text_eos] = layout.pad_id
    codes[0, text_eos] = last.codes[0, last.en_frames]
    codes[1 : 1 + q, :text_eos] = quiet[:, :text_eos]
    codes[1 + q :] = quiet
    codes[1 + q :, source_eos] = layout.card
    for piece in stream.pieces:
        row = rows[piece.row]
        source = row.codes
        if (source[0, : row.delay_frames] != layout.pad_id).any():
            raise RuntimeError(f"{row.id} has text before its delay")
        if int(source[1 + q, row.vi_frames]) != layout.card:
            raise RuntimeError(f"{row.id} has no source EOS at {row.vi_frames}")
        english = slice(row.delay_frames, row.en_frames)
        span = row.en_frames - row.delay_frames
        codes[: 1 + q, piece.en_start : piece.en_start + span] = source[: 1 + q, english]
        codes[1 + q :, piece.vi_start : piece.vi_start + row.vi_frames] = source[
            1 + q :, : row.vi_frames
        ]
    return codes


def silence_track(mimi: Any, frames: int) -> torch.Tensor:
    """Mimi codes of `frames` frames of steady digital silence, [codebooks, frames] int32 on the
    CPU.

    Encoding silence from a fresh state starts with a transient of ~11 frames (the codes a cached
    row's delay begins with), then settles into a few repeating patterns rather than one frame
    (codebook 0 is 1316 on ~99% of frames), so the transient is dropped and gaps take slices.
    """
    total = frames + SILENCE_WARMUP
    samples = int(round(mimi.sample_rate / mimi.frame_rate)) * total
    device = next(mimi.parameters()).device
    with torch.no_grad():
        codes = mimi.encode(torch.zeros(1, 1, samples, device=device))[0]
    if codes.shape[-1] != total:
        raise RuntimeError(f"Mimi encoded {codes.shape[-1]} silence frames, expected {total}")
    return codes[:, SILENCE_WARMUP:].to(device="cpu", dtype=torch.int32)


class StreamDataset(Dataset):
    """Streams planned once from `rows` with `seed`; each item is assembled on demand."""

    def __init__(
        self,
        rows: list[Row],
        layout: Layout,
        silence: torch.Tensor,
        stream_frames: tuple[int, int],
        seed: int,
    ):
        if silence.shape[0] != layout.dep_q:
            raise ValueError(f"Silence has {silence.shape[0]} codebooks, need {layout.dep_q}")
        self.rows = rows
        self.layout = layout
        self.silence = silence
        self.streams = plan_streams(rows, *stream_frames, int(silence.shape[1]), seed)

    def __len__(self) -> int:
        return len(self.streams)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stream = self.streams[index]
        return {
            "codes": assemble(stream, self.rows, self.silence, self.layout),
            "frames": stream.frames,
            "source_frames": sum(self.rows[piece.row].vi_frames for piece in stream.pieces),
        }

    def batches(self, batch_frames: int, shuffle_seed: int | None) -> list[list[int]]:
        """Length-sorted streams grouped so each padded batch holds at most `batch_frames`
        frames; the batch order is shuffled with `shuffle_seed`, or kept for None."""
        order = sorted(range(len(self.streams)), key=lambda index: self.streams[index].frames)
        batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            padded = math.ceil(self.streams[index].frames / FRAME_BUCKET) * FRAME_BUCKET
            if padded > batch_frames:
                raise ValueError(f"A {padded}-frame stream exceeds --batch-frames {batch_frames}")
            if current and (len(current) + 1) * padded > batch_frames:
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(batches)
        return batches

    def exposure(self) -> dict[str, float | int]:
        source_frames = sum(row.vi_frames for row in self.rows)
        return {
            "rows": len(self.rows),
            "streams": len(self.streams),
            "stream_frames": sum(stream.frames for stream in self.streams),
            "source_hours": source_frames / FRAME_RATE / 3600,
        }


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad a batch with -1 to a multiple of FRAME_BUCKET frames."""
    frames = max(int(sample["codes"].shape[1]) for sample in samples)
    frames = math.ceil(frames / FRAME_BUCKET) * FRAME_BUCKET
    batch = torch.full(
        (len(samples), int(samples[0]["codes"].shape[0]), frames), -1, dtype=torch.long
    )
    for index, sample in enumerate(samples):
        batch[index, :, : sample["codes"].shape[1]] = sample["codes"]
    return {
        "codes": batch,
        "frames": torch.tensor([sample["frames"] for sample in samples]),
        "source_frames": torch.tensor([sample["source_frames"] for sample in samples]),
    }
