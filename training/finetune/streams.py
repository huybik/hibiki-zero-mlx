"""Multi-sentence training streams assembled from grounded cache rows.

A stream joins cached rows (`[33, T]`: text, 16 target and 16 source codebooks) the way a
speaker talks on. A cache shard has a direction, Vietnamese to English (the diagrams and names
below) or English to Vietnamese; a stream holds rows of one direction, and a run may train on
streams of both. The rows of one document stay together, in order: a tranche
row's `doc` and `seq` (a PhoAudiobook reader's run, a GigaSpeech 2 recording; a viVoice row is its
own doc), and a PhoMT source chunk (`phomt_s<N>r<M>`, spanning talk boundaries); every other row
follows at random. Libriheavy segment numbers are not reading order, and about a quarter of a
recording's segments overlap another's text, so its rows are not grouped. Each row's Vietnamese follows the
previous one after a gap. Its English starts the row's cached delay after its Vietnamese, never
before the previous English ends, and a gap is stretched so the previous English overhangs the
next Vietnamese by at most MAX_OVERHANG.

Wherever nobody speaks, both audio streams hold encoded silence and the text holds PAD, all
supervised. The last row keeps the source EOS and the text EOS; the source after its EOS stays
silence (as upstream inference feeds it), while the target and text after the text EOS are -1.
With rooms (`rooms.run_rooms`; GRPO's, and SFT's with `train.py --room-share`) a share of streams hold a
room's noise in the source instead of silence, its lead, gaps and the stretch after its EOS
alike, as a phone's microphone hears a room between sentences.

Mimi codes depend on the encoder's state. Inference encodes one continuous stream from a fresh
state, so a stream opens with the encoder's start-up codes and is steady silence elsewhere. Every
cached clip, though, was encoded alone from a fresh state. Unless a piece opens the stream, its
Vietnamese frame 0 (a start-up marker: codebook 0 is one of three values in ~90% of rows, though
it can hold a syllable's onset) becomes silence, and so does English silence before the first word
within the first SILENCE_WARMUP frames of a clip. English speech that starts within those frames
keeps codes that differ from a continuous encoding (codebook 0 matches 60-80%), and residual
codebooks differ near every join (Mimi's 10 s context); re-encoding each clip after a silence
pre-roll would remove both.

    VI   lead [row 1]  gap  [row 2]  gap ... [row n] EOS  silence ...
    EN   silence  δ1 [row 1 en] [row 2 en]  ...   [row n en] EOS -1 ...

A SINGLE_ROW_STREAMS share of streams hold one row: lead, the row, both EOS. A LATE_START share
of rows, in any stream, start their English late: anywhere from half their Vietnamese (the cached
delays' cap) to 1 s past its end, so some start in the silence after it:

    VI   ... [row k]  gap  [row k+1] ...
    EN   ...        late [row k en]  [row k+1 en] ...

Which rows load, and where each row's target may start (`start_floor`), is `rows.py`'s;
the packs they load from are `packs.py`'s, and the rooms `rooms.py`'s.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from training.finetune.packs import FRAME_RATE, Layout
from training.finetune.rooms import encode_track
from training.finetune.rows import Row

# Stream lengths, seconds. Upstream's coarse stage trained on 30-75 s utterances and its RL stage
# on 120 s sequences; upstream Hibiki-Zero degrades after ~1.6 minutes of input, and a phone hears
# a stream that goes on.
STREAM_SECONDS = (30.0, 120.0)
# The share of streams that hold one row, as a single-sentence evaluation feeds it. Trained only on
# 30-120 s streams, the first run never saw the source end after one sentence, and never ended a
# third of the FLEURS rows.
SINGLE_ROW_STREAMS = 0.1
# The share of rows whose English starts late, its start drawn from [half the source, 1 s past its
# end] rather than the cached delay (at most half the source) after the source starts. In the
# caches a first word rarely comes after its source has ended (0.04% of English-to-Vietnamese rows,
# up to 2.6% of Vietnamese-to-English ones, sft-training.md), and a model whose start slipped past
# a short clip's end stayed silent; a late row whose stream would then pass its longest keeps its
# cached delay.
LATE_START = 0.05
LATE_START_FRAMES = 12  # past the source's end: the source EOS or the gap's first frame is frame 0
# Batches pad to a multiple of this many frames, so the model sees few shapes.
FRAME_BUCKET = 16
MAX_LEAD = 25  # frames of silence before the first Vietnamese, drawn from [0, 2 s]
SHORT_GAP = (4, 19)  # a pause between sentences, 0.3-1.5 s
LONG_GAP = (38, 125)  # a 3-10 s silence, to practise staying quiet
LONG_GAP_PROBABILITY = 0.1
MAX_OVERHANG = 25  # the previous English may run at most 2 s into the next Vietnamese
SILENCE_WARMUP = 16  # frames of Mimi's start-up codes when it encodes from a fresh state
# Teacher-forced validation streams and their batch order use one fixed seed, so every run and
# validate.py score the same streams, and --val-batches takes a random subset.
VALIDATION_SEED = 0


@dataclass(frozen=True)
class Piece:
    row: int  # index into the rows
    source_start: int
    target_start: int


@dataclass(frozen=True)
class Stream:
    pieces: tuple[Piece, ...]
    frames: int


def frame_range(stream_seconds: list[float]) -> tuple[int, int]:
    """--stream-seconds MIN MAX as stream lengths in frames."""
    shortest, longest = stream_seconds
    return round(shortest * FRAME_RATE), round(longest * FRAME_RATE)


def _gap(rng: random.Random) -> int:
    low, high = LONG_GAP if rng.random() < LONG_GAP_PROBABILITY else SHORT_GAP
    return rng.randint(low, high)


def plan_streams(rows: list[Row], min_frames: int, max_frames: int, seed: int) -> list[Stream]:
    """Cut every row, once, into streams of min_frames..max_frames frames (the last of a direction
    may be shorter), but for a SINGLE_ROW_STREAMS share of streams that hold one row. A stream
    holds one direction's rows: the directions are planned one after the other, from one draw."""
    longest = max(max(row.source_frames, row.target_end) for row in rows)
    if min_frames <= longest + MAX_LEAD + 1:
        raise ValueError(f"Streams of {min_frames} frames cannot hold a {longest}-frame row")
    # A stream still under min_frames (at most min_frames - 1) grows by at most a long gap and a
    # row, which must fit.
    if max_frames < min_frames - 1 + LONG_GAP[1] + longest:
        raise ValueError(
            f"Stream lengths {min_frames}..{max_frames} must differ by at least a "
            f"{longest}-frame row and a {LONG_GAP[1]}-frame gap"
        )
    rng = random.Random(seed)
    streams: list[Stream] = []
    for direction in sorted({row.direction for row in rows}):
        streams += _plan_direction(rows, direction, min_frames, max_frames, rng)
    return streams


def _plan_direction(
    rows: list[Row], direction: str, min_frames: int, max_frames: int, rng: random.Random
) -> list[Stream]:
    docs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    units: list[list[int]] = []
    for index, row in enumerate(rows):
        if row.direction != direction:
            continue
        if row.doc is None:
            units.append([index])
        else:
            docs[row.doc].append((row.seq, index))
    units.extend([index for _, index in sorted(doc)] for doc in docs.values())
    rng.shuffle(units)

    streams: list[Stream] = []
    pieces: list[Piece] = []
    budget = source_end = target_end = frames = 0
    single = False

    def close() -> None:
        streams.append(Stream(tuple(pieces), frames))

    for index in (index for unit in units for index in unit):
        row = rows[index]
        for _ in range(2):  # a row that overflows the stream starts the next one
            if pieces:
                source_start = max(source_end + _gap(rng), target_end - MAX_OVERHANG)
            else:
                budget = rng.randint(min_frames, max_frames)
                single = rng.random() < SINGLE_ROW_STREAMS
                source_start = rng.randint(0, MAX_LEAD)
            target_start = max(source_start + row.start_frames, target_end)
            if rng.random() < LATE_START:
                late = source_start + rng.randint(row.source_frames // 2, row.source_frames + LATE_START_FRAMES)
                late_end = max(source_start + row.source_frames, late + row.target_frames - row.delay_frames)
                if late_end + 1 <= max_frames:  # the length guarantees hold as for a cached delay
                    target_start = max(target_start, late)
            next_source_end = source_start + row.source_frames
            next_target_end = target_start + row.target_frames - row.delay_frames
            next_frames = max(next_source_end, next_target_end) + 1  # the EOS frames
            # Close a single-row stream after its row, any other when the row overflows its budget,
            # unless it is still short and the row fits the longest stream.
            if pieces and (
                single
                or (next_frames > budget and (frames >= min_frames or next_frames > max_frames))
            ):
                close()
                pieces, source_end, target_end = [], 0, 0
                continue
            pieces.append(Piece(index, source_start, target_start))
            source_end, target_end, frames = next_source_end, next_target_end, next_frames
            break
    if pieces:
        close()
    return streams


def assemble(
    stream: Stream, rows: list[Row], silence: torch.Tensor, layout: Layout,
    source_quiet: torch.Tensor | None = None,
) -> torch.Tensor:
    """The stream's int32 codes [33, frames]; `silence` is the `silence_track`. The source holds
    `source_quiet` (a `rooms.room_codes` track; default the silence) wherever nobody speaks; the target
    stays silent."""
    q = layout.dep_q
    frames = stream.frames
    # Silence encoded from a fresh state, as inference encodes a stream: start-up codes, then
    # steady silence.
    quiet = silence[:, :frames]
    last_piece = stream.pieces[-1]
    last = rows[last_piece.row]
    text_eos = last_piece.target_start + last.target_frames - last.delay_frames
    source_eos = last_piece.source_start + last.source_frames
    codes = torch.full((layout.rows, frames), -1, dtype=torch.int32)
    codes[0, :text_eos] = layout.pad_id
    codes[0, text_eos] = int(last.codes[0, last.target_frames])
    codes[1 : 1 + q, :text_eos] = quiet[:, :text_eos]
    codes[1 + q :] = quiet if source_quiet is None else source_quiet[:, :frames]
    codes[1 + q :, source_eos] = layout.card
    for piece in stream.pieces:
        row = rows[piece.row]
        source = torch.from_numpy(row.codes)
        if (source[0, : row.delay_frames] != layout.pad_id).any():
            raise RuntimeError(f"{row.id} has text before its delay")
        if int(source[1 + q, row.source_frames]) != layout.card:
            raise RuntimeError(f"{row.id} has no source EOS at {row.source_frames}")
        span = row.target_frames - row.delay_frames
        codes[0, piece.target_start : piece.target_start + span] = source[
            0, row.delay_frames : row.target_frames
        ]
        # Each clip was encoded from a fresh Mimi state: English silence before the first word
        # within the start-up window stays silence, and so does the Vietnamese first frame unless
        # the clip opens the stream, where inference starts fresh too.
        words = (source[0, row.delay_frames : row.target_frames] != layout.pad_id).nonzero()
        first_word = row.delay_frames + int(words[0]) if len(words) else row.target_frames
        audio_from = max(row.delay_frames, min(SILENCE_WARMUP, first_word))
        start = piece.target_start + audio_from - row.delay_frames
        codes[1 : 1 + q, start : piece.target_start + span] = source[
            1 : 1 + q, audio_from : row.target_frames
        ]
        source_from = 0 if piece.source_start == 0 else 1
        codes[1 + q :, piece.source_start + source_from : piece.source_start + row.source_frames] = source[
            1 + q :, source_from : row.source_frames
        ]
    return codes


def silence_track(mimi: Any, frames: int, codebooks: int | None = None) -> torch.Tensor:
    """Mimi codes of `frames` frames of digital silence encoded from a fresh state,
    [codebooks, frames] int32 on the CPU: `codebooks` of them (the cache's, which a student's
    Mimi has fewer of), or as many as `mimi` is set to.

    It is the silence a stream holds wherever nobody speaks: start-up codes in the first
    SILENCE_WARMUP frames (frames 0-14 differ, codebook 0 settles at frame 11; they equal a cached
    row's delay frames), then steady silence, a few repeating patterns rather than one frame
    (codebook 0 is 1316 on ~99% of frames).
    """
    samples = int(round(mimi.sample_rate / mimi.frame_rate)) * frames
    return encode_track(mimi, torch.zeros(samples), codebooks)


def model_rows(layout: Layout, codebooks: int) -> list[int]:
    """The rows of a stream a model with `codebooks` a stream reads: text, the target's first
    `codebooks` and the source's first (Mimi's codebooks are residual: a model with fewer reads the
    cache's first ones, [phone-student](../../docs/agent-memory/phone-student.md))."""
    if not 1 <= codebooks <= layout.dep_q:
        raise ValueError(f"a model with {codebooks} codebooks a stream cannot read a cache of {layout.dep_q}")
    q = layout.dep_q
    return [0, *range(1, 1 + codebooks), *range(1 + q, 1 + q + codebooks)]


class StreamDataset(Dataset):
    """Streams planned once from `rows` with `seed`; each item is assembled on demand, at the
    cache's layout, then cut to `codebooks` a stream (`model_rows`; None keeps them all). `layout`
    stays the cache's: a cut item's rows are `model_rows`, not `layout`'s."""

    def __init__(
        self,
        rows: list[Row],
        layout: Layout,
        silence: torch.Tensor,
        stream_frames: tuple[int, int],
        seed: int,
        codebooks: int | None = None,
        rooms: torch.Tensor | None = None,
        room_share: float = 0.0,
    ):
        if silence.shape[0] != layout.dep_q:
            raise ValueError(f"Silence has {silence.shape[0]} codebooks, need {layout.dep_q}")
        if silence.shape[1] < stream_frames[1]:
            raise ValueError("The silence track must cover the longest stream")
        if rooms is not None and rooms.shape[1:] != silence.shape:
            raise ValueError(f"Rooms of shape {tuple(rooms.shape[1:])} differ from the silence's {tuple(silence.shape)}")
        if not 0.0 <= room_share <= 1.0 or (room_share > 0) != (rooms is not None):
            raise ValueError(f"A room share of {room_share} needs rooms exactly when it is above 0")
        self.rows = rows
        self.layout = layout
        self.silence = silence
        self.rooms = rooms
        self.room_share = room_share
        self.seed = seed
        self.keep = None if codebooks is None or codebooks == layout.dep_q else model_rows(layout, codebooks)
        self.streams = plan_streams(rows, *stream_frames, seed)

    def __len__(self) -> int:
        return len(self.streams)

    def source_quiet(self, index: int) -> torch.Tensor:
        """What stream `index`'s source holds wherever nobody speaks, [dep_q, silence frames] at
        the cache's codebooks: a room drawn for it (with probability `room_share`, from the seed
        and the index, so the same on a resume), or the silence."""
        if self.rooms is None:
            return self.silence
        rng = np.random.default_rng([self.seed, index])
        if rng.random() >= self.room_share:
            return self.silence
        return self.rooms[int(rng.integers(len(self.rooms)))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        stream = self.streams[index]
        codes = assemble(stream, self.rows, self.silence, self.layout, self.source_quiet(index))
        return {
            "codes": codes if self.keep is None else codes[self.keep],
            "frames": stream.frames,
            "source_frames": sum(self.rows[piece.row].source_frames for piece in stream.pieces),
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
        source_frames = sum(row.source_frames for row in self.rows)
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
