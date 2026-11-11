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
"""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from training.paths import CACHE_FORMAT, DEFAULT_DIRECTION, DEFAULT_TOKENIZER, cpu_budget, swap_slots

FRAME_RATE = 12.5
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
# up to 1.7% of Vietnamese-to-English ones, sft-training.md), and a model whose start slipped past
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
PHOMT_ID = re.compile(r"^(phomt_s\d+)r(\d+)$")  # consecutive rows of one source chunk: doc, seq
PACK_WORKERS = 32


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
    codes: np.ndarray  # int32 [33, T], memory-mapped
    source_frames: int  # source length; the cached source EOS sits here
    target_frames: int  # target length including the delay; the cached text EOS sits here
    delay_frames: int
    doc: str | None  # rows of one doc join streams together, in seq order
    seq: int
    direction: str  # its shard's, <source>-<target>


@dataclass(frozen=True)
class Piece:
    row: int  # index into the rows
    source_start: int
    target_start: int


@dataclass(frozen=True)
class Stream:
    pieces: tuple[Piece, ...]
    frames: int


PACK_VERSION = 5  # 3: docs.json; 4: the shard's direction, the index by source and target; 5: tokenizer, text.bin
PACK_DIR = ".packed"  # beside the shards: <cache-dir>/.packed/<shard stem>/


def _pack_dir(shard_path: Path) -> Path:
    return shard_path.parent / PACK_DIR / shard_path.stem


def _pack_stamp(shard_path: Path) -> dict[str, int]:
    stat = shard_path.stat()
    return {"version": PACK_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _pack_is_current(shard_path: Path) -> bool:
    meta = _pack_dir(shard_path) / "pack.json"
    return meta.is_file() and json.loads(meta.read_text())["stamp"] == _pack_stamp(shard_path)


def pack_shard(shard_path: Path) -> None:
    """Write a shard's rows as one flat int32 array plus an index, which loads memory-mapped
    in milliseconds; unpickling the shard's per-row tensors takes ~0.2 s. The rows' text tracks
    also go to `text.bin` in row order, so a run on another tokenizer reads ~3% of the codes to
    check them (`load_rows`)."""
    payload = torch.load(shard_path, map_location="cpu")
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"Unsupported cache format in {shard_path}")
    if float(payload["frame_rate"]) != FRAME_RATE:
        raise RuntimeError(f"{shard_path} has frame rate {payload['frame_rate']}")
    config = payload["config"]
    layout = Layout(
        dep_q=int(config["dep_q"]),
        card=int(config["card"]),
        pad_id=int(config["existing_text_padding_id"]),
    )
    if 2 * layout.dep_q != int(config["n_q"]):
        raise RuntimeError(f"{shard_path} has unequal source and target codebooks")
    # A sample's vi_frames and en_frames are each language's clip; the direction says which is heard.
    direction = payload.get("direction", DEFAULT_DIRECTION)
    source, target = direction.split("-")
    chunks, texts, index, ids, docs = [], [], [], [], []
    offset = 0
    for sample in payload["samples"]:
        codes = sample["codes"]
        if codes.dtype != torch.int32 or codes.ndim != 2 or codes.shape[0] != layout.rows:
            raise RuntimeError(
                f"{shard_path} id={sample['id']} codes are {codes.dtype} {tuple(codes.shape)}"
            )
        chunks.append(codes.reshape(-1).numpy())
        texts.append(codes[0].numpy())
        frames = int(codes.shape[1])
        index.append(
            (
                offset,
                frames,
                int(sample[f"{source}_frames"]),
                int(sample[f"{target}_frames"]),
                int(sample["target_delay_frames"]),
            )
        )
        ids.append(str(sample["id"]))
        match = PHOMT_ID.match(str(sample["id"]))
        if "doc" in sample:
            docs.append([str(sample["doc"]), int(sample["seq"])])
        elif match:
            docs.append([match.group(1), int(match.group(2))])
        else:
            docs.append(None)
        offset += codes.numel()
    target = _pack_dir(shard_path)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    (np.concatenate(chunks) if chunks else np.empty(0, np.int32)).tofile(temporary / "codes.bin")
    (np.concatenate(texts) if texts else np.empty(0, np.int32)).tofile(temporary / "text.bin")
    np.save(temporary / "index.npy", np.array(index, dtype=np.int64).reshape(-1, 5))
    (temporary / "ids.json").write_text(json.dumps(ids))
    (temporary / "docs.json").write_text(json.dumps(docs))
    (temporary / "pack.json").write_text(
        json.dumps({"stamp": _pack_stamp(shard_path), "layout": dataclasses.asdict(layout), "direction": direction,
                    "tokenizer": payload.get("tokenizer", DEFAULT_TOKENIZER.name)})
    )
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)


def pack(cache_dirs: list[Path]) -> list[Path]:
    """The shards of `cache_dirs`, each with a current pack (`pack_shard`), packed in parallel
    where a shard is new, changed or packed by another `PACK_VERSION`. A shard is a
    `shard_*.pt`, or a pack whose `.pt` `--drop-shards` deleted (packs only on the box): that pack
    is used as it is, and must be of this `PACK_VERSION`; a pack left without its shard any other
    way raises, so deleting a shard never leaves its rows in a run unnoticed. Either way a
    shard's path is `<cache-dir>/<stem>.pt`, in name order, so dropping files keeps the rows' order."""
    shard_paths = []
    for d in cache_dirs:
        built = {path.name for path in d.glob("shard_*.pt") if path.exists()}  # not a broken link
        dropped = [d / f"{p.name}.pt" for p in (d / PACK_DIR).glob("shard_*") if f"{p.name}.pt" not in built]
        for path in dropped:
            meta = _pack_dir(path) / "pack.json"
            info = json.loads(meta.read_text()) if meta.is_file() else {}
            if not info.get("dropped"):
                raise RuntimeError(f"{_pack_dir(path)} has no shard {path.name} and was not dropped by --drop-shards: "
                                   f"fetch the shard from the Hub, or delete the pack")
            if info["stamp"]["version"] != PACK_VERSION:
                raise RuntimeError(f"{_pack_dir(path)} is not a version {PACK_VERSION} pack and {path.name} is gone: "
                                   f"fetch it from the Hub to pack it again")
        shard_paths += sorted([d / name for name in built] + dropped)
    stale = [path for path in shard_paths if path.exists() and not _pack_is_current(path)]
    if stale:
        print(f"[streams] packing {len(stale)} of {len(shard_paths)} cache shards", flush=True)
        # Forked like train.py's DataLoader workers (named: 3.14 defaults to forkserver): packing
        # touches only files and host memory, and fork neither re-runs the caller's script, pickles
        # the arguments' state nor starts a fresh BLAS thread pool per worker.
        workers = min(PACK_WORKERS, max(1, int(cpu_budget())), len(stale))
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            list(pool.map(pack_shard, stale))
    return shard_paths


def load_rows(cache_dirs: list[Path], max_frames: int, tokenizer: Path) -> tuple[list[Row], Layout]:
    """Every row of the grounded cache shards in `cache_dirs` that fits `max_frames` and whose
    text reads the same on `tokenizer`, the run's.

    Rows load memory-mapped from each shard's pack (`pack_shard`), built in parallel the first
    time a shard is seen or after it changes. Startup then reads only the indexes, and the codes
    (69 GiB for the full `en_speech` cache) stay on disk until a stream needs them.

    A shard tokenized on another tokenizer loads only if the run's is a swap of it (`swap_slots`),
    and then without the rows whose text uses a slot the swap gave a new piece: machine English
    holds a few (two-way.md). Any other mismatch raises."""
    shard_paths = pack(cache_dirs)
    rows: list[Row] = []
    layout: Layout | None = None
    dropped = 0
    swapped: dict[str, np.ndarray] = {}  # by a shard's tokenizer: the slots the run's reassigned
    reassigned = 0
    for shard_path in shard_paths:
        directory = _pack_dir(shard_path)
        meta = json.loads((directory / "pack.json").read_text())
        shard_layout = Layout(**meta["layout"])
        if layout is None:
            layout = shard_layout
        elif shard_layout != layout:
            raise RuntimeError(f"{shard_path} has layout {shard_layout}, others {layout}")
        # A private mapping that keeps no file descriptor open (numpy's memmap holds one per
        # shard, past the usual limit of 1,024), writable for torch.from_numpy, never written.
        path = directory / "codes.bin"
        codes = torch.from_file(
            str(path), shared=False, size=path.stat().st_size // 4, dtype=torch.int32
        ).numpy()
        ids = json.loads((directory / "ids.json").read_text())
        docs = json.loads((directory / "docs.json").read_text())
        index = np.load(directory / "index.npy")
        readable = np.ones(len(index), dtype=bool)
        on = meta["tokenizer"]
        if on != Path(tokenizer).name:
            if on not in swapped:
                try:
                    swapped[on] = np.array(swap_slots(tokenizer, on))
                except RuntimeError as exc:
                    raise RuntimeError(f"{shard_path}: {exc}") from None
            text = np.fromfile(directory / "text.bin", dtype=np.int32)
            if len(text) != int(index[:, 1].sum()):
                raise RuntimeError(f"{directory}: text.bin holds {len(text)} frames, its index {int(index[:, 1].sum())}")
            if len(index):
                starts = np.concatenate([[0], np.cumsum(index[:-1, 1])])
                readable = ~np.logical_or.reduceat(np.isin(text, swapped[on], kind="table"), starts)
                reassigned += int((~readable).sum())
        for row_id, doc, keep, (offset, frames, source_frames, target_frames, delay) in zip(
            ids, docs, readable.tolist(), index.tolist(), strict=True
        ):
            if not keep:
                continue
            if frames > max_frames:
                dropped += 1
                continue
            row_codes = codes[offset : offset + layout.rows * frames]
            rows.append(
                Row(
                    id=row_id,
                    codes=row_codes.reshape(layout.rows, frames),
                    source_frames=source_frames,
                    target_frames=target_frames,
                    delay_frames=delay,
                    doc=doc[0] if doc else None,
                    seq=doc[1] if doc else 0,
                    direction=meta["direction"],
                )
            )
    if layout is None or not rows:
        raise RuntimeError(f"No usable shard_*.pt cache rows in {cache_dirs}")
    if reassigned:
        print(f"[streams] dropped {reassigned} rows whose text uses a slot {Path(tokenizer).name} reassigned")
    if dropped:
        print(f"[streams] dropped {dropped} rows over {max_frames} frames; kept {len(rows)}")
    return rows, layout


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
    longest = max(max(row.source_frames, row.target_frames) for row in rows)
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
            target_start = max(source_start + row.delay_frames, target_end)
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
    stream: Stream, rows: list[Row], silence: torch.Tensor, layout: Layout
) -> torch.Tensor:
    """The stream's int32 codes [33, frames]; `silence` is the `silence_track`."""
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
    codes[1 + q :] = quiet
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


def silence_track(mimi: Any, frames: int) -> torch.Tensor:
    """Mimi codes of `frames` frames of digital silence encoded from a fresh state,
    [codebooks, frames] int32 on the CPU.

    It is the silence a stream holds wherever nobody speaks: start-up codes in the first
    SILENCE_WARMUP frames (frames 0-14 differ, codebook 0 settles at frame 11; they equal a cached
    row's delay frames), then steady silence, a few repeating patterns rather than one frame
    (codebook 0 is 1316 on ~99% of frames).
    """
    samples = int(round(mimi.sample_rate / mimi.frame_rate)) * frames
    device = next(mimi.parameters()).device
    with torch.no_grad():
        codes = mimi.encode(torch.zeros(1, 1, samples, device=device))[0]
    if codes.shape[-1] != frames:
        raise RuntimeError(f"Mimi encoded {codes.shape[-1]} silence frames, expected {frames}")
    return codes.to(device="cpu", dtype=torch.int32)


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
        if silence.shape[1] < stream_frames[1]:
            raise ValueError("The silence track must cover the longest stream")
        self.rows = rows
        self.layout = layout
        self.silence = silence
        self.streams = plan_streams(rows, *stream_frames, seed)

    def __len__(self) -> int:
        return len(self.streams)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stream = self.streams[index]
        return {
            "codes": assemble(stream, self.rows, self.silence, self.layout),
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


if __name__ == "__main__":
    # python -m training.finetune.streams [--drop-shards] CACHE_DIR...: build the packs before the first
    # run; --drop-shards then deletes each packed shard_*.pt (packs only: the trainer reads nothing else),
    # for caches whose shards are on the Hub.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-shards", action="store_true")
    parser.add_argument("cache_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    packed = pack(args.cache_dirs)
    print(f"{len(packed):,} shards' packs current in {len(args.cache_dirs)} cache directories")
    if args.drop_shards:
        built = [path for path in packed if path.exists()]
        freed = sum(path.lstat().st_size for path in built)  # a link frees only itself
        for path in built:
            meta = _pack_dir(path) / "pack.json"  # marked first: a pack without its shard loads only so
            meta.write_text(json.dumps(json.loads(meta.read_text()) | {"dropped": True}))
            path.unlink()
        print(f"dropped {len(built):,} shards, {freed / 1e9:.1f} GB")
