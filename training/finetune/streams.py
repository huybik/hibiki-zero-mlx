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
With rooms (`room_codes`; GRPO's) a share of streams hold a room's noise in the source instead
of silence, its lead, gaps and the stretch after its EOS alike, as a phone's microphone hears a
room between sentences.

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

A row's target never runs ahead of its source (`start_floor`): it starts at its cached delay, or
later where a target token would otherwise come before the source it translates has been heard.
The source's word times are the text track of the same row in the other direction's cache (its
pair), so a run reads both directions' caches. A row whose pair is missing (the other build
rejected it) is left out, unless it was rejected for fast speech: that row trains at its cached
delay. A row whose Vietnamese runs on past its last word (`Aligner.max_tail_s`) is left out too:

    EN   [I can't bear to go away from you]
    VI     [em không chịu nổi phải xa anh]        cached delay: ahead of the English
    VI              [em không chịu nổi phải xa anh] floored
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
from training.prep import augment
from training.prep.text_timing import ALIGNERS, CAPACITY_REASONS

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
# up to 2.6% of Vietnamese-to-English ones, sft-training.md), and a model whose start slipped past
# a short clip's end stayed silent; a late row whose stream would then pass its longest keeps its
# cached delay.
LATE_START = 0.05
LATE_START_FRAMES = 12  # past the source's end: the source EOS or the gap's first frame is frame 0
# A target text token starts at least START_MARGIN frames after its share of the source has been
# heard (`start_floor`). The cached delay counts from the source clip's start: Vietnamese to
# English runs behind anyway (its English is longer and opens on the recording's silence), but
# English to Vietnamese put a Vietnamese word over 1 s ahead of its English in 20-30% of rows and
# began before the English speaker in 4-23% (sft-training.md#target-timing).
START_MARGIN = 4  # 0.32 s
LAST_TOKEN = 5  # the source's last token is heard 0.4 s after it starts; any other by the next one's start
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
    start_frames: int  # where a plan starts the target after the source: the delay or `start_floor`
    doc: str | None  # rows of one doc join streams together, in seq order
    seq: int
    direction: str  # its shard's, <source>-<target>

    @property
    def target_end(self) -> int:
        """The target's end after the source's start, as a plan places it unless it waits."""
        return self.start_frames + self.target_frames - self.delay_frames


@dataclass(frozen=True)
class Piece:
    row: int  # index into the rows
    source_start: int
    target_start: int


@dataclass(frozen=True)
class Stream:
    pieces: tuple[Piece, ...]
    frames: int


# 3: docs.json; 4: the shard's direction, the index by source and target; 5: tokenizer, text.bin;
# 6: the ids the shard's build rejected for the aligner's frame grids (`capacity`)
PACK_VERSION = 6
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
                    "tokenizer": payload.get("tokenizer", DEFAULT_TOKENIZER.name),
                    "capacity": [str(item["id"]) for item in payload.get("rejected", [])
                                 if any(reason in item["reason"] for reason in CAPACITY_REASONS)]})
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


@dataclass(frozen=True)
class Tokens:
    """A pack's text tokens: each row's token frames from its target clip's start (the delay
    removed, the text EOS left out), in row order, `frames[offsets[r] : offsets[r + 1]]`, and each
    row's target clip in frames (`spans`, where its text EOS sits)."""

    frames: np.ndarray  # int16
    offsets: np.ndarray  # int64, one more than the rows
    spans: np.ndarray  # int64

    def rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The token frames of `rows`, back to back, and each row's count."""
        counts = self.offsets[rows + 1] - self.offsets[rows]
        firsts = np.repeat(self.offsets[rows] - (np.cumsum(counts) - counts), counts)
        return self.frames[firsts + np.arange(int(counts.sum()))], counts


def read_tokens(directory: Path, index: np.ndarray, pad_id: int) -> tuple[np.ndarray, Tokens]:
    """A pack's text track (`text.bin`, every row's frames back to back) and its `Tokens`."""
    text = np.fromfile(directory / "text.bin", dtype=np.int32)
    lengths = index[:, 1]
    if len(text) != int(lengths.sum()):
        raise RuntimeError(f"{directory}: text.bin holds {len(text)} frames, its index {int(lengths.sum())}")
    row = np.repeat(np.arange(len(index)), lengths)
    frame = np.arange(len(text)) - np.repeat(np.cumsum(lengths) - lengths, lengths) - index[row, 4]
    spans = index[:, 3] - index[:, 4]
    token = (text != pad_id) & (text >= 0) & (frame >= 0) & (frame < spans[row])
    counts = np.bincount(row[token], minlength=len(index))
    return text, Tokens(frame[token].astype(np.int16), np.concatenate([[0], np.cumsum(counts)]), spans)


def start_floor(source: np.ndarray, source_counts: np.ndarray, target: np.ndarray,
                target_counts: np.ndarray) -> np.ndarray:
    """Each row's smallest delay at which no target token starts before its share of the source
    has been heard: target token i of N waits START_MARGIN past source token ceil((i + 1) M / N) of
    M, as a proportional aligner reads a translation in order, a source token heard by the next
    one's start (the last LAST_TOKEN after its own). Token frames count from each clip's start,
    each side's rows back to back (`Tokens.rows`); a row with no token on either side gets 0."""
    floor = np.zeros(len(target_counts), dtype=np.int64)
    rows = np.repeat(np.arange(len(target_counts)), target_counts)
    n, m = target_counts[rows], source_counts[rows]
    i = np.arange(len(target)) - np.repeat(np.cumsum(target_counts) - target_counts, target_counts)
    first = np.repeat(np.cumsum(source_counts) - source_counts, target_counts)
    keep = m > 0
    rows, n, m, i, first = rows[keep], n[keep], m[keep], i[keep], first[keep]
    after = ((i + 1) * m + n - 1) // n  # the source token after the one target token i needs: 1..m
    source = source.astype(np.int64)
    heard = np.where(after < m, source[first + np.minimum(after, m - 1)], source[first + m - 1] + LAST_TOKEN)
    np.maximum.at(floor, rows, heard + START_MARGIN - target[keep])
    return floor


def tails(frames: np.ndarray, counts: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Each row's frames from its last text token to its clip's end (the whole clip without one);
    rows back to back, as `Tokens.rows` gives them."""
    if not len(frames):
        return spans.copy()
    last = np.maximum(np.cumsum(counts) - 1, 0)
    return np.where(counts > 0, spans - frames[last].astype(np.int64), spans)


@dataclass
class _Pack:
    """A pack as `load_rows` reads it, its text track left behind once read."""

    path: Path
    direction: str
    ids: list[str]
    index: np.ndarray
    tokens: Tokens
    readable: np.ndarray  # rows whose text reads the same on the run's tokenizer


def load_rows(cache_dirs: list[Path], max_frames: int, tokenizer: Path,
              pair_dirs: list[Path] | tuple[Path, ...] = ()) -> tuple[list[Row], Layout]:
    """Every row of the grounded cache shards in `cache_dirs` that fits `max_frames` once its target
    starts where `start_floor` puts it, whose text reads the same on `tokenizer`, the run's, and
    whose pair is found.

    A row's pair is the row of the same id in the other direction's cache, among the shards of
    `cache_dirs` and `pair_dirs` (read only for their text): the pair's text track times this
    row's source, for `start_floor`. A row is left out when its pair is missing (the other
    direction's build rejected it, as an alignment failure, so its speech does not match its text)
    or is not of the same clip (its target is not this row's source's length within a frame; a
    shard all of whose pairs are other clips raises), except that a row the other build rejected
    for its aligner's frame grids (`CAPACITY_REASONS`: fast or dense speech, a sound row) trains at
    its cached delay; and when its Vietnamese, its own or its pair's, runs on past its last text token longer than
    the aligner allows (`Aligner.max_tail_s`: TTS that babbled past its text). A shard none of whose
    rows has a pair raises: its other direction was not given.

    Rows load memory-mapped from each shard's pack (`pack_shard`), built in parallel the first
    time a shard is seen or after it changes. Startup then reads the indexes and text tracks, and
    the codes (69 GiB for the full `en_speech` cache) stay on disk until a stream needs them.

    A shard tokenized on another tokenizer loads only if the run's is a swap of it (`swap_slots`),
    and then without the rows whose text uses a slot the swap gave a new piece: machine English
    holds a few (two-way.md). Any other mismatch raises."""
    shard_paths = pack(cache_dirs)
    trained = set(shard_paths)
    layout: Layout | None = None
    swapped: dict[str, np.ndarray] = {}  # by a shard's tokenizer: the slots the run's reassigned
    packs: list[_Pack] = []
    untimed: dict[str, set[str]] = defaultdict(set)  # by direction: the ids its build could not time
    for shard_path in shard_paths + [path for path in pack(list(pair_dirs)) if path not in trained]:
        directory = _pack_dir(shard_path)
        meta = json.loads((directory / "pack.json").read_text())
        shard_layout = Layout(**meta["layout"])
        if layout is None:
            layout = shard_layout
        elif shard_layout != layout:
            raise RuntimeError(f"{shard_path} has layout {shard_layout}, others {layout}")
        index = np.load(directory / "index.npy")
        text, tokens = read_tokens(directory, index, layout.pad_id)
        readable = np.ones(len(index), dtype=bool)
        on = meta["tokenizer"]
        if shard_path in trained and on != Path(tokenizer).name:
            if on not in swapped:
                try:
                    swapped[on] = np.array(swap_slots(tokenizer, on))
                except RuntimeError as exc:
                    raise RuntimeError(f"{shard_path}: {exc}") from None
            if len(index):
                starts = np.concatenate([[0], np.cumsum(index[:-1, 1])])
                readable = ~np.logical_or.reduceat(np.isin(text, swapped[on], kind="table"), starts)
        ids = json.loads((directory / "ids.json").read_text())
        packs.append(_Pack(shard_path, meta["direction"], ids, index, tokens, readable))
        untimed[meta["direction"]].update(meta["capacity"])
    if layout is None:
        raise RuntimeError(f"No usable shard_*.pt cache rows in {cache_dirs}")
    # By direction, each row id's first pack and row, as pack << 32 | row: a clean cache and its
    # noisy copy hold the same rows and text.
    found: dict[str, dict[str, int]] = defaultdict(dict)
    for number, shard in enumerate(packs):
        place = found[shard.direction]
        for row, row_id in enumerate(shard.ids):
            place.setdefault(row_id, number << 32 | row)

    rows: list[Row] = []
    dropped: dict[str, int] = defaultdict(int)
    later: list[np.ndarray] = []  # frames past the cached delay, of each row the floor moved
    unfloored = 0  # rows kept at their cached delay: the other direction could not time them
    for shard in packs[: len(shard_paths)]:
        count = len(shard.index)
        source, target = shard.direction.split("-")
        place = found.get(f"{target}-{source}", {})
        # The pair's pack << 32 | row; -1 none, -3 none as the other build could not time the row,
        # -2 (below) a pair of another clip.
        fast = untimed.get(f"{target}-{source}", set())
        where = np.array([place.get(row_id, -3 if row_id in fast else -1) for row_id in shard.ids], dtype=np.int64)
        found_any = bool((where >= 0).any())
        floor = np.zeros(count, dtype=np.int64)
        source_tail = np.zeros(count, dtype=np.int64)
        paired = np.nonzero(where >= 0)[0]
        for number in np.unique(where[paired] >> 32).tolist():
            rows_here = paired[(where[paired] >> 32) == number]
            pair = packs[number].tokens
            pair_rows = where[rows_here] & 0xFFFFFFFF
            # The pair's target clip is this row's source: the same clip but for the delay's
            # rounding, or the id names another recording and its times are not this source's.
            other = np.abs(shard.index[rows_here, 2] - pair.spans[pair_rows]) > 1
            where[rows_here[other]] = -2
            rows_here, pair_rows = rows_here[~other], pair_rows[~other]
            heard, heard_counts = pair.rows(pair_rows)
            spoken, spoken_counts = shard.tokens.rows(rows_here)
            floor[rows_here] = start_floor(heard, heard_counts, spoken, spoken_counts)
            source_tail[rows_here] = tails(heard, heard_counts, pair.spans[pair_rows])
        if count and (where < 0).all():
            raise RuntimeError(
                f"{shard.path}: none of its {count} rows is in a {target}-{source} cache, which times its source: "
                f"pass that direction's cache with --cache-dir, or with --pair-cache-dir to read only its text"
                if not found_any else
                f"{shard.path}: every row's {target}-{source} pair found first is another clip (its target not this "
                f"row's source's length): a wrong cache among the other direction's"
            )
        # A row the other direction's aligner could not fit in its frame grids (fast or dense
        # speech) is sound: it trains at its cached delay, its source untimed.
        keep = shard.readable.copy()
        dropped["reassigned"] += int((~keep).sum())
        dropped["unpaired"] += int((keep & (where == -1)).sum())
        dropped["other clip"] += int((keep & (where == -2)).sum())
        keep &= (where >= 0) | (where == -3)
        target_tail = tails(shard.tokens.frames, np.diff(shard.tokens.offsets), shard.tokens.spans)
        for language, tail in ((target, target_tail), (source, source_tail)):
            limit = ALIGNERS[language].max_tail_s
            if limit is not None:
                over = keep & (tail > round(limit * FRAME_RATE))
                dropped["tail"] += int(over.sum())
                keep &= ~over
        delay = shard.index[:, 4]
        start = np.maximum(delay, floor)
        # The stream's frames this row needs alone: its source and EOS, its target and text EOS.
        needs = np.maximum(shard.index[:, 2], start + shard.tokens.spans) + 1
        over = keep & (needs > max_frames)
        dropped["long"] += int(over.sum())
        keep &= ~over
        later.append((start - delay)[keep & (start > delay)])
        unfloored += int((keep & (where == -3)).sum())
        if not keep.any():
            continue
        # A private mapping that keeps no file descriptor open (numpy's memmap holds one per
        # shard, past the usual limit of 1,024), writable for torch.from_numpy, never written.
        directory = _pack_dir(shard.path)
        path = directory / "codes.bin"
        codes = torch.from_file(
            str(path), shared=False, size=path.stat().st_size // 4, dtype=torch.int32
        ).numpy()
        docs = json.loads((directory / "docs.json").read_text())
        for row in np.nonzero(keep)[0].tolist():
            offset, frames, source_frames, target_frames, delay_frames = shard.index[row].tolist()
            doc = docs[row]
            rows.append(
                Row(
                    id=shard.ids[row],
                    codes=codes[offset : offset + layout.rows * frames].reshape(layout.rows, frames),
                    source_frames=source_frames,
                    target_frames=target_frames,
                    delay_frames=delay_frames,
                    start_frames=int(start[row]),
                    doc=doc[0] if doc else None,
                    seq=doc[1] if doc else 0,
                    direction=shard.direction,
                )
            )
    if not rows:
        raise RuntimeError(f"No usable shard_*.pt cache rows in {cache_dirs}")
    moved = np.concatenate(later)
    print(
        f"[streams] kept {len(rows)} rows; left out {dropped['reassigned']} whose text uses a slot "
        f"{Path(tokenizer).name} reassigned, {dropped['unpaired']} the other direction's cache lacks, "
        f"{dropped['other clip']} whose pair there is another clip, "
        f"{dropped['tail']} whose Vietnamese runs on past its last word, {dropped['long']} over {max_frames} "
        f"frames; {len(moved)} start after their cached delay"
        + (f", by {np.median(moved) / FRAME_RATE:.2f} s median" if len(moved) else "")
        + f"; {unfloored} at it, whose pair the other direction's aligner could not time (fast speech)",
        flush=True,
    )
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
    `source_quiet` (a `room_codes` track; default the silence) wherever nobody speaks; the target
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


def encode_track(mimi: Any, pcm: torch.Tensor, codebooks: int | None = None) -> torch.Tensor:
    """Mimi codes of `pcm` (1-D, Mimi's sample rate, whole frames) encoded from a fresh state, as
    inference encodes a stream, [codebooks, frames] int32 on the CPU: `codebooks` of them, or as
    many as `mimi` is set to."""
    frame = int(round(mimi.sample_rate / mimi.frame_rate))
    if pcm.ndim != 1 or len(pcm) % frame:
        raise ValueError(f"Mimi encodes whole {frame}-sample frames of one channel, not {tuple(pcm.shape)}")
    active = mimi.num_codebooks
    if codebooks is not None:
        mimi.set_num_codebooks(codebooks)
    device = next(mimi.parameters()).device
    try:
        with torch.no_grad():
            codes = mimi.encode(pcm.to(device=device, dtype=torch.float32)[None, None])[0]
    finally:
        mimi.set_num_codebooks(active)  # the caller's Mimi keeps its own count
    if codes.shape[-1] != len(pcm) // frame:
        raise RuntimeError(f"Mimi encoded {codes.shape[-1]} frames, expected {len(pcm) // frame}")
    return codes.to(device="cpu", dtype=torch.int32)


def room_codes(mimi: Any, bank: Any, count: int, frames: int, codebooks: int, seed: int) -> torch.Tensor:
    """Mimi codes of `count` rooms where nobody speaks (`augment.room_tracks` from the noise bank
    `bank`), each `frames` long from a fresh state: [count, codebooks, frames] int32 on the CPU."""
    samples = int(round(mimi.sample_rate / mimi.frame_rate)) * frames
    tracks = augment.room_tracks(bank, count, samples, seed)
    return torch.stack([encode_track(mimi, torch.from_numpy(track), codebooks) for track in tracks])


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
