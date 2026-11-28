"""The cache rows a run trains on (`load_rows`), memory-mapped from their packs (`packs.py`).

A row's target never runs ahead of its source (`start_floor`): it starts at its cached delay, or
later where a target token would otherwise come before the source it translates has been heard.
The source's word times are the text track of the same row in the other direction's cache (its
pair), so a run reads both directions' caches. A row whose pair is missing (the other build
rejected it) is left out, unless it was rejected for fast speech: that row trains at its cached
delay. A row whose Vietnamese runs on past its last word (`Aligner.max_tail_s`) is left out too:

    EN   [I can't bear to go away from you]
    VI     [em không chịu nổi phải xa anh]        cached delay: ahead of the English
    VI              [em không chịu nổi phải xa anh] floored

So is a Libriheavy row whose English text a bad book scan garbled (`packs.ocr_damaged`).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.cache.timing import ALIGNERS
from training.finetune.packs import FRAME_RATE, Layout, pack, pack_dir
from training.paths import swap_slots

# A target text token starts at least START_MARGIN frames after its share of the source has been
# heard (`start_floor`). The cached delay counts from the source clip's start: Vietnamese to
# English runs behind anyway (its English is longer and opens on the recording's silence), but
# English to Vietnamese put a Vietnamese word over 1 s ahead of its English in 20-30% of rows and
# began before the English speaker in 4-23% (sft-training.md#target-timing).
START_MARGIN = 4  # 0.32 s
LAST_TOKEN = 5  # the source's last token is heard 0.4 s after it starts; any other by the next one's start


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
    ocr: np.ndarray  # rows whose English is OCR-damaged (`packs.ocr_damaged`)


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
    the aligner allows (`Aligner.max_tail_s`: TTS that babbled past its text); and when it is a
    Libriheavy row whose English is OCR-damaged (`packs.ocr_damaged`, recorded in its pack). A shard none
    of whose rows has a pair raises: its other direction was not given.

    Rows load memory-mapped from each shard's pack (`packs.pack_shard`), built in parallel the first
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
        directory = pack_dir(shard_path)
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
        damaged = set(meta["ocr"])
        ocr = np.array([row_id in damaged for row_id in ids], dtype=bool)
        packs.append(_Pack(shard_path, meta["direction"], ids, index, tokens, readable, ocr))
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
        dropped["ocr"] += int((keep & shard.ocr).sum())
        keep &= ~shard.ocr
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
        directory = pack_dir(shard.path)
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
        f"{dropped['tail']} whose Vietnamese runs on past its last word, "
        f"{dropped['ocr']} whose English is OCR-damaged, {dropped['long']} over {max_frames} "
        f"frames; {len(moved)} start after their cached delay"
        + (f", by {np.median(moved) / FRAME_RATE:.2f} s median" if len(moved) else "")
        + f"; {unfloored} at it, whose pair the other direction's aligner could not time (fast speech)",
        flush=True,
    )
    return rows, layout
