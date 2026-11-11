"""Grounded-v2 cache rows, the CPU side of every cache build (`cache_build` runs the models): the
clips' decode, the row gates, the target's delay draw, the text timing in the workers, the
`[33, T]` code layout, a sample's fields and the shard file. A build has a direction, "vi-en"
(Vietnamese heard, English spoken and written) or "en-vi": its source and target languages
([cache-build](../../docs/agent-memory/cache-build.md),
[sft-data-caches](../../docs/agent-memory/sft-data-caches.md#row-layout)). No MLX, moshi or
transformers here, and torch only for a sample's codes and the write: the box imports it without
MLX, the Mac without CUDA."""

from __future__ import annotations

import math
import os
import random
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import CACHE_FORMAT, DEFAULT_DIRECTION, durable
from training.prep.audio import Rejected
from training.prep.text_timing import (
    ALIGNERS,
    Aligner,
    AlignmentError,
    CTCWordAligner,
    ctc_frames,
    sentencepiece_groups,
    timed_sentencepiece_tokens,
)

SAMPLE_RATE = 24_000  # Mimi
FRAME_SAMPLES = 1920  # one 12.5 Hz code frame
FRAME_RATE = 12.5
CTC_RATE = 16_000  # wav2vec2
DURATION_TOLERANCE_S = 0.02  # healthy clips decode exactly to their stored duration
CONFIG_KEYS = ("n_q", "dep_q", "card", "text_card", "existing_text_padding_id")
DIRECTIONS = ("vi-en", "en-vi")  # <source>-<target>


@dataclass(frozen=True)
class Settings:
    """What a build's rows depend on besides their audio and text; the shard records the first three."""

    max_frames: int = 280  # the trainer's frame cap
    min_alignment_score: float | None = None  # None: the target aligner's own
    direction: str = DEFAULT_DIRECTION
    delay_ratio: float = 0.5  # the delay's maximum, as a ratio of the source's duration
    seed: int = 1234  # the delay draw's

    @property
    def source(self) -> str:
        return self.direction.split("-")[0]

    @property
    def target(self) -> str:
        return self.direction.split("-")[1]

    @property
    def aligner(self) -> Aligner:
        return ALIGNERS[self.target]

    @property
    def min_score(self) -> float:
        return self.aligner.min_score if self.min_alignment_score is None else self.min_alignment_score


@dataclass
class Shard:
    """A source shard as its builder reads it, then as the build fills it.

    `rows` are the rows the builder's own gates kept, in source order, as (index in the source
    shard, row); a row holds `id`, `text_en`, `text_vi`, `vi_duration_s`, `en_duration_s`,
    `vi_audio` and `en_audio` (bytes, taken by the decode) and `fields`, copied into its sample
    (such as `doc`). `rejected` starts with the rows those gates dropped (`id`, `reason`)."""

    key: Any  # the builder's handle on it
    file: str  # its source path, in each sample's `source_row` (`<file>#r<index>`)
    split: str  # the delay draw's key
    size: int  # rows in the source shard
    rows: list[tuple[int, dict[str, Any]]]
    rejected: list[dict[str, str]]
    samples: list[dict[str, Any]] = field(default_factory=list)
    left: int = 0  # rows not yet sampled or rejected by the build
    dropped: list[tuple[int, dict[str, str]]] = field(default_factory=list)  # the build's rejections

    def __post_init__(self) -> None:
        self.left = len(self.rows)

    def reject(self, index: int, row_id: str, exc: Exception) -> None:
        self.dropped.append((index, {"id": row_id, "reason": f"{type(exc).__name__}: {exc}"}))
        self.left -= 1

    def done(self) -> Shard:
        """The shard once its last row is in: the build's rejections after the builder's own, in
        source order (the build rejects at decode or at alignment, windows apart)."""
        self.rejected += [entry for _, entry in sorted(self.dropped, key=lambda item: item[0])]
        self.dropped = []
        return self


@dataclass
class Row:
    """A row past the decode gates, waiting for its codes and text timing."""

    shard: Shard
    index: int
    data: dict[str, Any]  # the builder's row, its audio taken
    source24: np.ndarray
    target24: np.ndarray  # with the delay's silence in front
    target16: np.ndarray  # undelayed, normalized for wav2vec2
    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    source_frames: int
    target_frames: int


def prepare(source_audio: bytes, target_audio: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: both clips at 24 kHz for Mimi, and the target at 16 kHz for CTC, normalized to zero
    mean and unit variance over the clip as wav2vec2's feature extractor does."""
    import soxr

    from training.prep.audio import decode

    source, source_rate = decode(source_audio)
    target, target_rate = decode(target_audio)
    target16 = soxr.resample(target, target_rate, CTC_RATE).astype(np.float32)
    target16 = (target16 - target16.mean()) / np.sqrt(target16.var() + 1e-7)
    return (soxr.resample(source, source_rate, SAMPLE_RATE),
            soxr.resample(target, target_rate, SAMPLE_RATE), target16)


def target_delay_s(row_id: str, split: str, max_delay_s: float, seed: int) -> float:
    """Draw the target's left-silence delay from U[0, max_delay_s], seeded per row."""
    return random.Random(f"{seed}:{split}:{row_id}").uniform(0.0, max_delay_s)


def plan(
    shard: Shard,
    index: int,
    row: dict[str, Any],
    clips: tuple[np.ndarray, np.ndarray, np.ndarray],
    settings: Settings,
    tokenizer: Any,
) -> Row:
    """The decode gates, the delay draw and the text grouping; raises Rejected or AlignmentError."""
    source24, target24, target16 = clips
    source, target = settings.source, settings.target
    for side, pcm in ((source, source24), (target, target24)):
        # PyAV decodes a truncated or damaged clip without error, only shorter.
        seconds, stored = len(pcm) / SAMPLE_RATE, row[f"{side}_duration_s"]
        if abs(seconds - stored) > DURATION_TOLERANCE_S:
            raise Rejected(f"{side} audio decodes to {seconds:.2f} s, the row says {stored:.2f} s")
    source_frames = math.ceil(len(source24) / FRAME_SAMPLES)
    # Text EOS sits at target_frames, codec EOS at source_frames.
    frames = max(math.ceil(len(target24) / FRAME_SAMPLES), source_frames) + 1
    if frames > settings.max_frames:
        raise Rejected(f"{frames} frames exceed --max-frames {settings.max_frames}")
    if ctc_frames(len(target16)) < 1:
        raise Rejected(f"the {target} clip is shorter than one CTC frame")
    # The delay is drawn within the room the target clip leaves under the cap, so a long clip
    # gets a shorter delay instead of being rejected. A duration is float32 holding a multiple
    # of 0.01; round to the decimal it stores.
    room_s = ((settings.max_frames - 1) * FRAME_SAMPLES - len(target24)) / SAMPLE_RATE
    max_delay_s = min(settings.delay_ratio * round(row[f"{source}_duration_s"], 2), room_s)
    delay_s = target_delay_s(row["id"], shard.split, max_delay_s, settings.seed)
    pad = int(round(delay_s * SAMPLE_RATE))
    text = row[f"text_{target}"]
    if target == "vi":  # the tokenizer's normalizer is the identity, and its Vietnamese pieces are NFC
        text = unicodedata.normalize("NFC", text)
    lead, groups = sentencepiece_groups(text, tokenizer, settings.aligner.spoken)
    return Row(
        shard, index, row, source24, np.concatenate([np.zeros(pad, np.float32), target24]), target16,
        lead, groups, delay_s, int(round(delay_s * FRAME_RATE)), source_frames,
        math.ceil((pad + len(target24)) / FRAME_SAMPLES),
    )


_aligner: CTCWordAligner | None = None
_tokenizer: Any = None
_settings: Settings | None = None
_pcm_dir: Path | None = None


def init_worker(settings: Settings, tokenizer: Path, pcm_dir: Path | None) -> None:
    """Worker initializer: the process's aligner (the build's target language's), the text
    tokenizer, the build's settings and where to leave the clips (`pcm_dir`; None returns them in
    the reply)."""
    import sentencepiece

    global _aligner, _tokenizer, _settings, _pcm_dir
    _aligner = CTCWordAligner(settings.aligner, settings.min_score)
    _tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
    _settings, _pcm_dir = settings, pcm_dir


@dataclass
class Planned:
    """A row's `plan` from a worker without its shard: the clips inline (`pcm`), or as one raw
    float32 file of source24, target24 and target16 back to back (`file`, `lengths`) that `clips`
    reads."""

    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    source_frames: int
    target_frames: int
    pcm: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    file: str | None = None
    lengths: tuple[int, int, int] = (0, 0, 0)

    def clips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three clips; a file is read and unlinked at once. Read, not mapped: numpy's memmap
        keeps a descriptor open per file, and a window's 4,096 rows passed the box's limit of 1,024."""
        if self.pcm is not None:
            return self.pcm
        data = np.fromfile(self.file, np.float32)
        os.unlink(self.file)
        a, b = self.lengths[0], self.lengths[0] + self.lengths[1]
        return data[:a], data[a:b], data[b:]

    def row(self, shard: Shard, index: int, data: dict[str, Any]) -> Row:
        source24, target24, target16 = self.clips()
        return Row(shard, index, data, source24, target24, target16, self.lead, self.groups, self.delay_s,
                   self.delay_frames, self.source_frames, self.target_frames)


def prepare_row(row: dict[str, Any], source_audio: bytes, target_audio: bytes, split: str) -> Planned:
    """A worker's row of `prepare_rows`: `prepare` then `plan` (the decode gates, the delay draw, the
    text grouping), so the main process only reads the clips: piping ~1.4 MB of PCM a row through the pool's one result
    pipe held the main thread ~30 s per GigaSpeech 2 shard. Raises Rejected or AlignmentError."""
    shard = Shard(None, "", split, 0, [], [])
    p = plan(shard, 0, row, prepare(source_audio, target_audio), _settings, _tokenizer)
    out = Planned(p.lead, p.groups, p.delay_s, p.delay_frames, p.source_frames, p.target_frames)
    clips = (p.source24, p.target24, p.target16)
    if _pcm_dir is None:
        out.pcm = clips
        return out
    out.file = str(_pcm_dir / f"{uuid.uuid4().hex}.f32")
    with open(out.file, "wb") as f:
        for clip in clips:
            np.ascontiguousarray(clip, np.float32).tofile(f)
    out.lengths = tuple(len(c) for c in clips)
    return out


def prepare_rows(rows: list[tuple[dict[str, Any], bytes, bytes]], split: str) -> list[Planned | Rejected | AlignmentError]:
    """Worker: `prepare_row` over a job of rows (`row`, source audio, target audio), each row's
    Rejected or AlignmentError in its place. A job a row cost the main process each row's pool
    round trip under its one GIL, which the GPU thread shares (2026-11-11)."""
    out: list[Planned | Rejected | AlignmentError] = []
    for row, source_audio, target_audio in rows:
        try:
            out.append(prepare_row(row, source_audio, target_audio, split))
        except (Rejected, AlignmentError) as exc:
            out.append(exc)
    return out


def align_rows(
    log_probs: list[np.ndarray],
    rows: list[tuple[list[int], list[tuple[list[int], str]], int, int]],
    eos_id: int,
) -> list[tuple[list[int], list[int], float] | AlignmentError]:
    """Worker: for each row of a CTC batch (its log-probabilities, and its `lead`, `groups`, raw
    target frames and delay frames), its text tokens, their frames and the alignment score, or the
    AlignmentError that rejects it."""
    out: list[tuple[list[int], list[int], float] | AlignmentError] = []
    for probs, (lead, groups, raw_target_frames, delay_frames) in zip(log_probs, rows, strict=True):
        try:
            alignment = _aligner.align(probs, groups)
            tokens, frames = timed_sentencepiece_tokens(lead, groups, alignment, raw_target_frames, delay_frames,
                                                        eos_id)
        except AlignmentError as exc:
            out.append(exc)
            continue
        out.append((tokens, frames, alignment.score))
    return out


def assemble_codes(
    row_id: str,
    source_codes: np.ndarray,
    target_codes: np.ndarray,
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

    if source_codes.shape[0] < source_q:
        raise RuntimeError(f"The source's Mimi codes have {source_codes.shape[0]} codebooks, need {source_q}")
    if target_codes.shape[0] < dep_q:
        raise RuntimeError(f"The target's Mimi codes have {target_codes.shape[0]} codebooks, need {dep_q}")
    if any(token < 0 or token >= text_card for token in tokens):
        raise RuntimeError(f"Text token out of range for id={row_id}")
    if len(text_frames) != len(tokens) or len(set(text_frames)) != len(tokens):
        raise ValueError(f"Text frames must be unique and match tokens for id={row_id}")
    if text_frames != sorted(text_frames) or text_frames[0] < 0:
        raise ValueError(f"Text frames must be sorted and non-negative for id={row_id}")

    target_len = int(target_codes.shape[1])
    source_len = int(source_codes.shape[1])
    text_end = text_frames[-1] + 1
    codes = np.full((1 + n_q, max(text_end, target_len, source_len + 1)), -1, dtype=np.int32)
    codes[0, :text_end] = text_pad_id
    codes[0, text_frames] = tokens
    codes[1 : 1 + dep_q, :target_len] = target_codes[:dep_q]
    source = slice(1 + dep_q, 1 + n_q)
    codes[source, :source_len] = source_codes[:source_q]
    codes[source, source_len] = card
    return codes


def sample(
    row: Row,
    tokens: list[int],
    text_frames: list[int],
    score: float,
    source_codes: np.ndarray,
    target_codes: np.ndarray,
    cfg: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    """The row's cache sample: Python types and tensors only, since the trainer's `torch.load` keeps
    the `weights_only` default, which refuses NumPy scalars. `vi_frames` and `en_frames` are each
    language's clip, whichever is the source; the target's includes the delay."""
    import torch

    codes = assemble_codes(row.data["id"], source_codes, target_codes, tokens, text_frames, cfg)
    return {
        "id": row.data["id"],
        "split": row.shard.split,
        "codes": torch.from_numpy(codes),
        "frames": int(codes.shape[1]),
        f"{settings.source}_frames": row.source_frames,
        f"{settings.target}_frames": row.target_frames,
        "text_tokens": len(tokens),
        "target_delay_s": row.delay_s,
        "target_delay_frames": row.delay_frames,
        "text_en": row.data["text_en"],
        "text_vi": row.data["text_vi"],
        **row.data["fields"],
        "text_timing": settings.aligner.name,
        "alignment_score": score,
        "alignment_text": " ".join(spoken for _, spoken in row.groups),
        "source_row": f"{row.shard.file}#r{row.index}",
    }


def payload(shard: Shard, settings: Settings, cfg: dict[str, Any], source: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The shard file's contents. `source` describes the source shard (`file`, and its `bytes` or
    `rows`); `extra` holds the builder's own settings (such as `max_cer`)."""
    return {
        "format": CACHE_FORMAT,
        "sample_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "alignment_min_score": settings.min_score,
        "max_frames": settings.max_frames,
        "direction": settings.direction,
        "config": {key: int(cfg[key]) for key in CONFIG_KEYS},
        "source": source,
        **extra,
        "samples": shard.samples,
        "rejected": shard.rejected,
    }


def write(payload: dict[str, Any], dst: Path) -> None:
    """The shard file, written whole: a hidden `.tmp` flushed to the drive, then renamed."""
    import torch

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    torch.save(payload, tmp)
    durable(tmp)
    tmp.replace(dst)
    durable(dst.parent)


def mimi_hours(samples: list[dict[str, Any]]) -> float:
    """Hours of Mimi audio in `samples`: both clips, the delay included (the builds' speed unit)."""
    return sum(s["vi_frames"] + s["en_frames"] for s in samples) / FRAME_RATE / 3600
