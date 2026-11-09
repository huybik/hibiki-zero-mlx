"""Grounded-v2 cache rows, the CPU side of every cache build (`cache_build` runs the models): the
clips' decode, the row gates, the English delay draw, the text timing in the workers, the
`[33, T]` code layout, a sample's fields and the shard file
([cache-build](../../docs/agent-memory/cache-build.md),
[sft-data-caches](../../docs/agent-memory/sft-data-caches.md#row-layout)). No MLX, moshi or
transformers here, and torch only for a sample's codes and the write: the box imports it without
MLX, the Mac without CUDA."""

from __future__ import annotations

import math
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import CACHE_FORMAT, durable
from training.prep.audio import Rejected
from training.prep.text_timing import (
    ALIGNMENT_NAME,
    DEFAULT_MIN_ALIGNMENT_SCORE,
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


@dataclass(frozen=True)
class Settings:
    """What a build's rows depend on besides their audio and text; the shard records the first two."""

    max_frames: int = 280  # the trainer's frame cap
    min_alignment_score: float = DEFAULT_MIN_ALIGNMENT_SCORE
    delay_ratio: float = 0.5  # the delay's maximum, as a ratio of the Vietnamese duration
    seed: int = 1234  # the delay draw's


@dataclass
class Shard:
    """A source shard as its builder reads it, then as the build fills it.

    `rows` are the rows the builder's own gates kept, in source order, as (index in the source
    shard, row); a row holds `id`, `text_en`, `vi_duration_s`, `en_duration_s`, `vi_audio` and
    `en_audio` (bytes, taken by the decode) and `fields`, copied into its sample (such as
    `text_vi`). `rejected` starts with the rows those gates dropped (`id`, `reason`)."""

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
    source: dict[str, Any]  # the source row, its audio taken
    vi24: np.ndarray
    en24: np.ndarray  # with the delay's silence in front
    en16: np.ndarray  # undelayed, normalized for wav2vec2
    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    vi_frames: int
    en_frames: int


def prepare(vi_audio: bytes, en_audio: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: both clips at 24 kHz for Mimi, and the English at 16 kHz for CTC, normalized to zero
    mean and unit variance over the clip as wav2vec2's feature extractor does."""
    import soxr

    from training.prep.audio import decode

    vi, vi_rate = decode(vi_audio)
    en, en_rate = decode(en_audio)
    en16 = soxr.resample(en, en_rate, CTC_RATE).astype(np.float32)
    en16 = (en16 - en16.mean()) / np.sqrt(en16.var() + 1e-7)
    return soxr.resample(vi, vi_rate, SAMPLE_RATE), soxr.resample(en, en_rate, SAMPLE_RATE), en16


def target_delay_s(row_id: str, split: str, max_delay_s: float, seed: int) -> float:
    """Draw the English left-silence delay from U[0, max_delay_s], seeded per row."""
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
    vi24, en24, en16 = clips
    for side, pcm, stored in (("vi", vi24, row["vi_duration_s"]), ("en", en24, row["en_duration_s"])):
        # PyAV decodes a truncated or damaged clip without error, only shorter.
        seconds = len(pcm) / SAMPLE_RATE
        if abs(seconds - stored) > DURATION_TOLERANCE_S:
            raise Rejected(f"{side} audio decodes to {seconds:.2f} s, the row says {stored:.2f} s")
    vi_frames = math.ceil(len(vi24) / FRAME_SAMPLES)
    # Text EOS sits at en_frames, codec EOS at vi_frames.
    frames = max(math.ceil(len(en24) / FRAME_SAMPLES), vi_frames) + 1
    if frames > settings.max_frames:
        raise Rejected(f"{frames} frames exceed --max-frames {settings.max_frames}")
    if ctc_frames(len(en16)) < 1:
        raise Rejected("the English clip is shorter than one CTC frame")
    # The delay is drawn within the room the English clip leaves under the cap, so a long clip
    # gets a shorter delay instead of being rejected. vi_duration_s is float32 holding a multiple
    # of 0.01; round to the decimal it stores.
    room_s = ((settings.max_frames - 1) * FRAME_SAMPLES - len(en24)) / SAMPLE_RATE
    max_delay_s = min(settings.delay_ratio * round(row["vi_duration_s"], 2), room_s)
    delay_s = target_delay_s(row["id"], shard.split, max_delay_s, settings.seed)
    pad = int(round(delay_s * SAMPLE_RATE))
    lead, groups = sentencepiece_groups(row["text_en"], tokenizer)
    return Row(
        shard, index, row, vi24, np.concatenate([np.zeros(pad, np.float32), en24]), en16, lead, groups,
        delay_s, int(round(delay_s * FRAME_RATE)), vi_frames, math.ceil((pad + len(en24)) / FRAME_SAMPLES),
    )


_aligner: CTCWordAligner | None = None
_tokenizer: Any = None
_settings: Settings | None = None
_pcm_dir: Path | None = None


def init_worker(files: Path, min_score: float, tokenizer: Path | None = None, settings: Settings | None = None,
                pcm_dir: Path | None = None) -> None:
    """Worker initializer: the process's aligner, and for `prepare_row` the text tokenizer, the
    build's settings and where to leave the clips (`pcm_dir`; None returns them in the reply).
    `files`: the aligner snapshot, which the parent resolved."""
    global _aligner, _tokenizer, _settings, _pcm_dir
    _aligner = CTCWordAligner(files, min_score)
    if tokenizer is not None:
        import sentencepiece

        _tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
    _settings, _pcm_dir = settings, pcm_dir


@dataclass
class Planned:
    """A row's `plan` from a worker without its shard: the clips inline (`pcm`), or as one raw
    float32 file of vi24, en24 and en16 back to back (`file`, `lengths`) that `clips` maps."""

    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    vi_frames: int
    en_frames: int
    pcm: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    file: str | None = None
    lengths: tuple[int, int, int] = (0, 0, 0)

    def clips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three clips; a file is mapped and unlinked at once (the pages live as long as the
        arrays do)."""
        if self.pcm is not None:
            return self.pcm
        data = np.memmap(self.file, np.float32, "r")
        os.unlink(self.file)
        a, b = self.lengths[0], self.lengths[0] + self.lengths[1]
        return data[:a], data[a:b], data[b:]

    def row(self, shard: Shard, index: int, source: dict[str, Any]) -> Row:
        vi24, en24, en16 = self.clips()
        return Row(shard, index, source, vi24, en24, en16, self.lead, self.groups, self.delay_s,
                   self.delay_frames, self.vi_frames, self.en_frames)


def prepare_row(row: dict[str, Any], vi_audio: bytes, en_audio: bytes, split: str) -> Planned:
    """Worker: `prepare` then `plan` (the decode gates, the delay draw, the text grouping), so the
    main process only maps the clips: piping ~1.4 MB of PCM a row through the pool's one result
    pipe held the main thread ~30 s per GigaSpeech 2 shard. Raises Rejected or AlignmentError."""
    shard = Shard(None, "", split, 0, [], [])
    p = plan(shard, 0, row, prepare(vi_audio, en_audio), _settings, _tokenizer)
    out = Planned(p.lead, p.groups, p.delay_s, p.delay_frames, p.vi_frames, p.en_frames)
    clips = (p.vi24, p.en24, p.en16)
    if _pcm_dir is None:
        out.pcm = clips
        return out
    out.file = str(_pcm_dir / f"{uuid.uuid4().hex}.f32")
    with open(out.file, "wb") as f:
        for clip in clips:
            np.ascontiguousarray(clip, np.float32).tofile(f)
    out.lengths = tuple(len(c) for c in clips)
    return out


def align(
    log_probs: np.ndarray,
    lead: list[int],
    groups: list[tuple[list[int], str]],
    raw_target_frames: int,
    delay_frames: int,
    eos_id: int,
) -> tuple[list[int], list[int], float]:
    """Worker: the row's text tokens and their frames, and the alignment score; raises
    AlignmentError to reject the row."""
    alignment = _aligner.align(log_probs, groups)
    tokens, frames = timed_sentencepiece_tokens(lead, groups, alignment, raw_target_frames, delay_frames, eos_id)
    return tokens, frames, alignment.score


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


def sample(
    row: Row,
    tokens: list[int],
    text_frames: list[int],
    score: float,
    vi_codes: np.ndarray,
    en_codes: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """The row's cache sample: Python types and tensors only, since the trainer's `torch.load` keeps
    the `weights_only` default, which refuses NumPy scalars."""
    import torch

    codes = assemble_codes(row.source["id"], vi_codes, en_codes, tokens, text_frames, cfg)
    return {
        "id": row.source["id"],
        "split": row.shard.split,
        "codes": torch.from_numpy(codes),
        "frames": int(codes.shape[1]),
        "vi_frames": row.vi_frames,
        "en_frames": row.en_frames,
        "text_tokens": len(tokens),
        "target_delay_s": row.delay_s,
        "target_delay_frames": row.delay_frames,
        "text_en": row.source["text_en"],
        **row.source["fields"],
        "text_timing": ALIGNMENT_NAME,
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
        "alignment_min_score": settings.min_alignment_score,
        "max_frames": settings.max_frames,
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
