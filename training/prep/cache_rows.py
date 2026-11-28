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
import random
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from training.paths import CACHE_FORMAT, DEFAULT_DIRECTION, durable, exit_with_parent
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
    target24: np.ndarray  # raw 24 kHz clip; delay silence (`delay_samples`) applied at staging
    target16: np.ndarray  # undelayed, normalized for wav2vec2
    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    source_frames: int
    target_frames: int
    delay_samples: int = 0


def prepare(source_audio: bytes, target_audio: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: both clips at 24 kHz for Mimi, and the target at 16 kHz for CTC, normalized to zero
    mean and unit variance over the clip as wav2vec2's feature extractor does."""
    import soxr

    from training.prep.audio import decode

    source, source_rate = decode(source_audio)
    target, target_rate = decode(target_audio)
    target16 = soxr.resample(target, target_rate, CTC_RATE).astype(np.float32)
    target16 = (target16 - target16.mean()) / np.sqrt(target16.var() + 1e-7)
    return (
        soxr.resample(source, source_rate, SAMPLE_RATE).astype(np.float32, copy=False),
        soxr.resample(target, target_rate, SAMPLE_RATE).astype(np.float32, copy=False),
        target16,
    )


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
        shard, index, row, source24, target24, target16,
        lead, groups, delay_s, int(round(delay_s * FRAME_RATE)), source_frames,
        math.ceil((pad + len(target24)) / FRAME_SAMPLES), pad,
    )


_aligner: CTCWordAligner | None = None
_tokenizer: Any = None
_settings: Settings | None = None
_pcm_dir: Path | None = None
# Dual-direction worker state (`init_dual_worker`)
_dual_aligners: dict[str, CTCWordAligner] = {}
_dual_tokenizers: dict[str, Any] = {}
_dual_settings: dict[str, Settings] = {}


def init_worker(settings: Settings, tokenizer: Path, pcm_dir: Path | None) -> None:
    """Worker initializer: the process's aligner (the build's target language's), the text
    tokenizer, the build's settings and where to leave the clips (`pcm_dir`; None returns them in
    the reply)."""
    import sentencepiece

    exit_with_parent()
    global _aligner, _tokenizer, _settings, _pcm_dir
    _aligner = CTCWordAligner(settings.aligner, settings.min_score)
    _tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
    _settings, _pcm_dir = settings, pcm_dir


def init_dual_worker(
    settings_ve: Settings,
    tokenizer_ve: Path,
    settings_ev: Settings,
    tokenizer_ev: Path,
    pcm_dir: Path | None,
) -> None:
    """Dual-direction worker initializer: holds both EN and VI aligners and tokenizers."""
    import sentencepiece

    exit_with_parent()
    global _pcm_dir, _dual_aligners, _dual_tokenizers, _dual_settings
    _pcm_dir = pcm_dir
    _dual_settings = {"vi-en": settings_ve, "en-vi": settings_ev}
    _dual_aligners = {
        "vi-en": CTCWordAligner(settings_ve.aligner, settings_ve.min_score),
        "en-vi": CTCWordAligner(settings_ev.aligner, settings_ev.min_score),
    }
    _dual_tokenizers = {
        "vi-en": sentencepiece.SentencePieceProcessor(str(tokenizer_ve)),
        "en-vi": sentencepiece.SentencePieceProcessor(str(tokenizer_ev)),
    }


@dataclass
class Planned:
    """A row's `plan` from a worker without its shard: the clips inline (`pcm`), or as slices
    into a per-job float32 slab (`offset`, `lengths`)."""

    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    source_frames: int
    target_frames: int
    delay_samples: int = 0
    pcm: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    offset: int = 0
    lengths: tuple[int, int, int] = (0, 0, 0)

    def clips(self, slab: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.pcm is not None:
            return self.pcm
        assert slab is not None
        o = self.offset
        a = o + self.lengths[0]
        b = a + self.lengths[1]
        c = b + self.lengths[2]
        return slab[o:a], slab[a:b], slab[b:c]

    def row(self, shard: Shard, index: int, data: dict[str, Any], slab: np.ndarray | None = None) -> Row:
        source24, target24, target16 = self.clips(slab)
        return Row(
            shard, index, data, source24, target24, target16, self.lead, self.groups, self.delay_s,
            self.delay_frames, self.source_frames, self.target_frames, self.delay_samples,
        )


def prepare_rows(
    rows: list[tuple[dict[str, Any], bytes, bytes]], split: str
) -> tuple[str | None, list[Planned | Rejected | AlignmentError]]:
    """Worker: decode and plan a job of rows (`row`, source audio, target audio). Writes all kept
    clips of the job into ONE contiguous `/dev/shm` slab file (32x fewer open/write/unlink
    syscalls than per-row files)."""
    out: list[Planned | Rejected | AlignmentError] = []
    chunks: list[np.ndarray] = []
    offset = 0
    shard = Shard(None, "", split, 0, [], [])
    for row, source_audio, target_audio in rows:
        try:
            p = plan(shard, 0, row, prepare(source_audio, target_audio), _settings, _tokenizer)
        except (Rejected, AlignmentError) as exc:
            exc.__traceback__ = exc.__context__ = None
            out.append(exc)
            continue
        planned = Planned(
            p.lead, p.groups, p.delay_s, p.delay_frames, p.source_frames, p.target_frames, p.delay_samples
        )
        if _pcm_dir is None:
            planned.pcm = (p.source24, p.target24, p.target16)
        else:
            s24 = np.ascontiguousarray(p.source24, np.float32)
            t24 = np.ascontiguousarray(p.target24, np.float32)
            t16 = np.ascontiguousarray(p.target16, np.float32)
            chunks += [s24, t24, t16]
            planned.offset = offset
            planned.lengths = (len(s24), len(t24), len(t16))
            offset += len(s24) + len(t24) + len(t16)
        out.append(planned)
    if _pcm_dir is None or not chunks:
        return None, out
    slab_path = str(_pcm_dir / f"{uuid.uuid4().hex}.f32")
    with open(slab_path, "wb") as f:
        for chunk in chunks:
            chunk.tofile(f)
    return slab_path, out


@dataclass
class DualPlanned:
    """One row decoded once and planned for both `vi-en` and `en-vi` (either side may hold an
    AlignmentError if only that language's text failed grouping)."""

    vi_en: Planned | Rejected | AlignmentError
    en_vi: Planned | Rejected | AlignmentError
    offset: int = 0
    lengths: tuple[int, int, int, int] = (0, 0, 0, 0)  # (len(vi24), len(en24), len(vi16), len(en16))
    pcm: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None = None


def prepare_dual_rows(
    rows: list[tuple[dict[str, Any], bytes, bytes]], split: str
) -> tuple[str | None, list[DualPlanned | Rejected]]:
    """Dual-direction worker: decode `(vi_audio, en_audio)` ONCE per row, resample to 24 kHz and
    to normalized 16 kHz via `soxr`, then plan both `vi-en` and `en-vi` into a single `/dev/shm`
    slab."""
    import soxr

    from training.prep.audio import decode

    out: list[DualPlanned | Rejected] = []
    chunks: list[np.ndarray] = []
    offset = 0
    shard = Shard(None, "", split, 0, [], [])
    set_ve, set_ev = _dual_settings["vi-en"], _dual_settings["en-vi"]
    tok_ve, tok_ev = _dual_tokenizers["vi-en"], _dual_tokenizers["en-vi"]
    for row, vi_audio, en_audio in rows:
        try:
            vi_raw, vi_rate = decode(vi_audio)
            en_raw, en_rate = decode(en_audio)
            vi24 = soxr.resample(vi_raw, vi_rate, SAMPLE_RATE).astype(np.float32, copy=False)
            en24 = soxr.resample(en_raw, en_rate, SAMPLE_RATE).astype(np.float32, copy=False)
            vi16 = soxr.resample(vi_raw, vi_rate, CTC_RATE).astype(np.float32)
            vi16 = (vi16 - vi16.mean()) / np.sqrt(vi16.var() + 1e-7)
            en16 = soxr.resample(en_raw, en_rate, CTC_RATE).astype(np.float32)
            en16 = (en16 - en16.mean()) / np.sqrt(en16.var() + 1e-7)
        except Rejected as exc:
            exc.__traceback__ = exc.__context__ = None
            out.append(exc)
            continue

        def _plan_one(
            src24: np.ndarray, tgt24: np.ndarray, tgt16: np.ndarray, settings: Settings, tok: Any
        ) -> Planned | Rejected | AlignmentError:
            try:
                p = plan(shard, 0, row, (src24, tgt24, tgt16), settings, tok)
            except (Rejected, AlignmentError) as exc:
                exc.__traceback__ = exc.__context__ = None
                return exc
            return Planned(
                p.lead, p.groups, p.delay_s, p.delay_frames, p.source_frames, p.target_frames, p.delay_samples
            )

        p_ve = _plan_one(vi24, en24, en16, set_ve, tok_ve)
        p_ev = _plan_one(en24, vi24, vi16, set_ev, tok_ev)
        dp = DualPlanned(p_ve, p_ev)
        keep_ve = isinstance(p_ve, Planned)
        keep_ev = isinstance(p_ev, Planned)
        if keep_ve or keep_ev:
            if _pcm_dir is None:
                dp.pcm = (vi24, en24, vi16 if keep_ev else None, en16 if keep_ve else None)
            else:
                v = np.ascontiguousarray(vi24, np.float32)
                e = np.ascontiguousarray(en24, np.float32)
                chunks.append(v)
                chunks.append(e)
                v16_len = 0
                if keep_ev:
                    v16 = np.ascontiguousarray(vi16, np.float32)
                    chunks.append(v16)
                    v16_len = len(v16)
                e16_len = 0
                if keep_ve:
                    e16 = np.ascontiguousarray(en16, np.float32)
                    chunks.append(e16)
                    e16_len = len(e16)
                dp.offset = offset
                dp.lengths = (len(v), len(e), v16_len, e16_len)
                offset += len(v) + len(e) + v16_len + e16_len
        out.append(dp)
    if _pcm_dir is None or not chunks:
        return None, out
    slab_path = str(_pcm_dir / f"{uuid.uuid4().hex}.f32")
    with open(slab_path, "wb") as f:
        for chunk in chunks:
            chunk.tofile(f)
    return slab_path, out


def check_tail(tail_frames: int, aligner: Aligner) -> None:
    """Reject a row whose target runs on `tail_frames` past its last text token (to its text EOS)
    for longer than its aligner allows (`Aligner.max_tail_s`)."""
    if aligner.max_tail_s is not None and tail_frames > round(aligner.max_tail_s * FRAME_RATE):
        raise AlignmentError(f"Speech runs on {tail_frames / FRAME_RATE:.2f} s past the last word "
                             f"(at most {aligner.max_tail_s} s)")


def align_rows(
    log_probs: list[np.ndarray],
    rows: list[tuple[list[int], list[tuple[list[int], str]], int, int]],
    eos_id: int,
    direction: str | None = None,
) -> list[tuple[list[int], list[int], float] | AlignmentError]:
    """Worker: for each row of a CTC batch (its log-probabilities, and its `lead`, `groups`, raw
    target frames and delay frames), its text tokens, their frames and the alignment score, or the
    AlignmentError that rejects it."""
    aligner = _dual_aligners[direction] if direction is not None else _aligner
    settings = _dual_settings[direction] if direction is not None else _settings
    out: list[tuple[list[int], list[int], float] | AlignmentError] = []
    for probs, (lead, groups, raw_target_frames, delay_frames) in zip(log_probs, rows, strict=True):
        try:
            alignment = aligner.align(probs, groups)
            tokens, frames = timed_sentencepiece_tokens(lead, groups, alignment, raw_target_frames, delay_frames,
                                                        eos_id)
            check_tail(frames[-1] - frames[-2], settings.aligner)
        except AlignmentError as exc:
            exc.__traceback__ = exc.__context__ = None
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
        "max_tail_s": settings.aligner.max_tail_s,
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
