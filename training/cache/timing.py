"""Word-time a row's target text for the grounded-v2 training cache: English against English
audio, Vietnamese against Vietnamese audio, each with its language's CTC model (`ALIGNERS`)."""

from __future__ import annotations

import ctypes
import functools
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@functools.cache
def _viterbi_lib() -> Any:
    """`_ctc_viterbi.c`, compiled once per platform into `__pycache__` beside it and loaded; raises
    when it cannot be built or loaded."""
    src = Path(__file__).with_name("_ctc_viterbi.c")
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    cache_dir = src.parent / "__pycache__"
    tag = f"{platform.system().lower()}_{platform.machine().lower()}"
    so_path = cache_dir / f"_ctc_viterbi_{tag}{ext}"
    if not so_path.is_file() or so_path.stat().st_mtime < src.stat().st_mtime:
        cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            raise RuntimeError(f"no C compiler (cc, gcc or clang) to build {src.name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".viterbi_", suffix=ext, dir=cache_dir)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            subprocess.run([cc, "-O3", "-fPIC", "-shared", str(src), "-o", str(tmp_path), "-lm"],
                           check=True, stdout=subprocess.DEVNULL)
            tmp_path.replace(so_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(so_path))
    lib.ctc_viterbi_align.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ctc_viterbi_align.restype = ctypes.c_int
    return lib


# wav2vec2's feature encoder, (kernel, stride) per convolution: 16 kHz -> 50 Hz (base and large).
CTC_CONVS = ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2))


def ctc_frames(samples: int) -> int:
    """The CTC frames of a `samples`-long 16 kHz clip."""
    for kernel, stride in CTC_CONVS:
        samples = (samples - kernel) // stride + 1
    return samples


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _spoken_english(text: str) -> str:
    """One SentencePiece word group as wav2vec2-base-960h spells it: ASCII capitals, numbers in words."""
    from num2words import num2words

    text = _ascii(text)
    text = re.sub(
        r"\d+",
        lambda match: str(num2words(int(match.group(0)))).replace("-", " "),
        text,
    )
    return " ".join(re.sub(r"[^A-Za-z' ]+", " ", text).upper().split())


def _spoken_vietnamese(text: str) -> str:
    """One SentencePiece word group as the Vietnamese aligner spells it: lowercase NFC letters,
    numbers in Vietnamese words. A letter outside `_VIETNAMESE` rejects the row at alignment."""
    from num2words import num2words

    text = re.sub(r"\d+", lambda match: f" {num2words(int(match.group(0)), lang='vi')} ", text)
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(char if char.isalpha() else " " for char in text).split())


# Vietnamese letters, NFC: the twelve vowels in each of the six tones, the consonants and the four
# letters of loanwords (f, j, w, z).
_VIETNAMESE = "bcdđfghjklmnpqrstvwxz" + "".join(
    unicodedata.normalize("NFC", vowel + tone)
    for vowel in "aăâeêioôơuưy"
    for tone in ("", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323")
)


@dataclass(frozen=True)
class Aligner:
    """The CTC model that word-times one language's text; every machine times with `revision`."""

    name: str  # a sample's `text_timing`
    repo: str
    revision: str
    blank: str  # the vocabulary's CTC blank
    delimiter: str  # its token between words
    spoken: Callable[[str], str]  # a word group in the model's alphabet
    # The letters `spoken` may write: the Viterbi reads only their columns, the blank's and the
    # delimiter's (`vocab_columns`), and a row with any other letter is rejected.
    alphabet: str
    # The lowest mean posterior of the aligned characters a row may have, calibrated per model:
    # each model's scores are its own (docs/agent-memory/two-way.md#vietnamese-aligner).
    min_score: float
    # The longest a clip may run on past its last text token, or None. VieNeu sometimes keeps
    # speaking past its text, and the alignment score does not see it; English recordings end on
    # silence that token times cannot tell from speech (docs/agent-memory/cache-build.md#rows).
    max_tail_s: float | None


ALIGNERS = {  # by the language of the text
    "en": Aligner("wav2vec2_ctc_word_v1", "facebook/wav2vec2-base-960h",
                  "22aad52d435eb6dbaf354bdad9b0da84ce7d6156", "<pad>", "|", _spoken_english,
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ'", 0.5, None),
    # wav2vec2-base fine-tuned on 250 h of Vietnamese (non-commercial: its card says CC-BY-NC-4.0, the
    # repo also ships a CC-BY-NC-SA-4.0 text), at the Hub bot's safetensors
    # conversion (refs/pr/2), tensor for tensor its main branch's pytorch_model.bin. It replaced
    # omniASR-CTC-300M-v2 (Apache-2.0; 325M parameters against 95M) for the Mac's build speed
    # (user, 2026-11-22).
    "vi": Aligner("wav2vec2_vi250h_ctc_word_v1", "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
                  "bb07d01e57ce84fa73c3e4dd423b663ec6b36753", "<pad>", "|", _spoken_vietnamese,
                  _VIETNAMESE, 0.55, 1.5),
}


def vocab_columns(aligner: Aligner, files: Path) -> list[int]:
    """The vocabulary ids of the blank, the delimiter and the alphabet, in that order: the columns
    of the log-probabilities a device hands to `CTCWordAligner`, which reads no others."""
    import json

    vocab = json.loads((files / "vocab.json").read_text())
    tokens = [aligner.blank, aligner.delimiter, *aligner.alphabet]
    missing = [token for token in tokens if token not in vocab]
    if missing:
        raise ValueError(f"{aligner.repo} has no vocabulary entry for {missing}")
    return [int(vocab[token]) for token in tokens]


def sentencepiece_groups(
    text: str, tokenizer: Any, spoken: Callable[[str], str]
) -> tuple[list[int], list[tuple[list[int], str]]]:
    """SentencePiece ids grouped into spoken words for CTC alignment (`spoken`: the aligner's),
    after the ids of the unspoken lead (a stray "*", "-" or quote before the first word), which
    may be empty."""
    pieces = list(tokenizer.encode(text, out_type=str))
    if not pieces:
        raise AlignmentError("Cannot align empty text")
    groups: list[list[str]] = []
    for piece in pieces:
        if piece.startswith("▁") or not groups:
            groups.append([piece])
        else:
            groups[-1].append(piece)
    # A group with nothing to speak joins the word before it, or the lead before the first word.
    lead: list[int] = []
    result: list[tuple[list[int], str]] = []
    for group in groups:
        ids = [int(tokenizer.piece_to_id(piece)) for piece in group]
        word = spoken(tokenizer.decode(ids))
        if not word:
            (result[-1][0] if result else lead).extend(ids)
            continue
        result.append((ids, word))
    if not result:
        raise AlignmentError(f"No speakable text in {text!r}")
    return lead, result


class AlignmentError(ValueError):
    """The text cannot be timed against its audio: a data fault that rejects the row."""


# The rejections for the frame grids, not for speech and text disagreeing: more characters than
# CTC frames, or more tokens than 80 ms frames (fast or dense speech). The other direction's row
# of the same clips still trains (`streams.load_rows`). Shards record a rejection with its message
# ("AlignmentError: <message>"), and packs look for these in it when they are built: rewording one needs a new PACK_VERSION and
# keeps the old wording matched for the shards already built.
EMISSION_TOO_SHORT = "CTC emission is too short"
TEXT_OVER_FRAMES = "Aligned text needs"
CAPACITY_REASONS = (EMISSION_TOO_SHORT, TEXT_OVER_FRAMES)


@dataclass(frozen=True)
class Alignment:
    spans: list[tuple[float, float]]
    score: float


class CTCWordAligner:
    """An aligner's CTC word timing on the CPU: the Viterbi over a clip's log-probabilities in
    `vocab_columns` order, whichever device computed them (`mlx` or `cuda`)."""

    blank_id = 0  # the columns' order
    delimiter_id = 1

    def __init__(self, aligner: Aligner, min_score: float):
        if not 0 <= min_score <= 1:
            raise ValueError("Minimum CTC alignment score must be in [0, 1]")
        self.min_score = min_score
        self.letters = {letter: column for column, letter in enumerate(aligner.alphabet, start=2)}

    def _targets(
        self, groups: list[tuple[list[int], str]]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        target_ids: list[int] = []
        ranges: list[tuple[int, int]] = []
        for group_index, (_, spoken) in enumerate(groups):
            if group_index:
                target_ids.append(self.delimiter_id)
            start = len(target_ids)
            for char in spoken:
                if char == " ":
                    target_ids.append(self.delimiter_id)
                elif char not in self.letters:
                    raise AlignmentError(f"CTC transcript contains unsupported character: {char!r}")
                else:
                    target_ids.append(self.letters[char])
            ranges.append((start, len(target_ids)))
        return target_ids, ranges

    def align(self, log_probs: np.ndarray, groups: list[tuple[list[int], str]]) -> Alignment:
        """Word spans of `groups` as fractions of the clip, from its CTC log-probabilities; raises
        AlignmentError when the text cannot be aligned or the score is below the minimum."""
        targets, ranges = self._targets(groups)
        if not targets:
            raise AlignmentError("CTC target is empty")
        time_steps = int(log_probs.shape[0])
        needed = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
        if time_steps < needed:
            raise AlignmentError(f"{EMISSION_TOO_SHORT}: {time_steps} < {needed}")

        lib = _viterbi_lib()
        lp = np.ascontiguousarray(log_probs, dtype=np.float32)
        tgt = np.ascontiguousarray(targets, dtype=np.int32)
        starts = np.empty(len(targets), dtype=np.int32)
        ends = np.empty(len(targets), dtype=np.int32)
        score_c = ctypes.c_double(0.0)
        rc = lib.ctc_viterbi_align(
            lp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            time_steps,
            int(lp.shape[1]),
            tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(targets),
            self.blank_id,
            starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(score_c),
        )
        if rc == -2:
            raise AlignmentError("CTC alignment omitted a transcript token")
        if rc != 0:
            raise MemoryError("ctc_viterbi_align allocation failed")
        score = float(score_c.value)
        if score < self.min_score:
            raise AlignmentError(f"CTC alignment score {score:.3f} is below {self.min_score:.3f}")
        spans = [
            (int(starts[left:right].min()) / time_steps, (int(ends[left:right].max()) + 1) / time_steps)
            for left, right in ranges
        ]
        if any(
            not (0 <= left < right <= 1) or (index and left < spans[index - 1][0])
            for index, (left, right) in enumerate(spans)
        ):
            raise AlignmentError("CTC alignment produced invalid word spans")
        return Alignment(spans=spans, score=score)


def aligner_files(aligner: Aligner) -> Path:
    """The aligner's pinned snapshot: its weights, configuration and vocabulary."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            aligner.repo,
            revision=aligner.revision,
            allow_patterns=["model.safetensors", "config.json", "vocab.json"],
        )
    )


def timed_sentencepiece_tokens(
    lead: list[int],
    groups: list[tuple[list[int], str]],
    alignment: Alignment,
    raw_target_frames: int,
    delay_frames: int,
    eos_id: int,
) -> tuple[list[int], list[int]]:
    """Map word-aligned SentencePiece groups onto unique 12.5 Hz frames; the unspoken lead takes
    the frames just before the first word, from the delay at the earliest."""
    if len(groups) != len(alignment.spans):
        raise ValueError("Text groups and alignment spans differ")
    tokens: list[int] = []
    frames: list[int] = []
    previous = delay_frames - 1
    if lead:
        first = delay_frames + int(math.floor(alignment.spans[0][0] * raw_target_frames))
        for frame, token in enumerate(lead, start=max(delay_frames, first - len(lead))):
            tokens.append(token)
            frames.append(frame)
            previous = frame
    for (ids, _), (start_ratio, end_ratio) in zip(groups, alignment.spans, strict=True):
        start = delay_frames + int(math.floor(start_ratio * raw_target_frames))
        end = delay_frames + max(
            start - delay_frames, int(math.ceil(end_ratio * raw_target_frames)) - 1
        )
        if len(ids) == 1:
            candidates = [start]
        else:
            candidates = [
                round(start + index * max(0, end - start) / (len(ids) - 1))
                for index in range(len(ids))
            ]
        for token, frame in zip(ids, candidates, strict=True):
            frame = max(frame, previous + 1)
            tokens.append(token)
            frames.append(frame)
            previous = frame
    eos_frame = delay_frames + raw_target_frames
    if previous >= eos_frame:
        raise AlignmentError(
            f"{TEXT_OVER_FRAMES} {previous - delay_frames + 1} frames but target has "
            f"{raw_target_frames}"
        )
    tokens.append(int(eos_id))
    frames.append(eos_frame)
    return tokens, frames
