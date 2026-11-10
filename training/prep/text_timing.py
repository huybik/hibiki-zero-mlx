"""Word-time a row's target text for the grounded-v2 training cache: English against English
audio, Vietnamese against Vietnamese audio, each with its language's CTC model (`ALIGNERS`)."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    """One SentencePiece word group as omniASR spells it: lowercase NFC letters, numbers in
    Vietnamese words. A letter its vocabulary lacks rejects the row at alignment."""
    from num2words import num2words

    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\d+", lambda match: f" {num2words(int(match.group(0)), lang='vi')} ", text)
    return " ".join("".join(char if char.isalpha() else " " for char in text).split())


@dataclass(frozen=True)
class Aligner:
    """The CTC model that word-times one language's text; every machine times with `revision`."""

    name: str  # a sample's `text_timing`
    repo: str
    revision: str
    blank: str  # the vocabulary's CTC blank
    delimiter: str  # its token between words
    spoken: Callable[[str], str]  # a word group in the model's alphabet
    # The lowest mean posterior of the aligned characters a row may have. The scores are each
    # model's own: omniASR spreads its mass over 10,288 characters of every script.
    min_score: float


ALIGNERS = {  # by the language of the text
    "en": Aligner("wav2vec2_ctc_word_v1", "facebook/wav2vec2-base-960h",
                  "22aad52d435eb6dbaf354bdad9b0da84ce7d6156", "<pad>", "|", _spoken_english, 0.5),
    # A stock-transformers conversion of Meta's omniASR-CTC-300M-v2 (Apache-2.0).
    "vi": Aligner("omniasr_ctc_word_v1", "aadel4/omniASR-CTC-300M-v2",
                  "f2b3c39a62c061c27623c3f2d803294fbb954324", "<s>", " ", _spoken_vietnamese, 0.5),
}


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


@dataclass(frozen=True)
class Alignment:
    spans: list[tuple[float, float]]
    score: float


def _ctc_token_frames(
    log_probs: np.ndarray, targets: list[int], blank_id: int
) -> tuple[list[list[int]], float]:
    """Viterbi-align one CTC log-probability matrix [T, vocab] to a known token sequence.

    On ties a state keeps the first candidate (stay, advance, skip), and the path ends on the last
    token unless the trailing blank scores higher.
    """
    if not targets:
        raise AlignmentError("CTC target is empty")
    time_steps = int(log_probs.shape[0])
    states = 2 * len(targets) + 1
    if time_steps < states:
        raise AlignmentError(f"CTC emission is too short: {time_steps} < {states}")

    symbols = np.full(states, blank_id, dtype=np.int64)
    symbols[1::2] = targets
    can_skip = np.zeros(states, dtype=bool)
    can_skip[3::2] = symbols[3::2] != symbols[1:-2:2]
    emissions = log_probs[:, symbols]
    score = np.full(states, -np.inf, dtype=np.float32)
    score[:2] = emissions[0, :2]
    back = np.zeros((time_steps, states), dtype=np.int8)
    candidates = np.full((3, states), -np.inf, dtype=np.float32)
    every_state = np.arange(states)
    for time_index in range(1, time_steps):
        candidates[0] = score
        candidates[1, 1:] = score[:-1]
        candidates[2, 2:] = np.where(can_skip[2:], score[:-2], -np.inf)
        choice = candidates.argmax(axis=0)
        score = candidates[choice, every_state] + emissions[time_index]
        back[time_index] = choice

    state = states - 1 if score[-1] >= score[-2] else states - 2
    frames: list[list[int]] = [[] for _ in targets]
    times: list[int] = []
    tokens: list[int] = []
    for time_index in range(time_steps - 1, -1, -1):
        if state % 2:
            frames[state // 2].append(time_index)
            times.append(time_index)
            tokens.append(targets[state // 2])
        state -= int(back[time_index, state])
    if any(not item for item in frames):
        raise AlignmentError("CTC alignment omitted a transcript token")
    for item in frames:
        item.reverse()
    probabilities = np.exp(log_probs[times, tokens]).astype(np.float64)
    return frames, float(probabilities.sum() / len(probabilities))


class CTCWordAligner:
    """An aligner's CTC word timing on the CPU: the Viterbi over a clip's log-probabilities,
    whichever device computed them (`cache_build`: MLX or CUDA)."""

    def __init__(self, aligner: Aligner, files: Path, min_score: float):
        """`files`: the pinned snapshot, from `aligner_files`."""
        import json

        if not 0 <= min_score <= 1:
            raise ValueError("Minimum CTC alignment score must be in [0, 1]")
        self.min_score = min_score
        self.vocab = json.loads((files / "vocab.json").read_text())
        self.blank_id = int(self.vocab[aligner.blank])
        self.delimiter_id = int(self.vocab[aligner.delimiter])

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
                elif char not in self.vocab:
                    raise AlignmentError(f"CTC transcript contains unsupported character: {char!r}")
                else:
                    target_ids.append(int(self.vocab[char]))
            ranges.append((start, len(target_ids)))
        return target_ids, ranges

    def align(self, log_probs: np.ndarray, groups: list[tuple[list[int], str]]) -> Alignment:
        """Word spans of `groups` as fractions of the clip, from its CTC log-probabilities; raises
        AlignmentError when the text cannot be aligned or the score is below the minimum."""
        targets, ranges = self._targets(groups)
        token_frames, score = _ctc_token_frames(log_probs, targets, self.blank_id)
        if score < self.min_score:
            raise AlignmentError(f"CTC alignment score {score:.3f} is below {self.min_score:.3f}")
        denom = int(log_probs.shape[0])
        spans = []
        for left, right in ranges:
            frames = [frame for item in token_frames[left:right] for frame in item]
            spans.append((min(frames) / denom, (max(frames) + 1) / denom))
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
            f"Aligned text needs {previous - delay_frames + 1} frames but target has "
            f"{raw_target_frames}"
        )
    tokens.append(int(eos_id))
    frames.append(eos_frame)
    return tokens, frames
