"""GRPO's process rewards on rollout text (Hibiki-Zero, arXiv 2602.11072v2, §3.3).

A rollout's reward at frame t is (1 - α)·BLEU(ŷ_t, y_t) + α·BLEU(ŷ_T, y_T):

- ŷ_t is the text it emitted up to frame t.
- y_t joins the references of the sentences whose Vietnamese has started by t, the one being
  spoken included.
- T is the rollout's last frame.

Each reward frame's rewards are normalized over the rollouts of one stream (its group). A token
at frame t gets the sum of the normalized rewards at the reward frames after t.

The paper puts reward frames at the end of every 8 input words. The cache has no Vietnamese
words, so they sit every `REWARD_SPEECH_FRAMES` frames of Vietnamese speech, plus the last frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sacrebleu

# One reward frame per 2.5 s of Vietnamese speech, about 8 syllables at the TTS's pace (n_w = 8).
REWARD_SPEECH_FRAMES = 31
ALPHA = 0.4  # the final BLEU's share of each reward (the paper's α)


@dataclass(frozen=True)
class Sentence:
    """One cache row of a stream: where its Vietnamese runs in the stream's frames, and its
    English reference."""

    start: int
    end: int  # exclusive
    reference: str


def reward_frames(sentences: list[Sentence], frames: int, every: int = REWARD_SPEECH_FRAMES) -> np.ndarray:
    """The frames that close each `every` frames of Vietnamese speech, counted inside sentences,
    and the last frame, in order."""
    speech = np.zeros(frames, dtype=bool)
    for sentence in sentences:
        speech[sentence.start : min(sentence.end, frames)] = True
    spoken = np.cumsum(speech)
    marks = np.flatnonzero(speech & (spoken % every == 0))
    return np.unique(np.append(marks, frames - 1))


def prefix_references(sentences: list[Sentence], frames: np.ndarray) -> list[str]:
    """y_t at each frame: the references of the sentences started by then, in order."""
    return [
        " ".join(s.reference for s in sentences if s.start <= frame) for frame in frames.tolist()
    ]


def emitted_texts(
    tokens: np.ndarray, frames: np.ndarray, skip: set[int], decode
) -> list[str]:
    """ŷ_t at each frame: `decode` of a rollout's text tokens up to the frame, `skip` (PAD, EOS)
    left out. `tokens` is [T]."""
    keep = ~np.isin(tokens, list(skip))
    return [decode(tokens[: frame + 1][keep[: frame + 1]].tolist()) for frame in frames.tolist()]


def bleu(hypothesis: str, reference: str) -> float:
    """Sentence BLEU (0-100); nothing to translate yet scores 0."""
    if not reference:
        return 0.0
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


def process_rewards(hypotheses: list[list[str]], references: list[str], alpha: float = ALPHA) -> np.ndarray:
    """Rewards [G, J] of G rollouts at J reward frames. `hypotheses[i][j]` is rollout i's text
    at frame j and `references[j]` y at that frame; the last frame is T."""
    final = np.array([bleu(texts[-1], references[-1]) for texts in hypotheses])
    partial = np.array(
        [[bleu(text, reference) for text, reference in zip(texts, references, strict=True)]
         for texts in hypotheses]
    )
    return (1 - alpha) * partial + alpha * final[:, None]


def normalize(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Each reward frame's rewards over the group: (r - mean) / (std + eps). A frame where every
    rollout scored the same gets 0."""
    mean = rewards.mean(axis=0, keepdims=True)
    std = rewards.std(axis=0, ddof=1, keepdims=True)
    return (rewards - mean) / (std + eps)


def advantages(normalized: np.ndarray, frames: np.ndarray, length: int) -> np.ndarray:
    """[G, length]: a token at frame t gets the sum of the normalized rewards at reward frames
    after t (none after the last)."""
    suffix = np.concatenate(
        [np.cumsum(normalized[:, ::-1], axis=1)[:, ::-1], np.zeros((len(normalized), 1))], axis=1
    )
    first_after = np.searchsorted(frames, np.arange(length), side="right")
    return suffix[:, first_after]


def group_advantages(
    tokens: np.ndarray,
    sentences: list[Sentence],
    skip: set[int],
    decode,
    alpha: float = ALPHA,
    every: int = REWARD_SPEECH_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One stream's G rollouts, text tokens [G, T]: their reward frames [J], rewards [G, J] and
    per-frame advantages [G, T]."""
    length = tokens.shape[1]
    frames = reward_frames(sentences, length, every)
    references = prefix_references(sentences, frames)
    hypotheses = [emitted_texts(row, frames, skip, decode) for row in tokens]
    rewards = process_rewards(hypotheses, references, alpha)
    return frames, rewards, advantages(normalize(rewards), frames, length)
