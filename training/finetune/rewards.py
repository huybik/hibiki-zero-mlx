"""GRPO's process rewards on rollout text (Hibiki-Zero, arXiv 2602.11072v2, §3.3).

A rollout's reward at a frame boundary b is (1 - α)·BLEU(ŷ_b, y_b) + α·BLEU(ŷ_T, y_T):

- ŷ_b is the text it emitted in the frames before b, decoded as evaluation decodes it (words
  only, up to the first EOS).
- y_b joins the references of the sentences whose Vietnamese started before b, the one being
  spoken included.
- T is the rollout's length.

Each boundary's rewards are normalized over the rollouts of one stream (its group). A token at
frame t gets the sum of the normalized rewards at the boundaries after t. The tokens a reward
counts are therefore exactly the ones it credits.

The paper puts reward frames at the end of every 8 input words. The cache has no Vietnamese
words, so the boundaries close every `REWARD_SPEECH_FRAMES` frames of Vietnamese speech, plus
the end, T.
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
    """Reward boundaries, in order: after each `every`-th frame of Vietnamese speech, counted
    inside sentences, and at the end, `frames`."""
    speech = np.zeros(frames, dtype=bool)
    for sentence in sentences:
        speech[sentence.start : min(sentence.end, frames)] = True
    spoken = np.cumsum(speech)
    marks = np.flatnonzero(speech & (spoken % every == 0)) + 1
    return np.unique(np.append(marks, frames))


def prefix_references(sentences: list[Sentence], boundaries: np.ndarray) -> list[str]:
    """y at each boundary: the references of the sentences started before it, in order."""
    return [
        " ".join(s.reference for s in sentences if s.start < boundary)
        for boundary in boundaries.tolist()
    ]


def emitted_texts(
    tokens: np.ndarray, boundaries: np.ndarray, eos_id: int, pad_id: int, decode
) -> list[str]:
    """ŷ at each boundary: the words of a rollout's text tokens [T] before it, up to the first
    EOS, decoded as evaluation decodes (`common.decode_text_batch`: tokens above PAD)."""
    eos = np.flatnonzero(tokens == eos_id)
    end = int(eos[0]) if len(eos) else len(tokens)
    words = tokens > pad_id
    texts = []
    for boundary in boundaries.tolist():
        stop = min(boundary, end)
        texts.append(decode(tokens[:stop][words[:stop]].tolist()))
    return texts


def bleu(hypothesis: str, reference: str) -> float:
    """Sentence BLEU (0-100); nothing to translate yet scores 0."""
    if not reference:
        return 0.0
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


def process_rewards(hypotheses: list[list[str]], references: list[str], alpha: float = ALPHA) -> np.ndarray:
    """Rewards [G, J] of G rollouts at J reward boundaries. `hypotheses[i][j]` is rollout i's
    text at boundary j and `references[j]` y there; the last boundary is T."""
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


def advantages(normalized: np.ndarray, boundaries: np.ndarray, length: int) -> np.ndarray:
    """[G, length]: a token at frame t gets the sum of the normalized rewards at the boundaries
    after t, the rewards whose text includes it."""
    suffix = np.concatenate(
        [np.cumsum(normalized[:, ::-1], axis=1)[:, ::-1], np.zeros((len(normalized), 1))], axis=1
    )
    first_after = np.searchsorted(boundaries, np.arange(length), side="right")
    return suffix[:, first_after]


def group_advantages(
    tokens: np.ndarray,
    sentences: list[Sentence],
    eos_id: int,
    pad_id: int,
    decode,
    alpha: float = ALPHA,
    every: int = REWARD_SPEECH_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One stream's G rollouts, text tokens [G, T]: their reward boundaries [J], rewards [G, J]
    and per-frame advantages [G, T]."""
    length = tokens.shape[1]
    boundaries = reward_frames(sentences, length, every)
    references = prefix_references(sentences, boundaries)
    hypotheses = [emitted_texts(row, boundaries, eos_id, pad_id, decode) for row in tokens]
    rewards = process_rewards(hypotheses, references, alpha)
    return boundaries, rewards, advantages(normalize(rewards), boundaries, length)
