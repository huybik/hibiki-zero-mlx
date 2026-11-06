"""Grounded-v2 cache rows, shared by the MLX builder (`cache_codes`, the Mac) and the CUDA one
(`training.tranche.cache`, the box): the English delay draw and the `[33, T]` code layout
([sft-data-caches](../../docs/agent-memory/sft-data-caches.md#row-layout))."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

FRAME_RATE = 12.5
CONFIG_KEYS = ("n_q", "dep_q", "card", "text_card", "existing_text_padding_id")


def target_delay_s(row_id: str, split: str, max_delay_s: float, seed: int) -> float:
    """Draw the English left-silence delay from U[0, max_delay_s], seeded per row."""
    return random.Random(f"{seed}:{split}:{row_id}").uniform(0.0, max_delay_s)


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
