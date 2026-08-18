#!/usr/bin/env python
"""Verify a strict q4 student pack against its PyTorch parallel-head fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from hibiki_mlx import pipeline
from hibiki_mlx.student_pack import read_json
from moshi_mlx import models, utils


def errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return float(difference.max()), float(difference.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack_dir", type=Path)
    parser.add_argument("--embedding-max", type=float, default=0.25)
    parser.add_argument("--logits-max", type=float, default=0.75)
    parser.add_argument("--logits-mean", type=float, default=0.15)
    args = parser.parse_args()

    # pipeline.load() performs the single full strict pack validation (manifest,
    # receipts, q4, parity); read only the config here for the fixture paths.
    cfg = read_json(args.pack_dir / "config.json")
    model, lm_config, _tokenizer, _mimi_enc, _mimi_dec = pipeline.load(args.pack_dir)
    if model.parallel_head is None or model.depformer is not None:
        raise RuntimeError("Student pack did not instantiate exactly one parallel_v1 head")
    with np.load(args.pack_dir / cfg["parity_fixture_name"], allow_pickle=False) as fixture:
        hidden = fixture["hidden"]
        text_ids = fixture["text_ids"]
        text_embedding = fixture["text_embedding"]
        previous = fixture["previous_codes"]
        expected_logits = fixture["logits"]
        expected_next = fixture["next_previous_codes"]

    q4_embedding = model.text_emb(mx.array(text_ids))
    actual_logits = model.parallel_head(
        mx.array(hidden), mx.array(text_embedding), mx.array(previous)
    )
    mx.eval(q4_embedding, actual_logits)
    q4_embedding = np.array(q4_embedding)
    actual_logits = np.array(actual_logits)
    embedding_max, embedding_mean = errors(q4_embedding, text_embedding)
    logits_max, logits_mean = errors(actual_logits, expected_logits)
    actual_next = actual_logits.argmax(axis=-1)[:, -1].astype(np.int32)

    generator = models.LmGen(
        model=model,
        max_steps=2,
        text_sampler=utils.Sampler(temp=0),
        audio_sampler=utils.Sampler(temp=0),
    )
    initial_state = np.array(generator.previous_raw_audio_tokens)
    expected_initial = np.full((1, 8), int(cfg["card"]), dtype=np.int32)
    state_ok = np.array_equal(initial_state, expected_initial)
    next_ok = np.array_equal(actual_next, expected_next)
    numeric_ok = (
        embedding_max <= args.embedding_max
        and logits_max <= args.logits_max
        and logits_mean <= args.logits_mean
    )
    print(
        f"text embedding: max={embedding_max:.6f} mean={embedding_mean:.6f} "
        f"(max <= {args.embedding_max})"
    )
    print(
        f"parallel logits: max={logits_max:.6f} mean={logits_mean:.6f} "
        f"(max <= {args.logits_max}, mean <= {args.logits_mean})"
    )
    print(f"head state: initial_card={state_ok} next_raw_argmax={next_ok}")
    if not (numeric_ok and state_ok and next_ok):
        raise SystemExit("FAIL: q4 MLX/PyTorch parallel parity")
    print("PASS: q4 MLX/PyTorch parallel parity")


if __name__ == "__main__":
    main()
