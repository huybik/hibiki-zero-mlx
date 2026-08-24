"""Offline CPU tests for the GRPO rollout + reward engine (Phase A exit gate)."""
from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import sacrebleu

from finetune import grpo_rollout as gr


def _bleu(hyps: list[str], refs: list[list[str]]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sacrebleu.corpus_bleu(hyps, refs).score)


# --------------------------------------------------------------------------- #
# Frame / word mapping
# --------------------------------------------------------------------------- #
def test_frame_map_monotone_bounded_and_covering() -> None:
    text = "một hai ba. bốn năm sáu! bảy tám chín"
    frames = gr.build_source_frame_map(text, 100)
    assert len(frames) == 3
    assert all(1 <= f <= 100 for f in frames)
    assert frames == sorted(frames)
    positions = gr.reward_frame_positions(text, 100)
    assert positions[-1] == 100
    assert all(b > a for a, b in zip(positions, positions[1:]))
    assert all(1 <= p <= 100 for p in positions)


def test_reward_frames_every_eight_words() -> None:
    text = " ".join(f"w{i}" for i in range(20))
    # 20 words, every 8th -> word counts 8 and 16; final frame appended.
    positions = gr.reward_frame_positions(text, 50)
    assert positions == [round(50 * 8 / 20), round(50 * 16 / 20), 50]
    # Exactly divisible case must not duplicate the final frame.
    text2 = " ".join(f"w{i}" for i in range(16))
    assert gr.reward_frame_positions(text2, 40) == [20, 40]


def test_split_sentences_and_prefix_reference() -> None:
    text = "Xin chào. Tôi tên Minh! Còn bạn?"
    sentences = gr.split_sentences(text)
    assert sentences == ["Xin chào.", "Tôi tên Minh!", "Còn bạn?"]
    en = "hello. my name is Minh! and you?"
    assert gr.prefix_reference(en, 0) == "hello."
    assert gr.prefix_reference(en, 2) == "hello. my name is Minh! and you?"
    try:
        gr.prefix_reference(en, 3)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError beyond the last sentence")


# --------------------------------------------------------------------------- #
# Reward math (hand-computable fixtures)
# --------------------------------------------------------------------------- #
PERFECT_VI = "một hai ba. bốn năm sáu"
PERFECT_EN = "one two three. four five six"


def test_perfect_translation_scores_hundred_everywhere() -> None:
    result = gr.compute_rewards(
        [PERFECT_EN] * 4, PERFECT_EN, PERFECT_VI, vi_frames=40
    )
    assert result["reward_frames"] == gr.reward_frame_positions(PERFECT_VI, 40)
    for frame_rewards in result["rewards"]:
        for value in frame_rewards:
            assert math.isclose(value, 100.0)


def test_rewards_match_independent_sacrebleu_expression() -> None:
    hyps = [
        "one two three four five six",
        "one two three four five seven",
        "one two three four eight six",
        "",
    ]
    alpha = 0.4
    result = gr.compute_rewards(hyps, PERFECT_EN, PERFECT_VI, vi_frames=40, alpha=alpha)
    full_bleu = _bleu([h.strip() for h in hyps], [[PERFECT_EN.strip()] * 4])
    for frame_index, frame in enumerate(result["reward_frames"]):
        partial_hyps = [
            gr.partial_hypothesis(text, frame, 40) for text in hyps
        ]
        covered = sum(
            1
            for end in gr.build_source_frame_map(PERFECT_VI, 40)
            if end <= frame
        )
        reference = gr.prefix_reference(PERFECT_EN, max(covered - 1, 0))
        expected = (1 - alpha) * _bleu(partial_hyps, [[reference.strip()] * 4]) + (
            alpha * full_bleu
        )
        for member in range(4):
            assert math.isclose(result["rewards"][frame_index][member], expected), (
                frame_index,
                member,
            )


def test_partial_hypothesis_mapping() -> None:
    text = "a b c d e f g h i j"
    assert gr.partial_hypothesis(text, 10, 10) == text
    assert gr.partial_hypothesis(text, 5, 10) == "a b c d e"
    assert gr.partial_hypothesis("a b", 1, 10) == ""
    try:
        gr.partial_hypothesis(text, 0, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("t_frames=0 must be rejected")


# --------------------------------------------------------------------------- #
# Group statistics + advantages (Eq. 5-6)
# --------------------------------------------------------------------------- #
def test_normalize_and_advantage_properties() -> None:
    rewards = [
        [10.0, 20.0, 30.0, 40.0],
        [5.0, 5.0, 5.0, 5.0],  # zero std frame
        [1.0, 2.0, 3.0, 4.0],
    ]
    normalized = gr.normalize_rewards(rewards)
    mean = lambda values: sum(values) / len(values)
    std = lambda values: math.sqrt(mean([(v - mean(values)) ** 2 for v in values]))
    for index, frame in enumerate(rewards):
        m, s = mean(frame), std(frame)
        if s == 0:
            assert all(v == 0.0 for v in normalized[index])
        else:
            assert all(
                math.isclose(normalized[index][i], (frame[i] - m) / s) for i in range(4)
            )

    adv = gr.advantages(normalized)
    members = len(rewards[0])
    for member in range(members):
        # R_t equals the suffix sum over strictly later frames.
        for index in range(len(rewards)):
            expected = sum(
                normalized[later][member] for later in range(index + 1, len(rewards))
            )
            assert math.isclose(adv[index][member], expected)
        assert adv[-1][member] == 0.0

    # A strictly above-mean member earns positive normalized rewards at every
    # frame; earlier steps see more later frames, so R_t is non-increasing in t.
    rewards2 = [[1.0, 9.0], [1.0, 9.0], [1.0, 9.0]]
    adv2 = gr.advantages(gr.normalize_rewards(rewards2))
    column = [row[1] for row in adv2]
    assert column == sorted(column, reverse=True)
    assert all(value >= 0 for value in column)


# --------------------------------------------------------------------------- #
# Helpers: hashing + deterministic seeding
# --------------------------------------------------------------------------- #
def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 64
    path.write_bytes(payload)
    assert gr.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_batch_seed_deterministic() -> None:
    assert gr.batch_seed(42, 0) == 42
    assert gr.batch_seed(42, 3) == gr.batch_seed(42, 3)
    assert gr.batch_seed(42, 3) != gr.batch_seed(42, 4)
