"""Phase A rollout + reward engine for Hibiki-Zero style GRPO (docs/grpo_plan.md §3).

Batched free-running generation of G samples per input from any full-model
checkpoint, plus the paper-exact process-reward stack (arXiv 2602.11072v2
§3.3/§4.2.5): source-sentence frame mapping, reward frames every 8 input
words, prefix references, per-frame group normalization, and suffix-sum
advantages.

Two documented approximations (no Vietnamese forced alignment exists in this
repo), per plan §2:
  1. Source sentence boundaries and reward frames map to source frames
     proportionally by word count over `vi_frames`.
  2. A generated hypothesis is truncated at reward frame t by mapping frame t
     back to hypothesis words with the same proportional rule over the
     generated length T.

Seeding scheme (reproducible): before generating batch i (0-indexed over the
selected row list) we call common.seed_all(seed + i * 1000003). Group members
of one input are adjacent in every batch: order is row0*m0..m{G-1}, then
row1*m0..m{G-1}, and so on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from finetune import common  # noqa: E402
from finetune.utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_PAIRS_DIR,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_file,
    resolve_repo_path,
)

SENTENCE_TERMINATORS = ".!?…;:"
DEFAULT_WORDS_PER_GROUP = 8


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation into non-empty stripped sentences."""
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in SENTENCE_TERMINATORS:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _word_count(text: str) -> int:
    return len(text.split())


def build_source_frame_map(text_vi: str, vi_frames: int) -> list[int]:
    """Map each sentence-end word index to a source frame index.

    Plan §2 approximation: proportional by cumulative word count over
    vi_frames; no Vietnamese forced alignment exists.
    """
    if vi_frames <= 0:
        raise ValueError(f"vi_frames must be positive, got {vi_frames}")
    total_words = sum(_word_count(s) for s in split_sentences(text_vi))
    if total_words == 0:
        return [vi_frames]
    frames = []
    seen = 0
    for sentence in split_sentences(text_vi):
        seen += _word_count(sentence)
        frame = round(vi_frames * seen / total_words)
        frames.append(min(max(frame, 1), vi_frames))
    # Enforce monotone non-decreasing after clamping.
    for index in range(1, len(frames)):
        frames[index] = max(frames[index], frames[index - 1])
    return frames


def reward_frame_positions(
    text_vi: str, vi_frames: int, words_per_group: int = DEFAULT_WORDS_PER_GROUP
) -> list[int]:
    """Reward frames at the end timestamp of every words_per_group-th word."""
    if vi_frames <= 0:
        raise ValueError(f"vi_frames must be positive, got {vi_frames}")
    if words_per_group <= 0:
        raise ValueError("words_per_group must be positive")
    total_words = _word_count(text_vi)
    positions: list[int] = []
    for count in range(words_per_group, total_words + 1, words_per_group):
        frame = min(max(round(vi_frames * count / total_words), 1), vi_frames)
        if not positions or frame > positions[-1]:
            positions.append(frame)
    if not positions or positions[-1] != vi_frames:
        positions.append(vi_frames)
    return positions


def prefix_reference(text_en: str, sentence_index: int) -> str:
    """Gold translation of source sentences fully spoken up to a time.

    Progressive sentence-prefix of text_en: sentences 0..sentence_index joined
    with spaces (PhoMT rows are sentence-aligned pairs).
    """
    sentences = split_sentences(text_en)
    if not 0 <= sentence_index < len(sentences):
        raise IndexError(
            f"sentence_index {sentence_index} outside 0..{len(sentences) - 1}"
        )
    return " ".join(sentences[: sentence_index + 1])


def partial_hypothesis(text: str, t_frames: int, total_frames: int) -> str:
    """Hypothesis truncated at reward frame t via the plan's second approximation."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not 1 <= t_frames <= total_frames:
        raise ValueError(f"t_frames {t_frames} outside 1..{total_frames}")
    words = text.split()
    keep = round(len(words) * t_frames / total_frames)
    return " ".join(words[:keep])


def compute_rewards(
    hyps_by_member: list[str],
    text_en: str,
    text_vi: str,
    vi_frames: int,
    alpha: float = 0.4,
    words_per_group: int = DEFAULT_WORDS_PER_GROUP,
) -> dict[str, Any]:
    """Paper Eq. 3 process rewards across the group at each reward frame.

    r_t = (1 - alpha) * corpus-BLEU(partial hyps, prefix refs)
          + alpha * corpus-BLEU(full hyps, full ref)

    One corpus BLEU per reward frame over all G hypotheses, so rewards are
    comparable within a frame. Empty hypotheses score 0.0 (sacrebleu handles
    them without raising).
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")
    if not hyps_by_member:
        raise ValueError("compute_rewards requires at least one group member")
    frames = reward_frame_positions(text_vi, vi_frames, words_per_group)
    sentences_vi = split_sentences(text_vi)
    total_words = max(sum(_word_count(s) for s in sentences_vi), 1)
    rewards: list[list[float]] = []
    full_refs = [text_en.strip()] * len(hyps_by_member)
    full_hyps = [text.strip() for text in hyps_by_member]
    full_bleu = sacrebleu_corpus_bleu(full_hyps, [full_refs])
    for frame in frames:
        # Sentence whose end covers this frame (prefix reference selection).
        sentence_map = build_source_frame_map(text_vi, vi_frames)
        covered = sum(1 for end in sentence_map if end <= frame)
        reference = prefix_reference(text_en, min(max(covered - 1, 0), len(sentences_vi) - 1))
        partial_hyps = [
            partial_hypothesis(text, frame, vi_frames) for text in hyps_by_member
        ]
        refs = [reference.strip()] * len(hyps_by_member)
        partial_bleu = sacrebleu_corpus_bleu(partial_hyps, [refs])
        rewards.append(
            [(1 - alpha) * partial_bleu + alpha * full_bleu for _ in hyps_by_member]
        )
    return {"reward_frames": frames, "rewards": rewards}


def normalize_rewards(rewards: list[list[float]]) -> list[list[float]]:
    """Per-frame z-scores over the G members ((r - mean) / population std).

    A frame with zero std contributes zeros for every member.
    """
    normalized: list[list[float]] = []
    for frame_rewards in rewards:
        members = len(frame_rewards)
        mean = sum(frame_rewards) / members
        variance = sum((value - mean) ** 2 for value in frame_rewards) / members
        std = variance ** 0.5
        if std == 0:
            normalized.append([0.0] * members)
        else:
            normalized.append([(value - mean) / std for value in frame_rewards])
    return normalized


def advantages(normalized: list[list[float]]) -> list[list[float]]:
    """Paper Eq. 6: R_t = sum of normalized rewards at strictly later frames.

    The last reward frame yields advantage 0 for every member.
    """
    members = len(normalized[0]) if normalized else 0
    result = [[0.0] * members for _ in normalized]
    suffix = [0.0] * members
    for index in range(len(normalized) - 1, -1, -1):
        result[index] = list(suffix)
        for member in range(members):
            suffix[member] += normalized[index][member]
    return result


def shard_filename(batch_index: int) -> str:
    return f"rollout_batch{batch_index:06d}.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_seed(base_seed: int, batch_index: int) -> int:
    """Deterministic per-batch seed; documented in the module docstring."""
    return base_seed + batch_index * 1000003


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_DIR / "validation.jsonl")
    parser.add_argument("--checkpoint", type=Path, help="Optional full-model .safetensors checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--ids", default="")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=1, help="Inputs per batch (x group size rows).")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--gen-duration", type=float, default=0.0)
    parser.add_argument("--tail-s", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--words-per-group", type=int, default=DEFAULT_WORDS_PER_GROUP)
    parser.add_argument("--audio-temp", type=float, default=0.8)
    parser.add_argument("--text-temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-k-text", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="rollout")
    return parser.parse_args()


def generate_rollouts(args: argparse.Namespace) -> None:
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = common.check_device(args.device)
    dtype = common.dtype_from_name(args.dtype)
    checkpoint = require_file(args.checkpoint, "checkpoint") if args.checkpoint else None
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")

    ids = common.ids_from_args(args.ids, args.ids_file)
    rows = common.select_eval_rows(common.read_eval_rows(args.pairs), ids, "id", args.limit)
    common.validate_eval_rows(rows, "vi_audio", "text_en", "id")
    if not rows:
        raise RuntimeError(f"No rows selected from {args.pairs}")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = common.load_checkpoint_info(args)
    checkpoint_info.lm_gen_config["temp"] = args.audio_temp
    checkpoint_info.lm_gen_config["temp_text"] = args.text_temp
    checkpoint_info.lm_gen_config["top_k"] = args.top_k
    checkpoint_info.lm_gen_config["top_k_text"] = args.top_k_text

    print(f"Loading Mimi on {device} from {repo_display_path(args.mimi_weight)}")
    mimi = checkpoint_info.get_mimi(device=device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
    if checkpoint is not None:
        common.load_model(lm, checkpoint, dtype)
    else:
        print("Using the base model.")
    lm.eval()

    manifest_rows: list[dict[str, Any]] = []
    eos_id = int(text_tokenizer.eos_id())
    for batch_index, start in enumerate(range(0, len(rows), args.batch_size)):
        batch_rows = rows[start : start + args.batch_size]
        common.seed_all(batch_seed(args.seed, batch_index))
        shard_path, elapsed = generate_batch_shard(
            batch_rows,
            batch_index,
            args,
            device,
            mimi,
            lm,
            text_tokenizer,
            checkpoint_info,
            eos_id,
            out_dir,
        )
        manifest_rows.append(
            {
                "shard": shard_path.name,
                "sha256": sha256_file(shard_path),
                "row_ids": [row["id"] for row in batch_rows],
                "group_size": args.group_size,
            }
        )
        print(
            f"batch={batch_index} rows={len(batch_rows)} G={args.group_size} "
            f"elapsed={elapsed:.1f}s -> {repo_display_path(shard_path)}"
        )

    manifest_path = out_dir / "rollouts_manifest.jsonl"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"Wrote rollout manifest ({len(manifest_rows)} shards) -> {repo_display_path(manifest_path)}")


def generate_batch_shard(
    batch_rows: list[dict[str, str]],
    batch_index: int,
    args: argparse.Namespace,
    device: torch.device,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    eos_id: int,
    out_dir: Path,
) -> tuple[Path, float]:
    group_size = args.group_size
    files = [resolve_repo_path(row["vi_audio"]) for row in batch_rows]
    input_wavs = [
        common.audio_read(path, to_sample_rate=mimi.sample_rate, mono=True)[0] for path in files
    ]
    audio_durations = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    gen_duration = (
        args.gen_duration if args.gen_duration else max(audio_durations) + args.tail_s
    )
    if max(audio_durations) > gen_duration:
        raise RuntimeError(f"Source audio longer than generation duration: {max(audio_durations)}")

    expanded_wavs = [wav for wav in input_wavs for _ in range(group_size)]
    expanded_durations = [d for d in audio_durations for _ in range(group_size)]
    batch_wavs = common.stack_and_pad_audio(
        expanded_wavs, max_len=int(gen_duration * mimi.sample_rate)
    )
    lm_gen = common.get_lmgen(lm, checkpoint_info, batch_size=batch_wavs.shape[0])
    codes, warmup_codes = common.encode_inputs(batch_wavs, mimi, lm_gen, expanded_durations)

    output_text_tokens: list[Any] = []
    output_audio_tokens: list[Any] = []
    start_time = time.time()
    with torch.no_grad(), lm_gen.streaming(batch_wavs.shape[0]):
        for step in range(warmup_codes.shape[-1]):
            _ = lm_gen.step(warmup_codes[:, :, step : step + 1])
        for step in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, step : step + 1])
            if tokens is None:
                continue
            output_text_tokens.append(tokens[:, 0, :])
            output_audio_tokens.append(tokens[:, 1:, :])
    elapsed = time.time() - start_time

    if not output_text_tokens:
        raise RuntimeError("LM generation produced no output tokens; increase --gen-duration")
    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    batch_audio_codes = torch.concat(output_audio_tokens, dim=-1).to(torch.int16).cpu()
    decoded = common.decode_text_batch(batch_text_tokens, text_tokenizer, warn=True)

    shard: dict[str, Any] = {
        "batch_index": batch_index,
        "tag": args.tag,
        "row_ids": [row["id"] for row in batch_rows],
        "member_ids": [
            f"{row['id']}#m{member}" for row in batch_rows for member in range(group_size)
        ],
        "codes": batch_audio_codes,
        "text_token_ids": batch_text_tokens.cpu(),
        "texts": [item["text"] for item in decoded],
        "eos_flags": [bool(item["eos_found"]) for item in decoded],
        "gen_lengths": [int(item["generated_text_tokens"]) for item in decoded],
        "gen_duration_s": gen_duration,
        "generated_frames": int(batch_audio_codes.shape[-1]),
        "vi_frames": [
            round(float(row["vi_duration_s"]) * float(mimi.frame_rate)) for row in batch_rows
        ],
        "config": {
            "alpha": args.alpha,
            "words_per_group": args.words_per_group,
            "audio_temp": args.audio_temp,
            "text_temp": args.text_temp,
            "top_k": args.top_k,
            "top_k_text": args.top_k_text,
            "group_size": group_size,
            "seed": args.seed,
            "checkpoint": (
                repo_display_path(args.checkpoint) if args.checkpoint is not None else None
            ),
        },
        "reward_frames": [],
        "rewards_raw": [],
        "rewards_normalized": [],
        "advantages": [],
    }
    for local_row, row in enumerate(batch_rows):
        hyps = decoded[local_row * group_size : (local_row + 1) * group_size]
        rewards_result = compute_rewards(
            [item["text"] for item in hyps],
            row["text_en"],
            row["text_vi"],
            shard["vi_frames"][local_row],
            alpha=args.alpha,
            words_per_group=args.words_per_group,
        )
        normalized = normalize_rewards(rewards_result["rewards"])
        shard["reward_frames"].append(rewards_result["reward_frames"])
        shard["rewards_raw"].append(rewards_result["rewards"])
        shard["rewards_normalized"].append(normalized)
        shard["advantages"].append(advantages(normalized))

    shard_path = out_dir / shard_filename(batch_index)
    temporary = shard_path.with_name(f".{shard_path.name}.tmp")
    try:
        torch.save(shard, temporary)
        temporary.replace(shard_path)
    finally:
        temporary.unlink(missing_ok=True)
    flat_advantages = [value for row in shard["advantages"] for frame in row for value in frame]
    flat_rewards = [value for row in shard["rewards_raw"] for frame in row for value in frame]
    print(
        f"Generated batch {batch_index} in {elapsed:.1f}s "
        f"({gen_duration / elapsed:.2f}x RT); reward mean="
        f"{sum(flat_rewards) / max(len(flat_rewards), 1):.2f}; advantage range="
        f"[{min(flat_advantages, default=0.0):.3f}, {max(flat_advantages, default=0.0):.3f}]"
    )
    return shard_path, elapsed


def main() -> None:
    args = parse_args()
    generate_rollouts(args)


# Imported after helpers so module import stays light for CPU-only tests.
import sacrebleu  # noqa: E402


def sacrebleu_corpus_bleu(hypotheses: list[str], references: list[list[str]]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sacrebleu.corpus_bleu(hypotheses, references).score)


import warnings  # noqa: E402


if __name__ == "__main__":
    main()
