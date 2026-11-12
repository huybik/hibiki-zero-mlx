"""Shared training and evaluation logic for Vietnamese full-model SFT.

This module owns device helpers, exact full-model checkpoint I/O, the loss kernel and the
teacher-forced losses on it, free-running (sampled) translation, text metrics, and the
generation health gates. Training streams are built in `streams.py`; checkpoint pairs, resume
and logs are in `checkpoints.py`.

It runs on CUDA in the environment that `training/h100.sh setup` builds.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sacrebleu
import sphn
import torch
from moshi.models import loaders
from moshi.modules.transformer import set_attention_context
from moshi.run_inference import get_condition_tensors
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from inference.hibiki_mlx.level import Leveller
from training.finetune import lm_training
from training.finetune.hibiki_helpers import (
    audio_read,
    decode_wavs,
    encode_inputs,
    get_lmgen,
    stack_and_pad_audio,
)
from training.paths import DEFAULT_TOKENIZER, repo_display_path, require_file, resolve_repo_path


def ascii_text(text: str) -> str:
    """Remove Latin diacritics for stable word-metric normalization."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).encode(
        "ascii", "ignore"
    ).decode("ascii")


# --------------------------------------------------------------------------- #
# Device / dtype / seeding
# --------------------------------------------------------------------------- #
def check_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# moshi's training forward, faster (lm_training.py); inference keeps moshi's path.
lm_training.install()


def require_torch_compile() -> None:
    """The trainers' forward and loss are compiled functions (`lm_training`, `token_ce`)."""
    if os.environ.get("NO_TORCH_COMPILE"):
        raise ValueError("Full-model training requires torch compile; unset NO_TORCH_COMPILE")


# --------------------------------------------------------------------------- #
# Full-model checkpoint I/O
# --------------------------------------------------------------------------- #
def fsync_path(path: Path) -> None:
    """Flush a file, or a directory's entries, to disk: a rename alone can survive a host crash
    that the renamed file's data does not."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_state(state: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    """Write a model's state dict (host tensors) atomically as safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        save_file(state, str(temp_path), metadata=metadata)
        fsync_path(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_model(model: Any, path: Path, dtype: torch.dtype) -> None:
    """Load an exact full-model checkpoint; partial states are rejected."""
    state = load_file(str(path), device="cpu")
    expected = set(model.state_dict())
    actual = set(state)
    if missing := sorted(expected - actual):
        raise RuntimeError(f"Checkpoint is missing model tensors: {missing[:5]}")
    if unexpected := sorted(actual - expected):
        raise RuntimeError(f"Checkpoint has unexpected tensors: {unexpected[:5]}")
    # Converted in place, each fp32 tensor freed as it goes: a 12.5 GB fp32 checkpoint and its
    # bf16 copy whole at once held 18 GiB on the Mac.
    for name, tensor in state.items():
        if tensor.dtype.is_floating_point:
            state[name] = tensor.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} model tensors from {repo_display_path(path)}")


TOKENIZER_KEY = "text_tokenizer"  # a model weight's metadata: the tokenizer its text rows are on


def weight_tokenizer(path: Path) -> str:
    """The tokenizer a model weight's text rows are on: its `text_tokenizer` metadata, which
    every training checkpoint records since 2026-11-10; upstream's weight and older checkpoints
    are on the base."""
    with safe_open(str(path), framework="pt") as handle:
        return (handle.metadata() or {}).get(TOKENIZER_KEY, DEFAULT_TOKENIZER.name)


def run_tokenizer(tokenizer: Path | None, weight: Path) -> Path:
    """The run's text tokenizer: `tokenizer` (--tokenizer), by default the one `weight` is on,
    from `weights/`. Every entry point calls it at startup, before anything slow: a weight on
    another tokenizer is refused, since the Vietnamese swap gives 6,000 text rows new meanings
    (`vi_tokenizer.init_weights` moves a base weight onto it)."""
    on = weight_tokenizer(weight)
    if tokenizer is None and not (DEFAULT_TOKENIZER.parent / on).is_file():
        raise FileNotFoundError(f"{repo_display_path(weight)} is on {on}, missing from weights/: copy it there or pass --tokenizer")
    path = require_file(tokenizer or DEFAULT_TOKENIZER.parent / on, "tokenizer")
    if on != path.name:
        raise RuntimeError(f"{repo_display_path(weight)} is on {on}, not --tokenizer {path.name}")
    return path


def bf16_copy(checkpoint_info: Any, lm: Any, device: Any) -> Any:
    """A policy's weights in a bf16 generation model, as `eval.py` loads a checkpoint: 1.8x faster
    than generating on the fp32 masters. Built without reading the base weights, which the copy
    overwrites. moshi's streaming state holds it in reference cycles: after `del`, only
    `gc.collect()` frees its 5.8 GiB."""
    generator = checkpoint_info.get_moshi(device=device, dtype=torch.bfloat16, load_weight=False)
    generator.load_state_dict(lm.state_dict())
    generator.eval()
    return generator


def load_policy(checkpoint_info: Any, device: Any) -> Any:
    """The model a trainer updates (SFT's and GRPO's), from the checkpoint info's weight
    (--model-weight): fp32 masters, every parameter trainable, in training mode.

    Loading a Hibiki weight, moshi's `get_moshi` copies the text PAD's input embedding over the
    text EOS's (so a sampled early EOS reads on as PAD), so every run starts from that; the row
    trains from there (cosine 0.88 to PAD's after round 4), and a resumed pair restores its own.
    `load_lm` and `bf16_copy` keep a checkpoint's own row."""
    lm = checkpoint_info.get_moshi(
        device=device, dtype=torch.float32, lm_kwargs_overrides={"gradient_checkpointing": False}
    )
    lm.train()
    for param in lm.parameters():
        param.requires_grad_(True)
    return lm


def adamw(params: list[Any], args: argparse.Namespace) -> torch.optim.Optimizer:
    """The trainers' optimizer, from --lr, --adam-beta1/2 and --weight-decay: AdamW on the fp32
    weights, its two moments kept in 8 bits (bitsandbytes `AdamW8bit`: blocks of 256 values, each
    with its own scale; a tensor under 4,096 values keeps fp32 moments). For the 3.1B weights the
    moments take 6.4 GB, not 25, and a step moves 16 bytes a weight, not fused AdamW's 28
    (docs/agent-memory/sft-speed.md#8-bit-moments)."""
    # Imported here: bitsandbytes loads Triton, which turns the GIL back on in eval.py, validate.py
    # and the other importers of this module that never build an optimizer.
    import bitsandbytes

    return bitsandbytes.optim.AdamW8bit(
        params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )


def load_lm(checkpoint_info: Any, checkpoint: Path | None, device: Any, dtype: torch.dtype) -> Any:
    """The base model, or `checkpoint` in a model built without reading the base weights, which
    a full checkpoint overwrites anyway (12.5 GB of fp32 for a fine-tuned base)."""
    if checkpoint is None:
        print(f"Loading the base LM on {device} from {repo_display_path(checkpoint_info.moshi_weights)}")
        return checkpoint_info.get_moshi(device=device, dtype=dtype)
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype, load_weight=False)
    load_model(lm, checkpoint, dtype)
    return lm


# --------------------------------------------------------------------------- #
# Teacher-forced losses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossWeights:
    """Kyutai's moshi-finetune weighting: audio codebook 0 (semantic) counts `first_codebook`
    times an acoustic codebook and a text PAD `text_pad` times a word token, each within its own
    weighted mean; the audio and text means are then added with weights `audio` and `text`."""

    audio: float = 1.0
    text: float = 1.0
    first_codebook: float = 100.0
    text_pad: float = 0.5


# Each component's loss sum and the token count or weight it is divided by.
LOSS_SUMS = ("audio", "semantic", "acoustic", "text", "content", "pad")


def batch_condition_tensors(lm: Any, model_type: str, batch_size: int) -> Any | None:
    if lm.fuser is None:
        return None
    return get_condition_tensors(model_type, lm, batch_size=batch_size, cfg_coef=1.0)


def loss_means(sums: dict[str, Any], weights: LossWeights) -> dict[str, Any]:
    """Losses from `<name>_sum` / `<name>_n`, as tensors in training or floats in evaluation."""

    def mean(name: str) -> Any:
        total, count = sums[f"{name}_sum"], sums[f"{name}_n"]
        if isinstance(count, torch.Tensor):
            return total / count.clamp(min=1)
        return total / count if count else 0.0

    audio_loss = mean("audio")
    text_loss = mean("text")
    return {
        "loss": weights.audio * audio_loss + weights.text * text_loss,
        "audio_loss": audio_loss,
        "semantic_loss": mean("semantic"),
        "acoustic_loss": mean("acoustic"),
        "text_loss": text_loss,
        "content_text_loss": mean("content"),
        "pad_text_loss": mean("pad"),
    }


@torch.compile(dynamic=True)
def token_ce(logits: Any, targets: Any, valid: Any, temperature: float = 1.0) -> Any:
    """Each target token's cross-entropy under softmax(logits / temperature), float32 in the
    shape of `targets`, and 0 where not `valid` (moshi fills the logits there with NaN, or the
    target is -1). The one loss kernel: SFT's losses are its weighted sums, and GRPO's token
    logprobs its negative at the sampling temperature.

    Dense on purpose. The fp32 scores live only inside the fused kernel, so no full-vocabulary
    fp32 copy is kept for the backward pass, and no boolean index gathers the valid positions
    (a host sync, and 4.5% of an SFT step's kernels on the H200). No gradient reaches the model
    from a masked position: `torch.where` hands its CE a 0, and where the logits are moshi's NaN
    fill the resulting 0 x NaN stops at the fill (`_undelay_sequence` overwrites those slices in
    place, whose backward zeroes them). The trainers still check every gradient norm."""
    scores = logits.float() if temperature == 1.0 else logits.float() / temperature
    ce = torch.nn.functional.cross_entropy(
        scores.flatten(0, -2), torch.where(valid, targets, 0).flatten(), reduction="none"
    ).view(targets.shape)
    return torch.where(valid, ce, 0.0)


def compute_batch_losses(
    lm: Any,
    codes: Any,
    condition_tensors: Any | None,
    weights: LossWeights,
    accuracy: bool = False,
) -> dict[str, Any]:
    """Teacher-forced losses of one batch, as tensors on the GPU (no host sync), with every
    component's sum and count beside them so evaluation can aggregate exactly. Positions holding
    -1 are outside the masks. `accuracy` adds the counts of text tokens the argmax gets right
    (`content_correct`, `pad_correct`)."""
    output = lm(codes, condition_tensors=condition_tensors)
    audio_valid = output.mask  # [B, dep_q, T]
    audio_ce = token_ce(
        output.logits, codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q], audio_valid
    )
    semantic = torch.zeros_like(audio_valid)
    semantic[:, 0] = audio_valid[:, 0]
    acoustic = audio_valid & ~semantic
    audio_weight = torch.where(semantic, weights.first_codebook, 1.0) * audio_valid
    text_valid = output.text_mask  # [B, 1, T]
    text_targets = codes[:, :1]
    text_ce = token_ce(output.text_logits, text_targets, text_valid)
    pad = (text_targets == lm.text_padding_token_id) & text_valid
    content = text_valid & ~pad
    text_weight = torch.where(pad, weights.text_pad, 1.0) * text_valid
    sums = {
        "audio_sum": (audio_ce * audio_weight).sum(),
        "audio_n": audio_weight.sum(),
        "semantic_sum": (audio_ce * semantic).sum(),
        "semantic_n": semantic.sum(),
        "acoustic_sum": (audio_ce * acoustic).sum(),
        "acoustic_n": acoustic.sum(),
        "text_sum": (text_ce * text_weight).sum(),
        "text_n": text_weight.sum(),
        "content_sum": (text_ce * content).sum(),
        "content_n": content.sum(),
        "pad_sum": (text_ce * pad).sum(),
        "pad_n": pad.sum(),
    }
    result = {**sums, **loss_means(sums, weights)}
    if accuracy:
        correct = output.text_logits.argmax(-1) == text_targets
        result["content_correct"] = (correct & content).sum()
        result["pad_correct"] = (correct & pad).sum()
    return result


def evaluate_teacher_forced(
    lm: Any,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    weights: LossWeights,
    max_batches: int = 0,
) -> dict[str, float | int]:
    """Corpus-level teacher-forced losses and text accuracies over `dataloader`, under the bf16
    autocast that training computes its losses in (1.7x faster than fp32 weights without it)."""
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    keys = [f"{name}_{part}" for name in LOSS_SUMS for part in ("sum", "n")]
    keys += ["content_correct", "pad_correct"]
    totals = dict.fromkeys(keys, 0.0)
    batches = streams_seen = max_frames = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for batch_index, batch in enumerate(dataloader):
            if max_batches and batch_index >= max_batches:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(lm, model_type, batch_size)
            result = compute_batch_losses(
                lm, codes, condition_cache[batch_size], weights, accuracy=True
            )
            for key in keys:
                totals[key] += float(result[key])
            batches += 1
            streams_seen += batch_size
            max_frames = max(max_frames, int(batch["frames"].max()))
    if was_training:
        lm.train()
    if not batches:
        raise RuntimeError("Validation dataloader produced no batches.")
    content_tokens = int(totals["content_n"])
    pad_tokens = int(totals["pad_n"])
    return {
        **loss_means(totals, weights),
        "audio_tokens": int(totals["semantic_n"] + totals["acoustic_n"]),
        "content_tokens": content_tokens,
        "content_acc": totals["content_correct"] / content_tokens if content_tokens else 0.0,
        "pad_tokens": pad_tokens,
        "pad_acc": totals["pad_correct"] / pad_tokens if pad_tokens else 0.0,
        "batches": batches,
        "streams": streams_seen,
        "max_frames": max_frames,
    }


# --------------------------------------------------------------------------- #
# Eval row IO + selection
# --------------------------------------------------------------------------- #
EVAL_SET = "eval_set"  # the column read_eval_sets adds: the set a row belongs to
# The columns it adds for a row's direction: the clip heard and the text its translation is
# scored against. A pairs file holds the audio of one language (the key) and is translated into
# the other's text.
SOURCE, REFERENCE = "source_audio", "reference_text"
EVAL_DIRECTIONS = {"vi_audio": "text_en", "en_audio": "text_vi"}


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    path = require_file(path, "eval rows")
    if path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [{k: v or "" for k, v in row.items()} for row in csv.DictReader(handle)]
    if path.suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append({key: str(value) for key, value in row.items()})
        return rows
    raise ValueError(f"Eval rows must be .jsonl or .csv: {path}")


def read_eval_sets(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Each pairs file's rows by set name, in the order given, each row tagged under EVAL_SET and
    holding its clip and reference under SOURCE and REFERENCE. A set is named by its file's folder
    and stem, such as fleurs-vi/dev, and must have rows."""
    sets: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        path = require_file(path, "eval rows")
        name = f"{path.parent.name}/{path.stem}"
        if name in sets:
            raise ValueError(f"Two pairs files are both named {name}")
        rows = read_eval_rows(path)
        if not rows:
            raise ValueError(f"No rows in {path}")
        audio = [column for column in EVAL_DIRECTIONS if column in rows[0]]
        if len(audio) != 1:
            raise ValueError(f"{path} must hold one of the columns {', '.join(EVAL_DIRECTIONS)}, has {audio}")
        for index, row in enumerate(rows):
            if taken := {EVAL_SET, SOURCE, REFERENCE} & set(row):
                raise ValueError(f"{path} has columns {sorted(taken)} of its own")
            if [column for column in EVAL_DIRECTIONS if column in row] != audio:
                raise ValueError(f"{path} row {index} is heard in other audio than its first row's {audio[0]}")
            if missing := {audio[0], EVAL_DIRECTIONS[audio[0]]} - set(row):
                raise ValueError(f"{path} row {index} is missing columns: {', '.join(sorted(missing))}")
            row[EVAL_SET] = name
            row[SOURCE], row[REFERENCE] = row[audio[0]], row[EVAL_DIRECTIONS[audio[0]]]
        sets[name] = rows
    return sets


def selecting_sets(eval_sets: list[str], rows: list[dict[str, str]]) -> list[str]:
    """The sets that select a checkpoint: the first of each direction (the audio a set is heard
    in, `EVAL_DIRECTIONS`), in the order given."""
    heard: dict[str, str] = {}
    for row in rows:
        heard.setdefault(row[EVAL_SET], next(column for column in EVAL_DIRECTIONS if column in row))
    first: dict[str, str] = {}
    for name in eval_sets:
        first.setdefault(heard[name], name)
    return list(first.values())


def load_eval_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    """A trainer's evaluation rows: the set names in the order given (each direction's first
    selects the checkpoint, `selecting_sets`) and every set's rows, checked before the run starts rather than at its first
    validation: the columns, unique ids, a duration and the audio file of each."""
    cfg = TRAINER_GENERATION
    sets = read_eval_sets(paths)
    rows = [row for set_rows in sets.values() for row in set_rows]
    validate_eval_rows(rows, cfg.id_column, duration_column="duration_s")
    for row in rows:
        require_file(row[SOURCE], "evaluation audio")
    return list(sets), rows


def validate_eval_rows(
    rows: list[dict[str, str]],
    id_column: str,
    duration_column: str | None = None,
) -> None:
    """Rows of `read_eval_sets`: ids present and unique, and positive durations."""
    if not rows:
        return
    required = [id_column]
    if duration_column is not None:
        required.append(duration_column)
    for index, row in enumerate(rows):
        missing = [column for column in required if column not in row]
        if missing:
            raise ValueError(
                f"Eval row {index} is missing columns: {', '.join(missing)}"
            )
    ids = [row[id_column] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Eval rows must have unique {id_column!r} values")
    if duration_column is not None:
        for row in rows:
            try:
                duration = float(row[duration_column])
            except ValueError:
                duration = math.nan
            if not duration > 0:  # also rejects NaN, which would break the length sort
                raise ValueError(
                    f"Eval {duration_column} must be a positive number for "
                    f"{id_column}={row[id_column]}, got {row[duration_column]!r}"
                )


def select_eval_rows(
    rows: list[dict[str, str]], ids: list[str], id_column: str
) -> list[dict[str, str]]:
    """The rows of `ids`, in that order."""
    by_id = {row[id_column]: row for row in rows}
    missing = [row_id for row_id in ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Requested ids are missing: {missing[:10]}")
    return [by_id[row_id] for row_id in ids]


def ids_from_args(ids: str, ids_file: Path | None) -> list[str]:
    selected = [item.strip() for item in ids.split(",") if item.strip()]
    if ids_file is not None:
        path = require_file(ids_file, "id file")
        selected.extend(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate ids in --ids/--ids-file")
    return selected


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    text = ascii_text(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    edits = 0
    total_words = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_words = normalize(reference).split()
        edits += edit_distance(ref_words, normalize(hypothesis).split())
        total_words += len(ref_words)
    return edits / total_words if total_words else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def max_repeated_ngram(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for start in range(len(words) - n + 1):
        ngram = tuple(words[start : start + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return max(counts.values(), default=0)


def length_repetition_metrics(references: list[str], hypotheses: list[str]) -> dict[str, Any]:
    reference_lengths = [len(normalize(reference).split()) for reference in references]
    hypothesis_lengths = [len(normalize(hypothesis).split()) for hypothesis in hypotheses]
    length_ratios = [
        hyp_len / ref_len
        for ref_len, hyp_len in zip(reference_lengths, hypothesis_lengths, strict=True)
        if ref_len > 0
    ]
    repeated_4grams = [
        max_repeated_ngram(normalize(hypothesis).split(), 4) for hypothesis in hypotheses
    ]
    return {
        "mean_reference_words": mean([float(v) for v in reference_lengths]),
        "mean_prediction_words": mean([float(v) for v in hypothesis_lengths]),
        "mean_length_ratio": mean(length_ratios),
        "overlong_predictions": sum(
            1
            for ref_len, hyp_len in zip(reference_lengths, hypothesis_lengths, strict=True)
            if hyp_len > max(32, 2 * ref_len)
        ),
        "repeated_4gram_predictions": sum(1 for value in repeated_4grams if value >= 3),
        "max_repeated_4gram_count": max(repeated_4grams, default=0),
    }


def is_bool_text(value: Any, expected: bool) -> bool:
    return value is expected or str(value).lower() == str(expected).lower()


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    references = [str(record["reference_text"]).strip() for record in records]
    hypotheses = [str(record["prediction_text"]).strip() for record in records]
    nonempty_pairs = [
        (reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
        if hypothesis
    ]
    metrics: dict[str, Any] = {
        "num_predictions": len(records),
        "nonempty_predictions": sum(1 for hypothesis in hypotheses if hypothesis),
        "empty_predictions": sum(1 for hypothesis in hypotheses if not hypothesis),
        "exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if reference == hypothesis
        ),
        "normalized_exact_matches": sum(
            1
            for reference, hypothesis in zip(references, hypotheses, strict=True)
            if normalize(reference) == normalize(hypothesis)
        ),
        "eos_found": sum(1 for record in records if is_bool_text(record.get("eos_found"), True)),
        "eos_missing": sum(1 for record in records if is_bool_text(record.get("eos_found"), False)),
        "wer": word_error_rate(references, hypotheses),
    }
    metrics.update(length_repetition_metrics(references, hypotheses))
    offsets = [float(record["end_offset_s"]) for record in records if record["end_offset_s"] != ""]
    metrics["median_end_offset_s"] = float(np.median(offsets)) if offsets else float("nan")
    metrics["bleu"] = sacrebleu.corpus_bleu(hypotheses, [references]).score
    metrics["chrf"] = sacrebleu.corpus_chrf(hypotheses, [references]).score
    if nonempty_pairs:
        nonempty_refs, nonempty_hyps = zip(*nonempty_pairs, strict=True)
        metrics["nonempty_bleu"] = sacrebleu.corpus_bleu(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
        metrics["nonempty_chrf"] = sacrebleu.corpus_chrf(
            list(nonempty_hyps), [list(nonempty_refs)]
        ).score
    else:
        metrics["nonempty_bleu"] = 0.0
        metrics["nonempty_chrf"] = 0.0
    return metrics


# --------------------------------------------------------------------------- #
# Free-running (sampled) translation
# --------------------------------------------------------------------------- #
def decode_text_batch(batch_text_tokens: Any, text_tokenizer: Any) -> list[dict[str, Any]]:
    eos_id = int(text_tokenizer.eos_id())
    pad_id = int(text_tokenizer.pad_id())
    decoded: list[dict[str, Any]] = []
    for output_idx in range(batch_text_tokens.shape[0]):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        eos_found = eos_id in text_tokens
        # The output ends at the EOS; without one it runs to the end of generation.
        eos_idx = text_tokens.index(eos_id) if eos_found else len(text_tokens)
        words = [index for index, token in enumerate(text_tokens[:eos_idx]) if token > pad_id]
        content_tokens = [text_tokens[index] for index in words]
        decoded.append(
            {
                "text": text_tokenizer.decode(content_tokens),
                "eos_found": eos_found,
                "generated_text_tokens": len(content_tokens),
                "last_word_frame": words[-1] if words else None,
                "end_frame": eos_idx,
            }
        )
    return decoded


def output_paths(out_dir: Path, index: int, source_path: Path, tag: str | None) -> dict[str, Path]:
    suffix = "" if tag is None else f"_{tag}"
    stem = f"{index:04d}_{source_path.stem}{suffix}"
    return {
        "mono_wav": out_dir / f"{stem}_mono.wav",
        "stereo_wav": out_dir / f"{stem}_stereo.wav",
        "text": out_dir / f"{stem}.txt",
    }


def save_batch_outputs(
    rows: list[dict[str, str]],
    files: list[Path],
    input_wavs: list[Any],
    outputs: list[dict[str, Any]],
    sample_rate: int,
    out_dir: Path,
    cfg: GenerationConfig,
    start_index: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for local_index, (row, source_path, in_wav, output) in enumerate(
        zip(rows, files, input_wavs, outputs)
    ):
        global_index = start_index + local_index
        out_wav = output.get("wav")
        out_text = str(output["text"])
        paths = output_paths(out_dir, global_index, source_path, cfg.tag)
        mono_wav = ""
        stereo_wav = ""
        if out_wav is not None:
            stereo_audio = stack_and_pad_audio([in_wav, out_wav]).squeeze()
            sphn.write_wav(paths["mono_wav"], out_wav.numpy(), sample_rate)
            sphn.write_wav(paths["stereo_wav"], stereo_audio.numpy(), sample_rate)
            mono_wav = repo_display_path(paths["mono_wav"])
            stereo_wav = repo_display_path(paths["stereo_wav"])
        paths["text"].write_text(out_text, encoding="utf-8")
        records.append(
            {
                "id": row[cfg.id_column],
                EVAL_SET: row[EVAL_SET],
                SOURCE: row[SOURCE],
                REFERENCE: row[REFERENCE],
                "prediction_text": out_text,
                "eos_found": bool(output["eos_found"]),
                "generated_text_tokens": int(output["generated_text_tokens"]),
                "end_offset_s": output["end_offset_s"],
                "mono_wav": mono_wav,
                "stereo_wav": stereo_wav,
                "text_file": repo_display_path(paths["text"]),
            }
        )
    return records


@dataclass(frozen=True)
class GenerationConfig:
    """How free-running evaluation reads its rows, how long it generates and how it samples.
    The defaults are eval.py's, and train.py selects checkpoints with them."""

    id_column: str = "id"
    gen_duration: float = 0.0  # seconds; 0 generates the batch's longest source plus tail_s
    # The translation runs seconds behind the source: the text EOS came up to 14 s after a FLEURS
    # row ended, and 8 s cut a third of them.
    tail_s: float = 20.0
    stop_on_eos: bool = True
    text_only: bool = False  # skip decoding and writing the generated audio
    tag: str | None = None  # appended to every output file's name
    # The paper samples every token at temperature 0.8, top-k 250 (Sec. 4.5), as the checkpoint's
    # lm_gen_config does. A text temperature of 0.4 sharpens each frame's choice toward PAD, so the
    # model falls behind, stalls and seldom ends.
    audio_temp: float = 0.8
    text_temp: float = 0.8
    top_k: int = 250
    top_k_text: int = 250


# How the trainers validate: eval.py's defaults, text only (checkpoints are selected on the text).
TRAINER_GENERATION = GenerationConfig(text_only=True)


def read_source(path: Path, sample_rate: int) -> torch.Tensor:
    """A source clip at Mimi's rate (stereo averaged to mono, where the runtimes take the first
    channel), through the runtimes' level stage (inference/hibiki_mlx/level.py) as a stream of
    its own; [1, samples]."""
    wav = audio_read(path, to_sample_rate=sample_rate, mono=True)[0]
    try:
        return torch.from_numpy(Leveller()(wav[0].numpy()))[None]
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def generate_batch(
    rows: list[dict[str, str]],
    batch_start: int,
    cfg: GenerationConfig,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
) -> list[dict[str, str]]:
    """Sampled autoregressive decode of a batch of source clips."""
    files = [resolve_repo_path(row[SOURCE]) for row in rows]
    input_wavs = [read_source(path, mimi.sample_rate) for path in files]
    audio_durations = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    gen_duration = cfg.gen_duration if cfg.gen_duration else max(audio_durations) + cfg.tail_s
    if max(audio_durations) > gen_duration:
        raise RuntimeError(f"Source audio is longer than generation duration: {max(audio_durations)}")

    batch_wavs = stack_and_pad_audio(input_wavs, max_len=int(gen_duration * mimi.sample_rate))
    lm_gen = get_lmgen(
        lm,
        checkpoint_info,
        batch_size=batch_wavs.shape[0],
        temp=cfg.audio_temp,
        temp_text=cfg.text_temp,
        top_k=cfg.top_k,
        top_k_text=cfg.top_k_text,
    )
    codes, warmup_codes = encode_inputs(batch_wavs, mimi, lm_gen, audio_durations)

    output_text_tokens: list[Any] = []
    output_audio_tokens: list[Any] = []
    finished = torch.zeros(batch_wavs.shape[0], dtype=torch.bool, device=codes.device)
    eos_id = int(text_tokenizer.eos_id())
    # moshi sizes the main transformer's KV cache to its whole context (3,000 frames) and attends
    # over every slot at each step, written or not, while a batch takes only its own steps. A
    # cache of those steps gives the same attention (unwritten slots are masked out) and, at batch
    # 64 on the Mac, is 15 GiB smaller and steps faster.
    set_attention_context(lm.transformer, min(warmup_codes.shape[-1] + codes.shape[-1], lm.context))
    start_time = time.time()
    try:
        with torch.no_grad(), lm_gen.streaming(batch_wavs.shape[0]):
            for step in range(warmup_codes.shape[-1]):
                _ = lm_gen.step(warmup_codes[:, :, step : step + 1])
            for step in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, step : step + 1])
                if tokens is None:
                    continue
                output_text_tokens.append(tokens[:, 0, :])
                output_audio_tokens.append(tokens[:, 1:, :])
                finished |= tokens[:, 0, 0] == eos_id
                if cfg.stop_on_eos and bool(finished.all().detach().cpu()):
                    break
    finally:
        set_attention_context(lm.transformer, lm.context)
    elapsed = time.time() - start_time
    print(
        f"Generated batch {batch_start}-{batch_start + len(rows) - 1} "
        f"in {elapsed:.1f}s ({gen_duration / elapsed:.2f}x RT)"
    )

    if not output_text_tokens:
        raise RuntimeError("LM generation produced no output tokens; increase --gen-duration")
    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    text_outputs = decode_text_batch(batch_text_tokens, text_tokenizer)
    # How far the last generated word's text token lands after the source audio ends: the
    # paper's End Offset, measured to the end of the audio instead of its last word. Output
    # index i is the token sampled at step i, which is where the saved audio places it; the
    # model chose it having heard (i - 1) frames of source.
    for row, t, duration in zip(rows, text_outputs, audio_durations, strict=True):
        frame = t["last_word_frame"]
        t["end_offset_s"] = "" if frame is None else round(frame / mimi.frame_rate - duration, 2)
        if not t["eos_found"]:
            print(f"warning: no output EOS for {row[cfg.id_column]}; keeping all generated text.")
    if cfg.text_only:
        wavs = [None] * len(text_outputs)
    else:
        end_frames = [t["end_frame"] for t in text_outputs]
        wavs = decode_wavs(torch.concat(output_audio_tokens, dim=-1), end_frames, mimi)
    outputs = [{**t, "wav": wav} for t, wav in zip(text_outputs, wavs, strict=True)]
    return save_batch_outputs(
        rows, files, input_wavs, outputs, mimi.sample_rate, out_dir, cfg, batch_start
    )


def translate_rows(
    rows: list[dict[str, str]],
    cfg: GenerationConfig,
    batch_size: int,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
    evaluation_seed: int | None = None,
) -> list[dict[str, str]]:
    """Free-running translation of rows with `duration_s`, sampled as `cfg` says, in length order
    in batches of `batch_size`; batch i is reseeded with `evaluation_seed` + i. A row's sample
    therefore depends on the rows it shares a batch with: scores compare under one row list, seed
    and batch size."""
    # Batches of similar length pad less: on FLEURS dev, 6,000 generation steps instead of 8,300.
    rows = sorted(rows, key=lambda row: float(row["duration_s"]))
    records: list[dict[str, str]] = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        if evaluation_seed is not None:
            seed_all(evaluation_seed + batch_index)
        records.extend(
            generate_batch(
                rows[start : start + batch_size], start, cfg, mimi, lm, text_tokenizer,
                checkpoint_info, out_dir,
            )
        )
        if lm.device.type == "mps":
            # PyTorch's MPS cache keeps a finished batch's blocks without reusing them: it grew
            # 5 GiB a batch of 16 and swapped a 48 GB Mac while live tensors stayed at 6.3 GiB.
            torch.mps.empty_cache()
    return records


def generation_health(metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Gates against broken output: empty, looping or overlong. A tenth of a set's rows may be
    empty: a model that leaves 2-5% of short clips silent is not broken, and a tighter gate (8
    of 180 until 2026-11-10) failed on the sampling seed alone. The EOS count is reported, not
    gated: every checkpoint of the stream runs missed a 91% EOS gate (at best 268 of 361 FLEURS
    dev rows) while their chrF rose, so it blocked every promotion."""
    count = int(metrics["num_predictions"])
    gates = {
        "min_nonempty_predictions": count - count // 10,
        "max_repeated_4gram_predictions": math.floor(count * 12 / 128),
        "max_mean_length_ratio": 2.0,
    }
    eligible = (
        int(metrics["nonempty_predictions"]) >= gates["min_nonempty_predictions"]
        and int(metrics["repeated_4gram_predictions"])
        <= gates["max_repeated_4gram_predictions"]
        and float(metrics["mean_length_ratio"]) <= gates["max_mean_length_ratio"]
    )
    return eligible, gates


def score_sets(records: list[dict[str, str]], names: list[str]) -> dict[str, dict[str, Any]]:
    """Metrics and health gates of each set in `names` that has records, in that order. Sets are
    scored apart: corpus chrF and BLEU weigh rows by length, and the gates must hold on each."""
    scores: dict[str, dict[str, Any]] = {}
    for name in names:
        group = [record for record in records if record[EVAL_SET] == name]
        if not group:
            continue
        metrics = score_records(group)
        healthy, gates = generation_health(metrics)
        metrics.update(health_eligible=healthy, health_gates=gates)
        scores[name] = metrics
    return scores


def score_summary(metrics: dict[str, Any]) -> str:
    """One line of a set's metrics, as eval.py and train.py print them."""
    count = metrics["num_predictions"]
    return (
        f"chrf={metrics['chrf']:.2f} bleu={metrics['bleu']:.2f} "
        f"length_ratio={metrics['mean_length_ratio']:.2f} eos={metrics['eos_found']}/{count} "
        f"empty={metrics['empty_predictions']} repeat4={metrics['repeated_4gram_predictions']} "
        f"end_offset={metrics['median_end_offset_s']:.1f}s "
        f"health={'pass' if metrics['health_eligible'] else 'fail'}"
    )


def write_predictions(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", EVAL_SET, "source_audio", "reference_text", "prediction_text", "eos_found",
                "generated_text_tokens", "end_offset_s", "mono_wav", "stereo_wav", "text_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def validate_free_running(
    lm: Any,
    checkpoint_info: Any,
    mimi: Any,
    text_tokenizer: Any,
    rows: list[dict[str, str]],
    set_names: list[str],
    batch_size: int,
    out_dir: Path,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """A trainer's free-running validation of its policy `lm` (SFT's and GRPO's): translate
    `rows` on a bf16 copy with eval.py's sampling, write `out_dir`/predictions.csv and return each
    set's metrics and health gates (`score_sets`). The copy is freed before returning, so the
    next training step has its memory back."""
    generator = bf16_copy(checkpoint_info, lm, lm.device)
    try:
        records = translate_rows(
            rows, TRAINER_GENERATION, batch_size, mimi, generator, text_tokenizer,
            checkpoint_info, out_dir, seed,
        )
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
    write_predictions(out_dir / "predictions.csv", records)
    return score_sets(records, set_names)


# --------------------------------------------------------------------------- #
# Model loading (shared by every script)
# --------------------------------------------------------------------------- #
def load_checkpoint_info(args: argparse.Namespace) -> Any:
    return loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
