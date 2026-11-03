"""Shared training and evaluation logic for Vietnamese full-model SFT.

This module owns device helpers, exact full-model checkpoint I/O, teacher-forced
losses, autoregressive generation, text metrics, and correct-source generation
health diagnostics. Training streams are built in `streams.py`.

It runs on CUDA in the environment that `training/h100.sh setup` builds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from moshi.run_inference import get_condition_tensors
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from training.finetune import lm_training
from training.finetune.hibiki_helpers import (
    audio_read,
    decode_outputs,
    encode_inputs,
    get_lmgen,
    stack_and_pad_audio,
)
from training.paths import repo_display_path, require_file, resolve_repo_path


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


# --------------------------------------------------------------------------- #
# Full-model checkpoint I/O
# --------------------------------------------------------------------------- #
def enable_full_finetune(model: Any) -> None:
    """Train every language-model parameter."""
    for param in model.parameters():
        param.requires_grad_(True)


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def save_state(state: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    """Write a model's state dict (host tensors) atomically as safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        save_file(state, str(temp_path), metadata=metadata)
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
    state = {
        name: tensor.to(dtype=dtype) if tensor.dtype.is_floating_point else tensor
        for name, tensor in state.items()
    }
    model.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} model tensors from {repo_display_path(path)}")


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


def compute_batch_losses(
    lm: Any,
    codes: Any,
    condition_tensors: Any | None,
    weights: LossWeights,
    return_output: bool = False,
) -> dict[str, Any]:
    """Teacher-forced losses of one batch, with every component's sum and count beside them so
    evaluation can aggregate exactly. Positions holding -1 are outside the masks."""
    output = lm(codes, condition_tensors=condition_tensors)
    audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q][output.mask]
    audio_ce = torch.nn.functional.cross_entropy(
        output.logits[output.mask].float(), audio_targets, reduction="none"
    )
    codebook = torch.arange(lm.dep_q, device=codes.device)[None, :, None].expand_as(output.mask)
    semantic = codebook[output.mask] == 0
    audio_weight = torch.where(semantic, weights.first_codebook, 1.0)
    text_targets = codes[:, :1][output.text_mask]
    text_ce = torch.nn.functional.cross_entropy(
        output.text_logits[output.text_mask].float(), text_targets, reduction="none"
    )
    pad = text_targets == lm.text_padding_token_id
    text_weight = torch.where(pad, weights.text_pad, 1.0)
    # Sums and counts stay on the GPU: no host sync mid-step.
    sums = {
        "audio_sum": (audio_weight * audio_ce).sum(),
        "audio_n": audio_weight.sum(),
        "semantic_sum": audio_ce[semantic].sum(),
        "semantic_n": semantic.sum(),
        "acoustic_sum": audio_ce[~semantic].sum(),
        "acoustic_n": (~semantic).sum(),
        "text_sum": (text_weight * text_ce).sum(),
        "text_n": text_weight.sum(),
        "content_sum": text_ce[~pad].sum(),
        "content_n": (~pad).sum(),
        "pad_sum": text_ce[pad].sum(),
        "pad_n": pad.sum(),
    }
    result = {**sums, **loss_means(sums, weights)}
    if return_output:
        # Only for evaluation: keeping logits alive an extra step in the train loop costs
        # ~1 GB on a full-VRAM box.
        result["output"] = output
    return result


def evaluate_teacher_forced(
    lm: Any,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    weights: LossWeights,
    max_batches: int = 0,
) -> dict[str, float | int]:
    """Corpus-level teacher-forced losses and text accuracies over `dataloader`."""
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    totals = {f"{name}_{part}": 0.0 for name in LOSS_SUMS for part in ("sum", "n")}
    counts = {"batches": 0, "streams": 0, "max_frames": 0, "content_correct": 0, "pad_correct": 0}
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches and batch_index >= max_batches:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(lm, model_type, batch_size)
            result = compute_batch_losses(
                lm, codes, condition_cache[batch_size], weights, return_output=True
            )
            output = result.pop("output")
            for key in totals:
                totals[key] += float(result[key])
            targets = codes[:, :1][output.text_mask]
            correct = output.text_logits[output.text_mask].argmax(-1) == targets
            pad = targets == lm.text_padding_token_id
            counts["content_correct"] += int(correct[~pad].sum())
            counts["pad_correct"] += int(correct[pad].sum())
            counts["batches"] += 1
            counts["streams"] += batch_size
            counts["max_frames"] = max(counts["max_frames"], int(batch["frames"].max()))
    if was_training:
        lm.train()
    if not counts["batches"]:
        raise RuntimeError("Validation dataloader produced no batches.")
    content_tokens = int(totals["content_n"])
    pad_tokens = int(totals["pad_n"])
    return {
        **loss_means(totals, weights),
        "audio_tokens": int(totals["semantic_n"] + totals["acoustic_n"]),
        "content_tokens": content_tokens,
        "content_acc": counts["content_correct"] / content_tokens if content_tokens else 0.0,
        "pad_tokens": pad_tokens,
        "pad_acc": counts["pad_correct"] / pad_tokens if pad_tokens else 0.0,
        "batches": counts["batches"],
        "streams": counts["streams"],
        "max_frames": counts["max_frames"],
    }


# --------------------------------------------------------------------------- #
# Eval row IO + selection
# --------------------------------------------------------------------------- #
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


def validate_eval_rows(
    rows: list[dict[str, str]],
    source_column: str,
    reference_column: str,
    id_column: str,
    duration_column: str | None = None,
) -> None:
    if not rows:
        return
    required = [source_column, reference_column, id_column]
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
            duration = float(row[duration_column])
            if duration <= 0:
                raise ValueError(
                    f"Eval duration must be positive for {id_column}={row[id_column]}"
                )


def select_eval_rows(
    rows: list[dict[str, str]], ids: list[str], id_column: str, limit: int
) -> list[dict[str, str]]:
    if not ids:
        return rows[:limit]
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
# Text metrics (autoregressive greedy eval)
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
# Autoregressive greedy generation
# --------------------------------------------------------------------------- #
def decode_text_batch(batch_text_tokens: Any, text_tokenizer: Any, warn: bool) -> list[dict[str, Any]]:
    eos_id = int(text_tokenizer.eos_id())
    pad_id = int(text_tokenizer.pad_id())
    decoded: list[dict[str, Any]] = []
    for output_idx in range(batch_text_tokens.shape[0]):
        text_tokens: list[int] = batch_text_tokens[output_idx].tolist()
        if eos_id in text_tokens:
            eos_idx = text_tokens.index(eos_id)
            eos_found = True
        else:
            if warn:
                print(
                    "warning: model did not generate output EOS token for "
                    f"entry {output_idx}; truncating text after the last non-pad token."
                )
            eos_idx = len(text_tokens) - 1
            while eos_idx > 0 and text_tokens[eos_idx] == pad_id:
                eos_idx -= 1
            eos_found = False
        content_tokens = [token for token in text_tokens[:eos_idx] if token > pad_id]
        decoded.append(
            {
                "text": text_tokenizer.decode(content_tokens),
                "eos_found": eos_found,
                "generated_text_tokens": len(content_tokens),
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
                "source_audio": row[cfg.source_column],
                "reference_text": row[cfg.reference_column],
                "prediction_text": out_text,
                "eos_found": bool(output["eos_found"]),
                "generated_text_tokens": int(output["generated_text_tokens"]),
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

    source_column: str = "vi_audio"
    reference_column: str = "text_en"
    id_column: str = "id"
    gen_duration: float = 0.0  # seconds; 0 generates the batch's longest source plus tail_s
    tail_s: float = 8.0
    stop_on_eos: bool = True
    text_only: bool = False  # skip decoding and writing the generated audio
    tag: str | None = None  # appended to every output file's name
    audio_temp: float = 0.8
    text_temp: float = 0.4
    top_k: int = 250
    top_k_text: int = 250


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
    files = [resolve_repo_path(row[cfg.source_column]) for row in rows]
    input_wavs = [audio_read(path, to_sample_rate=mimi.sample_rate, mono=True)[0] for path in files]
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
            finished |= tokens[:, 0, 0] == eos_id
            if cfg.stop_on_eos and bool(finished.all().detach().cpu()):
                break
    elapsed = time.time() - start_time
    print(
        f"Generated batch {batch_start}-{batch_start + len(rows) - 1} "
        f"in {elapsed:.1f}s ({gen_duration / elapsed:.2f}x RT)"
    )

    if not output_text_tokens:
        raise RuntimeError("LM generation produced no output tokens; increase --gen-duration")
    batch_text_tokens = torch.concat(output_text_tokens, dim=-1)
    text_outputs = decode_text_batch(batch_text_tokens, text_tokenizer, warn=cfg.text_only)
    if cfg.text_only:
        outputs = [
            {
                "wav": None,
                "text": t["text"],
                "eos_found": t["eos_found"],
                "generated_text_tokens": t["generated_text_tokens"],
            }
            for t in text_outputs
        ]
    else:
        batch_codes = torch.concat(output_audio_tokens, dim=-1)
        decoded_outputs = decode_outputs(batch_codes, batch_text_tokens, mimi, text_tokenizer)
        outputs = [
            {
                "wav": wav,
                "text": text,
                "eos_found": t["eos_found"],
                "generated_text_tokens": t["generated_text_tokens"],
            }
            for (wav, text), t in zip(decoded_outputs, text_outputs, strict=True)
        ]
    return save_batch_outputs(
        rows, files, input_wavs, outputs, mimi.sample_rate, out_dir, cfg, batch_start
    )


def run_greedy_eval(
    rows: list[dict[str, str]],
    cfg: GenerationConfig,
    batch_size: int,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
    evaluation_seed: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Batched free-running eval; optionally reseed each batch deterministically."""
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
    return records, score_records(records)


def generation_health(metrics: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    count = int(metrics["num_predictions"])
    gates = {
        "min_nonempty_predictions": math.ceil(count * 122 / 128),
        "min_eos_found": math.ceil(count * 116 / 128),
        "max_repeated_4gram_predictions": math.floor(count * 12 / 128),
        "max_mean_length_ratio": 2.0,
    }
    eligible = (
        int(metrics["nonempty_predictions"]) >= gates["min_nonempty_predictions"]
        and int(metrics["eos_found"]) >= gates["min_eos_found"]
        and int(metrics["repeated_4gram_predictions"])
        <= gates["max_repeated_4gram_predictions"]
        and float(metrics["mean_length_ratio"]) <= gates["max_mean_length_ratio"]
    )
    return eligible, gates


def write_predictions(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "source_audio", "reference_text", "prediction_text", "eos_found",
                "generated_text_tokens", "mono_wav", "stereo_wav", "text_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


# --------------------------------------------------------------------------- #
# Model loading (shared by all three scripts)
# --------------------------------------------------------------------------- #
def load_checkpoint_info(args: argparse.Namespace) -> Any:
    return loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
