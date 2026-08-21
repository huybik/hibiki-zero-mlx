#!/usr/bin/env python
"""Capture strict pre-undelay AR targets for CUDA parallel-head training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from moshi.models import loaders
from moshi.models.lm_utils import _delay_sequence
from moshi.run_inference import get_condition_tensors

from student.cache import DISTILL_ROLE, artifact_identity, exact_keys, load_cache
from student.contract import DEFAULT_CONFIG, sha256, torch_lm_config
from student.harness import canonical_sha256
from student.parallel import validate_ar_checkpoint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIMI = ROOT / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
DEFAULT_TOKENIZER = ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"
PARALLEL_CACHE_FORMAT = "hibiki_parallel_capture_v1"
PARALLEL_METADATA_FORMAT = "hibiki_parallel_capture_metadata_v1"
PARALLEL_ROLE = "ar_parallel_distillation"
SAMPLE_FIELDS = {
    "id",
    "split",
    "hidden",
    "text_ids",
    "previous_codes",
    "hard_targets",
    "teacher_topk_ids",
    "teacher_topk_logprobs",
    "teacher_mask",
}


def input_cache_identity(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    shards = [
        {"name": path.name, "sha256": sha256(path)} for path in sorted(cache_dir.glob("shard_*.pt"))
    ]
    return {
        "role": DISTILL_ROLE,
        "metadata_sha256": canonical_sha256(metadata),
        "shards": shards,
    }


def make_metadata(
    cfg: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    source_metadata: dict[str, Any],
    source_identity: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    return {
        "format": PARALLEL_METADATA_FORMAT,
        "role": PARALLEL_ROLE,
        "architecture": cfg["architecture"],
        "layout": {
            "timeline": "ar_depformer_pattern_pre_undelay",
            "hidden_dim": cfg["dim"],
            "codebooks": cfg["dep_q"],
            "card": cfg["card"],
        },
        "backbone": {
            "config": cfg,
            "config_artifact": artifact_identity(config_path),
            "checkpoint_artifact": artifact_identity(checkpoint_path),
        },
        "mimi": source_metadata["mimi"],
        "tokenizer": source_metadata["tokenizer"],
        "input_cache": source_identity,
        "generation": {
            "method": "frozen_ar_pattern_topk_v1",
            "top_k": top_k,
            "text_condition": "current_pattern_text_argmax",
            "previous_codes": "initial_card_then_previous_pattern_target_frame",
            "audio_teacher": "frozen_12l_ar_depformer",
            "excluded": "3b_audio_logits",
        },
    }


def validate_metadata(metadata: Any) -> None:
    exact_keys(
        metadata,
        {
            "format",
            "role",
            "architecture",
            "layout",
            "backbone",
            "mimi",
            "tokenizer",
            "input_cache",
            "generation",
        },
        "parallel cache metadata",
    )
    if metadata["format"] != PARALLEL_METADATA_FORMAT or metadata["role"] != PARALLEL_ROLE:
        raise RuntimeError("Unsupported parallel capture metadata")
    cfg = metadata["backbone"].get("config")
    if not isinstance(cfg, dict) or cfg.get("head") != "ar" or cfg.get("num_layers") != 12:
        raise RuntimeError("Parallel cache backbone is not the frozen 12-layer AR student")
    expected_layout = {
        "timeline": "ar_depformer_pattern_pre_undelay",
        "hidden_dim": 2048,
        "codebooks": 8,
        "card": 2048,
    }
    if metadata["layout"] != expected_layout:
        raise RuntimeError("Parallel capture layout changed")
    exact_keys(
        metadata["backbone"],
        {"config", "config_artifact", "checkpoint_artifact"},
        "parallel cache backbone",
    )
    for key in ("config_artifact", "checkpoint_artifact"):
        exact_keys(metadata["backbone"][key], {"name", "sha256"}, key)
    for key in ("mimi", "tokenizer"):
        exact_keys(metadata[key], {"name", "sha256"}, key)
    exact_keys(metadata["input_cache"], {"role", "metadata_sha256", "shards"}, "input cache")
    if metadata["input_cache"]["role"] != DISTILL_ROLE:
        raise RuntimeError("Parallel capture did not consume the strict student distill cache")
    if (
        not isinstance(metadata["input_cache"]["shards"], list)
        or not metadata["input_cache"]["shards"]
    ):
        raise RuntimeError("Parallel capture lacks input shard hashes")
    generation = metadata["generation"]
    exact_keys(
        generation,
        {"method", "top_k", "text_condition", "previous_codes", "audio_teacher", "excluded"},
        "parallel target generation",
    )
    if (
        generation["method"] != "frozen_ar_pattern_topk_v1"
        or isinstance(generation["top_k"], bool)
        or not isinstance(generation["top_k"], int)
        or not 0 < generation["top_k"] <= 2048
        or generation["text_condition"] != "current_pattern_text_argmax"
        or generation["previous_codes"] != "initial_card_then_previous_pattern_target_frame"
        or generation["audio_teacher"] != "frozen_12l_ar_depformer"
        or generation["excluded"] != "3b_audio_logits"
    ):
        raise RuntimeError("Unsupported parallel target generation")


def validate_sample(sample: Any, metadata: dict[str, Any], location: str) -> None:
    exact_keys(sample, SAMPLE_FIELDS, location)
    if not all(isinstance(sample[key], str) and sample[key] for key in ("id", "split")):
        raise RuntimeError(f"{location} has an empty id or split")
    hidden = sample["hidden"]
    text_ids = sample["text_ids"]
    previous = sample["previous_codes"]
    targets = sample["hard_targets"]
    ids = sample["teacher_topk_ids"]
    values = sample["teacher_topk_logprobs"]
    mask = sample["teacher_mask"]
    integers = (torch.int16, torch.int32, torch.int64)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[1] != 2048:
        raise RuntimeError(f"{location} hidden must be [T, 2048]")
    frames = int(hidden.shape[0])
    top_k = int(metadata["generation"]["top_k"])
    if not hidden.is_floating_point() or not torch.isfinite(hidden.float()).all():
        raise RuntimeError(f"{location} hidden is not finite floating point")
    if (
        not isinstance(text_ids, torch.Tensor)
        or text_ids.dtype not in integers
        or tuple(text_ids.shape) != (frames,)
        or (text_ids < 0).any()
        or (text_ids >= 48_000).any()
    ):
        raise RuntimeError(f"{location} text IDs are invalid")
    for tensor, label in ((previous, "previous codes"), (targets, "hard targets")):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype not in integers
            or tuple(tensor.shape) != (frames, 8)
            or (tensor < 0).any()
            or (tensor > 2048).any()
        ):
            raise RuntimeError(f"{location} {label} must be [T, 8] in [0, 2048]")
    if frames == 0 or not torch.equal(previous[0], torch.full((8,), 2048, dtype=previous.dtype)):
        raise RuntimeError(f"{location} previous codes do not start with the initial card")
    if frames > 1 and not torch.equal(previous[1:], targets[:-1]):
        raise RuntimeError(f"{location} previous codes are not the prior pattern target frame")
    if (
        not isinstance(ids, torch.Tensor)
        or ids.dtype not in integers
        or tuple(ids.shape) != (frames, 8, top_k)
        or not isinstance(values, torch.Tensor)
        or not values.is_floating_point()
        or tuple(values.shape) != tuple(ids.shape)
        or not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (frames, 8)
        or not torch.equal(mask, targets < 2048)
    ):
        raise RuntimeError(f"{location} teacher targets are malformed or misaligned")
    if (ids[~mask] != -1).any() or (values[~mask] != 0).any():
        raise RuntimeError(f"{location} unmasked teacher storage is not empty")
    valid_ids = ids[mask]
    valid_values = values[mask].float()
    sorted_ids = valid_ids.sort(-1).values
    if valid_ids.numel() and (
        (valid_ids < 0).any()
        or (valid_ids >= 2048).any()
        or not torch.isfinite(valid_values).all()
        or (valid_values > 1e-3).any()
        or (valid_values.exp().sum(-1) > 1.001).any()
        or (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()
    ):
        raise RuntimeError(f"{location} top-k values are not an absolute distribution")


def save_shard(path: Path, metadata: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {"format": PARALLEL_CACHE_FORMAT, "metadata": metadata, "samples": samples},
        temporary,
    )
    temporary.replace(path)


def load_parallel_cache(
    cache_dir: Path,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    paths = sorted(cache_dir.glob("shard_*.pt"))
    if not paths:
        raise RuntimeError(f"No shard_*.pt files in {cache_dir}")
    metadata = None
    shards = []
    seen: set[str] = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        exact_keys(payload, {"format", "metadata", "samples"}, str(path))
        if payload["format"] != PARALLEL_CACHE_FORMAT:
            raise RuntimeError(f"Unsupported parallel cache format in {path}")
        validate_metadata(payload["metadata"])
        if metadata is not None and payload["metadata"] != metadata:
            raise RuntimeError(f"Parallel cache metadata differs in {path}")
        metadata = payload["metadata"]
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise RuntimeError(f"Empty parallel cache shard: {path}")
        for index, sample in enumerate(payload["samples"]):
            validate_sample(sample, metadata, f"{path}:{index}")
            if sample["id"] in seen:
                raise RuntimeError(f"Duplicate parallel cache id={sample['id']!r}")
            seen.add(sample["id"])
        shards.append((path, payload))
    return metadata, shards


def condition_parts(model: Any, cfg: dict[str, Any]) -> tuple[Any | None, Any | None]:
    if model.fuser is None:
        return None, None
    tensors = get_condition_tensors(
        str(cfg.get("model_type", "hibiki")), model, batch_size=1, cfg_coef=1.0
    )
    return model.fuser.get_sum(tensors), model.fuser.get_cross(tensors)


def capture_sample(
    model: Any,
    sample: dict[str, Any],
    top_k: int,
    sum_condition: Any | None,
    cross_condition: Any | None,
) -> dict[str, Any]:
    codes = sample["codes"][None].to(device="cuda", dtype=torch.long)
    frames = int(sample["target_frames"])
    initial = model._get_initial_token().expand(codes.shape[0], -1, -1)
    delayed = torch.cat([initial, _delay_sequence(model.delays, codes, initial)], dim=2)
    captured: list[torch.Tensor] = []
    hook = model.out_norm.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, text_logits = model.forward_text(
                delayed[:, :, :-1], sum_condition, cross_condition
            )
            text_ids = text_logits[:, 0].argmax(dim=-1)
            pattern_sequence = delayed[:, :, 1:].clone()
            pattern_sequence[:, 0] = text_ids
            pattern_logits = model.forward_depformer_training(pattern_sequence, hidden)
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0] is not hidden:
        raise RuntimeError("Failed to hook the normalized main hidden state")

    hidden = hidden[0, :frames]
    text_ids = text_ids[0, :frames]
    targets = delayed[0, model.audio_offset : model.audio_offset + model.dep_q, 1 : frames + 1]
    targets = targets.transpose(0, 1)
    if (targets < 0).any() or (targets > model.card).any():
        raise RuntimeError(f"id={sample['id']} pattern targets are outside [0, card]")
    mask = targets < model.card
    previous = torch.full_like(targets, model.card)
    previous[1:] = targets[:-1]

    logprobs = pattern_logits[0, :, :frames].transpose(0, 1).float().log_softmax(dim=-1)
    values, ids = logprobs.topk(top_k, dim=-1)
    ids = ids.to(dtype=torch.int32)
    values = values.to(dtype=torch.float16)
    ids[~mask] = -1
    values[~mask] = 0
    result = {
        "id": sample["id"],
        "split": sample["split"],
        "hidden": hidden.to(device="cpu", dtype=torch.bfloat16),
        "text_ids": text_ids.to(device="cpu", dtype=torch.int32),
        "previous_codes": previous.to(device="cpu", dtype=torch.int32),
        "hard_targets": targets.to(device="cpu", dtype=torch.int32),
        "teacher_topk_ids": ids.cpu(),
        "teacher_topk_logprobs": values.cpu(),
        "teacher_mask": mask.cpu(),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--ar-sha256", required=True)
    parser.add_argument("--mimi", type=Path, default=DEFAULT_MIMI)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel-head capture requires CUDA")
    if not 0 < args.top_k <= 2048:
        raise ValueError("--top-k must be in [1, 2048]")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite captures in {args.out_dir}")

    cfg = validate_ar_checkpoint(
        args.config,
        args.ar_checkpoint,
        args.ar_sha256,
    )
    source_metadata, source_shards = load_cache(args.cache_dir, DISTILL_ROLE)
    if source_metadata["model"]["config"] != cfg:
        raise RuntimeError("Student distill cache config differs from the frozen AR config")
    if source_metadata["model"]["config_artifact"] != artifact_identity(args.config):
        raise RuntimeError("Student distill cache config hash differs from --config")
    for key, path in (("mimi", args.mimi), ("tokenizer", args.tokenizer)):
        if source_metadata[key] != artifact_identity(path):
            raise RuntimeError(f"Student distill cache {key} hash differs from {path}")
    if source_metadata["generation"]["audio_heads"] != "unused":
        raise RuntimeError("Student distill input unexpectedly contains 3B audio-head targets")

    source_identity = input_cache_identity(args.cache_dir, source_metadata)
    metadata = make_metadata(
        cfg,
        args.config,
        args.ar_checkpoint,
        source_metadata,
        source_identity,
        args.top_k,
    )
    validate_metadata(metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = loaders.get_moshi_lm(
        args.ar_checkpoint,
        lm_kwargs=torch_lm_config(cfg),
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    model.requires_grad_(False)
    sum_condition, cross_condition = condition_parts(model, cfg)
    captured_count = 0
    for source_path, payload in source_shards:
        samples = [
            capture_sample(model, sample, args.top_k, sum_condition, cross_condition)
            for sample in payload["samples"]
        ]
        for sample in samples:
            validate_sample(sample, metadata, sample["id"])
        save_shard(args.out_dir / source_path.name, metadata, samples)
        captured_count += len(samples)
        print(f"Wrote {len(samples)} pattern-aligned samples -> {args.out_dir / source_path.name}")

    load_parallel_cache(args.out_dir)
    print(f"PASS: captured {captured_count} frozen-AR parallel-head samples")


if __name__ == "__main__":
    main()
