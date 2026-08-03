"""Shared training/eval logic for the Vietnamese LoRA finetune stack.

This module owns everything that used to be duplicated across train_lora.py /
eval_lora.py / validate_lora.py: device+dtype helpers, the cached-shard dataset
and loader, LoRA insertion and adapter load/save, teacher-forced losses,
autoregressive greedy generation + text metrics, and the piecewise-constant
schedule primitives used for loss-weight / replay / per-group-LR schedules.

It is a PyTorch training toolkit; torch, safetensors and the `moshi` pip package
are hard dependencies imported at module top (the conda base env ships them).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch

# moshi reads NO_TORCH_COMPILE once at import. Default-enable compile on CUDA:
# torch 2.13 fixed the 2.12 autograd break (invalid gradient shape in backward)
# and it's ~10% faster (measured H100). Keep it off elsewhere (no gain on MPS).
# Override with NO_TORCH_COMPILE=1.
os.environ.setdefault("NO_TORCH_COMPILE", "" if torch.cuda.is_available() else "1")

import numpy as np
import sphn
from moshi.models import loaders
from moshi.modules.lora import replace_all_linear_with_lora
from moshi.run_inference import get_condition_tensors
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from finetune.cache_codes import CACHE_FORMAT
from finetune.hibiki_helpers import (
    audio_read,
    decode_outputs,
    encode_inputs,
    get_lmgen,
    stack_and_pad_audio,
)
from finetune.utils import repo_display_path, require_file, resolve_repo_path

# LoRA target groups (used for freeze map, save/load and per-group LR schedules).
GROUP_PREFIXES = {
    "transformer": ("transformer.",),
    "text_linear": ("text_linear.",),
    "audio_heads": ("depformer_in.", "linears."),
}
SUPPORTED_TARGETS = {"LMModel.transformer", "text_linear", "audio_heads"}


# --------------------------------------------------------------------------- #
# Device / dtype / seeding
# --------------------------------------------------------------------------- #
def check_device(device_name: str) -> torch.device:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but torch.backends.mps is not available.")
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


def is_mps(device: torch.device) -> bool:
    return getattr(device, "type", str(device)) == "mps"


def mps_memory_stats(device: torch.device) -> dict[str, float]:
    if not is_mps(device):
        return {}
    return {
        "mps_allocated_gb": torch.mps.current_allocated_memory() / 1024**3,
        "mps_driver_gb": torch.mps.driver_allocated_memory() / 1024**3,
        "mps_recommended_gb": torch.mps.recommended_max_memory() / 1024**3,
    }


def empty_device_cache(device: torch.device) -> None:
    """Synchronize + free cached memory. No-op off MPS so CUDA runs stay clean."""
    if not is_mps(device):
        return
    torch.mps.synchronize()
    torch.mps.empty_cache()


class _CausalSDPA:
    """Namespace proxy injected as moshi.modules.transformer.F (rebinds only that
    module's global, not torch.nn.functional itself).

    moshi's causal attention passes a boolean attn_bias to SDPA, which blocks the
    flash kernel and falls back to the slower mem-efficient backend. In training
    (offset 0, T << cfg.context) that mask is exactly lower-triangular causal, so
    we rewrite it to is_causal=True. The verdict is verified by tril comparison
    ONCE per mask shape then cached (re-checking every call would reintroduce a
    host sync per layer) — sound here because every same-shape bool mask moshi
    builds at offset 0 is identical, and streaming decode has T_q=1 != T_k so it
    always falls through. Do not reuse this proxy where same-shape masks differ.
    """

    def __init__(self):
        self._causal_shapes: dict[tuple[int, ...], bool] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(torch.nn.functional, name)

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, **kwargs):
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and q.shape[-2] == k.shape[-2]
            and attn_mask.shape[-2:] == (q.shape[-2], k.shape[-2])
        ):
            key = tuple(attn_mask.shape)
            causal = self._causal_shapes.get(key)
            if causal is None:
                T = q.shape[-2]
                tril = torch.ones(T, T, dtype=torch.bool, device=attn_mask.device).tril()
                causal = bool((attn_mask == tril).all())
                self._causal_shapes[key] = causal
            if causal:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, **kwargs
                )
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, **kwargs)


def enable_causal_sdpa() -> None:
    import moshi.modules.transformer as _moshi_transformer

    _moshi_transformer.F = _CausalSDPA()


# Default-on for CUDA (~1% train speedup, free); streaming decode is unaffected
# (T_q=1 falls through). MPS SDPA has no flash backend to unlock.
if torch.cuda.is_available():
    enable_causal_sdpa()


# --------------------------------------------------------------------------- #
# Schedules: piecewise-constant "value@fraction" specs
# --------------------------------------------------------------------------- #
def parse_schedule(spec: str | float | int) -> list[tuple[float, float]]:
    """Parse "5@0,2@0.6" -> [(0.0, 5.0), (0.6, 2.0)] sorted by fraction.

    A bare number ("5" or 5.0) is the degenerate single-point schedule [(0.0, 5.0)]
    so all old static flags keep working unchanged.
    """
    text = str(spec).strip()
    if not text:
        raise ValueError("Empty schedule spec")
    points: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "@" in part:
            value_s, frac_s = part.split("@", 1)
            points.append((float(frac_s), float(value_s)))
        else:
            points.append((0.0, float(part)))
    points.sort(key=lambda item: item[0])
    if not points or points[0][0] != 0.0:
        raise ValueError(f"Schedule must define a value at fraction 0: {spec!r}")
    return points


def schedule_value(points: list[tuple[float, float]], step: int, total_steps: int) -> float:
    frac = step / total_steps if total_steps > 0 else 0.0
    value = points[0][1]
    for boundary, boundary_value in points:
        if frac >= boundary:
            value = boundary_value
        else:
            break
    return value


# --------------------------------------------------------------------------- #
# Cached-shard dataset / loader / replay sampler
# --------------------------------------------------------------------------- #
class CachedCodeDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path | list[Path],
        sort_by_length: bool,
        max_samples: int,
        max_frames: int = 0,
    ):
        self.samples: list[dict[str, Any]] = []
        self.frame_rate: float | None = None
        dropped = 0
        cache_dirs = [cache_dir] if isinstance(cache_dir, Path) else list(cache_dir)
        shard_paths = [p for d in cache_dirs for p in sorted(d.glob("shard_*.pt"))]
        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            if payload.get("format") != CACHE_FORMAT:
                raise RuntimeError(f"Unsupported cache format in {shard_path}")
            frame_rate = float(payload["frame_rate"])
            if self.frame_rate is None:
                self.frame_rate = frame_rate
            elif frame_rate != self.frame_rate:
                raise RuntimeError(
                    f"Cache frame-rate mismatch in {shard_path}: {frame_rate} != {self.frame_rate}"
                )
            for sample in payload["samples"]:
                codes = sample["codes"]
                if codes.ndim != 2:
                    raise RuntimeError(f"{shard_path} id={sample.get('id')} codes must be [K,T]")
                if max_frames and codes.shape[1] > max_frames:
                    dropped += 1
                    continue
                self.samples.append(
                    {
                        "id": str(sample["id"]),
                        # int32 in host RAM (halves footprint at ~700k samples);
                        # collate_cached casts to long on batch assembly.
                        "codes": codes.to(torch.int32),
                        "frames": int(codes.shape[1]),
                        "source_frames": int(sample["vi_frames"]),
                    }
                )
        if not self.samples:
            raise RuntimeError(f"No shard_*.pt cache files found in {cache_dir}")
        if max_frames and dropped:
            print(f"[dataset] dropped {dropped} samples over {max_frames} frames; kept {len(self.samples)}")
        if sort_by_length:
            self.samples.sort(key=lambda sample: sample["frames"])
        if max_samples:
            self.samples = self.samples[:max_samples]
            if not self.samples:
                raise RuntimeError("--max-samples selected no cached samples")

    def shuffle_batch_order(self, batch_size: int, seed: int = 1234) -> None:
        """Shuffle length-sorted samples in whole-batch blocks.

        Batches stay near-uniform length (minimal padding, few MPS shapes) but
        the epoch is no longer an ascending-length curriculum replayed in the
        same order every epoch. Deterministic, so sorted-resume skip still works.
        """
        blocks = [self.samples[i : i + batch_size] for i in range(0, len(self.samples), batch_size)]
        random.Random(seed).shuffle(blocks)
        self.samples = [sample for block in blocks for sample in block]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def exposure(self) -> dict[str, float | int]:
        source_frames = sum(sample["source_frames"] for sample in self.samples)
        return {
            "samples": len(self.samples),
            "assembled_frames": sum(sample["frames"] for sample in self.samples),
            "source_frames": source_frames,
            "source_hours": source_frames / float(self.frame_rate) / 3600,
        }


def parse_frame_batch_schedule(spec: str) -> list[tuple[int, int]]:
    """Parse cumulative max-frame buckets, e.g. ``288:10,384:8,512:5``."""
    buckets: list[tuple[int, int]] = []
    for part in spec.split(","):
        fields = part.strip().split(":")
        if len(fields) != 2:
            raise ValueError("--frame-batch-schedule must use MAX_FRAMES:BATCH_SIZE entries")
        try:
            max_frames, batch_size = (int(field) for field in fields)
        except ValueError as exc:
            raise ValueError("--frame-batch-schedule values must be positive integers") from exc
        if max_frames <= 0 or batch_size <= 0:
            raise ValueError("--frame-batch-schedule values must be positive integers")
        if buckets and max_frames <= buckets[-1][0]:
            raise ValueError("--frame-batch-schedule frame limits must strictly increase")
        buckets.append((max_frames, batch_size))
    if not buckets:
        raise ValueError("--frame-batch-schedule cannot be empty")
    return buckets


class FrameBudgetBatchSampler:
    """Length-homogeneous batches shuffled only as whole blocks per epoch."""

    def __init__(
        self,
        dataset: CachedCodeDataset,
        schedule: list[tuple[int, int]],
        seed: int,
    ):
        self.dataset = dataset
        self.schedule = schedule
        self.seed = seed
        self.epoch = 0
        self.batches: list[list[int]] = []
        self.bucket_exposure: list[dict[str, float | int]] = []

        lower = 0
        for upper, batch_size in schedule:
            indices = sorted(
                (
                    index
                    for index, sample in enumerate(dataset.samples)
                    if lower < sample["frames"] <= upper
                ),
                key=lambda index: dataset.samples[index]["frames"],
            )
            batches = [
                indices[start : start + batch_size] for start in range(0, len(indices), batch_size)
            ]
            self.batches.extend(batches)
            source_frames = sum(dataset.samples[index]["source_frames"] for index in indices)
            self.bucket_exposure.append(
                {
                    "min_frames": lower + 1,
                    "max_frames": upper,
                    "batch_size": batch_size,
                    "samples": len(indices),
                    "batches": len(batches),
                    "assembled_frames": sum(dataset.samples[index]["frames"] for index in indices),
                    "source_hours": source_frames / float(dataset.frame_rate) / 3600,
                }
            )
            lower = upper
        if sum(len(batch) for batch in self.batches) != len(dataset):
            raise ValueError("--frame-batch-schedule does not cover every selected cache sample")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = list(range(len(self.batches)))
        random.Random(f"{self.seed}:{self.epoch}").shuffle(order)
        return iter(self.batches[index] for index in order)

    def __len__(self) -> int:
        return len(self.batches)


# Pad each batch's frame length up to a multiple of this. MPS compiles+caches a
# Metal kernel graph per distinct tensor shape; the raw pool has 262 distinct
# lengths, which balloons the GPU working set (26 GB wired, swap-thrash). Bucketing
# collapses that to ~9 shapes. Loss-neutral: extra frames are -1 == zero_token_id,
# masked out of both CE terms (see LMModel.forward logits_mask). CUDA doesn't
# need the shape cap — HIBIKI_FRAME_BUCKET=1 pads to the exact batch max.
FRAME_BUCKET = int(os.environ.get("HIBIKI_FRAME_BUCKET", "32"))


def collate_cached(samples: list[dict[str, Any]]) -> dict[str, Any]:
    codebooks = int(samples[0]["codes"].shape[0])
    max_frames = max(int(sample["codes"].shape[1]) for sample in samples)
    max_frames = ((max_frames + FRAME_BUCKET - 1) // FRAME_BUCKET) * FRAME_BUCKET
    batch = torch.full((len(samples), codebooks, max_frames), -1, dtype=torch.long)
    ids: list[str] = []
    for index, sample in enumerate(samples):
        codes = sample["codes"]
        batch[index, :, : codes.shape[1]] = codes
        ids.append(sample["id"])
    return {
        "codes": batch,
        "ids": ids,
        "frames": torch.tensor([sample["frames"] for sample in samples]),
        "source_frames": torch.tensor([sample["source_frames"] for sample in samples]),
    }


def parse_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def make_replay_sampler(
    dataset: CachedCodeDataset,
    replay_ids: set[str],
    replay_weight: float,
    seed: int,
) -> Any | None:
    if not replay_ids or replay_weight == 1.0:
        return None
    sample_ids = {str(sample["id"]) for sample in dataset.samples}
    missing = sorted(replay_ids - sample_ids)
    if missing:
        raise ValueError(f"--replay-ids not present in selected cache samples: {missing[:10]}")
    weights = [
        float(replay_weight) if str(sample["id"]) in replay_ids else 1.0
        for sample in dataset.samples
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_cached_dataloader(
    dataset: CachedCodeDataset,
    batch_size: int,
    num_workers: int,
    sort_by_length: bool,
    sampler: Any | None = None,
    batch_sampler: Any | None = None,
    seed: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if batch_sampler is not None:
        if sampler is not None:
            raise ValueError("sampler and batch_sampler are mutually exclusive")
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_cached,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else not sort_by_length,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_cached,
        generator=generator,
    )


# --------------------------------------------------------------------------- #
# LoRA insertion / freeze map / adapter save+load
# --------------------------------------------------------------------------- #
def group_of(name: str) -> str | None:
    for group, prefixes in GROUP_PREFIXES.items():
        if name.startswith(prefixes):
            return group
    return None


def zero_lora_updates(model: Any) -> None:
    for module in model.modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is not None:
            lora_b.weight.data.zero_()


def adapter_target(train_text_head: bool, train_audio_heads: bool) -> str:
    targets = ["LMModel.transformer"]
    if train_text_head:
        targets.append("text_linear")
    if train_audio_heads:
        targets.append("audio_heads")
    return "+".join(targets)


def apply_lora_targets(
    model: Any, lora_rank: int, lora_scaling: float, train_text_head: bool, train_audio_heads: bool
) -> None:
    """Freeze everything, insert zero-init LoRA on the selected targets, unfreeze them."""
    for param in model.parameters():
        param.requires_grad_(False)
    replace_all_linear_with_lora(model.transformer, lora_rank, lora_scaling)
    if train_audio_heads:
        replace_all_linear_with_lora(model.depformer_in, lora_rank, lora_scaling)
        replace_all_linear_with_lora(model.linears, lora_rank, lora_scaling)
    zero_lora_updates(model)
    for name, param in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        train_transformer = is_lora and name.startswith("transformer.")
        train_text = train_text_head and name.startswith("text_linear.")
        train_audio = train_audio_heads and is_lora and (
            name.startswith("depformer_in.") or name.startswith("linears.")
        )
        param.requires_grad_(train_transformer or train_text or train_audio)

    bad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and group_of(name) is None
    ]
    if bad:
        raise RuntimeError(f"LoRA freeze map leaked trainable params outside targets: {bad[:5]}")


def apply_full_finetune(model: Any) -> None:
    """Full-model SFT (paper §4.6): unfreeze every LM parameter, no LoRA.

    Faithful to how the paper adds a language (full finetune from the base
    checkpoint). Saving/loading reuses the adapter helpers: with everything
    trainable, `adapter_state_dict` captures the whole model.
    """
    for param in model.parameters():
        param.requires_grad_(True)


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def adapter_state_dict(model: Any) -> dict[str, Any]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }
    if not state:
        raise RuntimeError("No trainable adapter tensors found to save.")
    return state


def save_adapter(model: Any, adapter_path: Path, metadata: dict[str, str]) -> None:
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state_dict(model), str(adapter_path), metadata=metadata)


def load_adapter_state(model: Any, adapter_path: Path, dtype: torch.dtype) -> int:
    """Load adapter tensors into a model that already has matching LoRA modules."""
    state = load_file(str(adapter_path), device="cpu")
    for key, value in state.items():
        if value.dtype.is_floating_point:
            state[key] = value.to(dtype=dtype)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys in {adapter_path}: {result.unexpected_keys[:5]}")
    return len(state)


def adapter_metadata(adapter_path: Path) -> tuple[int, float, set[str]]:
    with safe_open(str(adapter_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    missing = [key for key in ("target", "lora_rank", "lora_scaling") if key not in metadata]
    if missing:
        raise RuntimeError(f"Adapter is missing metadata: {', '.join(missing)}")
    targets = set(metadata["target"].split("+"))
    if "LMModel.transformer" not in targets or targets - SUPPORTED_TARGETS:
        raise RuntimeError(f"Unsupported adapter target: {metadata['target']}")
    return int(metadata["lora_rank"]), float(metadata["lora_scaling"]), targets


def load_main_lora(lm: Any, adapter_path: Path, device: torch.device, dtype: torch.dtype) -> None:
    """Insert LoRA per the adapter metadata, then load its weights (for eval/validate)."""
    rank, scaling, targets = adapter_metadata(adapter_path)
    replace_all_linear_with_lora(lm.transformer, rank, scaling, device=device, dtype=dtype)
    if "audio_heads" in targets:
        replace_all_linear_with_lora(lm.depformer_in, rank, scaling, device=device, dtype=dtype)
        replace_all_linear_with_lora(lm.linears, rank, scaling, device=device, dtype=dtype)
    state = load_file(str(adapter_path), device=str(device))
    allowed = ["transformer."]
    if "text_linear" in targets:
        allowed.append("text_linear.")
    if "audio_heads" in targets:
        allowed.extend(("depformer_in.", "linears."))
    bad = [key for key in state if not key.startswith(tuple(allowed))]
    if bad:
        raise RuntimeError(f"Adapter has unsupported tensors: {bad[:5]}")
    for key, value in state.items():
        if value.dtype.is_floating_point:
            state[key] = value.to(dtype=dtype)
    result = lm.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys[:5]}")
    print(f"Loaded {len(state)} adapter tensors from {repo_display_path(adapter_path)}")


def checkpoint_is_full(path: Path) -> bool:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return (handle.metadata() or {}).get("target") == "full"


def load_finetuned(lm: Any, path: Path, device: torch.device, dtype: torch.dtype) -> None:
    """Load a checkpoint into `lm`, dispatching on its metadata target.

    Full-finetune checkpoints hold every param key, so they load straight onto
    the base model (missing keys are just buffers); LoRA adapters get the LoRA
    modules inserted first.
    """
    if checkpoint_is_full(path):
        count = load_adapter_state(lm, path, dtype)
        print(f"Loaded {count} full-finetune tensors from {repo_display_path(path)}")
    else:
        load_main_lora(lm, path, device, dtype)


# --------------------------------------------------------------------------- #
# Optimizer param groups + per-group LR scheduling
# --------------------------------------------------------------------------- #
def build_param_groups(model: Any, lr_schedules: dict[str, list[tuple[float, float]]]) -> list[dict[str, Any]]:
    """One optimizer group per LoRA target that has trainable params.

    Collapses to a single unnamed group when all three groups share the same
    schedule, so plain `--lr` runs behave exactly as before.
    """
    buckets: dict[str, list[Any]] = {name: [] for name in GROUP_PREFIXES}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = group_of(name)
        if group is None:
            raise RuntimeError(f"Unexpected trainable parameter outside groups: {name}")
        buckets[group].append(param)

    active = {name: params for name, params in buckets.items() if params}
    specs = {name: lr_schedules[name] for name in active}
    uniform = len({tuple(points) for points in specs.values()}) <= 1
    if uniform:
        merged = [param for params in active.values() for param in params]
        return [{"name": "all", "params": merged, "points": next(iter(specs.values()))}]
    return [
        {"name": name, "params": params, "points": lr_schedules[name]}
        for name, params in active.items()
    ]


def full_param_groups(model: Any, lr_points: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """One optimizer group over all trainable params (full finetune = one LR schedule)."""
    params = [param for param in model.parameters() if param.requires_grad]
    return [{"name": "all", "params": params, "points": lr_points}]


def apply_lr_schedule(
    optimizer: Any, step: int, total_steps: int, warmup_steps: int
) -> float | dict[str, float]:
    warmup = 1.0
    if warmup_steps > 0:
        warmup = min(1.0, (step + 1) / warmup_steps)
    lrs: dict[str, float] = {}
    for group in optimizer.param_groups:
        base = schedule_value(group["points"], step, total_steps)
        group["lr"] = base * warmup
        lrs[str(group.get("name", "group"))] = group["lr"]
    if len(lrs) == 1:
        return next(iter(lrs.values()))
    return lrs


# --------------------------------------------------------------------------- #
# Teacher-forced losses
# --------------------------------------------------------------------------- #
def masked_cross_entropy(logits: Any, targets: Any, mask: Any) -> tuple[Any, Any]:
    # Sum/clamp form keeps everything on-GPU: no host sync mid-forward (the old
    # int(mask.sum().cpu()) stalled the CUDA pipeline twice per micro-batch).
    # Empty mask -> sum CE is 0.0, clamp avoids 0/0. Token count is a tensor;
    # callers int() it only at log/eval time.
    token_count = mask.sum()
    loss_sum = torch.nn.functional.cross_entropy(
        logits[mask].float(), targets[mask].long(), reduction="sum"
    )
    return loss_sum / token_count.clamp(min=1), token_count


def text_supervision_mask(base_mask: Any, targets: Any, pad_id: int) -> Any:
    non_pad = targets != pad_id
    seen_text = non_pad.long().cumsum(dim=-1) > 0
    prefix_pad = (targets == pad_id) & ~seen_text
    return base_mask & (non_pad | prefix_pad)


def weighted_text_cross_entropy(
    logits: Any, targets: Any, mask: Any, pad_id: int, prefix_pad_weight: float
) -> tuple[Any, Any]:
    token_count = mask.sum()
    selected_targets = targets[mask].long()
    token_losses = torch.nn.functional.cross_entropy(
        logits[mask].float(), selected_targets, reduction="none"
    )
    weights = torch.where(selected_targets == pad_id, prefix_pad_weight, 1.0)
    loss = (token_losses * weights).sum() / weights.sum().clamp(min=1)
    return loss, token_count


def batch_condition_tensors(lm: Any, model_type: str, batch_size: int) -> Any | None:
    if lm.fuser is None:
        return None
    return get_condition_tensors(model_type, lm, batch_size=batch_size, cfg_coef=1.0)


def compute_batch_losses(
    lm: Any,
    codes: Any,
    condition_tensors: Any | None,
    audio_loss_weight: float,
    text_loss_weight: float,
    text_prefix_pad_weight: float = 1.0,
    return_output: bool = False,
) -> dict[str, Any]:
    output = lm(codes, condition_tensors=condition_tensors)
    audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    text_targets = codes[:, :1]
    audio_loss, audio_tokens = masked_cross_entropy(output.logits, audio_targets, output.mask)
    text_mask = text_supervision_mask(output.text_mask, text_targets, lm.text_padding_token_id)
    text_loss, text_tokens = weighted_text_cross_entropy(
        output.text_logits,
        text_targets,
        text_mask,
        lm.text_padding_token_id,
        text_prefix_pad_weight,
    )
    loss = audio_loss_weight * audio_loss + text_loss_weight * text_loss
    result = {
        "loss": loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": audio_tokens,
        "text_tokens": text_tokens,
    }
    if return_output:
        # Only for eval: keeping logits alive an extra step in the train loop
        # would cost ~1 GB on a full-VRAM box.
        result["output"] = output
    return result


def evaluate_teacher_forced(
    lm: Any,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    audio_loss_weight: float,
    text_loss_weight: float,
    max_batches: int = 0,
) -> dict[str, float | int]:
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    totals = {
        "audio_loss_sum": 0.0,
        "text_loss_sum": 0.0,
        "audio_tokens": 0,
        "text_tokens": 0,
        "batches": 0,
        "samples": 0,
        "content_loss_sum": 0.0,
        "content_correct": 0,
        "content_tokens": 0,
        "silence_sum": 0.0,
        "silence_count": 0,
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches and batch_index >= max_batches:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(lm, model_type, batch_size)
            losses = compute_batch_losses(
                lm, codes, condition_cache[batch_size], audio_loss_weight, text_loss_weight,
                return_output=True,
            )
            output = losses.pop("output")
            audio_tokens = int(losses["audio_tokens"])
            text_tokens = int(losses["text_tokens"])
            totals["audio_loss_sum"] += float(losses["audio_loss"].detach().cpu()) * audio_tokens
            totals["text_loss_sum"] += float(losses["text_loss"].detach().cpu()) * text_tokens
            totals["audio_tokens"] += audio_tokens
            totals["text_tokens"] += text_tokens
            totals["batches"] += 1
            totals["samples"] += batch_size
            # Post-mortem metrics: the plain text CE is 57% prefix pads and kept
            # improving through the pad-collapse; these three see generation
            # health directly (content-only CE/acc, pad mass at first content).
            text_targets = codes[:, :1]
            pad_id = lm.text_padding_token_id
            content_mask = output.text_mask & (text_targets != pad_id)
            content_logits = output.text_logits[content_mask].float()
            content_targets = text_targets[content_mask].long()
            if content_targets.numel():
                totals["content_loss_sum"] += float(
                    torch.nn.functional.cross_entropy(content_logits, content_targets, reduction="sum")
                )
                totals["content_correct"] += int((content_logits.argmax(-1) == content_targets).sum())
                totals["content_tokens"] += int(content_targets.numel())
            first_content = content_mask & (content_mask.long().cumsum(-1) == 1)
            first_logits = output.text_logits[first_content].float()
            if first_logits.shape[0]:
                totals["silence_sum"] += float(first_logits.softmax(-1)[:, pad_id].sum())
                totals["silence_count"] += int(first_logits.shape[0])

    if was_training:
        lm.train()
    if not totals["batches"]:
        raise RuntimeError("Validation dataloader produced no batches.")

    audio_loss = totals["audio_loss_sum"] / totals["audio_tokens"] if totals["audio_tokens"] else 0.0
    text_loss = totals["text_loss_sum"] / totals["text_tokens"] if totals["text_tokens"] else 0.0
    content_tokens = totals["content_tokens"]
    return {
        "loss": audio_loss_weight * audio_loss + text_loss_weight * text_loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": totals["audio_tokens"],
        "text_tokens": totals["text_tokens"],
        "batches": totals["batches"],
        "samples": totals["samples"],
        "content_text_loss": totals["content_loss_sum"] / content_tokens if content_tokens else 0.0,
        "content_acc": totals["content_correct"] / content_tokens if content_tokens else 0.0,
        "content_tokens": content_tokens,
        "silence_score": totals["silence_sum"] / totals["silence_count"] if totals["silence_count"] else 0.0,
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
    rows: list[dict[str, str]], source_column: str, reference_column: str, id_column: str
) -> None:
    if not rows:
        return
    missing = [c for c in (source_column, reference_column, id_column) if c not in rows[0]]
    if missing:
        raise ValueError(f"Eval rows are missing columns: {', '.join(missing)}")


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
    text = text.lower()
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
    try:
        import sacrebleu
    except ImportError:
        metrics["sacrebleu"] = "missing"
        return metrics
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
    cfg: Any,
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


def generate_batch(
    rows: list[dict[str, str]],
    batch_start: int,
    cfg: Any,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
) -> list[dict[str, str]]:
    """Greedy autoregressive decode of a batch of source clips.

    `cfg` supplies: source_column, reference_column, id_column, gen_duration,
    tail_s, stop_on_eos, text_only, tag.
    """
    files = [resolve_repo_path(row[cfg.source_column]) for row in rows]
    input_wavs = [audio_read(path, to_sample_rate=mimi.sample_rate, mono=True)[0] for path in files]
    audio_durations = [wav.shape[-1] / mimi.sample_rate for wav in input_wavs]
    gen_duration = cfg.gen_duration if cfg.gen_duration else max(audio_durations) + cfg.tail_s
    if max(audio_durations) > gen_duration:
        raise RuntimeError(f"Source audio is longer than generation duration: {max(audio_durations)}")

    batch_wavs = stack_and_pad_audio(input_wavs, max_len=int(gen_duration * mimi.sample_rate))
    lm_gen = get_lmgen(lm, checkpoint_info, batch_size=batch_wavs.shape[0])
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
    cfg: Any,
    batch_size: int,
    mimi: Any,
    lm: Any,
    text_tokenizer: Any,
    checkpoint_info: Any,
    out_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Batched greedy eval over `rows`; returns (records, scored metrics)."""
    records: list[dict[str, str]] = []
    for start in range(0, len(rows), batch_size):
        records.extend(
            generate_batch(
                rows[start : start + batch_size], start, cfg, mimi, lm, text_tokenizer,
                checkpoint_info, out_dir,
            )
        )
    return records, score_records(records)


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
