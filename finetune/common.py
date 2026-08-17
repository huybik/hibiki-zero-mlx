"""Shared training and evaluation logic for Vietnamese full-model SFT.

This module owns device helpers, the cached-shard dataset, exact full-model
checkpoint I/O, teacher-forced losses, autoregressive generation, text metrics,
and correct-source generation health diagnostics.

It is a PyTorch training toolkit; torch, safetensors and the `moshi` pip package
are hard dependencies imported at module top (the conda base env ships them).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch

# moshi reads NO_TORCH_COMPILE once at import. Default-enable compile on CUDA:
# The pinned CUDA build supports compiled H100 training. Keep it off elsewhere
# (no gain on MPS).
# Override with NO_TORCH_COMPILE=1.
os.environ.setdefault("NO_TORCH_COMPILE", "" if torch.cuda.is_available() else "1")

import numpy as np
import sacrebleu
import sphn
from moshi.models import loaders
from moshi.run_inference import get_condition_tensors
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, Subset

from finetune.cache_codes import CACHE_FORMAT, GROUNDED_CACHE_FORMAT
from finetune.hibiki_helpers import (
    audio_read,
    decode_outputs,
    encode_inputs,
    get_lmgen,
    stack_and_pad_audio,
)
from finetune.utils import repo_display_path, require_file, resolve_repo_path

SUPPORTED_CACHE_FORMATS = {CACHE_FORMAT, GROUNDED_CACHE_FORMAT}


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
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
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
# Cached-shard dataset / loader
# --------------------------------------------------------------------------- #
class CachedCodeDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path | list[Path],
        sort_by_length: bool,
        max_samples: int,
        max_frames: int = 0,
        cache_weights: list[float] | None = None,
        seed: int = 0,
        sample_manifest: Path | None = None,
        sample_manifest_sha256: str | None = None,
    ):
        self.samples: list[dict[str, Any]] = []
        self.frame_rate: float | None = None
        dropped = 0
        cache_dirs = [cache_dir] if isinstance(cache_dir, Path) else list(cache_dir)
        self.cache_count = len(cache_dirs)
        if (sample_manifest is None) != (sample_manifest_sha256 is None):
            raise ValueError("Input sample manifest and SHA-256 must be set together")
        manifest_keys: list[tuple[int, str]] | None = None
        samples_by_key: dict[tuple[int, str], dict[str, Any]] = {}
        if sample_manifest is not None and sample_manifest_sha256 is not None:
            if max_samples or cache_weights is not None:
                raise ValueError("Input sample manifest forbids max-sample and cache-weight sampling")
            if sort_by_length:
                raise ValueError("Input sample manifest requires exact order; disable length sorting")
            content = sample_manifest.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != sample_manifest_sha256:
                raise RuntimeError(
                    f"Input sample manifest SHA-256 mismatch: {digest} != "
                    f"{sample_manifest_sha256}"
                )
            manifest_keys = []
            for line_no, line in enumerate(content.decode("utf-8").splitlines(), 1):
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {"cache_index", "id"}:
                    raise ValueError(f"Invalid input sample manifest row {line_no}")
                cache_index = row["cache_index"]
                sample_id = row["id"]
                if (
                    isinstance(cache_index, bool)
                    or not isinstance(cache_index, int)
                    or not 0 <= cache_index < len(cache_dirs)
                    or not isinstance(sample_id, str)
                    or not sample_id
                ):
                    raise ValueError(f"Invalid input sample manifest row {line_no}")
                manifest_keys.append((cache_index, sample_id))
            if not manifest_keys:
                raise ValueError("Input sample manifest is empty")
        shard_paths = [p for d in cache_dirs for p in sorted(d.glob("shard_*.pt"))]
        for shard_path in shard_paths:
            cache_index = next(
                index for index, directory in enumerate(cache_dirs) if shard_path.parent == directory
            )
            payload = torch.load(shard_path, map_location="cpu")
            cache_format = payload.get("format")
            if cache_format not in SUPPORTED_CACHE_FORMATS:
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
                if manifest_keys is None and max_frames and codes.shape[1] > max_frames:
                    dropped += 1
                    continue
                item = {
                    "id": str(sample["id"]),
                    # int32 in host RAM (halves footprint at ~700k samples);
                    # collate_cached casts to long on batch assembly.
                    "codes": codes.to(torch.int32),
                    "frames": int(codes.shape[1]),
                    "source_frames": int(sample["vi_frames"]),
                    "cache_format": str(cache_format),
                    "stratum": str(sample.get("stratum", "legacy_unspecified")),
                    "split": str(sample.get("split", "")),
                    "speaker_id": str(sample.get("speaker_id", "")),
                    "gender": str(sample.get("gender", "")),
                    "cache_index": cache_index,
                }
                if manifest_keys is None:
                    self.samples.append(item)
                else:
                    key = (cache_index, item["id"])
                    if key in samples_by_key:
                        raise RuntimeError(f"Duplicate cached sample for cache_index,id={key}")
                    samples_by_key[key] = item
        if manifest_keys is not None:
            missing = [key for key in manifest_keys if key not in samples_by_key]
            if missing:
                raise RuntimeError(f"Input sample manifest references missing samples: {missing[:10]}")
            self.samples = [samples_by_key[key] for key in manifest_keys]
            self.require_max_frames(max_frames, "Input sample manifest")
        if not self.samples:
            raise RuntimeError(f"No shard_*.pt cache files found in {cache_dir}")
        if max_frames and dropped:
            print(f"[dataset] dropped {dropped} samples over {max_frames} frames; kept {len(self.samples)}")
        if cache_weights is not None:
            self.select_weighted(cache_weights, max_samples, seed)
            max_samples = 0
        if sort_by_length:
            self.samples.sort(key=lambda sample: sample["frames"])
        if max_samples:
            self.samples = self.samples[:max_samples]
            if not self.samples:
                raise RuntimeError("--max-samples selected no cached samples")

    def select_weighted(
        self, cache_weights: list[float], max_samples: int, seed: int
    ) -> None:
        if len(cache_weights) != self.cache_count or any(
            weight <= 0 for weight in cache_weights
        ):
            raise ValueError("--cache-weights must provide one positive weight per --cache-dir")
        weight_total = sum(cache_weights)
        pools = [
            [sample for sample in self.samples if sample["cache_index"] == cache_index]
            for cache_index in range(len(cache_weights))
        ]
        if any(not pool for pool in pools):
            raise RuntimeError("Every weighted cache directory must have usable samples")
        target_total = max_samples or max(
            math.ceil(len(pool) * weight_total / weight)
            for pool, weight in zip(pools, cache_weights, strict=True)
        )
        counts: list[int] = []
        if max_samples:
            quotas = [target_total * weight / weight_total for weight in cache_weights]
            if any(
                not math.isclose(quota, round(quota), rel_tol=0.0, abs_tol=1e-8)
                for quota in quotas
            ):
                raise ValueError(
                    "--max-samples must permit exact integer --cache-weights counts"
                )
            counts = [round(quota) for quota in quotas]
            if sum(counts) != target_total:
                raise RuntimeError("Exact cache-weight counts do not sum to --max-samples")
        else:
            remaining = target_total
            for cache_index, weight in enumerate(cache_weights):
                count = (
                    remaining
                    if cache_index == len(cache_weights) - 1
                    else round(target_total * weight / weight_total)
                )
                counts.append(count)
                remaining -= count
        rng = random.Random(seed)
        balanced: list[dict[str, Any]] = []
        for pool, count in zip(pools, counts, strict=True):
            if count <= len(pool):
                balanced.extend(rng.sample(pool, count))
            else:
                balanced.extend(pool)
                balanced.extend(rng.choices(pool, k=count - len(pool)))
        self.samples = balanced

    def filter_max_frames(self, max_frames: int, label: str) -> None:
        if max_frames <= 0:
            raise ValueError("Frame filtering requires a positive maximum")
        before = len(self.samples)
        self.samples = [sample for sample in self.samples if sample["frames"] <= max_frames]
        if not self.samples:
            raise RuntimeError(f"{label} selected no samples at T<={max_frames}")
        print(
            f"[dataset] {label}: dropped {before - len(self.samples)} samples over "
            f"{max_frames} frames; kept {len(self.samples)}"
        )

    def require_max_frames(self, max_frames: int, label: str) -> None:
        if not max_frames or not self.samples:
            return
        observed = max(sample["frames"] for sample in self.samples)
        if observed > max_frames:
            over = sum(sample["frames"] > max_frames for sample in self.samples)
            raise RuntimeError(
                f"{label} exceeds --max-frames={max_frames}: "
                f"{over} entries, observed max T={observed}"
            )

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


def sample_manifest_bytes(dataset: CachedCodeDataset) -> bytes:
    return "".join(
        json.dumps(
            {"cache_index": sample["cache_index"], "id": sample["id"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for sample in dataset.samples
    ).encode("utf-8")


# Pad each batch's frame length up to a multiple of this. MPS compiles+caches a
# Metal kernel graph per distinct tensor shape; the raw pool has 262 distinct
# lengths, which balloons the GPU working set (26 GB wired, swap-thrash). Bucketing
# collapses that to ~9 shapes. Loss-neutral: extra frames are -1 == zero_token_id,
# masked out of both CE terms (see LMModel.forward logits_mask). CUDA launch
# recipes override this to balance padding against compiled-shape reuse.
FRAME_BUCKET = int(os.environ.get("HIBIKI_FRAME_BUCKET", "32"))


def collate_cached(samples: list[dict[str, Any]]) -> dict[str, Any]:
    codebooks = int(samples[0]["codes"].shape[0])
    max_frames = max(int(sample["codes"].shape[1]) for sample in samples)
    max_frames = ((max_frames + FRAME_BUCKET - 1) // FRAME_BUCKET) * FRAME_BUCKET
    batch = torch.full((len(samples), codebooks, max_frames), -1, dtype=torch.long)
    ids: list[str] = []
    strata: list[str] = []
    for index, sample in enumerate(samples):
        codes = sample["codes"]
        batch[index, :, : codes.shape[1]] = codes
        ids.append(sample["id"])
        strata.append(sample["stratum"])
    return {
        "codes": batch,
        "ids": ids,
        "frames": torch.tensor([sample["frames"] for sample in samples]),
        "source_frames": torch.tensor([sample["source_frames"] for sample in samples]),
        "strata": strata,
    }


def make_cached_dataloader(
    dataset: CachedCodeDataset,
    batch_size: int,
    num_workers: int,
    sort_by_length: bool,
    seed: int = 0,
    shuffle: bool | None = None,
    sample_order: list[int] | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_dataset: Dataset = dataset
    if sample_order is not None:
        loader_dataset = Subset(dataset, sample_order)
    if shuffle is None:
        shuffle = not sort_by_length
    return DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_cached,
        generator=generator,
    )


# --------------------------------------------------------------------------- #
# Full-model checkpoint I/O
# --------------------------------------------------------------------------- #
def enable_full_finetune(model: Any) -> None:
    """Train every language-model parameter."""
    for param in model.parameters():
        param.requires_grad_(True)


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def save_model(model: Any, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
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


def text_supervision_mask(base_mask: Any, targets: Any, pad_id: int, pad_mode: str) -> Any:
    if pad_mode == "all":
        return base_mask
    if pad_mode != "prefix":
        raise ValueError(f"Unsupported text PAD mode: {pad_mode}")
    non_pad = targets != pad_id
    seen_text = non_pad.long().cumsum(dim=-1) > 0
    prefix_pad = (targets == pad_id) & ~seen_text
    return base_mask & (non_pad | prefix_pad)


def combine_text_losses(
    content_loss: Any,
    pad_loss: Any,
    pad_loss_weight: float,
) -> Any:
    return (content_loss + pad_loss_weight * pad_loss) / (1.0 + pad_loss_weight)


def balanced_text_cross_entropy(
    logits: Any,
    targets: Any,
    mask: Any,
    pad_id: int,
    pad_loss_weight: float,
) -> dict[str, Any]:
    """Keep PAD pressure independent of the number of PAD frames."""
    token_count = mask.sum()
    selected_targets = targets[mask].long()
    token_losses = torch.nn.functional.cross_entropy(
        logits[mask].float(), selected_targets, reduction="none"
    )
    selected_pad = selected_targets == pad_id
    selected_content = ~selected_pad
    content_mask = mask & (targets != pad_id)
    first_content_mask = content_mask & (content_mask.long().cumsum(dim=-1) == 1)
    selected_first_content = first_content_mask[mask]

    pad_tokens = selected_pad.sum()
    content_tokens = selected_content.sum()
    first_content_tokens = selected_first_content.sum()
    pad_loss = token_losses[selected_pad].sum() / pad_tokens.clamp(min=1)
    content_loss = token_losses[selected_content].sum() / content_tokens.clamp(min=1)
    first_content_loss = token_losses[selected_first_content].sum() / first_content_tokens.clamp(
        min=1
    )
    loss = combine_text_losses(
        content_loss,
        pad_loss,
        pad_loss_weight,
    )
    return {
        "loss": loss,
        "tokens": token_count,
        "pad_loss": pad_loss,
        "pad_tokens": pad_tokens,
        "content_loss": content_loss,
        "content_tokens": content_tokens,
        "first_content_loss": first_content_loss,
        "first_content_tokens": first_content_tokens,
    }


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
    text_pad_loss_weight: float = 0.05,
    text_pad_mode: str = "prefix",
    return_output: bool = False,
) -> dict[str, Any]:
    output = lm(codes, condition_tensors=condition_tensors)
    audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    text_targets = codes[:, :1]
    audio_loss, audio_tokens = masked_cross_entropy(output.logits, audio_targets, output.mask)
    text_mask = text_supervision_mask(
        output.text_mask, text_targets, lm.text_padding_token_id, text_pad_mode
    )
    text_losses = balanced_text_cross_entropy(
        output.text_logits,
        text_targets,
        text_mask,
        lm.text_padding_token_id,
        text_pad_loss_weight,
    )
    text_loss = text_losses["loss"]
    loss = audio_loss_weight * audio_loss + text_loss_weight * text_loss
    result = {
        "loss": loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": audio_tokens,
        "text_tokens": text_losses["tokens"],
        "pad_text_loss": text_losses["pad_loss"],
        "pad_text_tokens": text_losses["pad_tokens"],
        "content_text_loss": text_losses["content_loss"],
        "content_text_tokens": text_losses["content_tokens"],
        "first_content_loss": text_losses["first_content_loss"],
        "first_content_tokens": text_losses["first_content_tokens"],
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
    text_pad_mode: str = "prefix",
    text_pad_loss_weight: float = 0.05,
) -> dict[str, float | int]:
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    totals = {
        "audio_loss_sum": 0.0,
        "audio_tokens": 0,
        "text_tokens": 0,
        "batches": 0,
        "samples": 0,
        "max_frames": 0,
        "max_padded_frames": 0,
        "content_loss_sum": 0.0,
        "content_correct": 0,
        "content_tokens": 0,
        "pad_loss_sum": 0.0,
        "pad_correct": 0,
        "pad_tokens": 0,
        "first_content_loss_sum": 0.0,
        "first_content_tokens": 0,
        "first_content_margin_sum": 0.0,
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
                text_pad_loss_weight=text_pad_loss_weight,
                text_pad_mode=text_pad_mode,
                return_output=True,
            )
            output = losses.pop("output")
            audio_tokens = int(losses["audio_tokens"])
            text_tokens = int(losses["text_tokens"])
            totals["audio_loss_sum"] += float(losses["audio_loss"].detach().cpu()) * audio_tokens
            totals["audio_tokens"] += audio_tokens
            totals["text_tokens"] += text_tokens
            component_keys = {
                "content": ("content_text_loss", "content_text_tokens"),
                "pad": ("pad_text_loss", "pad_text_tokens"),
                "first_content": ("first_content_loss", "first_content_tokens"),
            }
            for prefix, (loss_key, count_key) in component_keys.items():
                count = int(losses[count_key])
                totals[f"{prefix}_loss_sum"] += float(losses[loss_key].detach().cpu()) * count
                totals[f"{prefix}_tokens"] += count
            totals["batches"] += 1
            totals["samples"] += batch_size
            totals["max_frames"] = max(totals["max_frames"], int(batch["frames"].max()))
            totals["max_padded_frames"] = max(
                totals["max_padded_frames"], int(codes.shape[-1])
            )
            # Diagnostics for content quality, PAD behavior, and the critical
            # PAD-to-first-content transition.
            text_targets = codes[:, :1]
            pad_id = lm.text_padding_token_id
            content_mask = output.text_mask & (text_targets != pad_id)
            content_logits = output.text_logits[content_mask].float()
            content_targets = text_targets[content_mask].long()
            if content_targets.numel():
                totals["content_correct"] += int((content_logits.argmax(-1) == content_targets).sum())
            supervised_text_mask = text_supervision_mask(
                output.text_mask, text_targets, pad_id, text_pad_mode
            )
            pad_mask = supervised_text_mask & (text_targets == pad_id)
            pad_logits = output.text_logits[pad_mask].float()
            if pad_logits.shape[0]:
                totals["pad_correct"] += int((pad_logits.argmax(-1) == pad_id).sum())
            first_content = content_mask & (content_mask.long().cumsum(-1) == 1)
            first_logits = output.text_logits[first_content].float()
            if first_logits.shape[0]:
                totals["silence_sum"] += float(first_logits.softmax(-1)[:, pad_id].sum())
                first_targets = text_targets[first_content].long()
                target_logits = first_logits.gather(1, first_targets[:, None]).squeeze(1)
                totals["first_content_margin_sum"] += float(
                    (target_logits - first_logits[:, pad_id]).sum()
                )
                totals["silence_count"] += int(first_logits.shape[0])

    if was_training:
        lm.train()
    if not totals["batches"]:
        raise RuntimeError("Validation dataloader produced no batches.")

    audio_loss = totals["audio_loss_sum"] / totals["audio_tokens"] if totals["audio_tokens"] else 0.0
    content_tokens = totals["content_tokens"]
    pad_tokens = totals["pad_tokens"]
    first_content_tokens = totals["first_content_tokens"]
    content_loss = totals["content_loss_sum"] / content_tokens if content_tokens else 0.0
    pad_loss = totals["pad_loss_sum"] / pad_tokens if pad_tokens else 0.0
    first_content_loss = (
        totals["first_content_loss_sum"] / first_content_tokens if first_content_tokens else 0.0
    )
    text_loss = combine_text_losses(
        content_loss,
        pad_loss,
        text_pad_loss_weight,
    )
    return {
        "loss": audio_loss_weight * audio_loss + text_loss_weight * text_loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": totals["audio_tokens"],
        "text_tokens": totals["text_tokens"],
        "batches": totals["batches"],
        "samples": totals["samples"],
        "max_frames": totals["max_frames"],
        "max_padded_frames": totals["max_padded_frames"],
        "content_text_loss": content_loss,
        "content_acc": totals["content_correct"] / content_tokens if content_tokens else 0.0,
        "content_tokens": content_tokens,
        "pad_text_loss": pad_loss,
        "pad_acc": totals["pad_correct"] / pad_tokens if pad_tokens else 0.0,
        "pad_tokens": pad_tokens,
        "first_content_loss": first_content_loss,
        "first_content_tokens": first_content_tokens,
        "first_content_margin": (
            totals["first_content_margin_sum"] / totals["silence_count"]
            if totals["silence_count"]
            else 0.0
        ),
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
