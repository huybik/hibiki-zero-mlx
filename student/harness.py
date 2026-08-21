#!/usr/bin/env python
"""Shared run infrastructure for the mobile-student CUDA trainers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

from student.contract import sha256


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_identity(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    shards = [
        {"name": path.name, "sha256": sha256(path)} for path in sorted(cache_dir.glob("shard_*.pt"))
    ]
    return {
        "metadata_sha256": canonical_sha256(metadata),
        "shards_sha256": canonical_sha256(shards),
        "shards": shards,
    }


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_run_dir(out_dir: Path, contract: dict[str, Any], resume: bool, label: str = "") -> str:
    path = out_dir / "run.json"
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if resume:
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"{label}Resume contract differs from the original run.json")
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return canonical_sha256(contract)


def checkpoint_pairs(out_dir: Path, prefix: str, label: str = "") -> dict[int, tuple[Path, Path]]:
    model_re = re.compile(rf"{prefix}_step(\d+)\.safetensors")
    optimizer_re = re.compile(r"optimizer_step(\d+)\.pt")
    models = {
        int(match.group(1)): path
        for path in out_dir.glob(f"{prefix}_step*.safetensors")
        if (match := model_re.fullmatch(path.name)) is not None
    }
    optimizers = {
        int(match.group(1)): path
        for path in out_dir.glob("optimizer_step*.pt")
        if (match := optimizer_re.fullmatch(path.name)) is not None
    }
    if models.keys() != optimizers.keys():
        raise RuntimeError(f"Run directory contains an incomplete {label}checkpoint pair")
    return {step: (models[step], optimizers[step]) for step in models}


def prune_checkpoints(pairs: dict[int, tuple[Path, Path]], keep: int) -> None:
    for step in sorted(pairs)[: max(0, len(pairs) - keep)]:
        model_path, optimizer_path = pairs[step]
        model_path.unlink()
        optimizer_path.unlink()


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask & (targets >= 0) & (targets < logits.shape[-1])
    count = mask.sum()
    loss = F.cross_entropy(logits[mask].float(), targets[mask].long(), reduction="sum")
    return loss / count.clamp(min=1), count


def topk_residual_kl(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL over stored top-k atoms plus one aggregate residual-mass atom.

    Teacher values are absolute log-probabilities. The omitted probability mass
    is retained as one event; the top-k values are never renormalized.
    """
    if student_logits.ndim != teacher_ids.ndim or teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError(f"Malformed student or teacher {label} tensors")
    if teacher_ids.shape[:-1] != student_logits.shape[:-1] or mask.shape != teacher_ids.shape[:-1]:
        raise ValueError(f"Teacher {label} targets are not aligned with student logits")
    count = mask.sum()
    safe_ids = teacher_ids.clamp_min(0).long()
    logits = student_logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    student_top_logprobs = logits.gather(-1, safe_ids) - log_z.unsqueeze(-1)
    teacher_logprobs = teacher_logprobs.float()
    teacher_probs = teacher_logprobs.exp()

    teacher_top_mass = teacher_probs.sum(-1).clamp(max=1.0)
    student_top_mass = student_top_logprobs.exp().sum(-1).clamp(max=1.0 - 1e-7)
    teacher_other = 1.0 - teacher_top_mass
    student_other_logprob = torch.log1p(-student_top_mass)
    top_terms = (teacher_probs * (teacher_logprobs - student_top_logprobs)).sum(-1)
    other_terms = teacher_other * (teacher_other.clamp_min(1e-30).log() - student_other_logprob)
    frame_kl = top_terms + other_terms
    return frame_kl[mask].sum() / count.clamp(min=1), count


def validate_common_hyperparams(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "batch_size",
        "grad_accum_steps",
        "save_every",
        "keep_checkpoints",
        "log_every",
    )
    if any(getattr(args, key) <= 0 for key in positive):
        names = ", ".join("--" + key.replace("_", "-") for key in positive)
        raise ValueError(f"{names} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError("--grad-clip must be finite and positive")


def restore_optimizer_rng(optimizer: torch.optim.Optimizer, payload: dict[str, Any]) -> None:
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])


def checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}


def require_exact_shapes(
    expected: dict[str, tuple[int, ...]],
    actual: dict[str, tuple[int, ...]],
    label: str,
) -> None:
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong = sorted(
            name for name in set(actual) & set(expected) if actual[name] != expected[name]
        )
        raise RuntimeError(
            f"{label} missing={missing[:5]} unexpected={unexpected[:5]} shape={wrong[:5]}"
        )
