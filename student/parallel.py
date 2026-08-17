#!/usr/bin/env python
"""Frozen mobile-student parallel audio head and its strict lineage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from contract import build_meta_ar_model, read_config, sha256, validate_config

PARALLEL_PARAMETERS = 7_346_176
AR_QUALIFICATION_FORMAT = "hibiki_student_ar_qualification_v1"


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.in_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.out_proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.out_proj(torch.nn.functional.silu(self.in_proj(self.norm(value))))
        return value + residual


class ParallelHead(nn.Module):
    """Predict all eight target codebooks in one batched tensor operation."""

    def __init__(
        self,
        *,
        dim: int = 2048,
        head_dim: int = 512,
        codebooks: int = 8,
        card: int = 2048,
        layers: int = 2,
        passes: int = 1,
    ) -> None:
        super().__init__()
        if (dim, head_dim, codebooks, card, layers) != (2048, 512, 8, 2048, 2):
            raise ValueError("parallel_v1 shape changed without an architecture revision")
        if passes not in (1, 2):
            raise ValueError("parallel_v1 supports exactly one or two fixed passes")
        self.dim = dim
        self.head_dim = head_dim
        self.codebooks = codebooks
        self.card = card
        self.passes = passes
        self.context_projection = nn.Linear(dim, head_dim, bias=False)
        self.previous_embedding = nn.Embedding(card + 1, head_dim)
        self.position_embedding = nn.Parameter(torch.empty(codebooks, head_dim))
        self.blocks = nn.ModuleList(ResidualBlock(head_dim) for _ in range(layers))
        self.final_norm = nn.RMSNorm(head_dim, eps=1e-5)
        self.output_projection = nn.Linear(head_dim, card, bias=False)
        nn.init.normal_(self.position_embedding, std=head_dim**-0.5)
        if self.parameter_count != PARALLEL_PARAMETERS:
            raise RuntimeError(
                f"parallel_v1 has {self.parameter_count:,} parameters, "
                f"expected {PARALLEL_PARAMETERS:,}"
            )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ParallelHead:
        validate_config(cfg)
        if cfg["head"] != "parallel_v1":
            raise ValueError("ParallelHead requires a parallel_v1 config")
        return cls(
            dim=int(cfg["dim"]),
            head_dim=int(cfg["parallel_head_dim"]),
            codebooks=int(cfg["dep_q"]),
            card=int(cfg["card"]),
            layers=int(cfg["parallel_head_layers"]),
            passes=int(cfg["head_passes"]),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _predict(
        self,
        base: torch.Tensor,
        refinement_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = base
        if refinement_codes is not None:
            value = value + self.previous_embedding(refinement_codes)
        for block in self.blocks:
            value = block(value)
        return self.output_projection(self.final_norm(value))

    def forward(
        self,
        hidden: torch.Tensor,
        text_embedding: torch.Tensor,
        previous_codes: torch.Tensor,
    ) -> torch.Tensor:
        expected_context = (*hidden.shape[:2], self.dim)
        expected_codes = (*hidden.shape[:2], self.codebooks)
        if (
            tuple(hidden.shape) != expected_context
            or tuple(text_embedding.shape) != expected_context
        ):
            raise ValueError(f"hidden and text_embedding must be [B, T, {self.dim}]")
        if tuple(previous_codes.shape) != expected_codes:
            raise ValueError(f"previous_codes must be [B, T, {self.codebooks}]")
        if previous_codes.dtype not in (torch.int32, torch.int64):
            raise ValueError("previous_codes must be an integer tensor")
        if previous_codes.numel() and (
            int(previous_codes.min()) < 0 or int(previous_codes.max()) > self.card
        ):
            raise ValueError(f"previous_codes must be in [0, {self.card}]")

        context = self.context_projection(hidden + text_embedding).unsqueeze(2)
        base = (
            context
            + self.previous_embedding(previous_codes)
            + self.position_embedding.view(1, 1, self.codebooks, self.head_dim)
        )
        logits = self._predict(base)
        if self.passes == 2:
            # Every position observes the complete first-pass frame at once.
            logits = self._predict(base, logits.argmax(dim=-1))
        return logits


def require_compatible_configs(ar_cfg: dict[str, Any], parallel_cfg: dict[str, Any]) -> None:
    validate_config(ar_cfg)
    validate_config(parallel_cfg)
    if ar_cfg["head"] != "ar" or parallel_cfg["head"] != "parallel_v1":
        raise RuntimeError("Expected an AR base config and a parallel_v1 target config")
    parallel_base = dict(parallel_cfg)
    for key in ("parallel_head_dim", "parallel_head_layers"):
        parallel_base.pop(key)
    parallel_base["head"] = "ar"
    parallel_base["head_passes"] = 1
    if parallel_base != ar_cfg:
        raise RuntimeError("AR and parallel configs differ outside the frozen head boundary")


def checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()}


def require_exact_ar_checkpoint(path: Path, cfg: dict[str, Any]) -> None:
    expected = {
        name: tuple(tensor.shape) for name, tensor in build_meta_ar_model(cfg).state_dict().items()
    }
    actual = checkpoint_shapes(path)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong = sorted(
            name for name in set(actual) & set(expected) if actual[name] != expected[name]
        )
        raise RuntimeError(
            "Checkpoint does not exactly match the qualified 12-layer AR config: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} shape={wrong[:5]}"
        )


def validate_qualified_ar(
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    receipt_path: Path,
) -> dict[str, Any]:
    cfg = read_config(config_path)
    validate_config(cfg)
    if cfg["head"] != "ar" or int(cfg["num_layers"]) != 12:
        raise RuntimeError("Qualified backbone must be the 12-layer AR student")
    actual_checkpoint_sha = sha256(checkpoint_path)
    if actual_checkpoint_sha != checkpoint_sha256:
        raise RuntimeError("Qualified AR checkpoint SHA-256 does not match the explicit SHA")
    expected = {
        "format": AR_QUALIFICATION_FORMAT,
        "architecture": cfg["architecture"],
        "head": "ar",
        "decision": "pass",
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": actual_checkpoint_sha,
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != expected:
        raise RuntimeError("AR qualification receipt does not match the exact config/checkpoint")
    require_exact_ar_checkpoint(checkpoint_path, cfg)
    return cfg
