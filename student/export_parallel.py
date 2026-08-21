#!/usr/bin/env python
"""Merge an exact AR backbone and parallel head into a BF16 listening checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from student.contract import read_config, sha256
from student.harness import checkpoint_shapes, require_exact_shapes
from student.parallel import (
    PARALLEL_PARAMETERS,
    ParallelHead,
    require_compatible_configs,
    validate_ar_checkpoint,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AR_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_ar.json"
DEFAULT_PARALLEL_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_parallel_v1.json"
HEAD_CHECKPOINT_FORMAT = "hibiki_parallel_head_checkpoint_v1"
EXPORT_FORMAT = "hibiki_parallel_bf16_export_v1"


def validate_head_checkpoint(
    path: Path,
    expected_sha256: str,
    cfg: dict[str, Any],
    base_sha256: str,
    config_sha256: str,
) -> dict[str, str]:
    if sha256(path) != expected_sha256:
        raise RuntimeError("Parallel head SHA-256 does not match the explicit SHA")
    expected_shapes = {
        name: tuple(value.shape)
        for name, value in ParallelHead.from_config(cfg).state_dict().items()
    }
    actual_shapes = checkpoint_shapes(path)
    require_exact_shapes(expected_shapes, actual_shapes, "Parallel head checkpoint is not exact:")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    expected_keys = {
        "format",
        "step",
        "contract_sha256",
        "base_checkpoint_sha256",
        "parallel_config_sha256",
        "cache_sha256",
        "head_passes",
        "head_parameters",
    }
    if metadata is None or set(metadata) != expected_keys:
        raise RuntimeError("Parallel head checkpoint metadata is malformed")
    if (
        metadata["format"] != HEAD_CHECKPOINT_FORMAT
        or metadata["base_checkpoint_sha256"] != base_sha256
        or metadata["parallel_config_sha256"] != config_sha256
        or metadata["head_passes"] != str(cfg["head_passes"])
        or metadata["head_parameters"] != str(PARALLEL_PARAMETERS)
    ):
        raise RuntimeError("Parallel head was trained for a stale base/config")
    return metadata


def export(args: argparse.Namespace) -> None:
    if args.output_weights.exists() or args.output_config.exists():
        raise FileExistsError("Refusing to overwrite parallel export output")
    ar_cfg = validate_ar_checkpoint(
        args.ar_config,
        args.base_checkpoint,
        args.base_sha256,
    )
    parallel_cfg = read_config(args.parallel_config)
    require_compatible_configs(ar_cfg, parallel_cfg)
    parallel_config_sha = sha256(args.parallel_config)
    head_metadata = validate_head_checkpoint(
        args.head_checkpoint,
        args.head_sha256,
        parallel_cfg,
        args.base_sha256,
        parallel_config_sha,
    )

    obsolete = ("depformer", "linears.")
    state = {}
    with safe_open(args.base_checkpoint, framework="pt", device="cpu") as base:
        removed = [name for name in base.keys() if name.startswith(obsolete)]
        if not removed:
            raise RuntimeError("AR checkpoint contains no obsolete AR head tensors")
        for name in base.keys():
            if not name.startswith(obsolete):
                state[name] = base.get_tensor(name).to(torch.bfloat16).contiguous()
    with safe_open(args.head_checkpoint, framework="pt", device="cpu") as head:
        for name in head.keys():
            exported_name = f"parallel_head.{name}"
            if exported_name in state:
                raise RuntimeError(f"Export tensor collision: {exported_name}")
            state[exported_name] = head.get_tensor(name).to(torch.bfloat16).contiguous()

    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary_weights = args.output_weights.with_name(f".{args.output_weights.name}.tmp")
    temporary_config = args.output_config.with_name(f".{args.output_config.name}.tmp")
    try:
        save_file(
            state,
            str(temporary_weights),
            metadata={
                "format": EXPORT_FORMAT,
                "base_checkpoint_sha256": args.base_sha256,
                "parallel_head_sha256": args.head_sha256,
                "parallel_config_sha256": parallel_config_sha,
                "head_contract_sha256": head_metadata["contract_sha256"],
                "head_passes": str(parallel_cfg["head_passes"]),
                "dtype": "bfloat16",
            },
        )
        temporary_config.write_bytes(args.parallel_config.read_bytes())
        temporary_weights.replace(args.output_weights)
        temporary_config.replace(args.output_config)
    finally:
        temporary_weights.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)
    print(
        f"PASS: removed {len(removed)} AR tensors and wrote "
        f"{args.output_weights} ({sha256(args.output_weights)})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar-config", type=Path, default=DEFAULT_AR_CONFIG)
    parser.add_argument("--parallel-config", type=Path, default=DEFAULT_PARALLEL_CONFIG)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--head-sha256", required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
