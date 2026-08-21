#!/usr/bin/env python
"""Strictly initialize the 12-layer AR student from official Hibiki-M 1B."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from moshi.models import loaders
from safetensors import safe_open
from safetensors.torch import save_file

from student.contract import read_config, sha256, torch_lm_config, validate_config
from student.harness import checkpoint_shapes, require_exact_shapes

LAYER_RE = re.compile(r"^transformer\.layers\.(\d+)\.(.+)$")


def build_meta_model(cfg: dict[str, Any]) -> Any:
    return loaders.get_moshi_lm(
        None,
        lm_kwargs=torch_lm_config(cfg),
        device="meta",
        dtype=torch.bfloat16,
    )


def source_name(target_name: str, selected_layers: list[int]) -> str:
    match = LAYER_RE.match(target_name)
    if match is None:
        return target_name
    target_index = int(match.group(1))
    return f"transformer.layers.{selected_layers[target_index]}.{match.group(2)}"


def expected_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def validate_parent(target: dict[str, Any], parent: dict[str, Any]) -> None:
    if int(parent.get("num_layers", 0)) != 16:
        raise ValueError("The initialization parent must have 16 backbone layers")
    for key in (
        "card",
        "n_q",
        "dep_q",
        "delays",
        "dim",
        "text_card",
        "num_heads",
        "hidden_scale",
        "context",
        "max_period",
        "positional_embedding",
        "depformer_dim",
        "depformer_num_heads",
        "depformer_num_layers",
    ):
        if target.get(key) != parent.get(key):
            raise ValueError(
                f"Parent/student incompatibility at {key}: "
                f"{parent.get(key)!r} != {target.get(key)!r}"
            )


def initialize(
    target_cfg: dict[str, Any],
    parent_cfg: dict[str, Any],
    parent_weights: Path,
    output: Path,
) -> dict[str, Any]:
    validate_config(target_cfg)
    if target_cfg["head"] != "ar":
        raise ValueError("Full-model initialization is only valid for the AR student")
    validate_parent(target_cfg, parent_cfg)

    parent_expected = expected_shapes(build_meta_model(parent_cfg))
    parent_actual = checkpoint_shapes(parent_weights)
    require_exact_shapes(
        parent_expected, parent_actual, "Parent checkpoint does not exactly match its config:"
    )

    target_expected = expected_shapes(build_meta_model(target_cfg))
    selected = list(target_cfg["selected_parent_layers"])
    mapping = {name: source_name(name, selected) for name in target_expected}
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("Layer mapping is not one-to-one")
    for target_name, parent_name in mapping.items():
        if parent_name not in parent_actual:
            raise RuntimeError(f"Missing mapped parent tensor: {parent_name}")
        if target_expected[target_name] != parent_actual[parent_name]:
            raise RuntimeError(
                f"Mapped tensor shape mismatch for {target_name}: "
                f"{target_expected[target_name]} != {parent_actual[parent_name]}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(parent_weights, framework="pt", device="cpu") as handle:
        state = {
            target_name: handle.get_tensor(parent_name).contiguous()
            for target_name, parent_name in mapping.items()
        }
    parent_hash = sha256(parent_weights)
    save_file(
        state,
        str(output),
        metadata={
            "format": "hibiki_student_initialization_v1",
            "parent_sha256": parent_hash,
            "selected_parent_layers": ",".join(map(str, selected)),
        },
    )
    del state
    return {
        "format": "hibiki_student_initialization_receipt_v1",
        "parent_repo": target_cfg["parent_repo"],
        "parent_revision": target_cfg["parent_revision"],
        "parent_weights_sha256": parent_hash,
        "selected_parent_layers": selected,
        "student_weights": output.name,
        "student_weights_sha256": sha256(output),
        "tensors": len(mapping),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--parent-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.parent_config, args.parent_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Refusing to overwrite an initialized checkpoint or receipt")
    receipt = initialize(
        read_config(args.config),
        read_config(args.parent_config),
        args.parent_weights,
        args.output,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
