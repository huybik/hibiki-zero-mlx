#!/usr/bin/env python
"""Measure and validate the frozen mobile-student architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from moshi.models import loaders

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "student" / "configs" / "hibiki_m_12l_ar.json"
RECEIPT_FORMAT = "hibiki_student_shape_receipt_v1"

LOADER_METADATA_KEYS = {
    "architecture",
    "parent_repo",
    "parent_revision",
    "selected_parent_layers",
    "head",
    "head_passes",
    "parallel_head_dim",
    "parallel_head_layers",
    "sample_rate",
    "frame_rate",
    "frame_samples",
    "mimi_name",
    "tokenizer_name",
    "model_type",
    "lm_gen_config",
    "model_id",
}


def read_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(cfg: dict[str, Any]) -> None:
    expected = {
        "architecture": "hibiki_m_12l",
        "sample_rate": 24000,
        "frame_rate": 12.5,
        "frame_samples": 1920,
        "n_q": 16,
        "dep_q": 8,
        "dim": 2048,
        "num_layers": 12,
        "text_card": 48000,
    }
    mismatches = [
        f"{key}={cfg.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if cfg.get(key) != value
    ]
    if mismatches:
        raise ValueError("Student contract mismatch: " + "; ".join(mismatches))
    if cfg["sample_rate"] / cfg["frame_rate"] != cfg["frame_samples"]:
        raise ValueError("Audio sample rate, frame rate, and frame size disagree")
    if len(cfg.get("delays", [])) != 1 + cfg["n_q"]:
        raise ValueError("delays must contain text plus all audio streams")
    if cfg.get("selected_parent_layers") != [0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15]:
        raise ValueError("The frozen 16-to-12 parent layer mapping changed")
    head = cfg.get("head")
    passes = cfg.get("head_passes")
    if head == "ar" and passes != 1:
        raise ValueError("The AR head has exactly one autoregressive pass")
    if head == "parallel_v1":
        if passes not in (1, 2):
            raise ValueError("parallel_v1 supports one or two fixed passes")
        if cfg.get("parallel_head_dim") != 512 or cfg.get("parallel_head_layers") != 2:
            raise ValueError("parallel_v1 shape changed without an architecture revision")
    elif head != "ar":
        raise ValueError(f"Unsupported head: {head!r}")


def torch_lm_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return only fields understood by moshi's PyTorch model constructor."""
    return {key: value for key, value in cfg.items() if key not in LOADER_METADATA_KEYS}


def build_meta_ar_model(cfg: dict[str, Any]) -> Any:
    return loaders.get_moshi_lm(
        None,
        lm_kwargs=torch_lm_config(cfg),
        device="meta",
        dtype=torch.bfloat16,
    )


def parameter_groups(model: Any) -> dict[str, int]:
    groups = {"backbone": 0, "ar_head": 0}
    for name, parameter in model.named_parameters():
        group = "ar_head" if name.startswith(("depformer", "linears.")) else "backbone"
        groups[group] += parameter.numel()
    return groups


def parallel_head_parameters(cfg: dict[str, Any]) -> int:
    dim = int(cfg["dim"])
    head_dim = int(cfg["parallel_head_dim"])
    layers = int(cfg["parallel_head_layers"])
    codebooks = int(cfg["dep_q"])
    card = int(cfg["card"])
    context_in = dim * head_dim
    previous_embedding = (card + 1) * head_dim
    position_embedding = codebooks * head_dim
    blocks = layers * (head_dim + head_dim * 4 * head_dim + 4 * head_dim * head_dim)
    final_norm = head_dim
    output = head_dim * card
    return context_in + previous_embedding + position_embedding + blocks + final_norm + output


def state_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    head_dim = int(cfg["dim"]) // int(cfg["num_heads"])
    state: dict[str, Any] = {
        "mimi_encoder_io": {
            "pcm": ["batch", 1, cfg["frame_samples"]],
            "codes": ["batch", 1, cfg["dep_q"]],
        },
        "mimi_decoder_io": {
            "codes": ["batch", cfg["dep_q"], 1],
            "pcm": ["batch", 1, cfg["frame_samples"]],
        },
        "lm_kv_per_layer": {
            "layers": cfg["num_layers"],
            "key": ["batch", cfg["num_heads"], "frames<=1500", head_dim],
            "value": ["batch", cfg["num_heads"], "frames<=1500", head_dim],
        },
    }
    if cfg["head"] == "ar":
        state["head_state"] = {
            "lifetime": "one_frame",
            "layers": cfg["depformer_num_layers"],
            "key": ["batch", cfg["depformer_num_heads"], "codebooks<=8", 64],
            "value": ["batch", cfg["depformer_num_heads"], "codebooks<=8", 64],
        }
    else:
        state["head_state"] = {
            "previous_target_codes": ["batch", cfg["dep_q"]],
        }
    return state


def make_receipt(cfg: dict[str, Any]) -> dict[str, Any]:
    validate_config(cfg)
    model = build_meta_ar_model(cfg)
    groups = parameter_groups(model)
    if cfg["head"] == "ar":
        head_parameters = groups["ar_head"]
    else:
        head_parameters = parallel_head_parameters(cfg)
    total = groups["backbone"] + head_parameters
    return {
        "format": RECEIPT_FORMAT,
        "architecture": cfg["architecture"],
        "head": cfg["head"],
        "selected_parent_layers": cfg["selected_parent_layers"],
        "parameters": {
            "backbone": groups["backbone"],
            "head": head_parameters,
            "total": total,
        },
        "estimated_weight_bytes": {
            "bf16": total * 2,
        },
        "state": state_contract(cfg),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    receipt = sub.add_parser("receipt", help="measure a config without allocating model weights")
    receipt.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    receipt.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_receipt(read_config(args.config))
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
