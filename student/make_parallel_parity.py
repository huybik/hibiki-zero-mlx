#!/usr/bin/env python
"""Write a deterministic PyTorch parallel-head parity fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hibiki_mlx.student_pack import PARITY_FORMAT, read_json, validate_qualification
from safetensors import safe_open

from contract import read_config, sha256, validate_config
from parallel import ParallelHead

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_parallel_v1.json"
EXPORT_FORMAT = "hibiki_parallel_bf16_export_v1"


def make_fixture(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite parity fixture: {args.output}")
    cfg = read_config(args.config)
    validate_config(cfg)
    if cfg["head"] != "parallel_v1":
        raise RuntimeError("Parallel parity requires a parallel_v1 config")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError("BF16 checkpoint SHA-256 does not match --checkpoint-sha256")
    config_hash = sha256(args.config)
    validate_qualification(read_json(args.qualification_receipt), cfg, config_hash, checkpoint_hash)

    head = ParallelHead.from_config(cfg).bfloat16().eval()
    expected_head = set(head.state_dict())
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if (
            metadata is None
            or metadata.get("format") != EXPORT_FORMAT
            or metadata.get("parallel_config_sha256") != config_hash
        ):
            raise RuntimeError("Parity source is not a qualified parallel BF16 export")
        head_names = {
            name.removeprefix("parallel_head.")
            for name in checkpoint.keys()
            if name.startswith("parallel_head.")
        }
        if head_names != expected_head:
            raise RuntimeError("BF16 checkpoint parallel head tensors are not exact")
        state = {
            name: checkpoint.get_tensor(f"parallel_head.{name}").bfloat16()
            for name in expected_head
        }
        text_weight = checkpoint.get_tensor("text_emb.weight").bfloat16()
    head.load_state_dict(state, strict=True)
    if tuple(text_weight.shape) != (int(cfg["text_card"]) + 1, int(cfg["dim"])):
        raise RuntimeError("BF16 checkpoint text_emb.weight shape changed")

    hidden = ((torch.arange(2048, dtype=torch.float32) % 257) - 128) / 128
    hidden = hidden.view(1, 1, 2048).bfloat16()
    text_ids = torch.tensor([[17]], dtype=torch.int32)
    text_embedding = F.embedding(text_ids.long(), text_weight)
    previous = torch.tensor(
        [[[0, 1, 31, 32, 255, 511, 1023, int(cfg["card"]) - 1]]],
        dtype=torch.int32,
    )
    with torch.inference_mode():
        logits = head(hidden, text_embedding, previous)
    next_previous = logits.argmax(dim=-1)[:, -1].to(torch.int32)
    shapes = {
        "hidden": [1, 1, 2048],
        "text_ids": [1, 1],
        "text_embedding": [1, 1, 2048],
        "previous_codes": [1, 1, 8],
        "logits": [1, 1, 8, 2048],
        "next_previous_codes": [1, 8],
    }
    fixture_metadata = {
        "format": PARITY_FORMAT,
        "architecture": cfg["architecture"],
        "head": "parallel_v1",
        "head_passes": cfg["head_passes"],
        "config_sha256": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "reference_dtype": "bfloat16",
        "state_rule": "initial_card_then_previous_raw_pre_undelay_head_output",
        "shapes": shapes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez(
            handle,
            metadata_json=np.array(json.dumps(fixture_metadata, sort_keys=True)),
            hidden=hidden.float().numpy(),
            text_ids=text_ids.numpy(),
            text_embedding=text_embedding.float().numpy(),
            previous_codes=previous.numpy(),
            logits=logits.float().numpy(),
            next_previous_codes=next_previous.numpy(),
        )
    print(f"PASS: wrote deterministic parallel parity fixture {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    make_fixture(parse_args())
