#!/usr/bin/env python
"""Convert an exact Hibiki PyTorch checkpoint into a staged MLX bf16 model, with the text
tokenizer the checkpoint is on."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
from pathlib import Path

import mlx.core as mx

from moshi_mlx import models

MODEL_NAME = "hibiki.bf16.safetensors"
ARTIFACT_FORMAT = "hibiki_mlx_bf16_v1"
WEIGHTS = Path(__file__).resolve().parents[2] / "weights"  # inference/tools/ -> ../..
# Upstream's tokenizer, which a checkpoint without `text_tokenizer` is on (training/paths.py
# DEFAULT_TOKENIZER).
BASE_TOKENIZER = "tokenizer_spm_48k_multi6_2.model"


def checkpoint_tokenizer(checkpoint: Path) -> str:
    """The tokenizer file the checkpoint's text rows are on: its safetensors metadata
    `text_tokenizer`, which every training checkpoint records (`weight_tokenizer` in
    training/finetune/common.py), else upstream's (upstream's weight and checkpoints before
    2026-11-10)."""
    with open(checkpoint, "rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    return header.get("__metadata__", {}).get("text_tokenizer", BASE_TOKENIZER)


def convert(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model directory: {args.out_dir}")
    cfg = json.loads(args.config.read_text())
    on = checkpoint_tokenizer(args.checkpoint)
    tokenizer = args.tokenizer or WEIGHTS / on
    if tokenizer.name != on:
        raise ValueError(f"{args.checkpoint} is on {on}, not --tokenizer {tokenizer.name}")
    if not tokenizer.is_file():
        raise FileNotFoundError(
            f"missing {tokenizer}, the tokenizer {args.checkpoint} is on: "
            "copy it there or pass --tokenizer"
        )
    cfg["artifact_format"] = ARTIFACT_FORMAT
    cfg["moshi_name"] = MODEL_NAME
    cfg["weight_dtype"] = "bfloat16"
    cfg["tokenizer_name"] = on

    parent = args.out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.out_dir.name}.", dir=parent))
    try:
        lm_config = models.LmConfig.from_config_dict(cfg)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        print(f"loading PyTorch weights from {args.checkpoint} ...", flush=True)
        model.load_pytorch_weights(str(args.checkpoint), lm_config, strict=True)
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        print(f"saving MLX bf16 weights to {MODEL_NAME} ...", flush=True)
        model.save_weights(str(temporary / MODEL_NAME))

        (temporary / "config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(args.mimi, temporary / cfg["mimi_name"])
        shutil.copy2(tokenizer, temporary / on)
        temporary.replace(args.out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"PASS: wrote MLX bf16 model {args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path,
                        help="the tokenizer the checkpoint is on (default: that file in weights/)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
