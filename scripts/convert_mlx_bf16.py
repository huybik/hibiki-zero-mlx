#!/usr/bin/env python
"""Convert an exact Hibiki PyTorch checkpoint into a staged MLX bf16 model."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

from moshi_mlx import models

MODEL_NAME = "hibiki.bf16.safetensors"
ARTIFACT_FORMAT = "hibiki_mlx_bf16_v1"
EXPORT_FORMAT = "hibiki_parallel_bf16_export_v1"


def require_bf16_export(path: Path) -> None:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        if metadata.get("format") != EXPORT_FORMAT or metadata.get("dtype") != "bfloat16":
            raise RuntimeError("Checkpoint is not a parallel BF16 listening export")
        wrong = [
            name
            for name in checkpoint.keys()
            if checkpoint.get_slice(name).get_dtype() != "BF16"
        ]
    if wrong:
        raise RuntimeError(f"BF16 export contains non-BF16 tensors: {wrong[:5]}")


def convert(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model directory: {args.out_dir}")
    cfg = json.loads(args.config.read_text())
    if cfg.get("head") == "parallel_v1":
        require_bf16_export(args.checkpoint)
    cfg["artifact_format"] = ARTIFACT_FORMAT
    cfg["moshi_name"] = MODEL_NAME
    cfg["weight_dtype"] = "bfloat16"

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
        shutil.copy2(args.tokenizer, temporary / cfg["tokenizer_name"])
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
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
