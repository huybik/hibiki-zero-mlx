#!/usr/bin/env python
"""Stage an exact Hibiki PyTorch checkpoint as an MLX model directory, bf16 or (--q4)
quantized, with the text tokenizer the checkpoint is on.

  python inference/tools/convert_mlx.py --checkpoint CKPT.safetensors --config config.json \\
      --mimi mimi-pytorch-e351c8d8@125.safetensors [--tokenizer TOK.model] [--q4] --out-dir DIR

--q4 quantizes the transformer linears (main and depformer layers) to 4 bits and the
embeddings, the text head and each depformer slice's input and output projection to 8 bits,
group size 32, and records that per layer in config.json's `quantization`, which both runtimes
quantize by before loading strictly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from hibiki_mlx.pipeline import W as WEIGHTS
from hibiki_mlx.pipeline import quantize
from moshi_mlx import models

# Upstream's tokenizer, which a checkpoint without `text_tokenizer` is on (training/paths.py
# DEFAULT_TOKENIZER).
BASE_TOKENIZER = "tokenizer_spm_48k_multi6_2.model"
GROUP_SIZE = 32
# The layers 4 bits hurt most, at 8: the embeddings, the text head and the depformer slices'
# input and output projections (q3 there made the 3B babble; docs/vision/reports/
# inference_matrix.md). Every other Linear, the transformer layers', takes 4 bits.
EIGHT_BITS = re.compile(
    r"text_emb|text_linear|audio_embs\.\d+|depformer\.slices\.\d+\.(emb|linear_in|linear_out)"
)


def checkpoint_tokenizer(checkpoint: Path) -> str:
    """The tokenizer file the checkpoint's text rows are on: its safetensors metadata
    `text_tokenizer`, which every training checkpoint records (`weight_tokenizer` in
    training/finetune/common.py), else upstream's (upstream's weight and checkpoints before
    2026-11-10)."""
    with open(checkpoint, "rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    return header.get("__metadata__", {}).get("text_tokenizer", BASE_TOKENIZER)


def q4_record(lm: nn.Module) -> dict:
    """The `quantization` record of the q4 recipe: 4 bits, group size 32 by default, and an
    entry of 8 bits for each EIGHT_BITS layer."""
    record: dict = {"group_size": GROUP_SIZE, "bits": 4}
    for path, module in tree_flatten(lm.leaf_modules(), is_leaf=nn.Module.is_module):
        if hasattr(module, "to_quantized") and EIGHT_BITS.fullmatch(path):
            record[path] = {"group_size": GROUP_SIZE, "bits": 8}
    return record


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
    model_name = "hibiki.q4.safetensors" if args.q4 else "hibiki.bf16.safetensors"
    # A staged model's config may come in: its own record and the keys older stagings wrote
    # describe other weights.
    for key in ("quantization", "weight_dtype", "artifact_format"):
        cfg.pop(key, None)
    cfg["moshi_name"] = model_name
    cfg["tokenizer_name"] = on

    parent = args.out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.out_dir.name}.", dir=parent))
    try:
        lm_config = models.LmConfig.from_config_dict(cfg)
        lm = models.Lm(lm_config)
        print(f"loading PyTorch weights from {args.checkpoint} ...", flush=True)
        lm.load_pytorch_weights(str(args.checkpoint), lm_config, strict=True)
        # Quantized from bf16, so scales and biases are bf16 as the runtimes build them.
        lm.set_dtype(mx.bfloat16)
        if args.q4:
            cfg["quantization"] = q4_record(lm)
            quantize(lm, cfg["quantization"])
        mx.eval(lm.parameters())
        print(f"saving MLX weights to {model_name} ...", flush=True)
        lm.save_weights(str(temporary / model_name))

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
    size = (args.out_dir / model_name).stat().st_size / 1e9
    print(f"PASS: wrote MLX model {args.out_dir} ({model_name}, {size:.2f} GB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path,
                        help="the tokenizer the checkpoint is on (default: that file in weights/)")
    parser.add_argument("--q4", action="store_true",
                        help="quantize: 4 bits on the transformer linears, 8 on the embeddings "
                        "and heads, group size 32 (default: bf16)")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
