#!/usr/bin/env python
"""Convert hibiki-zero PyTorch LM weights to 4-bit MLX safetensors via moshi_mlx."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"  # scripts/ -> ..
CONFIG = WEIGHTS / "config.json"
PTH = WEIGHTS / "hibiki-pytorch-77f82164@110.safetensors"
OUT = WEIGHTS / "hibiki.q4.safetensors"

with open(CONFIG) as f:
    cfg = json.load(f)

lm_config = models.LmConfig.from_config_dict(cfg)
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)

print(f"loading PyTorch weights from {PTH.name} ...")
model.load_pytorch_weights(str(PTH), lm_config, strict=True)

print("quantizing to 4-bit (group_size=32) ...")
nn.quantize(model, bits=4, group_size=32)

print(f"saving {OUT.name} ...")
model.save_weights(str(OUT))

mb = OUT.stat().st_size / 1e6
print(f"done: {OUT} ({mb:.0f} MB)")
