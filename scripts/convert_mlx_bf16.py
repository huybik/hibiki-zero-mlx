#!/usr/bin/env python
"""Convert hibiki-zero PyTorch LM weights to bf16 MLX safetensors via moshi_mlx.

Same as convert_mlx_q4.py without the quantization step: produces a native MLX
bf16 checkpoint that loads without the runtime PyTorch->MLX conversion."""
import json
from pathlib import Path

import mlx.core as mx
from moshi_mlx import models

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"  # scripts/ -> ..
CONFIG = WEIGHTS / "config.json"
PTH = WEIGHTS / "hibiki-pytorch-77f82164@110.safetensors"
OUT = WEIGHTS / "hibiki.bf16.safetensors"

with open(CONFIG) as f:
    cfg = json.load(f)

lm_config = models.LmConfig.from_config_dict(cfg)
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)

print(f"loading PyTorch weights from {PTH.name} ...")
model.load_pytorch_weights(str(PTH), lm_config, strict=True)

print(f"saving {OUT.name} ...")
model.save_weights(str(OUT))

mb = OUT.stat().st_size / 1e6
print(f"done: {OUT} ({mb:.0f} MB)")
