#!/usr/bin/env python
"""Produce the q4-depq3 variant: depformer slice transformers 3-bit, rest 4-bit (all gs32).

The slice embeddings and linear_out stay q4: quantizing them to 3-bit makes the
3B babble/loop through the tail flush (see reports/inference_matrix.md).

No M4 speedup (depformer is launch-bound) but a smaller artifact -> phone
bandwidth/size win. Writes next to the model's q4 weights so
`hibiki_mlx.load(dir, quant="q4-depq3")` picks it up.

  python scripts/convert_depq3.py --model 3b   # from the PyTorch checkpoint
  python scripts/convert_depq3.py --model 1b   # from the staged MLX bf16 dir
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from hibiki_mlx.pipeline import _q4_model_name, _quant_predicate
from moshi_mlx import models

W = Path(__file__).resolve().parent.parent / "weights"  # scripts/ -> ..


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="3b", choices=["3b", "1b"])
    args = ap.parse_args()

    if args.model == "3b":
        out_dir = W
        src = W / "hibiki-pytorch-77f82164@110.safetensors"
    else:
        out_dir = W / "hibiki-m-mlx-q4"
        src = W / "hibiki-m-mlx-bf16" / "hibiki-mlx-dc2cf5a5@80.safetensors"

    cfg = json.loads((out_dir / "config.json").read_text())
    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    print(f"loading source weights from {src.name} ...", flush=True)
    if args.model == "3b":
        model.load_pytorch_weights(str(src), lm_config, strict=True)
    else:
        model.load_weights(str(src), strict=True)

    print("quantizing main q4 / depformer q3 (gs32) ...", flush=True)
    nn.quantize(model, bits=4, group_size=32, class_predicate=_quant_predicate("q4-depq3"))

    out = out_dir / _q4_model_name(cfg, "q4-depq3")
    model.save_weights(str(out))
    print(f"done: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
