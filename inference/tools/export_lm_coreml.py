"""Export the parallel-heads student's LM step (`lm_step.py`) as compiled Core ML models for the
Apple Neural Engine: fp16, one frame per call, the KV cache as state (iOS 18).

    python inference/tools/export_lm_coreml.py MODEL_DIR OUT_DIR [--bits 6]

MODEL_DIR is the student staged by `convert_mlx.py` in bf16. OUT_DIR becomes a model directory
for the Swift runtime (`NeuralLM`): `lm-trunk.mlmodelc` (the main transformer and the text head),
`lm-heads.mlmodelc` (the 16 heads), `lm-embeddings.bin` (the input tables in fp16, which the host
sums: text_emb, then each audio_embs), the tokenizer and Mimi checkpoint, and MODEL_DIR's
`config.json` with `lm_coreml` in place of the MLX weights (`moshi_name`). Mimi's decoder comes
from `export_mimi_coreml.py`, as for any model directory. `--bits N` palettizes the trunk's and the
heads' weights to N bits (k-means, one table per 16 output channels); without it they stay fp16.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch

import lm_step as ls
from export_mimi_coreml import convert, save_compiled


def _palettize(model: ct.models.MLModel, bits: int) -> ct.models.MLModel:
    op = cto.OpPalettizerConfig(nbits=bits, mode="kmeans", granularity="per_grouped_channel",
                                group_size=16, weight_threshold=1 << 20)
    return cto.palettize_weights(model, cto.OptimizationConfig(global_config=op))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", type=Path, help="the student staged in bf16")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), help="palettize to this many bits")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    cfg, w = ls.load(args.model_dir)

    ls.embeddings(cfg, w).tofile(args.out_dir / "lm-embeddings.bin")
    config = json.loads((args.model_dir / "config.json").read_text())
    for name in (config["tokenizer_name"], config["mimi_name"]):
        shutil.copy(args.model_dir / name, args.out_dir / name)
    del config["moshi_name"]
    config["lm_coreml"] = {"bits": args.bits}

    half = cfg.head_dim // 2
    cache = (cfg.layers, cfg.kv_heads, cfg.context, cfg.head_dim)
    trunk = convert(
        ls.Trunk(cfg, w),
        {
            "x": ((1, cfg.dim), np.float16),
            "cos": ((1, half), np.float16),
            "sin": ((1, half), np.float16),
            "mask": ((1, cfg.context), np.float16),
            "slot": ((1,), np.int32),
        },
        ["hidden", "text_logits"],
        {"keys": cache, "values": cache},
    )
    heads = convert(
        ls.Heads(cfg, w),
        {"hidden": ((1, cfg.dim), np.float16), "text": ((1, cfg.dim), np.float16)},
        ["audio_logits"],
    )
    if args.bits:
        trunk, heads = _palettize(trunk, args.bits), _palettize(heads, args.bits)
    save_compiled(trunk, args.out_dir / "lm-trunk.mlmodelc")
    save_compiled(heads, args.out_dir / "lm-heads.mlmodelc")
    # Last: a directory whose config names a Core ML LM holds one.
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
