"""Export a phone student's LM step (`lm_step.py`) as compiled Core ML models for the Apple Neural
Engine: fp16, one frame per call, the KV cache as state (iOS 18).

    python inference/tools/export_lm_coreml.py MODEL_DIR OUT_DIR [--bits 6]

MODEL_DIR is the student staged by `convert_mlx.py` in bf16. OUT_DIR becomes a model directory
for the Swift runtime (`NeuralLM`): `lm-trunk.mlmodelc` (the main transformer and the text head),
`lm-embeddings.bin` (the input tables in fp16, which the host sums: text_emb, then each
audio_embs), the audio head, the tokenizer and Mimi checkpoint, and MODEL_DIR's `config.json` with
`lm_coreml` in place of the MLX weights (`moshi_name`). The audio head is `lm-heads.mlmodelc` (the
16 parallel heads) or, for a `depformer_parallel` model, a single fused `lm-depformer.mlmodelc`
(all slices with in-graph top-k Gumbel-max sampling) and slice 0's table in
`lm-depformer-embeddings.bin` (or `--per-slice` for separate `lm-depformer-K.mlmodelc` models).
Mimi's decoder comes from `export_mimi_coreml.py`, as for any model directory. `--bits N`
palettizes every model's weights to N bits (k-means, one table per 16 output channels); `--trunk-bits`
overrides the trunk's bit width; without `--bits` weights stay fp16.
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
                                group_size=16, weight_threshold=1 << 16)
    return cto.palettize_weights(model, cto.OptimizationConfig(global_config=op))


def export_audio_head(
    cfg: ls.Config,
    w: dict[str, torch.Tensor],
    out_dir: Path,
    bits: int | None,
    *,
    top_k: int | None = None,
    per_slice: bool = False,
) -> dict:
    """The heads, or the depformer (one fused model by default, or per-slice with `per_slice`)
    and its host input table(s), into `out_dir`. Returns extra fields for `config["lm_coreml"]`."""
    if cfg.depformer is None:
        heads = convert(
            ls.Heads(cfg, w),
            {"hidden": ((1, cfg.dim), np.float16), "text": ((1, cfg.dim), np.float16)},
            ["audio_logits"],
        )
        save_compiled(_palettize(heads, bits) if bits else heads, out_dir / "lm-heads.mlmodelc")
        return {}
    d = cfg.depformer
    if per_slice:
        ls.depformer_embeddings(cfg, w).tofile(out_dir / "lm-depformer-embeddings.bin")
        for k in range(d.slices(cfg.dep_q)):
            inputs = {"hidden": ((1, cfg.dim), np.float16), "emb": ((1, d.dim), np.float16)}
            if k:
                past = (d.layers, d.heads, k, d.head_dim)
                inputs |= {"keys": (past, np.float16), "values": (past, np.float16)}
            model = convert(ls.DepformerSlice(cfg, w, k), inputs, ["logits", "key", "value"])
            save_compiled(_palettize(model, bits) if bits else model, out_dir / f"lm-depformer-{k}.mlmodelc")
        return {}
    ls.depformer_text_embeddings(cfg, w).tofile(out_dir / "lm-depformer-embeddings.bin")
    dep = ls.DepformerStep(cfg, w, top_k=top_k)
    model = convert(
        dep,
        {
            "hidden": ((1, cfg.dim), np.float16),
            "emb": ((1, d.dim), np.float16),
            "noise": ((cfg.dep_q, cfg.card), np.float16),
        },
        ["tokens"],
    )
    save_compiled(_palettize(model, bits) if bits else model, out_dir / "lm-depformer.mlmodelc")
    return {"depformer_top_k": dep.top_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir", type=Path, help="the student staged in bf16")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), help="palettize to this many bits")
    parser.add_argument("--trunk-bits", type=int, choices=(4, 6, 8),
                        help="override --bits for the trunk only (e.g. 4-bit trunk with 6-bit audio head)")
    parser.add_argument("--per-slice", action="store_true",
                        help="export depformer_parallel as separate lm-depformer-K models instead of one fused model")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    cfg, w = ls.load(args.model_dir)
    trunk_bits = args.trunk_bits if args.trunk_bits is not None else args.bits

    ls.embeddings(cfg, w).tofile(args.out_dir / "lm-embeddings.bin")
    config = json.loads((args.model_dir / "config.json").read_text())
    for name in (config["tokenizer_name"], config["mimi_name"]):
        shutil.copy(args.model_dir / name, args.out_dir / name)
    del config["moshi_name"]
    lm_coreml: dict = {"bits": args.bits}
    if trunk_bits != args.bits:
        lm_coreml["trunk_bits"] = trunk_bits

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
    save_compiled(_palettize(trunk, trunk_bits) if trunk_bits else trunk, args.out_dir / "lm-trunk.mlmodelc")
    top_k = config.get("lm_gen_config", {}).get("top_k")
    lm_coreml |= export_audio_head(cfg, w, args.out_dir, args.bits, top_k=top_k, per_slice=args.per_slice)
    config["lm_coreml"] = lm_coreml
    # Last: a directory whose config names a Core ML LM holds one.
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
