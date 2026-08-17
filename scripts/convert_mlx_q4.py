#!/usr/bin/env python
"""Convert one explicit qualified parallel BF16 student into a strict q4 pack."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from hibiki_mlx.student_pack import (
    MIMI_SHA256,
    TOKENIZER_SHA256,
    expected_shape_receipt,
    make_student_manifest,
    parity_metadata,
    read_json,
    sha256,
    validate_qualification,
    validate_student_config,
    validate_student_pack,
)
from moshi_mlx import models
from safetensors import safe_open

BF16_EXPORT_FORMAT = "hibiki_parallel_bf16_export_v1"


def q4_compatible(_: str, module: object) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and hasattr(module, "to_quantized") and weight.shape[-1] % 32 == 0


def convert(args: argparse.Namespace) -> None:
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite student pack: {args.out_dir}")
    cfg = read_json(args.config)
    validate_student_config(cfg)
    if cfg["head"] != "parallel_v1":
        raise RuntimeError("This conversion path requires a qualified parallel_v1 checkpoint")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError("BF16 checkpoint SHA-256 does not match --checkpoint-sha256")
    config_hash = sha256(args.config)
    validate_qualification(read_json(args.qualification_receipt), cfg, config_hash, checkpoint_hash)
    if read_json(args.shape_receipt) != expected_shape_receipt(cfg):
        raise RuntimeError("Shape receipt does not match the exact parallel config")
    if sha256(args.mimi) != MIMI_SHA256 or sha256(args.tokenizer) != TOKENIZER_SHA256:
        raise RuntimeError("Mimi/tokenizer hashes differ from the frozen student contract")
    fixture = parity_metadata(args.parity_fixture, cfg, config_hash)
    if fixture["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Parity fixture was generated from a stale BF16 checkpoint")
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        if (
            metadata is None
            or metadata.get("format") != BF16_EXPORT_FORMAT
            or metadata.get("parallel_config_sha256") != config_hash
        ):
            raise RuntimeError("BF16 export metadata does not match the parallel config")

    parent = args.out_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.out_dir.name}.", dir=parent))
    try:
        shutil.copy2(args.config, temporary / "config.json")
        shutil.copy2(args.mimi, temporary / cfg["mimi_name"])
        shutil.copy2(args.tokenizer, temporary / cfg["tokenizer_name"])
        shutil.copy2(args.parity_fixture, temporary / cfg["parity_fixture_name"])
        shutil.copy2(args.shape_receipt, temporary / "shape_receipt.json")
        shutil.copy2(args.qualification_receipt, temporary / "qualification_receipt.json")

        lm_config = models.LmConfig.from_config_dict(cfg)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        print(f"loading qualified PyTorch weights from {args.checkpoint} ...")
        model.load_pytorch_weights(str(args.checkpoint), lm_config, strict=True)
        model.set_dtype(mx.bfloat16)
        print("quantizing to q4 group_size=32 ...")
        nn.quantize(model, bits=4, group_size=32, class_predicate=q4_compatible)
        mx.eval(model.parameters())
        model.save_weights(str(temporary / cfg["moshi_name"]))

        manifest = make_student_manifest(temporary)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_student_pack(temporary)
        temporary.replace(args.out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"PASS: wrote strict q4 group-size-32 student pack {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--shape-receipt", type=Path, required=True)
    parser.add_argument("--parity-fixture", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
