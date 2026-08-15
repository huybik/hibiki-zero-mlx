#!/usr/bin/env python
"""Stage a standalone q4 MLX artifact repo for Hibiki-M.

Hibiki-M is already published as MLX bf16 weights, so unlike the Hibiki-Zero
3B converter this script loads the MLX safetensors directly before quantizing.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf_cache"))

from huggingface_hub import hf_hub_download  # noqa: E402
from moshi_mlx import models  # noqa: E402

SOURCE_REPO = "kyutai/hibiki-1b-mlx-bf16"
SOURCE_REVISION = "b3d6291f3dcf7954e1a502e4d66f32e3556f17ae"
SOURCE_HIBIKI_WEIGHTS = "hibiki-mlx-dc2cf5a5@80.safetensors"
Q4_HIBIKI_WEIGHTS = "hibiki-mlx-dc2cf5a5@80.q4.safetensors"
MIMI_WEIGHTS = "mimi-dbaa9758@125.safetensors"
TOKENIZER = "tokenizer_spm_48k_multi6_2.model"
STAGED_FILES = ["config.json", Q4_HIBIKI_WEIGHTS, MIMI_WEIGHTS, TOKENIZER, "README.md", ".gitattributes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=HERE / "weights" / "hibiki-m-mlx-bf16",
        help="directory containing or receiving the source Hibiki-M BF16 files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "weights" / "hibiki-m-mlx-q4",
        help="directory to stage the standalone q4 repo files",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use files already present in --source-dir",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip strict q4 reload verification",
    )
    return parser.parse_args()


def ensure_source_file(source_dir: Path, filename: str, skip_download: bool) -> Path:
    path = source_dir / filename
    if path.exists():
        return path
    if skip_download:
        raise FileNotFoundError(f"missing {path}")
    print(f"downloading {filename} from {SOURCE_REPO}@{SOURCE_REVISION} ...", flush=True)
    return Path(
        hf_hub_download(
            SOURCE_REPO,
            filename,
            revision=SOURCE_REVISION,
            local_dir=source_dir,
        )
    )


def stage_sidecar(source_dir: Path, out_dir: Path, filename: str, skip_download: bool) -> None:
    source = ensure_source_file(source_dir, filename, skip_download)
    destination = out_dir / filename
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def write_config(source_config: Path, out_dir: Path) -> dict:
    with source_config.open() as f:
        config = json.load(f)
    config["moshi_name"] = Q4_HIBIKI_WEIGHTS
    with (out_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    return config


def write_model_card(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        f"""---
license: cc-by-nc-sa-4.0
base_model: {SOURCE_REPO}
tags:
- mlx
- speech-translation
- text-to-speech
- audio
- quantized
---

# Hibiki-M 1B MLX q4

This is a 4-bit MLX quantization of [`{SOURCE_REPO}`](https://huggingface.co/{SOURCE_REPO})
at revision `{SOURCE_REVISION}`.

The language-model weights were quantized with MLX `nn.quantize(bits=4, group_size=32)`,
leaving tiny conditioner tensors dense when their last dimension is not divisible by 32.
Mimi, tokenizer, and model config files are copied from the source repository. The staged
`config.json` changes `moshi_name` to `{Q4_HIBIKI_WEIGHTS}` so compatible loaders can apply
q4 group-size-32 loading before strict weight load.

## Files

- `{Q4_HIBIKI_WEIGHTS}`: quantized Hibiki-M language-model weights
- `{MIMI_WEIGHTS}`: source Mimi codec weights
- `{TOKENIZER}`: source sentencepiece tokenizer
- `config.json`: source architecture config with q4 `moshi_name`

This derivative keeps the upstream non-commercial share-alike license.
""",
        encoding="utf-8",
    )


def write_gitattributes(out_dir: Path) -> None:
    (out_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )


def q4_compatible(_: str, module: object) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(module, "to_quantized"):
        return False
    return weight.shape[-1] % 32 == 0


def quantize(config: dict, source_weights: Path, out_weights: Path) -> None:
    lm_config = models.LmConfig.from_config_dict(config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    print(f"loading MLX BF16 weights from {source_weights} ...", flush=True)
    model.load_weights(str(source_weights), strict=True)

    print("quantizing to 4-bit (group_size=32) ...", flush=True)
    nn.quantize(model, bits=4, group_size=32, class_predicate=q4_compatible)

    print(f"saving {out_weights} ...", flush=True)
    model.save_weights(str(out_weights))


def verify_q4_load(config: dict, out_weights: Path) -> None:
    lm_config = models.LmConfig.from_config_dict(config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32, class_predicate=q4_compatible)
    print(f"verifying strict q4 reload from {out_weights.name} ...", flush=True)
    model.load_weights(str(out_weights), strict=True)


def main() -> None:
    args = parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_config = ensure_source_file(args.source_dir, "config.json", args.skip_download)
    source_weights = ensure_source_file(args.source_dir, SOURCE_HIBIKI_WEIGHTS, args.skip_download)

    config = write_config(source_config, args.out_dir)
    for filename in (MIMI_WEIGHTS, TOKENIZER):
        stage_sidecar(args.source_dir, args.out_dir, filename, args.skip_download)
    write_model_card(args.out_dir)
    write_gitattributes(args.out_dir)

    out_weights = args.out_dir / Q4_HIBIKI_WEIGHTS
    quantize(config, source_weights, out_weights)
    if not args.no_verify:
        verify_q4_load(config, out_weights)

    print("staged files:", ", ".join(STAGED_FILES), flush=True)
    print(f"staged repo directory: {args.out_dir}", flush=True)
    print(f"q4 size: {out_weights.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
