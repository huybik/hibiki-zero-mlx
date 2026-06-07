#!/usr/bin/env python
"""Run the fast MLX pipeline with full bf16 LM weights instead of q4."""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import rustymimi
import sentencepiece

HERE = Path(__file__).resolve().parent.parent  # repo root (scripts/ -> ..)
W = HERE / "weights"
VENDORED_MOSHI_MLX = HERE / "moshi-mlx"
if VENDORED_MOSHI_MLX.exists():
    sys.path.insert(0, str(VENDORED_MOSHI_MLX))
sys.path.insert(0, str(HERE / "src"))

from moshi_mlx import models
import infer_mlx_fast as f


def load_bf16(weights_dir: Path = W, use_mlx_weights: bool = True):
    cfg = json.loads((weights_dir / "config.json").read_text())
    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if use_mlx_weights:
        model.load_weights(str(weights_dir / "hibiki.bf16.safetensors"), strict=True)
    else:
        model.load_pytorch_weights(
            str(weights_dir / "hibiki-pytorch-77f82164@110.safetensors"),
            lm_config,
            strict=True,
        )
    mx.eval(model.parameters())
    tok = sentencepiece.SentencePieceProcessor(
        str(weights_dir / "tokenizer_spm_48k_multi6_2.model"))
    mimi_path = str(weights_dir / "mimi-pytorch-e351c8d8@125.safetensors")
    nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
    return (model, lm_config, tok,
            rustymimi.Tokenizer(mimi_path, num_codebooks=nq),
            rustymimi.Tokenizer(mimi_path, num_codebooks=nq))


def main():
    parser = argparse.ArgumentParser(description="MLX bf16 Hibiki-Zero translation")
    parser.add_argument("input", nargs="?", default=str(HERE / "hibiki_zero" / "samples" / "leon.wav"))
    parser.add_argument(
        "-o",
        "--out",
        default=str(HERE / "translations" / "leon_mlx_bf16.wav"),
        help="output wav path",
    )
    parser.add_argument(
        "--text-out",
        help="output transcript path; default matches output wav with .txt",
    )
    parser.add_argument(
        "--from-pytorch",
        action="store_true",
        help="load the original PyTorch safetensors instead of weights/hibiki.bf16.safetensors",
    )
    args = parser.parse_args()

    f.load = lambda weights_dir=W: load_bf16(
        weights_dir,
        use_mlx_weights=not args.from_pytorch,
    )
    mx.random.seed(299792458)
    f.run(args.input, args.out, text_outfile=args.text_out)


if __name__ == "__main__":
    main()
