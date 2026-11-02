#!/usr/bin/env python
"""Teacher-forced losses of a base or full-SFT model on the validation streams of a cache.

The streams and batches are planned as `train.py` plans its validation (fixed seed, silence track
encoded on the CPU). To reproduce the `teacher_forced` metrics of a run's `val_log.jsonl`, pass its
--stream-seconds, --max-frames and --batch-frames and use --dtype float32, as training validates
its fp32 weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.finetune import common, streams
from training.finetune.train import stream_frames
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional full-model .safetensors checkpoint."
    )
    parser.add_argument("--out-json", type=Path, help="Optional metrics JSON path.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model dtype.",
    )
    parser.add_argument("--batch-frames", type=int, default=4608)
    parser.add_argument(
        "--stream-seconds", type=float, nargs=2, default=list(streams.STREAM_SECONDS)
    )
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means every batch.")
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--first-codebook-weight", type=float, default=100.0)
    parser.add_argument("--text-pad-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = common.check_device(args.device)
    dtype = common.dtype_from_name(args.dtype)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls, as training validates

    cache_dir = require_dir(args.cache_dir, "code cache directory")
    checkpoint = require_file(args.checkpoint, "checkpoint") if args.checkpoint else None
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")

    checkpoint_info = common.load_checkpoint_info(args)
    frames = stream_frames(args.stream_seconds)
    silence = streams.silence_track(checkpoint_info.get_mimi(device="cpu"), frames[1])
    rows, layout = streams.load_rows([cache_dir], args.max_frames)
    dataset = streams.StreamDataset(rows, layout, silence, frames, streams.VALIDATION_SEED)
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.batches(args.batch_frames, streams.VALIDATION_SEED),
        collate_fn=streams.collate,
        num_workers=args.num_workers,
    )
    print(f"Planned {len(dataset)} streams from {len(rows)} rows of {repo_display_path(cache_dir)}")

    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
    if checkpoint is not None:
        common.load_model(lm, checkpoint, dtype)
    else:
        print("Using the base model.")

    weights = common.LossWeights(
        audio=args.audio_loss_weight,
        text=args.text_loss_weight,
        first_codebook=args.first_codebook_weight,
        text_pad=args.text_pad_weight,
    )
    metrics = common.evaluate_teacher_forced(
        lm, dataloader, device, checkpoint_info.model_type, weights, args.max_batches
    )
    result = {
        "cache_dir": repo_display_path(cache_dir),
        "checkpoint": repo_display_path(checkpoint) if checkpoint is not None else "",
        **metrics,
    }
    print(
        f"loss={metrics['loss']:.4f} audio={metrics['audio_loss']:.4f} "
        f"semantic={metrics['semantic_loss']:.4f} text={metrics['text_loss']:.4f} "
        f"content={metrics['content_text_loss']:.4f} pad={metrics['pad_text_loss']:.4f} "
        f"acc={metrics['content_acc']:.3f} pad_acc={metrics['pad_acc']:.3f} "
        f"streams={metrics['streams']}"
    )
    if args.out_json:
        out_path = resolve_repo_path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote metrics -> {repo_display_path(out_path)}")


if __name__ == "__main__":
    main()
