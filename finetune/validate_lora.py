#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.eval_lora import load_main_lora  # noqa: E402
from finetune.train_lora import (  # noqa: E402
    check_device,
    dtype_from_name,
    evaluate_teacher_forced,
    make_cached_dataloader,
    make_dataset_class,
)
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
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
    parser = argparse.ArgumentParser(
        description="Compute teacher-forced CE on cached codes for a base model or LoRA adapter."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT / "validation")
    parser.add_argument("--adapter", type=Path, help="Optional adapter .safetensors file.")
    parser.add_argument("--out-json", type=Path, help="Optional metrics JSON path.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model/LoRA dtype.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all selected batches.")
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sort-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort cached samples by frame length to reduce padding.",
    )
    return parser.parse_args()


def require_runtime_deps() -> tuple[Any, ...]:
    try:
        import torch
        from moshi.models import loaders
        from moshi.modules.lora import replace_all_linear_with_lora
        from moshi.run_inference import get_condition_tensors
        from safetensors import safe_open
        from safetensors.torch import load_file
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(f"Missing validation dependency: {exc.name}") from exc
    return (
        torch,
        DataLoader,
        Dataset,
        loaders,
        replace_all_linear_with_lora,
        get_condition_tensors,
        safe_open,
        load_file,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.max_batches < 0:
        raise ValueError("--max-batches must be non-negative")

    (
        torch,
        DataLoader,
        Dataset,
        loaders,
        replace_all_linear_with_lora,
        get_condition_tensors,
        safe_open,
        load_file,
    ) = require_runtime_deps()
    device = check_device(torch, args.device)
    dtype = dtype_from_name(torch, args.dtype)

    cache_dir = require_dir(args.cache_dir, "code cache directory")
    adapter = require_file(args.adapter, "LoRA adapter") if args.adapter else None
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")

    DatasetClass = make_dataset_class(torch, Dataset)
    dataset = DatasetClass(cache_dir, args.sort_by_length, args.max_samples)
    dataloader = make_cached_dataloader(
        DataLoader,
        dataset,
        torch,
        args.batch_size,
        args.num_workers,
        args.sort_by_length,
    )
    print(f"Loaded {len(dataset)} cached samples from {repo_display_path(cache_dir)}")

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
    if adapter is not None:
        load_main_lora(
            lm,
            adapter,
            torch,
            safe_open,
            load_file,
            replace_all_linear_with_lora,
            device,
            dtype,
        )
    else:
        print("Using base model without adapter.")

    metrics = evaluate_teacher_forced(
        torch,
        lm,
        dataloader,
        device,
        checkpoint_info.model_type,
        get_condition_tensors,
        args.audio_loss_weight,
        args.text_loss_weight,
        args.max_batches,
    )
    result = {
        "cache_dir": repo_display_path(cache_dir),
        "adapter": repo_display_path(adapter) if adapter is not None else "",
        "loss": metrics["loss"],
        "audio_loss": metrics["audio_loss"],
        "text_loss": metrics["text_loss"],
        "audio_tokens": metrics["audio_tokens"],
        "text_tokens": metrics["text_tokens"],
        "batches": metrics["batches"],
        "samples": metrics["samples"],
    }
    print(
        f"loss={metrics['loss']:.4f} "
        f"audio={metrics['audio_loss']:.4f} text={metrics['text_loss']:.4f} "
        f"samples={metrics['samples']}"
    )
    if args.out_json:
        out_path = resolve_repo_path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote metrics -> {repo_display_path(out_path)}")


if __name__ == "__main__":
    main()
