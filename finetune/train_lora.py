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

from finetune.cache_codes import CACHE_FORMAT  # noqa: E402
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_RUN_DIR,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Hibiki-Zero main-transformer LoRA trainer for cached vi->en codes."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT / "train")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-weight", type=Path, default=DEFAULT_MODEL_WEIGHT)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--device", default="mps", help="Torch device. No automatic fallback.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model/LoRA dtype. bfloat16 is the stable smoke-test default on MPS.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-scaling", type=float, default=2.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Optimizer steps, 0 means all.")
    parser.add_argument("--save-every", type=int, default=50, help="Optimizer steps between saves.")
    parser.add_argument("--log-every", type=int, default=1, help="Optimizer steps between logs.")
    parser.add_argument(
        "--mps-empty-cache-every",
        type=int,
        default=10,
        help="Optimizer steps between MPS synchronize+empty_cache calls, 0 disables.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="trainer_step*.pt checkpoint to resume from.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Pass gradient_checkpointing=True when constructing LMModel.",
    )
    parser.add_argument(
        "--sort-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort cached samples by frame length to reduce padding.",
    )
    return parser.parse_args()


def require_runtime_deps() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
        from torch.utils.data import DataLoader, Dataset
        from moshi.models import loaders
        from moshi.modules.lora import replace_all_linear_with_lora
        from moshi.run_inference import get_condition_tensors
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency: {exc.name}") from exc
    return (
        torch,
        load_file,
        save_file,
        DataLoader,
        Dataset,
        loaders,
        replace_all_linear_with_lora,
        get_condition_tensors,
    )


def check_device(torch: Any, device_name: str) -> Any:
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but torch.backends.mps is not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_dataset_class(torch: Any, Dataset: Any) -> Any:
    class CachedCodeDataset(Dataset):
        def __init__(self, cache_dir: Path, sort_by_length: bool):
            self.samples: list[dict[str, Any]] = []
            for shard_path in sorted(cache_dir.glob("shard_*.pt")):
                payload = torch.load(shard_path, map_location="cpu")
                if payload.get("format") != CACHE_FORMAT:
                    raise RuntimeError(f"Unsupported cache format in {shard_path}")
                for sample in payload["samples"]:
                    codes = sample["codes"]
                    if codes.ndim != 2:
                        raise RuntimeError(
                            f"{shard_path} id={sample.get('id')} codes must be [K,T]"
                        )
                    self.samples.append(
                        {
                            "id": str(sample["id"]),
                            "codes": codes.long(),
                            "frames": int(codes.shape[1]),
                        }
                    )
            if not self.samples:
                raise RuntimeError(f"No shard_*.pt cache files found in {cache_dir}")
            if sort_by_length:
                self.samples.sort(key=lambda sample: sample["frames"])

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.samples[index]

    return CachedCodeDataset


def collate_cached(samples: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    codebooks = int(samples[0]["codes"].shape[0])
    max_frames = max(int(sample["codes"].shape[1]) for sample in samples)
    batch = torch.full((len(samples), codebooks, max_frames), -1, dtype=torch.long)
    ids: list[str] = []
    for index, sample in enumerate(samples):
        codes = sample["codes"]
        batch[index, :, : codes.shape[1]] = codes
        ids.append(sample["id"])
    return {"codes": batch, "ids": ids}


def apply_main_lora(
    model: Any, replace_all_linear_with_lora: Any, args: argparse.Namespace
) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    replace_all_linear_with_lora(model.transformer, args.lora_rank, args.lora_scaling)
    for name, param in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        param.requires_grad_(is_lora and name.startswith("transformer."))

    bad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("transformer.")
    ]
    if bad:
        raise RuntimeError(
            f"LoRA freeze map leaked trainable params outside transformer: {bad[:5]}"
        )


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def adapter_state_dict(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if ".lora_A." in name or ".lora_B." in name:
            state[name] = tensor.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No LoRA tensors found to save.")
    return state


def save_checkpoint(
    torch: Any,
    save_file: Any,
    model: Any,
    optimizer: Any,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / f"adapter_step{step:06d}.safetensors"
    metadata = {
        "lora_rank": str(args.lora_rank),
        "lora_scaling": str(args.lora_scaling),
        "target": "LMModel.transformer",
        "base_model": repo_display_path(args.model_weight),
    }
    save_file(adapter_state_dict(model), str(adapter_path), metadata=metadata)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "adapter": str(adapter_path),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        out_dir / f"trainer_step{step:06d}.pt",
    )


def load_resume_checkpoint(
    torch: Any,
    load_file: Any,
    model: Any,
    optimizer: Any,
    resume_path: Path,
    device: Any,
    dtype: Any,
) -> int:
    checkpoint = torch.load(resume_path, map_location="cpu")
    adapter_path = require_file(checkpoint["adapter"], "resume adapter")
    adapter_state = load_file(str(adapter_path), device="cpu")
    for key, value in adapter_state.items():
        if value.dtype.is_floating_point:
            adapter_state[key] = value.to(dtype=dtype)
    result = model.load_state_dict(adapter_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys while resuming: {result.unexpected_keys[:5]}")

    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)

    step = int(checkpoint["step"])
    print(f"Resumed step {step} from {repo_display_path(resume_path)}")
    return step


def resume_batch_offset(global_step: int, grad_accum_steps: int, dataloader_len: int) -> int:
    if dataloader_len <= 0:
        return 0
    return (global_step * grad_accum_steps) % dataloader_len


def masked_cross_entropy(torch: Any, logits: Any, targets: Any, mask: Any) -> Any:
    if not bool(mask.any()):
        return logits.float().sum() * 0.0
    return torch.nn.functional.cross_entropy(logits[mask].float(), targets[mask].long())


def batch_condition_tensors(
    lm: Any, model_type: str, batch_size: int, get_condition_tensors: Any
) -> Any | None:
    if lm.fuser is None:
        return None
    return get_condition_tensors(model_type, lm, batch_size=batch_size, cfg_coef=1.0)


def mps_memory_stats(torch: Any, device: Any) -> dict[str, float]:
    if getattr(device, "type", str(device)) != "mps":
        return {}
    return {
        "mps_allocated_gb": torch.mps.current_allocated_memory() / 1024**3,
        "mps_driver_gb": torch.mps.driver_allocated_memory() / 1024**3,
        "mps_recommended_gb": torch.mps.recommended_max_memory() / 1024**3,
    }


def empty_mps_cache(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) != "mps":
        return
    torch.mps.synchronize()
    torch.mps.empty_cache()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")

    (
        torch,
        load_file,
        save_file,
        DataLoader,
        Dataset,
        loaders,
        replace_all_linear_with_lora,
        get_condition_tensors,
    ) = require_runtime_deps()
    device = check_device(torch, args.device)
    dtype = dtype_from_name(torch, args.dtype)

    cache_dir = require_dir(args.cache_dir, "code cache directory")
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DatasetClass = make_dataset_class(torch, Dataset)
    dataset = DatasetClass(cache_dir, args.sort_by_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.sort_by_length,
        num_workers=args.num_workers,
        collate_fn=lambda samples: collate_cached(samples, torch),
    )

    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
    print(f"Loading LM on {device} from {repo_display_path(args.model_weight)}")
    lm = checkpoint_info.get_moshi(
        device=device,
        dtype=dtype,
        lm_kwargs_overrides={"gradient_checkpointing": args.gradient_checkpointing},
    )
    lm.train()
    apply_main_lora(lm, replace_all_linear_with_lora, args)

    params = trainable_parameters(lm)
    if not params:
        raise RuntimeError("No trainable LoRA parameters after freeze map.")
    trainable_count = sum(param.numel() for param in params)
    total_count = sum(param.numel() for param in lm.parameters())
    print(f"Trainable LoRA params: {trainable_count:,} / {total_count:,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    global_step = 0
    micro_step = 0
    resume_skip_batches = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(
            torch, load_file, lm, optimizer, args.resume_checkpoint, device, dtype
        )
        micro_step = global_step * args.grad_accum_steps
        if args.sort_by_length:
            resume_skip_batches = resume_batch_offset(
                global_step, args.grad_accum_steps, len(dataloader)
            )
            if resume_skip_batches:
                print(f"Skipping {resume_skip_batches} sorted batches already covered by resume.")
        else:
            print("Resume with shuffled data starts a fresh shuffle; use --sort-by-length to continue deterministically.")
    optimizer.zero_grad(set_to_none=True)

    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    log_path = out_dir / "train_log.jsonl"
    condition_cache: dict[int, Any | None] = {}

    for epoch in range(args.epochs):
        for batch in dataloader:
            if resume_skip_batches:
                resume_skip_batches -= 1
                continue
            if args.max_steps and global_step >= args.max_steps:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(
                    lm, checkpoint_info.model_type, batch_size, get_condition_tensors
                )
            condition_tensors = condition_cache[batch_size]
            output = lm(codes, condition_tensors=condition_tensors)

            audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
            text_targets = codes[:, :1]
            audio_loss = masked_cross_entropy(torch, output.logits, audio_targets, output.mask)
            text_mask = output.text_mask & (text_targets != lm.text_padding_token_id)
            text_loss = masked_cross_entropy(
                torch, output.text_logits, text_targets, text_mask
            )
            loss = args.audio_loss_weight * audio_loss + args.text_loss_weight * text_loss
            if not bool(torch.isfinite(loss.detach()).cpu()):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch + 1} micro_step={micro_step + 1}. "
                    "On MPS, use --dtype bfloat16 or lower --lr."
                )
            (loss / args.grad_accum_steps).backward()
            micro_step += 1

            if micro_step % args.grad_accum_steps != 0:
                continue

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                args.mps_empty_cache_every
                and device.type == "mps"
                and global_step % args.mps_empty_cache_every == 0
            ):
                empty_mps_cache(torch, device)

            if args.log_every and global_step % args.log_every == 0:
                log_item = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "audio_loss": float(audio_loss.detach().cpu()),
                    "text_loss": float(text_loss.detach().cpu()),
                    "text_tokens": int(text_mask.sum().detach().cpu()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log_item.update(mps_memory_stats(torch, device))
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(log_item, sort_keys=True) + "\n")
                memory_msg = ""
                if "mps_driver_gb" in log_item:
                    memory_msg = (
                        f" mps={log_item['mps_allocated_gb']:.1f}/"
                        f"{log_item['mps_driver_gb']:.1f}GB"
                    )
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"audio={float(audio_loss.detach().cpu()):.4f} "
                    f"text={float(text_loss.detach().cpu()):.4f}"
                    f"{memory_msg}"
                )
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(torch, save_file, lm, optimizer, args, global_step, out_dir)

        if args.max_steps and global_step >= args.max_steps:
            break

    if global_step == 0:
        raise RuntimeError(
            "Training ended before any optimizer step. Check cache size and grad accumulation."
        )
    save_checkpoint(torch, save_file, lm, optimizer, args, global_step, out_dir)
    print(
        f"Saved final LoRA adapter/checkpoint at step {global_step} in "
        f"{repo_display_path(out_dir)}"
    )


if __name__ == "__main__":
    main()
