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
    parser.add_argument(
        "--val-cache-dir",
        type=Path,
        help="Optional cached validation split for teacher-forced CE logging.",
    )
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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use only the first N cached samples after optional length-sort; 0 means all.",
    )
    parser.add_argument(
        "--replay-ids",
        default="",
        help="Comma-separated sample ids to upweight in the training sampler.",
    )
    parser.add_argument(
        "--replay-weight",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for --replay-ids; 1 disables replay weighting.",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=0,
        help="Deterministic seed for the replay-weighted sampler.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--text-head-lr",
        type=float,
        default=0.0,
        help="Optional LR for text_linear params; 0 uses --lr.",
    )
    parser.add_argument(
        "--audio-head-lr",
        type=float,
        default=0.0,
        help="Optional LR for audio-head LoRA params; 0 uses --lr.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-scaling", type=float, default=2.0)
    parser.add_argument(
        "--train-text-head",
        action="store_true",
        help="Also train/save LMModel.text_linear for tiny text-overfit experiments.",
    )
    parser.add_argument(
        "--train-audio-heads",
        action="store_true",
        help="Also add/train LoRA on depformer_in and audio output linears.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Optimizer steps, 0 means all.")
    parser.add_argument(
        "--val-every",
        type=int,
        default=0,
        help="Optimizer steps between teacher-forced validation runs; 0 means final only.",
    )
    parser.add_argument(
        "--val-max-samples",
        type=int,
        default=0,
        help="Use only the first N validation cache samples after sort; 0 means all.",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=0,
        help="Validate only the first N batches; 0 means all selected validation batches.",
    )
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
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="Load adapter weights before training without optimizer/global-step resume.",
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
        def __init__(self, cache_dir: Path, sort_by_length: bool, max_samples: int):
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
            if max_samples:
                self.samples = self.samples[:max_samples]
                if not self.samples:
                    raise RuntimeError("--max-samples selected no cached samples")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.samples[index]

    return CachedCodeDataset


def parse_replay_ids(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def make_replay_sampler(
    torch: Any,
    dataset: Any,
    replay_ids: set[str],
    replay_weight: float,
    seed: int,
) -> Any | None:
    if not replay_ids or replay_weight == 1.0:
        return None
    sample_ids = {str(sample["id"]) for sample in dataset.samples}
    missing = sorted(replay_ids - sample_ids)
    if missing:
        raise ValueError(f"--replay-ids not present in selected cache samples: {missing[:10]}")

    weights = [
        float(replay_weight) if str(sample["id"]) in replay_ids else 1.0
        for sample in dataset.samples
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


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


def make_cached_dataloader(
    DataLoader: Any,
    dataset: Any,
    torch: Any,
    batch_size: int,
    num_workers: int,
    sort_by_length: bool,
    sampler: Any | None = None,
) -> Any:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else not sort_by_length,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lambda samples: collate_cached(samples, torch),
    )


def adapter_target(args: argparse.Namespace) -> str:
    targets = ["LMModel.transformer"]
    if args.train_text_head:
        targets.append("text_linear")
    if args.train_audio_heads:
        targets.append("audio_heads")
    return "+".join(targets)


def zero_lora_updates(model: Any) -> None:
    for module in model.modules():
        lora_b = getattr(module, "lora_B", None)
        if lora_b is not None:
            lora_b.weight.data.zero_()


def apply_lora_targets(
    model: Any, replace_all_linear_with_lora: Any, args: argparse.Namespace
) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    replace_all_linear_with_lora(model.transformer, args.lora_rank, args.lora_scaling)
    if args.train_audio_heads:
        replace_all_linear_with_lora(model.depformer_in, args.lora_rank, args.lora_scaling)
        replace_all_linear_with_lora(model.linears, args.lora_rank, args.lora_scaling)
    zero_lora_updates(model)
    for name, param in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        train_transformer = is_lora and name.startswith("transformer.")
        train_text_head = args.train_text_head and name.startswith("text_linear.")
        train_audio_head = args.train_audio_heads and is_lora and (
            name.startswith("depformer_in.") or name.startswith("linears.")
        )
        param.requires_grad_(train_transformer or train_text_head or train_audio_head)

    bad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and not name.startswith("transformer.")
        and not name.startswith("text_linear.")
        and not name.startswith("depformer_in.")
        and not name.startswith("linears.")
    ]
    if bad:
        raise RuntimeError(
            f"LoRA freeze map leaked trainable params outside selected targets: {bad[:5]}"
        )


def trainable_parameters(model: Any) -> list[Any]:
    return [param for param in model.parameters() if param.requires_grad]


def optimizer_parameters(model: Any, args: argparse.Namespace) -> list[Any] | list[dict[str, Any]]:
    text_lr = args.text_head_lr or args.lr
    audio_lr = args.audio_head_lr or args.lr
    if text_lr == args.lr and audio_lr == args.lr:
        return trainable_parameters(model)

    groups: dict[str, dict[str, Any]] = {
        "transformer": {"params": [], "lr": args.lr, "name": "transformer"},
        "text_linear": {"params": [], "lr": text_lr, "name": "text_linear"},
        "audio_heads": {"params": [], "lr": audio_lr, "name": "audio_heads"},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("transformer."):
            groups["transformer"]["params"].append(param)
        elif name.startswith("text_linear."):
            groups["text_linear"]["params"].append(param)
        elif name.startswith("depformer_in.") or name.startswith("linears."):
            groups["audio_heads"]["params"].append(param)
        else:
            raise RuntimeError(f"Unexpected trainable parameter outside optimizer groups: {name}")

    return [group for group in groups.values() if group["params"]]


def optimizer_lrs(optimizer: Any) -> float | dict[str, float]:
    if len(optimizer.param_groups) == 1:
        return float(optimizer.param_groups[0]["lr"])
    return {
        str(group.get("name", f"group{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def adapter_state_dict(model: Any) -> dict[str, Any]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state: dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if name in trainable_names:
            state[name] = tensor.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No trainable adapter tensors found to save.")
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
        "target": adapter_target(args),
        "train_text_head": str(args.train_text_head),
        "train_audio_heads": str(args.train_audio_heads),
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


def load_adapter_state(
    load_file: Any,
    model: Any,
    adapter_path: Path,
    dtype: Any,
) -> None:
    adapter_state = load_file(str(adapter_path), device="cpu")
    for key, value in adapter_state.items():
        if value.dtype.is_floating_point:
            adapter_state[key] = value.to(dtype=dtype)
    result = model.load_state_dict(adapter_state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys while loading {adapter_path}: {result.unexpected_keys[:5]}")
    print(f"Loaded {len(adapter_state)} adapter tensors from {repo_display_path(adapter_path)}")


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
    load_adapter_state(load_file, model, adapter_path, dtype)

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


def masked_cross_entropy(torch: Any, logits: Any, targets: Any, mask: Any) -> tuple[Any, int]:
    token_count = int(mask.sum().detach().cpu())
    if token_count == 0:
        return logits.float().sum() * 0.0, 0
    loss = torch.nn.functional.cross_entropy(logits[mask].float(), targets[mask].long())
    return loss, token_count


def text_supervision_mask(torch: Any, base_mask: Any, targets: Any, pad_id: int) -> Any:
    non_pad = targets != pad_id
    seen_text = non_pad.long().cumsum(dim=-1) > 0
    prefix_pad = (targets == pad_id) & ~seen_text
    return base_mask & (non_pad | prefix_pad)


def compute_batch_losses(
    torch: Any,
    lm: Any,
    codes: Any,
    condition_tensors: Any | None,
    audio_loss_weight: float,
    text_loss_weight: float,
) -> dict[str, Any]:
    output = lm(codes, condition_tensors=condition_tensors)
    audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    text_targets = codes[:, :1]
    audio_loss, audio_tokens = masked_cross_entropy(
        torch, output.logits, audio_targets, output.mask
    )
    text_mask = text_supervision_mask(
        torch, output.text_mask, text_targets, lm.text_padding_token_id
    )
    text_loss, text_tokens = masked_cross_entropy(
        torch, output.text_logits, text_targets, text_mask
    )
    loss = audio_loss_weight * audio_loss + text_loss_weight * text_loss
    return {
        "loss": loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": audio_tokens,
        "text_tokens": text_tokens,
    }


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


def evaluate_teacher_forced(
    torch: Any,
    lm: Any,
    dataloader: Any,
    device: Any,
    model_type: str,
    get_condition_tensors: Any,
    audio_loss_weight: float,
    text_loss_weight: float,
    max_batches: int = 0,
) -> dict[str, float | int]:
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    totals = {
        "audio_loss_sum": 0.0,
        "text_loss_sum": 0.0,
        "audio_tokens": 0,
        "text_tokens": 0,
        "batches": 0,
        "samples": 0,
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches and batch_index >= max_batches:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(
                    lm, model_type, batch_size, get_condition_tensors
                )
            losses = compute_batch_losses(
                torch,
                lm,
                codes,
                condition_cache[batch_size],
                audio_loss_weight,
                text_loss_weight,
            )
            audio_tokens = int(losses["audio_tokens"])
            text_tokens = int(losses["text_tokens"])
            totals["audio_loss_sum"] += float(losses["audio_loss"].detach().cpu()) * audio_tokens
            totals["text_loss_sum"] += float(losses["text_loss"].detach().cpu()) * text_tokens
            totals["audio_tokens"] += audio_tokens
            totals["text_tokens"] += text_tokens
            totals["batches"] += 1
            totals["samples"] += batch_size

    if was_training:
        lm.train()
    if not totals["batches"]:
        raise RuntimeError("Validation dataloader produced no batches.")

    audio_loss = (
        totals["audio_loss_sum"] / totals["audio_tokens"]
        if totals["audio_tokens"]
        else 0.0
    )
    text_loss = (
        totals["text_loss_sum"] / totals["text_tokens"]
        if totals["text_tokens"]
        else 0.0
    )
    return {
        "loss": audio_loss_weight * audio_loss + text_loss_weight * text_loss,
        "audio_loss": audio_loss,
        "text_loss": text_loss,
        "audio_tokens": totals["audio_tokens"],
        "text_tokens": totals["text_tokens"],
        "batches": totals["batches"],
        "samples": totals["samples"],
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.val_max_samples < 0:
        raise ValueError("--val-max-samples must be non-negative")
    if args.val_batches < 0:
        raise ValueError("--val-batches must be non-negative")
    if args.val_every < 0:
        raise ValueError("--val-every must be non-negative")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.replay_weight <= 0:
        raise ValueError("--replay-weight must be positive")
    if args.text_head_lr < 0 or args.audio_head_lr < 0:
        raise ValueError("Custom LR values must be non-negative")
    if args.resume_checkpoint is not None and args.init_adapter is not None:
        raise ValueError("--resume-checkpoint and --init-adapter are mutually exclusive")

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
    val_cache_dir = (
        require_dir(args.val_cache_dir, "validation code cache directory")
        if args.val_cache_dir is not None
        else None
    )
    args.config_path = require_file(args.config_path, "config")
    args.model_weight = require_file(args.model_weight, "model weight")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.tokenizer = require_file(args.tokenizer, "tokenizer")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = require_file(args.resume_checkpoint, "resume checkpoint")
    if args.init_adapter is not None:
        args.init_adapter = require_file(args.init_adapter, "initial adapter")
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DatasetClass = make_dataset_class(torch, Dataset)
    dataset = DatasetClass(cache_dir, args.sort_by_length, args.max_samples)
    print(f"Loaded {len(dataset)} cached samples from {repo_display_path(cache_dir)}")
    replay_ids = parse_replay_ids(args.replay_ids)
    replay_sampler = make_replay_sampler(
        torch, dataset, replay_ids, args.replay_weight, args.replay_seed
    )
    if replay_sampler is not None:
        print(
            f"Replay-weighted sampler: {len(replay_ids)} ids at "
            f"{args.replay_weight:g}x weight, seed={args.replay_seed}"
        )
    dataloader = make_cached_dataloader(
        DataLoader,
        dataset,
        torch,
        args.batch_size,
        args.num_workers,
        args.sort_by_length,
        replay_sampler,
    )
    val_dataloader = None
    if val_cache_dir is not None:
        val_dataset = DatasetClass(val_cache_dir, args.sort_by_length, args.val_max_samples)
        val_dataloader = make_cached_dataloader(
            DataLoader,
            val_dataset,
            torch,
            args.batch_size,
            args.num_workers,
            args.sort_by_length,
        )
        print(f"Loaded {len(val_dataset)} validation cached samples from {repo_display_path(val_cache_dir)}")

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
    apply_lora_targets(lm, replace_all_linear_with_lora, args)
    if args.init_adapter is not None:
        load_adapter_state(load_file, lm, args.init_adapter, dtype)

    params = trainable_parameters(lm)
    if not params:
        raise RuntimeError("No trainable LoRA parameters after freeze map.")
    trainable_count = sum(param.numel() for param in params)
    total_count = sum(param.numel() for param in lm.parameters())
    print(f"Trainable params: {trainable_count:,} / {total_count:,}")

    optimizer = torch.optim.AdamW(optimizer_parameters(lm, args), lr=args.lr)
    global_step = 0
    micro_step = 0
    resume_skip_batches = 0
    if args.resume_checkpoint is not None:
        global_step = load_resume_checkpoint(
            torch, load_file, lm, optimizer, args.resume_checkpoint, device, dtype
        )
        micro_step = global_step * args.grad_accum_steps
        if args.sort_by_length and replay_sampler is None:
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
    val_log_path = out_dir / "val_log.jsonl"
    condition_cache: dict[int, Any | None] = {}
    log_sums = {"loss": 0.0, "audio_loss": 0.0, "text_loss": 0.0}
    log_steps = 0
    log_text_tokens = 0

    def run_validation(step: int, epoch: int) -> None:
        if val_dataloader is None:
            return
        metrics = evaluate_teacher_forced(
            torch,
            lm,
            val_dataloader,
            device,
            checkpoint_info.model_type,
            get_condition_tensors,
            args.audio_loss_weight,
            args.text_loss_weight,
            args.val_batches,
        )
        log_item = {
            "epoch": epoch,
            "step": step,
            "loss": metrics["loss"],
            "audio_loss": metrics["audio_loss"],
            "text_loss": metrics["text_loss"],
            "audio_tokens": metrics["audio_tokens"],
            "text_tokens": metrics["text_tokens"],
            "batches": metrics["batches"],
            "samples": metrics["samples"],
        }
        log_item.update(mps_memory_stats(torch, device))
        with val_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_item, sort_keys=True) + "\n")
        print(
            f"val step={step} loss={metrics['loss']:.4f} "
            f"audio={metrics['audio_loss']:.4f} text={metrics['text_loss']:.4f} "
            f"samples={metrics['samples']}"
        )

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
            losses = compute_batch_losses(
                torch,
                lm,
                codes,
                condition_tensors,
                args.audio_loss_weight,
                args.text_loss_weight,
            )
            loss = losses["loss"]
            audio_loss = losses["audio_loss"]
            text_loss = losses["text_loss"]
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

            loss_value = float(loss.detach().cpu())
            audio_loss_value = float(audio_loss.detach().cpu())
            text_loss_value = float(text_loss.detach().cpu())
            log_sums["loss"] += loss_value
            log_sums["audio_loss"] += audio_loss_value
            log_sums["text_loss"] += text_loss_value
            log_steps += 1
            log_text_tokens += int(losses["text_tokens"])

            if args.log_every and global_step % args.log_every == 0:
                log_item = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": log_sums["loss"] / log_steps,
                    "audio_loss": log_sums["audio_loss"] / log_steps,
                    "text_loss": log_sums["text_loss"] / log_steps,
                    "text_tokens": log_text_tokens,
                    "log_steps": log_steps,
                    "lr": optimizer_lrs(optimizer),
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
                    f"loss={log_item['loss']:.4f} "
                    f"audio={log_item['audio_loss']:.4f} "
                    f"text={log_item['text_loss']:.4f}"
                    f"{memory_msg}"
                )
                log_sums = {"loss": 0.0, "audio_loss": 0.0, "text_loss": 0.0}
                log_steps = 0
                log_text_tokens = 0
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(torch, save_file, lm, optimizer, args, global_step, out_dir)
            if (
                val_dataloader is not None
                and args.val_every
                and global_step % args.val_every == 0
            ):
                run_validation(global_step, epoch + 1)

        if args.max_steps and global_step >= args.max_steps:
            break

    if global_step == 0:
        raise RuntimeError(
            "Training ended before any optimizer step. Check cache size and grad accumulation."
        )
    if val_dataloader is not None and (not args.val_every or global_step % args.val_every != 0):
        run_validation(global_step, min(args.epochs, epoch + 1))
    save_checkpoint(torch, save_file, lm, optimizer, args, global_step, out_dir)
    print(
        f"Saved final LoRA adapter/checkpoint at step {global_step} in "
        f"{repo_display_path(out_dir)}"
    )


if __name__ == "__main__":
    main()
