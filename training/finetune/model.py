"""The model as every trainer and tool loads it, on CUDA in the environment that
`training/h100.sh setup` builds: devices and seeds, exact full-model checkpoint I/O, the run's text
tokenizer, moshi's model for a checkpoint (a phone student's audio part installed), the policy a
trainer updates, its optimizer and the host offload of its states, and the distillation teacher.

Importing it installs `lm_training`'s faster moshi, as every trainer needs.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moshi.models import loaders
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from training.finetune import bf16_adamw, fp8_linear, lm_training
from training.finetune.run_files import fsync_path
from training.student import heads, slices
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TOKENIZER,
    repo_display_path,
    require_file,
)


def check_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# moshi's training forward and generation step, faster (lm_training.py).
lm_training.install()


def require_torch_compile() -> None:
    """The trainers' forward and loss are compiled functions (`lm_training`, `losses.token_ce`)."""
    if os.environ.get("NO_TORCH_COMPILE"):
        raise ValueError("Full-model training requires torch compile; unset NO_TORCH_COMPILE")


def save_state(state: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    """Write a model's state dict (host tensors) atomically as safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        save_file(state, str(temp_path), metadata=metadata)
        fsync_path(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_model(model: Any, path: Path, dtype: torch.dtype | None) -> None:
    """Load an exact full-model checkpoint; partial states are rejected.
    When `dtype` is None, each floating tensor is cast to its destination parameter's own dtype
    (preserving mixed bf16/fp32 policies such as `--bf16-adamw`)."""
    state = load_file(str(path), device="cpu")
    model_state = model.state_dict()
    expected = set(model_state)
    actual = set(state)
    if missing := sorted(expected - actual):
        raise RuntimeError(f"Checkpoint is missing model tensors: {missing[:5]}")
    if unexpected := sorted(actual - expected):
        raise RuntimeError(f"Checkpoint has unexpected tensors: {unexpected[:5]}")
    # Converted in place, each fp32 tensor freed as it goes: a 12.5 GB fp32 checkpoint and its
    # bf16 copy whole at once held 18 GiB on the Mac.
    for name, tensor in state.items():
        if tensor.dtype.is_floating_point:
            target_dtype = model_state[name].dtype if dtype is None else dtype
            state[name] = tensor.to(dtype=target_dtype)
    model.load_state_dict(state, strict=True)
    print(f"Loaded {len(state)} model tensors from {repo_display_path(path)}")


TOKENIZER_KEY = "text_tokenizer"  # a model weight's metadata: the tokenizer its text rows are on


def weight_tokenizer(path: Path) -> str:
    """The tokenizer a model weight's text rows are on: its `text_tokenizer` metadata, which
    every training checkpoint records since 2026-11-10; upstream's weight and older checkpoints
    are on the base."""
    with safe_open(str(path), framework="pt") as handle:
        return (handle.metadata() or {}).get(TOKENIZER_KEY, DEFAULT_TOKENIZER.name)


def run_tokenizer(tokenizer: Path | None, weight: Path) -> Path:
    """The run's text tokenizer: `tokenizer` (--tokenizer), by default the one `weight` is on,
    from `weights/`. Every entry point calls it at startup, before anything slow: a weight on
    another tokenizer is refused, since the Vietnamese swap gives 6,000 text rows new meanings
    (`vi_tokenizer.init_weights` moves a base weight onto it)."""
    on = weight_tokenizer(weight)
    if tokenizer is None and not (DEFAULT_TOKENIZER.parent / on).is_file():
        raise FileNotFoundError(f"{repo_display_path(weight)} is on {on}, missing from weights/: copy it there or pass --tokenizer")
    path = require_file(tokenizer or DEFAULT_TOKENIZER.parent / on, "tokenizer")
    if on != path.name:
        raise RuntimeError(f"{repo_display_path(weight)} is on {on}, not --tokenizer {path.name}")
    return path


def build_lm(checkpoint_info: Any, device: Any, dtype: torch.dtype, load_weight: bool = True, **kwargs: Any) -> Any:
    """moshi's model for the checkpoint info, with its weight (`load_weight`) or without. A phone
    student's audio part (the config's `audio_heads`, which `load_checkpoint_info` took out: parallel
    heads, or depformer slices) is installed in a model built without it, then loaded, with the text
    EOS row copied from PAD's as moshi's own load of a Hibiki weight does."""
    if checkpoint_info.audio_heads is None:
        return checkpoint_info.get_moshi(device=device, dtype=dtype, load_weight=load_weight, **kwargs)
    lm = checkpoint_info.get_moshi(device=device, dtype=dtype, load_weight=False, **kwargs)
    if checkpoint_info.audio_heads == heads.KIND:
        lm = heads.install(lm)
    else:
        lm = slices.install(lm, checkpoint_info.parallel_codebooks)
    if load_weight:
        load_model(lm, Path(checkpoint_info.moshi_weights), dtype)
        if checkpoint_info.model_type == "hibiki":
            lm.text_emb.weight.data[2] = lm.text_emb.weight.data[3]
    return lm


def bf16_copy(checkpoint_info: Any, lm: Any, device: Any) -> Any:
    """A policy's weights in a bf16 generation model, as `eval.py` loads a checkpoint: 1.8x faster
    than generating on the fp32 masters. Built without reading the base weights, which the copy
    overwrites. moshi's streaming state holds it in reference cycles: after `del`, only
    `gc.collect()` frees its 5.8 GiB."""
    generator = build_lm(checkpoint_info, device, torch.bfloat16, load_weight=False)
    generator.load_state_dict(lm.state_dict())
    generator.eval()
    return generator


def load_policy(checkpoint_info: Any, device: Any, bf16_adamw_mode: bool = False) -> Any:
    """The model a trainer updates (SFT's and GRPO's), from the checkpoint info's weight
    (--model-weight): fp32 masters by default, or bf16 2D+ weights + fp32 1D norms/scales when
    `bf16_adamw_mode` is True (all-GPU 3B SFT on 32 GB cards), every parameter trainable, in
    training mode.

    Loading a Hibiki weight, `build_lm` (moshi's `get_moshi`, or its own load for a heads student)
    copies the text PAD's input embedding over the text EOS's (so a sampled early EOS reads on as PAD), so every run starts from that; the row
    trains from there (cosine 0.88 to PAD's after round 4), and a resumed pair restores its own.
    `load_lm` and `bf16_copy` keep a checkpoint's own row."""
    dtype = torch.bfloat16 if bf16_adamw_mode else torch.float32
    lm = build_lm(checkpoint_info, device, dtype, lm_kwargs_overrides={"gradient_checkpointing": False})
    if bf16_adamw_mode:
        bf16_adamw.cast_policy_bf16(lm)
    lm.train()
    for param in lm.parameters():
        param.requires_grad_(True)
    return lm


def fused_adamw(params: list[Any], args: argparse.Namespace) -> torch.optim.AdamW:
    """The trainers' optimizer, from --lr, --adam-beta1/2 and --weight-decay."""
    return torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )


class HostOffload:
    """GPU storages parked in pinned host memory while a phase that never reads them runs: GRPO's
    window generates on a bf16 copy, so the fp32 policy and AdamW's states (37.5 GB on the 3.1B
    model) wait on the host and the generation batches get the GPU's memory (grpo.md#running);
    SFT's free-running validation, with --park-states, parks AdamW's states and the teacher.
    `to_host` copies each storage out and shrinks it to nothing, so nothing may read those tensors
    until `to_device` restores them exactly; the tensors, the optimizer's and the compiled
    forward's alike, stay the same objects. The pinned buffers are allocated once per storage and
    reused (AdamW's states appear at the first update)."""

    def __init__(self) -> None:
        self.host: dict[int, torch.Tensor] = {}
        self.out: list[tuple[torch.UntypedStorage, torch.Tensor]] = []

    def to_host(self, tensors: list[torch.Tensor]) -> None:
        if self.out:
            raise RuntimeError("HostOffload.to_host while storages are already out")
        storages = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.nbytes():
                storages.setdefault(storage._cdata, storage)  # its StorageImpl: shared views once
        for key, storage in storages.items():
            if key not in self.host:
                self.host[key] = torch.empty(storage.nbytes(), dtype=torch.uint8, pin_memory=True)
            self.host[key].untyped_storage().copy_(storage, non_blocking=True)
            self.out.append((storage, self.host[key]))
        torch.cuda.synchronize()  # every copy has landed before its storage goes
        for storage, _ in self.out:
            storage.resize_(0)
        torch.cuda.empty_cache()  # whole segments for the generation batches

    def to_device(self) -> None:
        for storage, host in self.out:
            storage.resize_(host.numel())
            storage.copy_(host.untyped_storage(), non_blocking=True)
        torch.cuda.synchronize()
        self.out = []


def adam_states(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """AdamW's CUDA state tensors (none before its first step)."""
    return [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_cuda
    ]


def load_lm(checkpoint_info: Any, checkpoint: Path | None, device: Any, dtype: torch.dtype) -> Any:
    """The base model, or `checkpoint` in a model built without reading the base weights, which
    a full checkpoint overwrites anyway (12.5 GB of fp32 for a fine-tuned base)."""
    if checkpoint is None:
        print(f"Loading the base LM on {device} from {repo_display_path(checkpoint_info.moshi_weights)}")
        return build_lm(checkpoint_info, device, dtype)
    lm = build_lm(checkpoint_info, device, dtype, load_weight=False)
    load_model(lm, checkpoint, dtype)
    return lm


def teacher_checkpoint_info(args: argparse.Namespace) -> Any:
    """The checkpoint info of a run's distillation teacher: --teacher-weight, with
    --teacher-config (by default the 3B's), on the run's tokenizer, which it must be on."""
    args.teacher_weight = require_file(args.teacher_weight, "teacher weight")
    args.teacher_config = require_file(args.teacher_config or DEFAULT_CONFIG_PATH, "teacher config")
    run_tokenizer(args.tokenizer, args.teacher_weight)  # refuses another tokenizer
    return load_checkpoint_info(
        argparse.Namespace(
            hf_repo=args.hf_repo,
            model_weight=args.teacher_weight,
            mimi_weight=args.mimi_weight,
            tokenizer=args.tokenizer,
            config_path=args.teacher_config,
        )
    )


def load_teacher(info: Any, weight: Path, device: Any, dep_q: int, fp8: bool = False) -> Any:
    """A frozen bf16 teacher in the compiled training forward (`lm_training`), run without a
    backward. Its depformer runs only the first `dep_q` codebooks: steps past them change no
    earlier codebook's logits (the depformer is causal over steps). With `fp8`, pre-quantizes the
    frozen teacher's main-transformer and depformer linear weights once to rowwise E4M3."""
    teacher = load_lm(info, weight, device, torch.bfloat16)
    teacher.train()
    teacher.requires_grad_(False)
    teacher.dep_q = dep_q
    if fp8:
        n_q = fp8_linear.prepare_fp8_teacher(teacher)
        print(f"Pre-quantized {n_q} frozen teacher linears to rowwise FP8")
    return teacher


def load_checkpoint_info(args: argparse.Namespace) -> Any:
    """moshi's checkpoint info, its `audio_heads` (None, or a phone student's parallel heads,
    `heads.KIND`, or depformer slices, `slices.KIND`, with their `parallel_codebooks`) taken out
    of the LM config, which moshi hands whole to the model."""
    info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.model_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )
    info.audio_heads = info.lm_config.pop(heads.CONFIG_KEY, None)
    info.parallel_codebooks = info.lm_config.pop(slices.PARALLEL_KEY, None)
    if info.audio_heads not in (None, heads.KIND, slices.KIND):
        raise ValueError(f"unknown {heads.CONFIG_KEY} {info.audio_heads!r} in {args.config_path}")
    if (info.audio_heads == slices.KIND) != (info.parallel_codebooks is not None):
        raise ValueError(f"{slices.PARALLEL_KEY} goes with {heads.CONFIG_KEY} {slices.KIND!r} only, in {args.config_path}")
    return info
