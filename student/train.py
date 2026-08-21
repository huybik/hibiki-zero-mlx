#!/usr/bin/env python
"""CUDA-only full-model training for the 12-layer AR mobile student."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from moshi.models import loaders
from moshi.run_inference import get_condition_tensors
from safetensors.torch import save_file

from student.cache import DISTILL_ROLE, artifact_identity, load_cache
from student.contract import (
    DEFAULT_CONFIG,
    build_meta_ar_model,
    read_config,
    sha256,
    torch_lm_config,
    validate_config,
)
from student.harness import (
    atomic_torch_save,
    cache_identity,
    checkpoint_pairs,
    checkpoint_shapes,
    masked_cross_entropy,
    prepare_run_dir,
    prune_checkpoints,
    require_exact_shapes,
    restore_optimizer_rng,
    topk_residual_kl,
    validate_common_hyperparams,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIMI = ROOT / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
DEFAULT_TOKENIZER = ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"
RUN_FORMAT = "hibiki_student_ar_run_v1"
CHECKPOINT_FORMAT = "hibiki_student_ar_checkpoint_v1"
OPTIMIZER_FORMAT = "hibiki_student_ar_optimizer_v1"


def rollout_conditioning(
    codes: torch.Tensor,
    audio_logits: torch.Tensor,
    valid: torch.Tensor,
    audio_offset: int,
    fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Rollout replacement fraction must be in [0, 1]")
    audio = codes[:, audio_offset : audio_offset + audio_logits.shape[1]]
    eligible = valid & (audio >= 0)
    selected = eligible & (
        torch.rand(eligible.shape, device=eligible.device, generator=generator) < fraction
    )
    conditioned = codes.clone()
    conditioned_audio = conditioned[:, audio_offset : audio_offset + audio_logits.shape[1]]
    conditioned_audio[selected] = audio_logits.argmax(-1)[selected]
    return conditioned, selected


def collate(samples: list[dict[str, Any]], cfg: dict[str, Any], top_k: int) -> dict[str, Any]:
    dep_q = int(cfg["dep_q"])
    frames = max(
        max(
            int(sample["codes"].shape[1]),
            int(sample.get("teacher_sequence_codes", sample["codes"][:0]).shape[1]),
        )
        for sample in samples
    )
    batch = len(samples)
    rows = 1 + int(cfg["n_q"])
    codes = torch.full((batch, rows, frames), -1, dtype=torch.long)
    teacher_sequences = torch.full((batch, dep_q, frames), -1, dtype=torch.long)
    teacher_ids = torch.full((batch, frames, top_k), -1, dtype=torch.long)
    teacher_logprobs = torch.zeros((batch, frames, top_k), dtype=torch.float16)
    teacher_mask = torch.zeros((batch, frames), dtype=torch.bool)
    for index, sample in enumerate(samples):
        sample_frames = sample["codes"].shape[1]
        codes[index, :, :sample_frames] = sample["codes"].long()
        teacher_ids[index, :sample_frames] = sample["teacher_text_topk_ids"].long()
        teacher_logprobs[index, :sample_frames] = sample["teacher_text_topk_logprobs"]
        teacher_mask[index, :sample_frames] = sample["teacher_text_mask"]
        sequence = sample.get("teacher_sequence_codes")
        if sequence is not None:
            teacher_sequences[index, :, : sequence.shape[1]] = sequence.long()
    return {
        "codes": codes,
        "teacher_sequences": teacher_sequences,
        "teacher_ids": teacher_ids,
        "teacher_logprobs": teacher_logprobs,
        "teacher_mask": teacher_mask,
    }


def require_exact_checkpoint(path: Path, cfg: dict[str, Any]) -> None:
    expected = {
        name: tuple(tensor.shape) for name, tensor in build_meta_ar_model(cfg).state_dict().items()
    }
    require_exact_shapes(
        expected,
        checkpoint_shapes(path),
        "Checkpoint does not exactly match the 12-layer AR config:",
    )


def run_contract(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    cache_hashes: dict[str, Any],
) -> dict[str, Any]:
    training_keys = (
        "split",
        "steps",
        "batch_size",
        "grad_accum_steps",
        "lr",
        "adam_beta1",
        "adam_beta2",
        "weight_decay",
        "grad_clip",
        "hard_audio_weight",
        "hard_text_weight",
        "teacher_sequence_weight",
        "teacher_text_weight",
        "rollout_start",
        "rollout_fraction",
        "save_every",
        "keep_checkpoints",
        "log_every",
        "gradient_checkpointing",
        "seed",
    )
    return {
        "format": RUN_FORMAT,
        "architecture": metadata["model"]["config"]["architecture"],
        "config": artifact_identity(args.config),
        "mimi": artifact_identity(args.mimi),
        "tokenizer": artifact_identity(args.tokenizer),
        "initial_checkpoint": artifact_identity(args.init_checkpoint),
        "cache": cache_hashes,
        "training": {key: getattr(args, key) for key in training_keys},
    }


def save_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    step: int,
    contract: dict[str, Any],
    contract_hash: str,
) -> Path:
    model_path = out_dir / f"model_step{step:06d}.safetensors"
    optimizer_path = out_dir / f"optimizer_step{step:06d}.pt"
    if model_path.exists() or optimizer_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint pair at step {step}")
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    try:
        save_file(
            state,
            str(temporary),
            metadata={
                "format": CHECKPOINT_FORMAT,
                "step": str(step),
                "contract_sha256": contract_hash,
                "config_sha256": contract["config"]["sha256"],
                "cache_sha256": contract["cache"]["shards_sha256"],
                "initial_checkpoint_sha256": contract["initial_checkpoint"]["sha256"],
            },
        )
        temporary.replace(model_path)
        model_hash = sha256(model_path)
        atomic_torch_save(
            {
                "format": OPTIMIZER_FORMAT,
                "step": step,
                "model": model_path.name,
                "model_sha256": model_hash,
                "contract_sha256": contract_hash,
                "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            optimizer_path,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
        raise
    return optimizer_path


def load_resume(
    model: Any,
    optimizer: torch.optim.Optimizer,
    path: Path,
    contract_hash: str,
) -> int:
    pairs = checkpoint_pairs(path.parent, "model")
    if not pairs or path.resolve() != pairs[max(pairs)][1].resolve():
        raise RuntimeError("--resume-optimizer must be the newest complete checkpoint pair")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "format",
        "step",
        "model",
        "model_sha256",
        "contract_sha256",
        "optimizer",
        "python_rng",
        "torch_rng",
        "cuda_rng",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("Malformed optimizer checkpoint")
    step = int(payload["step"])
    model_path, optimizer_path = pairs.get(step, (None, None))
    if (
        payload["format"] != OPTIMIZER_FORMAT
        or optimizer_path != path
        or model_path is None
        or payload["model"] != model_path.name
        or payload["contract_sha256"] != contract_hash
        or sha256(model_path) != payload["model_sha256"]
    ):
        raise RuntimeError("Resume checkpoint pair or run contract changed")
    restore_optimizer_rng(optimizer, payload)
    return step


def condition_tensors(model: Any, cfg: dict[str, Any], batch_size: int) -> Any | None:
    if model.fuser is None:
        return None
    return get_condition_tensors(
        str(cfg.get("model_type", "hibiki")), model, batch_size=batch_size, cfg_coef=1.0
    )


def validate_args(args: argparse.Namespace) -> None:
    validate_common_hyperparams(args)
    weights = (
        args.hard_audio_weight,
        args.hard_text_weight,
        args.teacher_sequence_weight,
        args.teacher_text_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or not any(weights):
        raise ValueError("Loss weights must be finite and non-negative, with at least one positive")
    if not 0.0 <= args.rollout_start <= 1.0 or not 0.0 <= args.rollout_fraction <= 1.0:
        raise ValueError("Rollout start and replacement fraction must be in [0, 1]")
    if args.init_sha256 != sha256(args.init_checkpoint):
        raise RuntimeError("Initial checkpoint SHA-256 does not match --init-sha256")


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AR student training requires CUDA; MLX is only for quantization/inference"
        )
    if (
        args.resume_optimizer is not None
        and args.resume_optimizer.parent.resolve() != args.out_dir.resolve()
    ):
        raise ValueError("Resume optimizer must be inside --out-dir")

    cfg = read_config(args.config)
    validate_config(cfg)
    if cfg["head"] != "ar" or int(cfg["num_layers"]) != 12:
        raise RuntimeError("Training requires the frozen 12-layer AR student config")
    metadata, shards = load_cache(args.cache_dir, DISTILL_ROLE)
    if metadata["model"]["config"] != cfg:
        raise RuntimeError("Cache model config differs from --config")
    for label, path in (
        ("config_artifact", args.config),
        ("mimi", args.mimi),
        ("tokenizer", args.tokenizer),
    ):
        cached = metadata["model"][label] if label == "config_artifact" else metadata[label]
        if cached != artifact_identity(path):
            raise RuntimeError(f"Cache {label} identity differs from {path}")
    samples = [
        sample
        for _, payload in shards
        for sample in payload["samples"]
        if sample["split"] == args.split
    ]
    if not samples:
        raise RuntimeError(f"No cache samples for split={args.split!r}")
    max_frames = max(
        max(
            sample["codes"].shape[1],
            sample.get("teacher_sequence_codes", sample["codes"][:0]).shape[1],
        )
        for sample in samples
    )
    if max_frames > int(cfg["context"]):
        raise RuntimeError(f"Cache reaches {max_frames} frames, beyond context={cfg['context']}")
    sequence_samples = sum("teacher_sequence_codes" in sample for sample in samples)
    if (
        args.teacher_sequence_weight > 0
        and sequence_samples != len(samples)
    ):
        raise RuntimeError(
            "--teacher-sequence-weight requires teacher_sequence_codes for every selected sample"
        )

    cache_hashes = cache_identity(args.cache_dir, metadata)
    contract = run_contract(args, metadata, cache_hashes)
    initial_path = args.init_checkpoint
    if args.resume_optimizer is not None:
        pairs = checkpoint_pairs(args.out_dir, "model")
        if not pairs:
            raise RuntimeError("No complete checkpoint pair to resume")
        initial_path = pairs[max(pairs)][0]
    require_exact_checkpoint(initial_path, cfg)
    contract_hash = prepare_run_dir(args.out_dir, contract, args.resume_optimizer is not None)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = loaders.get_moshi_lm(
        initial_path,
        lm_kwargs=torch_lm_config(cfg),
        device="cuda",
        dtype=torch.float32,
        lm_kwargs_overrides={"gradient_checkpointing": args.gradient_checkpointing},
    )
    model.requires_grad_(True)
    model.train()
    parameters = list(model.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("Every AR student parameter must be trainable")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise RuntimeError("AR student optimizer masters must remain fp32")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = 0
    if args.resume_optimizer is not None:
        start_step = load_resume(model, optimizer, args.resume_optimizer, contract_hash)
    if start_step >= args.steps:
        raise RuntimeError(f"Resume step {start_step} already reached --steps={args.steps}")
    conditions = condition_tensors(model, cfg, args.batch_size)

    top_k = int(metadata["generation"]["top_k"])
    samples.sort(key=lambda sample: (sample["codes"].shape[1], sample["id"]))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    last_saved = start_step
    optimizer.zero_grad(set_to_none=True)
    for step_index in range(start_step, args.steps):
        totals = {"hard_audio": 0.0, "hard_text": 0.0, "teacher_sequence": 0.0, "teacher_text": 0.0}
        rollout_tokens = 0
        for accumulation_index in range(args.grad_accum_steps):
            microstep = step_index * args.grad_accum_steps + accumulation_index
            begin = microstep * args.batch_size
            selected = [
                samples[order[(begin + offset) % len(order)]] for offset in range(args.batch_size)
            ]
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in collate(selected, cfg, top_k).items()
            }
            codes = batch["codes"]
            progress = step_index / max(1, args.steps - 1)
            use_rollout = args.rollout_fraction > 0 and progress >= args.rollout_start
            model_inputs = codes
            if use_rollout:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    proposal = model(
                        codes,
                        conditions,
                    )
                generator = torch.Generator(device="cuda")
                generator.manual_seed(args.seed + microstep)
                model_inputs, replaced = rollout_conditioning(
                    codes,
                    proposal.logits,
                    proposal.mask,
                    model.audio_offset,
                    args.rollout_fraction,
                    generator,
                )
                rollout_tokens += int(replaced.sum())

            hard_weight = args.hard_audio_weight + args.hard_text_weight + args.teacher_text_weight
            if hard_weight > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        model_inputs,
                        conditions,
                    )
                    hard_audio, _ = masked_cross_entropy(
                        output.logits,
                        codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                        output.mask,
                    )
                    hard_text, _ = masked_cross_entropy(
                        output.text_logits,
                        codes[:, :1],
                        output.text_mask,
                    )
                    teacher_text, _ = topk_residual_kl(
                        output.text_logits[:, 0],
                        batch["teacher_ids"],
                        batch["teacher_logprobs"],
                        batch["teacher_mask"] & output.text_mask[:, 0],
                        "text",
                    )
                    loss = (
                        args.hard_audio_weight * hard_audio
                        + args.hard_text_weight * hard_text
                        + args.teacher_text_weight * teacher_text
                    ) / args.grad_accum_steps
                loss.backward()
                totals["hard_audio"] += float(hard_audio.detach())
                totals["hard_text"] += float(hard_text.detach())
                totals["teacher_text"] += float(teacher_text.detach())

            if args.teacher_sequence_weight > 0 and (batch["teacher_sequences"] >= 0).any():
                sequence_inputs = codes.clone()
                sequence_inputs[:, model.audio_offset : model.audio_offset + model.dep_q] = batch[
                    "teacher_sequences"
                ]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sequence_output = model(
                        sequence_inputs,
                        conditions,
                    )
                    sequence_loss, _ = masked_cross_entropy(
                        sequence_output.logits,
                        batch["teacher_sequences"],
                        sequence_output.mask,
                    )
                    weighted_sequence = (
                        args.teacher_sequence_weight * sequence_loss / args.grad_accum_steps
                    )
                weighted_sequence.backward()
                totals["teacher_sequence"] += float(sequence_loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = step_index + 1
        if step % args.log_every == 0:
            scale = 1.0 / args.grad_accum_steps
            metrics = {key: round(value * scale, 6) for key, value in totals.items()}
            print(
                json.dumps(
                    {
                        "step": step,
                        "losses": metrics,
                        "grad_norm": round(float(grad_norm), 6),
                        "rollout_tokens": rollout_tokens,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            saved = save_checkpoint(model, optimizer, args.out_dir, step, contract, contract_hash)
            prune_checkpoints(checkpoint_pairs(args.out_dir, "model"), args.keep_checkpoints)
            last_saved = step
            print(f"Saved exact checkpoint pair through {saved}", flush=True)
    assert last_saved == args.steps


def self_check() -> None:
    logits = torch.tensor([[[2.0, -1.0, 0.5], [0.0, 1.0, 2.0]]])
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    actual_ce, count = masked_cross_entropy(logits, targets, mask)
    expected_ce = F.cross_entropy(logits[:, :1].reshape(1, 3), torch.tensor([0]))
    torch.testing.assert_close(actual_ce, expected_ce)
    assert int(count) == 1

    teacher_distribution = torch.tensor([0.6, 0.3, 0.1])
    ids = torch.tensor([[[0, 1]]])
    teacher_logprobs = teacher_distribution[:2].log().reshape(1, 1, 2)
    student_logits = torch.tensor([[[1.2, 0.1, -0.7]]])
    actual_kl, count = topk_residual_kl(
        student_logits, ids, teacher_logprobs, torch.ones(1, 1, dtype=torch.bool), "text"
    )
    student_distribution = student_logits.softmax(-1)[0, 0]
    coarse_teacher = torch.tensor([0.6, 0.3, 0.1])
    coarse_student = torch.stack(
        [student_distribution[0], student_distribution[1], student_distribution[2]]
    )
    expected_kl = (coarse_teacher * (coarse_teacher.log() - coarse_student.log())).sum()
    torch.testing.assert_close(actual_kl, expected_kl)
    assert int(count) == 1 and float(actual_kl) >= 0

    codes = torch.tensor([[[9, 9, -1], [1, 2, -1], [3, 4, 5], [7, 7, 7]]])
    audio_logits = torch.zeros(1, 2, 3, 8)
    audio_logits[..., 6] = 1
    valid = torch.tensor([[[True, True, False], [True, False, False]]])
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(7)
    rolled, selected = rollout_conditioning(codes, audio_logits, valid, 1, 0.5, first)
    repeated, repeated_selected = rollout_conditioning(codes, audio_logits, valid, 1, 0.5, second)
    assert torch.equal(rolled, repeated) and torch.equal(selected, repeated_selected)
    assert not selected[~valid].any()
    untouched, none_selected = rollout_conditioning(
        codes, audio_logits, valid, 1, 0.0, torch.Generator().manual_seed(1)
    )
    assert torch.equal(untouched, codes) and not none_selected.any()
    replaced, all_selected = rollout_conditioning(
        codes, audio_logits, valid, 1, 1.0, torch.Generator().manual_seed(1)
    )
    assert torch.equal(all_selected, valid & (codes[:, 1:3] >= 0))
    assert (replaced[:, 1:3][all_selected] == 6).all()
    assert torch.equal(replaced[:, 3], codes[:, 3])
    print("PASS: masked CE, absolute top-k KL, and deterministic rollout invariants")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("train")
    command.add_argument("--cache-dir", type=Path, required=True)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--mimi", type=Path, default=DEFAULT_MIMI)
    command.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    command.add_argument("--init-checkpoint", type=Path, required=True)
    command.add_argument("--init-sha256", required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--resume-optimizer", type=Path)
    command.add_argument("--split", default="train")
    command.add_argument("--steps", type=int, required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--grad-accum-steps", type=int, default=4)
    command.add_argument("--lr", type=float, default=1e-4)
    command.add_argument("--adam-beta1", type=float, default=0.9)
    command.add_argument("--adam-beta2", type=float, default=0.95)
    command.add_argument("--weight-decay", type=float, default=0.1)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--hard-audio-weight", type=float, default=1.0)
    command.add_argument("--hard-text-weight", type=float, default=1.0)
    command.add_argument("--teacher-sequence-weight", type=float, default=0.0)
    command.add_argument("--teacher-text-weight", type=float, default=1.0)
    command.add_argument("--rollout-start", type=float, default=0.8)
    command.add_argument("--rollout-fraction", type=float, default=0.25)
    command.add_argument("--save-every", type=int, default=100)
    command.add_argument("--keep-checkpoints", type=int, default=2)
    command.add_argument("--log-every", type=int, default=1)
    command.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument("--seed", type=int, default=1234)
    subparsers.add_parser("self-check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
    else:
        train(args)


if __name__ == "__main__":
    main()
