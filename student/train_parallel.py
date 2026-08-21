#!/usr/bin/env python
"""CUDA head-only training for the frozen 12-layer student's parallel_v1 head."""

from __future__ import annotations

import argparse
import json
import math
import random
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from student.cache import artifact_identity
from student.capture_parallel import load_parallel_cache
from student.contract import read_config, sha256
from student.harness import (
    atomic_torch_save,
    cache_identity,
    checkpoint_pairs,
    masked_cross_entropy,
    prepare_run_dir,
    prune_checkpoints,
    restore_optimizer_rng,
    topk_residual_kl,
    validate_common_hyperparams,
)
from student.parallel import (
    PARALLEL_PARAMETERS,
    ParallelHead,
    require_compatible_configs,
    validate_ar_checkpoint,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AR_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_ar.json"
DEFAULT_PARALLEL_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_parallel_v1.json"
DEFAULT_TOKENIZER = ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"
RUN_FORMAT = "hibiki_parallel_head_run_v1"
CHECKPOINT_FORMAT = "hibiki_parallel_head_checkpoint_v1"
OPTIMIZER_FORMAT = "hibiki_parallel_head_optimizer_v1"


def collate(samples: list[dict[str, Any]], top_k: int, card: int) -> dict[str, torch.Tensor]:
    frames = max(int(sample["hidden"].shape[0]) for sample in samples)
    batch = len(samples)
    hidden = torch.zeros(batch, frames, 2048, dtype=torch.bfloat16)
    text_ids = torch.zeros(batch, frames, dtype=torch.long)
    previous = torch.full((batch, frames, 8), card, dtype=torch.long)
    targets = torch.full((batch, frames, 8), card, dtype=torch.long)
    teacher_ids = torch.full((batch, frames, 8, top_k), -1, dtype=torch.long)
    teacher_logprobs = torch.zeros(batch, frames, 8, top_k, dtype=torch.float16)
    mask = torch.zeros(batch, frames, 8, dtype=torch.bool)
    for index, sample in enumerate(samples):
        length = int(sample["hidden"].shape[0])
        hidden[index, :length] = sample["hidden"]
        text_ids[index, :length] = sample["text_ids"].long()
        previous[index, :length] = sample["previous_codes"].long()
        targets[index, :length] = sample["hard_targets"].long()
        teacher_ids[index, :length] = sample["teacher_topk_ids"].long()
        teacher_logprobs[index, :length] = sample["teacher_topk_logprobs"]
        mask[index, :length] = sample["teacher_mask"]
    return {
        "hidden": hidden,
        "text_ids": text_ids,
        "previous_codes": previous,
        "targets": targets,
        "teacher_ids": teacher_ids,
        "teacher_logprobs": teacher_logprobs,
        "mask": mask,
    }


def run_contract(
    args: argparse.Namespace,
    cfg: dict[str, Any],
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
        "ce_weight",
        "kl_weight",
        "save_every",
        "keep_checkpoints",
        "log_every",
        "seed",
    )
    return {
        "format": RUN_FORMAT,
        "architecture": cfg["architecture"],
        "head": "parallel_v1",
        "head_passes": cfg["head_passes"],
        "parallel_config": artifact_identity(args.parallel_config),
        "ar_config": artifact_identity(args.ar_config),
        "base_checkpoint": artifact_identity(args.ar_checkpoint),
        "tokenizer": artifact_identity(args.tokenizer),
        "cache": cache_hashes,
        "training": {key: getattr(args, key) for key in training_keys},
    }


def save_checkpoint(
    head: ParallelHead,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    step: int,
    contract: dict[str, Any],
    contract_hash: str,
) -> Path:
    head_path = out_dir / f"head_step{step:06d}.safetensors"
    optimizer_path = out_dir / f"optimizer_step{step:06d}.pt"
    if head_path.exists() or optimizer_path.exists():
        raise FileExistsError(f"Refusing to overwrite parallel checkpoint pair at step {step}")
    state = {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}
    temporary = head_path.with_name(f".{head_path.name}.tmp")
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "step": str(step),
        "contract_sha256": contract_hash,
        "base_checkpoint_sha256": contract["base_checkpoint"]["sha256"],
        "parallel_config_sha256": contract["parallel_config"]["sha256"],
        "cache_sha256": contract["cache"]["shards_sha256"],
        "head_passes": str(contract["head_passes"]),
        "head_parameters": str(PARALLEL_PARAMETERS),
    }
    try:
        save_file(state, str(temporary), metadata=metadata)
        temporary.replace(head_path)
        head_hash = sha256(head_path)
        atomic_torch_save(
            {
                "format": OPTIMIZER_FORMAT,
                "step": step,
                "head": head_path.name,
                "head_sha256": head_hash,
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
        head_path.unlink(missing_ok=True)
        raise
    return optimizer_path


def expected_checkpoint_metadata(
    step: int, contract: dict[str, Any], contract_hash: str
) -> dict[str, str]:
    return {
        "format": CHECKPOINT_FORMAT,
        "step": str(step),
        "contract_sha256": contract_hash,
        "base_checkpoint_sha256": contract["base_checkpoint"]["sha256"],
        "parallel_config_sha256": contract["parallel_config"]["sha256"],
        "cache_sha256": contract["cache"]["shards_sha256"],
        "head_passes": str(contract["head_passes"]),
        "head_parameters": str(PARALLEL_PARAMETERS),
    }


def load_resume(
    head: ParallelHead,
    optimizer: torch.optim.Optimizer,
    path: Path,
    contract: dict[str, Any],
    contract_hash: str,
) -> int:
    pairs = checkpoint_pairs(path.parent, "head", label="parallel ")
    if not pairs or path.resolve() != pairs[max(pairs)][1].resolve():
        raise RuntimeError("--resume-optimizer must be the newest complete checkpoint pair")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "format",
        "step",
        "head",
        "head_sha256",
        "contract_sha256",
        "optimizer",
        "python_rng",
        "torch_rng",
        "cuda_rng",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("Malformed parallel optimizer checkpoint")
    step = int(payload["step"])
    head_path, optimizer_path = pairs.get(step, (None, None))
    if (
        payload["format"] != OPTIMIZER_FORMAT
        or optimizer_path != path
        or head_path is None
        or payload["head"] != head_path.name
        or payload["contract_sha256"] != contract_hash
        or sha256(head_path) != payload["head_sha256"]
    ):
        raise RuntimeError("Parallel resume checkpoint pair or lineage changed")
    with safe_open(head_path, framework="pt", device="cpu") as handle:
        if handle.metadata() != expected_checkpoint_metadata(step, contract, contract_hash):
            raise RuntimeError("Parallel head checkpoint metadata changed")
    head.load_state_dict(load_file(head_path, device="cpu"), strict=True)
    head.cuda()
    restore_optimizer_rng(optimizer, payload)
    return step


def load_frozen_text_embedding(path: Path, cfg: dict[str, Any]) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("text_emb.weight")
    expected = (int(cfg["text_card"]) + 1, int(cfg["dim"]))
    if tuple(weight.shape) != expected:
        raise RuntimeError(f"text_emb.weight shape {tuple(weight.shape)} != {expected}")
    weight = weight.to(device="cuda", dtype=torch.bfloat16)
    weight.requires_grad_(False)
    return weight


def validate_args(args: argparse.Namespace) -> None:
    validate_common_hyperparams(args)
    if any(
        not math.isfinite(weight) or weight < 0 for weight in (args.ce_weight, args.kl_weight)
    ) or not (args.ce_weight > 0 or args.kl_weight > 0):
        raise ValueError("CE/KL weights must be non-negative with at least one positive")


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel-head training requires CUDA; MLX is quantize/inference only")
    if (
        args.resume_optimizer is not None
        and args.resume_optimizer.parent.resolve() != args.out_dir.resolve()
    ):
        raise ValueError("Resume optimizer must be inside --out-dir")

    ar_cfg = validate_ar_checkpoint(
        args.ar_config,
        args.ar_checkpoint,
        args.ar_sha256,
    )
    cfg = read_config(args.parallel_config)
    require_compatible_configs(ar_cfg, cfg)
    metadata, shards = load_parallel_cache(args.cache_dir)
    if metadata["backbone"]["config"] != ar_cfg:
        raise RuntimeError("Capture cache backbone config differs from --ar-config")
    expected_artifacts = {
        "config_artifact": artifact_identity(args.ar_config),
        "checkpoint_artifact": artifact_identity(args.ar_checkpoint),
    }
    for key, value in expected_artifacts.items():
        if metadata["backbone"][key] != value:
            raise RuntimeError(f"Capture cache {key} differs from the exact frozen base")
    if metadata["tokenizer"] != artifact_identity(args.tokenizer):
        raise RuntimeError("Capture cache tokenizer differs from --tokenizer")

    samples = [
        sample
        for _, payload in shards
        for sample in payload["samples"]
        if sample["split"] == args.split
    ]
    if not samples:
        raise RuntimeError(f"No parallel capture samples for split={args.split!r}")
    if max(sample["hidden"].shape[0] for sample in samples) > int(cfg["context"]):
        raise RuntimeError("Parallel capture exceeds the frozen backbone context")

    cache_hashes = cache_identity(args.cache_dir, metadata)
    contract = run_contract(args, cfg, cache_hashes)
    contract_hash = prepare_run_dir(args.out_dir, contract, args.resume_optimizer is not None)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    head = ParallelHead.from_config(cfg).cuda().float()
    parameters = list(head.parameters())
    if sum(parameter.numel() for parameter in parameters) != PARALLEL_PARAMETERS:
        raise RuntimeError("Parallel head parameter count changed")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        fused=True,
    )
    start_step = 0
    if args.resume_optimizer is not None:
        start_step = load_resume(head, optimizer, args.resume_optimizer, contract, contract_hash)
    if start_step >= args.steps:
        raise RuntimeError(f"Resume step {start_step} already reached --steps={args.steps}")
    text_weight = load_frozen_text_embedding(args.ar_checkpoint, ar_cfg)

    top_k = int(metadata["generation"]["top_k"])
    samples.sort(key=lambda sample: (sample["hidden"].shape[0], sample["id"]))
    order = list(range(len(samples)))
    random.Random(args.seed).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    last_saved = start_step
    for step_index in range(start_step, args.steps):
        totals = {"ce": 0.0, "kl": 0.0}
        for accumulation_index in range(args.grad_accum_steps):
            microstep = step_index * args.grad_accum_steps + accumulation_index
            begin = microstep * args.batch_size
            selected = [
                samples[order[(begin + offset) % len(order)]] for offset in range(args.batch_size)
            ]
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in collate(selected, top_k, int(cfg["card"])).items()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                text_embedding = F.embedding(batch["text_ids"], text_weight)
                logits = head(batch["hidden"], text_embedding, batch["previous_codes"])
                ce, _ = masked_cross_entropy(logits, batch["targets"], batch["mask"])
                kl, _ = topk_residual_kl(
                    logits,
                    batch["teacher_ids"],
                    batch["teacher_logprobs"],
                    batch["mask"],
                    "audio",
                )
                loss = (args.ce_weight * ce + args.kl_weight * kl) / args.grad_accum_steps
            loss.backward()
            totals["ce"] += float(ce.detach())
            totals["kl"] += float(kl.detach())
        if text_weight.grad is not None:
            raise RuntimeError("Frozen text_emb.weight unexpectedly received a gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = step_index + 1
        if step % args.log_every == 0:
            scale = 1.0 / args.grad_accum_steps
            print(
                json.dumps(
                    {
                        "step": step,
                        "losses": {key: round(value * scale, 6) for key, value in totals.items()},
                        "grad_norm": round(float(grad_norm), 6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            saved = save_checkpoint(head, optimizer, args.out_dir, step, contract, contract_hash)
            prune_checkpoints(
                checkpoint_pairs(args.out_dir, "head", label="parallel "), args.keep_checkpoints
            )
            last_saved = step
            print(f"Saved exact parallel checkpoint pair through {saved}", flush=True)
    assert last_saved == args.steps


def self_check() -> None:
    head = ParallelHead(passes=1)
    assert head.parameter_count == PARALLEL_PARAMETERS
    logits = head(
        torch.randn(2, 3, 2048),
        torch.randn(2, 3, 2048),
        torch.full((2, 3, 8), 2048, dtype=torch.long),
    )
    assert tuple(logits.shape) == (2, 3, 8, 2048)
    del head, logits

    ce_logits = torch.tensor([[[[2.0, -1.0, 0.5], [0.0, 1.0, 2.0]]]])
    targets = torch.tensor([[[0, 1]]])
    mask = torch.tensor([[[True, False]]])
    actual_ce, count = masked_cross_entropy(ce_logits, targets, mask)
    expected_ce = F.cross_entropy(ce_logits[:, :, :1].reshape(1, 3), torch.tensor([0]))
    torch.testing.assert_close(actual_ce, expected_ce)
    assert int(count) == 1

    teacher_distribution = torch.tensor([0.6, 0.3, 0.1])
    ids = torch.tensor([[[[0, 1]]]])
    teacher_logprobs = teacher_distribution[:2].log().reshape(1, 1, 1, 2)
    student_logits = torch.tensor([[[[1.2, 0.1, -0.7]]]])
    actual_kl, count = topk_residual_kl(
        student_logits, ids, teacher_logprobs, torch.ones(1, 1, 1, dtype=torch.bool), "audio"
    )
    student_distribution = student_logits.softmax(-1)[0, 0, 0]
    expected_kl = (
        teacher_distribution * (teacher_distribution.log() - student_distribution.log())
    ).sum()
    torch.testing.assert_close(actual_kl, expected_kl)
    assert int(count) == 1 and float(actual_kl) >= 0

    two_pass = ParallelHead(passes=2)
    refinements: list[torch.Tensor | None] = []

    def record_predict(
        module: ParallelHead,
        base: torch.Tensor,
        refinement_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        refinements.append(None if refinement_codes is None else refinement_codes.clone())
        controlled = torch.zeros(*base.shape[:-1], module.card)
        tokens = torch.arange(module.codebooks).view(1, 1, -1, 1).expand(*base.shape[:-1], 1)
        return controlled.scatter(-1, tokens, 1.0)

    two_pass._predict = types.MethodType(record_predict, two_pass)
    two_pass(
        torch.zeros(1, 2, 2048),
        torch.zeros(1, 2, 2048),
        torch.full((1, 2, 8), 2048, dtype=torch.long),
    )
    expected = torch.arange(8).view(1, 1, 8).expand(1, 2, 8)
    assert len(refinements) == 2 and refinements[0] is None
    assert refinements[1] is not None and torch.equal(refinements[1], expected)
    print("PASS: parallel_v1 shapes, 7,346,176 parameters, CE/KL, and simultaneous pass-2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--cache-dir", type=Path, required=True)
    command.add_argument("--ar-config", type=Path, default=DEFAULT_AR_CONFIG)
    command.add_argument("--parallel-config", type=Path, default=DEFAULT_PARALLEL_CONFIG)
    command.add_argument("--ar-checkpoint", type=Path, required=True)
    command.add_argument("--ar-sha256", required=True)
    command.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--resume-optimizer", type=Path)
    command.add_argument("--split", default="train")
    command.add_argument("--steps", type=int, required=True)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--grad-accum-steps", type=int, default=1)
    command.add_argument("--lr", type=float, default=1e-4)
    command.add_argument("--adam-beta1", type=float, default=0.9)
    command.add_argument("--adam-beta2", type=float, default=0.95)
    command.add_argument("--weight-decay", type=float, default=0.1)
    command.add_argument("--grad-clip", type=float, default=1.0)
    command.add_argument("--ce-weight", type=float, default=1.0)
    command.add_argument("--kl-weight", type=float, default=1.0)
    command.add_argument("--save-every", type=int, default=100)
    command.add_argument("--keep-checkpoints", type=int, default=2)
    command.add_argument("--log-every", type=int, default=1)
    command.add_argument("--seed", type=int, default=1234)
    sub.add_parser("self-check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
    else:
        train(args)


if __name__ == "__main__":
    main()
