#!/usr/bin/env python
"""CUDA head-only training for the frozen 12-layer student's parallel_v1 head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from cache import artifact_identity
from capture_parallel import load_parallel_cache
from contract import read_config, sha256
from parallel import (
    PARALLEL_PARAMETERS,
    ParallelHead,
    require_compatible_configs,
    validate_qualified_ar,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AR_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_ar.json"
DEFAULT_PARALLEL_CONFIG = ROOT / "student" / "configs" / "hibiki_m_12l_parallel_v1.json"
DEFAULT_TOKENIZER = ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"
RUN_FORMAT = "hibiki_parallel_head_run_v1"
CHECKPOINT_FORMAT = "hibiki_parallel_head_checkpoint_v1"
OPTIMIZER_FORMAT = "hibiki_parallel_head_optimizer_v1"
HEAD_RE = re.compile(r"head_step(\d+)\.safetensors")
OPTIMIZER_RE = re.compile(r"optimizer_step(\d+)\.pt")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask & (targets >= 0) & (targets < logits.shape[-1])
    count = mask.sum()
    loss = F.cross_entropy(logits[mask].float(), targets[mask].long(), reduction="sum")
    return loss / count.clamp(min=1), count


def residual_bucket_topk_kl(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL over absolute top-k atoms plus one aggregate omitted-mass atom."""
    if student_logits.ndim != 4 or teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError("Malformed student or teacher audio tensors")
    if teacher_ids.shape[:3] != student_logits.shape[:3] or mask.shape != teacher_ids.shape[:3]:
        raise ValueError("Teacher audio targets are not aligned with student logits")
    count = mask.sum()
    safe_ids = teacher_ids.clamp_min(0).long()
    logits = student_logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    student_top_logprobs = logits.gather(-1, safe_ids) - log_z.unsqueeze(-1)
    teacher_logprobs = teacher_logprobs.float()
    teacher_probs = teacher_logprobs.exp()
    teacher_top_mass = teacher_probs.sum(-1).clamp(max=1.0)
    student_top_mass = student_top_logprobs.exp().sum(-1).clamp(max=1.0 - 1e-7)
    teacher_other = 1.0 - teacher_top_mass
    student_other_logprob = torch.log1p(-student_top_mass)
    top_terms = (teacher_probs * (teacher_logprobs - student_top_logprobs)).sum(-1)
    other_terms = teacher_other * (teacher_other.clamp_min(1e-30).log() - student_other_logprob)
    frame_kl = top_terms + other_terms
    return frame_kl[mask].sum() / count.clamp(min=1), count


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


def cache_identity(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    shards = [
        {"name": path.name, "sha256": sha256(path)} for path in sorted(cache_dir.glob("shard_*.pt"))
    ]
    return {
        "metadata_sha256": canonical_sha256(metadata),
        "shards_sha256": canonical_sha256(shards),
        "shards": shards,
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
        "qualification_receipt": artifact_identity(args.qualification_receipt),
        "tokenizer": artifact_identity(args.tokenizer),
        "cache": cache_hashes,
        "training": {key: getattr(args, key) for key in training_keys},
    }


def prepare_run_dir(out_dir: Path, contract: dict[str, Any], resume: bool) -> str:
    path = out_dir / "run.json"
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if resume:
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Parallel resume contract differs from the original run.json")
    else:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty run directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return canonical_sha256(contract)


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_pairs(out_dir: Path) -> dict[int, tuple[Path, Path]]:
    heads = {
        int(match.group(1)): path
        for path in out_dir.glob("head_step*.safetensors")
        if (match := HEAD_RE.fullmatch(path.name)) is not None
    }
    optimizers = {
        int(match.group(1)): path
        for path in out_dir.glob("optimizer_step*.pt")
        if (match := OPTIMIZER_RE.fullmatch(path.name)) is not None
    }
    if heads.keys() != optimizers.keys():
        raise RuntimeError("Run directory contains an incomplete parallel checkpoint pair")
    return {step: (heads[step], optimizers[step]) for step in heads}


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


def prune_checkpoints(out_dir: Path, keep: int) -> None:
    pairs = checkpoint_pairs(out_dir)
    for step in sorted(pairs)[: max(0, len(pairs) - keep)]:
        head_path, optimizer_path = pairs[step]
        head_path.unlink()
        optimizer_path.unlink()


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
    pairs = checkpoint_pairs(path.parent)
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
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.cuda()
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
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
    positive = (
        "steps",
        "batch_size",
        "grad_accum_steps",
        "save_every",
        "keep_checkpoints",
        "log_every",
    )
    if any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Step, batch, accumulation, save, keep, and log values must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        raise ValueError("--grad-clip must be finite and positive")
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

    ar_cfg = validate_qualified_ar(
        args.ar_config,
        args.ar_checkpoint,
        args.ar_sha256,
        args.qualification_receipt,
    )
    cfg = read_config(args.parallel_config)
    require_compatible_configs(ar_cfg, cfg)
    metadata, shards = load_parallel_cache(args.cache_dir)
    if metadata["backbone"]["config"] != ar_cfg:
        raise RuntimeError("Capture cache backbone config differs from --ar-config")
    expected_artifacts = {
        "config_artifact": artifact_identity(args.ar_config),
        "checkpoint_artifact": artifact_identity(args.ar_checkpoint),
        "qualification_receipt": artifact_identity(args.qualification_receipt),
    }
    for key, value in expected_artifacts.items():
        if metadata["backbone"][key] != value:
            raise RuntimeError(f"Capture cache {key} differs from the exact qualified base")
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
    if any(parameter is text_weight for parameter in parameters):
        raise RuntimeError("Frozen text embedding entered the head optimizer")

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
                kl, _ = residual_bucket_topk_kl(
                    logits,
                    batch["teacher_ids"],
                    batch["teacher_logprobs"],
                    batch["mask"],
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
            prune_checkpoints(args.out_dir, args.keep_checkpoints)
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
    actual_kl, count = residual_bucket_topk_kl(
        student_logits, ids, teacher_logprobs, torch.ones(1, 1, 1, dtype=torch.bool)
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
    command.add_argument("--qualification-receipt", type=Path, required=True)
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
