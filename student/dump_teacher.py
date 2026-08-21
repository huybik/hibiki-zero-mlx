#!/usr/bin/env python
"""Materialize compact CUDA teacher text targets onto strict student caches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from moshi.models import loaders
from moshi.run_inference import get_condition_tensors

from student.cache import (
    DISTILL_ROLE,
    STUDENT_ROLE,
    TEACHER_ROLE,
    load_cache,
    model_identity,
    save_shard,
    validate_metadata,
    validate_sample,
)
from student.contract import read_config, torch_lm_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--student-cache", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    return parser.parse_args()


def aligned_topk(
    model: Any,
    condition: Any,
    teacher_codes: torch.Tensor,
    student_frames: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(teacher_codes[None].to(device="cuda", dtype=torch.long), condition)
        logprobs = output.text_logits[0, 0].float().log_softmax(dim=-1)
        values, ids = logprobs.topk(top_k, dim=-1)
        teacher_mask = output.text_mask[0, 0]

    frames = min(student_frames, int(ids.shape[0]))
    aligned_ids = torch.full((student_frames, top_k), -1, dtype=torch.int32)
    aligned_values = torch.zeros((student_frames, top_k), dtype=torch.float16)
    aligned_mask = torch.zeros(student_frames, dtype=torch.bool)
    if frames:
        mask = teacher_mask[:frames].to(device="cpu", dtype=torch.bool)
        aligned_mask[:frames] = mask
        cpu_ids = ids[:frames].to(device="cpu", dtype=torch.int32)
        cpu_values = values[:frames].to(device="cpu", dtype=torch.float16)
        aligned_ids[:frames][mask] = cpu_ids[mask]
        aligned_values[:frames][mask] = cpu_values[mask]
    return aligned_ids, aligned_values, aligned_mask


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-teacher target materialization")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite targets in {args.out_dir}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    student_metadata, student_shards = load_cache(args.student_cache, STUDENT_ROLE)
    teacher_metadata, teacher_shards = load_cache(args.teacher_cache, TEACHER_ROLE)
    teacher_cfg = read_config(args.teacher_config)
    expected_teacher = model_identity(
        args.teacher_config,
        args.teacher_weights,
        args.teacher_repo,
        args.teacher_revision,
    )
    if teacher_metadata["model"] != expected_teacher:
        raise RuntimeError("Teacher cache does not match the requested teacher config and weights")
    if teacher_metadata["tokenizer"]["sha256"] != student_metadata["tokenizer"]["sha256"]:
        raise RuntimeError("Teacher and student tokenizer hashes differ")
    text_card = int(teacher_cfg["text_card"])
    if text_card != int(student_metadata["model"]["config"]["text_card"]):
        raise RuntimeError("Teacher/student text vocabularies differ")
    if args.top_k > text_card:
        raise ValueError(f"--top-k exceeds teacher text_card={text_card}")

    teacher_by_id: dict[str, dict[str, Any]] = {}
    for _, payload in teacher_shards:
        for sample in payload["samples"]:
            teacher_by_id[sample["id"]] = sample
    student_ids = {sample["id"] for _, payload in student_shards for sample in payload["samples"]}
    if set(teacher_by_id) != student_ids:
        raise RuntimeError(
            "Teacher/student cache IDs differ: "
            f"missing_teacher={sorted(student_ids - set(teacher_by_id))[:10]} "
            f"extra_teacher={sorted(set(teacher_by_id) - student_ids)[:10]}"
        )

    model = loaders.get_moshi_lm(
        args.teacher_weights,
        lm_kwargs=torch_lm_config(teacher_cfg),
        device="cuda",
        dtype=torch.bfloat16,
    )
    model.eval()
    model.requires_grad_(False)
    condition = (
        None
        if model.fuser is None
        else get_condition_tensors(
            str(teacher_cfg.get("model_type", "hibiki")), model, batch_size=1, cfg_coef=1.0
        )
    )

    output_metadata = dict(student_metadata)
    output_metadata["role"] = DISTILL_ROLE
    output_metadata["teacher"] = {
        **teacher_metadata["model"],
        "cache_generation": teacher_metadata["generation"],
    }
    output_metadata["generation"] = {
        "method": "frozen_teacher_forced_text_topk_v1",
        "top_k": args.top_k,
        "alignment": "same_frame_12.5hz",
        "audio_heads": "unused",
    }
    validate_metadata(output_metadata, DISTILL_ROLE)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for source_path, payload in student_shards:
        output_samples: list[dict[str, Any]] = []
        for student_sample in payload["samples"]:
            teacher_sample = teacher_by_id[student_sample["id"]]
            if teacher_sample["split"] != student_sample["split"]:
                raise RuntimeError(f"Split mismatch for id={student_sample['id']}")
            ids, logprobs, mask = aligned_topk(
                model,
                condition,
                teacher_sample["codes"],
                int(student_sample["codes"].shape[1]),
                args.top_k,
            )
            sample = dict(student_sample)
            sample.update(
                {
                    "teacher_text_topk_ids": ids,
                    "teacher_text_topk_logprobs": logprobs,
                    "teacher_text_mask": mask,
                }
            )
            validate_sample(sample, output_metadata, student_sample["id"])
            output_samples.append(sample)
            written += 1
        output_path = args.out_dir / source_path.name
        save_shard(output_path, output_metadata, output_samples)
        print(f"Wrote {len(output_samples)} samples -> {output_path}")

    load_cache(args.out_dir, DISTILL_ROLE)
    print(f"PASS: materialized text-only teacher targets for {written} student samples")


if __name__ == "__main__":
    main()
