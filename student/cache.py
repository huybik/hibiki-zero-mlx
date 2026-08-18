#!/usr/bin/env python
"""Build and strictly validate CUDA training caches for the mobile student."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from contract import DEFAULT_CONFIG, read_config, sha256, validate_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIMI = ROOT / "weights" / "mimi-pytorch-e351c8d8@125.safetensors"
DEFAULT_TOKENIZER = ROOT / "weights" / "tokenizer_spm_48k_multi6_2.model"
CACHE_FORMAT = "hibiki_student_cache_v1"
METADATA_FORMAT = "hibiki_student_cache_metadata_v1"
STUDENT_ROLE = "student_hard"
TEACHER_ROLE = "teacher_context"
DISTILL_ROLE = "student_text_distillation"
HASH_RE = re.compile(r"[0-9a-f]{64}")
ROLES = {
    STUDENT_ROLE: (16, 8),
    TEACHER_ROLE: (32, 16),
    DISTILL_ROLE: (16, 8),
}
PAIR_FIELDS = {
    "id",
    "split",
    "codes",
    "source_frames",
    "target_frames",
    "text_tokens",
    "source_audio",
    "target_audio",
    "target_text",
    "text_frames",
}
DISTILL_FIELDS = {
    "teacher_text_topk_ids",
    "teacher_text_topk_logprobs",
    "teacher_text_mask",
}


def artifact_identity(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"name": path.name, "sha256": sha256(path)}


def model_identity(
    config_path: Path,
    weights_path: Path | None,
    repo: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    return {
        "repo": repo,
        "revision": revision,
        "config": read_config(config_path),
        "config_artifact": artifact_identity(config_path),
        "weights_artifact": None if weights_path is None else artifact_identity(weights_path),
    }


def make_metadata(
    role: str,
    config_path: Path,
    mimi_path: Path,
    tokenizer_path: Path,
    *,
    weights_path: Path | None = None,
    teacher: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    model_repo: str | None = None,
    model_revision: str | None = None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(f"Unsupported cache role: {role}")
    model = model_identity(config_path, weights_path, model_repo, model_revision)
    n_q, dep_q = int(model["config"]["n_q"]), int(model["config"]["dep_q"])
    return {
        "format": METADATA_FORMAT,
        "role": role,
        "audio": {"sample_rate": 24_000, "frame_rate": 12.5, "frame_samples": 1_920},
        "layout": {
            "n_q": n_q,
            "dep_q": dep_q,
            "rows": 1 + n_q,
            "target_rows": [1, 1 + dep_q],
            "source_rows": [1 + dep_q, 1 + n_q],
        },
        "model": model,
        "mimi": artifact_identity(mimi_path),
        "tokenizer": artifact_identity(tokenizer_path),
        "teacher": teacher,
        "generation": generation,
    }


def exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"{label} must have exactly {sorted(keys)}")


def validate_artifact(value: Any, label: str) -> None:
    exact_keys(value, {"name", "sha256"}, label)
    if not isinstance(value["name"], str) or not value["name"]:
        raise RuntimeError(f"{label} has no name")
    if not isinstance(value["sha256"], str) or HASH_RE.fullmatch(value["sha256"]) is None:
        raise RuntimeError(f"{label} does not contain a SHA-256")


def validate_metadata(metadata: Any, expected_role: str | None = None) -> None:
    keys = {
        "format",
        "role",
        "audio",
        "layout",
        "model",
        "mimi",
        "tokenizer",
        "teacher",
        "generation",
    }
    exact_keys(metadata, keys, "cache metadata")
    role = metadata["role"]
    if metadata["format"] != METADATA_FORMAT or role not in ROLES:
        raise RuntimeError("Unsupported student cache metadata")
    if expected_role is not None and role != expected_role:
        raise RuntimeError(f"Cache role mismatch: {role!r} != {expected_role!r}")
    if metadata["audio"] != {"sample_rate": 24_000, "frame_rate": 12.5, "frame_samples": 1_920}:
        raise RuntimeError("Cache audio contract must be 24 kHz and 12.5 Hz")
    exact_keys(
        metadata["model"],
        {"repo", "revision", "config", "config_artifact", "weights_artifact"},
        "model",
    )
    cfg = metadata["model"]["config"]
    if not isinstance(cfg, dict):
        raise RuntimeError("Cache must embed the complete model config")
    validate_artifact(metadata["model"]["config_artifact"], "model config")
    validate_artifact(metadata["mimi"], "Mimi")
    validate_artifact(metadata["tokenizer"], "tokenizer")
    if metadata["model"]["weights_artifact"] is not None:
        validate_artifact(metadata["model"]["weights_artifact"], "model weights")
    n_q, dep_q = ROLES[role]
    if (cfg.get("n_q"), cfg.get("dep_q")) != (n_q, dep_q):
        raise RuntimeError(f"{role} requires n_q={n_q}, dep_q={dep_q}")
    if role != TEACHER_ROLE:
        validate_config(cfg)
    elif metadata["model"]["weights_artifact"] is None:
        raise RuntimeError("Teacher cache must identify teacher weights")
    elif not all(
        isinstance(metadata["model"][key], str) and metadata["model"][key]
        for key in ("repo", "revision")
    ):
        raise RuntimeError("Teacher cache must pin its repository and revision")
    layout = {
        "n_q": n_q,
        "dep_q": dep_q,
        "rows": 1 + n_q,
        "target_rows": [1, 1 + dep_q],
        "source_rows": [1 + dep_q, 1 + n_q],
    }
    if metadata["layout"] != layout:
        raise RuntimeError(f"{role} row layout changed")
    if role == STUDENT_ROLE and (
        metadata["teacher"] is not None or metadata["generation"] is not None
    ):
        raise RuntimeError("Hard-pair cache cannot claim teacher provenance")
    if role == STUDENT_ROLE and metadata["model"]["weights_artifact"] is not None:
        raise RuntimeError("Hard-pair cache must not depend on a student checkpoint")
    if role == TEACHER_ROLE and not isinstance(metadata["generation"], dict):
        raise RuntimeError("Teacher cache must record how its context was assembled")
    if role == DISTILL_ROLE and not all(
        isinstance(metadata[key], dict) for key in ("teacher", "generation")
    ):
        raise RuntimeError("Distillation cache lacks teacher provenance")
    if role == DISTILL_ROLE:
        teacher = metadata["teacher"]
        exact_keys(
            teacher,
            {
                "repo",
                "revision",
                "config",
                "config_artifact",
                "weights_artifact",
                "cache_generation",
            },
            "distillation teacher",
        )
        if (
            not isinstance(teacher["repo"], str)
            or not teacher["repo"]
            or not isinstance(teacher["revision"], str)
            or not teacher["revision"]
            or not isinstance(teacher["config"], dict)
            or (teacher["config"].get("n_q"), teacher["config"].get("dep_q")) != (32, 16)
            or not isinstance(teacher["cache_generation"], dict)
        ):
            raise RuntimeError("Distillation teacher identity is incomplete")
        validate_artifact(teacher["config_artifact"], "teacher config")
        validate_artifact(teacher["weights_artifact"], "teacher weights")
        generation = metadata["generation"]
        exact_keys(
            generation,
            {"method", "top_k", "alignment", "audio_heads"},
            "distillation generation",
        )
        if (
            generation["method"] != "frozen_teacher_forced_text_topk_v1"
            or isinstance(generation["top_k"], bool)
            or not isinstance(generation["top_k"], int)
            or generation["top_k"] <= 0
            or generation["alignment"] != "same_frame_12.5hz"
            or generation["audio_heads"] != "unused"
        ):
            raise RuntimeError("Unsupported distillation target generation")


def tensor_range(tensor: torch.Tensor, low: int, high: int, label: str) -> None:
    if tensor.numel() and (int(tensor.min()) < low or int(tensor.max()) > high):
        raise RuntimeError(f"{label} values are outside [{low}, {high}]")


def validate_sample(sample: Any, metadata: dict[str, Any], location: str) -> None:
    role = metadata["role"]
    required = {"id", "split", "codes"}
    if role in ROLES:
        required |= PAIR_FIELDS
    if role == DISTILL_ROLE:
        required |= DISTILL_FIELDS
    actual = set(sample) if isinstance(sample, dict) else set()
    if not required <= actual or actual - required - {"teacher_sequence_codes"}:
        raise RuntimeError(f"{location} has invalid sample fields")
    if not all(isinstance(sample[key], str) and sample[key] for key in ("id", "split")):
        raise RuntimeError(f"{location} has an empty id or split")
    codes = sample["codes"]
    rows = metadata["layout"]["rows"]
    integers = (torch.int16, torch.int32, torch.int64)
    if (
        not isinstance(codes, torch.Tensor)
        or codes.dtype not in integers
        or codes.ndim != 2
        or tuple(codes.shape)[0] != rows
        or codes.shape[1] == 0
    ):
        raise RuntimeError(f"{location} id={sample['id']} codes must be [{rows}, T]")
    cfg = metadata["model"]["config"]
    tensor_range(codes[:1], -1, int(cfg["text_card"]) - 1, "text")
    tensor_range(codes[1:], -1, int(cfg["card"]), "audio")
    sequence = sample.get("teacher_sequence_codes")
    if sequence is not None:
        dep_q = metadata["layout"]["dep_q"]
        if (
            role == TEACHER_ROLE
            or not isinstance(sequence, torch.Tensor)
            or sequence.dtype not in integers
            or sequence.ndim != 2
            or tuple(sequence.shape)[0] != dep_q
            or sequence.shape[1] == 0
        ):
            raise RuntimeError(f"{location} teacher_sequence_codes must be [{dep_q}, T]")
        tensor_range(sequence, 0, int(cfg["card"]) - 1, "teacher sequence")
    if role in ROLES:
        counts = [sample[key] for key in ("source_frames", "target_frames", "text_tokens")]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts
        ):
            raise RuntimeError(f"{location} has invalid frame/token counts")
        if sample["source_frames"] >= codes.shape[1] or sample["target_frames"] > codes.shape[1]:
            raise RuntimeError(f"{location} frame counts exceed codes")
        text_frames = sample["text_frames"]
        if (
            not isinstance(text_frames, list)
            or len(text_frames) != sample["text_tokens"]
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in text_frames)
            or text_frames != sorted(set(text_frames))
            or text_frames[0] < 0
            or text_frames[-1] >= codes.shape[1]
        ):
            raise RuntimeError(f"{location} has invalid aligned text_frames")
    if role == DISTILL_ROLE:
        ids, values, mask = (sample[key] for key in DISTILL_FIELDS)
        if (
            not isinstance(ids, torch.Tensor)
            or not isinstance(values, torch.Tensor)
            or ids.dtype not in integers
            or not values.is_floating_point()
            or ids.ndim != 2
            or ids.shape != values.shape
        ):
            raise RuntimeError(f"{location} has malformed top-k tensors")
        if (
            ids.shape[0] != codes.shape[1]
            or ids.shape[1] != int(metadata["generation"]["top_k"])
            or not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (codes.shape[1],)
        ):
            raise RuntimeError(f"{location} teacher targets are not aligned to student frames")
        tensor_range(ids[mask], 0, int(cfg["text_card"]) - 1, "teacher top-k ids")
        if not torch.isfinite(values[mask]).all() or ((ids[~mask] != -1).any()):
            raise RuntimeError(f"{location} has invalid teacher top-k values")
        valid_ids = ids[mask]
        valid_logprobs = values[mask].float()
        sorted_ids = valid_ids.sort(-1).values
        if valid_ids.shape[0] and (
            (valid_logprobs > 1e-3).any()
            or (valid_logprobs.exp().sum(-1) > 1.001).any()
            or (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()
        ):
            raise RuntimeError(
                f"{location} teacher top-k IDs/probabilities are not an absolute distribution"
            )


def load_cache(
    cache_dir: Path, expected_role: str | None = None
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    paths = sorted(cache_dir.glob("shard_*.pt"))
    if not paths:
        raise RuntimeError(f"No shard_*.pt files in {cache_dir}")
    metadata, shards, ids = None, [], set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        exact_keys(payload, {"format", "metadata", "samples"}, str(path))
        if payload["format"] != CACHE_FORMAT:
            raise RuntimeError(f"Unsupported cache format in {path}")
        validate_metadata(payload["metadata"], expected_role)
        if metadata is not None and payload["metadata"] != metadata:
            raise RuntimeError(f"Shard metadata differs in {path}")
        metadata = payload["metadata"]
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise RuntimeError(f"Empty shard: {path}")
        for index, sample in enumerate(payload["samples"]):
            validate_sample(sample, metadata, f"{path}:{index}")
            if sample["id"] in ids:
                raise RuntimeError(f"Duplicate id={sample['id']!r}")
            ids.add(sample["id"])
        shards.append((path, payload))
    return metadata, shards


def save_shard(path: Path, metadata: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"format": CACHE_FORMAT, "metadata": metadata, "samples": samples}, tmp)
    tmp.replace(path)


def read_pairs(path: Path) -> list[dict[str, Any]]:
    fields = {"id", "split", "source_audio", "target_audio", "target_text", "text_frames"}
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fields <= set(reader.fieldnames or []):
                raise ValueError(f"{path} must contain {sorted(fields)}")
            raw = list(reader)
    else:
        raise ValueError(f"Pair file must be JSONL or CSV: {path}")
    rows: list[dict[str, Any]] = []
    for raw_row in raw:
        row = {key: str(raw_row.get(key, "")).strip() for key in fields - {"text_frames"}}
        value = raw_row.get("text_frames")
        frames = json.loads(value) if isinstance(value, str) else value
        if (
            not isinstance(frames, list)
            or not frames
            or any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames)
            or frames != sorted(set(frames))
            or frames[0] < 0
        ):
            raise ValueError(f"id={row.get('id')!r} has invalid text_frames")
        row["text_frames"] = frames
        rows.append(row)
    if not rows or any(not row[key] for row in rows for key in fields - {"text_frames"}):
        raise ValueError(f"{path} is empty or has an empty required field")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path} has duplicate IDs")
    return rows


def encode(path: Path, mimi: Any, codebooks: int, device: torch.device) -> torch.Tensor:
    import sphn

    wav, rate = sphn.read(str(path), sample_rate=24_000)
    wav = torch.as_tensor(wav).float()
    if wav.ndim == 1:
        wav = wav[None]
    if rate != 24_000 or wav.ndim != 2:
        raise RuntimeError(f"Invalid audio: {path}")
    with torch.inference_mode():
        codes = mimi.encode(wav.mean(0, keepdim=True)[None].to(device))
    if tuple(codes.shape[:2]) != (1, codebooks):
        raise RuntimeError(f"Mimi returned {tuple(codes.shape)}, expected [1, {codebooks}, T]")
    return codes[0].cpu().to(torch.int32)


def assemble(
    cfg: dict[str, Any],
    source: torch.Tensor,
    target: torch.Tensor,
    tokens: list[int],
    text_frames: list[int],
) -> torch.Tensor:
    n_q, dep_q = int(cfg["n_q"]), int(cfg["dep_q"])
    if n_q != 2 * dep_q or source.shape[0] != dep_q or target.shape[0] != dep_q:
        raise RuntimeError("Cache assembly requires equal source and target stream counts")
    if len(tokens) != len(text_frames):
        raise RuntimeError("text_frames must locate every SentencePiece token including EOS")
    codes = torch.full(
        (1 + n_q, max(source.shape[1] + 1, target.shape[1], text_frames[-1] + 1)),
        -1,
        dtype=torch.int32,
    )
    codes[0].fill_(cfg["existing_text_padding_id"])
    codes[0, text_frames] = torch.tensor(tokens, dtype=torch.int32)
    if text_frames[-1] + 1 < codes.shape[1]:
        codes[0, text_frames[-1] + 1 :] = -1
    codes[1 : 1 + dep_q, : target.shape[1]] = target
    codes[1 + dep_q :, : source.shape[1]] = source
    codes[1 + dep_q :, source.shape[1]] = cfg["card"]
    return codes


def build(args: argparse.Namespace) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Student cache construction requires CUDA")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg, rows = read_config(args.config), read_pairs(args.pairs)
    if args.role == STUDENT_ROLE:
        validate_config(cfg)
        weights = None
        generation = None
    else:
        if (cfg.get("n_q"), cfg.get("dep_q")) != (32, 16):
            raise ValueError("teacher_context requires a 32-stream, 16-head config")
        if args.weights is None:
            raise ValueError("teacher_context requires --weights")
        if not args.repo or not args.revision:
            raise ValueError("teacher_context requires --repo and --revision")
        weights = args.weights
        generation = {"method": "aligned_hard_pair_context_v1"}
    metadata = make_metadata(
        args.role,
        args.config,
        args.mimi,
        args.tokenizer,
        weights_path=weights,
        generation=generation,
        model_repo=args.repo,
        model_revision=args.revision,
    )
    import sentencepiece
    from moshi.models import loaders

    device = torch.device(args.device)
    codebooks = int(cfg["dep_q"])
    mimi = loaders.get_mimi(args.mimi, num_codebooks=codebooks, device=device)
    if (mimi.sample_rate, mimi.frame_rate, mimi.cardinality) != (24_000, 12.5, cfg["card"]):
        raise RuntimeError("Mimi does not match the student contract")
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    if tokenizer.vocab_size() != cfg["text_card"] or tokenizer.eos_id() < 0:
        raise RuntimeError("Tokenizer does not match the student contract")
    for start in range(0, len(rows), args.shard_size):
        samples = []
        for row in rows[start : start + args.shard_size]:
            paths = []
            for key in ("source_audio", "target_audio"):
                path = Path(row[key])
                path = path if path.is_absolute() else args.pairs.parent / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(path)
            source = encode(paths[0], mimi, codebooks, device)
            target = encode(paths[1], mimi, codebooks, device)
            tokens = list(tokenizer.encode(row["target_text"], out_type=int)) + [tokenizer.eos_id()]
            if len(tokens) != len(row["text_frames"]):
                raise RuntimeError(
                    f"id={row['id']} text_frames has {len(row['text_frames'])} entries "
                    f"for {len(tokens)} tokens including EOS"
                )
            sample = {
                **row,
                "codes": assemble(cfg, source, target, tokens, row["text_frames"]),
                "source_frames": source.shape[1],
                "target_frames": target.shape[1],
                "text_tokens": len(tokens),
            }
            validate_sample(sample, metadata, row["id"])
            samples.append(sample)
        save_shard(args.out_dir / f"shard_{start // args.shard_size:05d}.pt", metadata, samples)
    load_cache(args.out_dir, args.role)
    print(f"PASS: {len(rows)} strict {args.role} samples in {args.out_dir}")


def self_check() -> None:
    metadata = make_metadata(STUDENT_ROLE, DEFAULT_CONFIG, DEFAULT_MIMI, DEFAULT_TOKENIZER)
    cfg = metadata["model"]["config"]
    sample = {
        "id": "synthetic",
        "split": "smoke",
        "codes": assemble(
            cfg,
            torch.zeros(8, 2, dtype=torch.int32),
            torch.ones(8, 2, dtype=torch.int32),
            [2],
            [0],
        ),
        "source_frames": 2,
        "target_frames": 2,
        "text_tokens": 1,
        "source_audio": "source.wav",
        "target_audio": "target.wav",
        "target_text": "text",
        "text_frames": [0],
        "teacher_sequence_codes": torch.zeros(8, 2, dtype=torch.int32),
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        student_dir = root / "student"
        teacher_dir = root / "teacher"
        student_dir.mkdir()
        teacher_dir.mkdir()
        save_shard(student_dir / "shard_00000.pt", metadata, [sample])
        load_cache(student_dir, STUDENT_ROLE)

        fake_weights = root / "teacher.safetensors"
        fake_weights.write_bytes(b"synthetic")
        teacher_config = ROOT / "weights" / "config.json"
        teacher_metadata = make_metadata(
            TEACHER_ROLE,
            teacher_config,
            DEFAULT_MIMI,
            DEFAULT_TOKENIZER,
            weights_path=fake_weights,
            generation={"method": "aligned_hard_pair_context_v1"},
            model_repo="kyutai/hibiki-zero-3b-pytorch-bf16",
            model_revision="73175ce6243f8ad66b2138b0264a80044b35c1bd",
        )
        teacher_cfg = teacher_metadata["model"]["config"]
        teacher_sample = dict(sample)
        teacher_sample["codes"] = assemble(
            teacher_cfg,
            torch.zeros(16, 2, dtype=torch.int32),
            torch.ones(16, 2, dtype=torch.int32),
            [2],
            [0],
        )
        teacher_sample.pop("teacher_sequence_codes")
        save_shard(teacher_dir / "shard_00000.pt", teacher_metadata, [teacher_sample])
        load_cache(teacher_dir, TEACHER_ROLE)

        torch.save(
            {"format": "hibiki_vn_lora_cache_v1"},
            student_dir / "shard_00000.pt",
        )
        try:
            load_cache(student_dir, STUDENT_ROLE)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Legacy 3B cache was accepted")
    print("PASS: strict 17/33-row caches and optional teacher sequence; legacy cache rejected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build")
    command.add_argument("--pairs", type=Path, required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--mimi", type=Path, default=DEFAULT_MIMI)
    command.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    command.add_argument("--role", choices=(STUDENT_ROLE, TEACHER_ROLE), default=STUDENT_ROLE)
    command.add_argument("--weights", type=Path)
    command.add_argument("--repo")
    command.add_argument("--revision")
    command.add_argument("--device", default="cuda")
    command.add_argument("--shard-size", type=int, default=32)
    command = sub.add_parser("validate")
    command.add_argument("cache_dir", type=Path)
    command.add_argument("--role", choices=sorted(ROLES), required=True)
    sub.add_parser("self-check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "validate":
        metadata, shards = load_cache(args.cache_dir, args.role)
        count = sum(len(payload["samples"]) for _, payload in shards)
        print(f"PASS: {count} {metadata['role']} samples; identical metadata")
    else:
        self_check()


if __name__ == "__main__":
    main()
