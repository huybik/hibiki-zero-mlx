#!/usr/bin/env python
"""Build and audit provenance-preserving VIVOS Mimi cache v2 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from finetune.cache_codes import (
    FRAME_RATE,
    SAMPLE_RATE,
    assemble_codes,
    check_device,
    encode_audio,
    require_runtime_deps,
    target_delay_s,
    text_tokens,
)
from finetune.utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    read_json,
    require_file,
)

CACHE_FORMAT = "hibiki_vn_lora_cache_v2"
ALIGNMENT_SCHEMA = "hibiki_vivos_single_sentence_coarse_alignment_v1"
QA_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_qa_v1"
CAMPAIGN_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_v1"
STRATUM = "real_source_st_core"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gender-files", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--alignment-seed", type=int, default=1234)
    parser.add_argument("--target-delay-ratio", type=float, default=0.5)
    return parser.parse_args()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise RuntimeError(f"Empty JSONL line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def immutable_write(path: Path, value: bytes) -> None:
    if path.exists() and path.read_bytes() != value:
        raise RuntimeError(f"Refusing to change immutable cache artifact: {path}")
    atomic_write(path, value)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def load_genders(paths: list[Path]) -> tuple[dict[str, str], list[dict[str, str]]]:
    genders: dict[str, str] = {}
    records = []
    for path in (item.expanduser().resolve() for item in paths):
        records.append(attestation(path))
        for line in path.read_text(encoding="utf-8").splitlines():
            speaker, value = line.split()
            gender = "male" if value.casefold().startswith("m") else "female"
            if speaker in genders and genders[speaker] != gender:
                raise RuntimeError(f"Conflicting gender for {speaker}")
            genders[speaker] = gender
    return genders, records


def load_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = args.plan.expanduser().resolve()
    accepted_path = args.accepted.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    report_path = args.qa_report.expanduser().resolve()
    plan = read_jsonl(plan_path)
    accepted = read_jsonl(accepted_path)
    selections = read_jsonl(selection_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != QA_SCHEMA or report.get("decision") != "go":
        raise RuntimeError("Final full QA report must have decision=go")
    outputs = report.get("outputs", {})
    if outputs.get("accepted") != attestation(accepted_path) or outputs.get(
        "selection"
    ) != attestation(selection_path):
        raise RuntimeError("Accepted/selection manifests are not bound to the QA report")
    plan_record = report.get("inputs", {}).get("plan", {})
    if plan_record != attestation(plan_path):
        raise RuntimeError("Generation plan is not bound to the QA report")
    campaign_config = (
        report.get("inputs", {}).get("campaign_artifacts", {}).get("campaign_config", {})
    )
    if campaign_config != attestation(Path(str(campaign_config.get("path", "")))):
        raise RuntimeError("Frozen TTS campaign config changed")
    plan_by_id = {str(row.get("id", "")): row for row in plan}
    accepted_by_id = {str(row.get("id", "")): row for row in accepted}
    selection_by_id = {str(row.get("id", "")): row for row in selections}
    if (
        not accepted
        or any(row.get("schema_version") != CAMPAIGN_SCHEMA for row in plan)
        or any(row.get("eligibility_split") not in {"train", "dev"} for row in plan)
        or len(plan_by_id) != len(plan)
        or len(accepted_by_id) != len(accepted)
        or len(selection_by_id) != len(selections)
    ):
        raise RuntimeError("Invalid scope, schema, or duplicate ids in cache inputs")
    if set(accepted_by_id) - set(plan_by_id) or set(selection_by_id) != set(plan_by_id):
        raise RuntimeError("Plan, selection, and accepted scopes disagree")
    for row_id, row in accepted_by_id.items():
        planned = plan_by_id[row_id]
        selected = selection_by_id[row_id]
        if (
            selected.get("status") != "accepted"
            or row.get("eligibility_split") not in {"train", "dev"}
            or planned.get("eligibility_split") != row.get("eligibility_split")
            or row.get("schema_version") != QA_SCHEMA
        ):
            raise RuntimeError(f"Invalid accepted cache row: {row_id}")
        if (
            row.get("source_audio") != planned.get("source_audio")
            or row.get("source_provenance") != planned.get("source_provenance")
            or row.get("reference") != planned.get("reference")
        ):
            raise RuntimeError(f"Accepted provenance differs from plan: {row_id}")
        selected_candidate = next(
            (
                candidate
                for candidate in selected.get("candidates", [])
                if candidate.get("candidate_id") == selected.get("selected_candidate_id")
            ),
            None,
        )
        if (
            selected_candidate is None
            or selected_candidate.get("attempt") != row.get("target_audio", {}).get("attempt")
            or selected_candidate.get("metric_sha256")
            != row.get("target_qa", {}).get("metric_sha256")
            or selected_candidate.get("audio_sha256") != row.get("target_audio", {}).get("sha256")
        ):
            raise RuntimeError(f"Accepted QA selection provenance mismatch: {row_id}")
        target = Path(str(row.get("target_audio", {}).get("path", "")))
        if not target.is_file() or sha256_file(target) != row.get("target_audio", {}).get("sha256"):
            raise RuntimeError(f"Selected target audio changed: {row_id}")
    return sorted(accepted, key=lambda row: (row["eligibility_split"], row["id"])), {
        "plan": attestation(plan_path),
        "accepted": attestation(accepted_path),
        "selection": attestation(selection_path),
        "qa_report": attestation(report_path),
        "campaign_config": campaign_config,
    }


def supervision_counts(text_start: int, token_count: int, frames: int) -> dict[str, Any]:
    content = token_count - 1
    eos = 1
    ignored = frames - text_start - token_count
    if min(content, ignored, text_start) < 0:
        raise RuntimeError("Invalid text supervision accounting")
    total = text_start + content + eos
    return {
        "prefix_pad": text_start,
        "content": content,
        "eos": eos,
        "ignored_tail": ignored,
        "batch_pad": "schedule_dependent_unreported",
        "effective_loss_mass": {
            "prefix_pad_weight_1.0": {
                "pad": text_start,
                "content_plus_eos": content + eos,
                "pad_fraction": text_start / total if total else 0.0,
            },
            "prefix_pad_weight_0.5": {
                "pad": 0.5 * text_start,
                "content_plus_eos": content + eos,
                "pad_fraction": (0.5 * text_start) / (0.5 * text_start + content + eos),
            },
        },
    }


def shard_path(out_root: Path, split: str, index: int) -> Path:
    return out_root / split / f"shard_{index:05d}.pt"


def save_shard(torch: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_shard(
    torch: Any, path: Path, config_sha: str, expected_ids: list[str]
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    ids = [str(sample.get("id", "")) for sample in payload.get("samples", [])]
    if (
        payload.get("format") != CACHE_FORMAT
        or payload.get("cache_config_sha256") != config_sha
        or ids != expected_ids
    ):
        raise RuntimeError(f"Existing shard does not match its frozen slice: {path}")
    return payload


def audit_shards(
    torch: Any, out_root: Path, accepted_ids: set[str], cfg: dict[str, Any], config_sha: str
) -> dict[str, Any]:
    n_q, dep_q, card = int(cfg["n_q"]), int(cfg["dep_q"]), int(cfg["card"])
    source_q = n_q - dep_q
    samples: list[dict[str, Any]] = []
    duplicate_ids: set[str] = set()
    seen: set[str] = set()
    failures: list[dict[str, str]] = []
    totals = {key: 0 for key in ("prefix_pad", "content", "eos", "ignored_tail")}
    for path in sorted([*out_root.glob("train/shard_*.pt"), *out_root.glob("dev/shard_*.pt")]):
        payload = torch.load(path, map_location="cpu")
        if (
            payload.get("format") != CACHE_FORMAT
            or payload.get("cache_config_sha256") != config_sha
        ):
            raise RuntimeError(f"Cache shard contract mismatch: {path}")
        for sample in payload["samples"]:
            row_id = str(sample["id"])
            if row_id in seen:
                duplicate_ids.add(row_id)
            seen.add(row_id)
            codes = sample["codes"]
            reason = []
            if codes.ndim != 2 or tuple(codes.shape) != (1 + n_q, sample["frames"]):
                reason.append("shape")
            elif not bool(((codes[0] >= 0) & (codes[0] < int(cfg["text_card"]))).all()):
                reason.append("text_range")
            else:
                target = codes[1 : 1 + dep_q]
                source = codes[1 + dep_q :]
                if not bool(((target >= -1) & (target < card)).all()):
                    reason.append("target_range")
                if not bool(((source >= -1) & (source <= card)).all()):
                    reason.append("source_range")
                for codebook in range(source_q):
                    eos = (source[codebook] == card).nonzero().flatten()
                    if eos.numel() != 1:
                        reason.append("source_eos")
                        continue
                    content = source[codebook, : int(eos[0])]
                    if content.numel() and content.unique().numel() <= 1:
                        reason.append("degenerate_source")
                for codebook in range(dep_q):
                    content = target[codebook][(target[codebook] >= 0) & (target[codebook] < card)]
                    if content.numel() and content.unique().numel() <= 1:
                        reason.append("degenerate_target")
            if reason:
                failures.append({"id": row_id, "reasons": ",".join(sorted(set(reason)))})
            for key in totals:
                totals[key] += int(sample["supervision_counts"][key])
            samples.append(sample)
    content_eos = totals["content"] + totals["eos"]
    weighted_pad = 0.5 * totals["prefix_pad"]
    report = {
        "schema_version": CACHE_FORMAT,
        "accepted_rows": len(accepted_ids),
        "cache_rows": len(samples),
        "missing_ids": sorted(accepted_ids - seen),
        "unexpected_ids": sorted(seen - accepted_ids),
        "duplicate_ids": sorted(duplicate_ids),
        "invalid_rows": failures,
        "supervision_totals": {
            **totals,
            "batch_pad": "schedule_dependent_unreported",
            "effective_loss_mass": {
                "prefix_pad_weight_1.0": {
                    "pad": totals["prefix_pad"],
                    "content_plus_eos": content_eos,
                    "pad_fraction": totals["prefix_pad"] / (totals["prefix_pad"] + content_eos),
                },
                "prefix_pad_weight_0.5": {
                    "pad": weighted_pad,
                    "content_plus_eos": content_eos,
                    "pad_fraction": weighted_pad / (weighted_pad + content_eos),
                },
            },
        },
    }
    report["complete"] = (
        not any(
            (
                report["missing_ids"],
                report["unexpected_ids"],
                report["duplicate_ids"],
                report["invalid_rows"],
            )
        )
        and report["cache_rows"] == report["accepted_rows"]
    )
    return report


def write_indexes(torch: Any, out_root: Path) -> None:
    fields = [
        "id",
        "split",
        "speaker_id",
        "gender",
        "stratum",
        "shard",
        "frames",
        "source_frames",
        "target_frames",
        "text_tokens",
        "target_delay_s",
        "target_delay_frames",
        "source_manifest_sha256",
    ]
    for split in ("train", "dev"):
        rows = []
        for path in sorted((out_root / split).glob("shard_*.pt")):
            payload = torch.load(path, map_location="cpu")
            for sample in payload["samples"]:
                rows.append({key: sample.get(key, "") for key in fields} | {"shard": path.name})
        output = [",".join(fields)]
        for row in rows:
            output.append(",".join(str(row[key]) for key in fields))
        atomic_write(out_root / split / "index.csv", ("\n".join(output) + "\n").encode())


def main() -> None:
    args = parse_args()
    if args.device != "mps" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("VIVOS cache construction requires native PyTorch Mimi on MPS")
    if args.shard_size <= 0 or args.target_delay_ratio != 0.5:
        raise RuntimeError("Full VIVOS cache requires shard-size > 0 and delay ratio 0.5")
    accepted, inputs = load_inputs(args)
    genders, gender_files = load_genders(args.gender_files)
    dataset_root = args.dataset_root.expanduser().resolve()
    config_path = require_file(args.config_path, "Hibiki config")
    mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    tokenizer_path = require_file(args.tokenizer, "text tokenizer")
    cfg = read_json(config_path)
    source_rows_path = Path(str(accepted[0]["source_audit"]["row_metrics"]["path"]))
    source_rows_record = accepted[0]["source_audit"]["row_metrics"]
    if attestation(source_rows_path) != source_rows_record:
        raise RuntimeError("Frozen source-audit row metrics changed")
    source_rows = {str(row["id"]): row for row in read_jsonl(source_rows_path)}
    config = {
        "schema_version": CACHE_FORMAT,
        "repository_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": inputs,
        "gender_files": gender_files,
        "weights": {
            "config": attestation(config_path),
            "mimi": attestation(mimi_weight),
            "tokenizer": attestation(tokenizer_path),
        },
        "alignment": {
            "schema_version": ALIGNMENT_SCHEMA,
            "seed": args.alignment_seed,
            "target_delay_ratio": 0.5,
            "distribution": "U(0, 0.5 * source_duration_s)",
            "punctuation_pauses": "not_applicable_single_sentence",
        },
        "scope": {
            "rows": len(accepted),
            "splits": {
                split: sum(row["eligibility_split"] == split for row in accepted)
                for split in ("train", "dev")
            },
            "test_sealed": True,
        },
        "shard_size": args.shard_size,
    }
    out_root = args.out_root.expanduser().resolve()
    config_path_out = out_root / "cache_config.json"
    config_value = json_bytes(config)
    config_sha = sha256_bytes(config_value)
    immutable_write(config_path_out, config_value)
    torch, sphn, sentencepiece, loaders = require_runtime_deps()
    device = check_device(torch, args.device)
    num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
    mimi = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks, device=device)
    if int(cfg["card"]) != int(mimi.cardinality):
        raise RuntimeError("Mimi cardinality does not match Hibiki config")
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
    runtime = {
        "backend": "pytorch_mimi",
        "device": "mps",
        "torch": package_version("torch"),
        "moshi": package_version("moshi"),
        "sphn": package_version("sphn"),
        "sentencepiece": package_version("sentencepiece"),
    }
    for split in ("train", "dev"):
        split_rows = [row for row in accepted if row["eligibility_split"] == split]
        for shard_index in range(math.ceil(len(split_rows) / args.shard_size)):
            chunk = split_rows[shard_index * args.shard_size : (shard_index + 1) * args.shard_size]
            path = shard_path(out_root, split, shard_index)
            expected_ids = [str(row["id"]) for row in chunk]
            if path.exists():
                validate_shard(torch, path, config_sha, expected_ids)
                continue
            samples = []
            for row in chunk:
                row_id = str(row["id"])
                source_path = (
                    dataset_root / row["source_audio"]["dataset_relative_path"]
                ).resolve()
                target_path = Path(str(row["target_audio"]["path"])).resolve()
                if (
                    not source_path.is_relative_to(dataset_root)
                    or sha256_file(source_path) != row["source_audio"]["sha256"]
                    or sha256_file(target_path) != row["target_audio"]["sha256"]
                ):
                    raise RuntimeError(f"Audio provenance changed: {row_id}")
                alignment_row = {
                    "id": row_id,
                    "split": split,
                    "vi_duration_s": str(row["source_audio"]["duration_s"]),
                }
                delay_s = target_delay_s(alignment_row, 0.5, args.alignment_seed)
                delay_frames = int(round(delay_s * FRAME_RATE))
                source_codes = encode_audio(source_path, mimi, sphn, torch, device)
                target_codes = encode_audio(
                    target_path, mimi, sphn, torch, device, left_pad_s=delay_s
                )
                tokens = text_tokens(str(row["text_en"]), tokenizer)
                codes = assemble_codes(
                    torch, alignment_row, source_codes, target_codes, tokens, cfg, delay_frames
                )
                audit = source_rows[row_id]
                if (
                    sha256_bytes(canonical_json(audit).encode())
                    != row["source_audit"]["row_sha256"]
                    or row["source_audit"]["row_metrics"] != source_rows_record
                ):
                    raise RuntimeError(f"Source-audit row provenance changed: {row_id}")
                gender = genders.get(str(row["speaker_id"]))
                if gender is None:
                    raise RuntimeError(f"Missing gender for {row['speaker_id']}")
                translation = row["source_provenance"]["translation"]
                generation_sidecar = row["target_audio"]["generation_sidecar"]
                sidecar_path = Path(str(generation_sidecar["path"])).resolve()
                if sha256_file(sidecar_path) != generation_sidecar["sha256"]:
                    raise RuntimeError(f"Generation sidecar changed: {row_id}")
                generated = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if (
                    generated.get("id") != row_id
                    or generated.get("attempt") != row["target_audio"]["attempt"]
                    or generated.get("seed") != row["target_audio"]["seed"]
                    or generated.get("synthesis") != row["target_audio"]["synthesis"]
                    or generated.get("audio_sha256") != row["target_audio"]["sha256"]
                ):
                    raise RuntimeError(f"Generation sidecar provenance mismatch: {row_id}")
                sample = {
                    "id": row_id,
                    "split": split,
                    "speaker_id": row["speaker_id"],
                    "gender": gender,
                    "stratum": STRATUM,
                    "codes": codes,
                    "frames": int(codes.shape[1]),
                    "vi_frames": int(source_codes.shape[1]),
                    "en_frames": int(target_codes.shape[1]),
                    "source_frames": int(source_codes.shape[1]),
                    "target_frames": int(target_codes.shape[1]),
                    "text_tokens": len(tokens),
                    "target_delay_s": delay_s,
                    "target_delay_frames": delay_frames,
                    "source_manifest_sha256": row["source_provenance"]["accepted_manifest"][
                        "sha256"
                    ],
                    "alignment": {
                        **config["alignment"],
                        "source_duration_s": row["source_audio"]["duration_s"],
                        "delay_s": delay_s,
                        "delay_frames": delay_frames,
                    },
                    "supervision_counts": supervision_counts(
                        delay_frames, len(tokens), int(codes.shape[1])
                    ),
                    "source": {
                        "corpus": row["source_provenance"]["corpus"],
                        "corpus_revision": row["source_provenance"]["corpus_revision"],
                        "license": row["source_provenance"]["license"],
                        "source_repo": row["source_provenance"]["source_repo"],
                        "source_archive_sha256": row["source_provenance"]["source_archive_sha256"],
                        "source_file": row["source_provenance"]["source_file"],
                        "accepted_manifest": row["source_provenance"]["accepted_manifest"],
                        "audio": row["source_audio"],
                        "text_vi": row["text_vi"],
                        "text_vi_sha256": row["text_vi_sha256"],
                        "duration_slice": audit["duration_slice"],
                    },
                    "translation": translation,
                    "tts": {
                        "campaign": inputs["plan"],
                        "campaign_config": inputs["campaign_config"],
                        "model": row["target_audio"]["synthesis"],
                        "selected_attempt": row["target_audio"]["attempt"],
                        "seed": row["target_audio"]["seed"],
                        "reference_id": row["reference"]["reference_id"],
                        "reference_audio_sha256": row["reference"]["reference_audio_sha256"],
                        "reference_text_vi_sha256": row["reference"]["reference_text_vi_sha256"],
                        "reference_source_audit_row_sha256": row["reference"][
                            "source_audit_row_sha256"
                        ],
                        "generation_sidecar": {
                            **generation_sidecar,
                        },
                        "model_snapshot": generated["model_snapshot"],
                        "generation_runtime": generated["runtime"],
                        "target_wav": {
                            "path": str(target_path),
                            "sha256": row["target_audio"]["sha256"],
                        },
                        "qa_row_sha256": row["target_qa"]["metric_sha256"],
                    },
                    "mimi": {"weights": config["weights"], "runtime": runtime},
                    "text_en": row["text_en"],
                    "text_en_sha256": row["text_en_sha256"],
                }
                samples.append(sample)
            save_shard(
                torch,
                {
                    "format": CACHE_FORMAT,
                    "sample_rate": SAMPLE_RATE,
                    "frame_rate": FRAME_RATE,
                    "config": {
                        key: int(cfg[key])
                        for key in ("n_q", "dep_q", "card", "text_card", "existing_text_padding_id")
                    },
                    "cache_config_sha256": config_sha,
                    "samples": samples,
                },
                path,
            )
            print(f"Wrote {len(samples)} samples -> {path}", flush=True)
    write_indexes(torch, out_root)
    report = audit_shards(torch, out_root, {str(row["id"]) for row in accepted}, cfg, config_sha)
    report["inputs"] = inputs
    report["cache_config"] = attestation(config_path_out)
    report["shards"] = [
        attestation(path)
        for path in sorted([*out_root.glob("train/shard_*.pt"), *out_root.glob("dev/shard_*.pt")])
    ]
    report["indexes"] = {
        split: attestation(out_root / split / "index.csv") for split in ("train", "dev")
    }
    immutable_write(out_root / "cache_audit.json", json_bytes(report))
    if not report["complete"]:
        raise RuntimeError("VIVOS cache audit failed")
    print(f"Cache audit passed: {report['cache_rows']} rows")


if __name__ == "__main__":
    main()
