"""Audit pinned VIVOS source speech against its Vietnamese transcript."""

from __future__ import annotations

import argparse
import os
import unicodedata
from collections import defaultdict
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any

from qa_vivos_tts import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    SCIPY_VERSION,
    TRANSFORMERS_VERSION,
    acoustic_metrics,
    atomic_write_json,
    read_audio,
    require_package,
    resample,
    transcribe,
    word_error_counts,
)
from synthesize_vivos import atomic_write_jsonl, read_jsonl, sha256_file

SCHEMA = "hibiki_vivos_source_asr_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        help="Audit only the unique target source rows selected by a TTS pilot plan.",
    )
    return parser.parse_args()


def normalize_characters(text: str) -> list[str]:
    normalized = " ".join(unicodedata.normalize("NFC", text).casefold().split())
    return list(normalized)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, start=1):
        current = [index]
        for other_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[other_index] + 1,
                    current[other_index - 1] + 1,
                    previous[other_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def resolve_audio(row: dict[str, Any], dataset_root: Path | None) -> Path:
    planned = Path(str(row["audio_path"])).expanduser()
    if dataset_root is None:
        return planned.resolve()
    parts = planned.parts
    try:
        relative = Path(*parts[parts.index("raw") :])
    except ValueError as error:
        raise RuntimeError(
            f"Cannot relocate source path without a raw/ suffix: {planned}"
        ) from error
    return (dataset_root.expanduser().resolve() / relative).resolve()


def duration_boundaries(rows: list[dict[str, Any]]) -> tuple[float, float]:
    durations = sorted(float(row["duration_s"]) for row in rows)
    if not durations:
        raise RuntimeError("No source rows selected")
    return (
        durations[(len(durations) - 1) // 4],
        durations[(3 * (len(durations) - 1)) // 4],
    )


def duration_slice(duration: float, boundaries: tuple[float, float]) -> str:
    if duration <= boundaries[0]:
        return "short"
    if duration <= boundaries[1]:
        return "medium"
    return "long"


def load_selected_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    manifest_records: list[dict[str, str]] = []
    for source_path in args.manifests:
        path = source_path.expanduser().resolve()
        record = {"path": str(path), "sha256": sha256_file(path)}
        manifest_records.append(record)
        for row in read_jsonl(path):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in rows_by_id:
                raise RuntimeError(f"Empty or duplicate source id: {row_id!r}")
            if not row.get("text_vi") or not row.get("audio_sha256"):
                raise RuntimeError(f"Incomplete source row: {row_id}")
            rows_by_id[row_id] = row
    if args.pilot_plan:
        plan = read_jsonl(args.pilot_plan.expanduser().resolve())
        selected_ids = {str(row["target_id"]) for row in plan}
        missing = selected_ids - set(rows_by_id)
        if missing:
            raise RuntimeError(
                f"Pilot ids missing from source manifests: {sorted(missing)}"
            )
        rows_by_id = {row_id: rows_by_id[row_id] for row_id in selected_ids}
    return [rows_by_id[key] for key in sorted(rows_by_id)], manifest_records


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    words = sum(int(row["asr_reference_words"]) for row in rows)
    word_errors = sum(int(row["asr_word_errors"]) for row in rows)
    characters = sum(int(row["asr_reference_characters"]) for row in rows)
    character_errors = sum(int(row["asr_character_errors"]) for row in rows)
    rms_values = [
        float(row["acoustic"]["rms"])
        for row in rows
        if row["acoustic"]["rms"] is not None
    ]
    return {
        "rows": len(rows),
        "hours": round(sum(float(row["duration_s"]) for row in rows) / 3600, 6),
        "wer": word_errors / words if words else 0.0,
        "cer": character_errors / characters if characters else 0.0,
        "median_row_wer": median(float(row["asr_wer"]) for row in rows),
        "median_row_cer": median(float(row["asr_cer"]) for row in rows),
        "median_rms": median(rms_values) if rms_values else None,
        "non_finite_rows": sum(not row["acoustic"]["finite"] for row in rows),
        "all_zero_rows": sum(not row["acoustic"]["nonzero"] for row in rows),
    }


def audit(args: argparse.Namespace) -> None:
    rows, manifests = load_selected_rows(args)
    out_dir = args.out_dir.expanduser().resolve()
    row_metrics_path = out_dir / "row_metrics.jsonl"
    report_path = out_dir / "audit_report.json"
    boundaries = duration_boundaries(rows)

    require_package("transformers", TRANSFORMERS_VERSION)
    require_package("scipy", SCIPY_VERSION)
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from scipy import signal as scipy_signal
        from transformers import AutoProcessor, WhisperForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Source audit requires torch, transformers, soundfile, scipy, and numpy"
        ) from error
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK; fallback is not allowed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU fallback is implemented")

    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, revision=ASR_MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    runtime = {
        "device": args.device,
        "model_dtype": "torch.float32",
        "attention_implementation": "eager",
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "scipy": package_version("scipy"),
        "soundfile": package_version("soundfile"),
        "numpy": package_version("numpy"),
    }

    completed = read_jsonl(row_metrics_path) if row_metrics_path.exists() else []
    completed_by_id = {str(row.get("id", "")): row for row in completed}
    if len(completed_by_id) != len(completed) or set(completed_by_id) - {
        str(row["id"]) for row in rows
    }:
        raise RuntimeError(f"Invalid resumable row artifact: {row_metrics_path}")
    manifest_hashes = sorted(record["sha256"] for record in manifests)
    for row in completed:
        if (
            row.get("schema_version") != SCHEMA
            or row.get("source_manifest_sha256") not in manifest_hashes
            or row.get("models", {}).get("asr")
            != {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION}
        ):
            raise RuntimeError(f"Completed source provenance mismatch: {row.get('id')}")

    source_manifest_by_id: dict[str, dict[str, str]] = {}
    for record, manifest_path in zip(manifests, args.manifests):
        for row in read_jsonl(manifest_path.expanduser().resolve()):
            source_manifest_by_id[str(row["id"])] = record

    pending = [row for row in rows if row["id"] not in completed_by_id]
    for number, row in enumerate(pending, start=1):
        path = resolve_audio(row, args.dataset_root)
        if not path.is_file() or sha256_file(path) != row["audio_sha256"]:
            raise RuntimeError(f"Missing or hash-mismatched source audio: {path}")
        audio, sample_rate = read_audio(path, sf, np)
        acoustic = acoustic_metrics(audio, sample_rate, np)
        if not acoustic["finite"] or not acoustic["nonzero"]:
            transcript = ""
        else:
            transcript = transcribe(
                resample(audio, sample_rate, 16_000, scipy_signal),
                processor,
                model,
                torch,
                args.device,
                "vi",
            )
        word_errors, reference_words = word_error_counts(row["text_vi"], transcript)
        reference_characters = normalize_characters(str(row["text_vi"]))
        hypothesis_characters = normalize_characters(transcript)
        character_errors = edit_distance(reference_characters, hypothesis_characters)
        duration_s = len(audio) / sample_rate if sample_rate else 0.0
        record = source_manifest_by_id[str(row["id"])]
        metric = {
            "schema_version": SCHEMA,
            "id": row["id"],
            "corpus": row["corpus"],
            "speaker_id": row["speaker_id"],
            "eligibility_split": row["eligibility_split"],
            "duration_slice": duration_slice(float(row["duration_s"]), boundaries),
            "source_manifest": record["path"],
            "source_manifest_sha256": record["sha256"],
            "audio_path": str(path),
            "audio_sha256": row["audio_sha256"],
            "duration_s": round(duration_s, 6),
            "manifest_duration_s": row["duration_s"],
            "duration_delta_s": round(duration_s - float(row["duration_s"]), 6),
            "acoustic": acoustic,
            "reference_text_vi": row["text_vi"],
            "asr_transcript_vi": transcript,
            "asr_word_errors": word_errors,
            "asr_reference_words": reference_words,
            "asr_wer": word_errors / reference_words if reference_words else 0.0,
            "asr_character_errors": character_errors,
            "asr_reference_characters": len(reference_characters),
            "asr_cer": (
                character_errors / len(reference_characters)
                if reference_characters
                else 0.0
            ),
            "models": {
                "asr": {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION},
            },
            "runtime": runtime,
        }
        completed_by_id[str(row["id"])] = metric
        atomic_write_jsonl(
            row_metrics_path, [completed_by_id[key] for key in sorted(completed_by_id)]
        )
        print(f"[{number}/{len(pending)}] {row['id']}", flush=True)

    scored = [completed_by_id[str(row["id"])] for row in rows]
    slices: dict[str, dict[str, Any]] = {}
    for field in ("eligibility_split", "speaker_id", "duration_slice"):
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            groups[str(row[field])].append(row)
        slices[field] = {key: aggregate(group) for key, group in sorted(groups.items())}
    report = {
        "schema_version": SCHEMA,
        "status": "measured_no_acceptance_threshold_frozen",
        "rows": len(scored),
        "speakers": len({row["speaker_id"] for row in scored}),
        "duration_slice_boundaries_s": {
            "short_max": boundaries[0],
            "medium_max": boundaries[1],
        },
        "aggregate": aggregate(scored),
        "slices": slices,
        "models": {"asr": {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION}},
        "runtime": runtime,
        "source_manifests": manifests,
        "pilot_plan": (
            str(args.pilot_plan.expanduser().resolve()) if args.pilot_plan else None
        ),
        "row_metrics": str(row_metrics_path),
    }
    atomic_write_json(report_path, report)
    print(f"Source audit: {report_path}")


if __name__ == "__main__":
    audit(parse_args())
