"""Audit pinned VIVOS source speech against its Vietnamese transcript."""

from __future__ import annotations

import argparse
import json
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
from synthesize_vivos import (
    atomic_write_jsonl,
    immutable_write,
    read_jsonl,
    sha256_file,
)
from synthesize_vivos import (
    SOURCE_AUDIT_SCHEMA,
    canonical_json,
    expected_pilot_sources,
    sha256_bytes,
    translation_record,
    validate_plan,
)

SCHEMA = SOURCE_AUDIT_SCHEMA
FULL_SCHEMA = "hibiki_vivos_source_asr_mps_full_v1"
FULL_OUT_DIR_NAME = "vivos_source_asr_mps_full_v1"
FULL_EXPECTED_MANIFESTS = {
    "d2276dcda8b664ca918dd53d215b11b159da98fd817fec89d7cb3701f6bc92fb": {
        "split": "train",
        "rows": 9_844,
        "speakers": 41,
    },
    "6fae77d42d6580fc0c36754ce284acb26a3be34bc17de4378979a549f727579d": {
        "split": "dev",
        "rows": 1_106,
        "speakers": 5,
    },
}
FULL_EXPECTED_ROWS = 10_950
FULL_EXPECTED_SPEAKERS = 46
SOURCE_WAVEFORM_THRESHOLDS = {
    "clipping_ratio_max": 0.0001,
    "rms_min": 0.0001,
    "silence_ratio_max": 0.50,
    "leading_trailing_silence_max_s": 2.0,
    "duration_delta_abs_max_s": 0.00001,
}
SOURCE_REVIEW_WER_THRESHOLD = 0.50
SOURCE_ASR_CONFIG = {
    "language": "vi",
    "task": "transcribe",
    "max_new_tokens": 256,
    "sampling_rate_hz": 16_000,
    "return_attention_mask": True,
}
PROVENANCE_FIELDS = (
    "corpus",
    "corpus_revision",
    "license",
    "source_repo",
    "source_file",
    "source_archive_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Freeze the exact accepted train+dev 10,950-row full-audit contract.",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        help="Audit the unique target and clone-reference rows selected by a TTS pilot plan.",
    )
    args = parser.parse_args()
    if args.full and args.pilot_plan:
        parser.error("--full and --pilot-plan are mutually exclusive")
    return args


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
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, Any] | None,
]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    manifest_by_id: dict[str, dict[str, str]] = {}
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
            manifest_by_id[row_id] = record
    pilot_context = None
    if args.pilot_plan:
        plan_path = args.pilot_plan.expanduser().resolve()
        plan = read_jsonl(plan_path)
        validate_plan(plan)
        pilot_sources, pilot_coverage = expected_pilot_sources(plan)
        selected_ids = {str(row["id"]) for row in pilot_sources}
        missing = selected_ids - set(rows_by_id)
        if missing:
            raise RuntimeError(
                f"Pilot ids missing from source manifests: {sorted(missing)}"
            )
        expected_by_id = {str(row["id"]): row for row in pilot_sources}
        for row_id in selected_ids:
            row = rows_by_id[row_id]
            expected = expected_by_id[row_id]
            record = manifest_by_id[row_id]
            if (
                row.get("speaker_id") != expected["speaker_id"]
                or row.get("audio_sha256") != expected["audio_sha256"]
                or sha256_bytes(str(row["text_vi"]).encode("utf-8"))
                != expected["text_vi_sha256"]
                or record["sha256"] != expected["accepted_manifest_sha256"]
            ):
                raise RuntimeError(f"Pilot source identity mismatch: {row_id}")
        rows_by_id = {row_id: rows_by_id[row_id] for row_id in selected_ids}
        manifest_by_id = {row_id: manifest_by_id[row_id] for row_id in selected_ids}
        pilot_context = {
            "pilot_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
                "schema_version": plan[0]["schema_version"],
            },
            "pilot_coverage": pilot_coverage,
            "pilot_sources": pilot_sources,
        }
    if getattr(args, "full", False):
        records_by_sha = {record["sha256"]: record for record in manifest_records}
        if set(records_by_sha) != set(FULL_EXPECTED_MANIFESTS):
            raise RuntimeError(
                "Full audit requires the exact accepted VIVOS train+dev manifests: "
                f"found={sorted(records_by_sha)}"
            )
        for digest, expected in FULL_EXPECTED_MANIFESTS.items():
            ids = [
                row_id
                for row_id, record in manifest_by_id.items()
                if record["sha256"] == digest
            ]
            selected = [rows_by_id[row_id] for row_id in ids]
            if (
                len(selected) != expected["rows"]
                or {str(row["eligibility_split"]) for row in selected}
                != {expected["split"]}
                or len({str(row["speaker_id"]) for row in selected})
                != expected["speakers"]
            ):
                raise RuntimeError(
                    f"Full audit manifest scope mismatch for {expected['split']}"
                )
            records_by_sha[digest]["eligibility_split"] = expected["split"]
        if (
            len(rows_by_id) != FULL_EXPECTED_ROWS
            or len({str(row["speaker_id"]) for row in rows_by_id.values()})
            != FULL_EXPECTED_SPEAKERS
        ):
            raise RuntimeError("Full audit row/speaker coverage mismatch")
        speaker_splits: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows_by_id.values():
            speaker_splits[str(row["speaker_id"])].add(str(row["eligibility_split"]))
        if any(len(splits) != 1 for splits in speaker_splits.values()):
            raise RuntimeError("Full audit speakers are not split-contained")
        manifest_records.sort(key=lambda record: record["eligibility_split"])
    return (
        [rows_by_id[key] for key in sorted(rows_by_id)],
        manifest_records,
        manifest_by_id,
        pilot_context,
    )


def source_provenance(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in PROVENANCE_FIELDS if not row.get(field)]
    if missing:
        raise RuntimeError(f"Incomplete source provenance for {row['id']}: {missing}")
    return {
        **{field: row[field] for field in PROVENANCE_FIELDS},
        "translation": translation_record(row),
    }


def unreadable_acoustic() -> dict[str, Any]:
    return {
        "finite": False,
        "nonzero": False,
        "peak": None,
        "rms": None,
        "dc_offset": None,
        "clipping_ratio": None,
        "silence_ratio": None,
        "leading_silence_s": None,
        "trailing_silence_s": None,
    }


def source_hard_gate(metric: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    acoustic = metric["acoustic"]
    if not metric["readable"]:
        failures.append("unreadable")
    if not acoustic["finite"]:
        failures.append("non_finite")
    if not acoustic["nonzero"]:
        failures.append("all_zero")
    checks = (
        (
            "clipping_ratio",
            acoustic["clipping_ratio"],
            "<=",
            SOURCE_WAVEFORM_THRESHOLDS["clipping_ratio_max"],
        ),
        (
            "rms_below_min",
            acoustic["rms"],
            ">=",
            SOURCE_WAVEFORM_THRESHOLDS["rms_min"],
        ),
        (
            "silence_ratio",
            acoustic["silence_ratio"],
            "<=",
            SOURCE_WAVEFORM_THRESHOLDS["silence_ratio_max"],
        ),
        (
            "leading_silence",
            acoustic["leading_silence_s"],
            "<=",
            SOURCE_WAVEFORM_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "trailing_silence",
            acoustic["trailing_silence_s"],
            "<=",
            SOURCE_WAVEFORM_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "duration_disagreement",
            (
                abs(metric["duration_delta_s"])
                if metric["duration_delta_s"] is not None
                else None
            ),
            "<=",
            SOURCE_WAVEFORM_THRESHOLDS["duration_delta_abs_max_s"],
        ),
    )
    for name, value, operator, threshold in checks:
        if (
            value is None
            or (operator == "<=" and value > threshold)
            or (operator == ">=" and value < threshold)
        ):
            failures.append(name)
    return failures


def immutable_write_json(path: Path, value: object) -> None:
    immutable_write(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


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
        "hours": round(
            sum(
                float(row.get("manifest_duration_s", row["duration_s"])) for row in rows
            )
            / 3600,
            6,
        ),
        "wer": word_errors / words if words else 0.0,
        "cer": character_errors / characters if characters else 0.0,
        "median_row_wer": median(float(row["asr_wer"]) for row in rows),
        "median_row_cer": median(float(row["asr_cer"]) for row in rows),
        "median_rms": median(rms_values) if rms_values else None,
        "non_finite_rows": sum(not row["acoustic"]["finite"] for row in rows),
        "all_zero_rows": sum(not row["acoustic"]["nonzero"] for row in rows),
        "hard_gate_failed_rows": sum(
            bool(row.get("hard_failure_reasons")) for row in rows
        ),
        "source_review_flagged_rows": sum(
            bool(row.get("source_review_required")) for row in rows
        ),
    }


def audit(args: argparse.Namespace) -> None:
    rows, manifests, source_manifest_by_id, pilot_context = load_selected_rows(args)
    out_dir = args.out_dir.expanduser().resolve()
    full = bool(args.full)
    schema = FULL_SCHEMA if full else SCHEMA
    if full and out_dir.name != FULL_OUT_DIR_NAME:
        raise RuntimeError(
            f"Full audit output directory must end in {FULL_OUT_DIR_NAME}: {out_dir}"
        )
    row_metrics_path = out_dir / "row_metrics.jsonl"
    report_path = out_dir / "audit_report.json"
    resume_rows_dir = out_dir / "rows"
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
    if full:
        runtime["audit_script_sha256"] = sha256_file(Path(__file__).resolve())

    if row_metrics_path.exists():
        completed = read_jsonl(row_metrics_path)
    elif full and resume_rows_dir.exists():
        completed = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(resume_rows_dir.glob("*.json"))
        ]
    else:
        completed = []
    completed_by_id = {str(row.get("id", "")): row for row in completed}
    if len(completed_by_id) != len(completed) or set(completed_by_id) - {
        str(row["id"]) for row in rows
    }:
        raise RuntimeError(f"Invalid resumable row artifact: {row_metrics_path}")
    pilot_source_by_id = {
        str(row["id"]): row for row in (pilot_context or {}).get("pilot_sources", [])
    }
    source_by_id = {str(row["id"]): row for row in rows}
    for row in completed:
        row_id = str(row.get("id", ""))
        source = source_by_id[row_id]
        source_record = source_manifest_by_id[row_id]
        provenance = source_provenance(source)
        if (
            row.get("schema_version") != schema
            or row.get("source_manifest_sha256") != source_record["sha256"]
            or row.get("audio_sha256") != source["audio_sha256"]
            or row.get("reference_text_vi_sha256")
            != sha256_bytes(str(source["text_vi"]).encode("utf-8"))
            or row.get("source_provenance") != provenance
            or row.get("source_provenance_sha256")
            != sha256_bytes(canonical_json(provenance).encode("utf-8"))
            or row.get("pilot_roles")
            != pilot_source_by_id.get(row_id, {}).get("pilot_roles", [])
            or row.get("pilot_plan_sha256")
            != (pilot_context or {}).get("pilot_plan", {}).get("sha256")
            or row.get("models", {}).get("asr")
            != {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION}
            or row.get("runtime") != runtime
            or (
                full
                and (
                    row.get("asr_config") != SOURCE_ASR_CONFIG
                    or row.get("hard_gate_thresholds") != SOURCE_WAVEFORM_THRESHOLDS
                    or row.get("hard_failure_reasons") != source_hard_gate(row)
                    or row.get("source_review_required")
                    != (float(row["asr_wer"]) > SOURCE_REVIEW_WER_THRESHOLD)
                )
            )
        ):
            raise RuntimeError(f"Completed source provenance mismatch: {row.get('id')}")

    pending = [row for row in rows if row["id"] not in completed_by_id]
    for number, row in enumerate(pending, start=1):
        path = resolve_audio(row, args.dataset_root)
        if not path.is_file() or sha256_file(path) != row["audio_sha256"]:
            raise RuntimeError(f"Missing or hash-mismatched source audio: {path}")
        try:
            audio, sample_rate = read_audio(path, sf, np)
            readable = True
            read_error = None
            acoustic = acoustic_metrics(audio, sample_rate, np)
            duration_s = len(audio) / sample_rate if sample_rate else 0.0
            duration_delta_s = round(duration_s - float(row["duration_s"]), 6)
        except Exception as error:
            audio = None
            sample_rate = None
            readable = False
            read_error = {"type": type(error).__name__, "message": str(error)}
            acoustic = unreadable_acoustic()
            duration_s = None
            duration_delta_s = None
        if not readable or not acoustic["finite"] or not acoustic["nonzero"]:
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
        record = source_manifest_by_id[str(row["id"])]
        provenance = source_provenance(row)
        metric = {
            "schema_version": schema,
            "id": row["id"],
            "corpus": row["corpus"],
            "speaker_id": row["speaker_id"],
            "eligibility_split": row["eligibility_split"],
            "duration_slice": duration_slice(float(row["duration_s"]), boundaries),
            "source_manifest": record["path"],
            "source_manifest_sha256": record["sha256"],
            "pilot_roles": pilot_source_by_id.get(str(row["id"]), {}).get(
                "pilot_roles", []
            ),
            "pilot_plan_sha256": (pilot_context or {})
            .get("pilot_plan", {})
            .get("sha256"),
            "audio_path": str(path),
            "audio_sha256": row["audio_sha256"],
            "reference_text_vi_sha256": sha256_bytes(
                str(row["text_vi"]).encode("utf-8")
            ),
            "source_provenance": provenance,
            "source_provenance_sha256": sha256_bytes(
                canonical_json(provenance).encode("utf-8")
            ),
            "readable": readable,
            "read_error": read_error,
            "sample_rate_hz": sample_rate,
            "duration_s": round(duration_s, 6) if duration_s is not None else None,
            "manifest_duration_s": row["duration_s"],
            "duration_delta_s": duration_delta_s,
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
        if full:
            metric["asr_config"] = SOURCE_ASR_CONFIG
            metric["hard_gate_thresholds"] = SOURCE_WAVEFORM_THRESHOLDS
            metric["hard_failure_reasons"] = source_hard_gate(metric)
            metric["hard_gate_pass"] = not metric["hard_failure_reasons"]
            metric["source_review_required"] = (
                metric["asr_wer"] > SOURCE_REVIEW_WER_THRESHOLD
            )
            metric["source_review_reason"] = (
                "wer_above_0.50" if metric["source_review_required"] else None
            )
        completed_by_id[str(row["id"])] = metric
        if full:
            sidecar = resume_rows_dir / f"{sha256_bytes(str(row['id']).encode())}.json"
            atomic_write_json(sidecar, metric)
        else:
            atomic_write_jsonl(
                row_metrics_path,
                [completed_by_id[key] for key in sorted(completed_by_id)],
            )
        print(f"[{number}/{len(pending)}] {row['id']}", flush=True)

    scored = [completed_by_id[str(row["id"])] for row in rows]
    if full:
        immutable_write(
            row_metrics_path,
            "".join(canonical_json(row) + "\n" for row in scored).encode("utf-8"),
        )
    slices: dict[str, dict[str, Any]] = {}
    for field in ("eligibility_split", "speaker_id", "duration_slice"):
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            groups[str(row[field])].append(row)
        slices[field] = {key: aggregate(group) for key, group in sorted(groups.items())}
    report = {
        "schema_version": schema,
        "status": (
            "complete_with_source_review_flags"
            if full and any(row["source_review_required"] for row in scored)
            else "complete" if full else "measured_no_acceptance_threshold_frozen"
        ),
        "complete": len(scored) == len(rows),
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
        "pilot_plan": (pilot_context or {}).get("pilot_plan"),
        "pilot_coverage": (pilot_context or {}).get("pilot_coverage"),
        "pilot_sources": (pilot_context or {}).get("pilot_sources"),
        "row_metrics": {
            "path": str(row_metrics_path),
            "sha256": sha256_file(row_metrics_path),
        },
    }
    if full:
        report.update(
            {
                "audit_scope": {
                    "name": "accepted_vivos_train_dev_full_v1",
                    "expected_rows": FULL_EXPECTED_ROWS,
                    "expected_speakers": FULL_EXPECTED_SPEAKERS,
                    "split_contained_speakers": True,
                    "output_directory_name": FULL_OUT_DIR_NAME,
                },
                "hard_gate_thresholds": SOURCE_WAVEFORM_THRESHOLDS,
                "asr_config": SOURCE_ASR_CONFIG,
                "hard_gate_failed_ids": [
                    str(row["id"]) for row in scored if row["hard_failure_reasons"]
                ],
                "source_review_policy": {
                    "rule": "asr_wer > 0.50 is a manual-review flag, not an automatic rejection",
                    "wer_threshold_exclusive": SOURCE_REVIEW_WER_THRESHOLD,
                    "flagged_ids": [
                        str(row["id"])
                        for row in scored
                        if row["source_review_required"]
                    ],
                },
                "resume_rows_directory": str(resume_rows_dir),
            }
        )
        immutable_write_json(report_path, report)
    else:
        atomic_write_json(report_path, report)
    print(f"Source audit: {report_path}")


if __name__ == "__main__":
    audit(parse_args())
