"""Score and finalize the immutable full VIVOS MLX target campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any

from qa_vivos_tts import (
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    SCIPY_VERSION,
    SPEAKER_MODEL_ID,
    SPEAKER_MODEL_REVISION,
    TRANSFORMERS_VERSION,
    acoustic_metrics,
    prompt_leak_evidence,
    read_audio,
    require_package,
    resample,
    transcribe,
    word_error_counts,
)
from synthesize_vivos import (
    atomic_write_bytes,
    canonical_json,
    immutable_write,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from synthesize_vivos_full import (
    APPROVAL_SCHEMA,
    ATTEMPT_SCHEMA,
    EXPECTED_ROWS,
    EXPECTED_SPEAKERS,
    MLX_MODEL_FILES_SHA256,
    SCHEMA as CAMPAIGN_SCHEMA,
    SOURCE_AUDIT_SCHEMA,
    SYNTHESIS,
    load_campaign,
    output_path,
    sidecar_path,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_qa_v1"
MANUAL_REVIEW_SEED = "hibiki-vivos-qwen3-tts-mlx-full-v1-manual-v1"
MANUAL_REVIEW_SAMPLE_SIZE = 120
ROW_THRESHOLDS = {
    "clipping_ratio_max": 0.0001,
    "silence_ratio_max": 0.50,
    "leading_trailing_silence_max_s": 2.0,
    "rms_min": 0.0001,
    "duration_ratio_min": 0.40,
    "duration_ratio_max": 1.80,
    "asr_wer_max": 0.50,
    "reference_only_3gram_matches_max": 0,
    "speaker_cosine_min": 0.85,
}
CORPUS_THRESHOLDS = {
    "selected_asr_wer_max": 0.08,
    "selected_speaker_cosine_median_min": 0.90,
    "accepted_prompt_leak_matches_max": 0,
    "manual_review_sample_size": MANUAL_REVIEW_SAMPLE_SIZE,
    "manual_review_seed": MANUAL_REVIEW_SEED,
}
MODELS = {
    "asr": {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION},
    "speaker": {"id": SPEAKER_MODEL_ID, "revision": SPEAKER_MODEL_REVISION},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    score = commands.add_parser("score-attempt")
    final = commands.add_parser("finalize")
    for command in (score, final):
        command.add_argument("plan", type=Path)
        command.add_argument("--campaign-config", type=Path, required=True)
        command.add_argument("--approval", type=Path, required=True)
        command.add_argument("--source-audit-report", type=Path, required=True)
        command.add_argument("--reference-map", type=Path, required=True)
        command.add_argument("--reference-report", type=Path, required=True)
        command.add_argument("--generation-manifest", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--attempt", type=int, choices=(0, 1), required=True)
    score.add_argument("--retry-ids", type=Path)
    score.add_argument("--device", default="mps")
    final.add_argument("--attempt0-metrics", type=Path, required=True)
    final.add_argument("--retry-ids", type=Path, required=True)
    final.add_argument("--attempt1-metrics", type=Path)
    final.add_argument(
        "--manual-review",
        type=Path,
        help="TSV columns: candidate_id, status (pass/fail), prompt_leak (yes/no), notes.",
    )
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode()


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, json_bytes(value))


def attestation(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def require_exact_artifact(path: Path, expected: dict[str, str], label: str) -> Path:
    resolved = path.expanduser().resolve()
    if str(resolved) != str(Path(expected["path"]).expanduser().resolve()):
        raise RuntimeError(f"{label} path is not the frozen campaign artifact: {resolved}")
    if sha256_file(resolved) != expected["sha256"]:
        raise RuntimeError(f"{label} hash changed: {resolved}")
    return resolved


def load_inputs(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], str, str, dict[str, Any]]:
    plan_path = args.plan.expanduser().resolve()
    rows, config, plan_sha, config_sha = load_campaign(plan_path)
    config_path = args.campaign_config.expanduser().resolve()
    if (
        config_path != plan_path.parent / "campaign_config.json"
        or sha256_file(config_path) != config_sha
    ):
        raise RuntimeError("--campaign-config is not the exact config bound to the plan")
    approval_path = require_exact_artifact(args.approval, config["approval"], "Approval")
    source_report_path = require_exact_artifact(
        args.source_audit_report, config["source_audit"]["report"], "Source audit report"
    )
    reference_map_path = require_exact_artifact(
        args.reference_map, config["references"]["map"], "Reference map"
    )
    reference_report_path = require_exact_artifact(
        args.reference_report, config["references"]["report"], "Reference report"
    )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if (
        approval.get("schema_version") != APPROVAL_SCHEMA
        or approval.get("campaign_schema_version") != CAMPAIGN_SCHEMA
    ):
        raise RuntimeError("Approval schema does not authorize this campaign")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_rows_path = require_exact_artifact(
        Path(str(source_report.get("row_metrics", {}).get("path", ""))),
        config["source_audit"]["row_metrics"],
        "Source audit row metrics",
    )
    source_rows = read_jsonl(source_rows_path)
    source_by_id = {str(row.get("id", "")): row for row in source_rows}
    references = read_jsonl(reference_map_path)
    reference_by_speaker = {str(row.get("speaker_id", "")): row for row in references}
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    if (
        source_report.get("schema_version") != SOURCE_AUDIT_SCHEMA
        or source_report.get("rows") != EXPECTED_ROWS
        or source_report.get("speakers") != EXPECTED_SPEAKERS
        or len(source_by_id) != EXPECTED_ROWS
        or len(reference_by_speaker) != EXPECTED_SPEAKERS
        or reference_report.get("references") != references
    ):
        raise RuntimeError("Source-audit or reference-map scope changed")
    for row in rows:
        row_id = str(row["id"])
        source = source_by_id.get(row_id)
        if source is None:
            raise RuntimeError(f"Source audit is missing {row_id}")
        if (
            sha256_bytes(canonical_json(source).encode()) != row["source_audit"]["row_sha256"]
            or source.get("audio_sha256") != row["source_audio"]["sha256"]
            or source.get("reference_text_vi_sha256") != row["text_vi_sha256"]
            or source.get("eligibility_split") != row["eligibility_split"]
        ):
            raise RuntimeError(f"Source audit provenance mismatch: {row_id}")
        frozen_reference = reference_by_speaker.get(str(row["speaker_id"]))
        planned_reference = dict(row["reference"])
        planned_reference.pop("reference_audio_dataset_relative_path", None)
        if frozen_reference is None or planned_reference != frozen_reference:
            raise RuntimeError(f"Reference-map provenance mismatch: {row_id}")
    artifacts = {
        "campaign_config": attestation(config_path),
        "approval": attestation(approval_path),
        "source_audit_report": attestation(source_report_path),
        "source_audit_rows": attestation(source_rows_path),
        "reference_map": attestation(reference_map_path),
        "reference_report": attestation(reference_report_path),
    }
    return plan_path, rows, config, plan_sha, config_sha, artifacts


def load_retry_ids(
    path: Path | None, attempt: int, plan_ids: set[str]
) -> tuple[list[str], dict[str, str] | None]:
    if attempt == 0:
        if path is not None:
            raise RuntimeError("--retry-ids is only valid for attempt 1")
        return sorted(plan_ids), None
    if path is None:
        raise RuntimeError("Attempt 1 requires --retry-ids")
    resolved = path.expanduser().resolve()
    rows = read_jsonl(resolved)
    ids = [str(row.get("id", "")) for row in rows]
    if (
        not ids
        or len(ids) != len(set(ids))
        or set(ids) - plan_ids
        or any(set(row) != {"id"} for row in rows)
    ):
        raise RuntimeError("Retry manifest must contain unique in-plan {id} rows")
    return ids, attestation(resolved)


def validate_generation_sidecar(
    path: Path,
    row: dict[str, Any],
    attempt: int,
    plan_path: Path,
    plan_sha: str,
    config_sha: str,
    retry: dict[str, str] | None,
) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    output = output_path(plan_path, row, attempt)
    seed_key = "attempt0" if attempt == 0 else "attempt1_retry_1"
    if (
        path != sidecar_path(plan_path, row, attempt)
        or item.get("schema_version") != ATTEMPT_SCHEMA
        or item.get("id") != row["id"]
        or item.get("speaker_id") != row["speaker_id"]
        or item.get("eligibility_split") != row["eligibility_split"]
        or item.get("attempt") != attempt
        or item.get("seed") != row["seeds"][seed_key]
        or item.get("plan_path") != str(plan_path)
        or item.get("plan_sha256") != plan_sha
        or item.get("config_sha256") != config_sha
        or item.get("retry_ids") != retry
        or item.get("synthesis") != SYNTHESIS
        or item.get("reference") != row["reference"]
        or item.get("source_audit") != row["source_audit"]
        or item.get("output_wav") != str(output)
        or not output.is_file()
        or sha256_file(output) != item.get("audio_sha256")
        or item.get("model_snapshot", {}).get("files_sha256") != MLX_MODEL_FILES_SHA256
    ):
        raise RuntimeError(f"Generation sidecar provenance mismatch: {path}")
    return item


def discover_attempts(
    plan_path: Path,
    rows: list[dict[str, Any]],
    attempt: int,
    selected_ids: set[str],
    plan_sha: str,
    config_sha: str,
    retry: dict[str, str] | None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    by_id = {str(row["id"]): row for row in rows}
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    root = plan_path.parent / "attempts" / f"attempt{attempt}"
    for path in sorted(root.glob("*/*/*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        row_id = str(item.get("id", ""))
        row = by_id.get(row_id)
        if row is None or row_id not in selected_ids or row_id in found:
            raise RuntimeError(f"Unexpected, unselected, or duplicate attempt sidecar: {path}")
        found[row_id] = (
            path,
            validate_generation_sidecar(path, row, attempt, plan_path, plan_sha, config_sha, retry),
        )
    return found


def validate_generation_manifest(
    path: Path, plan_path: Path, rows: list[dict[str, Any]]
) -> tuple[Path, dict[tuple[str, int], dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    if resolved != plan_path.parent / "generation_attempts.jsonl":
        raise RuntimeError("--generation-manifest must be the campaign assembly manifest")
    plan_by_id = {str(row["id"]): row for row in rows}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for item in read_jsonl(resolved):
        key = (str(item.get("id", "")), int(item.get("attempt", -1)))
        row = plan_by_id.get(key[0])
        if row is None or key[1] not in (0, 1) or key in found:
            raise RuntimeError(f"Unexpected or duplicate generation manifest row: {key}")
        source_path = sidecar_path(plan_path, row, key[1])
        if not source_path.is_file() or json.loads(source_path.read_text(encoding="utf-8")) != item:
            raise RuntimeError(f"Generation manifest differs from sidecar: {key}")
        found[key] = item
    return resolved, found


def resolve_audio(
    record: dict[str, Any],
    relative_key: str,
    hash_key: str,
    root: Path,
    hashes: dict[Path, str],
) -> Path:
    relative = Path(str(record[relative_key]))
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Frozen input audio is unavailable: {path}")
    if path not in hashes:
        hashes[path] = sha256_file(path)
    actual_hash = hashes[path]
    if actual_hash != record[hash_key]:
        raise RuntimeError(f"Frozen input audio changed: {path}")
    return path


def metric_path(out_dir: Path, row: dict[str, Any], attempt: int) -> Path:
    safe_id = str(row["id"]).replace(":", "_")
    return (
        out_dir
        / f"attempt{attempt}"
        / "rows"
        / row["eligibility_split"]
        / row["speaker_id"]
        / f"{safe_id}.json"
    )


def row_failures(metric: dict[str, Any]) -> list[str]:
    acoustic = metric["acoustic"]
    failures: list[str] = []
    if not acoustic.get("readable"):
        failures.append("unreadable")
    if not acoustic.get("finite"):
        failures.append("non_finite")
    if not acoustic.get("nonzero"):
        failures.append("all_zero")
    checks = (
        ("rms_below_min", acoustic.get("rms"), ">=", ROW_THRESHOLDS["rms_min"]),
        (
            "clipping_ratio",
            acoustic.get("clipping_ratio"),
            "<=",
            ROW_THRESHOLDS["clipping_ratio_max"],
        ),
        ("silence_ratio", acoustic.get("silence_ratio"), "<=", ROW_THRESHOLDS["silence_ratio_max"]),
        (
            "leading_silence",
            acoustic.get("leading_silence_s"),
            "<=",
            ROW_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "trailing_silence",
            acoustic.get("trailing_silence_s"),
            "<=",
            ROW_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "duration_ratio_below_min",
            metric.get("duration_ratio_target_source"),
            ">=",
            ROW_THRESHOLDS["duration_ratio_min"],
        ),
        (
            "duration_ratio_above_max",
            metric.get("duration_ratio_target_source"),
            "<=",
            ROW_THRESHOLDS["duration_ratio_max"],
        ),
        ("asr_wer", metric.get("asr_wer"), "<=", ROW_THRESHOLDS["asr_wer_max"]),
        (
            "prompt_leak",
            metric.get("prompt_leak", {}).get("reference_only_3gram_match_count"),
            "<=",
            ROW_THRESHOLDS["reference_only_3gram_matches_max"],
        ),
        (
            "speaker_cosine",
            metric.get("speaker_cosine"),
            ">=",
            ROW_THRESHOLDS["speaker_cosine_min"],
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


def speaker_embedding(audio_16k: Any, extractor: Any, model: Any, torch: Any, device: str) -> Any:
    inputs = extractor(
        audio_16k,
        sampling_rate=16_000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        embedding = model(**model_inputs).embeddings[0]
    return torch.nn.functional.normalize(embedding.float(), dim=0).cpu()


def load_models(device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if device != "mps":
        raise RuntimeError("Full-campaign scoring is Apple-Metal-only; use --device mps")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK; fallback is not allowed")
    versions = {
        "transformers": require_package("transformers", TRANSFORMERS_VERSION),
        "scipy": require_package("scipy", SCIPY_VERSION),
    }
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from scipy import signal as scipy_signal
        from transformers import (
            AutoFeatureExtractor,
            AutoProcessor,
            WavLMForXVector,
            WhisperForConditionalGeneration,
        )
    except ImportError as error:
        raise RuntimeError(
            "Scoring requires torch, transformers, soundfile, scipy, and numpy"
        ) from error
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no fallback is implemented")
    dtype = torch.float32
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, revision=ASR_MODEL_REVISION)
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    asr_model.eval()
    speaker_features = AutoFeatureExtractor.from_pretrained(
        SPEAKER_MODEL_ID, revision=SPEAKER_MODEL_REVISION
    )
    speaker_model = WavLMForXVector.from_pretrained(
        SPEAKER_MODEL_ID, revision=SPEAKER_MODEL_REVISION, torch_dtype=dtype
    ).to(device)
    speaker_model.eval()
    versions.update(
        {
            "torch": package_version("torch"),
            "soundfile": package_version("soundfile"),
            "numpy": package_version("numpy"),
            "device": device,
            "model_dtype": str(dtype),
            "attention_implementation": "eager",
            "attention_masks": {"whisper": True, "wavlm": True},
        }
    )
    objects = {
        "np": np,
        "sf": sf,
        "torch": torch,
        "scipy_signal": scipy_signal,
        "asr_processor": asr_processor,
        "asr_model": asr_model,
        "speaker_features": speaker_features,
        "speaker_model": speaker_model,
    }
    return objects, versions


def score_row(
    row: dict[str, Any],
    attempt: int,
    generated_path: Path,
    generated: dict[str, Any],
    plan_path: Path,
    plan_sha: str,
    config_sha: str,
    artifacts: dict[str, Any],
    dataset_root: Path,
    objects: dict[str, Any],
    runtime: dict[str, Any],
    reference_embeddings: dict[str, Any],
    input_hashes: dict[Path, str],
) -> dict[str, Any]:
    np = objects["np"]
    sf = objects["sf"]
    torch = objects["torch"]
    scipy_signal = objects["scipy_signal"]
    output = Path(str(generated["output_wav"]))
    reference_path = resolve_audio(
        row["reference"],
        "reference_audio_dataset_relative_path",
        "reference_audio_sha256",
        dataset_root,
        input_hashes,
    )
    resolve_audio(
        row["source_audio"],
        "dataset_relative_path",
        "sha256",
        dataset_root,
        input_hashes,
    )
    duration_s: float | None = None
    ratio: float | None = None
    asr_text: str | None = None
    auto_text: str | None = None
    errors: int | None = None
    reference_words: int | None = None
    wer: float | None = None
    cosine: float | None = None
    read_error: str | None = None
    try:
        audio, sample_rate = read_audio(output, sf, np)
        acoustic = {"readable": True, **acoustic_metrics(audio, sample_rate, np)}
    except Exception as error:
        acoustic = {
            "readable": False,
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
        read_error = f"{type(error).__name__}: {error}"
    if acoustic["readable"]:
        duration_s = len(audio) / sample_rate
        ratio = duration_s / float(row["source_audio"]["duration_s"])
    if acoustic["finite"] and acoustic["nonzero"]:
        audio_16k = resample(audio, sample_rate, 16_000, scipy_signal)
        asr_text = transcribe(
            audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", "en"
        )
        auto_text = transcribe(
            audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", None
        )
        errors, reference_words = word_error_counts(row["text_en"], asr_text)
        wer = errors / reference_words if reference_words else None
        speaker = str(row["speaker_id"])
        if speaker not in reference_embeddings:
            reference_audio, reference_rate = read_audio(reference_path, sf, np)
            reference_16k = resample(reference_audio, reference_rate, 16_000, scipy_signal)
            reference_embeddings[speaker] = speaker_embedding(
                reference_16k, objects["speaker_features"], objects["speaker_model"], torch, "mps"
            )
        output_embedding = speaker_embedding(
            audio_16k, objects["speaker_features"], objects["speaker_model"], torch, "mps"
        )
        cosine = float(torch.dot(reference_embeddings[speaker], output_embedding))
    leak = (
        prompt_leak_evidence(row["reference"]["reference_text_vi"], row["text_en"], auto_text)
        if auto_text is not None
        else {
            "asr_auto_transcript": None,
            "reference_only_3gram_match_count": None,
            "reference_only_3gram_matches": [],
        }
    )
    metric = {
        "schema_version": SCHEMA,
        "id": row["id"],
        "candidate_id": f"{row['id']}|attempt{attempt}",
        "attempt": attempt,
        "speaker_id": row["speaker_id"],
        "eligibility_split": row["eligibility_split"],
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "campaign_config_sha256": config_sha,
        "campaign_artifacts": artifacts,
        "generation_sidecar": attestation(generated_path),
        "output_wav": str(output),
        "audio_sha256": generated["audio_sha256"],
        "source_audio_sha256": row["source_audio"]["sha256"],
        "reference_audio_sha256": row["reference"]["reference_audio_sha256"],
        "text_en_sha256": row["text_en_sha256"],
        "text_vi_sha256": row["text_vi_sha256"],
        "duration_s": round(duration_s, 6) if duration_s is not None else None,
        "duration_ratio_target_source": round(ratio, 6) if ratio is not None else None,
        "acoustic": acoustic,
        "waveform_read_error": read_error,
        "asr_transcript_en": asr_text,
        "asr_word_errors": errors,
        "asr_reference_words": reference_words,
        "asr_wer": wer,
        "prompt_leak": leak,
        "speaker_cosine": cosine,
        "models": MODELS,
        "runtime": runtime,
        "thresholds": ROW_THRESHOLDS,
    }
    metric["failure_reasons"] = row_failures(metric)
    metric["retry_gate_pass"] = not metric["failure_reasons"]
    return metric


def validate_metric(
    metric: dict[str, Any],
    row: dict[str, Any],
    attempt: int,
    generated_path: Path,
    generated: dict[str, Any],
    plan_path: Path,
    plan_sha: str,
    config_sha: str,
    artifacts: dict[str, Any],
) -> None:
    if (
        metric.get("schema_version") != SCHEMA
        or metric.get("id") != row["id"]
        or metric.get("candidate_id") != f"{row['id']}|attempt{attempt}"
        or metric.get("attempt") != attempt
        or metric.get("speaker_id") != row["speaker_id"]
        or metric.get("eligibility_split") != row["eligibility_split"]
        or metric.get("plan") != {"path": str(plan_path), "sha256": plan_sha}
        or metric.get("campaign_config_sha256") != config_sha
        or metric.get("campaign_artifacts") != artifacts
        or metric.get("generation_sidecar") != attestation(generated_path)
        or metric.get("output_wav") != generated["output_wav"]
        or metric.get("audio_sha256") != generated["audio_sha256"]
        or metric.get("source_audio_sha256") != row["source_audio"]["sha256"]
        or metric.get("reference_audio_sha256") != row["reference"]["reference_audio_sha256"]
        or metric.get("text_en_sha256") != row["text_en_sha256"]
        or metric.get("text_vi_sha256") != row["text_vi_sha256"]
        or metric.get("models") != MODELS
        or metric.get("thresholds") != ROW_THRESHOLDS
        or metric.get("runtime", {}).get("device") != "mps"
        or metric.get("runtime", {}).get("model_dtype") != "torch.float32"
        or metric.get("runtime", {}).get("attention_implementation") != "eager"
        or metric.get("runtime", {}).get("attention_masks") != {"whisper": True, "wavlm": True}
        or metric.get("runtime", {}).get("transformers") != TRANSFORMERS_VERSION
        or metric.get("runtime", {}).get("scipy") != SCIPY_VERSION
        or metric.get("failure_reasons") != row_failures(metric)
        or metric.get("retry_gate_pass") != (not metric.get("failure_reasons"))
    ):
        raise RuntimeError(
            f"Resumable QA metric provenance mismatch: {row['id']} attempt {attempt}"
        )


def score_attempt(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("Full-campaign scoring is Apple-Metal-only; use --device mps")
    plan_path, rows, _, plan_sha, config_sha, artifacts = load_inputs(args)
    plan_order = {str(row["id"]): index for index, row in enumerate(rows)}
    selected_order, retry = load_retry_ids(args.retry_ids, args.attempt, set(plan_order))
    selected = set(selected_order)
    generation_path, generation_rows = validate_generation_manifest(
        args.generation_manifest, plan_path, rows
    )
    generated = discover_attempts(
        plan_path, rows, args.attempt, selected, plan_sha, config_sha, retry
    )
    out_dir = args.out_dir.expanduser().resolve()
    by_id = {str(row["id"]): row for row in rows}
    metrics: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for row_id in selected_order:
        candidate = generated.get(row_id)
        if candidate is None:
            continue
        path = metric_path(out_dir, by_id[row_id], args.attempt)
        if path.is_file():
            metric = json.loads(path.read_text(encoding="utf-8"))
            validate_metric(
                metric,
                by_id[row_id],
                args.attempt,
                *candidate,
                plan_path,
                plan_sha,
                config_sha,
                artifacts,
            )
            metrics[row_id] = metric
        else:
            pending.append(row_id)
    objects: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    reference_embeddings: dict[str, Any] = {}
    input_hashes: dict[Path, str] = {}
    dataset_root = args.dataset_root.expanduser().resolve()
    for number, row_id in enumerate(pending, start=1):
        if objects is None or runtime is None:
            objects, runtime = load_models(args.device)
        row = by_id[row_id]
        generated_path, generated_row = generated[row_id]
        metric = score_row(
            row,
            args.attempt,
            generated_path,
            generated_row,
            plan_path,
            plan_sha,
            config_sha,
            artifacts,
            dataset_root,
            objects,
            runtime,
            reference_embeddings,
            input_hashes,
        )
        immutable_write(metric_path(out_dir, row, args.attempt), json_bytes(metric))
        metrics[row_id] = metric
        print(f"[{number}/{len(pending)}] {row_id}: {metric['failure_reasons']}", flush=True)
    complete_sidecars = set(generated) == selected
    manifest_ids = {row_id for row_id, attempt in generation_rows if attempt == args.attempt}
    complete_manifest = manifest_ids == selected
    complete_metrics = set(metrics) == selected
    progress = {
        "schema_version": SCHEMA,
        "attempt": args.attempt,
        "expected_rows": len(selected),
        "generation_sidecars": len(generated),
        "generation_manifest_rows": len(manifest_ids),
        "scored_rows": len(metrics),
        "failed_rows": sum(not metric["retry_gate_pass"] for metric in metrics.values()),
        "complete": complete_sidecars and complete_manifest and complete_metrics,
        "generation_manifest": attestation(generation_path),
        "retry_ids": retry,
    }
    atomic_write_json(out_dir / f"attempt{args.attempt}_progress.json", progress)
    if not progress["complete"]:
        print(f"Attempt {args.attempt} QA progress: {len(metrics)}/{len(selected)} scored")
        return
    ordered_metrics = [metrics[row_id] for row_id in sorted(selected, key=plan_order.get)]
    metrics_path = out_dir / f"attempt{args.attempt}_metrics.jsonl"
    generation_snapshot_path = out_dir / f"attempt{args.attempt}_generation.jsonl"
    immutable_write(metrics_path, jsonl_bytes(ordered_metrics))
    immutable_write(
        generation_snapshot_path,
        jsonl_bytes([generated[str(metric["id"])][1] for metric in ordered_metrics]),
    )
    report = {
        **{key: value for key, value in progress.items() if key != "generation_manifest"},
        "row_metrics": attestation(metrics_path),
        "generation_snapshot": attestation(generation_snapshot_path),
        "thresholds": ROW_THRESHOLDS,
        "models": MODELS,
        "status": "complete",
    }
    immutable_write(out_dir / f"attempt{args.attempt}_report.json", json_bytes(report))
    if args.attempt == 0:
        retry_rows = [
            {"id": metric["id"]} for metric in ordered_metrics if not metric["retry_gate_pass"]
        ]
        immutable_write(out_dir / "retry_ids.jsonl", jsonl_bytes(retry_rows))
        print(f"Attempt 0 complete; {len(retry_rows)} deterministic retries")
    else:
        print(f"Attempt 1 complete; {report['failed_rows']} rows still fail")


def validate_metrics_manifest(
    path: Path,
    attempt: int,
    expected_ids: list[str],
    by_id: dict[str, dict[str, Any]],
    plan_path: Path,
    plan_sha: str,
    config_sha: str,
    artifacts: dict[str, Any],
    retry: dict[str, str] | None,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    resolved = path.expanduser().resolve()
    if resolved.name != f"attempt{attempt}_metrics.jsonl":
        raise RuntimeError(f"Unexpected attempt-{attempt} metrics filename: {resolved}")
    rows = read_jsonl(resolved)
    ids = [str(row.get("id", "")) for row in rows]
    if ids != expected_ids:
        raise RuntimeError(f"Attempt-{attempt} metrics do not match deterministic expected scope")
    output: dict[str, dict[str, Any]] = {}
    for metric in rows:
        row_id = str(metric["id"])
        generation_path = Path(str(metric.get("generation_sidecar", {}).get("path", "")))
        generated = validate_generation_sidecar(
            generation_path,
            by_id[row_id],
            attempt,
            plan_path,
            plan_sha,
            config_sha,
            retry,
        )
        validate_metric(
            metric,
            by_id[row_id],
            attempt,
            generation_path,
            generated,
            plan_path,
            plan_sha,
            config_sha,
            artifacts,
        )
        if sha256_file(Path(str(metric["output_wav"]))) != metric["audio_sha256"]:
            raise RuntimeError(f"Selected candidate audio changed: {row_id} attempt {attempt}")
        output[row_id] = metric
    return resolved, output


def load_manual_reviews(
    path: Path | None, allowed: set[str]
) -> tuple[dict[str, dict[str, str]], dict[str, str] | None]:
    if path is None:
        return {}, None
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"candidate_id", "status", "prompt_leak", "notes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Manual review TSV must contain {sorted(required)}")
        reviews: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            candidate_id = row["candidate_id"].strip()
            status = row["status"].strip().casefold()
            prompt_leak = row["prompt_leak"].strip().casefold()
            if not candidate_id or candidate_id not in allowed or candidate_id in reviews:
                raise RuntimeError(f"Invalid or duplicate candidate_id at {resolved}:{line_number}")
            if not status and not prompt_leak:
                continue
            if status not in {"pass", "fail"} or prompt_leak not in {"yes", "no"}:
                raise RuntimeError(f"Invalid review values at {resolved}:{line_number}")
            reviews[candidate_id] = {
                "status": status,
                "prompt_leak": prompt_leak,
                "notes": row["notes"].strip(),
            }
    return reviews, attestation(resolved)


def manual_sample(candidate_ids: list[str]) -> list[str]:
    return sorted(
        candidate_ids,
        key=lambda candidate_id: sha256_bytes(f"{MANUAL_REVIEW_SEED}\0{candidate_id}".encode()),
    )[: min(MANUAL_REVIEW_SAMPLE_SIZE, len(candidate_ids))]


def write_review_template(
    path: Path,
    required: dict[str, set[str]],
    candidates: dict[str, dict[str, Any]],
) -> None:
    if path.exists():
        return
    lines = [
        "candidate_id\tid\tattempt\tobligation\toutput_wav\taudio_sha256\t"
        "failure_reasons\tstatus\tprompt_leak\tnotes\n"
    ]
    for candidate_id in sorted(required):
        row_id, attempt_text = candidate_id.rsplit("|attempt", 1)
        candidate = candidates[candidate_id]
        obligations = ",".join(sorted(required[candidate_id]))
        failures = ",".join(candidate["failure_reasons"])
        lines.append(
            f"{candidate_id}\t{row_id}\t{attempt_text}\t{obligations}\t"
            f"{candidate['output_wav']}\t{candidate['audio_sha256']}\t{failures}\t\t\t\n"
        )
    immutable_write(path, "".join(lines).encode())


def finalize(args: argparse.Namespace) -> None:
    plan_path, rows, _, plan_sha, config_sha, artifacts = load_inputs(args)
    generation_path, generation_rows = validate_generation_manifest(
        args.generation_manifest, plan_path, rows
    )
    by_id = {str(row["id"]): row for row in rows}
    order = [str(row["id"]) for row in rows]
    attempt0_path, attempt0 = validate_metrics_manifest(
        args.attempt0_metrics,
        0,
        order,
        by_id,
        plan_path,
        plan_sha,
        config_sha,
        artifacts,
        None,
    )
    retry_path = args.retry_ids.expanduser().resolve()
    retry_rows = read_jsonl(retry_path)
    retry_order = [str(row.get("id", "")) for row in retry_rows]
    expected_retry_order = [row_id for row_id in order if not attempt0[row_id]["retry_gate_pass"]]
    if retry_order != expected_retry_order or any(set(row) != {"id"} for row in retry_rows):
        raise RuntimeError("Retry ids are not the deterministic attempt-0 failures")
    if retry_order:
        if args.attempt1_metrics is None:
            raise RuntimeError("Finalization requires complete attempt-1 metrics for every retry")
        attempt1_path, attempt1 = validate_metrics_manifest(
            args.attempt1_metrics,
            1,
            retry_order,
            by_id,
            plan_path,
            plan_sha,
            config_sha,
            artifacts,
            attestation(retry_path),
        )
    else:
        if args.attempt1_metrics is not None:
            raise RuntimeError("Attempt-1 metrics are invalid when the retry set is empty")
        attempt1_path, attempt1 = None, {}
    expected_generation = {(row_id, 0) for row_id in order} | {
        (row_id, 1) for row_id in retry_order
    }
    if set(generation_rows) != expected_generation:
        raise RuntimeError(
            "Generation manifest does not exactly cover attempt 0 plus frozen retries"
        )
    dataset_root = args.dataset_root.expanduser().resolve()
    input_hashes: dict[Path, str] = {}
    for row in rows:
        resolve_audio(
            row["source_audio"],
            "dataset_relative_path",
            "sha256",
            dataset_root,
            input_hashes,
        )
        resolve_audio(
            row["reference"],
            "reference_audio_dataset_relative_path",
            "reference_audio_sha256",
            dataset_root,
            input_hashes,
        )

    selections: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    failed_candidates: set[str] = set()
    for row_id in order:
        row = by_id[row_id]
        candidates = [attempt0[row_id]]
        if row_id in attempt1:
            candidates.append(attempt1[row_id])
        for metric in candidates:
            if not metric["retry_gate_pass"]:
                failed_candidates.add(str(metric["candidate_id"]))
        selected = next((metric for metric in candidates if metric["retry_gate_pass"]), None)
        candidate_summaries = [
            {
                "candidate_id": metric["candidate_id"],
                "attempt": metric["attempt"],
                "retry_gate_pass": metric["retry_gate_pass"],
                "failure_reasons": metric["failure_reasons"],
                "output_wav": metric["output_wav"],
                "audio_sha256": metric["audio_sha256"],
                "metric_sha256": sha256_bytes(canonical_json(metric).encode()),
            }
            for metric in candidates
        ]
        selection = {
            "schema_version": SCHEMA,
            "id": row_id,
            "speaker_id": row["speaker_id"],
            "eligibility_split": row["eligibility_split"],
            "status": "accepted" if selected is not None else "rejected",
            "selected_attempt": selected["attempt"] if selected is not None else None,
            "selected_candidate_id": selected["candidate_id"] if selected is not None else None,
            "candidates": candidate_summaries,
        }
        selections.append(selection)
        if selected is None:
            rejected.append({**selection, "rejection_reasons": candidate_summaries})
            continue
        selected_metrics.append(selected)
        generated = generation_rows[(row_id, int(selected["attempt"]))]
        accepted.append(
            {
                "schema_version": SCHEMA,
                "id": row_id,
                "speaker_id": row["speaker_id"],
                "eligibility_split": row["eligibility_split"],
                "text_vi": row["text_vi"],
                "text_en": row["text_en"],
                "text_vi_sha256": row["text_vi_sha256"],
                "text_en_sha256": row["text_en_sha256"],
                "source_audio": row["source_audio"],
                "source_provenance": row["source_provenance"],
                "source_audit": row["source_audit"],
                "reference": row["reference"],
                "target_audio": {
                    "path": selected["output_wav"],
                    "sha256": selected["audio_sha256"],
                    "duration_s": selected["duration_s"],
                    "attempt": selected["attempt"],
                    "seed": generated["seed"],
                    "synthesis": generated["synthesis"],
                    "generation_sidecar": selected["generation_sidecar"],
                },
                "target_qa": {
                    "candidate_id": selected["candidate_id"],
                    "asr_transcript_en": selected["asr_transcript_en"],
                    "asr_word_errors": selected["asr_word_errors"],
                    "asr_reference_words": selected["asr_reference_words"],
                    "asr_wer": selected["asr_wer"],
                    "speaker_cosine": selected["speaker_cosine"],
                    "prompt_leak": selected["prompt_leak"],
                    "acoustic": selected["acoustic"],
                    "duration_ratio_target_source": selected["duration_ratio_target_source"],
                    "models": selected["models"],
                    "runtime": selected["runtime"],
                    "thresholds": selected["thresholds"],
                    "metric_sha256": sha256_bytes(canonical_json(selected).encode()),
                },
            }
        )

    selected_candidates = [str(metric["candidate_id"]) for metric in selected_metrics]
    sample = set(manual_sample(selected_candidates))
    required: dict[str, set[str]] = {}
    for candidate_id in sample:
        required.setdefault(candidate_id, set()).add("seeded_selected_sample")
    rejected_ids = {str(row["id"]) for row in rejected}
    for candidate_id in failed_candidates:
        required.setdefault(candidate_id, set()).add("machine_failure")
        if candidate_id.rsplit("|attempt", 1)[0] in rejected_ids:
            required[candidate_id].add("rejected_row")
    out_dir = args.out_dir.expanduser().resolve()
    review_template = out_dir / "manual_review_required.tsv"
    candidate_metrics = {
        str(metric["candidate_id"]): metric for metric in [*attempt0.values(), *attempt1.values()]
    }
    write_review_template(review_template, required, candidate_metrics)
    allowed_candidates = {
        f"{row_id}|attempt{attempt}"
        for row_id in order
        for attempt in ((0, 1) if row_id in attempt1 else (0,))
    }
    reviews, review_attestation = load_manual_reviews(args.manual_review, allowed_candidates)
    missing_reviews = sorted(set(required) - set(reviews))
    sampled_pass = not missing_reviews and all(
        reviews[candidate_id]["status"] == "pass" and reviews[candidate_id]["prompt_leak"] == "no"
        for candidate_id in sample
    )
    failures_reviewed = not missing_reviews

    total_errors = sum(int(metric["asr_word_errors"]) for metric in selected_metrics)
    total_words = sum(int(metric["asr_reference_words"]) for metric in selected_metrics)
    selected_wer = total_errors / total_words if total_words else None
    cosine_median = (
        median(float(metric["speaker_cosine"]) for metric in selected_metrics)
        if selected_metrics
        else None
    )
    prompt_matches = sum(
        int(metric["prompt_leak"]["reference_only_3gram_match_count"])
        for metric in selected_metrics
    )
    machine_checks = {
        "exact_plan_selection_scope": len(selections) == EXPECTED_ROWS
        and len(accepted) + len(rejected) == EXPECTED_ROWS,
        "all_accepted_rows_pass_retry_gate": all(
            metric["retry_gate_pass"] for metric in selected_metrics
        ),
        "selected_asr_wer": selected_wer is not None
        and selected_wer <= CORPUS_THRESHOLDS["selected_asr_wer_max"],
        "selected_speaker_cosine_median": cosine_median is not None
        and cosine_median >= CORPUS_THRESHOLDS["selected_speaker_cosine_median_min"],
        "zero_accepted_prompt_leaks": prompt_matches
        <= CORPUS_THRESHOLDS["accepted_prompt_leak_matches_max"],
    }
    manual_checks = {
        "manual_review_complete": not missing_reviews,
        "seeded_selected_sample_pass": sampled_pass,
        "every_rejection_and_failure_reviewed": failures_reviewed,
    }
    if not all(machine_checks.values()) or (
        not missing_reviews and not all(manual_checks.values())
    ):
        decision = "no_go"
    elif missing_reviews:
        decision = "pending_manual_review"
    else:
        decision = "go"

    selection_path = out_dir / "selection.jsonl"
    accepted_path = out_dir / "accepted.jsonl"
    rejected_path = out_dir / "rejected.jsonl"
    immutable_write(selection_path, jsonl_bytes(selections))
    immutable_write(accepted_path, jsonl_bytes(accepted))
    immutable_write(rejected_path, jsonl_bytes(rejected))
    report = {
        "schema_version": SCHEMA,
        "decision": decision,
        "machine_checks": machine_checks,
        "manual_checks": manual_checks,
        "thresholds": {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS},
        "scope": {
            "plan_rows": len(rows),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "attempt0_failures": len(expected_retry_order),
            "attempt1_failures": sum(not metric["retry_gate_pass"] for metric in attempt1.values()),
        },
        "metrics": {
            "selected_asr_word_errors": total_errors,
            "selected_asr_reference_words": total_words,
            "selected_asr_wer": selected_wer,
            "selected_speaker_cosine_median": cosine_median,
            "accepted_prompt_leak_matches": prompt_matches,
        },
        "manual_review": {
            "required_tsv": str(review_template),
            "required_candidates": len(required),
            "seeded_sample_candidates": sorted(sample),
            "failed_candidates": sorted(failed_candidates),
            "missing_candidates": missing_reviews,
            "review_file": review_attestation,
        },
        "inputs": {
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "campaign_artifacts": artifacts,
            "generation_manifest": attestation(generation_path),
            "attempt0_metrics": attestation(attempt0_path),
            "retry_ids": attestation(retry_path),
            "attempt1_metrics": attestation(attempt1_path) if attempt1_path is not None else None,
        },
        "outputs": {
            "selection": attestation(selection_path),
            "accepted": attestation(accepted_path),
            "rejected": attestation(rejected_path),
        },
    }
    atomic_write_json(out_dir / "aggregate_report.json", report)
    print(f"Full-campaign decision: {decision}")
    print(
        f"Accepted {len(accepted)}; rejected {len(rejected)}; manual reviews pending {len(missing_reviews)}"
    )


def main() -> None:
    args = parse_args()
    score_attempt(args) if args.action == "score-attempt" else finalize(args)


if __name__ == "__main__":
    main()
