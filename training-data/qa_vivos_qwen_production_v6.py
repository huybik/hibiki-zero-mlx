"""Score, select, and finalize the immutable Qwen MLX retry-v6 campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_vivos_qwen_mlx_retry_v6 import production_attestation_path
from qa_vivos_full import (
    CORPUS_THRESHOLDS,
    MODELS,
    ROW_THRESHOLDS,
    load_models,
    speaker_embedding,
)
from qa_vivos_qwen_mlx_batch_v2 import failures as frozen_failures
from qa_vivos_tts import (
    SCIPY_VERSION,
    TRANSFORMERS_VERSION,
    acoustic_metrics,
    prompt_leak_evidence,
    read_audio,
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
from validate_vivos_qwen_production_v6 import (
    ATTEMPTS,
    InvalidGeneration,
    attestation,
    load_production_plan,
    validate_production_attempt,
)


SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_qa_v6"
SELECTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_selection_v6"
FINAL_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_final_v6"
RNG_HELPER_SHA256 = "cb96149414e1c991c0ea29908b3d99a02dd73a12dcd849fde3d6e025eb5dbe82"
RNG_SCHEMA = "hibiki-qwen-mlx-row-rng-v1"
TERMINAL_NO_RETRY_POLICY = {
    "enabled": True,
    "reason": "user_requested_drop_after_attempt0_validation",
    "corpus_wer_pruning": {
        "objective": "minimum_rows_removed_to_pass_corpus_wer",
        "ranking": "descending_word_error_surplus_then_row_id",
        "threshold": CORPUS_THRESHOLDS["selected_asr_wer_max"],
    },
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode()


def runtime_contract(device: str) -> dict[str, Any]:
    return {
        "transformers": package_version("transformers"),
        "scipy": package_version("scipy"),
        "torch": package_version("torch"),
        "soundfile": package_version("soundfile"),
        "numpy": package_version("numpy"),
        "device": device,
        "model_dtype": "torch.float32",
        "attention_implementation": "eager",
        "attention_masks": {"whisper": True, "wavlm": True},
    }


def require_runtime(device: str) -> dict[str, Any]:
    runtime = runtime_contract(device)
    if (
        device != "mps"
        or runtime["transformers"] != TRANSFORMERS_VERSION
        or runtime["scipy"] != SCIPY_VERSION
    ):
        raise RuntimeError("QA requires the exact pinned MPS environment and model dependencies")
    return runtime


def _candidate_rows(
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    candidate = validation.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("Generation has not completed candidate assembly")
    root = Path(str(candidate["raw_results"])).resolve().parent
    generation_manifest = root.parent / f"generation_attempt{validation['attempt']}_manifest.json"
    for record in validation["records"]:
        group_path = root / "groups" / record["group"]["group_id"] / "group.json"
        group_record = attestation(group_path)
        for row in record["rows"]:
            row_id = str(row["id"])
            rows.append(row)
            provenance[row_id] = {
                "generation_manifest": attestation(generation_manifest),
                "candidate_record": attestation(root / "candidate.json"),
                "raw_results": attestation(root / "raw_results.jsonl"),
                "group_record": group_record,
                "candidate_row_sha256": sha256_bytes(canonical_json(row).encode()),
            }
    return rows, provenance


def metric_path(out_dir: Path, row: dict[str, Any]) -> Path:
    return (
        out_dir
        / "rows"
        / str(row["eligibility_split"])
        / str(row["speaker_id"])
        / f"{str(row['id']).replace(':', '_')}.json"
    )


def _binding(
    plan_path: Path,
    plan: dict[str, Any],
    row: dict[str, Any],
    provenance: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    source = {
        "id": row["id"],
        "speaker_id": row["speaker_id"],
        "eligibility_split": row["eligibility_split"],
        "text_en_sha256": sha256_bytes(str(row["text_en"]).encode()),
        "source_plan_row_sha256": row["source_plan_row_sha256"],
        "source_audio_sha256": row["source_audio"]["sha256"],
        "reference_audio_sha256": row["reference"]["reference_audio_sha256"],
    }
    value = {
        "production_plan": attestation(plan_path),
        "production_attestation": attestation(production_attestation_path(plan_path)),
        "policy": plan["policy"],
        "source_plan": plan["source_plan"],
        "source": source,
        "candidate": {
            **provenance,
            "attempt": row["attempt"],
            "attempt_name": row["attempt_name"],
            "output_wav": row["output_wav"],
            "audio_sha256": row["audio_sha256"],
            "codes": row["codes"],
            "codes_sha256": row["codes_sha256"],
        },
        "models": MODELS,
        "thresholds": ROW_THRESHOLDS,
        "runtime": runtime,
    }
    value["binding_sha256"] = sha256_bytes(canonical_json(value).encode())
    return value


def _empty_acoustic() -> dict[str, Any]:
    return {
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


def _reference_embedding(
    row: dict[str, Any],
    out_dir: Path,
    objects: dict[str, Any],
    memory: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    speaker = str(row["speaker_id"])
    path = out_dir / "reference_embeddings" / f"{speaker}.json"
    reference = row["reference"]
    contract = {
        "schema_version": SCHEMA,
        "speaker_id": speaker,
        "reference_audio_path": reference["reference_audio_path"],
        "reference_audio_sha256": reference["reference_audio_sha256"],
        "speaker_model": MODELS["speaker"],
    }
    if speaker in memory:
        return memory[speaker], attestation(path)
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if {key: record.get(key) for key in contract} != contract:
            raise RuntimeError(f"Reference embedding provenance changed: {speaker}")
        vector = objects["torch"].tensor(record["embedding"], dtype=objects["torch"].float32)
        if sha256_bytes(canonical_json(record["embedding"]).encode()) != record.get(
            "embedding_sha256"
        ):
            raise RuntimeError(f"Reference embedding content changed: {speaker}")
    else:
        audio, rate = read_audio(
            Path(reference["reference_audio_path"]), objects["sf"], objects["np"]
        )
        if (
            sha256_file(Path(reference["reference_audio_path"]))
            != reference["reference_audio_sha256"]
        ):
            raise RuntimeError(f"Reference audio changed: {speaker}")
        audio_16k = resample(audio, rate, 16_000, objects["scipy_signal"])
        vector = speaker_embedding(
            audio_16k,
            objects["speaker_features"],
            objects["speaker_model"],
            objects["torch"],
            "mps",
        )
        values = [float(value) for value in vector.tolist()]
        record = {
            **contract,
            "embedding": values,
            "embedding_sha256": sha256_bytes(canonical_json(values).encode()),
        }
        immutable_write(path, json_bytes(record))
    memory[speaker] = vector
    return vector, attestation(path)


def score_row(
    row: dict[str, Any],
    binding: dict[str, Any],
    generation_issues: list[str],
    out_dir: Path,
    objects: dict[str, Any] | None,
    references: dict[str, Any],
) -> dict[str, Any]:
    duration: float | None = None
    ratio: float | None = None
    asr_en: str | None = None
    asr_auto: str | None = None
    errors: int | None = None
    words: int | None = None
    cosine: float | None = None
    processing_error: str | None = None
    reference_record: dict[str, str] | None = None
    acoustic = _empty_acoustic()
    if not generation_issues:
        assert objects is not None
        try:
            audio, rate = read_audio(Path(row["output_wav"]), objects["sf"], objects["np"])
            acoustic = {"readable": True, **acoustic_metrics(audio, rate, objects["np"])}
            duration = len(audio) / rate
            ratio = duration / float(row["source_audio"]["duration_s"])
            if acoustic["finite"] and acoustic["nonzero"]:
                audio_16k = resample(audio, rate, 16_000, objects["scipy_signal"])
                asr_en = transcribe(
                    audio_16k,
                    objects["asr_processor"],
                    objects["asr_model"],
                    objects["torch"],
                    "mps",
                    "en",
                )
                asr_auto = transcribe(
                    audio_16k,
                    objects["asr_processor"],
                    objects["asr_model"],
                    objects["torch"],
                    "mps",
                    None,
                )
                errors, words = word_error_counts(str(row["text_en"]), asr_en)
                if len(audio_16k) >= 8_000:
                    reference, reference_record = _reference_embedding(
                        row, out_dir, objects, references
                    )
                    output = speaker_embedding(
                        audio_16k,
                        objects["speaker_features"],
                        objects["speaker_model"],
                        objects["torch"],
                        "mps",
                    )
                    cosine = float(objects["torch"].dot(reference, output))
                else:
                    reference_record = None
            else:
                reference_record = None
        except Exception as error:
            processing_error = f"{type(error).__name__}: {error}"
            reference_record = None
    else:
        processing_error = "generation_validation:" + ",".join(generation_issues)
        reference_record = None
    leak = (
        prompt_leak_evidence(row["reference"]["reference_text_vi"], str(row["text_en"]), asr_auto)
        if asr_auto is not None
        else {
            "asr_auto_transcript": None,
            "reference_only_3gram_match_count": None,
            "reference_only_3gram_matches": [],
        }
    )
    metric = {
        "schema_version": SCHEMA,
        "id": row["id"],
        "candidate_id": f"{row['id']}|{row['attempt_name']}",
        "attempt": row["attempt"],
        "attempt_name": row["attempt_name"],
        "speaker_id": row["speaker_id"],
        "eligibility_split": row["eligibility_split"],
        "binding": binding,
        "output_wav": row["output_wav"],
        "audio_sha256": row["audio_sha256"],
        "codes": row["codes"],
        "codes_sha256": row["codes_sha256"],
        "duration_s": round(duration, 6) if duration is not None else None,
        "duration_ratio_target_source": round(ratio, 6) if ratio is not None else None,
        "acoustic": acoustic,
        "qa_processing_error": processing_error,
        "asr_transcript_en": asr_en,
        "asr_auto_transcript": asr_auto,
        "asr_word_errors": errors,
        "asr_reference_words": words,
        "asr_wer": errors / words if errors is not None and words else None,
        "prompt_leak": leak,
        "speaker_cosine": cosine,
        "reference_embedding": reference_record,
        "models": MODELS,
        "runtime": binding["runtime"],
        "thresholds": ROW_THRESHOLDS,
        "generation_validation_errors": generation_issues,
    }
    reasons = frozen_failures(metric)
    if processing_error is not None:
        reasons.append("qa_processing_error")
    reasons.extend(f"generation_{issue}" for issue in generation_issues)
    metric["failure_reasons"] = sorted(set(reasons))
    metric["row_gate_pass"] = not metric["failure_reasons"]
    return metric


def expected_failures(metric: dict[str, Any], generation_issues: list[str]) -> list[str]:
    reasons = frozen_failures(metric)
    if metric.get("qa_processing_error") is not None:
        reasons.append("qa_processing_error")
    reasons.extend(f"generation_{issue}" for issue in generation_issues)
    return sorted(set(reasons))


def validate_metric(
    metric: dict[str, Any],
    row: dict[str, Any],
    binding: dict[str, Any],
    generation_issues: list[str],
) -> None:
    failures = expected_failures(metric, generation_issues)
    if (
        metric.get("schema_version") != SCHEMA
        or metric.get("id") != row["id"]
        or metric.get("candidate_id") != f"{row['id']}|{row['attempt_name']}"
        or metric.get("attempt") != row["attempt"]
        or metric.get("attempt_name") != row["attempt_name"]
        or metric.get("binding") != binding
        or metric.get("output_wav") != row["output_wav"]
        or metric.get("audio_sha256") != row["audio_sha256"]
        or metric.get("codes_sha256") != row["codes_sha256"]
        or metric.get("models") != MODELS
        or metric.get("thresholds") != ROW_THRESHOLDS
        or metric.get("runtime") != binding["runtime"]
        or metric.get("generation_validation_errors") != generation_issues
        or metric.get("failure_reasons") != failures
        or metric.get("row_gate_pass") != (not failures)
    ):
        raise RuntimeError(f"Resumable QA provenance mismatch: {row['id']}")


def score_attempt(args: argparse.Namespace) -> None:
    runtime = require_runtime(args.device)
    plan_path, plan, _, _, _ = load_production_plan(args.production_plan)
    validation = validate_production_attempt(plan_path, args.attempt, args.retry_manifest)
    if validation["state"] == "incomplete":
        raise RuntimeError("Generation attempt is incomplete; QA was not started")
    candidates, provenance = _candidate_rows(validation)
    expected_ids = [str(row["id"]) for row in candidates]
    out_dir = args.out_dir.expanduser().resolve()
    metrics: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in candidates:
        binding = _binding(plan_path, plan, row, provenance[str(row["id"])], runtime)
        path = metric_path(out_dir, row)
        if path.is_file():
            metric = json.loads(path.read_text(encoding="utf-8"))
            validate_metric(
                metric,
                row,
                binding,
                validation["media_errors"].get(str(row["id"]), []),
            )
            metrics[str(row["id"])] = metric
        else:
            pending.append((row, binding))
    objects: dict[str, Any] | None = None
    loaded_runtime: dict[str, Any] | None = None
    references: dict[str, Any] = {}
    for number, (row, binding) in enumerate(pending, 1):
        issues = validation["media_errors"].get(str(row["id"]), [])
        if not issues and objects is None:
            objects, loaded_runtime = load_models(args.device)
            if loaded_runtime != runtime:
                raise RuntimeError("Loaded QA runtime differs from the resumable runtime contract")
        metric = score_row(row, binding, issues, out_dir, objects, references)
        immutable_write(metric_path(out_dir, row), json_bytes(metric))
        metrics[str(row["id"])] = metric
        print(f"[{number}/{len(pending)}] {row['id']}: {metric['failure_reasons']}", flush=True)
    if set(metrics) != set(expected_ids):
        raise RuntimeError("QA metrics are not an exact candidate bijection")
    ordered = [metrics[row_id] for row_id in expected_ids]
    metrics_path = out_dir / "metrics.jsonl"
    immutable_write(metrics_path, jsonl_bytes(ordered))
    errors = sum(int(row["asr_word_errors"] or 0) for row in ordered)
    words = sum(int(row["asr_reference_words"] or 0) for row in ordered)
    cosines = [float(row["speaker_cosine"]) for row in ordered if row["speaker_cosine"] is not None]
    leaks = sum(int(row["prompt_leak"]["reference_only_3gram_match_count"] or 0) for row in ordered)
    report = {
        "schema_version": SCHEMA,
        "status": "complete",
        "attempt": args.attempt,
        "attempt_name": ATTEMPTS[args.attempt]["name"],
        "scope_rows": len(ordered),
        "production_plan": attestation(plan_path),
        "generation_manifest": attestation(
            plan_path.parent / f"generation_attempt{args.attempt}_manifest.json"
        ),
        "retry_manifest": attestation(args.retry_manifest) if args.retry_manifest else None,
        "row_metrics": attestation(metrics_path),
        "models": MODELS,
        "runtime": runtime,
        "thresholds": {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS},
        "row_gate_failures": sum(not row["row_gate_pass"] for row in ordered),
        "asr_word_errors": errors,
        "asr_reference_words": words,
        "asr_wer": errors / words if words else None,
        "speaker_cosine_median": median(cosines) if cosines else None,
        "prompt_leak_matches": leaks,
        "models_loaded": objects is not None,
    }
    immutable_write(out_dir / "qa_report.json", json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_qa_attempt(
    plan_path: Path,
    plan: dict[str, Any],
    attempt: int,
    retry_manifest: Path | None,
    qa_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    validation = validate_production_attempt(plan_path, attempt, retry_manifest)
    if validation["state"] not in {"complete", "complete_with_media_errors"}:
        raise RuntimeError(f"Attempt {attempt} generation is not complete and valid")
    rows, provenance = _candidate_rows(validation)
    directory = qa_root / ATTEMPTS[attempt]["name"]
    report_path = directory / "qa_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    if (
        report.get("schema_version") != SCHEMA
        or report.get("status") != "complete"
        or report.get("attempt") != attempt
        or report.get("production_plan") != attestation(plan_path)
        or report.get("generation_manifest")
        != attestation(plan_path.parent / f"generation_attempt{attempt}_manifest.json")
        or report.get("retry_manifest") != (attestation(retry_manifest) if retry_manifest else None)
        or report.get("row_metrics") != attestation(metrics_path)
        or report.get("models") != MODELS
        or report.get("runtime") != require_runtime("mps")
        or report.get("thresholds") != {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS}
        or [metric.get("id") for metric in metrics] != [row["id"] for row in rows]
    ):
        raise RuntimeError(f"Attempt-{attempt} QA report/scope binding mismatch")
    output: dict[str, dict[str, Any]] = {}
    by_id = {str(row["id"]): row for row in rows}
    for metric in metrics:
        row = by_id[str(metric["id"])]
        binding = _binding(plan_path, plan, row, provenance[str(row["id"])], report["runtime"])
        validate_metric(
            metric,
            row,
            binding,
            validation["media_errors"].get(str(row["id"]), []),
        )
        if sha256_file(Path(metric["output_wav"])) != metric["audio_sha256"]:
            raise RuntimeError(f"QA-selected audio changed: {metric['candidate_id']}")
        output[str(metric["id"])] = metric
    return rows, output, report


def _select(candidates: list[dict[str, Any]], order: list[int]) -> dict[str, Any] | None:
    rank = {attempt: index for index, attempt in enumerate(order)}
    passing = [
        metric
        for metric in candidates
        if metric.get("row_gate_pass") is True
        and metric.get("failure_reasons") == []
        and metric.get("asr_word_errors") is not None
        and metric.get("asr_wer") is not None
        and float(metric["asr_wer"]) <= ROW_THRESHOLDS["asr_wer_max"]
    ]
    return (
        min(
            passing,
            key=lambda metric: (
                int(metric["asr_word_errors"]),
                float(metric["asr_wer"]),
                rank[int(metric["attempt"])],
            ),
        )
        if passing
        else None
    )


def _selection_policy_hash(policy: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(policy["retry_policy"]).encode())


def _speaker_exclusion_record(
    source_rows: list[dict[str, Any]], speaker_ids: list[str]
) -> dict[str, Any]:
    requested = sorted(speaker_ids)
    if len(requested) != len(set(requested)):
        raise RuntimeError("Speaker exclusions contain duplicates")
    available = {str(row["speaker_id"]) for row in source_rows}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise RuntimeError(f"Unknown speaker exclusions: {unknown}")
    rows = sum(str(row["speaker_id"]) in set(requested) for row in source_rows)
    return {
        "speaker_ids": requested,
        "rows": rows,
        "reason": "user_requested_quality_exclusion",
    }


def _selection_scope_hash(
    policy: dict[str, Any],
    exclusions: dict[str, Any],
    terminal_policy: dict[str, Any] | None = None,
) -> str:
    if not exclusions["speaker_ids"] and terminal_policy is None:
        return _selection_policy_hash(policy)
    scope = {"retry_policy": policy["retry_policy"]}
    if exclusions["speaker_ids"]:
        scope["speaker_exclusions"] = exclusions
    if terminal_policy is not None:
        scope["terminal_policy"] = terminal_policy
    return sha256_bytes(canonical_json(scope).encode())


def _word_error_surplus(metric: dict[str, Any]) -> int:
    threshold = Fraction(str(CORPUS_THRESHOLDS["selected_asr_wer_max"]))
    return (
        int(metric["asr_word_errors"]) * threshold.denominator
        - int(metric["asr_reference_words"]) * threshold.numerator
    )


def _terminal_prune_ids(selected: dict[str, dict[str, Any]]) -> list[str]:
    threshold = Fraction(str(CORPUS_THRESHOLDS["selected_asr_wer_max"]))
    errors = sum(int(metric["asr_word_errors"]) for metric in selected.values())
    words = sum(int(metric["asr_reference_words"]) for metric in selected.values())
    ranked = sorted(
        selected.items(),
        key=lambda item: (-_word_error_surplus(item[1]), item[0]),
    )
    pruned = []
    for row_id, metric in ranked:
        if words and errors * threshold.denominator <= words * threshold.numerator:
            break
        if _word_error_surplus(metric) <= 0:
            break
        pruned.append(row_id)
        errors -= int(metric["asr_word_errors"])
        words -= int(metric["asr_reference_words"])
    return pruned


def _terminal_exclusion() -> dict[str, str]:
    return {
        "kind": "terminal_corpus_wer_prune",
        "reason": TERMINAL_NO_RETRY_POLICY["reason"],
    }


def _terminal_pruning_record(row_ids: list[str]) -> dict[str, Any]:
    ordered = sorted(row_ids)
    return {
        "rows": len(ordered),
        "row_ids_sha256": sha256_bytes(canonical_json(ordered).encode()),
    }


def select_round(args: argparse.Namespace) -> None:
    plan_path, plan, policy, source_rows, _ = load_production_plan(args.production_plan)
    qa_root = args.qa_root.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    source_order = [str(row["id"]) for row in source_rows]
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempt_records = []
    retry_paths: dict[int, Path | None] = {0: None}
    for attempt in range(1, args.through_round + 1):
        retry_paths[attempt] = out / f"retry_round{attempt}.jsonl"
    for attempt in range(args.through_round + 1):
        _, values, qa_report = load_qa_attempt(
            plan_path, plan, attempt, retry_paths[attempt], qa_root
        )
        for row_id, metric in values.items():
            candidates[row_id].append(metric)
        attempt_records.append(
            {
                "attempt": attempt,
                "generation_manifest": attestation(
                    plan_path.parent / f"generation_attempt{attempt}_manifest.json"
                ),
                "qa_report": attestation(qa_root / ATTEMPTS[attempt]["name"] / "qa_report.json"),
                "retry_manifest": attestation(retry_paths[attempt])
                if retry_paths[attempt]
                else None,
            }
        )
    selections = []
    selected_by_id: dict[str, dict[str, Any]] = {}
    retry = []
    exclusions = _speaker_exclusion_record(source_rows, args.exclude_speaker)
    excluded_speakers = set(exclusions["speaker_ids"])
    source_by_id = {str(row["id"]): row for row in source_rows}
    terminal_policy = TERMINAL_NO_RETRY_POLICY if args.terminal_no_retry else None
    policy_hash = _selection_scope_hash(policy, exclusions, terminal_policy)
    for row_id in source_order:
        choices = candidates[row_id]
        speaker_id = str(source_by_id[row_id]["speaker_id"])
        exclusion = (
            {
                "kind": "speaker",
                "speaker_id": speaker_id,
                "reason": exclusions["reason"],
            }
            if speaker_id in excluded_speakers
            else None
        )
        selected = None if exclusion else _select(choices, list(range(args.through_round + 1)))
        summaries = [
            {
                "candidate_id": metric["candidate_id"],
                "attempt": metric["attempt"],
                "row_gate_pass": metric["row_gate_pass"],
                "failure_reasons": metric["failure_reasons"],
                "asr_word_errors": metric["asr_word_errors"],
                "asr_reference_words": metric["asr_reference_words"],
                "asr_wer": metric["asr_wer"],
                "speaker_cosine": metric["speaker_cosine"],
                "prompt_leak_matches": metric["prompt_leak"]["reference_only_3gram_match_count"],
                "output_wav": metric["output_wav"],
                "audio_sha256": metric["audio_sha256"],
                "metric_sha256": sha256_bytes(canonical_json(metric).encode()),
            }
            for metric in choices
        ]
        selections.append(
            {
                "schema_version": SELECTION_SCHEMA,
                "id": row_id,
                "status": "accepted" if selected else "rejected",
                "selected_candidate_id": selected["candidate_id"] if selected else None,
                "selected_attempt": selected["attempt"] if selected else None,
                "exclusion": exclusion,
                "candidates": summaries,
            }
        )
        if selected:
            selected_by_id[row_id] = selected
        if exclusion is None and (selected is None or int(selected["asr_word_errors"]) >= int(
            policy["retry_policy"]["word_errors_min"]
        )):
            retry.append(
                {
                    "id": row_id,
                    "retry_round": args.through_round + 1,
                    "trigger": "no_passing_candidate"
                    if selected is None
                    else "selected_word_errors_gte_4",
                    "selected_word_errors": selected["asr_word_errors"] if selected else None,
                    "selection_through_round": args.through_round,
                    "production_plan": attestation(plan_path),
                    "policy": plan["policy"],
                    "selection_policy_sha256": policy_hash,
                    "speaker_exclusions": exclusions,
                }
            )
    pruned_ids = _terminal_prune_ids(selected_by_id) if terminal_policy else []
    pruned = set(pruned_ids)
    if pruned:
        for selection in selections:
            if selection["id"] in pruned:
                selection.update(
                    {
                        "status": "rejected",
                        "selected_candidate_id": None,
                        "selected_attempt": None,
                        "exclusion": _terminal_exclusion(),
                    }
                )
        selected_by_id = {
            row_id: metric for row_id, metric in selected_by_id.items() if row_id not in pruned
        }
    selected_metrics = list(selected_by_id.values())
    errors = sum(int(metric["asr_word_errors"]) for metric in selected_metrics)
    words = sum(int(metric["asr_reference_words"]) for metric in selected_metrics)
    cosines = [float(metric["speaker_cosine"]) for metric in selected_metrics]
    leaks = sum(
        int(metric["prompt_leak"]["reference_only_3gram_match_count"])
        for metric in selected_metrics
    )
    checks = {
        "selected_corpus_wer": bool(words)
        and errors / words <= CORPUS_THRESHOLDS["selected_asr_wer_max"],
        "selected_speaker_cosine_median": bool(cosines)
        and median(cosines) >= CORPUS_THRESHOLDS["selected_speaker_cosine_median_min"],
        "zero_selected_prompt_leaks": leaks == 0,
        "all_selected_candidates_pass_every_row_gate": bool(selected_metrics)
        and all(
            metric["row_gate_pass"] and not metric["failure_reasons"] for metric in selected_metrics
        ),
    }
    decision = "go" if all(checks.values()) else (
        "no_go"
        if terminal_policy is not None
        else (
            "continue"
            if args.through_round < int(policy["retry_policy"]["maximum_new_rounds"])
            else "no_go"
        )
    )
    if decision != "continue":
        retry = []
    selection_path = out / f"selection_rows_round{args.through_round}.jsonl"
    immutable_write(selection_path, jsonl_bytes(selections))
    retry_path = out / f"retry_round{args.through_round + 1}.jsonl"
    if decision == "continue":
        selection_record = attestation(selection_path)
        retry = [{**row, "selection_rows": selection_record} for row in retry]
        immutable_write(retry_path, jsonl_bytes(retry))
    report = {
        "schema_version": SELECTION_SCHEMA,
        "production_plan": attestation(plan_path),
        "policy": plan["policy"],
        "selection_policy_sha256": policy_hash,
        "speaker_exclusions": exclusions,
        "terminal_policy": terminal_policy,
        "terminal_pruning": _terminal_pruning_record(pruned_ids),
        "through_round": args.through_round,
        "attempts": attempt_records,
        "decision": decision,
        "machine_checks": checks,
        "accepted_rows": len(selected_metrics),
        "rejected_rows": len(source_order) - len(selected_metrics),
        "metrics": {
            "asr_word_errors": errors,
            "asr_reference_words": words,
            "asr_wer": errors / words if words else None,
            "speaker_cosine_median": median(cosines) if cosines else None,
            "prompt_leak_matches": leaks,
        },
        "selection_rows": attestation(selection_path),
        "next_retry": attestation(retry_path) if decision == "continue" else None,
    }
    immutable_write(out / f"selection_round{args.through_round}.json", json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit_historical_selection(args: argparse.Namespace) -> None:
    policy_path = args.policy.expanduser().resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cohort = read_jsonl(Path(policy["cohort"]["path"]))
    report_path = args.selection_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    through_round = int(report["through_round"])
    if (
        report.get("policy") != attestation(policy_path)
        or report.get("decision") not in {"go", "continue", "no_go"}
        or through_round not in (0, 1, 2)
    ):
        raise RuntimeError("Historical selection does not bind the frozen policy")
    qa_root = args.qa_root.expanduser().resolve()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in range(through_round + 1):
        directory = qa_root / ATTEMPTS[attempt]["name"]
        qa_report = json.loads((directory / "qa_report.json").read_text(encoding="utf-8"))
        metrics = read_jsonl(directory / "metrics.jsonl")
        if qa_report.get("candidate_record") != report["attempts"][attempt]:
            raise RuntimeError(f"Historical attempt-{attempt} QA/candidate binding changed")
        if report["qa"][attempt] != attestation(directory / "qa_report.json"):
            raise RuntimeError(f"Historical attempt-{attempt} QA attestation changed")
        for metric in metrics:
            candidate = {**metric, "attempt": attempt}
            if candidate.get("failure_reasons") != frozen_failures(candidate):
                raise RuntimeError(f"Historical frozen row gates changed: {candidate['id']}")
            candidate["row_gate_pass"] = not candidate["failure_reasons"]
            candidates[str(candidate["id"])].append(candidate)
    archived = read_jsonl(Path(report["selection_rows"]["path"]))
    if [row.get("id") for row in archived] != [row["id"] for row in cohort]:
        raise RuntimeError("Historical selection row order changed")
    selected = []
    for source, stored in zip(cohort, archived):
        choice = _select(candidates[str(source["id"])], list(range(through_round + 1)))
        expected_attempt = choice["attempt"] if choice else None
        expected_status = "accepted" if choice else "rejected"
        if (
            stored.get("selected_attempt") != expected_attempt
            or stored.get("status") != expected_status
            or stored.get("selected_word_errors") != (choice["asr_word_errors"] if choice else None)
        ):
            raise RuntimeError(f"Historical selection rule mismatch: {source['id']}")
        if choice:
            selected.append(choice)
    errors = sum(int(metric["asr_word_errors"]) for metric in selected)
    words = sum(int(metric["asr_reference_words"]) for metric in selected)
    cosines = [float(metric["speaker_cosine"]) for metric in selected]
    leaks = sum(
        int(metric["prompt_leak"]["reference_only_3gram_match_count"]) for metric in selected
    )
    recomputed = {
        "asr_word_errors": errors,
        "asr_reference_words": words,
        "asr_wer": errors / words if words else None,
        "speaker_cosine_median": median(cosines) if cosines else None,
        "prompt_leak_matches": leaks,
    }
    if recomputed != report.get("metrics"):
        raise RuntimeError("Historical aggregate selection metrics changed")
    print(
        json.dumps(
            {
                "valid": True,
                "decision": report["decision"],
                "through_round": through_round,
                "rows": len(cohort),
                "accepted_rows": len(selected),
                "metrics": recomputed,
                "selection_report": attestation(report_path),
            },
            indent=2,
        )
    )


def rng_provenance(policy: dict[str, Any], row_id: str, attempt: int) -> dict[str, Any]:
    payload = f"{RNG_SCHEMA}\0{policy['campaign_revision']}\0{row_id}\0attempt={attempt}".encode()
    return {
        "frozen_policy_rng": policy["rng"],
        "erratum": {
            "date": "2026-08-04",
            "classification": "prose-only provenance correction; frozen policy unchanged",
            "executable_root_formula": "SHA256(f'{schema}\\0{campaign_revision}\\0{row_id}\\0attempt={attempt}')",
            "attempt2_is_distinct": True,
            "row_root_digest_helper_sha256": RNG_HELPER_SHA256,
        },
        "derived_row_root_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _duration_band(seconds: float) -> str:
    if seconds < 3:
        return "under_3s"
    if seconds <= 6:
        return "3_to_6s"
    return "over_6s"


def _manual_reviews(
    path: Path | None, allowed: set[str]
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if path is None:
        return {}, None
    path = path.expanduser().resolve()
    reviews = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"candidate_id", "status", "prompt_leak", "notes"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise RuntimeError("Manual review TSV columns changed")
        for line, row in enumerate(reader, 2):
            candidate = row["candidate_id"].strip()
            status = row["status"].strip().casefold()
            leak = row["prompt_leak"].strip().casefold()
            if not status and not leak:
                continue
            if (
                candidate not in allowed
                or candidate in reviews
                or status not in {"pass", "fail"}
                or leak not in {"yes", "no"}
            ):
                raise RuntimeError(f"Invalid manual review at {path}:{line}")
            reviews[candidate] = {
                "status": status,
                "prompt_leak": leak,
                "notes": row["notes"].strip(),
            }
    return reviews, attestation(path)


def finalize(args: argparse.Namespace) -> None:
    plan_path, plan, policy, source_rows, _ = load_production_plan(args.production_plan)
    selection_report_path = args.selection_report.expanduser().resolve()
    report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    exclusions = _speaker_exclusion_record(
        source_rows, list(report.get("speaker_exclusions", {}).get("speaker_ids", []))
    )
    terminal_policy = report.get("terminal_policy")
    if terminal_policy not in (None, TERMINAL_NO_RETRY_POLICY):
        raise RuntimeError("Unknown terminal selection policy")
    if (
        report.get("schema_version") != SELECTION_SCHEMA
        or report.get("production_plan") != attestation(plan_path)
        or report.get("policy") != plan["policy"]
        or report.get("selection_policy_sha256")
        != _selection_scope_hash(policy, exclusions, terminal_policy)
        or report.get("speaker_exclusions") != exclusions
        or report.get("decision") not in {"go", "no_go"}
        or report.get("next_retry") is not None
        or report.get("selection_rows") != attestation(Path(report["selection_rows"]["path"]))
    ):
        raise RuntimeError("Finalization requires an exact terminal v6 selection")
    selection_rows = read_jsonl(Path(report["selection_rows"]["path"]))
    order = [str(row["id"]) for row in source_rows]
    if [row.get("id") for row in selection_rows] != order:
        raise RuntimeError("Selection is not in exact source-plan order")
    through_round = int(report["through_round"])
    qa_root = args.qa_root.expanduser().resolve()
    retry_paths: dict[int, Path | None] = {0: None}
    for attempt in range(1, through_round + 1):
        retry_paths[attempt] = selection_report_path.parent / f"retry_round{attempt}.jsonl"
    metric_by_candidate: dict[str, dict[str, Any]] = {}
    generation_by_attempt: dict[int, dict[str, dict[str, Any]]] = {}
    for attempt in range(through_round + 1):
        candidate_rows, metrics, _ = load_qa_attempt(
            plan_path, plan, attempt, retry_paths[attempt], qa_root
        )
        generation_by_attempt[attempt] = {str(row["id"]): row for row in candidate_rows}
        metric_by_candidate.update({metric["candidate_id"]: metric for metric in metrics.values()})
    source_by_id = {str(row["id"]): row for row in source_rows}
    excluded_speakers = set(exclusions["speaker_ids"])
    initially_selected = {}
    for selected_row in selection_rows:
        row_id = str(selected_row["id"])
        if str(source_by_id[row_id]["speaker_id"]) in excluded_speakers:
            continue
        choice = _select(
            [metric_by_candidate[candidate["candidate_id"]] for candidate in selected_row["candidates"]],
            list(range(through_round + 1)),
        )
        if choice is not None:
            initially_selected[row_id] = choice
    pruned_ids = _terminal_prune_ids(initially_selected) if terminal_policy else []
    if report.get("terminal_pruning") != _terminal_pruning_record(pruned_ids):
        raise RuntimeError("Terminal corpus-WER pruning scope changed")
    pruned = set(pruned_ids)
    selections = []
    accepted = []
    rejected = []
    failed_candidates = set()
    for selected_row in selection_rows:
        row_id = str(selected_row["id"])
        source = source_by_id[row_id]
        candidates = selected_row["candidates"]
        choice_metrics = [
            metric_by_candidate[candidate["candidate_id"]] for candidate in candidates
        ]
        is_excluded = str(source["speaker_id"]) in excluded_speakers
        expected_exclusion = (
            {
                "kind": "speaker",
                "speaker_id": source["speaker_id"],
                "reason": exclusions["reason"],
            }
            if is_excluded
            else (_terminal_exclusion() if row_id in pruned else None)
        )
        recomputed = (
            None
            if expected_exclusion is not None
            else _select(choice_metrics, list(range(through_round + 1)))
        )
        recomputed_id = recomputed["candidate_id"] if recomputed else None
        if (
            selected_row.get("selected_candidate_id") != recomputed_id
            or selected_row.get("selected_attempt")
            != (recomputed["attempt"] if recomputed else None)
            or selected_row.get("status") != ("accepted" if recomputed else "rejected")
            or selected_row.get("exclusion") != expected_exclusion
            or any(
                candidate.get("metric_sha256")
                != sha256_bytes(
                    canonical_json(metric_by_candidate[candidate["candidate_id"]]).encode()
                )
                for candidate in candidates
            )
        ):
            raise RuntimeError(f"Terminal selection differs from frozen selector: {row_id}")
        if expected_exclusion is None:
            for candidate in candidates:
                metric = metric_by_candidate[candidate["candidate_id"]]
                if not metric["row_gate_pass"]:
                    failed_candidates.add(candidate["candidate_id"])
        selected_id = selected_row["selected_candidate_id"]
        selections.append(selected_row)
        if selected_id is None:
            rejected.append({**selected_row, "rejection_reasons": candidates})
            continue
        metric = metric_by_candidate[selected_id]
        attempt = int(metric["attempt"])
        generated = generation_by_attempt[attempt][row_id]
        accepted.append(
            {
                "schema_version": FINAL_SCHEMA,
                "id": row_id,
                "speaker_id": source["speaker_id"],
                "eligibility_split": source["eligibility_split"],
                "text_vi": source["text_vi"],
                "text_en": source["text_en"],
                "text_vi_sha256": source["text_vi_sha256"],
                "text_en_sha256": source["text_en_sha256"],
                "source_audio": source["source_audio"],
                "source_provenance": source["source_provenance"],
                "source_audit": source["source_audit"],
                "reference": source["reference"],
                "target_audio": {
                    "path": generated["output_wav"],
                    "sha256": generated["audio_sha256"],
                    "codes": generated["codes"],
                    "codes_sha256": generated["codes_sha256"],
                    "duration_s": generated["duration_s"],
                    "attempt": attempt,
                    "attempt_name": generated["attempt_name"],
                    "generation_provenance": metric["binding"]["candidate"],
                    "synthesis": {
                        **plan["synthesis"],
                        "temperature": ATTEMPTS[attempt]["temperature"],
                    },
                    "model": plan["model"],
                    "rng": rng_provenance(policy, row_id, attempt),
                },
                "target_qa": {
                    "candidate_id": selected_id,
                    "metric_sha256": sha256_bytes(canonical_json(metric).encode()),
                    "asr_transcript_en": metric["asr_transcript_en"],
                    "asr_word_errors": metric["asr_word_errors"],
                    "asr_reference_words": metric["asr_reference_words"],
                    "asr_wer": metric["asr_wer"],
                    "speaker_cosine": metric["speaker_cosine"],
                    "prompt_leak": metric["prompt_leak"],
                    "acoustic": metric["acoustic"],
                    "duration_ratio_target_source": metric["duration_ratio_target_source"],
                    "models": metric["models"],
                    "runtime": metric["runtime"],
                    "thresholds": metric["thresholds"],
                },
            }
        )
    selected_ids = [row["target_qa"]["candidate_id"] for row in accepted]
    selected_metrics = [metric_by_candidate[candidate_id] for candidate_id in selected_ids]
    selected_errors = sum(int(metric["asr_word_errors"]) for metric in selected_metrics)
    selected_words = sum(int(metric["asr_reference_words"]) for metric in selected_metrics)
    selected_cosines = [float(metric["speaker_cosine"]) for metric in selected_metrics]
    selected_leaks = sum(
        int(metric["prompt_leak"]["reference_only_3gram_match_count"])
        for metric in selected_metrics
    )
    recomputed_metrics = {
        "asr_word_errors": selected_errors,
        "asr_reference_words": selected_words,
        "asr_wer": selected_errors / selected_words if selected_words else None,
        "speaker_cosine_median": median(selected_cosines) if selected_cosines else None,
        "prompt_leak_matches": selected_leaks,
    }
    if recomputed_metrics != report.get("metrics"):
        raise RuntimeError("Terminal aggregate differs from selected row metrics")
    recomputed_checks = {
        "selected_corpus_wer": bool(selected_words)
        and recomputed_metrics["asr_wer"] <= CORPUS_THRESHOLDS["selected_asr_wer_max"],
        "selected_speaker_cosine_median": bool(selected_cosines)
        and recomputed_metrics["speaker_cosine_median"]
        >= CORPUS_THRESHOLDS["selected_speaker_cosine_median_min"],
        "zero_selected_prompt_leaks": selected_leaks == 0,
        "all_selected_candidates_pass_every_row_gate": bool(selected_metrics)
        and all(
            metric["row_gate_pass"] and not metric["failure_reasons"] for metric in selected_metrics
        ),
    }
    recomputed_decision = "go" if all(recomputed_checks.values()) else (
        "no_go"
        if terminal_policy is not None
        else (
            "continue"
            if through_round < int(policy["retry_policy"]["maximum_new_rounds"])
            else "no_go"
        )
    )
    if recomputed_checks != report.get("machine_checks") or recomputed_decision != report.get(
        "decision"
    ):
        raise RuntimeError("Terminal machine decision differs from selected row metrics")
    seed = CORPUS_THRESHOLDS["manual_review_seed"]
    sample = set(
        sorted(selected_ids, key=lambda value: sha256_bytes(f"{seed}\0{value}".encode()))[
            : min(int(CORPUS_THRESHOLDS["manual_review_sample_size"]), len(selected_ids))
        ]
    )
    required: dict[str, set[str]] = defaultdict(set)
    for candidate in sample:
        required[candidate].add("seeded_selected_sample")
    for candidate in failed_candidates:
        required[candidate].add("machine_failure")
    for row in rejected:
        if row.get("exclusion") is not None:
            continue
        for candidate in row["candidates"]:
            required[candidate["candidate_id"]].add("rejected_row")
    out = args.out_dir.expanduser().resolve()
    review_path = out / "manual_review_required.tsv"
    lines = [
        "candidate_id\tid\tattempt\tobligation\toutput_wav\taudio_sha256\tfailure_reasons\tstatus\tprompt_leak\tnotes\n"
    ]
    for candidate_id in sorted(required):
        metric = metric_by_candidate[candidate_id]
        lines.append(
            f"{candidate_id}\t{metric['id']}\t{metric['attempt']}\t{','.join(sorted(required[candidate_id]))}\t"
            f"{metric['output_wav']}\t{metric['audio_sha256']}\t{','.join(metric['failure_reasons'])}\t\t\t\n"
        )
    immutable_write(review_path, "".join(lines).encode())
    reviews, review_record = _manual_reviews(args.manual_review, set(metric_by_candidate))
    waiver_record = None
    waived = False
    required_hash = sha256_bytes(
        canonical_json(
            {candidate: sorted(obligations) for candidate, obligations in sorted(required.items())}
        ).encode()
    )
    if args.manual_waiver is not None:
        waiver_path = args.manual_waiver.expanduser().resolve()
        waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
        if (
            waiver.get("schema_version") != "hibiki_vivos_qwen3_tts_mlx_manual_waiver_v6"
            or waiver.get("waive_manual_review") is not True
            or waiver.get("production_plan") != attestation(plan_path)
            or waiver.get("selection_report") != attestation(selection_report_path)
            or waiver.get("required_candidates_sha256") != required_hash
            or not str(waiver.get("rationale", "")).strip()
        ):
            raise RuntimeError("Manual waiver does not explicitly bind this final scope")
        waiver_record = attestation(waiver_path)
        waived = True
    missing = sorted(set(required) - set(reviews)) if not waived else []
    selected_sample_pass = waived or (
        not (sample - set(reviews))
        and all(
            reviews[candidate]["status"] == "pass" and reviews[candidate]["prompt_leak"] == "no"
            for candidate in sample
        )
    )
    review_pass = (waived or not missing) and selected_sample_pass
    failure_review_ids = failed_candidates | {
        candidate["candidate_id"]
        for row in rejected
        if row.get("exclusion") is None
        for candidate in row["candidates"]
    }
    failures_review_complete = waived or not (failure_review_ids - set(reviews))
    status = (
        "no_go"
        if report["decision"] == "no_go" or (not missing and not review_pass)
        else ("go" if review_pass else "pending_manual_review")
    )
    selection_path = out / "selection.jsonl"
    accepted_path = out / "accepted.jsonl"
    rejected_path = out / "rejected.jsonl"
    candidate_manifest = out / "selected_candidates.jsonl"
    immutable_write(selection_path, jsonl_bytes(selections))
    immutable_write(accepted_path, jsonl_bytes(accepted))
    immutable_write(rejected_path, jsonl_bytes(rejected))
    immutable_write(
        candidate_manifest,
        jsonl_bytes(
            [
                {
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "eligibility_split": row["eligibility_split"],
                    **row["target_audio"],
                }
                for row in accepted
            ]
        ),
    )
    split_summary = {}
    for split in ("train", "dev"):
        source = [row for row in source_rows if row["eligibility_split"] == split]
        chosen = [row for row in accepted if row["eligibility_split"] == split]
        split_summary[split] = {
            "source_rows": len(source),
            "accepted_rows": len(chosen),
            "acceptance_rate": len(chosen) / len(source) if source else None,
            "accepted_source_hours": sum(float(row["source_audio"]["duration_s"]) for row in chosen)
            / 3600,
            "accepted_target_hours": sum(float(row["target_audio"]["duration_s"]) for row in chosen)
            / 3600,
            "accepted_speakers": len({row["speaker_id"] for row in chosen}),
        }
    per_speaker = {}
    for speaker in sorted({str(row["speaker_id"]) for row in source_rows}):
        source = [row for row in source_rows if row["speaker_id"] == speaker]
        chosen = [row for row in accepted if row["speaker_id"] == speaker]
        per_speaker[speaker] = {
            "source_rows": len(source),
            "accepted_rows": len(chosen),
            "acceptance_rate": len(chosen) / len(source),
        }
    per_duration = {}
    for band in ("under_3s", "3_to_6s", "over_6s"):
        source = [
            row
            for row in source_rows
            if _duration_band(float(row["source_audio"]["duration_s"])) == band
        ]
        chosen = [
            row
            for row in accepted
            if _duration_band(float(row["source_audio"]["duration_s"])) == band
        ]
        per_duration[band] = {
            "source_rows": len(source),
            "accepted_rows": len(chosen),
            "acceptance_rate": len(chosen) / len(source) if source else None,
        }
    aggregate = {
        "schema_version": FINAL_SCHEMA,
        "status": status,
        "downstream_cache_compatibility": "requires_v6_cache_adapter",
        "machine_selection_decision": report["decision"],
        "production_plan": attestation(plan_path),
        "selection_report": attestation(selection_report_path),
        "speaker_exclusions": exclusions,
        "terminal_policy": terminal_policy,
        "terminal_pruning": _terminal_pruning_record(pruned_ids),
        "scope": {
            "plan_rows": len(source_rows),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "accepted_speakers": len({row["speaker_id"] for row in accepted}),
            "splits": split_summary,
            "per_speaker": per_speaker,
            "per_source_duration": per_duration,
        },
        "machine_metrics": report["metrics"],
        "machine_checks": report["machine_checks"],
        "manual_review": {
            "required": len(required),
            "required_candidates_sha256": required_hash,
            "seeded_sample": sorted(sample),
            "failed_candidates": sorted(failed_candidates),
            "missing": missing,
            "review_file": review_record,
            "waiver": waiver_record,
            "selected_sample_pass": selected_sample_pass,
            "failures_and_rejections_review_complete": failures_review_complete,
        },
        "outputs": {
            "selection": attestation(selection_path),
            "accepted": attestation(accepted_path),
            "rejected": attestation(rejected_path),
            "selected_candidates": attestation(candidate_manifest),
            "manual_review_required": attestation(review_path),
        },
    }
    finalization_key = sha256_bytes(
        canonical_json(
            {
                "selection_report": attestation(selection_report_path),
                "review_file": review_record,
                "waiver": waiver_record,
                "status": status,
            }
        ).encode()
    )[:16]
    immutable_write(
        out / f"aggregate_report_{status}_{finalization_key}.json",
        json_bytes(aggregate),
    )
    atomic_write_bytes(out / "aggregate_report.json", json_bytes(aggregate))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    score = commands.add_parser("score-attempt")
    score.add_argument("production_plan", type=Path)
    score.add_argument("--attempt", type=int, choices=(0, 1, 2), required=True)
    score.add_argument("--retry-manifest", type=Path)
    score.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--device", default="mps")
    select = commands.add_parser("select")
    select.add_argument("production_plan", type=Path)
    select.add_argument("--through-round", type=int, choices=(0, 1, 2), required=True)
    select.add_argument("--qa-root", type=Path, required=True)
    select.add_argument("--out-dir", type=Path, required=True)
    select.add_argument("--exclude-speaker", action="append", default=[])
    select.add_argument("--terminal-no-retry", action="store_true")
    final = commands.add_parser("finalize")
    final.add_argument("production_plan", type=Path)
    final.add_argument("--selection-report", type=Path, required=True)
    final.add_argument("--qa-root", type=Path, required=True)
    final.add_argument("--out-dir", type=Path, required=True)
    final.add_argument("--manual-review", type=Path)
    final.add_argument("--manual-waiver", type=Path)
    historical = commands.add_parser("audit-historical-selection")
    historical.add_argument("policy", type=Path)
    historical.add_argument("--qa-root", type=Path, required=True)
    historical.add_argument("--selection-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        {
            "score-attempt": score_attempt,
            "select": select_round,
            "finalize": finalize,
            "audit-historical-selection": audit_historical_selection,
        }[args.action](args)
    except InvalidGeneration as error:
        raise RuntimeError(f"Generation validation failed: {error}") from error


if __name__ == "__main__":
    main()
