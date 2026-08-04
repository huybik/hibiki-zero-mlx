"""Run pinned automatic QA for the VIVOS Qwen MLX batch-v2 cohort."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from qa_vivos_full import (
    CORPUS_THRESHOLDS,
    MODELS,
    ROW_THRESHOLDS,
    load_models,
    speaker_embedding,
)
from qa_vivos_tts import (
    acoustic_metrics,
    prompt_leak_evidence,
    read_audio,
    resample,
    transcribe,
    word_error_counts,
)
from synthesize_vivos import atomic_write_bytes, canonical_json, immutable_write, read_jsonl, sha256_bytes, sha256_file

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_qa_v2"
VARIANTS = ("scalar", "batch8", "batch16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_plan", type=Path)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--skip-batch16", action="store_true")
    return parser.parse_args()


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def batch_candidates(root: Path, size: int) -> dict[str, dict[str, Any]]:
    output = {}
    for path in sorted((root / f"batch_size_{size}" / "batches").glob("*/batch.json")):
        record = json.loads(path.read_text())
        for row in record["rows"]:
            row_id = str(row["id"])
            if row_id in output or sha256_file(Path(row["output_wav"])) != row["audio_sha256"]:
                raise RuntimeError(f"Duplicate or changed B{size} output: {row_id}")
            output[row_id] = {**row, "batch_record": attestation(path)}
    return output


def failures(metric: dict[str, Any]) -> list[str]:
    acoustic = metric["acoustic"]
    checks = [
        ("unreadable", acoustic.get("readable") is True),
        ("finite", acoustic.get("finite") is True),
        ("nonzero", acoustic.get("nonzero") is True),
        ("rms", acoustic.get("rms") is not None and acoustic["rms"] >= ROW_THRESHOLDS["rms_min"]),
        (
            "clipping_ratio",
            acoustic.get("clipping_ratio") is not None
            and acoustic["clipping_ratio"] <= ROW_THRESHOLDS["clipping_ratio_max"],
        ),
        (
            "silence_ratio",
            acoustic.get("silence_ratio") is not None
            and acoustic["silence_ratio"] <= ROW_THRESHOLDS["silence_ratio_max"],
        ),
        (
            "leading_silence",
            acoustic.get("leading_silence_s") is not None
            and acoustic["leading_silence_s"] <= ROW_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "trailing_silence",
            acoustic.get("trailing_silence_s") is not None
            and acoustic["trailing_silence_s"] <= ROW_THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "duration_ratio",
            metric.get("duration_ratio_target_source") is not None
            and ROW_THRESHOLDS["duration_ratio_min"]
            <= metric["duration_ratio_target_source"]
            <= ROW_THRESHOLDS["duration_ratio_max"],
        ),
        ("asr_wer", metric.get("asr_wer") is not None and metric["asr_wer"] <= ROW_THRESHOLDS["asr_wer_max"]),
        (
            "prompt_leak",
            metric.get("prompt_leak", {}).get("reference_only_3gram_match_count")
            == ROW_THRESHOLDS["reference_only_3gram_matches_max"],
        ),
        (
            "speaker_cosine",
            metric.get("speaker_cosine") is not None
            and metric["speaker_cosine"] >= ROW_THRESHOLDS["speaker_cosine_min"],
        ),
    ]
    return [name for name, passed in checks if not passed]


def metric_path(out: Path, variant: str, row_id: str) -> Path:
    return out / variant / f"{row_id.replace(':', '_')}.json"


def score(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("Batch-v2 QA is MPS-only")
    cohort_path = args.cohort_plan.expanduser().resolve()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    rows = read_jsonl(cohort_path)
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64:
        raise RuntimeError("Expected the exact 64-row v2 cohort")
    by_id = {str(row["id"]): row for row in rows}
    b8 = batch_candidates(benchmark_root, 8)
    b16 = {} if args.skip_batch16 else batch_candidates(benchmark_root, 16)
    if set(b8) != set(by_id) or (not args.skip_batch16 and set(b16) != set(by_id)):
        raise RuntimeError("Requested batch variants are not exact cohort bijections")
    candidates = {
        "scalar": {
            row_id: {
                "output_wav": row["scalar_baseline"]["output_wav"],
                "audio_sha256": row["scalar_baseline"]["audio_sha256"],
                "sidecar": row["scalar_baseline"]["sidecar"],
            }
            for row_id, row in by_id.items()
        },
        "batch8": b8,
    }
    if not args.skip_batch16:
        candidates["batch16"] = b16
    variants = ("scalar", "batch8") if args.skip_batch16 else VARIANTS
    for variant in variants:
        for row_id, candidate in candidates[variant].items():
            if sha256_file(Path(candidate["output_wav"])) != candidate["audio_sha256"]:
                raise RuntimeError(f"Changed candidate {variant}/{row_id}")
    objects, runtime = load_models(args.device)
    np = objects["np"]
    sf = objects["sf"]
    torch = objects["torch"]
    scipy_signal = objects["scipy_signal"]
    out = args.out_dir.expanduser().resolve()
    cohort_attestation = attestation(cohort_path)
    reference_embeddings: dict[str, Any] = {}
    all_metrics: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in variants}
    total = len(rows) * len(variants)
    completed = 0
    for variant in variants:
        for row_id in by_id:
            row = by_id[row_id]
            candidate = candidates[variant][row_id]
            path = metric_path(out, variant, row_id)
            if path.is_file():
                metric = json.loads(path.read_text())
                if (
                    metric.get("schema_version") != SCHEMA
                    or metric.get("variant") != variant
                    or metric.get("id") != row_id
                    or metric.get("cohort") != cohort_attestation
                    or metric.get("audio_sha256") != candidate["audio_sha256"]
                    or metric.get("models") != MODELS
                    or metric.get("thresholds") != ROW_THRESHOLDS
                    or metric.get("failure_reasons") != failures(metric)
                ):
                    raise RuntimeError(f"Resumable QA mismatch: {variant}/{row_id}")
                all_metrics[variant][row_id] = metric
                completed += 1
                continue
            audio_path = Path(candidate["output_wav"])
            audio, sample_rate = read_audio(audio_path, sf, np)
            acoustic = {"readable": True, **acoustic_metrics(audio, sample_rate, np)}
            audio_16k = resample(audio, sample_rate, 16_000, scipy_signal)
            asr_en = transcribe(
                audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", "en"
            )
            asr_auto = transcribe(
                audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", None
            )
            errors, words = word_error_counts(row["text_en"], asr_en)
            speaker = str(row["speaker_id"])
            if speaker not in reference_embeddings:
                ref_audio, ref_rate = read_audio(Path(row["reference"]["reference_audio_path"]), sf, np)
                ref_16k = resample(ref_audio, ref_rate, 16_000, scipy_signal)
                reference_embeddings[speaker] = speaker_embedding(
                    ref_16k,
                    objects["speaker_features"],
                    objects["speaker_model"],
                    torch,
                    "mps",
                )
            embedding = (
                speaker_embedding(
                    audio_16k,
                    objects["speaker_features"],
                    objects["speaker_model"],
                    torch,
                    "mps",
                )
                if len(audio_16k) >= 8_000
                else None
            )
            metric = {
                "schema_version": SCHEMA,
                "variant": variant,
                "id": row_id,
                "speaker_id": speaker,
                "cohort": cohort_attestation,
                "candidate_provenance": candidate,
                "output_wav": str(audio_path),
                "audio_sha256": candidate["audio_sha256"],
                "duration_s": len(audio) / sample_rate,
                "duration_ratio_target_source": len(audio)
                / sample_rate
                / float(row["source_audio"]["duration_s"]),
                "acoustic": acoustic,
                "asr_transcript_en": asr_en,
                "asr_auto_transcript": asr_auto,
                "asr_word_errors": errors,
                "asr_reference_words": words,
                "asr_wer": errors / words if words else None,
                "prompt_leak": prompt_leak_evidence(
                    row["reference"]["reference_text_vi"], row["text_en"], asr_auto
                ),
                "speaker_cosine": (
                    float(torch.dot(reference_embeddings[speaker], embedding))
                    if embedding is not None
                    else None
                ),
                "models": MODELS,
                "runtime": runtime,
                "thresholds": ROW_THRESHOLDS,
            }
            metric["failure_reasons"] = failures(metric)
            metric["retry_gate_pass"] = not metric["failure_reasons"]
            immutable_write(path, json_bytes(metric))
            all_metrics[variant][row_id] = metric
            completed += 1
            print(f"[{completed}/{total}] {variant}/{row_id}: {metric['failure_reasons']}", flush=True)

    summaries = {}
    for variant in variants:
        metrics = list(all_metrics[variant].values())
        errors = sum(metric["asr_word_errors"] for metric in metrics)
        words = sum(metric["asr_reference_words"] for metric in metrics)
        wer = errors / words
        cosine_values = [
            metric["speaker_cosine"]
            for metric in metrics
            if metric["speaker_cosine"] is not None
        ]
        cosine = median(cosine_values) if cosine_values else None
        leaks = sum(metric["prompt_leak"]["reference_only_3gram_match_count"] for metric in metrics)
        machine_checks = {
            "scope_64": len(metrics) == 64,
            "corpus_wer": wer <= CORPUS_THRESHOLDS["selected_asr_wer_max"],
            "speaker_cosine_median": cosine is not None
            and cosine >= CORPUS_THRESHOLDS["selected_speaker_cosine_median_min"],
            "zero_prompt_leaks": leaks == 0,
            "all_waveforms_finite_nonzero": all(
                metric["acoustic"]["finite"] and metric["acoustic"]["nonzero"]
                for metric in metrics
            ),
        }
        summaries[variant] = {
            "automatic_quality_pass": all(machine_checks.values()),
            "machine_checks": machine_checks,
            "rows": len(metrics),
            "row_gate_failures": sum(bool(metric["failure_reasons"]) for metric in metrics),
            "failure_reasons": {
                reason: sum(reason in metric["failure_reasons"] for metric in metrics)
                for reason in sorted({reason for metric in metrics for reason in metric["failure_reasons"]})
            },
            "asr_word_errors": errors,
            "asr_reference_words": words,
            "asr_wer": wer,
            "speaker_cosine_median": cosine,
            "prompt_leak_matches": leaks,
        }
        metrics_path = out / f"{variant}_metrics.jsonl"
        immutable_write(metrics_path, jsonl_bytes([all_metrics[variant][row["id"]] for row in rows]))
        summaries[variant]["row_metrics"] = attestation(metrics_path)
    comparisons = []
    for row in rows:
        row_id = row["id"]
        scalar = all_metrics["scalar"][row_id]
        for variant in variants[1:]:
            metric = all_metrics[variant][row_id]
            comparisons.append(
                {
                    "id": row_id,
                    "variant": variant,
                    "scalar_audio_sha256": scalar["audio_sha256"],
                    "candidate_audio_sha256": metric["audio_sha256"],
                    "duration_delta_s": metric["duration_s"] - scalar["duration_s"],
                    "wer_delta": metric["asr_wer"] - scalar["asr_wer"],
                    "speaker_cosine_delta": (
                        metric["speaker_cosine"] - scalar["speaker_cosine"]
                        if metric["speaker_cosine"] is not None
                        and scalar["speaker_cosine"] is not None
                        else None
                    ),
                    "scalar_failure_reasons": scalar["failure_reasons"],
                    "candidate_failure_reasons": metric["failure_reasons"],
                }
            )
    comparisons_path = out / "scalar_comparison.jsonl"
    immutable_write(comparisons_path, jsonl_bytes(comparisons))
    report = {
        "schema_version": SCHEMA,
        "decision": "automatic_go"
        if summaries["batch8"]["automatic_quality_pass"]
        else "no_go",
        "cohort": cohort_attestation,
        "benchmark_root": str(benchmark_root),
        "models": MODELS,
        "runtime": runtime,
        "thresholds": {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS},
        "variants": summaries,
        "row_by_row_scalar_comparison": attestation(comparisons_path),
        "remaining_gate": "manual listening; automatic duration/row failures remain deterministic retry candidates",
    }
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(report_dir / "qa_report.json", json_bytes(report))
    atomic_write_bytes(report_dir / "qa_scalar_comparison.jsonl", comparisons_path.read_bytes())
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "sample",
            "variant",
            "audio_file",
            "duration_s",
            "transcript_vi",
            "reference_en",
            "asr_output_en",
            "asr_wer",
            "speaker_cosine",
            "failure_reasons",
            "audio_sha256",
        ],
    )
    writer.writeheader()
    for row in rows:
        for variant in variants:
            metric = all_metrics[variant][row["id"]]
            writer.writerow(
                {
                    "sample": row["id"],
                    "variant": variant,
                    "audio_file": metric["output_wav"],
                    "duration_s": metric["duration_s"],
                    "transcript_vi": row["text_vi"],
                    "reference_en": row["text_en"],
                    "asr_output_en": metric["asr_transcript_en"],
                    "asr_wer": metric["asr_wer"],
                    "speaker_cosine": metric["speaker_cosine"],
                    "failure_reasons": ",".join(metric["failure_reasons"]),
                    "audio_sha256": metric["audio_sha256"],
                }
            )
    atomic_write_bytes(report_dir / "translations.csv", buffer.getvalue().encode())
    print(f"Decision: {report['decision']}")


if __name__ == "__main__":
    score(parse_args())
