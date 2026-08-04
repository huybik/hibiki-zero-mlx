"""Pinned automatic QA for one Qwen MLX efficiency-v3 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from qa_vivos_full import CORPUS_THRESHOLDS, MODELS, ROW_THRESHOLDS, load_models, speaker_embedding
from qa_vivos_qwen_mlx_batch_v2 import failures
from qa_vivos_tts import (
    acoustic_metrics,
    prompt_leak_evidence,
    read_audio,
    resample,
    transcribe,
    word_error_counts,
)
from synthesize_vivos import immutable_write, read_jsonl, sha256_file

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_efficiency_qa_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_plan", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def load_candidates(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads((root / "candidate.json").read_text())
    records = [json.loads(line) for line in (root / "raw_results.jsonl").read_text().splitlines()]
    rows = [row for record in records for row in record["rows"]]
    if len(rows) != report["scope_rows"] or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Candidate rows are not the reported exact scope")
    for row in rows:
        if sha256_file(Path(row["output_wav"])) != row["audio_sha256"]:
            raise RuntimeError(f"Candidate WAV changed: {row['id']}")
    return report, rows


def main() -> None:
    args = parse_args()
    if args.device != "mps":
        raise RuntimeError("Efficiency QA is pinned to MPS")
    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort = {row["id"]: row for row in read_jsonl(cohort_path)}
    candidate_root = args.candidate_root.expanduser().resolve()
    candidate, rows = load_candidates(candidate_root)
    if any(row["id"] not in cohort for row in rows):
        raise RuntimeError("Candidate is outside the pinned cohort")

    objects, runtime = load_models(args.device)
    np = objects["np"]
    sf = objects["sf"]
    torch = objects["torch"]
    scipy_signal = objects["scipy_signal"]
    references: dict[str, Any] = {}
    metrics = []
    out = args.out_dir.expanduser().resolve()
    cohort_record = attestation(cohort_path)
    for number, row in enumerate(rows, 1):
        source = cohort[row["id"]]
        path = out / "rows" / f"{row['id'].replace(':', '_')}.json"
        if path.is_file():
            metric = json.loads(path.read_text())
            if (
                metric.get("schema_version") != SCHEMA
                or metric.get("candidate") != candidate["candidate"]
                or metric.get("audio_sha256") != row["audio_sha256"]
                or metric.get("models") != MODELS
                or metric.get("thresholds") != ROW_THRESHOLDS
                or metric.get("failure_reasons") != failures(metric)
            ):
                raise RuntimeError(f"Resumable QA mismatch: {row['id']}")
            metrics.append(metric)
            continue
        audio, rate = read_audio(Path(row["output_wav"]), sf, np)
        audio_16k = resample(audio, rate, 16_000, scipy_signal)
        asr_en = transcribe(
            audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", "en"
        )
        asr_auto = transcribe(
            audio_16k, objects["asr_processor"], objects["asr_model"], torch, "mps", None
        )
        errors, words = word_error_counts(source["text_en"], asr_en)
        speaker = source["speaker_id"]
        if speaker not in references:
            ref, ref_rate = read_audio(Path(source["reference"]["reference_audio_path"]), sf, np)
            references[speaker] = speaker_embedding(
                resample(ref, ref_rate, 16_000, scipy_signal),
                objects["speaker_features"], objects["speaker_model"], torch, "mps"
            )
        embedding = speaker_embedding(
            audio_16k,
            objects["speaker_features"], objects["speaker_model"], torch, "mps"
        ) if len(audio_16k) >= 8_000 else None
        metric = {
            "schema_version": SCHEMA,
            "candidate": candidate["candidate"],
            "id": row["id"],
            "speaker_id": speaker,
            "cohort": cohort_record,
            "output_wav": row["output_wav"],
            "audio_sha256": row["audio_sha256"],
            "duration_s": len(audio) / rate,
            "duration_ratio_target_source": len(audio) / rate / float(source["source_audio"]["duration_s"]),
            "acoustic": {"readable": True, **acoustic_metrics(audio, rate, np)},
            "asr_transcript_en": asr_en,
            "asr_auto_transcript": asr_auto,
            "asr_word_errors": errors,
            "asr_reference_words": words,
            "asr_wer": errors / words if words else None,
            "prompt_leak": prompt_leak_evidence(
                source["reference"]["reference_text_vi"], source["text_en"], asr_auto
            ),
            "speaker_cosine": float(torch.dot(references[speaker], embedding)) if embedding is not None else None,
            "models": MODELS,
            "runtime": runtime,
            "thresholds": ROW_THRESHOLDS,
        }
        metric["failure_reasons"] = failures(metric)
        metric["row_gate_pass"] = not metric["failure_reasons"]
        immutable_write(path, json_bytes(metric))
        metrics.append(metric)
        print(f"[{number}/{len(rows)}] {candidate['candidate']}/{row['id']}: {metric['failure_reasons']}", flush=True)

    errors = sum(row["asr_word_errors"] for row in metrics)
    words = sum(row["asr_reference_words"] for row in metrics)
    cosine = median(row["speaker_cosine"] for row in metrics if row["speaker_cosine"] is not None)
    leaks = sum(row["prompt_leak"]["reference_only_3gram_match_count"] for row in metrics)
    checks = {
        "exact_scope": len(metrics) == candidate["scope_rows"],
        "corpus_wer": errors / words <= CORPUS_THRESHOLDS["selected_asr_wer_max"],
        "speaker_cosine_median": cosine >= CORPUS_THRESHOLDS["selected_speaker_cosine_median_min"],
        "zero_prompt_leaks": leaks == 0,
        "all_waveforms_finite_nonzero": all(
            row["acoustic"]["finite"] and row["acoustic"]["nonzero"] for row in metrics
        ),
    }
    report = {
        "schema_version": SCHEMA,
        "candidate": candidate["candidate"],
        "scope_rows": len(metrics),
        "candidate_record": attestation(candidate_root / "candidate.json"),
        "models": MODELS,
        "runtime": runtime,
        "thresholds": {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS},
        "automatic_absolute_pass": all(checks.values()),
        "machine_checks": checks,
        "row_gate_failures": sum(bool(row["failure_reasons"]) for row in metrics),
        "asr_word_errors": errors,
        "asr_reference_words": words,
        "asr_wer": errors / words,
        "speaker_cosine_median": cosine,
        "prompt_leak_matches": leaks,
        "row_metrics": str(out / "metrics.jsonl"),
    }
    immutable_write(out / "metrics.jsonl", jsonl_bytes(metrics))
    immutable_write(out / "qa_report.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
