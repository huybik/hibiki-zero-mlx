"""Score the pinned VIVOS TTS pilot and emit its go/no-go gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import unicodedata
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from statistics import median
from typing import Any

from synthesize_vivos import (
    MLX_PILOT_SPECS,
    atomic_write_bytes,
    atomic_write_jsonl,
    baseline_output_path,
    read_jsonl,
    sha256_file,
    planned_output_path,
    validate_audio_inputs,
    validate_plan,
    validate_source_audit_report,
)

SCHEMA = "hibiki_vivos_tts_qa_v1"
ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
ASR_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SPEAKER_MODEL_ID = "microsoft/wavlm-base-plus-sv"
SPEAKER_MODEL_REVISION = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
TRANSFORMERS_VERSION = "4.57.3"
SACREBLEU_VERSION = "2.6.0"
SCIPY_VERSION = "1.16.2"

# Hard waveform/content safety bounds are frozen before synthesis. Quality and
# identity use the matched-Kokoro pilot as their calibration boundary.
THRESHOLDS = {
    "catastrophic_wer_max": 0.50,
    "aggregate_wer_margin_vs_kokoro": 0.03,
    "qwen_timbre_win_rate_min": 0.75,
    "clipping_ratio_max": 0.0001,
    "silence_ratio_max": 0.50,
    "leading_trailing_silence_max_s": 2.0,
    "rms_min": 0.0001,
    "duration_ratio_min": 0.40,
    "duration_ratio_max": 1.80,
    "reference_only_3gram_matches_max": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("generation", type=Path)
    parser.add_argument("kokoro_generation", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--manual-review",
        type=Path,
        help="TSV columns: candidate_id, status (pass/fail), prompt_leak (yes/no), notes.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--source-audit-report", type=Path)
    return parser.parse_args()


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(
        path,
        (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def require_package(name: str, expected: str) -> str:
    try:
        installed = package_version(name)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"Scoring requires {name}=={expected} in the separate QA environment"
        ) from error
    if installed != expected:
        raise RuntimeError(f"Scoring requires {name}=={expected}, found {installed}")
    return installed


def load_manual_reviews(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"candidate_id", "status", "prompt_leak", "notes"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Manual review TSV must contain {sorted(required)}: {resolved}"
            )
        reviews: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            candidate_id = row["candidate_id"].strip()
            status = row["status"].strip().casefold()
            prompt_leak = row["prompt_leak"].strip().casefold()
            if not candidate_id or candidate_id in reviews:
                raise RuntimeError(
                    f"Empty or duplicate candidate_id at {resolved}:{line_number}"
                )
            if status not in {"pass", "fail"} or prompt_leak not in {"yes", "no"}:
                raise RuntimeError(f"Invalid review values at {resolved}:{line_number}")
            reviews[candidate_id] = {
                "status": status,
                "prompt_leak": prompt_leak,
                "notes": row["notes"].strip(),
                "review_file": str(resolved),
            }
    return reviews


def normalize_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", normalized, flags=re.UNICODE)


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    left = normalize_words(reference)
    right = normalize_words(hypothesis)
    previous = list(range(len(right) + 1))
    for index, left_word in enumerate(left, start=1):
        current = [index]
        for other_index, right_word in enumerate(right, start=1):
            current.append(
                min(
                    previous[other_index] + 1,
                    current[other_index - 1] + 1,
                    previous[other_index - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1], len(left)


def ngrams(words: list[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


def prompt_leak_evidence(
    reference: str, target: str, auto_transcript: str
) -> dict[str, Any]:
    reference_only = ngrams(normalize_words(reference), 3) - ngrams(
        normalize_words(target), 3
    )
    matches = sorted(reference_only & ngrams(normalize_words(auto_transcript), 3))
    return {
        "asr_auto_transcript": auto_transcript,
        "reference_only_3gram_match_count": len(matches),
        "reference_only_3gram_matches": [" ".join(match) for match in matches],
    }


def read_audio(path: Path, soundfile: Any, numpy: Any) -> tuple[Any, int]:
    audio, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = numpy.asarray(audio, dtype=numpy.float32).mean(axis=1)
    return audio, int(sample_rate)


def resample(audio: Any, source_rate: int, target_rate: int, scipy_signal: Any) -> Any:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return scipy_signal.resample_poly(
        audio, target_rate // divisor, source_rate // divisor
    ).astype("float32")


def acoustic_metrics(audio: Any, sample_rate: int, numpy: Any) -> dict[str, Any]:
    finite = bool(numpy.isfinite(audio).all())
    if not finite or audio.size == 0:
        return {
            "finite": finite,
            "nonzero": False,
            "peak": None,
            "rms": None,
            "dc_offset": None,
            "clipping_ratio": None,
            "silence_ratio": None,
            "leading_silence_s": None,
            "trailing_silence_s": None,
        }
    absolute = numpy.abs(audio)
    rms = float(numpy.sqrt(numpy.mean(numpy.square(audio, dtype=numpy.float64))))
    frame_size = max(1, round(sample_rate * 0.02))
    hop = max(1, round(sample_rate * 0.01))
    if audio.size < frame_size:
        frame_rms = numpy.asarray([rms])
    else:
        starts = list(range(0, audio.size - frame_size + 1, hop))
        frame_rms = numpy.asarray(
            [
                numpy.sqrt(numpy.mean(numpy.square(audio[start : start + frame_size])))
                for start in starts
            ]
        )
    speech = numpy.flatnonzero(frame_rms > 0.01)
    if speech.size:
        leading_silence_s = float(speech[0] * hop / sample_rate)
        trailing_silence_s = float(
            (len(frame_rms) - 1 - speech[-1]) * hop / sample_rate
        )
    else:
        leading_silence_s = trailing_silence_s = float(audio.size / sample_rate)
    return {
        "finite": True,
        "nonzero": bool(absolute.max(initial=0.0) > 0),
        "peak": float(absolute.max(initial=0.0)),
        "rms": rms,
        "dc_offset": float(numpy.mean(audio)),
        "clipping_ratio": float(numpy.mean(absolute >= 0.999)),
        "silence_ratio": float(numpy.mean(frame_rms <= 0.01)),
        "leading_silence_s": leading_silence_s,
        "trailing_silence_s": trailing_silence_s,
    }


def transcribe(
    audio_16k: Any,
    processor: Any,
    model: Any,
    torch: Any,
    device: str,
    language: str | None,
) -> str:
    inputs = processor(
        audio_16k,
        sampling_rate=16_000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    kwargs: dict[str, Any] = {"task": "transcribe", "max_new_tokens": 256}
    if language:
        kwargs["language"] = language
    with torch.inference_mode():
        tokens = model.generate(**model_inputs, **kwargs)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def speaker_embedding(
    audio_16k: Any,
    feature_extractor: Any,
    model: Any,
    torch: Any,
    device: str,
) -> Any:
    inputs = feature_extractor(audio_16k, sampling_rate=16_000, return_tensors="pt")
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        embedding = model(**model_inputs).embeddings[0]
    return torch.nn.functional.normalize(embedding.float(), dim=0).cpu()


def row_gate(metrics: dict[str, Any], *, qwen_candidate: bool) -> list[str]:
    failures: list[str] = []
    acoustic = metrics["acoustic"]
    if not acoustic["finite"]:
        failures.append("non_finite")
    if not acoustic["nonzero"]:
        failures.append("all_zero")
    checks = (
        ("rms_below_min", acoustic["rms"], ">=", THRESHOLDS["rms_min"]),
        (
            "clipping_ratio",
            acoustic["clipping_ratio"],
            "<=",
            THRESHOLDS["clipping_ratio_max"],
        ),
        (
            "silence_ratio",
            acoustic["silence_ratio"],
            "<=",
            THRESHOLDS["silence_ratio_max"],
        ),
        (
            "leading_silence",
            acoustic["leading_silence_s"],
            "<=",
            THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "trailing_silence",
            acoustic["trailing_silence_s"],
            "<=",
            THRESHOLDS["leading_trailing_silence_max_s"],
        ),
        (
            "duration_ratio_below_min",
            metrics["duration_ratio_target_source"],
            ">=",
            THRESHOLDS["duration_ratio_min"],
        ),
        (
            "duration_ratio_above_max",
            metrics["duration_ratio_target_source"],
            "<=",
            THRESHOLDS["duration_ratio_max"],
        ),
    )
    if qwen_candidate:
        checks += (
            (
                "catastrophic_wer",
                metrics["asr_wer"],
                "<=",
                THRESHOLDS["catastrophic_wer_max"],
            ),
            (
                "prompt_leak",
                metrics["prompt_leak"]["reference_only_3gram_match_count"],
                "<=",
                THRESHOLDS["reference_only_3gram_matches_max"],
            ),
        )
    for name, value, operator, threshold in checks:
        if (
            value is None
            or (operator == "<=" and value > threshold)
            or (operator == ">=" and value < threshold)
        ):
            failures.append(name)
    manual = metrics["manual_review"]
    if manual.get("status") != "pass":
        failures.append("manual_review")
    if qwen_candidate and manual.get("prompt_leak") != "no":
        failures.append("manual_prompt_leak")
    return failures


def validate_inputs(
    plan_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    plan_sha: str,
    plan_path: Path,
    source_audit_attestation: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    plan_by_id = {str(row.get("pilot_id", "")): row for row in plan_rows}
    generation_by_id = {str(row.get("pilot_id", "")): row for row in generation_rows}
    if len(plan_by_id) != len(plan_rows) or len(generation_by_id) != len(
        generation_rows
    ):
        raise RuntimeError(
            "Plan or generation manifest contains empty/duplicate pilot ids"
        )
    if set(plan_by_id) != set(generation_by_id):
        missing = sorted(set(plan_by_id) - set(generation_by_id))
        extra = sorted(set(generation_by_id) - set(plan_by_id))
        raise RuntimeError(
            f"Generation is not a plan bijection: missing={missing}, extra={extra}"
        )
    for pilot_id, generated in generation_by_id.items():
        planned = plan_by_id[pilot_id]
        output = planned_output_path(planned, plan_path)
        if (
            generated.get("schema_version") != planned["schema_version"]
            or generated.get("plan_sha256") != plan_sha
            or generated.get("output_wav") != str(output)
            or generated.get("speaker_id") != planned["speaker_id"]
            or generated.get("target_id") != planned["target_id"]
            or generated.get("replicate_seed") != planned["synthesis"]["seed"]
            or generated.get("synthesis") != planned["synthesis"]
            or (
                source_audit_attestation is not None
                and generated.get("source_audit_attestation")
                != source_audit_attestation
            )
        ):
            raise RuntimeError(
                f"Generation provenance does not match plan for {pilot_id}"
            )
        if sha256_file(output) != generated.get("audio_sha256"):
            raise RuntimeError(f"Generated audio hash mismatch for {pilot_id}")
    return generation_by_id


def validate_kokoro_inputs(
    plan_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    plan_sha: str,
    plan_path: Path,
) -> dict[str, dict[str, Any]]:
    plan_by_target: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        target_id = str(row["target_id"])
        previous = plan_by_target.setdefault(target_id, row)
        if previous["kokoro_baseline"] != row["kokoro_baseline"]:
            raise RuntimeError(
                f"Kokoro baseline differs across replicates: {target_id}"
            )
    generated_by_target = {
        str(row.get("target_id", "")): row for row in generation_rows
    }
    if len(generated_by_target) != len(generation_rows) or set(
        generated_by_target
    ) != set(plan_by_target):
        raise RuntimeError("Kokoro generation is not a one-per-target plan bijection")
    for target_id, generated in generated_by_target.items():
        planned = plan_by_target[target_id]
        output = baseline_output_path(planned, plan_path)
        if (
            generated.get("plan_sha256") != plan_sha
            or generated.get("output_wav") != str(output)
            or generated.get("speaker_id") != planned["speaker_id"]
            or generated.get("baseline") != planned["kokoro_baseline"]
            or sha256_file(output) != generated.get("audio_sha256")
        ):
            raise RuntimeError(f"Kokoro provenance does not match plan for {target_id}")
    return generated_by_target


def score(args: argparse.Namespace) -> None:
    plan_path = args.plan.expanduser().resolve()
    generation_path = args.generation.expanduser().resolve()
    kokoro_generation_path = args.kokoro_generation.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    plan_rows = read_jsonl(plan_path)
    validate_plan(plan_rows)
    pilot_schema = str(plan_rows[0]["schema_version"])
    mlx_spec = MLX_PILOT_SPECS.get(pilot_schema)
    if mlx_spec is not None and mlx_spec["source_audit_required"]:
        if args.source_audit_report is None:
            raise RuntimeError("MLX pilot v2 QA requires --source-audit-report")
        source_audit_attestation = validate_source_audit_report(
            plan_path, plan_rows, args.source_audit_report
        )
    else:
        if args.source_audit_report is not None:
            raise RuntimeError("--source-audit-report is only valid for MLX pilot v2")
        source_audit_attestation = None
    input_paths = validate_audio_inputs(plan_rows, args.dataset_root)
    config_path = plan_path.parent / "pilot_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("plan_sha256") != sha256_file(plan_path):
        raise RuntimeError(f"Plan hash does not match {config_path}")
    generation_rows = read_jsonl(generation_path)
    plan_sha = sha256_file(plan_path)
    generated_by_id = validate_inputs(
        plan_rows,
        generation_rows,
        plan_sha,
        plan_path,
        source_audit_attestation,
    )
    kokoro_generated = validate_kokoro_inputs(
        plan_rows, read_jsonl(kokoro_generation_path), plan_sha, plan_path
    )
    plan_by_target: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        plan_by_target.setdefault(str(row["target_id"]), row)
    candidate_ids = {str(row["pilot_id"]) for row in plan_rows} | {
        f"{target_id}|kokoro" for target_id in plan_by_target
    }
    reviews = load_manual_reviews(args.manual_review)
    unknown_reviews = set(reviews) - candidate_ids
    if unknown_reviews:
        raise RuntimeError(
            f"Manual reviews contain ids outside the plan: {sorted(unknown_reviews)}"
        )
    if args.device != "mps" and not args.device.startswith("cuda:"):
        raise RuntimeError("Pilot scoring requires explicit --device mps or cuda:N")

    versions = {
        "transformers": require_package("transformers", TRANSFORMERS_VERSION),
        "sacrebleu": require_package("sacrebleu", SACREBLEU_VERSION),
        "scipy": require_package("scipy", SCIPY_VERSION),
    }
    try:
        import numpy as np
        import sacrebleu
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
            "Scoring requires torch, transformers, soundfile, scipy, numpy, and sacrebleu"
        ) from error
    if args.device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
            raise RuntimeError(
                "Unset PYTORCH_ENABLE_MPS_FALLBACK; fallback is not allowed"
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; no CPU fallback is implemented")
        model_dtype = torch.float32
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; no CPU or MPS fallback is implemented"
            )
        model_dtype = torch.bfloat16
    versions.update(
        {
            "torch": package_version("torch"),
            "soundfile": package_version("soundfile"),
            "numpy": package_version("numpy"),
            "cuda": torch.version.cuda,
            "model_dtype": str(model_dtype),
            "attention_implementation": "eager" if args.device == "mps" else "default",
        }
    )

    asr_processor = AutoProcessor.from_pretrained(
        ASR_MODEL_ID, revision=ASR_MODEL_REVISION
    )
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID,
        revision=ASR_MODEL_REVISION,
        torch_dtype=model_dtype,
        **({"attn_implementation": "eager"} if args.device == "mps" else {}),
    ).to(args.device)
    asr_model.eval()
    speaker_features = AutoFeatureExtractor.from_pretrained(
        SPEAKER_MODEL_ID, revision=SPEAKER_MODEL_REVISION
    )
    speaker_model = WavLMForXVector.from_pretrained(
        SPEAKER_MODEL_ID, revision=SPEAKER_MODEL_REVISION, torch_dtype=model_dtype
    ).to(args.device)
    speaker_model.eval()

    primary_backend = (
        "qwen_mlx"
        if plan_rows[0].get("synthesis", {}).get("package") == "mlx-audio"
        else "qwen"
    )
    generation_hashes = {
        primary_backend: sha256_file(generation_path),
        "kokoro": sha256_file(kokoro_generation_path),
    }
    candidates: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
        (
            str(plan["pilot_id"]),
            primary_backend,
            plan,
            generated_by_id[str(plan["pilot_id"])],
        )
        for plan in plan_rows
    ]
    candidates.extend(
        (
            f"{target_id}|kokoro",
            "kokoro",
            plan,
            kokoro_generated[target_id],
        )
        for target_id, plan in sorted(plan_by_target.items())
    )

    reference_embeddings: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []
    for number, (candidate_id, backend, plan, generated) in enumerate(
        candidates, start=1
    ):
        output_path = Path(str(generated["output_wav"]))
        audio, sample_rate = read_audio(output_path, sf, np)
        acoustic = acoustic_metrics(audio, sample_rate, np)
        if not acoustic["finite"] or not acoustic["nonzero"]:
            raise RuntimeError(f"Cannot run ASR on invalid waveform: {output_path}")
        audio_16k = resample(audio, sample_rate, 16_000, scipy_signal)
        asr_text = transcribe(
            audio_16k, asr_processor, asr_model, torch, args.device, "en"
        )
        auto_text = (
            transcribe(audio_16k, asr_processor, asr_model, torch, args.device, None)
            if backend == primary_backend
            else asr_text
        )
        errors, reference_words = word_error_counts(plan["text_en"], asr_text)

        speaker = str(plan["speaker_id"])
        if speaker not in reference_embeddings:
            reference_path = input_paths[plan["reference"]["audio"]["path"]]
            reference_audio, reference_rate = read_audio(reference_path, sf, np)
            reference_16k = resample(
                reference_audio, reference_rate, 16_000, scipy_signal
            )
            reference_embeddings[speaker] = speaker_embedding(
                reference_16k, speaker_features, speaker_model, torch, args.device
            )
        output_embedding = speaker_embedding(
            audio_16k, speaker_features, speaker_model, torch, args.device
        )
        cosine = float(torch.dot(reference_embeddings[speaker], output_embedding))
        duration_s = len(audio) / sample_rate
        manual = reviews.get(
            candidate_id,
            {
                "status": "pending",
                "prompt_leak": "pending",
                "notes": "",
                "review_file": None,
            },
        )
        metrics = {
            "schema_version": SCHEMA,
            "candidate_id": candidate_id,
            "backend": backend,
            "pilot_id": str(plan["pilot_id"]) if backend == primary_backend else None,
            "speaker_id": speaker,
            "gender": plan["gender"],
            "eligibility_split": plan["eligibility_split"],
            "duration_slice": plan["duration_slice"],
            "target_id": plan["target_id"],
            "replicate_seed": generated.get("replicate_seed"),
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "generation_manifest": str(
                generation_path
                if backend == primary_backend
                else kokoro_generation_path
            ),
            "generation_manifest_sha256": generation_hashes[backend],
            "output_wav": str(output_path),
            "audio_sha256": generated["audio_sha256"],
            "candidate_provenance": (
                plan["synthesis"]
                if backend == primary_backend
                else plan["kokoro_baseline"]
            ),
            "source_audit_attestation": source_audit_attestation,
            "duration_s": round(duration_s, 6),
            "duration_ratio_target_source": round(
                duration_s / float(plan["source_audio"]["duration_s"]), 6
            ),
            "acoustic": acoustic,
            "asr_transcript_en": asr_text,
            "asr_word_errors": errors,
            "asr_reference_words": reference_words,
            "asr_wer": errors / reference_words if reference_words else 0.0,
            "asr_chrf": sacrebleu.sentence_chrf(asr_text, [plan["text_en"]]).score,
            "speaker_cosine": cosine,
            "prompt_leak": prompt_leak_evidence(
                plan["reference"]["text_vi"], plan["text_en"], auto_text
            ),
            "manual_review": manual,
            "models": {
                "asr": {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION},
                "speaker": {"id": SPEAKER_MODEL_ID, "revision": SPEAKER_MODEL_REVISION},
            },
            "runtime": {**versions, "device": args.device},
        }
        metrics["failure_reasons"] = row_gate(
            metrics, qwen_candidate=backend == primary_backend
        )
        metrics["machine_and_manual_pass"] = not metrics["failure_reasons"]
        scored_rows.append(metrics)
        atomic_write_json(
            output_path.with_suffix(output_path.suffix + ".qa.json"), metrics
        )
        print(
            f"[{number}/{len(candidates)}] {candidate_id}: {metrics['failure_reasons']}",
            flush=True,
        )

    qwen_rows = [row for row in scored_rows if row["backend"] == primary_backend]
    kokoro_rows = [row for row in scored_rows if row["backend"] == "kokoro"]

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        errors = sum(int(row["asr_word_errors"]) for row in rows)
        words = sum(int(row["asr_reference_words"]) for row in rows)
        return {
            "asr_wer": errors / words if words else 0.0,
            "asr_chrf": sacrebleu.corpus_chrf(
                [str(row["asr_transcript_en"]) for row in rows],
                [
                    [
                        str(plan_by_target[str(row["target_id"])]["text_en"])
                        for row in rows
                    ]
                ],
            ).score,
        }

    qwen_aggregate = aggregate(qwen_rows)
    kokoro_aggregate = aggregate(kokoro_rows)

    def slices(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field in fields:
            output[field] = {}
            for value in sorted({str(row[field]) for row in rows}):
                group = [row for row in rows if str(row[field]) == value]
                output[field][value] = {
                    **aggregate(group),
                    "rows": len(group),
                    "failed_rows": sum(bool(row["failure_reasons"]) for row in group),
                    "speaker_cosine_median": median(
                        float(row["speaker_cosine"]) for row in group
                    ),
                }
        return output

    slice_metrics = {
        primary_backend: slices(
            qwen_rows,
            ("gender", "eligibility_split", "duration_slice", "replicate_seed"),
        ),
        "kokoro": slices(
            kokoro_rows, ("gender", "eligibility_split", "duration_slice")
        ),
    }
    qwen_cosine_by_speaker = {
        speaker: sum(
            float(row["speaker_cosine"])
            for row in qwen_rows
            if row["speaker_id"] == speaker
        )
        / sum(row["speaker_id"] == speaker for row in qwen_rows)
        for speaker in {str(row["speaker_id"]) for row in qwen_rows}
    }
    kokoro_cosine_by_speaker = {
        str(row["speaker_id"]): float(row["speaker_cosine"]) for row in kokoro_rows
    }
    timbre_wins = sum(
        qwen_cosine_by_speaker[speaker] > kokoro_cosine_by_speaker[speaker]
        for speaker in qwen_cosine_by_speaker
    )
    timbre_win_rate = timbre_wins / len(qwen_cosine_by_speaker)
    qwen_cosine_median = median(qwen_cosine_by_speaker.values())
    kokoro_cosine_median = median(kokoro_cosine_by_speaker.values())
    failed_rows = sum(bool(row["failure_reasons"]) for row in scored_rows)
    manual_complete = set(reviews) == candidate_ids
    checks = {
        "plan_generation_bijections": len(qwen_rows) == len(plan_rows)
        and len(kokoro_rows) == len(plan_by_target),
        "sealed_test_split": all(
            row["eligibility_split"] != "test" for row in scored_rows
        ),
        "two_qwen_replicates_and_one_kokoro_per_speaker": all(
            sum(row["speaker_id"] == speaker for row in qwen_rows) == 2
            and sum(row["speaker_id"] == speaker for row in kokoro_rows) == 1
            for speaker in qwen_cosine_by_speaker
        ),
        "all_rows_pass_hard_gates": failed_rows == 0,
        "qwen_wer_vs_kokoro": qwen_aggregate["asr_wer"]
        <= kokoro_aggregate["asr_wer"] + THRESHOLDS["aggregate_wer_margin_vs_kokoro"],
        "qwen_timbre_median_vs_kokoro": qwen_cosine_median > kokoro_cosine_median,
        "qwen_timbre_speaker_win_rate": timbre_win_rate
        >= THRESHOLDS["qwen_timbre_win_rate_min"],
        "manual_review_complete": manual_complete,
        "manual_review_pass": manual_complete
        and all(review["status"] == "pass" for review in reviews.values()),
        "manual_prompt_leak_clear": manual_complete
        and all(
            reviews[str(row["pilot_id"])]["prompt_leak"] == "no" for row in plan_rows
        ),
    }
    report = {
        "schema_version": SCHEMA,
        "candidate_backend": primary_backend,
        "decision": "go" if all(checks.values()) else "no_go",
        "checks": checks,
        "thresholds": THRESHOLDS,
        "rows": {primary_backend: len(qwen_rows), "kokoro": len(kokoro_rows)},
        "speakers": len({row["speaker_id"] for row in scored_rows}),
        "failed_rows": failed_rows,
        "metrics": {
            primary_backend: {
                **qwen_aggregate,
                "speaker_cosine_median": qwen_cosine_median,
                "speaker_cosine_by_speaker": qwen_cosine_by_speaker,
                "prompt_leak_matches": sum(
                    row["prompt_leak"]["reference_only_3gram_match_count"]
                    for row in qwen_rows
                ),
            },
            "kokoro": {
                **kokoro_aggregate,
                "speaker_cosine_median": kokoro_cosine_median,
                "speaker_cosine_by_speaker": kokoro_cosine_by_speaker,
            },
            "comparison": {
                "qwen_minus_kokoro_wer": qwen_aggregate["asr_wer"]
                - kokoro_aggregate["asr_wer"],
                "qwen_minus_kokoro_speaker_cosine_median": qwen_cosine_median
                - kokoro_cosine_median,
                "qwen_timbre_wins": timbre_wins,
                "qwen_timbre_win_rate": timbre_win_rate,
            },
            "slices": slice_metrics,
        },
        "models": {
            "asr": {"id": ASR_MODEL_ID, "revision": ASR_MODEL_REVISION},
            "speaker": {"id": SPEAKER_MODEL_ID, "revision": SPEAKER_MODEL_REVISION},
        },
        "runtime": {**versions, "device": args.device},
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "source_audit_attestation": source_audit_attestation,
        "generation": {
            primary_backend: {
                "path": str(generation_path),
                "sha256": generation_hashes[primary_backend],
            },
            "kokoro": {
                "path": str(kokoro_generation_path),
                "sha256": generation_hashes["kokoro"],
            },
        },
        "row_metrics": str((out_dir / "row_metrics.jsonl").resolve()),
    }
    atomic_write_jsonl(out_dir / "row_metrics.jsonl", scored_rows)
    atomic_write_json(out_dir / "gate_report.json", report)
    print(f"Decision: {report['decision']}")
    print(f"Gate report: {out_dir / 'gate_report.json'}")


def main() -> None:
    score(parse_args())


if __name__ == "__main__":
    main()
