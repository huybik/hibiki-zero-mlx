"""Benchmark and run same-speaker batched VIVOS Qwen3-TTS on Apple Silicon."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthesize_vivos import (
    MLX_MODEL_FILES_SHA256,
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    MLX_PACKAGE_COMMIT,
    MLX_PACKAGE_VERSION,
    MLX_SOURCE_MODEL_ID,
    MLX_SOURCE_MODEL_REVISION,
    MLX_V3_GENERATION_CONFIG,
    atomic_write_bytes,
    atomic_write_wav,
    canonical_json,
    git_commit,
    immutable_write,
    package_version,
    read_jsonl,
    require_mlx_audio_commit,
    require_package,
    sha256_bytes,
    sha256_file,
    verify_mlx_snapshot,
)
from synthesize_vivos_full import (
    ATTEMPT_SCHEMA as SCALAR_ATTEMPT_SCHEMA,
    SCHEMA as SCALAR_SCHEMA,
    SYNTHESIS as SCALAR_SYNTHESIS,
    load_campaign,
    output_path as scalar_output_path,
)

BENCHMARK_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_benchmark_v1"
BENCHMARK_ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_benchmark_row_v1"
PRODUCTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_full_v1"
PRODUCTION_ATTEMPT_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_attempt_v1"
BENCHMARK_DIR_NAME = "vivos_qwen3_tts_mlx_batch_v1_benchmark_2026-08-04"
PRODUCTION_DIR_NAME = "vivos_qwen3_tts_mlx_batch_v1_full"
COHORT_SEED = "hibiki-vivos-qwen3-tts-mlx-batch-benchmark-v1"
PRODUCTION_SEED = "hibiki-vivos-qwen3-tts-mlx-batch-full-v1"
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_SPEAKERS = 4
DEFAULT_ROWS_PER_SPEAKER = 16
INITIAL_SCRIPT_SHA256 = "8191098997b93e7b2e86a8c178cdfbe898c934d80fafcfec6dec516a1e57f28b"
GENERATION = {
    "api": "Model.batch_generate",
    "same_reference_per_batch": True,
    "target_length_key": "unicode_codepoints(text_en)",
    "max_tokens": MLX_V3_GENERATION_CONFIG["max_tokens"],
    "temperature": MLX_V3_GENERATION_CONFIG["temperature"],
    "top_k": MLX_V3_GENERATION_CONFIG["top_k"],
    "top_p": MLX_V3_GENERATION_CONFIG["top_p"],
    "repetition_penalty_requested": MLX_V3_GENERATION_CONFIG[
        "repetition_penalty_requested"
    ],
    "repetition_penalty_effective_icl": MLX_V3_GENERATION_CONFIG[
        "repetition_penalty_effective_icl"
    ],
    "lang_code": MLX_V3_GENERATION_CONFIG["lang_code"],
    "stream": False,
}
ACOUSTIC_GATES = {
    "rms_min": 0.0001,
    "clipping_ratio_max": 0.0001,
    "silence_ratio_max": 0.50,
    "duration_ratio_min": 0.40,
    "duration_ratio_max": 1.80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-benchmark")
    prepare.add_argument("source_plan", type=Path)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--speakers", type=int, default=DEFAULT_SPEAKERS)
    prepare.add_argument("--rows-per-speaker", type=int, default=DEFAULT_ROWS_PER_SPEAKER)
    run = commands.add_parser("benchmark")
    run.add_argument("cohort_plan", type=Path)
    run.add_argument("--report-dir", type=Path, required=True)
    run.add_argument("--device", default="mps")
    production = commands.add_parser("prepare-production")
    production.add_argument("source_plan", type=Path)
    production.add_argument("--out-dir", type=Path, required=True)
    production.add_argument("--batch-size", type=int, required=True)
    generate = commands.add_parser("generate-production")
    generate.add_argument("production_plan", type=Path)
    generate.add_argument("--dataset-root", type=Path, required=True)
    generate.add_argument("--device", default="mps")
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode()


def attestation(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def script_path() -> Path:
    return Path(__file__).resolve()


def compatible_script_attestation(record: dict[str, str]) -> bool:
    return record.get("path") == str(script_path()) and record.get("sha256") in {
        INITIAL_SCRIPT_SHA256,
        sha256_file(script_path()),
    }


def seed(namespace: str, *parts: object) -> int:
    material = "\0".join((namespace, *(str(part) for part in parts))).encode()
    return int.from_bytes(bytes.fromhex(sha256_bytes(material))[:4], "big")


def group_id(batch_size: int, speaker: str, index: int, ids: list[str]) -> str:
    digest = sha256_bytes("\0".join(ids).encode())[:12]
    return f"b{batch_size:02d}_{speaker}_{index:03d}_{digest}"


def source_row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode())


def scalar_sidecar_path(source_plan: Path, row: dict[str, Any]) -> Path:
    safe_id = str(row["id"]).replace(":", "_")
    return (
        source_plan.parent
        / "attempts"
        / "attempt0"
        / row["eligibility_split"]
        / row["speaker_id"]
        / f"{safe_id}.json"
    )


def load_completed_scalar(
    source_plan: Path, rows: list[dict[str, Any]], plan_sha: str, config_sha: str
) -> dict[str, dict[str, Any]]:
    by_id = {str(row["id"]): row for row in rows}
    completed: dict[str, dict[str, Any]] = {}
    root = source_plan.parent / "attempts" / "attempt0"
    for path in sorted(root.glob("*/*/*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        row_id = str(item.get("id", ""))
        row = by_id.get(row_id)
        if row is None or row_id in completed or path != scalar_sidecar_path(source_plan, row):
            raise RuntimeError(f"Unexpected scalar sidecar: {path}")
        output = scalar_output_path(source_plan, row, 0)
        if (
            item.get("schema_version") != SCALAR_ATTEMPT_SCHEMA
            or item.get("attempt") != 0
            or item.get("plan_sha256") != plan_sha
            or item.get("config_sha256") != config_sha
            or item.get("synthesis") != SCALAR_SYNTHESIS
            or item.get("seed") != row["seeds"]["attempt0"]
            or item.get("output_wav") != str(output)
            or not output.is_file()
            or sha256_file(output) != item.get("audio_sha256")
            or item.get("model_snapshot", {}).get("files_sha256")
            != MLX_MODEL_FILES_SHA256
        ):
            raise RuntimeError(f"Scalar baseline provenance mismatch: {path}")
        completed[row_id] = {**item, "sidecar": attestation(path)}
    return completed


def quantile_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (len(str(row["text_en"])), str(row["id"])))
    if len(ordered) < count:
        raise RuntimeError(f"Need {count} completed rows, found {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("Quantile selection produced duplicate source rows")
    return [ordered[index] for index in indices]


def make_groups(rows: list[dict[str, Any]], batch_size: int, namespace: str) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    groups: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        ordered = sorted(
            by_speaker[speaker], key=lambda row: (len(str(row["text_en"])), str(row["id"]))
        )
        for index, offset in enumerate(range(0, len(ordered), batch_size)):
            items = ordered[offset : offset + batch_size]
            ids = [str(row["id"]) for row in items]
            key = group_id(batch_size, speaker, index, ids)
            groups.append(
                {
                    "group_id": key,
                    "speaker_id": speaker,
                    "batch_size_requested": batch_size,
                    "batch_size_actual": len(items),
                    "ids": ids,
                    "target_chars": [len(str(row["text_en"])) for row in items],
                    "seed": seed(namespace, batch_size, key, *ids),
                }
            )
    return groups


def prepare_benchmark(args: argparse.Namespace) -> None:
    source_plan = args.source_plan.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.name != BENCHMARK_DIR_NAME:
        raise RuntimeError(f"Benchmark directory must be named {BENCHMARK_DIR_NAME}")
    if args.speakers < 2 or args.rows_per_speaker < max(DEFAULT_BATCH_SIZES):
        raise RuntimeError("Benchmark needs >=2 speakers and >=8 rows per speaker")
    if any(args.rows_per_speaker % size for size in DEFAULT_BATCH_SIZES):
        raise RuntimeError("--rows-per-speaker must be divisible by 1, 2, 4, and 8")
    rows, source_config, plan_sha, config_sha = load_campaign(source_plan)
    scalar = load_completed_scalar(source_plan, rows, plan_sha, config_sha)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["id"] in scalar:
            by_speaker[str(row["speaker_id"])].append(row)
    eligible = sorted(
        speaker
        for speaker, speaker_rows in by_speaker.items()
        if len(speaker_rows) >= args.rows_per_speaker
    )
    if len(eligible) < args.speakers:
        raise RuntimeError(
            f"Stopped scalar campaign has only {len(eligible)} eligible speakers; "
            f"need {args.speakers}"
        )
    selected_speakers = sorted(
        eligible, key=lambda speaker: sha256_bytes(f"{COHORT_SEED}\0{speaker}".encode())
    )[: args.speakers]
    selected: list[dict[str, Any]] = []
    for speaker in selected_speakers:
        selected.extend(quantile_rows(by_speaker[speaker], args.rows_per_speaker))
    selected.sort(key=lambda row: (row["speaker_id"], len(row["text_en"]), row["id"]))
    cohort_rows = []
    for row in selected:
        baseline = scalar[str(row["id"])]
        cohort_rows.append(
            {
                "schema_version": BENCHMARK_SCHEMA,
                "id": row["id"],
                "speaker_id": row["speaker_id"],
                "eligibility_split": row["eligibility_split"],
                "text_vi": row["text_vi"],
                "text_en": row["text_en"],
                "target_chars": len(row["text_en"]),
                "source_audio": row["source_audio"],
                "reference": row["reference"],
                "source_plan_row_sha256": source_row_hash(row),
                "scalar_baseline": {
                    "seed": baseline["seed"],
                    "output_wav": baseline["output_wav"],
                    "audio_sha256": baseline["audio_sha256"],
                    "duration_s": baseline["duration_s"],
                    "generation_seconds": baseline["generation_seconds"],
                    "sidecar": baseline["sidecar"],
                },
            }
        )
    variants = {
        str(size): make_groups(selected, size, f"{COHORT_SEED}:batch")
        for size in DEFAULT_BATCH_SIZES
    }
    cohort_path = out_dir / "cohort_plan.jsonl"
    cohort_data = jsonl_bytes(cohort_rows)
    config = {
        "schema_version": BENCHMARK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "script": {"path": str(script_path()), "sha256": sha256_file(script_path())},
        "command": sys.argv,
        "source_campaign": {
            "schema_version": SCALAR_SCHEMA,
            "plan": {"path": str(source_plan), "sha256": plan_sha},
            "config": {
                "path": str(source_plan.parent / "campaign_config.json"),
                "sha256": config_sha,
            },
            "completed_scalar_rows_at_freeze": len(scalar),
            "source_campaign_repository_commit": source_config["repository_commit"],
        },
        "cohort": {
            "seed": COHORT_SEED,
            "selection": "seed-ranked eligible speakers; target-character quantiles per speaker",
            "eligible_speakers": eligible,
            "selected_speakers": selected_speakers,
            "speakers": args.speakers,
            "rows_per_speaker": args.rows_per_speaker,
            "rows": len(cohort_rows),
            "plan": {"path": str(cohort_path), "sha256": sha256_bytes(cohort_data)},
        },
        "generation": GENERATION,
        "batch_sizes": list(DEFAULT_BATCH_SIZES),
        "variants": variants,
        "acoustic_gates": ACOUSTIC_GATES,
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "source_id": MLX_SOURCE_MODEL_ID,
            "source_revision": MLX_SOURCE_MODEL_REVISION,
            "files_sha256": MLX_MODEL_FILES_SHA256,
        },
    }
    immutable_write(cohort_path, cohort_data)
    immutable_write(out_dir / "benchmark_config.json", json_bytes(config))
    print(
        f"Prepared {len(cohort_rows)} rows across {len(selected_speakers)} speakers: "
        f"{cohort_path}"
    )


def load_benchmark(
    cohort_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    cohort_path = cohort_path.expanduser().resolve()
    if cohort_path.name != "cohort_plan.jsonl" or cohort_path.parent.name != BENCHMARK_DIR_NAME:
        raise RuntimeError("Unexpected benchmark cohort path")
    config_path = cohort_path.parent / "benchmark_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(cohort_path)
    source_plan = Path(config["source_campaign"]["plan"]["path"])
    source_rows, _, plan_sha, config_sha = load_campaign(source_plan)
    source_by_id = {str(row["id"]): row for row in source_rows}
    if (
        config.get("schema_version") != BENCHMARK_SCHEMA
        or not compatible_script_attestation(config.get("script", {}))
        or config["cohort"]["plan"] != attestation(cohort_path)
        or config["source_campaign"]["plan"]["sha256"] != plan_sha
        or config["source_campaign"]["config"]["sha256"] != config_sha
        or len(rows) != config["cohort"]["rows"]
        or len({str(row.get("id", "")) for row in rows}) != len(rows)
    ):
        raise RuntimeError("Benchmark config/cohort contract mismatch")
    for row in rows:
        source = source_by_id.get(str(row["id"]))
        baseline = row["scalar_baseline"]
        sidecar = Path(baseline["sidecar"]["path"])
        if (
            source is None
            or row.get("schema_version") != BENCHMARK_SCHEMA
            or row["source_plan_row_sha256"] != source_row_hash(source)
            or sha256_file(sidecar) != baseline["sidecar"]["sha256"]
            or sha256_file(Path(baseline["output_wav"])) != baseline["audio_sha256"]
        ):
            raise RuntimeError(f"Benchmark row provenance mismatch: {row.get('id')}")
    return rows, config, source_by_id


def runtime_environment(mx: Any) -> dict[str, Any]:
    memory_bytes = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "physical_memory_bytes": memory_bytes,
        "mlx-audio": package_version("mlx-audio"),
        "mlx-audio-commit": MLX_PACKAGE_COMMIT,
        "mlx": package_version("mlx"),
        "numpy": package_version("numpy"),
        "soundfile": package_version("soundfile"),
        "mlx_default_device": str(mx.default_device()),
        "device": "mps",
    }


def set_rng(value: int, mx: Any, np: Any) -> None:
    random.seed(value)
    np.random.seed(value)
    mx.random.seed(value)


def acoustic(audio: Any, source_duration: float, np: Any) -> tuple[dict[str, Any], list[str]]:
    finite = bool(audio.size and np.isfinite(audio).all())
    nonzero = bool(finite and np.any(audio != 0))
    peak = float(np.max(np.abs(audio))) if finite else None
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if finite else None
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if finite else None
    silence = float(np.mean(np.abs(audio) < 1e-4)) if finite else None
    metrics = {
        "finite": finite,
        "nonzero": nonzero,
        "peak": peak,
        "rms": rms,
        "clipping_ratio": clipping,
        "silence_ratio": silence,
        "duration_ratio_target_source": None,
    }
    failures: list[str] = []
    if not finite:
        failures.append("non_finite")
    if not nonzero:
        failures.append("all_zero")
    if finite:
        if rms < ACOUSTIC_GATES["rms_min"]:
            failures.append("rms_below_min")
        if clipping > ACOUSTIC_GATES["clipping_ratio_max"]:
            failures.append("clipping_ratio")
        if silence > ACOUSTIC_GATES["silence_ratio_max"]:
            failures.append("silence_ratio")
    return metrics, failures


def batch_dir(root: Path, batch_size: int, key: str) -> Path:
    return root / f"batch_size_{batch_size}" / "batches" / key


def validate_batch_dir(
    path: Path, expected_group: dict[str, Any], plan_sha: str, row_schema: str
) -> dict[str, Any]:
    record_path = path / "batch.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("group") != expected_group
        or record.get("plan", {}).get("sha256") != plan_sha
        or record.get("row_schema_version") != row_schema
        or len(record.get("rows", [])) != len(expected_group["ids"])
        or (
            record.get("runner_script") is not None
            and not compatible_script_attestation(record["runner_script"])
        )
    ):
        raise RuntimeError(f"Batch provenance mismatch: {path}")
    for row in record["rows"]:
        output = Path(str(row["output_wav"]))
        if not output.is_file() or sha256_file(output) != row.get("audio_sha256"):
            raise RuntimeError(f"Batch output changed: {output}")
    return record


def generate_group(
    *,
    model: Any,
    mx: Any,
    np: Any,
    sf: Any,
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    output: Path,
    plan_path: Path,
    plan_sha: str,
    schema: str,
    row_schema: str,
    environment: dict[str, Any],
    model_snapshot: dict[str, str],
) -> dict[str, Any]:
    if output.exists():
        return validate_batch_dir(output, group, plan_sha, row_schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{group['group_id']}.", dir=output.parent))
    by_id = {str(row["id"]): row for row in rows}
    selected = [by_id[row_id] for row_id in group["ids"]]
    reference = selected[0]["reference"]
    if any(row["speaker_id"] != group["speaker_id"] or row["reference"] != reference for row in selected):
        raise RuntimeError(f"Batch is not one frozen speaker/reference: {group['group_id']}")
    try:
        set_rng(int(group["seed"]), mx, np)
        mx.reset_peak_memory()
        started = time.monotonic()
        results = list(
            model.batch_generate(
                texts=[str(row["text_en"]) for row in selected],
                ref_audio=str(reference["reference_audio_path"]),
                ref_text=str(reference["reference_text_vi"]),
                max_tokens=GENERATION["max_tokens"],
                temperature=GENERATION["temperature"],
                top_k=GENERATION["top_k"],
                top_p=GENERATION["top_p"],
                repetition_penalty=GENERATION["repetition_penalty_requested"],
                lang_code=GENERATION["lang_code"],
                stream=False,
            )
        )
        wall_seconds = time.monotonic() - started
        result_by_index = {int(result.sequence_idx): result for result in results}
        if set(result_by_index) != set(range(len(selected))):
            raise RuntimeError(
                f"Expected sequence indices 0..{len(selected) - 1}, found {sorted(result_by_index)}"
            )
        row_records: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            generated = result_by_index[index]
            mx.eval(generated.audio)
            audio = np.asarray(generated.audio, dtype=np.float32).reshape(-1)
            sample_rate = int(generated.sample_rate)
            metrics, failures = acoustic(audio, float(row["source_audio"]["duration_s"]), np)
            duration = audio.size / sample_rate
            ratio = duration / float(row["source_audio"]["duration_s"])
            metrics["duration_ratio_target_source"] = ratio
            if ratio < ACOUSTIC_GATES["duration_ratio_min"]:
                failures.append("duration_ratio_below_min")
            if ratio > ACOUSTIC_GATES["duration_ratio_max"]:
                failures.append("duration_ratio_above_max")
            filename = f"{str(row['id']).replace(':', '_')}.wav"
            temporary_wav = temporary / "wavs" / filename
            atomic_write_wav(temporary_wav, audio, sample_rate, sf)
            final_wav = output / "wavs" / filename
            row_records.append(
                {
                    "schema_version": row_schema,
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "eligibility_split": row["eligibility_split"],
                    "batch_group_id": group["group_id"],
                    "batch_size_requested": group["batch_size_requested"],
                    "batch_size_actual": group["batch_size_actual"],
                    "sequence_index": index,
                    "batch_seed": group["seed"],
                    "text_vi": row["text_vi"],
                    "text_en": row["text_en"],
                    "source_audio": row["source_audio"],
                    "reference": row["reference"],
                    "source_plan_row_sha256": row["source_plan_row_sha256"],
                    "output_wav": str(final_wav),
                    "audio_sha256": sha256_file(temporary_wav),
                    "sample_rate_hz": sample_rate,
                    "num_samples": int(audio.size),
                    "duration_s": round(duration, 6),
                    "token_count": int(generated.token_count),
                    "batch_processing_time_seconds_reported": float(
                        generated.processing_time_seconds
                    ),
                    "acoustic": metrics,
                    "acoustic_failure_reasons": failures,
                    "quality_inputs": {
                        "reference_text_en": row["text_en"],
                        "reference_text_vi": row["reference"]["reference_text_vi"],
                        "reference_audio_sha256": row["reference"]["reference_audio_sha256"],
                        "source_audio_sha256": row["source_audio"]["sha256"],
                    },
                }
            )
        record = {
            "schema_version": schema,
            "row_schema_version": row_schema,
            "plan": {"path": str(plan_path), "sha256": plan_sha},
            "runner_script": {
                "path": str(script_path()),
                "sha256": sha256_file(script_path()),
            },
            "group": group,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(wall_seconds, 6),
            "target_audio_seconds": round(sum(row["duration_s"] for row in row_records), 6),
            "rows_per_minute": 60 * len(row_records) / wall_seconds,
            "audio_seconds_per_wall_second": sum(row["duration_s"] for row in row_records)
            / wall_seconds,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "active_memory_bytes_after": int(mx.get_active_memory()),
            "cache_memory_bytes_after": int(mx.get_cache_memory()),
            "environment": environment,
            "model_snapshot": {
                "id": MLX_MODEL_ID,
                "revision": MLX_MODEL_REVISION,
                "files_sha256": model_snapshot,
            },
            "generation": GENERATION,
            "rows": row_records,
        }
        immutable_write(temporary / "batch.json", json_bytes(record))
        os.replace(temporary, output)
        return record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prime_references(model: Any, rows: list[dict[str, Any]], mx: Any, np: Any) -> list[dict[str, Any]]:
    shortest: dict[str, dict[str, Any]] = {}
    for row in rows:
        speaker = str(row["speaker_id"])
        current = shortest.get(speaker)
        if current is None or (len(row["text_en"]), row["id"]) < (
            len(current["text_en"]),
            current["id"],
        ):
            shortest[speaker] = row
    records = []
    for speaker in sorted(shortest):
        row = shortest[speaker]
        value = seed(COHORT_SEED, "reference-prime", speaker, row["id"])
        set_rng(value, mx, np)
        started = time.monotonic()
        results = list(
            model.batch_generate(
                texts=[row["text_en"]],
                ref_audio=row["reference"]["reference_audio_path"],
                ref_text=row["reference"]["reference_text_vi"],
                max_tokens=GENERATION["max_tokens"],
                temperature=GENERATION["temperature"],
                top_k=GENERATION["top_k"],
                top_p=GENERATION["top_p"],
                repetition_penalty=GENERATION["repetition_penalty_requested"],
                lang_code=GENERATION["lang_code"],
                stream=False,
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"Reference prime failed for {speaker}")
        mx.eval(results[0].audio)
        records.append(
            {
                "speaker_id": speaker,
                "id": row["id"],
                "seed": value,
                "wall_seconds": round(time.monotonic() - started, 6),
                "purpose": "populate pinned mlx-audio ICL reference cache before timed variants",
            }
        )
    return records


def write_report(
    report_dir: Path,
    cohort_path: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    primes: list[dict[str, Any]],
    environment: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for batch_size in DEFAULT_BATCH_SIZES:
        subset = [record for record in records if record["group"]["batch_size_requested"] == batch_size]
        expected = len(config["variants"][str(batch_size)])
        complete = len(subset) == expected
        wall = sum(record["wall_seconds"] for record in subset)
        audio_seconds = sum(record["target_audio_seconds"] for record in subset)
        rows = sum(len(record["rows"]) for record in subset)
        acoustic_failures = sum(
            bool(row["acoustic_failure_reasons"])
            for record in subset
            for row in record["rows"]
        )
        summaries.append(
            {
                "batch_size": batch_size,
                "complete": complete,
                "batches": len(subset),
                "rows": rows,
                "wall_seconds": wall,
                "target_audio_seconds": audio_seconds,
                "rows_per_minute": 60 * rows / wall if wall else None,
                "audio_seconds_per_wall_second": audio_seconds / wall if wall else None,
                "peak_memory_bytes": max((record["peak_memory_bytes"] for record in subset), default=None),
                "acoustic_gate_failures": acoustic_failures,
                "candidate": complete and acoustic_failures == 0,
            }
        )
    candidates = [row for row in summaries if row["candidate"]]
    winner = max(candidates, key=lambda row: row["audio_seconds_per_wall_second"]) if candidates else None
    cohort_rows = read_jsonl(cohort_path)
    scalar_wall = sum(float(row["scalar_baseline"]["generation_seconds"]) for row in cohort_rows)
    scalar_audio = sum(float(row["scalar_baseline"]["duration_s"]) for row in cohort_rows)
    result = {
        "schema_version": BENCHMARK_SCHEMA,
        "decision": "provisional_batch_candidate" if winner else "no_go",
        "winner_batch_size": winner["batch_size"] if winner else None,
        "selection_metric": "maximum audio_seconds_per_wall_second among complete acoustic-pass variants",
        "quality_scope": "waveform/acoustic only; English ASR WER and speaker similarity still required",
        "source_plan": config["source_campaign"]["plan"],
        "cohort_plan": attestation(cohort_path),
        "script": config["script"],
        "environment": environment,
        "commands": {"prepare": config["command"], "benchmark": command},
        "reference_primes": primes,
        "scalar_baseline": {
            "rows": len(cohort_rows),
            "sum_per_row_generation_seconds": scalar_wall,
            "target_audio_seconds": scalar_audio,
            "rows_per_minute": 60 * len(cohort_rows) / scalar_wall,
            "audio_seconds_per_generation_second": scalar_audio / scalar_wall,
            "caveat": "historical scalar sidecar timings, not a fresh controlled wall-time run",
        },
        "variants": summaries,
        "failures": failures,
    }
    immutable_write(report_dir / "raw_results.jsonl", jsonl_bytes(records))
    immutable_write(report_dir / "failures.jsonl", jsonl_bytes(failures))
    immutable_write(report_dir / "environment.json", json_bytes(environment))
    immutable_write(report_dir / "benchmark_summary.json", json_bytes(result))
    csv_path = report_dir / "translations.csv"
    if not csv_path.exists():
        fieldnames = [
            "sample",
            "batch_size",
            "audio_file",
            "duration_s",
            "transcript_vi",
            "reference_en",
            "model_output_sha256",
            "scalar_baseline_sha256",
            "acoustic_failures",
        ]
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        cohort_by_id = {str(row["id"]): row for row in cohort_rows}
        for record in records:
            for row in record["rows"]:
                cohort = cohort_by_id[str(row["id"])]
                writer.writerow(
                    {
                        "sample": row["id"],
                        "batch_size": row["batch_size_requested"],
                        "audio_file": row["output_wav"],
                        "duration_s": row["duration_s"],
                        "transcript_vi": row["text_vi"],
                        "reference_en": row["text_en"],
                        "model_output_sha256": row["audio_sha256"],
                        "scalar_baseline_sha256": cohort["scalar_baseline"]["audio_sha256"],
                        "acoustic_failures": ",".join(row["acoustic_failure_reasons"]),
                    }
                )
        immutable_write(csv_path, buffer.getvalue().encode())
    rows_md = [
        "| Batch | Rows | Wall s | Rows/min | Audio s / wall s | Peak GB | Acoustic failures |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        rows_md.append(
            "| {batch_size} | {rows} | {wall_seconds:.3f} | {rpm} | {rt} | {peak} | "
            "{acoustic_gate_failures} |".format(
                **summary,
                rpm=(f"{summary['rows_per_minute']:.3f}" if summary["rows_per_minute"] else "—"),
                rt=(
                    f"{summary['audio_seconds_per_wall_second']:.3f}"
                    if summary["audio_seconds_per_wall_second"]
                    else "—"
                ),
                peak=(f"{summary['peak_memory_bytes'] / 1e9:.3f}" if summary["peak_memory_bytes"] else "—"),
            )
        )
    winner_text = f"Batch {winner['batch_size']}" if winner else "None"
    metrics = f"""# Qwen3-TTS MLX same-speaker batch benchmark

Date: 2026-08-04 · Machine: Apple Silicon `{environment['machine']}` with {environment['physical_memory_bytes'] / 2**30:.0f} GiB unified memory · Cohort: {len(cohort_rows)} frozen rows / {config['cohort']['speakers']} speakers.

{chr(10).join(rows_md)}

Provisional throughput winner: **{winner_text}**. This is not a corpus-generation quality approval: all generated WAVs, hashes, references, texts, source durations, and acoustic checks are preserved for the existing English-ASR WER and speaker-similarity QA. The scalar figures in `benchmark_summary.json` reuse immutable stopped-campaign sidecar timings and are not a fresh controlled wall-time baseline.

The timed path uses installed `Model.batch_generate`, one frozen reference per same-speaker batch, target-character sorting, fixed group order, and one SHA-256-derived RNG seed per batch. Reference prompts were primed before timing; first-shape compilation remains included. Raw batch records, peak MLX memory, output hashes, environment, commands, and failures are archived beside this report.
"""
    immutable_write(report_dir / "metrics.md", metrics.encode())
    return result


def benchmark(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("MLX benchmark is Apple-Metal-only; use --device mps")
    cohort_path = args.cohort_plan.expanduser().resolve()
    rows, config, _ = load_benchmark(cohort_path)
    require_package("mlx-audio", MLX_PACKAGE_VERSION, "Qwen MLX batch benchmark")
    require_mlx_audio_commit()
    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model
    except ImportError as error:
        raise RuntimeError("Pinned MLX benchmark environment is incomplete") from error
    for row in rows:
        reference = Path(str(row["reference"]["reference_audio_path"]))
        if not reference.is_file() or sha256_file(reference) != row["reference"]["reference_audio_sha256"]:
            raise RuntimeError(f"Frozen reference changed: {reference}")
    model_root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    snapshot = verify_mlx_snapshot(model_root)
    model = load_model(model_root)
    environment = runtime_environment(mx)
    primes = prime_references(model, rows, mx, np)
    plan_sha = sha256_file(cohort_path)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stop_higher = False
    for batch_size in DEFAULT_BATCH_SIZES:
        if stop_higher:
            failures.append(
                {
                    "batch_size": batch_size,
                    "status": "not_run_after_lower_batch_failure",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue
        for number, group in enumerate(config["variants"][str(batch_size)], 1):
            path = batch_dir(cohort_path.parent, batch_size, group["group_id"])
            try:
                record = generate_group(
                    model=model,
                    mx=mx,
                    np=np,
                    sf=sf,
                    rows=rows,
                    group=group,
                    output=path,
                    plan_path=cohort_path,
                    plan_sha=plan_sha,
                    schema=BENCHMARK_SCHEMA,
                    row_schema=BENCHMARK_ROW_SCHEMA,
                    environment=environment,
                    model_snapshot=snapshot,
                )
                records.append(record)
                print(
                    f"[B{batch_size} {number}/{len(config['variants'][str(batch_size)])}] "
                    f"{group['group_id']}: {record['audio_seconds_per_wall_second']:.3f}x audio",
                    flush=True,
                )
            except BaseException as error:
                failures.append(
                    {
                        "batch_size": batch_size,
                        "group": group,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                        "peak_memory_bytes": int(mx.get_peak_memory()),
                        "active_memory_bytes": int(mx.get_active_memory()),
                        "cache_memory_bytes": int(mx.get_cache_memory()),
                    }
                )
                stop_higher = True
                break
    report = write_report(
        args.report_dir.expanduser().resolve(),
        cohort_path,
        config,
        records,
        failures,
        primes,
        environment,
        sys.argv,
    )
    print(
        f"Benchmark decision {report['decision']}; winner batch "
        f"{report['winner_batch_size']}"
    )


def prepare_production(args: argparse.Namespace) -> None:
    source_plan = args.source_plan.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.name != PRODUCTION_DIR_NAME:
        raise RuntimeError(f"Production directory must be named {PRODUCTION_DIR_NAME}")
    if args.batch_size not in DEFAULT_BATCH_SIZES:
        raise RuntimeError(f"Production batch size must be one of {DEFAULT_BATCH_SIZES}")
    rows, _, plan_sha, config_sha = load_campaign(source_plan)
    groups = make_groups(rows, args.batch_size, PRODUCTION_SEED)
    plan = {
        "schema_version": PRODUCTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "script": {"path": str(script_path()), "sha256": sha256_file(script_path())},
        "command": sys.argv,
        "source_campaign": {
            "schema_version": SCALAR_SCHEMA,
            "plan": {"path": str(source_plan), "sha256": plan_sha},
            "config": {
                "path": str(source_plan.parent / "campaign_config.json"),
                "sha256": config_sha,
            },
        },
        "batch_size": args.batch_size,
        "order": "speaker_id, unicode_codepoints(text_en), id",
        "rng_contract": "uint32_be(SHA256(namespace + NUL + batch_size + NUL + group_id + NUL + ordered ids)[:4])",
        "seed_namespace": PRODUCTION_SEED,
        "rows": len(rows),
        "speakers": len({str(row["speaker_id"]) for row in rows}),
        "groups": groups,
        "generation": GENERATION,
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "files_sha256": MLX_MODEL_FILES_SHA256,
        },
        "output_namespace": str(out_dir),
        "atomicity": "one complete batch directory atomically renamed from same-filesystem staging",
    }
    plan_path = out_dir / "production_plan.json"
    immutable_write(plan_path, json_bytes(plan))
    print(f"Prepared {len(groups)} production batches / {len(rows)} rows: {plan_path}")


def load_production(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = plan_path.expanduser().resolve()
    if plan_path.name != "production_plan.json" or plan_path.parent.name != PRODUCTION_DIR_NAME:
        raise RuntimeError("Unexpected production plan path")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_path = Path(plan["source_campaign"]["plan"]["path"])
    rows, _, source_sha, config_sha = load_campaign(source_path)
    ids = [row_id for group in plan["groups"] for row_id in group["ids"]]
    if (
        plan.get("schema_version") != PRODUCTION_SCHEMA
        or not compatible_script_attestation(plan.get("script", {}))
        or plan["source_campaign"]["plan"]["sha256"] != source_sha
        or plan["source_campaign"]["config"]["sha256"] != config_sha
        or len(ids) != len(rows)
        or len(set(ids)) != len(rows)
        or set(ids) != {str(row["id"]) for row in rows}
        or plan["groups"] != make_groups(rows, int(plan["batch_size"]), PRODUCTION_SEED)
    ):
        raise RuntimeError("Production plan contract mismatch")
    return plan, rows


def generate_production(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("MLX production is Apple-Metal-only; use --device mps")
    plan_path = args.production_plan.expanduser().resolve()
    plan, source_rows = load_production(plan_path)
    dataset_root = args.dataset_root.expanduser().resolve()
    row_by_id = {str(row["id"]): row for row in source_rows}
    rows = []
    for source in source_rows:
        reference = source["reference"]
        reference_path = (dataset_root / reference["reference_audio_dataset_relative_path"]).resolve()
        if (
            not reference_path.is_relative_to(dataset_root)
            or not reference_path.is_file()
            or sha256_file(reference_path) != reference["reference_audio_sha256"]
        ):
            raise RuntimeError(f"Frozen production reference changed: {reference_path}")
        copy = dict(source)
        copy["reference"] = {**reference, "reference_audio_path": str(reference_path)}
        copy["source_plan_row_sha256"] = source_row_hash(source)
        rows.append(copy)
    require_package("mlx-audio", MLX_PACKAGE_VERSION, "Qwen MLX batch production")
    require_mlx_audio_commit()
    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model
    except ImportError as error:
        raise RuntimeError("Pinned MLX production environment is incomplete") from error
    model_root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    snapshot = verify_mlx_snapshot(model_root)
    model = load_model(model_root)
    environment = runtime_environment(mx)
    plan_sha = sha256_file(plan_path)
    completed: list[dict[str, Any]] = []
    for number, group in enumerate(plan["groups"], 1):
        path = plan_path.parent / "attempt0" / "batches" / group["group_id"]
        record = generate_group(
            model=model,
            mx=mx,
            np=np,
            sf=sf,
            rows=rows,
            group=group,
            output=path,
            plan_path=plan_path,
            plan_sha=plan_sha,
            schema=PRODUCTION_ATTEMPT_SCHEMA,
            row_schema=PRODUCTION_ATTEMPT_SCHEMA,
            environment=environment,
            model_snapshot=snapshot,
        )
        completed.extend(record["rows"])
        atomic_write_bytes(plan_path.parent / "generation_attempt0.jsonl", jsonl_bytes(completed))
        print(f"[{number}/{len(plan['groups'])}] {group['group_id']}", flush=True)
    if set(row_by_id) != {str(row["id"]) for row in completed}:
        raise RuntimeError("Production generation did not cover the exact source plan")


def main() -> None:
    args = parse_args()
    if args.action == "prepare-benchmark":
        prepare_benchmark(args)
    elif args.action == "benchmark":
        benchmark(args)
    elif args.action == "prepare-production":
        prepare_production(args)
    else:
        generate_production(args)


if __name__ == "__main__":
    main()
