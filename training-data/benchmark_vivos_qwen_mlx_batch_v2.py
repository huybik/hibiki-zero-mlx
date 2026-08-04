"""Scale and quality-gate same-speaker Qwen3-TTS MLX batching on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_vivos_qwen_mlx_batch import (
    ACOUSTIC_GATES,
    GENERATION,
    attestation,
    json_bytes,
    jsonl_bytes,
    load_completed_scalar,
    make_groups,
    quantile_rows,
    source_row_hash,
)
from synthesize_vivos import (
    MLX_MODEL_FILES_SHA256,
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    MLX_PACKAGE_COMMIT,
    MLX_PACKAGE_VERSION,
    atomic_write_bytes,
    atomic_write_wav,
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
from synthesize_vivos_full import load_campaign

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_benchmark_v2"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_benchmark_row_v2"
PRODUCTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_full_v2"
PRODUCTION_ATTEMPT_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_batch_attempt_v2"
BENCHMARK_DIR_NAME = "vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04"
PRODUCTION_DIR_NAME = "vivos_qwen3_tts_mlx_batch_v2_full"
COHORT_SEED = "hibiki-vivos-qwen3-tts-mlx-batch-benchmark-v2"
PRODUCTION_SEED = "hibiki-vivos-qwen3-tts-mlx-batch-full-v2"
BATCH_SIZES = (8, 16, 32, 64)
COHORT_ROWS = 64
ACTIVE_MEMORY_LIMIT_BYTES = 36 * 2**30
SATURATION_MIN_GAIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-benchmark")
    prepare.add_argument("source_plan", type=Path)
    prepare.add_argument("--out-dir", type=Path, required=True)
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


def script_attestation() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def system_state() -> dict[str, Any]:
    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout.strip()
    pressure = subprocess.run(
        ["memory_pressure"], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()
    thermal = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, text=True, check=False
    )
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "swapusage": swap,
        "memory_pressure_tail": pressure[-4:],
        "pmset_therm": thermal.stdout.strip() or thermal.stderr.strip(),
        "load_average": os.getloadavg(),
    }


class RssSampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pid = str(os.getpid())
        while not self.stop.wait(0.1):
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True
            )
            value = result.stdout.strip()
            if value:
                self.samples.append(
                    {"elapsed_sample": time.monotonic(), "rss_bytes": int(value) * 1024}
                )

    def __enter__(self) -> "RssSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()

    @property
    def peak(self) -> int:
        sampled = max((row["rss_bytes"] for row in self.samples), default=0)
        return max(sampled, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def runtime_environment(mx: Any) -> dict[str, Any]:
    memory = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "physical_memory_bytes": memory,
        "active_memory_safety_limit_bytes": ACTIVE_MEMORY_LIMIT_BYTES,
        "mlx-audio": package_version("mlx-audio"),
        "mlx-audio-commit": MLX_PACKAGE_COMMIT,
        "mlx": package_version("mlx"),
        "numpy": package_version("numpy"),
        "soundfile": package_version("soundfile"),
        "mlx_default_device": str(mx.default_device()),
        "device": "mps",
    }


def prepare_benchmark(args: argparse.Namespace) -> None:
    source_plan = args.source_plan.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.name != BENCHMARK_DIR_NAME:
        raise RuntimeError(f"Benchmark directory must be named {BENCHMARK_DIR_NAME}")
    rows, source_config, plan_sha, config_sha = load_campaign(source_plan)
    scalar = load_completed_scalar(source_plan, rows, plan_sha, config_sha)
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["id"] in scalar:
            by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    eligible = sorted(
        speaker for speaker, speaker_rows in by_speaker.items() if len(speaker_rows) >= COHORT_ROWS
    )
    if not eligible:
        raise RuntimeError("No speaker has 64 immutable scalar completions")
    speaker = min(
        eligible, key=lambda value: sha256_bytes(f"{COHORT_SEED}\0{value}".encode())
    )
    selected = quantile_rows(by_speaker[speaker], COHORT_ROWS)
    selected.sort(key=lambda row: (len(str(row["text_en"])), str(row["id"])))
    cohort = []
    for row in selected:
        baseline = scalar[str(row["id"])]
        cohort.append(
            {
                "schema_version": SCHEMA,
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
    cohort_path = out_dir / "cohort_plan.jsonl"
    cohort_data = jsonl_bytes(cohort)
    config = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "script": script_attestation(),
        "command": sys.argv,
        "source_campaign": {
            "plan": {"path": str(source_plan), "sha256": plan_sha},
            "config": {
                "path": str(source_plan.parent / "campaign_config.json"),
                "sha256": config_sha,
            },
            "repository_commit": source_config["repository_commit"],
            "completed_scalar_rows_at_freeze": len(scalar),
        },
        "cohort": {
            "seed": COHORT_SEED,
            "selection": "seed-ranked >=64-completion speaker; target-character quantiles",
            "eligible_speakers": eligible,
            "speaker_id": speaker,
            "rows": len(cohort),
            "plan": {"path": str(cohort_path), "sha256": sha256_bytes(cohort_data)},
        },
        "batch_sizes": list(BATCH_SIZES),
        "variants": {
            str(size): make_groups(cohort, size, f"{COHORT_SEED}:batch")
            for size in BATCH_SIZES
        },
        "generation": GENERATION,
        "acoustic_gates": ACOUSTIC_GATES,
        "safety": {
            "active_memory_limit_bytes": ACTIVE_MEMORY_LIMIT_BYTES,
            "stop_on_failure": True,
            "stop_on_saturation_gain_below": SATURATION_MIN_GAIN,
        },
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "files_sha256": MLX_MODEL_FILES_SHA256,
        },
    }
    immutable_write(cohort_path, cohort_data)
    immutable_write(out_dir / "benchmark_config.json", json_bytes(config))
    print(f"Prepared {len(cohort)} rows for {speaker}: {cohort_path}")


def load_benchmark(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.name != "cohort_plan.jsonl" or path.parent.name != BENCHMARK_DIR_NAME:
        raise RuntimeError("Unexpected v2 cohort path")
    config = json.loads((path.parent / "benchmark_config.json").read_text())
    rows = read_jsonl(path)
    if (
        config.get("schema_version") != SCHEMA
        or config.get("script") != script_attestation()
        or config.get("cohort", {}).get("plan") != attestation(path)
        or len(rows) != COHORT_ROWS
        or len({row["id"] for row in rows}) != COHORT_ROWS
        or {row["speaker_id"] for row in rows} != {config["cohort"]["speaker_id"]}
    ):
        raise RuntimeError("V2 benchmark contract mismatch")
    for row in rows:
        sidecar = Path(row["scalar_baseline"]["sidecar"]["path"])
        scalar = Path(row["scalar_baseline"]["output_wav"])
        if (
            row.get("schema_version") != SCHEMA
            or sha256_file(sidecar) != row["scalar_baseline"]["sidecar"]["sha256"]
            or sha256_file(scalar) != row["scalar_baseline"]["audio_sha256"]
        ):
            raise RuntimeError(f"Scalar provenance changed: {row['id']}")
    return rows, config


def batch_path(root: Path, size: int, group_id: str) -> Path:
    return root / f"batch_size_{size}" / "batches" / group_id


def validate_record(
    path: Path,
    group: dict[str, Any],
    plan_sha: str,
    schema: str,
    row_schema: str,
) -> dict[str, Any]:
    record = json.loads((path / "batch.json").read_text())
    if (
        record.get("schema_version") != schema
        or record.get("row_schema_version") != row_schema
        or record.get("group") != group
        or record.get("plan", {}).get("sha256") != plan_sha
    ):
        raise RuntimeError(f"Batch record mismatch: {path}")
    for row in record["rows"]:
        if sha256_file(Path(row["output_wav"])) != row["audio_sha256"]:
            raise RuntimeError(f"Batch WAV changed: {row['output_wav']}")
    return record


def set_rng(value: int, mx: Any, np: Any) -> None:
    random.seed(value)
    np.random.seed(value)
    mx.random.seed(value)


def acoustic(audio: Any, source_duration: float, np: Any) -> tuple[dict[str, Any], list[str]]:
    finite = bool(audio.size and np.isfinite(audio).all())
    absolute = np.abs(audio) if finite else None
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if finite else None
    clipping = float(np.mean(absolute >= 0.999)) if finite else None
    silence = float(np.mean(absolute < 1e-4)) if finite else None
    ratio = float(audio.size / 24_000 / source_duration) if finite else None
    metrics = {
        "finite": finite,
        "nonzero": bool(finite and np.any(audio != 0)),
        "peak": float(np.max(absolute)) if finite else None,
        "rms": rms,
        "clipping_ratio": clipping,
        "silence_ratio": silence,
        "duration_ratio_target_source": ratio,
    }
    failures = []
    if not finite:
        failures.append("non_finite")
    if not metrics["nonzero"]:
        failures.append("all_zero")
    for name, value, low, high in (
        ("rms", rms, ACOUSTIC_GATES["rms_min"], None),
        ("clipping_ratio", clipping, None, ACOUSTIC_GATES["clipping_ratio_max"]),
        ("silence_ratio", silence, None, ACOUSTIC_GATES["silence_ratio_max"]),
        (
            "duration_ratio",
            ratio,
            ACOUSTIC_GATES["duration_ratio_min"],
            ACOUSTIC_GATES["duration_ratio_max"],
        ),
    ):
        if value is not None and ((low is not None and value < low) or (high is not None and value > high)):
            failures.append(name)
    return metrics, failures


def generate_group(
    model: Any,
    mx: Any,
    np: Any,
    sf: Any,
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    output: Path,
    plan: Path,
    environment: dict[str, Any],
    snapshot: dict[str, str],
    schema: str = SCHEMA,
    row_schema: str = ROW_SCHEMA,
) -> dict[str, Any]:
    plan_sha = sha256_file(plan)
    if output.exists():
        return validate_record(output, group, plan_sha, schema, row_schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{group['group_id']}.", dir=output.parent))
    by_id = {str(row["id"]): row for row in rows}
    selected = [by_id[row_id] for row_id in group["ids"]]
    reference = selected[0]["reference"]
    if any(row["reference"] != reference for row in selected):
        raise RuntimeError("A batch crossed a frozen reference boundary")
    try:
        set_rng(group["seed"], mx, np)
        before = system_state()
        mx.reset_peak_memory()
        started = time.monotonic()
        yields = []
        results = []
        with RssSampler() as rss:
            for result in model.batch_generate(
                texts=[row["text_en"] for row in selected],
                ref_audio=reference["reference_audio_path"],
                ref_text=reference["reference_text_vi"],
                max_tokens=GENERATION["max_tokens"],
                temperature=GENERATION["temperature"],
                top_k=GENERATION["top_k"],
                top_p=GENERATION["top_p"],
                repetition_penalty=GENERATION["repetition_penalty_requested"],
                lang_code=GENERATION["lang_code"],
                stream=False,
            ):
                mx.eval(result.audio)
                results.append(result)
                yields.append(time.monotonic() - started)
        wall = time.monotonic() - started
        after = system_state()
        result_by_index = {int(result.sequence_idx): result for result in results}
        if set(result_by_index) != set(range(len(selected))):
            raise RuntimeError(f"Missing sequence results: {sorted(result_by_index)}")
        talker_seconds = max(float(result.processing_time_seconds) for result in results)
        rows_out = []
        for index, row in enumerate(selected):
            result = result_by_index[index]
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            sample_rate = int(result.sample_rate)
            metrics, failures = acoustic(audio, float(row["source_audio"]["duration_s"]), np)
            filename = f"{str(row['id']).replace(':', '_')}.wav"
            temp_wav = temporary / "wavs" / filename
            atomic_write_wav(temp_wav, audio, sample_rate, sf)
            rows_out.append(
                {
                    "schema_version": row_schema,
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "eligibility_split": row["eligibility_split"],
                    "text_vi": row["text_vi"],
                    "text_en": row["text_en"],
                    "source_audio": row["source_audio"],
                    "reference": row["reference"],
                    "source_plan_row_sha256": row["source_plan_row_sha256"],
                    "scalar_baseline": row["scalar_baseline"],
                    "batch_group_id": group["group_id"],
                    "batch_size_requested": group["batch_size_requested"],
                    "batch_size_actual": group["batch_size_actual"],
                    "sequence_index": index,
                    "batch_seed": group["seed"],
                    "output_wav": str(output / "wavs" / filename),
                    "audio_sha256": sha256_file(temp_wav),
                    "sample_rate_hz": sample_rate,
                    "num_samples": int(audio.size),
                    "duration_s": round(audio.size / sample_rate, 6),
                    "token_count": int(result.token_count),
                    "per_sequence_token_cap": min(
                        GENERATION["max_tokens"],
                        max(75, len(model.tokenizer.encode(row["text_en"])) * 6),
                    ),
                    "processing_time_seconds_reported": float(result.processing_time_seconds),
                    "yield_elapsed_seconds": yields[index],
                    "acoustic": metrics,
                    "acoustic_failure_reasons": failures,
                }
            )
        target_audio = sum(row["duration_s"] for row in rows_out)
        record = {
            "schema_version": schema,
            "row_schema_version": row_schema,
            "plan": attestation(plan),
            "runner_script": script_attestation(),
            "group": group,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall,
            "stage_timing": {
                "prepare_prefill_talker_seconds_reported": talker_seconds,
                "sequential_decode_and_yield_seconds": wall - talker_seconds,
                "yield_elapsed_seconds": yields,
                "caveat": "reported talker time includes batch input preparation/prefill/generation; residual includes sequential decoder/yields",
            },
            "target_audio_seconds": target_audio,
            "rows_per_minute": 60 * len(rows_out) / wall,
            "audio_seconds_per_wall_second": target_audio / wall,
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "active_mlx_memory_bytes_after": int(mx.get_active_memory()),
            "cache_mlx_memory_bytes_after": int(mx.get_cache_memory()),
            "peak_process_rss_bytes": rss.peak,
            "within_active_memory_safety_limit": max(int(mx.get_peak_memory()), rss.peak)
            <= ACTIVE_MEMORY_LIMIT_BYTES,
            "system_before": before,
            "system_after": after,
            "environment": environment,
            "model_snapshot": {
                "id": MLX_MODEL_ID,
                "revision": MLX_MODEL_REVISION,
                "files_sha256": snapshot,
            },
            "generation": GENERATION,
            "rows": rows_out,
        }
        immutable_write(temporary / "batch.json", json_bytes(record))
        os.replace(temporary, output)
        return record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def write_report(
    report_dir: Path,
    cohort: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for size in BATCH_SIZES:
        subset = [r for r in records if r["group"]["batch_size_requested"] == size]
        wall = sum(r["wall_seconds"] for r in subset)
        count = sum(len(r["rows"]) for r in subset)
        audio = sum(r["target_audio_seconds"] for r in subset)
        complete = len(subset) == len(config["variants"][str(size)])
        summaries.append(
            {
                "batch_size": size,
                "complete": complete,
                "batches": len(subset),
                "rows": count,
                "wall_seconds": wall,
                "rows_per_minute": 60 * count / wall if wall else None,
                "audio_seconds_per_wall_second": audio / wall if wall else None,
                "talker_seconds": sum(
                    r["stage_timing"]["prepare_prefill_talker_seconds_reported"] for r in subset
                ),
                "sequential_decode_seconds": sum(
                    r["stage_timing"]["sequential_decode_and_yield_seconds"] for r in subset
                ),
                "peak_mlx_memory_bytes": max(
                    (r["peak_mlx_memory_bytes"] for r in subset), default=None
                ),
                "peak_process_rss_bytes": max(
                    (r["peak_process_rss_bytes"] for r in subset), default=None
                ),
                "acoustic_failures": sum(
                    bool(row["acoustic_failure_reasons"]) for r in subset for row in r["rows"]
                ),
                "within_memory_limit": all(r["within_active_memory_safety_limit"] for r in subset),
            }
        )
    complete = [row for row in summaries if row["complete"] and row["within_memory_limit"]]
    winner = max(complete, key=lambda row: row["rows_per_minute"]) if complete else None
    for index, row in enumerate(summaries):
        previous = summaries[index - 1] if index else None
        row["throughput_gain_vs_previous"] = (
            row["rows_per_minute"] / previous["rows_per_minute"] - 1
            if previous and row["rows_per_minute"] and previous["rows_per_minute"]
            else None
        )
    result = {
        "schema_version": SCHEMA,
        "decision": "pending_quality" if winner else "no_go",
        "winner_batch_size": winner["batch_size"] if winner else None,
        "selection_metric": "maximum rows/min among complete <=36 GiB variants",
        "cohort_plan": attestation(cohort),
        "script": script_attestation(),
        "environment": environment,
        "variants": summaries,
        "failures": failures,
        "remaining_gate": "complete automatic QA for B8, winner, and scalar; then manual listening",
    }
    atomic_write_bytes(report_dir / "raw_results.jsonl", jsonl_bytes(records))
    atomic_write_bytes(report_dir / "failures.jsonl", jsonl_bytes(failures))
    atomic_write_bytes(report_dir / "benchmark_summary.json", json_bytes(result))
    atomic_write_bytes(report_dir / "environment.json", json_bytes(environment))
    return result


def benchmark(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("MLX benchmark is Metal-only")
    cohort, config = load_benchmark(args.cohort_plan)
    require_package("mlx-audio", MLX_PACKAGE_VERSION, "Qwen MLX batch v2")
    require_mlx_audio_commit()
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model

    model_root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    snapshot = verify_mlx_snapshot(model_root)
    model = load_model(model_root)
    environment = runtime_environment(mx)
    records = []
    failures = []
    previous_rpm = None
    stop_reason = None
    for size in BATCH_SIZES:
        if stop_reason:
            failures.append(
                {
                    "batch_size": size,
                    "status": "not_run",
                    "reason": stop_reason,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue
        size_records = []
        for number, group in enumerate(config["variants"][str(size)], 1):
            try:
                record = generate_group(
                    model,
                    mx,
                    np,
                    sf,
                    cohort,
                    group,
                    batch_path(args.cohort_plan.parent, size, group["group_id"]),
                    args.cohort_plan.resolve(),
                    environment,
                    snapshot,
                )
                size_records.append(record)
                records.append(record)
                print(f"[B{size} {number}/{len(config['variants'][str(size)])}] {record['rows_per_minute']:.3f} rows/min", flush=True)
            except BaseException as error:
                failures.append(
                    {
                        "batch_size": size,
                        "group": group,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "system": system_state(),
                        "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
                        "completed_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                stop_reason = "lower_batch_runtime_or_memory_failure"
                break
        report_dir = args.report_dir.expanduser().resolve()
        write_report(report_dir, args.cohort_plan.resolve(), config, records, failures, environment)
        if not size_records:
            continue
        wall = sum(r["wall_seconds"] for r in size_records)
        rpm = 60 * sum(len(r["rows"]) for r in size_records) / wall
        if any(not r["within_active_memory_safety_limit"] for r in size_records):
            stop_reason = "36_gib_active_working_set_limit_exceeded"
        elif previous_rpm is not None and rpm / previous_rpm - 1 < SATURATION_MIN_GAIN:
            stop_reason = f"clear_throughput_saturation_below_{SATURATION_MIN_GAIN:.0%}_gain"
        previous_rpm = rpm
    result = write_report(
        args.report_dir.expanduser().resolve(),
        args.cohort_plan.resolve(),
        config,
        records,
        failures,
        environment,
    )
    print(f"Winner B{result['winner_batch_size']}; decision {result['decision']}")


def production_groups(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return make_groups(rows, size, PRODUCTION_SEED)


def prepare_production(args: argparse.Namespace) -> None:
    source = args.source_plan.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    if out.name != PRODUCTION_DIR_NAME or args.batch_size not in BATCH_SIZES:
        raise RuntimeError("Unexpected production namespace or batch size")
    rows, _, plan_sha, config_sha = load_campaign(source)
    groups = production_groups(rows, args.batch_size)
    plan = {
        "schema_version": PRODUCTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "script": script_attestation(),
        "command": sys.argv,
        "source_campaign": {
            "plan": {"path": str(source), "sha256": plan_sha},
            "config": {
                "path": str(source.parent / "campaign_config.json"),
                "sha256": config_sha,
            },
        },
        "batch_size": args.batch_size,
        "rows": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "groups": groups,
        "generation": GENERATION,
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "files_sha256": MLX_MODEL_FILES_SHA256,
        },
        "output_namespace": str(out),
        "atomicity": "one complete same-speaker batch directory atomically renamed",
        "resumability": "validate immutable batch record and every WAV hash before skip",
    }
    immutable_write(out / "production_plan.json", json_bytes(plan))
    print(f"Prepared {len(groups)} groups / {len(rows)} rows")


def load_production(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    if path.name != "production_plan.json" or path.parent.name != PRODUCTION_DIR_NAME:
        raise RuntimeError("Unexpected production plan")
    plan = json.loads(path.read_text())
    rows, _, plan_sha, config_sha = load_campaign(Path(plan["source_campaign"]["plan"]["path"]))
    if (
        plan.get("schema_version") != PRODUCTION_SCHEMA
        or plan.get("script") != script_attestation()
        or plan["source_campaign"]["plan"]["sha256"] != plan_sha
        or plan["source_campaign"]["config"]["sha256"] != config_sha
        or plan["groups"] != production_groups(rows, plan["batch_size"])
    ):
        raise RuntimeError("Production contract mismatch")
    return plan, rows


def generate_production(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("MLX production is Metal-only")
    plan, source_rows = load_production(args.production_plan)
    dataset_root = args.dataset_root.expanduser().resolve()
    rows = []
    for source in source_rows:
        reference = source["reference"]
        path = (dataset_root / reference["reference_audio_dataset_relative_path"]).resolve()
        if sha256_file(path) != reference["reference_audio_sha256"]:
            raise RuntimeError(f"Reference changed: {path}")
        rows.append(
            {
                **source,
                "reference": {**reference, "reference_audio_path": str(path)},
                "source_plan_row_sha256": source_row_hash(source),
                "scalar_baseline": None,
            }
        )
    require_package("mlx-audio", MLX_PACKAGE_VERSION, "Qwen MLX batch v2 production")
    require_mlx_audio_commit()
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    snapshot = verify_mlx_snapshot(root)
    model = load_model(root)
    environment = runtime_environment(mx)
    completed = []
    for number, group in enumerate(plan["groups"], 1):
        record = generate_group(
            model,
            mx,
            np,
            sf,
            rows,
            group,
            args.production_plan.parent / "attempt0" / "batches" / group["group_id"],
            args.production_plan.resolve(),
            environment,
            snapshot,
            PRODUCTION_ATTEMPT_SCHEMA,
            PRODUCTION_ATTEMPT_SCHEMA,
        )
        completed.extend(record["rows"])
        atomic_write_bytes(
            args.production_plan.parent / "generation_attempt0.jsonl", jsonl_bytes(completed)
        )
        print(f"[{number}/{len(plan['groups'])}] {group['group_id']}", flush=True)


def main() -> None:
    args = parse_args()
    actions = {
        "prepare-benchmark": prepare_benchmark,
        "benchmark": benchmark,
        "prepare-production": prepare_production,
        "generate-production": generate_production,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
