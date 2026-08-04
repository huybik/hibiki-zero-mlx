"""Benchmark deterministic active-lane Qwen3-TTS execution on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_batch import GENERATION, json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import RssSampler, runtime_environment, system_state
from qwen_mlx_compaction import generate_lanes, rng_contract, row_root_digest
from qwen_mlx_recurrent import FunctionalCodePredictor
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    atomic_write_wav,
    git_commit,
    immutable_write,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_mlx_snapshot,
)


SCHEMA = "hibiki_vivos_qwen3_tts_mlx_compaction_benchmark_v5"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_compaction_row_v5"
REPORT_NAME = "2026-08-04_qwen_mlx_compaction_v5"
EXTERNAL_NAME = "vivos_qwen3_tts_mlx_compaction_v5"
CAMPAIGN_REVISION = "hibiki-vivos-qwen3-tts-mlx-compaction-v5"
QUALITY_SEED = "hibiki-vivos-qwen3-tts-mlx-compaction-v5-quality"
GLOBAL_SEED = 2211536431
CANDIDATES = (
    "global_b8",
    "row_rng_b8",
    "row_rng_length_b8",
    "row_rng_length_compact_b8",
    "row_rng_length_compact_recurrent_b8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("full_plan", type=Path)
    prepare.add_argument("throughput_plan", type=Path)
    prepare.add_argument("length_model", type=Path)
    prepare.add_argument("--out-root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("plan", type=Path)
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--candidate", choices=CANDIDATES, required=True)
    run.add_argument("--cohort", choices=("throughput64", "quality16", "quality64"), required=True)
    run.add_argument("--attempt", type=int, choices=(0, 1), default=0)
    run.add_argument("--temperature", type=float, choices=(0.7, 0.8), default=0.8)
    run.add_argument("--retry-metrics", type=Path)
    compare = commands.add_parser("compare-exact")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify-resume")
    verify.add_argument("candidate_root", type=Path)
    verify.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def script_attestation() -> dict[str, str]:
    return attestation(Path(__file__))


def load_model() -> tuple[Any, Path, dict[str, str]]:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model as load

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    return load(root), root, verify_mlx_snapshot(root)


def prediction(row: dict[str, Any], tokenizer: Any, coefficients: list[float]) -> tuple[int, float]:
    tokens = len(tokenizer.encode(str(row["text_en"])))
    value = max(
        1.0,
        coefficients[0]
        + coefficients[1] * tokens
        + coefficients[2] * float(row["source_audio"]["duration_s"]),
    )
    return tokens, value


def ranked_speakers(rows: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["speaker_id"])] = counts.get(str(row["speaker_id"]), 0) + 1
    eligible = [speaker for speaker, count in counts.items() if count >= 8]
    return sorted(
        eligible,
        key=lambda speaker: sha256_bytes(f"{QUALITY_SEED}\0{speaker}".encode()),
    )[:8]


def quantiles(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["predicted_frames"], str(row["id"])))
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def groups(rows: list[dict[str, Any]], *, length_aware: bool) -> list[dict[str, Any]]:
    ordered = list(rows)
    if length_aware:
        ordered.sort(key=lambda row: (str(row["speaker_id"]), row["predicted_frames"], str(row["id"])))
    output = []
    for speaker in sorted({str(row["speaker_id"]) for row in ordered}):
        subset = [row for row in ordered if str(row["speaker_id"]) == speaker]
        for number in range(0, len(subset), 8):
            chunk = subset[number : number + 8]
            if len(chunk) != 8:
                raise RuntimeError(f"Cohort is not an exact B8 multiple for {speaker}")
            ids = [str(row["id"]) for row in chunk]
            output.append(
                {
                    "group_id": f"{speaker}_{number // 8:03d}_{sha256_bytes(chr(0).join(ids).encode())[:12]}",
                    "speaker_id": speaker,
                    "ids": ids,
                    "predicted_frames": [row["predicted_frames"] for row in chunk],
                }
            )
    return output


def prepare(args: argparse.Namespace) -> None:
    out = args.out_root.expanduser().resolve()
    if out.name != EXTERNAL_NAME:
        raise RuntimeError(f"Output root must be named {EXTERNAL_NAME}")
    full_path = args.full_plan.expanduser().resolve()
    throughput_path = args.throughput_plan.expanduser().resolve()
    length_path = args.length_model.expanduser().resolve()
    full = read_jsonl(full_path)
    throughput = read_jsonl(throughput_path)
    length = json.loads(length_path.read_text())
    if length.get("schema_version") != "hibiki_vivos_qwen_length_model_v5":
        raise RuntimeError("Unexpected length model")
    model, _, _ = load_model()
    coefficients = length["frozen_model"]["coefficients"]

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        tokens, predicted = prediction(row, model.tokenizer, coefficients)
        return {**row, "qwen_token_count": tokens, "predicted_frames": predicted}

    throughput_rows = [enrich(row) for row in throughput]
    full_rows = [enrich(row) for row in full]
    chosen_speakers = ranked_speakers(full_rows)
    quality64 = []
    for speaker in chosen_speakers:
        quality64.extend(quantiles([row for row in full_rows if row["speaker_id"] == speaker], 8))
    quality16_speakers = chosen_speakers[:2]
    quality16 = [row for row in quality64 if row["speaker_id"] in quality16_speakers]
    cohorts = {
        "throughput64": throughput_rows,
        "quality16": quality16,
        "quality64": quality64,
    }
    for name, rows in cohorts.items():
        immutable_write(out / f"{name}.jsonl", jsonl_bytes(rows))
    plan = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "script": script_attestation(),
        "command": sys.argv,
        "campaign_revision": CAMPAIGN_REVISION,
        "rng": rng_contract(CAMPAIGN_REVISION),
        "sources": {
            "full_plan": attestation(full_path),
            "throughput_plan": attestation(throughput_path),
            "length_model": attestation(length_path),
        },
        "cohorts": {
            name: {
                "rows": len(rows),
                "speakers": sorted({str(row["speaker_id"]) for row in rows}),
                "plan": attestation(out / f"{name}.jsonl"),
                "baseline_groups": groups(rows, length_aware=False),
                "length_groups": groups(rows, length_aware=True),
            }
            for name, rows in cohorts.items()
        },
        "generation": GENERATION,
        "model": {"id": MLX_MODEL_ID, "revision": MLX_MODEL_REVISION},
        "controls": [
            "global_b8: installed group-global RNG, original target-character grouping",
            "row_rng_b8: row-owned keys, same grouping, no compaction",
            "row_rng_length_b8: fitted-length grouping, no compaction",
            "row_rng_length_compact_b8: fitted-length grouping plus active removal",
            "row_rng_length_compact_recurrent_b8: active removal plus v4 functional predictor",
        ],
    }
    immutable_write(out / "benchmark_plan.json", json_bytes(plan))
    print(json.dumps({name: value["rows"] for name, value in plan["cohorts"].items()}, indent=2))


def load_plan(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = path.expanduser().resolve()
    plan = json.loads(path.read_text())
    if path.name != "benchmark_plan.json" or path.parent.name != EXTERNAL_NAME:
        raise RuntimeError("Unexpected v5 plan namespace")
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("script", {}).get("path") != str(Path(__file__).resolve())
    ):
        raise RuntimeError("V5 plan contract changed")
    rows = {
        name: read_jsonl(Path(record["plan"]["path"]))
        for name, record in plan["cohorts"].items()
    }
    for name, cohort_rows in rows.items():
        if attestation(Path(plan["cohorts"][name]["plan"]["path"])) != plan["cohorts"][name]["plan"]:
            raise RuntimeError(f"Changed cohort: {name}")
        if len(cohort_rows) != plan["cohorts"][name]["rows"]:
            raise RuntimeError(f"Changed cohort scope: {name}")
    return plan, rows


def set_global_rng(seed: int, mx: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def save_npy(path: Path, value: Any, np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_group(path: Path, group: dict[str, Any], candidate: str, attempt: int) -> dict[str, Any]:
    record = json.loads((path / "group.json").read_text())
    if record.get("group") != group or record.get("candidate") != candidate or record.get("attempt") != attempt:
        raise RuntimeError(f"Resume contract mismatch: {path}")
    for row in record["rows"]:
        if sha256_file(Path(row["output_wav"])) != row["audio_sha256"]:
            raise RuntimeError(f"Changed WAV: {row['id']}")
        if row.get("codes") and sha256_file(Path(row["codes"])) != row["codes_sha256"]:
            raise RuntimeError(f"Changed codes: {row['id']}")
    return record


def global_group(model: Any, rows: list[dict[str, Any]], generation: dict[str, Any], mx: Any, np: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = rows[0]["reference"]
    set_global_rng(GLOBAL_SEED, mx, np)
    started = time.monotonic()
    results = list(
        model.batch_generate(
            texts=[row["text_en"] for row in rows],
            ref_audio=reference["reference_audio_path"],
            ref_text=reference["reference_text_vi"],
            max_tokens=generation["max_tokens"],
            temperature=generation["temperature"],
            top_k=generation["top_k"],
            top_p=generation["top_p"],
            repetition_penalty=generation["repetition_penalty_requested"],
            lang_code=generation["lang_code"],
            stream=False,
        )
    )
    mx.eval(*[result.audio for result in results])
    wall = time.monotonic() - started
    ordered = {int(result.sequence_idx): result for result in results}
    token_counts = [int(ordered[index].token_count) for index in range(len(rows))]
    maximum = max(token_counts) + 1
    return [
        {"audio": np.asarray(ordered[index].audio, dtype=np.float32), "token_count": token_counts[index]}
        for index in range(len(rows))
    ], {
        "prepare_seconds": None,
        "generation_seconds": max(float(result.processing_time_seconds) for result in results),
        "decode_seconds": wall - max(float(result.processing_time_seconds) for result in results),
        "wall_seconds": wall,
        "talker_lane_steps": len(rows) * maximum,
        "useful_talker_lane_steps": sum(value + 1 for value in token_counts),
        "predictor_lane_steps": len(rows) * maximum * 15,
        "useful_predictor_lane_steps": sum(token_counts) * 15,
        "active_widths": [len(rows)] * maximum,
        "stop_reasons": ["eos"] * len(rows),
    }


def overlay_group(
    model: Any,
    rows: list[dict[str, Any]],
    generation: dict[str, Any],
    mx: Any,
    np: Any,
    *,
    attempt: int,
    compact: bool,
    adapter: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from mlx_audio.utils import load_audio

    reference = rows[0]["reference"]
    ref_audio = load_audio(reference["reference_audio_path"], sample_rate=model.sample_rate)
    prepare_started = time.monotonic()
    generated = generate_lanes(
        model,
        rows,
        ref_audio=ref_audio,
        ref_text=reference["reference_text_vi"],
        generation=generation,
        campaign_revision=CAMPAIGN_REVISION,
        attempts=[attempt] * len(rows),
        compact=compact,
        adapter=adapter,
    )
    prepare_and_generation = time.monotonic() - prepare_started
    decode_started = time.monotonic()
    decoded = []
    for codes in generated.codes:
        if not codes:
            raise RuntimeError("A row completed without codec frames")
        audio = model._decode_generated_codes(codes)
        mx.eval(audio)
        decoded.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    decode = time.monotonic() - decode_started
    prepare = max(0.0, prepare_and_generation - generated.generation_seconds)
    return [
        {"audio": audio, "codes": np.asarray(mx.stack(codes, axis=1)), "token_count": len(codes)}
        for audio, codes in zip(decoded, generated.codes)
    ], {
        "prepare_seconds": prepare,
        "generation_seconds": generated.generation_seconds,
        "prefill_seconds": generated.prefill_seconds,
        "decode_seconds": decode,
        "wall_seconds": prepare_and_generation + decode,
        "talker_lane_steps": generated.talker_lane_steps,
        "useful_talker_lane_steps": generated.useful_talker_lane_steps,
        "predictor_lane_steps": generated.predictor_lane_steps,
        "useful_predictor_lane_steps": generated.useful_predictor_lane_steps,
        "active_widths": generated.active_widths,
        "completed_order": generated.completed_order,
        "stop_reasons": generated.stop_reasons,
        "token_caps": generated.token_caps,
    }


def run_group(
    model: Any,
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    output: Path,
    generation: dict[str, Any],
    candidate: str,
    attempt: int,
    mx: Any,
    np: Any,
    sf: Any,
    adapter: Any,
) -> dict[str, Any]:
    if output.exists():
        return validate_group(output, group, candidate, attempt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{group['group_id']}.", dir=output.parent))
    try:
        before = system_state()
        mx.reset_peak_memory()
        with RssSampler() as rss:
            if candidate == "global_b8":
                values, timing = global_group(model, rows, generation, mx, np)
            else:
                values, timing = overlay_group(
                    model,
                    rows,
                    generation,
                    mx,
                    np,
                    attempt=attempt,
                    compact="compact" in candidate,
                    adapter=adapter,
                )
        after = system_state()
        output_rows = []
        for row, value in zip(rows, values):
            stem = str(row["id"]).replace(":", "_")
            wav = temporary / "wavs" / f"{stem}.wav"
            atomic_write_wav(wav, value["audio"], model.sample_rate, sf)
            codes = None
            codes_hash = None
            if "codes" in value:
                codes = temporary / "codes" / f"{stem}.npy"
                save_npy(codes, value["codes"], np)
                codes_hash = sha256_file(codes)
            output_rows.append(
                {
                    "schema_version": ROW_SCHEMA,
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "text_en": row["text_en"],
                    "source_audio": row["source_audio"],
                    "reference": row["reference"],
                    "qwen_token_count": row["qwen_token_count"],
                    "predicted_frames": row["predicted_frames"],
                    "attempt": attempt,
                    "output_wav": str(output / "wavs" / wav.name),
                    "audio_sha256": sha256_file(wav),
                    "codes": str(output / "codes" / codes.name) if codes else None,
                    "codes_sha256": codes_hash,
                    "sample_rate_hz": model.sample_rate,
                    "num_samples": int(value["audio"].size),
                    "duration_s": value["audio"].size / model.sample_rate,
                    "token_count": value["token_count"],
                }
            )
        record = {
            "schema_version": SCHEMA,
            "candidate": candidate,
            "attempt": attempt,
            "group": group,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "timing": timing,
            "rows_per_minute": 60 * len(rows) / timing["wall_seconds"],
            "audio_seconds_per_wall_second": sum(row["duration_s"] for row in output_rows) / timing["wall_seconds"],
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "peak_process_rss_bytes": rss.peak,
            "system_before": before,
            "system_after": after,
            "rows": output_rows,
        }
        immutable_write(temporary / "group.json", json_bytes(record))
        os.replace(temporary, output)
        return record
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import soundfile as sf

    plan, cohorts = load_plan(args.plan)
    rows = cohorts[args.cohort]
    by_id = {str(row["id"]): row for row in rows}
    group_key = "length_groups" if "length" in args.candidate else "baseline_groups"
    selected_groups = plan["cohorts"][args.cohort][group_key]
    retry_record = None
    if args.retry_metrics is not None:
        if args.attempt != 1 or args.candidate == "global_b8":
            raise RuntimeError("Retry metrics require a row-owned attempt 1")
        retry_path = args.retry_metrics.expanduser().resolve()
        metrics = read_jsonl(retry_path)
        retry_ids = {str(row["id"]) for row in metrics if row["failure_reasons"]}
        if not retry_ids or not retry_ids <= set(by_id):
            raise RuntimeError("Retry metrics do not select an in-cohort nonempty subset")
        rows = [row for row in rows if str(row["id"]) in retry_ids]
        by_id = {str(row["id"]): row for row in rows}
        selected_groups = []
        for speaker in sorted({str(row["speaker_id"]) for row in rows}):
            subset = sorted(
                [row for row in rows if str(row["speaker_id"]) == speaker],
                key=lambda row: (row["predicted_frames"], str(row["id"])),
            )
            for number in range(0, len(subset), 8):
                chunk = subset[number : number + 8]
                ids = [str(row["id"]) for row in chunk]
                selected_groups.append(
                    {
                        "group_id": f"retry_{speaker}_{number // 8:03d}_{sha256_bytes(chr(0).join(ids).encode())[:12]}",
                        "speaker_id": speaker,
                        "ids": ids,
                        "predicted_frames": [row["predicted_frames"] for row in chunk],
                    }
                )
        retry_record = {"metrics": attestation(retry_path), "ids": sorted(retry_ids)}
    if args.candidate == "global_b8" and args.attempt != 0:
        raise RuntimeError("The group-global control has no row-owned retry contract")
    generation = {**plan["generation"], "temperature": args.temperature}
    model, model_root, snapshot = load_model()
    mx.eval(model.parameters())
    adapter = (
        FunctionalCodePredictor(model, mx, nn, compiled=True)
        if args.candidate.endswith("recurrent_b8")
        else None
    )
    output = (
        args.out_root.expanduser().resolve()
        / (
            f"{args.candidate}_t{str(args.temperature).replace('.', '')}_{args.cohort}_attempt{args.attempt}"
            + ("_retry" if retry_record else "")
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    # One complete B8 warm-up exercises the same generation and decode path but is excluded.
    warm_rows = [by_id[row_id] for row_id in selected_groups[0]["ids"]]
    warm_started = time.monotonic()
    if args.candidate == "global_b8":
        global_group(model, warm_rows, generation, mx, np)
    else:
        overlay_group(
            model,
            warm_rows,
            generation,
            mx,
            np,
            attempt=args.attempt,
            compact="compact" in args.candidate,
            adapter=adapter,
        )
    warm_seconds = time.monotonic() - warm_started
    mx.clear_cache()
    records = []
    for number, group in enumerate(selected_groups, 1):
        selected = [by_id[row_id] for row_id in group["ids"]]
        record = run_group(
            model,
            selected,
            group,
            output / "groups" / group["group_id"],
            generation,
            args.candidate,
            args.attempt,
            mx,
            np,
            sf,
            adapter,
        )
        records.append(record)
        print(f"[{number}/{len(selected_groups)}] {record['rows_per_minute']:.3f} rows/min", flush=True)
    flat = [row for record in records for row in record["rows"]]
    wall = sum(record["timing"]["wall_seconds"] for record in records)
    generation_seconds = sum(record["timing"]["generation_seconds"] for record in records)
    decode_seconds = sum(record["timing"]["decode_seconds"] for record in records)
    talker = sum(record["timing"]["talker_lane_steps"] for record in records)
    useful_talker = sum(record["timing"]["useful_talker_lane_steps"] for record in records)
    predictor = sum(record["timing"]["predictor_lane_steps"] for record in records)
    useful_predictor = sum(record["timing"]["useful_predictor_lane_steps"] for record in records)
    candidate = {
        "schema_version": SCHEMA,
        "candidate": args.candidate,
        "cohort": args.cohort,
        "scope_rows": len(flat),
        "attempt": args.attempt,
        "temperature": args.temperature,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "script": script_attestation(),
        "plan": attestation(args.plan),
        "campaign_revision": CAMPAIGN_REVISION,
        "rng": None if args.candidate == "global_b8" else plan["rng"],
        "retry": retry_record,
        "warmup": {"scope": "one complete B8 generation+decode, excluded", "seconds": warm_seconds},
        "timing": {
            "wall_seconds": wall,
            "generation_seconds": generation_seconds,
            "decode_seconds": decode_seconds,
            "rows_per_minute": 60 * len(flat) / wall,
            "generation_rows_per_minute": 60 * len(flat) / generation_seconds,
            "audio_seconds_per_wall_second": sum(row["duration_s"] for row in flat) / wall,
        },
        "lane_accounting": {
            "talker_lane_steps": talker,
            "useful_talker_lane_steps": useful_talker,
            "talker_dead_lane_steps": talker - useful_talker,
            "talker_waste_fraction": (talker - useful_talker) / talker,
            "predictor_lane_steps": predictor,
            "useful_predictor_lane_steps": useful_predictor,
            "predictor_dead_lane_steps": predictor - useful_predictor,
            "predictor_waste_fraction": (predictor - useful_predictor) / predictor,
        },
        "memory": {
            "peak_mlx_memory_bytes": max(record["peak_mlx_memory_bytes"] for record in records),
            "peak_process_rss_bytes": max(record["peak_process_rss_bytes"] for record in records),
        },
        "thermal": [record["system_after"]["pmset_therm"] for record in records],
        "model": {"id": MLX_MODEL_ID, "revision": MLX_MODEL_REVISION, "root": str(model_root), "files_sha256": snapshot},
        "adapter": adapter.timing_report() if adapter is not None else None,
        "raw_results": str(output / "raw_results.jsonl"),
    }
    atomic_write_bytes(output / "raw_results.jsonl", jsonl_bytes(records))
    immutable_write(output / "candidate.json", json_bytes(candidate))
    print(json.dumps({"candidate": args.candidate, **candidate["timing"], **candidate["lane_accounting"]}, indent=2))


def compare_exact(args: argparse.Namespace) -> None:
    import numpy as np

    def load(root: Path) -> dict[str, dict[str, Any]]:
        records = read_jsonl(root.expanduser().resolve() / "raw_results.jsonl")
        return {str(row["id"]): row for record in records for row in record["rows"]}

    left = load(args.left)
    right = load(args.right)
    if set(left) != set(right):
        raise RuntimeError("Exactness scopes differ")
    rows = []
    for row_id in sorted(left):
        left_codes = np.load(left[row_id]["codes"], allow_pickle=False)
        right_codes = np.load(right[row_id]["codes"], allow_pickle=False)
        rows.append(
            {
                "id": row_id,
                "codes_exact": bool(np.array_equal(left_codes, right_codes)),
                "wav_sha256_exact": left[row_id]["audio_sha256"] == right[row_id]["audio_sha256"],
                "left_codes_sha256": left[row_id]["codes_sha256"],
                "right_codes_sha256": right[row_id]["codes_sha256"],
            }
        )
    report = {
        "schema_version": SCHEMA,
        "left": attestation(args.left / "candidate.json"),
        "right": attestation(args.right / "candidate.json"),
        "rows": len(rows),
        "codes_exact": sum(row["codes_exact"] for row in rows),
        "wav_sha256_exact": sum(row["wav_sha256_exact"] for row in rows),
        "all_exact": all(row["codes_exact"] and row["wav_sha256_exact"] for row in rows),
        "comparisons": rows,
    }
    immutable_write(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps({key: report[key] for key in ("rows", "codes_exact", "wav_sha256_exact", "all_exact")}, indent=2))


def verify_resume(args: argparse.Namespace) -> None:
    root = args.candidate_root.expanduser().resolve()
    candidate = json.loads((root / "candidate.json").read_text())
    records = read_jsonl(root / "raw_results.jsonl")
    verified = []
    for record in records:
        group = record["group"]
        current = validate_group(
            root / "groups" / group["group_id"],
            group,
            candidate["candidate"],
            candidate["attempt"],
        )
        ids = [str(row["id"]) for row in current["rows"]]
        if ids != group["ids"]:
            raise RuntimeError(f"Original-row mapping changed: {group['group_id']}")
        verified.extend(ids)
    if len(verified) != candidate["scope_rows"] or len(set(verified)) != len(verified):
        raise RuntimeError("Resume validation is not an exact row bijection")
    roots_are_distinct = all(
        row_root_digest(CAMPAIGN_REVISION, row_id, 0)
        != row_root_digest(CAMPAIGN_REVISION, row_id, 1)
        for row_id in verified
    )
    report = {
        "schema_version": SCHEMA,
        "candidate": attestation(root / "candidate.json"),
        "records": attestation(root / "raw_results.jsonl"),
        "groups_validated_without_generation": len(records),
        "rows_validated": len(verified),
        "original_row_bijection": True,
        "all_wav_and_code_hashes_valid": True,
        "attempt0_attempt1_root_keys_distinct": roots_are_distinct,
        "resume_decision": "skip all completed groups",
    }
    immutable_write(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    {
        "prepare": prepare,
        "run": run,
        "compare-exact": compare_exact,
        "verify-resume": verify_resume,
    }[args.action](args)


if __name__ == "__main__":
    main()
