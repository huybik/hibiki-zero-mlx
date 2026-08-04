"""Benchmark selective Qwen3-TTS MLX optimizations on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_batch import GENERATION, json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import (
    generate_group,
    load_benchmark,
    runtime_environment,
)
from qwen_mlx_efficiency import (
    QUANTIZATION_CANDIDATES,
    apply_quantization,
    capture_generated_codes,
    install_full_reference_cache,
)
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    sha256_file,
    verify_mlx_snapshot,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_efficiency_benchmark_v3"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_efficiency_row_v3"
REPORT_NAME = "2026-08-04_qwen_mlx_efficiency_v3"
SCREEN_ROWS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run-candidate")
    run.add_argument("cohort_plan", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--report-dir", type=Path, required=True)
    run.add_argument(
        "--candidate",
        required=True,
        choices=("bf16", "bf16_prefix", *QUANTIZATION_CANDIDATES),
    )
    run.add_argument("--rows", type=int, choices=(SCREEN_ROWS, 64), default=SCREEN_ROWS)
    compare = commands.add_parser("compare-exact")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def script_attestation() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def load_model() -> tuple[Any, Path, dict[str, str]]:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model as load

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    return load(root), root, verify_mlx_snapshot(root)


def set_rng(value: int, mx: Any, np: Any) -> None:
    random.seed(value)
    np.random.seed(value)
    mx.random.seed(value)


def smoke_logits(model: Any, rows: list[dict[str, Any]], group: dict[str, Any], mx: Any, np: Any) -> dict[str, Any]:
    from mlx_audio.utils import load_audio

    by_id = {row["id"]: row for row in rows}
    selected = [by_id[row_id] for row_id in group["ids"]]
    reference = selected[0]["reference"]
    ref_audio = load_audio(reference["reference_audio_path"], sample_rate=model.sample_rate)
    inputs = model._prepare_batch_inputs(
        [row["text_en"] for row in selected],
        language=GENERATION["lang_code"],
        ref_audio=ref_audio,
        ref_text=reference["reference_text_vi"],
        return_metadata=True,
    )
    cache = model.talker.make_cache()
    logits, hidden = model.talker(
        inputs.input_embeds,
        cache=cache,
        attention_mask=inputs.attention_mask,
    )
    main_logits = logits[:, -1, :].astype(mx.float32)
    first = mx.argmax(main_logits, axis=-1)[:, None]
    code_cache = model.talker.code_predictor.make_cache()
    code_input = mx.concatenate(
        [hidden[:, -1:, :], model.talker.get_input_embeddings()(first)], axis=1
    )
    code_logits, _, _ = model.talker.code_predictor(
        code_input, cache=code_cache, generation_step=0
    )
    code_logits = code_logits[:, -1, :].astype(mx.float32)
    mx.eval(main_logits, code_logits)
    return {
        "main_logits": np.asarray(main_logits),
        "code_predictor_step0_logits": np.asarray(code_logits),
        "main_top1": np.asarray(mx.argmax(main_logits, axis=-1), dtype=np.int32),
        "code_predictor_step0_top1": np.asarray(
            mx.argmax(code_logits, axis=-1), dtype=np.int32
        ),
    }


def atomic_save_npy(path: Path, value: Any, np: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def candidate_dir(root: Path, candidate: str, rows: int) -> Path:
    return root / f"{candidate}_n{rows}"


def run_candidate(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import soundfile as sf

    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    groups = config["variants"]["8"][: args.rows // 8]
    selected_ids = {row_id for group in groups for row_id in group["ids"]}
    selected_rows = [row for row in cohort if row["id"] in selected_ids]
    if len(selected_rows) != args.rows:
        raise RuntimeError("Screen/full cohort selection is not an exact bijection")

    model, model_root, snapshot = load_model()
    mx.eval(model.parameters())
    active_bf16 = int(mx.get_active_memory())
    quantization = None
    if args.candidate in QUANTIZATION_CANDIDATES:
        quantization = apply_quantization(model, args.candidate, nn, mx)
    active_candidate = int(mx.get_active_memory())
    prefix_stats = None
    if args.candidate != "bf16":
        prefix_stats = install_full_reference_cache(model, mx)

    output = candidate_dir(args.output_root.expanduser().resolve(), args.candidate, args.rows)
    output.mkdir(parents=True, exist_ok=True)
    smoke = smoke_logits(model, selected_rows, groups[0], mx, np)
    smoke_path = output / "logit_smoke.npz"
    np.savez(smoke_path, **smoke)

    records = []
    code_rows = []
    environment = runtime_environment(mx)
    for number, group in enumerate(groups, 1):
        set_rng(group["seed"], mx, np)
        group_output = output / "batches" / group["group_id"]
        captured = []
        if group_output.exists():
            record = generate_group(
                model,
                mx,
                np,
                sf,
                selected_rows,
                group,
                group_output,
                cohort_path,
                environment,
                snapshot,
                SCHEMA,
                ROW_SCHEMA,
            )
        else:
            captured, restore = capture_generated_codes(model, mx, np)
            try:
                record = generate_group(
                    model,
                    mx,
                    np,
                    sf,
                    selected_rows,
                    group,
                    group_output,
                    cohort_path,
                    environment,
                    snapshot,
                    SCHEMA,
                    ROW_SCHEMA,
                )
            finally:
                restore()
            if len(captured) != len(record["rows"]):
                raise RuntimeError(
                    f"Captured {len(captured)} code arrays for {len(record['rows'])} rows"
                )
        for index, row in enumerate(record["rows"]):
            code_path = output / "codes" / f"{row['id'].replace(':', '_')}.npy"
            if captured:
                atomic_save_npy(code_path, captured[index], np)
            if not code_path.is_file():
                raise RuntimeError(f"Missing resumable code array: {code_path}")
            codes = np.load(code_path, allow_pickle=False)
            code_rows.append(
                {
                    "id": row["id"],
                    "audio_sha256": row["audio_sha256"],
                    "code_path": str(code_path),
                    "code_sha256": sha256_file(code_path),
                    "code_shape": list(codes.shape),
                    "code_dtype": str(codes.dtype),
                }
            )
        records.append(record)
        print(
            f"[{args.candidate} {number}/{len(groups)}] "
            f"{record['rows_per_minute']:.3f} rows/min",
            flush=True,
        )

    wall = sum(record["wall_seconds"] for record in records)
    rows_out = [row for record in records for row in record["rows"]]
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report = {
        "schema_version": SCHEMA,
        "candidate": args.candidate,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_attestation(),
        "command": sys.argv,
        "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        "scope_rows": args.rows,
        "group_ids": [group["group_id"] for group in groups],
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "root": str(model_root),
            "files_sha256": snapshot,
        },
        "generation": GENERATION,
        "prefix_cache": {
            "enabled": prefix_stats is not None,
            "stats": prefix_stats,
            "allocator_clear_behavior": "unchanged; mlx-audio batch_generate owns mx.clear_cache",
        },
        "quantization": quantization,
        "wall_seconds": wall,
        "rows_per_minute": 60 * len(rows_out) / wall,
        "audio_seconds_per_wall_second": sum(row["duration_s"] for row in rows_out) / wall,
        "talker_seconds": sum(
            record["stage_timing"]["prepare_prefill_talker_seconds_reported"]
            for record in records
        ),
        "decode_seconds": sum(
            record["stage_timing"]["sequential_decode_and_yield_seconds"]
            for record in records
        ),
        "active_bf16_bytes": active_bf16,
        "active_candidate_bytes": active_candidate,
        "active_memory_reduction": 1 - active_candidate / active_bf16,
        "peak_mlx_memory_bytes": max(record["peak_mlx_memory_bytes"] for record in records),
        "peak_process_rss_bytes": peak_rss,
        "system": {
            "platform": platform.platform(),
            "load_average": os.getloadavg(),
            "thermal": subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True
            ).stdout.strip(),
            "power_measurement": "not_available_without_privileged_powermetrics; no throughput-per-watt claim",
        },
        "logit_smoke": {"path": str(smoke_path), "sha256": sha256_file(smoke_path)},
        "code_manifest": str(output / "codes.jsonl"),
        "raw_records": str(output / "raw_results.jsonl"),
    }
    atomic_write_bytes(output / "codes.jsonl", jsonl_bytes(code_rows))
    atomic_write_bytes(output / "raw_results.jsonl", jsonl_bytes(records))
    atomic_write_bytes(output / "candidate.json", json_bytes(report))
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        report_dir / f"{args.candidate}_n{args.rows}.json", json_bytes(report)
    )
    print(json.dumps({key: report[key] for key in (
        "candidate", "scope_rows", "rows_per_minute", "active_memory_reduction"
    )}, indent=2))


def compare_exact(args: argparse.Namespace) -> None:
    import numpy as np

    baseline = args.baseline.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    base_codes = {
        row["id"]: row for row in map(json.loads, (baseline / "codes.jsonl").read_text().splitlines())
    }
    cand_codes = {
        row["id"]: row for row in map(json.loads, (candidate / "codes.jsonl").read_text().splitlines())
    }
    if set(base_codes) != set(cand_codes):
        raise RuntimeError("Exact comparison ID sets differ")
    rows = []
    for row_id in sorted(base_codes):
        left = base_codes[row_id]
        right = cand_codes[row_id]
        left_codes = np.load(left["code_path"], allow_pickle=False)
        right_codes = np.load(right["code_path"], allow_pickle=False)
        rows.append(
            {
                "id": row_id,
                "code_sha256_match": left["code_sha256"] == right["code_sha256"],
                "code_array_exact": bool(np.array_equal(left_codes, right_codes)),
                "audio_sha256_match": left["audio_sha256"] == right["audio_sha256"],
            }
        )
    base_smoke = np.load(baseline / "logit_smoke.npz", allow_pickle=False)
    cand_smoke = np.load(candidate / "logit_smoke.npz", allow_pickle=False)
    smoke = {}
    for name in ("main_logits", "code_predictor_step0_logits"):
        smoke[name] = {
            "max_abs_delta": float(np.max(np.abs(base_smoke[name] - cand_smoke[name]))),
            "top1_agreement": float(np.mean(
                np.argmax(base_smoke[name], axis=-1) == np.argmax(cand_smoke[name], axis=-1)
            )),
        }
    report = {
        "schema_version": SCHEMA,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "rows": len(rows),
        "exact_code_arrays": sum(row["code_array_exact"] for row in rows),
        "exact_code_hashes": sum(row["code_sha256_match"] for row in rows),
        "exact_audio_hashes": sum(row["audio_sha256_match"] for row in rows),
        "logit_smoke": smoke,
        "row_comparison": rows,
    }
    atomic_write_bytes(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps({key: report[key] for key in (
        "rows", "exact_code_arrays", "exact_code_hashes", "exact_audio_hashes"
    )}, indent=2))


def main() -> None:
    args = parse_args()
    if args.action == "run-candidate":
        run_candidate(args)
    else:
        compare_exact(args)


if __name__ == "__main__":
    main()
