"""Benchmark functional Qwen3-TTS recurrent compilation on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_batch import GENERATION, json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import (
    generate_group,
    load_benchmark,
    runtime_environment,
)
from qwen_mlx_efficiency import capture_generated_codes
from qwen_mlx_recurrent import (
    FunctionalCodePredictor,
    install_functional_code_predictor,
    install_talker_layer_split,
)
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    sha256_file,
    verify_mlx_snapshot,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_recurrent_benchmark_v4"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_recurrent_row_v4"
REPORT_NAME = "2026-08-04_qwen_mlx_recurrent_v4"
SCREEN_ROWS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    exact = commands.add_parser("exactness")
    exact.add_argument("cohort_plan", type=Path)
    exact.add_argument("--out", type=Path, required=True)
    probe = commands.add_parser("probe-shapeless")
    probe.add_argument("cohort_plan", type=Path)
    probe.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run-candidate")
    run.add_argument("cohort_plan", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--report-dir", type=Path, required=True)
    run.add_argument(
        "--candidate",
        choices=("eager", "recurrent_compiled", "recurrent_compiled_talker_split"),
        required=True,
    )
    run.add_argument("--rows", type=int, choices=(SCREEN_ROWS, 64), default=SCREEN_ROWS)
    compare = commands.add_parser("compare-exact")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
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


def set_rng(seed: int, mx: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def array_hash(value: Any, np: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_prefill(
    model: Any, selected: list[dict[str, Any]], mx: Any
) -> tuple[Any, Any, float]:
    from mlx_audio.utils import load_audio

    reference = selected[0]["reference"]
    ref_audio = load_audio(
        reference["reference_audio_path"], sample_rate=model.sample_rate
    )
    inputs = model._prepare_batch_inputs(
        [row["text_en"] for row in selected],
        language=GENERATION["lang_code"],
        ref_audio=ref_audio,
        ref_text=reference["reference_text_vi"],
        return_metadata=True,
    )
    cache = model.talker.make_cache()
    started = time.monotonic()
    logits, hidden = model.talker(
        inputs.input_embeds,
        cache=cache,
        attention_mask=inputs.attention_mask,
    )
    mx.eval(logits, hidden)
    return hidden[:, -1:, :], mx.argmax(logits[:, -1, :], axis=-1)[:, None], time.monotonic() - started


def installed_trace(model: Any, hidden: Any, first: Any, mx: Any, np: Any) -> list[dict[str, Any]]:
    cache = model.talker.code_predictor.make_cache()
    token = first
    rows = []
    for index in range(model.config.talker_config.num_code_groups - 1):
        if index == 0:
            embed = model.talker.get_input_embeddings()(token)
            inputs = mx.concatenate([hidden, embed], axis=1)
        else:
            inputs = model.talker.code_predictor.codec_embedding[index - 1](token)
        started = time.monotonic()
        logits, _, _ = model.talker.code_predictor(
            inputs, cache=cache, generation_step=index
        )
        token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        live = [value for item in cache for value in (item.keys, item.values)]
        mx.eval(logits, token, *live)
        elapsed = time.monotonic() - started
        rows.append(
            {
                "logits": np.asarray(logits),
                "token": np.asarray(token),
                "kv": [
                    np.asarray(value[..., : item.offset, :])
                    for item in cache
                    for value in (item.keys, item.values)
                ],
                "seconds": elapsed,
            }
        )
    return rows


def functional_trace(
    adapter: FunctionalCodePredictor,
    hidden: Any,
    first: Any,
    mx: Any,
    np: Any,
) -> list[dict[str, Any]]:
    token = first
    flat_kv: tuple[Any, ...] = ()
    rows = []
    for index in range(adapter.num_steps):
        before = adapter.timings[index].seconds
        logits, flat_kv = adapter.run_step(
            index, hidden, token, flat_kv, evaluate=True
        )
        token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.eval(token)
        rows.append(
            {
                "logits": np.asarray(logits),
                "token": np.asarray(token),
                "kv": [np.asarray(value) for value in flat_kv],
                "seconds": adapter.timings[index].seconds - before,
            }
        )
    return rows


def delta(left: Any, right: Any, np: Any) -> float:
    return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def talker_split_exactness(
    model: Any,
    selected: list[dict[str, Any]],
    mx: Any,
    np: Any,
) -> dict[str, Any]:
    from mlx_audio.utils import load_audio

    reference = selected[0]["reference"]
    ref_audio = load_audio(
        reference["reference_audio_path"], sample_rate=model.sample_rate
    )
    inputs = model._prepare_batch_inputs(
        [row["text_en"] for row in selected],
        language=GENERATION["lang_code"],
        ref_audio=ref_audio,
        ref_text=reference["reference_text_vi"],
        return_metadata=True,
    )
    base = model.talker.make_cache()
    logits, _ = model.talker(
        inputs.input_embeds, cache=base, attention_mask=inputs.attention_mask
    )
    mx.eval(logits, *[value for item in base for value in (item.keys, item.values)])

    def clone_cache() -> list[Any]:
        cloned = model.talker.make_cache()
        for target, source in zip(cloned, base):
            target.keys = source.keys
            target.values = source.values
            target.offset = source.offset
        return cloned

    first = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    initial = inputs.tts_pad_embed + model.talker.get_input_embeddings()(first)
    eager_cache = clone_cache()
    split_cache = clone_cache()
    eager_rows = []
    mask = mx.concatenate(
        [inputs.attention_mask, mx.ones((len(selected), 1))], axis=1
    )
    current = initial
    for _ in range(5):
        started = time.monotonic()
        step_logits, hidden = model.talker(
            current, cache=eager_cache, attention_mask=mask
        )
        mx.eval(
            step_logits,
            hidden,
            *[value for item in eager_cache for value in (item.keys, item.values)],
        )
        eager_rows.append(
            {
                "logits": np.asarray(step_logits),
                "hidden": np.asarray(hidden),
                "kv": [
                    np.asarray(value[..., : item.offset, :])
                    for item in eager_cache
                    for value in (item.keys, item.values)
                ],
                "seconds": time.monotonic() - started,
            }
        )
        current = hidden
        mask = mx.concatenate([mask, mx.ones((len(selected), 1))], axis=1)

    wrappers = install_talker_layer_split(model, mx)
    split_rows = []
    mask = mx.concatenate(
        [inputs.attention_mask, mx.ones((len(selected), 1))], axis=1
    )
    current = initial
    for _ in range(5):
        started = time.monotonic()
        step_logits, hidden = model.talker(
            current, cache=split_cache, attention_mask=mask
        )
        mx.eval(
            step_logits,
            hidden,
            *[value for item in split_cache for value in (item.keys, item.values)],
        )
        split_rows.append(
            {
                "logits": np.asarray(step_logits),
                "hidden": np.asarray(hidden),
                "kv": [
                    np.asarray(value[..., : item.offset, :])
                    for item in split_cache
                    for value in (item.keys, item.values)
                ],
                "seconds": time.monotonic() - started,
            }
        )
        current = hidden
        mask = mx.concatenate([mask, mx.ones((len(selected), 1))], axis=1)

    rows = []
    for index, (eager, split) in enumerate(zip(eager_rows, split_rows)):
        rows.append(
            {
                "step": index,
                "eager_seconds": eager["seconds"],
                "split_seconds": split["seconds"],
                "logit_max_abs_delta": delta(eager["logits"], split["logits"], np),
                "hidden_max_abs_delta": delta(eager["hidden"], split["hidden"], np),
                "cache_max_abs_delta": max(
                    delta(left, right, np)
                    for left, right in zip(eager["kv"], split["kv"])
                ),
            }
        )
    return {
        "steps": len(rows),
        "layers": len(wrappers),
        "compiled_closures": 2 * len(wrappers),
        "logit_max_abs_delta": max(row["logit_max_abs_delta"] for row in rows),
        "hidden_max_abs_delta": max(row["hidden_max_abs_delta"] for row in rows),
        "cache_max_abs_delta": max(row["cache_max_abs_delta"] for row in rows),
        "exact": all(
            row["logit_max_abs_delta"] == 0
            and row["hidden_max_abs_delta"] == 0
            and row["cache_max_abs_delta"] == 0
            for row in rows
        ),
        "raw_timings": rows,
    }


def run_exactness(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    groups = config["variants"]["8"][:2]
    by_id = {row["id"]: row for row in cohort}
    model, model_root, snapshot = load_model()
    mx.eval(model.parameters())
    functional = FunctionalCodePredictor(model, mx, nn, compiled=False)
    compiled = FunctionalCodePredictor(model, mx, nn, compiled=True)
    comparisons = []
    prefills = []
    for group_number, group in enumerate(groups):
        selected = [by_id[row_id] for row_id in group["ids"]]
        hidden, first, talker_seconds = frozen_prefill(model, selected, mx)
        mx.eval(hidden, first)
        prefills.append(
            {
                "group_id": group["group_id"],
                "ids": group["ids"],
                "hidden_sha256": array_hash(hidden, np),
                "first_token_sha256": array_hash(first, np),
                "talker_prefill_seconds": talker_seconds,
            }
        )
        # Compile before running either eager predictor so this is a true cold
        # first-call measurement rather than a warmed Metal-kernel cache cost.
        compiled_rows = functional_trace(compiled, hidden, first, mx, np)
        eager_rows = installed_trace(model, hidden, first, mx, np)
        functional_rows = functional_trace(functional, hidden, first, mx, np)
        for index, (eager, func, comp) in enumerate(
            zip(eager_rows, functional_rows, compiled_rows)
        ):
            comparisons.append(
                {
                    "group_id": group["group_id"],
                    "group_number": group_number,
                    "position": index,
                    "cache_length": int(comp["kv"][0].shape[2]),
                    "eager_seconds": eager["seconds"],
                    "functional_eager_seconds": func["seconds"],
                    "compiled_seconds": comp["seconds"],
                    "functional_eager_logit_max_abs_delta": delta(
                        eager["logits"], func["logits"], np
                    ),
                    "compiled_logit_max_abs_delta": delta(
                        eager["logits"], comp["logits"], np
                    ),
                    "functional_eager_top1_exact": bool(
                        np.array_equal(eager["token"], func["token"])
                    ),
                    "compiled_top1_exact": bool(
                        np.array_equal(eager["token"], comp["token"])
                    ),
                    "functional_eager_cache_max_abs_delta": max(
                        delta(left, right, np)
                        for left, right in zip(eager["kv"], func["kv"])
                    ),
                    "compiled_cache_max_abs_delta": max(
                        delta(left, right, np)
                        for left, right in zip(eager["kv"], comp["kv"])
                    ),
                }
            )

    first_selected = [by_id[row_id] for row_id in groups[0]["ids"]]
    talker_split = talker_split_exactness(model, first_selected, mx, np)
    report = {
        "schema_version": SCHEMA,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_attestation(),
        "command": sys.argv,
        "cohort": attestation(cohort_path),
        "groups": groups,
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "root": str(model_root),
            "files_sha256": snapshot,
        },
        "weights": {
            "transforms": "none",
            "code_predictor_dtypes": sorted(
                {
                    str(value.dtype)
                    for _, value in tree_flatten(
                        model.talker.code_predictor.parameters()
                    )
                }
            ),
        },
        "prefills": prefills,
        "positions_per_frame": compiled.num_steps,
        "layers_per_position": compiled.num_layers,
        "compiled_closure_count": len(compiled.steps),
        "compile_shape_policy": "fixed B8; shapeless probe failed and is archived separately",
        "functional_eager": {
            "logit_max_abs_delta": max(
                row["functional_eager_logit_max_abs_delta"] for row in comparisons
            ),
            "top1_exact_all": all(
                row["functional_eager_top1_exact"] for row in comparisons
            ),
            "cache_max_abs_delta": max(
                row["functional_eager_cache_max_abs_delta"] for row in comparisons
            ),
        },
        "compiled": {
            "logit_max_abs_delta": max(
                row["compiled_logit_max_abs_delta"] for row in comparisons
            ),
            "top1_exact_all": all(row["compiled_top1_exact"] for row in comparisons),
            "cache_max_abs_delta": max(
                row["compiled_cache_max_abs_delta"] for row in comparisons
            ),
            "cold_compile_seconds_by_position": compiled.compile_seconds,
            "cold_compile_seconds_total": sum(compiled.compile_seconds),
            "warm_predictor_seconds_by_group": [
                sum(
                    row["compiled_seconds"]
                    for row in comparisons
                    if row["group_number"] == number
                )
                for number in range(len(groups))
            ],
        },
        "talker_split": talker_split,
        "raw_timings": comparisons,
    }
    atomic_write_bytes(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps({
        "functional_eager": report["functional_eager"],
        "compiled": report["compiled"],
        "talker_split": report["talker_split"],
    }, indent=2))


def probe_shapeless(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    group = config["variants"]["8"][0]
    by_id = {row["id"]: row for row in cohort}
    model, _, _ = load_model()
    mx.eval(model.parameters())
    hidden, first, _ = frozen_prefill(
        model, [by_id[row_id] for row_id in group["ids"]], mx
    )
    started = time.monotonic()
    status = "unexpected_success"
    error = None
    try:
        adapter = FunctionalCodePredictor(model, mx, nn, compiled=False)
        adapter.steps = [mx.compile(step, shapeless=True) for step in adapter.eager_steps]
        adapter.compiled = True
        adapter.run_step(0, hidden, first, (), evaluate=True)
    except BaseException as exc:
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    report = {
        "schema_version": SCHEMA,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_attestation(),
        "command": sys.argv,
        "boundary": "code-predictor functional position-0 closure, B8, shapeless=True",
        "status": status,
        "seconds": time.monotonic() - started,
        "error": error,
        "decision": "use fixed-B8 compilation; do not broaden the recurrent rewrite",
    }
    atomic_write_bytes(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps(report, indent=2))


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
    model, model_root, snapshot = load_model()
    mx.eval(model.parameters())
    active_before = int(mx.get_active_memory())
    adapter = None
    split_wrappers = None
    if args.candidate in ("recurrent_compiled", "recurrent_compiled_talker_split"):
        adapter = install_functional_code_predictor(model, mx, nn, compiled=True)
    if args.candidate == "recurrent_compiled_talker_split":
        split_wrappers = install_talker_layer_split(model, mx)

    output = args.output_root.expanduser().resolve() / f"{args.candidate}_n{args.rows}"
    output.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(mx)
    records = []
    code_rows = []
    for number, group in enumerate(groups, 1):
        set_rng(group["seed"], mx, np)
        group_output = output / "batches" / group["group_id"]
        captured = []
        if group_output.exists():
            record = generate_group(
                model, mx, np, sf, selected_rows, group, group_output,
                cohort_path, environment, snapshot, SCHEMA, ROW_SCHEMA,
            )
        else:
            captured, restore = capture_generated_codes(model, mx, np)
            try:
                record = generate_group(
                    model, mx, np, sf, selected_rows, group, group_output,
                    cohort_path, environment, snapshot, SCHEMA, ROW_SCHEMA,
                )
            finally:
                restore()
        if captured and len(captured) != len(record["rows"]):
            raise RuntimeError("Generated-code capture count mismatch")
        for index, row in enumerate(record["rows"]):
            code_path = output / "codes" / f"{row['id'].replace(':', '_')}.npy"
            if captured:
                atomic_save_npy(code_path, captured[index], np)
            if not code_path.is_file():
                raise RuntimeError(f"Missing generated codes: {code_path}")
            codes = np.load(code_path, allow_pickle=False)
            code_rows.append(
                {
                    "id": row["id"],
                    "audio_sha256": row["audio_sha256"],
                    "code_path": str(code_path),
                    "code_sha256": sha256_file(code_path),
                    "code_shape": list(codes.shape),
                }
            )
        records.append(record)
        print(f"[{args.candidate} {number}/{len(groups)}] {record['rows_per_minute']:.3f} rows/min", flush=True)

    wall = sum(row["wall_seconds"] for row in records)
    rows = [row for record in records for row in record["rows"]]
    report = {
        "schema_version": SCHEMA,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "script": script_attestation(),
        "helper_scripts": [
            attestation(Path(__file__).with_name("qwen_mlx_recurrent.py")),
            attestation(Path(__file__).with_name("benchmark_vivos_qwen_mlx_batch_v2.py")),
        ],
        "command": sys.argv,
        "cohort": attestation(cohort_path),
        "scope_rows": len(rows),
        "group_ids": [group["group_id"] for group in groups],
        "model": {
            "id": MLX_MODEL_ID,
            "revision": MLX_MODEL_REVISION,
            "root": str(model_root),
            "files_sha256": snapshot,
        },
        "generation": GENERATION,
        "rng_contract": "unchanged group-owned Python/NumPy/MLX seed; sampling remains outside compilation",
        "adapter": adapter.timing_report() if adapter else None,
        "talker_split": (
            {
                "layers": len(split_wrappers),
                "compiled_closures": 2 * len(split_wrappers),
                "pre_calls": sum(layer.pre_calls for layer in split_wrappers),
                "post_calls": sum(layer.post_calls for layer in split_wrappers),
            }
            if split_wrappers
            else None
        ),
        "compile_timing_caveat": "authoritative cold/warm compile timings are in exactness.json where each output is explicitly evaluated",
        "wall_seconds": wall,
        "rows_per_minute": 60 * len(rows) / wall,
        "group_rows_per_minute": [row["rows_per_minute"] for row in records],
        "warm_group_rows_per_minute": records[-1]["rows_per_minute"],
        "talker_seconds": sum(
            row["stage_timing"]["prepare_prefill_talker_seconds_reported"]
            for row in records
        ),
        "decode_seconds": sum(
            row["stage_timing"]["sequential_decode_and_yield_seconds"]
            for row in records
        ),
        "active_memory_bytes": int(mx.get_active_memory()),
        "active_memory_before_adapter_bytes": active_before,
        "peak_mlx_memory_bytes": max(row["peak_mlx_memory_bytes"] for row in records),
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "system": {
            "platform": platform.platform(),
            "load_average": os.getloadavg(),
            "thermal": subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True
            ).stdout.strip(),
        },
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
    print(json.dumps({key: report[key] for key in ("candidate", "rows_per_minute", "warm_group_rows_per_minute")}, indent=2))


def compare_exact(args: argparse.Namespace) -> None:
    import numpy as np

    baseline = args.baseline.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    left = {row["id"]: row for row in map(json.loads, (baseline / "codes.jsonl").read_text().splitlines())}
    right = {row["id"]: row for row in map(json.loads, (candidate / "codes.jsonl").read_text().splitlines())}
    if set(left) != set(right):
        raise RuntimeError("Generated-code comparison ID sets differ")
    rows = []
    for row_id in sorted(left):
        left_codes = np.load(left[row_id]["code_path"], allow_pickle=False)
        right_codes = np.load(right[row_id]["code_path"], allow_pickle=False)
        rows.append(
            {
                "id": row_id,
                "code_array_exact": bool(np.array_equal(left_codes, right_codes)),
                "code_sha256_match": left[row_id]["code_sha256"] == right[row_id]["code_sha256"],
                "audio_sha256_match": left[row_id]["audio_sha256"] == right[row_id]["audio_sha256"],
            }
        )
    report = {
        "schema_version": SCHEMA,
        "baseline": attestation(baseline / "candidate.json"),
        "candidate": attestation(candidate / "candidate.json"),
        "rows": len(rows),
        "exact_code_arrays": sum(row["code_array_exact"] for row in rows),
        "exact_code_hashes": sum(row["code_sha256_match"] for row in rows),
        "exact_audio_hashes": sum(row["audio_sha256_match"] for row in rows),
        "row_comparison": rows,
    }
    atomic_write_bytes(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps({key: report[key] for key in ("rows", "exact_code_arrays", "exact_audio_hashes")}, indent=2))


def main() -> None:
    args = parse_args()
    if args.action == "exactness":
        run_exactness(args)
    elif args.action == "probe-shapeless":
        probe_shapeless(args)
    elif args.action == "run-candidate":
        run_candidate(args)
    else:
        compare_exact(args)


if __name__ == "__main__":
    main()
