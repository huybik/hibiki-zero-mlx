"""Isolate Qwen3-TTS decoder scheduling with preserved bf16 code arrays."""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
import types
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_batch import GENERATION, json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import load_benchmark
from qwen_mlx_efficiency import install_full_reference_cache
from synthesize_vivos import MLX_MODEL_ID, MLX_MODEL_REVISION, atomic_write_wav, atomic_write_bytes, sha256_file

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_decoder_benchmark_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_plan", type=Path)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def load_model() -> Any:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model as load

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    return load(root)


def decode_full(tokenizer: Any, generated: Any, reference: Any, mx: Any) -> Any:
    generated_mx = mx.array(generated)
    reference_mx = mx.array(reference)
    reference_t = mx.transpose(reference_mx, (0, 2, 1))
    full = mx.concatenate([reference_t, generated_mx], axis=1)
    audio, _ = tokenizer.decode(full)
    start = reference.shape[2] * tokenizer.decode_upsample_rate
    length = generated.shape[1] * tokenizer.decode_upsample_rate
    audio = audio[0, start : start + length]
    mx.eval(audio)
    return audio


def main() -> None:
    args = parse_args()
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.utils import load_audio

    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    groups = config["variants"]["8"][: args.groups]
    by_id = {row["id"]: row for row in cohort}
    selected_ids = [row_id for group in groups for row_id in group["ids"]]
    baseline_root = args.baseline_root.expanduser().resolve()
    code_manifest = {
        row["id"]: row
        for row in map(json.loads, (baseline_root / "codes.jsonl").read_text().splitlines())
    }
    if any(row_id not in code_manifest for row_id in selected_ids):
        raise RuntimeError("Preserved-code manifest does not cover pipeline cohort")
    baseline_records = [
        json.loads(line) for line in (baseline_root / "raw_results.jsonl").read_text().splitlines()
    ][: args.groups]
    baseline_rows = {row["id"]: row for record in baseline_records for row in record["rows"]}

    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Static single-row buckets on Metal: later right-padding is causally inert,
    # then the output is trimmed to the exact generated-code length.
    gpu_model = load_model()
    install_full_reference_cache(gpu_model, mx)
    first = by_id[selected_ids[0]]
    ref_audio = load_audio(first["reference"]["reference_audio_path"], sample_rate=24_000)
    ref_codes = gpu_model.speech_tokenizer.encode(ref_audio[None, None, :])
    mx.eval(ref_codes)
    ref_np = np.asarray(ref_codes, dtype=np.int32)
    static_rows = []
    for row_id in selected_ids[:8]:
        generated = np.load(code_manifest[row_id]["code_path"], allow_pickle=False)
        started = time.monotonic()
        exact = np.asarray(
            decode_full(gpu_model.speech_tokenizer, generated, ref_np, mx),
            dtype=np.float32,
        )
        exact_seconds = time.monotonic() - started
        total = ref_np.shape[2] + generated.shape[1]
        bucket_total = ((total + 15) // 16) * 16
        pad = bucket_total - total
        padded_generated = np.pad(generated, ((0, 0), (0, pad), (0, 0)))
        started = time.monotonic()
        bucketed = np.asarray(
            decode_full(gpu_model.speech_tokenizer, padded_generated, ref_np, mx),
            dtype=np.float32,
        )[: exact.size]
        bucket_seconds = time.monotonic() - started
        static_rows.append(
            {
                "id": row_id,
                "generated_tokens": generated.shape[1],
                "full_tokens": total,
                "bucket_tokens": bucket_total,
                "exact_seconds": exact_seconds,
                "bucket_seconds": bucket_seconds,
                "speedup": exact_seconds / bucket_seconds,
                "waveform_exact": bool(np.array_equal(exact, bucketed)),
                "waveform_max_abs_delta": float(np.max(np.abs(exact - bucketed))),
            }
        )

    if args.static_only:
        report = {
            "schema_version": SCHEMA,
            "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
            "static_length_bucket": {
                "bucket_multiple": 16,
                "rows": static_rows,
                "exact_waveforms": sum(row["waveform_exact"] for row in static_rows),
                "median_speedup": float(np.median([row["speedup"] for row in static_rows])),
                "decision": "candidate" if all(row["waveform_exact"] for row in static_rows) else "reject_non_exact",
            },
        }
        report_dir = args.report_dir.expanduser().resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(report_dir / "decoder_static_report.json", json_bytes(report))
        print(json.dumps(report, indent=2))
        return

    # A decoder model and CPU stream are created and used only on the consumer
    # thread. The producer crosses the thread boundary with eager NumPy codes.
    work: queue.Queue[Any] = queue.Queue(maxsize=16)
    ready = threading.Event()
    consumer_error: list[BaseException] = []
    cpu_rows: list[dict[str, Any]] = []

    def consume() -> None:
        try:
            cpu_stream = mx.new_stream(mx.cpu)
            with mx.stream(cpu_stream):
                cpu_model = load_model()
                tokenizer = cpu_model.speech_tokenizer
                mx.eval(tokenizer.parameters())
                del cpu_model
                ready.set()
                while True:
                    item = work.get()
                    if item is None:
                        work.task_done()
                        break
                    row_id, generated, reference = item
                    started = time.monotonic()
                    audio = np.asarray(
                        decode_full(tokenizer, generated, reference, mx), dtype=np.float32
                    )
                    seconds = time.monotonic() - started
                    path = output / "cpu_pipeline" / "wavs" / f"{row_id.replace(':', '_')}.wav"
                    atomic_write_wav(path, audio, 24_000, sf)
                    baseline_audio, baseline_rate = sf.read(
                        baseline_rows[row_id]["output_wav"], dtype="float32"
                    )
                    if baseline_rate != 24_000:
                        raise RuntimeError(f"CPU decoder sample-rate mismatch: {row_id}")
                    overlap = min(audio.size, baseline_audio.size)
                    cpu_rows.append(
                        {
                            "id": row_id,
                            "output_wav": str(path),
                            "audio_sha256": sha256_file(path),
                            "decode_seconds": seconds,
                            "duration_s": audio.size / 24_000,
                            "baseline_duration_s": baseline_audio.size / 24_000,
                            "duration_delta_s": (audio.size - baseline_audio.size) / 24_000,
                            "waveform_exact_float": bool(np.array_equal(audio, baseline_audio)),
                            "waveform_max_abs_delta_overlap": float(
                                np.max(np.abs(audio[:overlap] - baseline_audio[:overlap]))
                            ),
                        }
                    )
                    work.task_done()
        except BaseException as error:
            consumer_error.append(error)
            ready.set()

    consumer = threading.Thread(target=consume, name="qwen-cpu-decoder")
    consumer.start()
    if not ready.wait(180):
        raise RuntimeError("CPU decoder model did not become ready")
    if consumer_error:
        raise consumer_error[0]

    current_ids: list[str] = []
    current_index = 0
    original_decode = gpu_model._decode_icl_generated_codes

    def enqueue(self: Any, generated_codes: list[Any], reference: Any) -> Any:
        nonlocal current_index
        generated_mx = mx.stack(generated_codes, axis=1)
        mx.eval(generated_mx, reference)
        item = (
            current_ids[current_index],
            np.asarray(generated_mx, dtype=np.int32),
            np.asarray(reference, dtype=np.int32),
        )
        while True:
            if consumer_error:
                raise RuntimeError("CPU decoder consumer failed") from consumer_error[0]
            try:
                work.put(item, timeout=1)
                break
            except queue.Full:
                continue
        current_index += 1
        return mx.zeros((1,), dtype=mx.float32)

    gpu_model._decode_icl_generated_codes = types.MethodType(enqueue, gpu_model)
    producer_groups = []
    pipeline_started = time.monotonic()
    try:
        for group in groups:
            current_ids = list(group["ids"])
            current_index = 0
            rows = [by_id[row_id] for row_id in current_ids]
            reference = rows[0]["reference"]
            random.seed(group["seed"])
            np.random.seed(group["seed"])
            mx.random.seed(group["seed"])
            started = time.monotonic()
            results = list(
                gpu_model.batch_generate(
                    texts=[row["text_en"] for row in rows],
                    ref_audio=reference["reference_audio_path"],
                    ref_text=reference["reference_text_vi"],
                    max_tokens=GENERATION["max_tokens"],
                    temperature=GENERATION["temperature"],
                    top_k=GENERATION["top_k"],
                    top_p=GENERATION["top_p"],
                    repetition_penalty=GENERATION["repetition_penalty_requested"],
                    lang_code=GENERATION["lang_code"],
                    stream=False,
                )
            )
            seconds = time.monotonic() - started
            if current_index != len(rows) or len(results) != len(rows):
                raise RuntimeError("Generator-only enqueue did not cover the batch")
            producer_groups.append(
                {"group_id": group["group_id"], "rows": len(rows), "generation_seconds": seconds}
            )
    finally:
        gpu_model._decode_icl_generated_codes = original_decode
    generation_only_wall = time.monotonic() - pipeline_started
    if consumer_error:
        raise RuntimeError("CPU decoder consumer failed") from consumer_error[0]
    work.put(None)
    work.join()
    consumer.join()
    end_to_end_wall = time.monotonic() - pipeline_started
    if consumer_error:
        raise consumer_error[0]

    cpu_rows.sort(key=lambda row: selected_ids.index(row["id"]))
    cpu_candidate = {
        "candidate": "cpu_decoder_pipeline",
        "scope_rows": len(cpu_rows),
        "schema_version": SCHEMA,
    }
    atomic_write_bytes(output / "cpu_pipeline" / "candidate.json", json_bytes(cpu_candidate))
    atomic_write_bytes(
        output / "cpu_pipeline" / "raw_results.jsonl",
        jsonl_bytes([{"rows": cpu_rows}]),
    )
    baseline_wall = sum(record["wall_seconds"] for record in baseline_records)
    report = {
        "schema_version": SCHEMA,
        "cohort": {"path": str(cohort_path), "sha256": sha256_file(cohort_path)},
        "groups": len(groups),
        "rows": len(selected_ids),
        "static_length_bucket": {
            "bucket_multiple": 16,
            "rows": static_rows,
            "exact_waveforms": sum(row["waveform_exact"] for row in static_rows),
            "median_speedup": float(np.median([row["speedup"] for row in static_rows])),
            "decision": "candidate" if all(row["waveform_exact"] for row in static_rows) else "reject_non_exact",
        },
        "cpu_pipeline": {
            "queue_payload": "eager NumPy generated/reference code arrays",
            "consumer": "dedicated MLX CPU model and thread-local CPU stream",
            "producer_groups": producer_groups,
            "generation_only_wall_seconds": generation_only_wall,
            "decoder_sum_seconds": sum(row["decode_seconds"] for row in cpu_rows),
            "end_to_end_wall_seconds": end_to_end_wall,
            "baseline_gpu_end_to_end_wall_seconds": baseline_wall,
            "end_to_end_speedup_vs_bf16_prefix": baseline_wall / end_to_end_wall,
            "exact_pcm_wavs": sum(
                row["audio_sha256"] == baseline_rows[row["id"]]["audio_sha256"]
                for row in cpu_rows
            ),
            "exact_float_waveforms": sum(row["waveform_exact_float"] for row in cpu_rows),
            "max_abs_waveform_delta_overlap": max(
                row["waveform_max_abs_delta_overlap"] for row in cpu_rows
            ),
            "candidate_root": str(output / "cpu_pipeline"),
        },
        "metal_stream_experiment": {
            "status": "not_run",
            "reason": "mlx-audio clears the global Metal allocator at each batch boundary and its lazy arrays are stream/thread bound; a shared-model second Metal stream is not a safe ownership boundary",
        },
        "power": {
            "throughput_per_watt_reported": False,
            "reason": "authoritative powermetrics requires privileges unavailable to this run",
        },
    }
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(report_dir / "decoder_report.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
