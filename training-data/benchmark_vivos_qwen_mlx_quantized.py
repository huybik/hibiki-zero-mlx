"""Benchmark a separately quality-gated Qwen3-TTS MLX q4 talker candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import (
    batch_path,
    generate_group,
    load_benchmark,
    runtime_environment,
)
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    verify_mlx_snapshot,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_talker_q4_benchmark_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_plan", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort_path = args.cohort_plan.expanduser().resolve()
    cohort, config = load_benchmark(cohort_path)
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    snapshot = verify_mlx_snapshot(root)
    model = load_model(root)
    mx.eval(model.parameters())
    active_before = int(mx.get_active_memory())

    def predicate(_: str, module: object) -> bool:
        return isinstance(module, (nn.Linear, nn.Embedding)) and module.weight.shape[-1] % 64 == 0

    nn.quantize(model.talker, group_size=64, bits=4, class_predicate=predicate)
    mx.eval(model.talker.parameters())
    active_after = int(mx.get_active_memory())
    records = []
    output = args.output_root.expanduser().resolve()
    environment = runtime_environment(mx)
    for number, group in enumerate(config["variants"]["8"], 1):
        record = generate_group(
            model,
            mx,
            np,
            sf,
            cohort,
            group,
            batch_path(output, 8, group["group_id"]),
            cohort_path,
            environment,
            snapshot,
        )
        records.append(record)
        print(f"[q4 {number}/8] {record['rows_per_minute']:.3f} rows/min", flush=True)
    wall = sum(record["wall_seconds"] for record in records)
    rows = [row for record in records for row in record["rows"]]
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "source_id": MLX_MODEL_ID,
            "source_revision": MLX_MODEL_REVISION,
            "new_synthesis_model": True,
            "quantized_module": "model.talker",
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "predicate": "Linear or Embedding with input dimension divisible by 64",
        },
        "rows": len(rows),
        "wall_seconds": wall,
        "rows_per_minute": 60 * len(rows) / wall,
        "audio_seconds_per_wall_second": sum(row["duration_s"] for row in rows) / wall,
        "talker_seconds": sum(record["stage_timing"]["prepare_prefill_talker_seconds_reported"] for record in records),
        "decode_seconds": sum(record["stage_timing"]["sequential_decode_and_yield_seconds"] for record in records),
        "active_memory_before_quantization_bytes": active_before,
        "active_memory_after_quantization_bytes": active_after,
        "active_memory_reduction": 1 - active_after / active_before,
        "peak_mlx_memory_bytes": max(record["peak_mlx_memory_bytes"] for record in records),
        "peak_process_rss_bytes": max(record["peak_process_rss_bytes"] for record in records),
        "automatic_quality_status": "not_yet_scored",
        "records": str((args.report_dir / "quantization_raw_results.jsonl").resolve()),
    }
    report_dir = args.report_dir.expanduser().resolve()
    atomic_write_bytes(report_dir / "quantization_report.json", json_bytes(report))
    atomic_write_bytes(report_dir / "quantization_raw_results.jsonl", jsonl_bytes(records))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
