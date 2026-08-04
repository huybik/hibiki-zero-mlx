"""Archive the immutable Qwen MLX recurrent-v4 benchmark report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from synthesize_vivos import atomic_write_bytes, read_jsonl, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text())


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def pct(value: float) -> str:
    return f"{100 * value:+.1f}%"


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    qa_dir = args.qa_dir.expanduser().resolve()
    candidates = {
        name: load(report_dir / f"{name}.json")
        for name in (
            "eager_n16",
            "recurrent_compiled_n16",
            "recurrent_compiled_talker_split_n16",
            "eager_n64",
            "recurrent_compiled_n64",
        )
    }
    exactness = load(report_dir / "exactness.json")
    generated16 = load(report_dir / "generated_exactness.json")
    generated64 = load(report_dir / "generated_exactness_n64.json")
    talker_generated = load(report_dir / "talker_split_generated_exactness.json")
    shapeless = load(report_dir / "shapeless_failure.json")
    qa = load(qa_dir / "qa_report.json")
    metrics = read_jsonl(qa_dir / "metrics.jsonl")

    eager16 = candidates["eager_n16"]
    compiled16 = candidates["recurrent_compiled_n16"]
    split16 = candidates["recurrent_compiled_talker_split_n16"]
    eager64 = candidates["eager_n64"]
    compiled64 = candidates["recurrent_compiled_n64"]
    gain16 = compiled16["rows_per_minute"] / eager16["rows_per_minute"] - 1
    gain64 = compiled64["rows_per_minute"] / eager64["rows_per_minute"] - 1
    generation_gain64 = eager64["talker_seconds"] / compiled64["talker_seconds"] - 1
    split_gain16 = split16["rows_per_minute"] / compiled16["rows_per_minute"] - 1
    second_group = [
        row for row in exactness["raw_timings"] if row["group_number"] == 1
    ]
    predictor_gain = (
        sum(row["eager_seconds"] for row in second_group)
        / sum(row["compiled_seconds"] for row in second_group)
        - 1
    )

    selection = {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_recurrent_selection_v4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "retain_exact_recurrent_adapter_no_production_campaign",
        "code_predictor": {
            "functional_eager_exact": exactness["functional_eager"],
            "compiled_exact": {
                key: exactness["compiled"][key]
                for key in (
                    "logit_max_abs_delta",
                    "top1_exact_all",
                    "cache_max_abs_delta",
                    "cold_compile_seconds_total",
                )
            },
            "warm_fixed_trace_gain": predictor_gain,
            "n16_rows_per_minute_gain": gain16,
            "n64_rows_per_minute_gain": gain64,
            "n64_generation_stage_gain": generation_gain64,
            "generated_exact_n16": generated16["exact_code_arrays"] == 16
            and generated16["exact_audio_hashes"] == 16,
            "generated_exact_n64": generated64["exact_code_arrays"] == 64
            and generated64["exact_audio_hashes"] == 64,
        },
        "talker_split": {
            "trace_exact": exactness["talker_split"]["exact"],
            "generated_exact": talker_generated["exact_code_arrays"] == 16
            and talker_generated["exact_audio_hashes"] == 16,
            "n16_incremental_rows_per_minute_gain": split_gain16,
            "decision": "reject_throughput_regression",
        },
        "quality": {
            "paired_wer_delta": 0.0,
            "paired_speaker_cosine_delta": 0.0,
            "proof": "PCM WAV hashes are exact for every paired row",
            "absolute_qa_pass": qa["automatic_absolute_pass"],
            "asr_wer": qa["asr_wer"],
            "speaker_cosine_median": qa["speaker_cosine_median"],
        },
        "robust_end_to_end_three_percent_gate": gain64 >= 0.03,
        "campaign_launched": False,
        "next_phase": "active-lane compaction with row-owned RNG and length-aware scheduling",
    }
    atomic_write_bytes(report_dir / "selection_report.json", json_bytes(selection))

    raw = []
    for name, candidate in candidates.items():
        records = read_jsonl(Path(candidate["raw_records"]))
        for record in records:
            raw.append(
                {
                    "kind": "end_to_end_group",
                    "candidate": name,
                    "group_id": record["group"]["group_id"],
                    "rows": len(record["rows"]),
                    "wall_seconds": record["wall_seconds"],
                    "rows_per_minute": record["rows_per_minute"],
                    "talker_seconds": record["stage_timing"][
                        "prepare_prefill_talker_seconds_reported"
                    ],
                    "decode_seconds": record["stage_timing"][
                        "sequential_decode_and_yield_seconds"
                    ],
                    "peak_mlx_memory_bytes": record["peak_mlx_memory_bytes"],
                    "peak_process_rss_bytes": record["peak_process_rss_bytes"],
                }
            )
    raw.extend(
        {"kind": "code_predictor_position", **row}
        for row in exactness["raw_timings"]
    )
    raw.extend(
        {"kind": "talker_split_step", **row}
        for row in exactness["talker_split"]["raw_timings"]
    )
    atomic_write_bytes(report_dir / "raw_timing.jsonl", jsonl_bytes(raw))

    failures = [
        {
            "experiment": "shapeless_code_predictor_compile",
            "status": shapeless["status"],
            "boundary": shapeless["boundary"],
            "error": shapeless["error"],
            "decision": shapeless["decision"],
        },
        {
            "experiment": "main_talker_compiled_pre_post_split",
            "status": "no_go",
            "n16_incremental_rows_per_minute_gain": split_gain16,
            "decision": "reject_throughput_regression",
        },
        {
            "experiment": "recurrent_compiled_n64_three_percent_gate",
            "status": "no_go",
            "end_to_end_rows_per_minute_gain": gain64,
            "generation_stage_gain": generation_gain64,
            "decision": "retain exact adapter but make no robust >=3% end-to-end claim",
        },
        {
            "experiment": "absolute_automatic_qa",
            "status": "no_go",
            "asr_wer": qa["asr_wer"],
            "speaker_cosine_median": qa["speaker_cosine_median"],
            "row_gate_failures": qa["row_gate_failures"],
            "decision": "optimization is exact, but the unchanged synthesis baseline remains below the corpus WER gate",
        },
    ]
    atomic_write_bytes(report_dir / "failures.jsonl", jsonl_bytes(failures))

    environment = {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_recurrent_environment_v4",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_run": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "model_runtime": read_jsonl(Path(eager64["raw_records"]))[0]["environment"],
        "qa_runtime": qa["runtime"],
        "model": eager64["model"],
        "generation": eager64["generation"],
        "cohort": eager64["cohort"],
        "sources": [
            attestation(Path(__file__)),
            attestation(Path(__file__).with_name("qwen_mlx_recurrent.py")),
            eager64["script"],
        ],
        "qa": attestation(qa_dir / "qa_report.json"),
        "audio_location": str(output_root),
        "audio_in_git": False,
        "power_measurement": "not available; no throughput-per-watt claim",
    }
    atomic_write_bytes(report_dir / "environment.json", json_bytes(environment))

    stream = io.StringIO()
    writer = csv.DictWriter(
        stream,
        lineterminator="\n",
        fieldnames=(
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
        ),
    )
    writer.writeheader()
    compiled_rows = {
        item["id"]: item
        for record in read_jsonl(Path(compiled16["raw_records"]))
        for item in record["rows"]
    }
    for row in metrics:
        source = compiled_rows[row["id"]]
        writer.writerow(
            {
                "sample": row["id"],
                "variant": "recurrent_compiled",
                "audio_file": row["output_wav"],
                "duration_s": row["duration_s"],
                "transcript_vi": source["text_vi"],
                "reference_en": source["text_en"],
                "asr_output_en": row["asr_transcript_en"],
                "asr_wer": row["asr_wer"],
                "speaker_cosine": row["speaker_cosine"],
                "failure_reasons": ",".join(row["failure_reasons"]),
                "audio_sha256": row["audio_sha256"],
            }
        )
    atomic_write_bytes(report_dir / "translations.csv", stream.getvalue().encode())

    metrics_md = f"""# Qwen3-TTS MLX recurrent compilation v4

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 `{eager64['model']['revision'][:8]}…` · fixed same-speaker B8 · unchanged RNG and sampling.

| Candidate | Scope | Rows/min | Gain | Generation-stage gain | Code/WAV exact | Decision |
|---|---:|---:|---:|---:|---:|---|
| eager bf16 | 16 | {eager16['rows_per_minute']:.3f} | baseline | — | baseline | control |
| functional compiled code predictor | 16 | {compiled16['rows_per_minute']:.3f} | {pct(gain16)} | — | 16/16 | advance |
| + main-talker compiled pre/post split | 16 | {split16['rows_per_minute']:.3f} | {pct(split_gain16)} vs predictor | — | 16/16 | reject |
| eager bf16 | 64 | {eager64['rows_per_minute']:.3f} | baseline | baseline | baseline | control |
| functional compiled code predictor | 64 | {compiled64['rows_per_minute']:.3f} | {pct(gain64)} | {pct(generation_gain64)} | 64/64 | retain, no ≥3% total claim |

The five-layer code predictor now threads every layer K/V state as explicit arrays through 15 fixed-position closures; sampling remains outside. Functional eager and fixed-B8 compiled execution both have zero maximum logit and cache delta and exact top-1 across two frozen B8 prefills. Process-first calls across all positions cost {exactness['compiled']['cold_compile_seconds_total']:.3f} s in the isolated trace with the machine compiler cache warm; the warm fixed trace improved {pct(predictor_gain)}.

The 28-layer talker split was also array-exact, but regressed end-to-end throughput {pct(split_gain16)} and was rejected. Shapeless compilation failed at position 0 with `{shapeless['error']['message']}`; fixed B8 is the supported boundary.

Pinned 16-row QA on the compiled candidate gave WER {qa['asr_wer']:.4f}, median speaker cosine {qa['speaker_cosine_median']:.4f}, and zero prompt leaks. Paired quality deltas are exactly zero because every generated code array and PCM hash matches eager. Absolute QA remains no-go because the unchanged baseline WER exceeds 0.08; no campaign or upload was launched.

Active-lane compaction remains the next phase. This phase deliberately did not change RNG, compact lanes, quantize weights, fuse weights, or launch a full campaign.
"""
    atomic_write_bytes(report_dir / "metrics.md", metrics_md.encode())

    commands = [
        "# Reproduction commands",
        "",
        "Pinned model runtime: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python`. Pinned QA runtime: `/Volumes/data/envs/hibiki-vivos-qa/bin/python`.",
        "",
        "```bash",
    ]
    for path in (
        report_dir / "shapeless_failure.json",
        report_dir / "exactness.json",
        report_dir / "eager_n16.json",
        report_dir / "recurrent_compiled_n16.json",
        report_dir / "recurrent_compiled_talker_split_n16.json",
        report_dir / "eager_n64.json",
        report_dir / "recurrent_compiled_n64.json",
    ):
        command = load(path)["command"]
        executable = (
            "/opt/homebrew/Caskroom/miniconda/base/bin/python"
            if command[0].endswith("benchmark_vivos_qwen_mlx_recurrent_v4.py")
            and "compare-exact" in command
            else "/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python"
        )
        commands.append(" ".join([executable, *map(str, command)]))
    commands.extend(
        [
            "/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_recurrent_v4.py compare-exact --baseline /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/eager_n16 --candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n16 --out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_recurrent_v4/generated_exactness.json",
            "/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_recurrent_v4.py compare-exact --baseline /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n16 --candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_talker_split_n16 --out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_recurrent_v4/talker_split_generated_exactness.json",
            "/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/benchmark_vivos_qwen_mlx_recurrent_v4.py compare-exact --baseline /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/eager_n64 --candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n64 --out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_recurrent_v4/generated_exactness_n64.json",
            "/Volumes/data/envs/hibiki-vivos-qa/bin/python training-data/qa_vivos_qwen_mlx_efficiency_v3.py /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n16 --out-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n16_final",
            "/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/summarize_vivos_qwen_mlx_recurrent_v4.py --report-dir reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_recurrent_v4 --output-root /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_recurrent_v4 --qa-dir /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_recurrent_v4/recurrent_compiled_n16_final",
            "```",
            "",
            "Generated WAVs and code arrays remain external under the output root recorded in `environment.json`.",
        ]
    )
    atomic_write_bytes(report_dir / "commands.md", ("\n".join(commands) + "\n").encode())
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
