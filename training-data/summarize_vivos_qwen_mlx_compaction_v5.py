"""Archive the immutable Qwen MLX compaction-v5 benchmark record."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import shlex
import subprocess
from pathlib import Path

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from synthesize_vivos import atomic_write_bytes, package_version, read_jsonl, sha256_file


def attest(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def candidate(root: Path, name: str) -> dict:
    return json.loads((root / name / "candidate.json").read_text())


def gain(value: float, baseline: float) -> float:
    return value / baseline - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    external = args.external_root.expanduser().resolve()
    qa = args.qa_root.expanduser().resolve()
    report = args.report_dir.expanduser().resolve()
    report.mkdir(parents=True, exist_ok=True)
    names = [
        "global_b8_t08_throughput64_attempt0",
        "row_rng_b8_t08_throughput64_attempt0",
        "row_rng_length_b8_t08_throughput64_attempt0",
        "row_rng_length_compact_b8_t08_throughput64_attempt0",
        "row_rng_length_compact_recurrent_b8_t08_throughput64_attempt0",
        "row_rng_b8_t08_quality16_attempt0",
        "row_rng_length_b8_t08_quality16_attempt0",
        "row_rng_length_compact_b8_t08_quality16_attempt0",
        "row_rng_length_compact_recurrent_b8_t08_quality16_attempt0",
        "row_rng_length_compact_recurrent_b8_t07_quality16_attempt0",
        "row_rng_length_b8_t08_quality64_attempt0",
        "row_rng_length_compact_recurrent_b8_t08_quality64_attempt0",
        "row_rng_length_compact_recurrent_b8_t07_quality64_attempt0",
        "row_rng_length_compact_recurrent_b8_t08_quality64_attempt1_retry",
    ]
    records = [candidate(external, name) for name in names]
    atomic_write_bytes(report / "raw_timing.jsonl", jsonl_bytes(records))
    selection = json.loads((report / "selection_report.json").read_text())
    quality = {
        "baseline16": json.loads((qa / "row_rng_length_b8_t08_quality16_attempt0/qa_report.json").read_text()),
        "candidate16": json.loads((qa / "row_rng_length_compact_recurrent_b8_t08_quality16_attempt0/qa_report.json").read_text()),
        "temperature07_16": json.loads((qa / "row_rng_length_compact_recurrent_b8_t07_quality16_attempt0/qa_report.json").read_text()),
        "baseline64": json.loads((qa / "row_rng_length_b8_t08_quality64_attempt0/qa_report.json").read_text()),
        "candidate64": json.loads((qa / "row_rng_length_compact_recurrent_b8_t08_quality64_attempt0/qa_report.json").read_text()),
        "temperature07_64": json.loads((qa / "row_rng_length_compact_recurrent_b8_t07_quality64_attempt0/qa_report.json").read_text()),
        "retry13": json.loads((qa / "row_rng_length_compact_recurrent_b8_t08_quality64_attempt1_retry/qa_report.json").read_text()),
    }
    atomic_write_bytes(report / "qa_summary.json", json_bytes(quality))
    length_record = json.loads((report / "length_model.json").read_text())
    plan_record = json.loads((external / "benchmark_plan.json").read_text())
    command_lines = [
        "# Reproduction commands",
        "",
        "Pinned MLX runtime: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python`. Pinned QA runtime: `/Volumes/data/envs/hibiki-vivos-qa/bin/python`. Candidate JSON files and the length/plan records retain the exact argv used for every executed run.",
        "",
        "```bash",
        shlex.join(["/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python", *length_record["command"]]),
        shlex.join(["/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python", *plan_record["command"]]),
        *[
            shlex.join(["/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python", *record["command"]])
            for record in records
        ],
        shlex.join(json.loads((report / "qa_attempt0_timing.json").read_text())["command"]),
        shlex.join(json.loads((report / "qa_attempt1_timing.json").read_text())["command"]),
        "/opt/homebrew/Caskroom/miniconda/base/bin/python training-data/select_vivos_qwen_compaction_v5.py --baseline-candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_compaction_v5/row_rng_length_b8_t08_quality64_attempt0 --baseline-qa /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_compaction_v5/row_rng_length_b8_t08_quality64_attempt0 --attempt0-candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_compaction_v5/row_rng_length_compact_recurrent_b8_t08_quality64_attempt0 --attempt0-qa /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_compaction_v5/timed_row_rng_length_compact_recurrent_b8_t08_quality64_attempt0 --attempt1-candidate /Volumes/data/datasets/hibiki_vi_v2/tts/vivos_qwen3_tts_mlx_compaction_v5/row_rng_length_compact_recurrent_b8_t08_quality64_attempt1_retry --attempt1-qa /Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_compaction_v5/timed_row_rng_length_compact_recurrent_b8_t08_quality64_attempt1_retry --qa0-timing reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_compaction_v5/qa_attempt0_timing.json --qa1-timing reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_compaction_v5/qa_attempt1_timing.json --out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_compaction_v5/selection_report.json --rows-out reports/benchmarks/vivos_tts/2026-08-04_qwen_mlx_compaction_v5/selection_rows.jsonl",
        "```",
        "",
        "The initial length-manifest, missing-parent-directory, and development-plan hash failures are preserved in `failures.jsonl`; no candidate outputs were published by those failed invocations.",
        "",
    ]
    atomic_write_bytes(report / "commands.md", "\n".join(command_lines).encode())
    by_name = dict(zip(names, records))
    global_b8 = by_name[names[0]]["timing"]["rows_per_minute"]
    row_rng = by_name[names[1]]["timing"]["rows_per_minute"]
    length = by_name[names[2]]["timing"]["rows_per_minute"]
    compact = by_name[names[3]]["timing"]["rows_per_minute"]
    recurrent = by_name[names[4]]["timing"]["rows_per_minute"]
    qa_base_rpm = by_name["row_rng_length_b8_t08_quality64_attempt0"]["timing"]["rows_per_minute"]
    qa_candidate_rpm = by_name["row_rng_length_compact_recurrent_b8_t08_quality64_attempt0"]["timing"]["rows_per_minute"]
    metrics = f"""# Qwen3-TTS MLX deterministic compaction v5

Date: 2026-08-04 · Apple M4 Pro / 48 GiB · Qwen3-TTS 1.7B Base bf16 `a6eb4f68…` · frozen B8 same-speaker throughput and 8-speaker quality cohorts.

| Candidate | Rows/min | Gain | Generation s | Decode s | Audio s/wall s | Talker/predictor waste | Peak MLX GiB | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| installed B8, group-global RNG | {global_b8:.3f} | baseline | {by_name[names[0]]['timing']['generation_seconds']:.2f} | {by_name[names[0]]['timing']['decode_seconds']:.2f} | {by_name[names[0]]['timing']['audio_seconds_per_wall_second']:.2f} | {by_name[names[0]]['lane_accounting']['talker_waste_fraction']:.1%}/{by_name[names[0]]['lane_accounting']['predictor_waste_fraction']:.1%} | {by_name[names[0]]['memory']['peak_mlx_memory_bytes']/2**30:.2f} | control |
| row-owned RNG, original groups | {row_rng:.3f} | {gain(row_rng, global_b8):+.1%} | {by_name[names[1]]['timing']['generation_seconds']:.2f} | {by_name[names[1]]['timing']['decode_seconds']:.2f} | {by_name[names[1]]['timing']['audio_seconds_per_wall_second']:.2f} | {by_name[names[1]]['lane_accounting']['talker_waste_fraction']:.1%}/{by_name[names[1]]['lane_accounting']['predictor_waste_fraction']:.1%} | {by_name[names[1]]['memory']['peak_mlx_memory_bytes']/2**30:.2f} | deterministic control |
| fitted-length groups, no compaction | {length:.3f} | {gain(length, row_rng):+.1%} | {by_name[names[2]]['timing']['generation_seconds']:.2f} | {by_name[names[2]]['timing']['decode_seconds']:.2f} | {by_name[names[2]]['timing']['audio_seconds_per_wall_second']:.2f} | {by_name[names[2]]['lane_accounting']['talker_waste_fraction']:.1%}/{by_name[names[2]]['lane_accounting']['predictor_waste_fraction']:.1%} | {by_name[names[2]]['memory']['peak_mlx_memory_bytes']/2**30:.2f} | reject: waste worsened |
| active-lane compaction | {compact:.3f} | {gain(compact, length):+.1%} | {by_name[names[3]]['timing']['generation_seconds']:.2f} | {by_name[names[3]]['timing']['decode_seconds']:.2f} | {by_name[names[3]]['timing']['audio_seconds_per_wall_second']:.2f} | 0.0%/0.0% | {by_name[names[3]]['memory']['peak_mlx_memory_bytes']/2**30:.2f} | advance |
| compaction + compiled predictor | {recurrent:.3f} | {gain(recurrent, compact):+.1%} | {by_name[names[4]]['timing']['generation_seconds']:.2f} | {by_name[names[4]]['timing']['decode_seconds']:.2f} | {by_name[names[4]]['timing']['audio_seconds_per_wall_second']:.2f} | 0.0%/0.0% | {by_name[names[4]]['memory']['peak_mlx_memory_bytes']/2**30:.2f} | additive gain {gain(recurrent, compact):+.1%} |

On the eight-speaker n=64 quality cohort, compact+compiled improved total throughput **{qa_base_rpm:.3f}→{qa_candidate_rpm:.3f} rows/min ({gain(qa_candidate_rpm, qa_base_rpm):+.1%})**. Attempt-0 quality was paired non-inferior: WER {quality['baseline64']['asr_wer']:.4f}→{quality['candidate64']['asr_wer']:.4f}; median speaker cosine {quality['baseline64']['speaker_cosine_median']:.4f}→{quality['candidate64']['speaker_cosine_median']:.4f}; zero prompt leaks for both. The compiled predictor is exact against eager compaction on 64/64 code arrays and WAV hashes.

Quality-adjusted selection retried {selection['selection']['retried_rows']} failing rows with distinct row-owned attempt-1 keys, preserved every attempt, and selected the lowest WER among candidates passing waveform/duration/leak/speaker gates. It accepted {selection['selection']['accepted_rows']}/64, rejected {selection['selection']['rejected_rows']}, and selected attempt 1 for {selection['selection']['selected_attempt1_rows']} rows. Selected WER is **{selection['selection']['selected_wer']:.4f}**, above the frozen 0.08 gate, so the decision is **NO-GO** and no production plan/campaign was created.

Accepted throughput is {selection['accepted_throughput']['baseline_attempt0_accepted_rows_per_minute']:.3f} rows/min for the bf16 B8 control, {selection['accepted_throughput']['candidate_attempt0_accepted_rows_per_minute']:.3f} rows/min for optimized attempt 0 ({selection['accepted_throughput']['candidate_attempt0_gain_vs_baseline']:+.1%}), and {selection['accepted_throughput']['candidate_generation_plus_retry_accepted_rows_per_minute']:.3f} rows/min after retry cost. Fresh QA cost {selection['accepted_throughput']['qa_wall_seconds_separate']:.3f} s separately; including it gives {selection['accepted_throughput']['candidate_generation_plus_retry_and_qa_rows_per_minute']:.3f} selected rows/min.

The frozen length model used all 1,489 immutable scalar sidecars and held out 307 rows (MAE 6.65 codec frames / about 0.53 s), but its schedule increased dead-lane waste from 24.6% to 26.6%; it is rejected as a scheduling improvement. Temperature 0.7 reduced n=16 row failures but regressed n=64 WER (0.1092→0.1126), median cosine, and total throughput; it is rejected. Compaction changes Metal batch width after EOS: 58/64 outputs were bit-exact versus no compaction, with six legitimate stochastic trajectories diverging under batch-width numeric drift. Pinned QA, not bit identity, therefore owns the quality decision.

Every timed candidate used one complete excluded B8 generation+decode warm-up. `pmset -g therm` reported no thermal, performance, or CPU-power warning in every group; raw group system snapshots, active widths/lane-steps, RSS, warm-up times, and model/source hashes are retained in the external candidate records and `raw_timing.jsonl`.
"""
    atomic_write_bytes(report / "metrics.md", metrics.encode())

    selected = {row["id"]: row for row in read_jsonl(report / "selection_rows.jsonl")}
    qa0 = {row["id"]: row for row in read_jsonl(qa / "row_rng_length_compact_recurrent_b8_t08_quality64_attempt0/metrics.jsonl")}
    qa1 = {row["id"]: row for row in read_jsonl(qa / "row_rng_length_compact_recurrent_b8_t08_quality64_attempt1_retry/metrics.jsonl")}
    cohort = {row["id"]: row for row in read_jsonl(external / "quality64.jsonl")}
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["sample", "audio_file", "duration_s", "transcript_vi", "reference_en", "asr_output_en", "asr_wer", "speaker_cosine", "failure_reasons", "attempt", "decision", "audio_sha256"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row_id in sorted(cohort):
        decision = selected[row_id]
        attempt = decision["selected_attempt"]
        metric = (qa1 if attempt == 1 else qa0)[row_id] if attempt is not None else qa0[row_id]
        writer.writerow(
            {
                "sample": row_id,
                "audio_file": metric["output_wav"],
                "duration_s": metric["duration_s"],
                "transcript_vi": cohort[row_id]["text_vi"],
                "reference_en": cohort[row_id]["text_en"],
                "asr_output_en": metric["asr_transcript_en"],
                "asr_wer": metric["asr_wer"],
                "speaker_cosine": metric["speaker_cosine"],
                "failure_reasons": ",".join(metric["failure_reasons"]),
                "attempt": attempt if attempt is not None else "rejected",
                "decision": decision["decision"],
                "audio_sha256": metric["audio_sha256"],
            }
        )
    atomic_write_bytes(report / "translations.csv", buffer.getvalue().encode())

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("qwen_mlx_compaction.py"),
        Path(__file__).with_name("benchmark_vivos_qwen_mlx_compaction_v5.py"),
        Path(__file__).with_name("fit_vivos_qwen_length_v5.py"),
        Path(__file__).with_name("run_vivos_qwen_qa_v5.py"),
        Path(__file__).with_name("select_vivos_qwen_compaction_v5.py"),
        Path(__file__).with_name("qwen_mlx_recurrent.py"),
        Path("/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/lib/python3.13/site-packages/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py"),
        Path("/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/lib/python3.13/site-packages/mlx_audio/tts/models/qwen3_tts/talker.py"),
        Path("/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/lib/python3.13/site-packages/mlx_audio/tts/models/qwen3_tts/continuous_batching.py"),
        Path("/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/lib/python3.13/site-packages/mlx_lm/models/cache.py"),
    ]
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "packages": {name: package_version(name) for name in ("mlx", "mlx-audio", "numpy", "soundfile")},
        "model": by_name[names[0]]["model"],
        "benchmark_plan": attest(external / "benchmark_plan.json"),
        "cohorts": {name: attest(external / f"{name}.jsonl") for name in ("throughput64", "quality16", "quality64")},
        "sources": [attest(path) for path in source_paths],
        "candidate_runner_hash_note": "candidate.json records the runner hash at execution; final source adds retry/resume/report orchestration without changing generate_lanes",
    }
    atomic_write_bytes(report / "environment.json", json_bytes(environment))

    failures = [
        {"phase": "length_fit_preflight", "status": "failed", "reason": "sidecar shasum manifest serialized path before digest; corrected to the frozen shasum digest-then-path format", "data_changed": False},
        {"phase": "first_candidate_preflight", "status": "failed", "reason": "temporary group parent directory was not created; fixed before any output group was published"},
        {"phase": "plan_preflight", "status": "failed", "reason": "development runner hash changed after immutable prepare; loader now binds schema/path while every execution archives its actual source hash"},
        {"phase": "length_scheduling", "status": "rejected", "reason": "dead-lane waste increased 24.6% to 26.6% on throughput64"},
        {"phase": "compaction_exactness", "status": "not_exact", "reason": "58/64 exact; batch-width numeric drift changed six stochastic trajectories; paired QA required"},
        {"phase": "temperature_0.7", "status": "rejected", "reason": "n64 WER, cosine, and throughput regressed versus 0.8"},
        {"phase": "attempt1_retry", "status": "no_go", "reason": "58/64 accepted but selected WER 0.10285 exceeds 0.08"},
        {"phase": "production_plan", "status": "not_run", "reason": "no candidate passed the absolute corpus WER gate"},
        {"phase": "full_campaign", "status": "not_run", "reason": "explicitly gated on an approved production plan"},
    ]
    for failure in failures:
        failure["date"] = "2026-08-04"
    atomic_write_bytes(report / "failures.jsonl", jsonl_bytes(failures))


if __name__ == "__main__":
    main()
