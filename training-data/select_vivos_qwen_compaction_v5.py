"""Select VIVOS Qwen attempts and compute quality-adjusted accepted throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from synthesize_vivos import immutable_write, read_jsonl, sha256_file


NON_ASR_FAILURES = {
    "unreadable",
    "finite",
    "nonzero",
    "rms",
    "clipping_ratio",
    "silence_ratio",
    "leading_silence",
    "trailing_silence",
    "duration_ratio",
    "prompt_leak",
    "speaker_cosine",
}


def attest(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def load_candidate(path: Path):
    root = path.expanduser().resolve()
    report = json.loads((root / "candidate.json").read_text())
    rows = {
        row["id"]: row
        for record in read_jsonl(root / "raw_results.jsonl")
        for row in record["rows"]
    }
    return report, rows


def load_qa(path: Path):
    root = path.expanduser().resolve()
    report = json.loads((root / "qa_report.json").read_text())
    rows = {row["id"]: row for row in read_jsonl(root / "metrics.jsonl")}
    return report, rows


def eligible(metric: dict) -> bool:
    return not (set(metric["failure_reasons"]) & NON_ASR_FAILURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-candidate", type=Path, required=True)
    parser.add_argument("--baseline-qa", type=Path, required=True)
    parser.add_argument("--attempt0-candidate", type=Path, required=True)
    parser.add_argument("--attempt0-qa", type=Path, required=True)
    parser.add_argument("--attempt1-candidate", type=Path, required=True)
    parser.add_argument("--attempt1-qa", type=Path, required=True)
    parser.add_argument("--qa0-timing", type=Path, required=True)
    parser.add_argument("--qa1-timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    args = parser.parse_args()
    baseline_candidate, _ = load_candidate(args.baseline_candidate)
    baseline_qa, baseline_metrics = load_qa(args.baseline_qa)
    attempt0_candidate, attempt0_rows = load_candidate(args.attempt0_candidate)
    attempt0_qa, attempt0_metrics = load_qa(args.attempt0_qa)
    attempt1_candidate, attempt1_rows = load_candidate(args.attempt1_candidate)
    attempt1_qa, attempt1_metrics = load_qa(args.attempt1_qa)
    if set(attempt0_metrics) != set(baseline_metrics) or set(attempt1_metrics) != set(attempt1_rows):
        raise RuntimeError("Selection scopes do not match")

    selections = []
    for row_id in sorted(attempt0_metrics):
        choices = []
        for attempt, metrics, rows in (
            (0, attempt0_metrics, attempt0_rows),
            (1, attempt1_metrics, attempt1_rows),
        ):
            metric = metrics.get(row_id)
            if metric is not None and eligible(metric):
                choices.append((metric["asr_wer"], attempt, metric, rows[row_id]))
        if choices:
            _, attempt, metric, row = min(choices, key=lambda value: (value[0], value[1]))
            selections.append(
                {
                    "id": row_id,
                    "decision": "accepted",
                    "selected_attempt": attempt,
                    "selected_audio_sha256": metric["audio_sha256"],
                    "selected_output_wav": row["output_wav"],
                    "selected_wer": metric["asr_wer"],
                    "selected_word_errors": metric["asr_word_errors"],
                    "reference_words": metric["asr_reference_words"],
                    "selected_speaker_cosine": metric["speaker_cosine"],
                    "selected_failure_reasons": metric["failure_reasons"],
                    "attempts_preserved": [
                        {
                            "attempt": candidate_attempt,
                            "audio_sha256": candidate_metric["audio_sha256"],
                            "wer": candidate_metric["asr_wer"],
                            "failure_reasons": candidate_metric["failure_reasons"],
                            "eligible": eligible(candidate_metric),
                        }
                        for candidate_attempt, candidate_metric in (
                            (0, attempt0_metrics[row_id]),
                            *(([(1, attempt1_metrics[row_id])]) if row_id in attempt1_metrics else []),
                        )
                    ],
                }
            )
        else:
            selections.append(
                {
                    "id": row_id,
                    "decision": "rejected",
                    "selected_attempt": None,
                    "attempts_preserved": [
                        {
                            "attempt": candidate_attempt,
                            "audio_sha256": candidate_metric["audio_sha256"],
                            "wer": candidate_metric["asr_wer"],
                            "failure_reasons": candidate_metric["failure_reasons"],
                            "eligible": False,
                        }
                        for candidate_attempt, candidate_metric in (
                            (0, attempt0_metrics[row_id]),
                            *(([(1, attempt1_metrics[row_id])]) if row_id in attempt1_metrics else []),
                        )
                    ],
                }
            )
    accepted = [row for row in selections if row["decision"] == "accepted"]
    selected_wer = sum(row["selected_word_errors"] for row in accepted) / sum(
        row["reference_words"] for row in accepted
    )
    selected_cosine = median(row["selected_speaker_cosine"] for row in accepted)
    qa_wall = json.loads(args.qa0_timing.read_text())["wall_seconds"] + json.loads(
        args.qa1_timing.read_text()
    )["wall_seconds"]
    generation0 = attempt0_candidate["timing"]["wall_seconds"]
    generation1 = attempt1_candidate["timing"]["wall_seconds"]
    attempt0_accepted = sum(eligible(metric) for metric in attempt0_metrics.values())
    baseline_accepted = sum(eligible(metric) for metric in baseline_metrics.values())
    baseline_rpm = (
        60 * baseline_accepted / baseline_candidate["timing"]["wall_seconds"]
    )
    attempt0_rpm = 60 * attempt0_accepted / generation0
    selected_rpm = 60 * len(accepted) / (generation0 + generation1)
    report = {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_selection_v5",
        "inputs": {
            name: attest(path / "candidate.json") if "candidate" in name else attest(path / "qa_report.json")
            for name, path in {
                "baseline_candidate": args.baseline_candidate,
                "baseline_qa": args.baseline_qa,
                "attempt0_candidate": args.attempt0_candidate,
                "attempt0_qa": args.attempt0_qa,
                "attempt1_candidate": args.attempt1_candidate,
                "attempt1_qa": args.attempt1_qa,
            }.items()
        },
        "selection_rule": "lowest row WER among attempts passing waveform, duration, prompt-leak, and speaker gates; attempt 0 wins exact WER ties",
        "paired_attempt0": {
            "baseline_wer": baseline_qa["asr_wer"],
            "candidate_wer": attempt0_qa["asr_wer"],
            "wer_delta": attempt0_qa["asr_wer"] - baseline_qa["asr_wer"],
            "baseline_median_cosine": baseline_qa["speaker_cosine_median"],
            "candidate_median_cosine": attempt0_qa["speaker_cosine_median"],
            "median_cosine_delta": attempt0_qa["speaker_cosine_median"] - baseline_qa["speaker_cosine_median"],
            "noninferiority_pass": (
                attempt0_qa["asr_wer"] - baseline_qa["asr_wer"] <= 0.01
                and attempt0_qa["speaker_cosine_median"] - baseline_qa["speaker_cosine_median"] >= -0.01
            ),
        },
        "selection": {
            "attempt0_rows": len(attempt0_metrics),
            "retried_rows": len(attempt1_metrics),
            "accepted_rows": len(accepted),
            "rejected_rows": len(selections) - len(accepted),
            "selected_attempt1_rows": sum(row["selected_attempt"] == 1 for row in accepted),
            "selected_wer": selected_wer,
            "selected_median_cosine": selected_cosine,
            "absolute_wer_gate": 0.08,
            "absolute_wer_pass": selected_wer <= 0.08,
            "zero_selected_prompt_leaks": all(
                "prompt_leak" not in row["selected_failure_reasons"] for row in accepted
            ),
        },
        "accepted_throughput": {
            "baseline_attempt0_accepted_rows_per_minute": baseline_rpm,
            "candidate_attempt0_accepted_rows_per_minute": attempt0_rpm,
            "candidate_attempt0_gain_vs_baseline": attempt0_rpm / baseline_rpm - 1,
            "candidate_generation_plus_retry_accepted_rows_per_minute": selected_rpm,
            "candidate_generation_plus_retry_and_qa_rows_per_minute": 60 * len(accepted) / (generation0 + generation1 + qa_wall),
            "attempt0_generation_wall_seconds": generation0,
            "retry_generation_wall_seconds": generation1,
            "qa_wall_seconds_separate": qa_wall,
        },
        "decision": "go" if selected_wer <= 0.08 else "no_go",
        "decision_reason": "selected corpus WER exceeds the frozen 0.08 gate" if selected_wer > 0.08 else "paired noninferiority and absolute gates pass",
        "rows": None,
    }
    immutable_write(args.rows_out.expanduser().resolve(), jsonl_bytes(selections))
    # Fill the row attestation only after the immutable row file exists.
    report["rows"] = attest(args.rows_out)
    immutable_write(args.out.expanduser().resolve(), json_bytes(report))
    print(json.dumps({"selection": report["selection"], "accepted_throughput": report["accepted_throughput"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
