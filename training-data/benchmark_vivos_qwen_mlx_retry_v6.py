"""Run the preregistered Qwen MLX retry-v6 validation and production plan."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_vivos_qwen_mlx_batch import json_bytes, jsonl_bytes
from benchmark_vivos_qwen_mlx_batch_v2 import RssSampler, runtime_environment, system_state
from qa_vivos_full import CORPUS_THRESHOLDS, ROW_THRESHOLDS, select_best_passing_candidate
from qwen_mlx_compaction import generate_lanes, rng_contract, row_root_digest
from qwen_mlx_recurrent import FunctionalCodePredictor
from synthesize_vivos import (
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    atomic_write_bytes,
    atomic_write_wav,
    canonical_json,
    git_commit,
    immutable_write,
    package_version,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_mlx_snapshot,
)


SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_policy_v6"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_row_v6"
PRODUCTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_production_v6"
EXTERNAL_NAME = "vivos_qwen3_tts_mlx_retry_v6"
PRODUCTION_NAME = "vivos_qwen3_tts_mlx_retry_v6_full"
REPORT_NAME = "2026-08-04_qwen_mlx_retry_v6"
CAMPAIGN_REVISION = "hibiki-vivos-qwen3-tts-mlx-retry-v6"
SYNTHESIS_REVISION = "qwen3-tts-1.7b-base-bf16-mlx-b8-compact-recurrent-rp105-v6"
COHORT_SEED = "hibiki-vivos-qwen3-tts-mlx-retry-v6-untouched-cohort"
ATTEMPTS = (
    {"name": "attempt0_t08", "attempt": 0, "temperature": 0.8},
    {"name": "retry1_t07", "attempt": 1, "temperature": 0.7},
    {"name": "retry2_t08", "attempt": 2, "temperature": 0.8},
)
GENERATION_BASE = {
    "api": "repository qwen_mlx_compaction.generate_lanes",
    "batch_size": 8,
    "active_lane_compaction": True,
    "code_predictor": "exact fixed-width compiled functional recurrence",
    "length_scheduler": "none; rejected v5 scheduler is not used",
    "max_tokens": 2048,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty_requested": 1.05,
    "repetition_penalty_effective": 1.05,
    "lang_code": "English",
    "stream": False,
    "weight_dtype": "bfloat16",
    "synthesis_revision": SYNTHESIS_REVISION,
}
RETRY_WORD_ERRORS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare-validation")
    prepare.add_argument("full_plan", type=Path)
    prepare.add_argument("--tts-root", type=Path, required=True)
    prepare.add_argument("--gender-file", type=Path, required=True)
    prepare.add_argument("--out-root", type=Path, required=True)
    run = actions.add_parser("run-validation")
    run.add_argument("policy", type=Path)
    run.add_argument("--out-root", type=Path, required=True)
    run.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--retry-ids", type=Path)
    select = actions.add_parser("select")
    select.add_argument("policy", type=Path)
    select.add_argument("--external-root", type=Path, required=True)
    select.add_argument("--qa-root", type=Path, required=True)
    select.add_argument("--report-dir", type=Path, required=True)
    select.add_argument("--through-round", type=int, choices=(0, 1, 2), required=True)
    production = actions.add_parser("prepare-production")
    production.add_argument("policy", type=Path)
    production.add_argument("full_plan", type=Path)
    production.add_argument("selection_report", type=Path)
    production.add_argument("--out-root", type=Path, required=True)
    validate = actions.add_parser("validate-production")
    validate.add_argument("production_plan", type=Path)
    generate = actions.add_parser("run-production")
    generate.add_argument("production_plan", type=Path)
    generate.add_argument("--round", type=int, choices=(0, 1, 2), required=True)
    generate.add_argument("--retry-ids", type=Path)
    archive = actions.add_parser("archive")
    archive.add_argument("policy", type=Path)
    archive.add_argument("--external-root", type=Path, required=True)
    archive.add_argument("--qa-root", type=Path, required=True)
    archive.add_argument("--report-dir", type=Path, required=True)
    archive.add_argument("--through-round", type=int, choices=(0, 1, 2), required=True)
    return parser.parse_args()


def attest(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def script_attest() -> dict[str, str]:
    return attest(Path(__file__))


def source_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode())


def history_paths(root: Path) -> list[Path]:
    relative = [
        "vivos_qwen3_tts_mlx_pilot_v1/pilot_plan.jsonl",
        "vivos_qwen3_tts_mlx_pilot_v2/pilot_plan.jsonl",
        "vivos_qwen3_tts_mlx_pilot_v3/pilot_plan.jsonl",
        "vivos_qwen3_tts_mlx_batch_v1_benchmark_2026-08-04/cohort_plan.jsonl",
        "vivos_qwen3_tts_mlx_batch_v2_superseded_preflight_2026-08-04/cohort_plan.jsonl",
        "vivos_qwen3_tts_mlx_batch_v2r1_benchmark_2026-08-04/cohort_plan.jsonl",
        "vivos_qwen3_tts_mlx_efficiency_v3/bf16_n64/raw_results.jsonl",
        "vivos_qwen3_tts_mlx_recurrent_v4/eager_n64/raw_results.jsonl",
        "vivos_qwen3_tts_mlx_compaction_v5/throughput64.jsonl",
        "vivos_qwen3_tts_mlx_compaction_v5/quality64.jsonl",
    ]
    paths = [(root / value).resolve() for value in relative]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing v1-v5 history artifacts: {missing}")
    return paths


def collect_ids(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        row_id = value.get("id")
        if isinstance(row_id, str) and row_id.startswith("vivos:"):
            output.add(row_id)
        for child in value.values():
            collect_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_ids(child, output)


def history_ids(paths: list[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            collect_ids(row, output)
    return output


def read_genders(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        speaker, gender = line.split()
        values[speaker] = gender
    return values


def quantiles(rows: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            len(str(row["text_en"]).split()),
            len(str(row["text_en"])),
            float(row["source_audio"]["duration_s"]),
            str(row["id"]),
        ),
    )
    if len(ordered) < count:
        raise RuntimeError("Speaker lacks eight untouched rows")
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def group_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    groups = []
    for speaker in sorted(by_speaker):
        subset = by_speaker[speaker]
        for number in range(0, len(subset), 8):
            chunk = subset[number : number + 8]
            ids = [str(row["id"]) for row in chunk]
            groups.append(
                {
                    "group_id": f"{prefix}_{speaker}_{number // 8:04d}_{sha256_bytes(chr(0).join(ids).encode())[:12]}",
                    "speaker_id": speaker,
                    "ids": ids,
                }
            )
    return groups


def prepare_validation(args: argparse.Namespace) -> None:
    out = args.out_root.expanduser().resolve()
    if out.name != EXTERNAL_NAME:
        raise RuntimeError(f"Validation root must be named {EXTERNAL_NAME}")
    full_path = args.full_plan.expanduser().resolve()
    full = read_jsonl(full_path)
    if len(full) != 10_950 or len({row["id"] for row in full}) != len(full):
        raise RuntimeError("Expected the frozen 10,950-row full source plan")
    histories = history_paths(args.tts_root.expanduser().resolve())
    excluded = history_ids(histories)
    genders = read_genders(args.gender_file.expanduser().resolve())
    eligible: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full:
        if row["id"] not in excluded:
            by_speaker[str(row["speaker_id"])].append(row)
    for speaker, rows in by_speaker.items():
        if len(rows) >= 8:
            eligible[(str(rows[0]["eligibility_split"]), genders[speaker])].append(speaker)
    quotas = {("train", "f"): 3, ("train", "m"): 1, ("dev", "f"): 1, ("dev", "m"): 3}
    speakers = []
    for category, count in quotas.items():
        candidates = sorted(
            eligible[category],
            key=lambda speaker: sha256_bytes(
                f"{COHORT_SEED}\0{category[0]}\0{category[1]}\0{speaker}".encode()
            ),
        )
        if len(candidates) < count:
            raise RuntimeError(f"Cannot satisfy frozen speaker quota {category}: {count}")
        speakers.extend(candidates[:count])
    selected = []
    for speaker in sorted(speakers):
        selected.extend(quantiles(by_speaker[speaker]))
    if len(selected) != 64 or {row["id"] for row in selected} & excluded:
        raise RuntimeError("Untouched cohort construction failed")
    cohort_path = out / "cohort.jsonl"
    immutable_write(cohort_path, jsonl_bytes(selected))
    prior_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full:
        if row["id"] in excluded:
            prior_by_speaker[str(row["speaker_id"])].append(row)
    warm_speaker = min(
        (speaker for speaker, rows in prior_by_speaker.items() if len(rows) >= 8),
        key=lambda speaker: sha256_bytes(f"{COHORT_SEED}\0warmup\0{speaker}".encode()),
    )
    warmup = sorted(prior_by_speaker[warm_speaker], key=lambda row: str(row["id"]))[:8]
    warmup_path = out / "warmup.jsonl"
    immutable_write(warmup_path, jsonl_bytes(warmup))
    policy = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_freeze": git_commit(),
        "command": sys.argv,
        "script": script_attest(),
        "helpers": [
            attest(Path(__file__).with_name("qwen_mlx_compaction.py")),
            attest(Path(__file__).with_name("qwen_mlx_recurrent.py")),
            attest(Path(__file__).with_name("qa_vivos_full.py")),
            attest(Path(__file__).with_name("qa_vivos_qwen_mlx_efficiency_v3.py")),
        ],
        "source_plan": attest(full_path),
        "history": [attest(path) for path in histories],
        "excluded_unique_ids": len(excluded),
        "cohort": attest(cohort_path),
        "warmup": attest(warmup_path),
        "cohort_contract": {
            "rows": 64,
            "speakers": sorted(speakers),
            "rows_per_speaker": 8,
            "split_counts": {
                split: sum(row["eligibility_split"] == split for row in selected)
                for split in ("train", "dev")
            },
            "gender_counts": {
                gender: sum(genders[str(row["speaker_id"])] == gender for row in selected)
                for gender in ("f", "m")
            },
            "selection": "SHA256-ranked fixed speaker quotas; eight target-word-count quantiles per speaker after exact v1-v5/pilot exclusion",
            "disjoint_from_every_attested_history": True,
        },
        "campaign_revision": CAMPAIGN_REVISION,
        "rng": rng_contract(CAMPAIGN_REVISION),
        "synthesis": GENERATION_BASE,
        "attempt_order": list(ATTEMPTS),
        "retry_policy": {
            "trigger": "no passing candidate under every frozen row gate, or selected integer ASR word errors >= 4",
            "word_errors_min": RETRY_WORD_ERRORS,
            "maximum_new_rounds": 2,
            "selection": "only candidates passing every frozen row gate; minimum integer word errors, then row WER, then immutable attempt order",
            "acceptance": "accept every row with a passing candidate",
            "stop": "stop immediately when selected corpus WER <= 0.08, median speaker cosine >= 0.90, zero prompt leaks, and every accepted waveform/row gate passes",
            "adaptive_changes": "forbidden after freeze",
        },
        "thresholds": {"row": ROW_THRESHOLDS, "corpus": CORPUS_THRESHOLDS},
        "model": {"id": MLX_MODEL_ID, "revision": MLX_MODEL_REVISION},
    }
    policy_path = out / "policy.json"
    immutable_write(policy_path, json_bytes(policy))
    print(json.dumps({"policy": attest(policy_path), "cohort": policy["cohort_contract"]}, indent=2))


def load_policy(path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    policy = json.loads(path.read_text(encoding="utf-8"))
    if path.name != "policy.json" or path.parent.name != EXTERNAL_NAME:
        raise RuntimeError("Unexpected v6 policy namespace")
    if policy.get("schema_version") != SCHEMA or policy.get("attempt_order") != list(ATTEMPTS):
        raise RuntimeError("V6 policy contract changed")
    cohort_path = Path(policy["cohort"]["path"])
    warmup_path = Path(policy["warmup"]["path"])
    if attest(cohort_path) != policy["cohort"] or attest(warmup_path) != policy["warmup"]:
        raise RuntimeError("V6 cohort or warmup changed")
    cohort = read_jsonl(cohort_path)
    warmup = read_jsonl(warmup_path)
    if len(cohort) != 64 or len({row["id"] for row in cohort}) != 64 or len(warmup) != 8:
        raise RuntimeError("V6 frozen scope changed")
    historical = history_ids([Path(item["path"]) for item in policy["history"]])
    if {row["id"] for row in cohort} & historical:
        raise RuntimeError("V6 cohort is no longer disjoint from attested history")
    return path, policy, cohort, warmup


def load_model() -> tuple[Any, Path, dict[str, str]]:
    from huggingface_hub import snapshot_download
    from mlx_audio.tts.utils import load_model as load

    root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    return load(root), root, verify_mlx_snapshot(root)


def save_npy(path: Path, value: Any, np: Any) -> None:
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


def validate_group(path: Path, group: dict[str, Any], attempt: dict[str, Any], policy: dict[str, str]) -> dict[str, Any]:
    record = json.loads((path / "group.json").read_text(encoding="utf-8"))
    if record.get("group") != group or record.get("attempt_config") != attempt or record.get("policy") != policy:
        raise RuntimeError(f"Resume contract mismatch: {path}")
    for row in record["rows"]:
        if sha256_file(Path(row["output_wav"])) != row["audio_sha256"]:
            raise RuntimeError(f"Changed v6 WAV: {row['id']}")
        if sha256_file(Path(row["codes"])) != row["codes_sha256"]:
            raise RuntimeError(f"Changed v6 codes: {row['id']}")
    return record


def run_group(
    model: Any,
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    output: Path,
    attempt: dict[str, Any],
    policy_record: dict[str, str],
    mx: Any,
    np: Any,
    sf: Any,
    adapter: Any,
) -> dict[str, Any]:
    if output.exists():
        return validate_group(output, group, attempt, policy_record)
    from mlx_audio.utils import load_audio

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{group['group_id']}.", dir=output.parent))
    try:
        reference = rows[0]["reference"]
        ref_audio = load_audio(reference["reference_audio_path"], sample_rate=model.sample_rate)
        generation = {**GENERATION_BASE, "temperature": attempt["temperature"]}
        before = system_state()
        mx.reset_peak_memory()
        with RssSampler() as rss:
            prepare_started = time.monotonic()
            generated = generate_lanes(
                model,
                rows,
                ref_audio=ref_audio,
                ref_text=reference["reference_text_vi"],
                generation=generation,
                campaign_revision=CAMPAIGN_REVISION,
                attempts=[attempt["attempt"]] * len(rows),
                compact=True,
                adapter=adapter,
            )
            prepare_and_generation = time.monotonic() - prepare_started
            decode_started = time.monotonic()
            values = []
            for codes in generated.codes:
                audio = model._decode_generated_codes(codes)
                mx.eval(audio)
                values.append((np.asarray(audio, dtype=np.float32).reshape(-1), np.asarray(mx.stack(codes, axis=1))))
            decode_seconds = time.monotonic() - decode_started
        after = system_state()
        output_rows = []
        for row, (audio, codes) in zip(rows, values):
            stem = str(row["id"]).replace(":", "_")
            wav = temporary / "wavs" / f"{stem}.wav"
            code = temporary / "codes" / f"{stem}.npy"
            atomic_write_wav(wav, audio, model.sample_rate, sf)
            save_npy(code, codes, np)
            output_rows.append(
                {
                    "schema_version": ROW_SCHEMA,
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "eligibility_split": row["eligibility_split"],
                    "text_en": row["text_en"],
                    "source_audio": row["source_audio"],
                    "reference": row["reference"],
                    "source_plan_row_sha256": source_hash(row),
                    "attempt": attempt["attempt"],
                    "attempt_name": attempt["name"],
                    "output_wav": str(output / "wavs" / wav.name),
                    "audio_sha256": sha256_file(wav),
                    "codes": str(output / "codes" / code.name),
                    "codes_sha256": sha256_file(code),
                    "sample_rate_hz": model.sample_rate,
                    "num_samples": int(audio.size),
                    "duration_s": audio.size / model.sample_rate,
                    "token_count": len(codes),
                }
            )
        timing = {
            "prepare_seconds": max(0.0, prepare_and_generation - generated.generation_seconds),
            "generation_seconds": generated.generation_seconds,
            "decode_seconds": decode_seconds,
            "wall_seconds": prepare_and_generation + decode_seconds,
            "prefill_seconds": generated.prefill_seconds,
        }
        record = {
            "schema_version": SCHEMA,
            "policy": policy_record,
            "attempt_config": attempt,
            "group": group,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "timing": timing,
            "lane_accounting": {
                "talker_lane_steps": generated.talker_lane_steps,
                "useful_talker_lane_steps": generated.useful_talker_lane_steps,
                "predictor_lane_steps": generated.predictor_lane_steps,
                "useful_predictor_lane_steps": generated.useful_predictor_lane_steps,
                "active_widths": generated.active_widths,
                "stop_reasons": generated.stop_reasons,
            },
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


def retry_rows(path: Path | None, allowed: set[str], expected_round: int) -> tuple[list[str], dict[str, str] | None]:
    if expected_round == 0:
        if path is not None:
            raise RuntimeError("Attempt 0 cannot take retry ids")
        return sorted(allowed), None
    if path is None:
        raise RuntimeError("Retry rounds require the frozen selector manifest")
    path = path.expanduser().resolve()
    rows = read_jsonl(path)
    ids = [str(row.get("id", "")) for row in rows]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise RuntimeError("Invalid v6 retry manifest")
    if any(row.get("retry_round") != expected_round for row in rows):
        raise RuntimeError("Retry manifest round changed")
    return ids, attest(path)


def execute(
    policy_path: Path,
    policy: dict[str, Any],
    cohort: list[dict[str, Any]],
    warmup: list[dict[str, Any]],
    output_root: Path,
    round_index: int,
    retry_path: Path | None,
) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    import soundfile as sf

    attempt = ATTEMPTS[round_index]
    by_id = {str(row["id"]): row for row in cohort}
    ids, retry_record = retry_rows(retry_path, set(by_id), round_index)
    selected = [by_id[row_id] for row_id in ids]
    groups = group_rows(selected, attempt["name"])
    output = output_root / attempt["name"]
    candidate_path = output / "candidate.json"
    if candidate_path.is_file():
        verify_candidate(output)
        print(f"Validated existing {candidate_path}; no generation repeated")
        return
    model, model_root, snapshot = load_model()
    mx.eval(model.parameters())
    adapter = FunctionalCodePredictor(model, mx, nn, compiled=True)
    policy_record = attest(policy_path)
    warm_group = group_rows(warmup, f"warm_{attempt['name']}")[0]
    warm_root = Path(tempfile.mkdtemp(prefix="hibiki-v6-warmup-"))
    warm_started = time.monotonic()
    try:
        run_group(model, warmup, warm_group, warm_root / warm_group["group_id"], attempt, policy_record, mx, np, sf, adapter)
    finally:
        shutil.rmtree(warm_root, ignore_errors=True)
    warm_seconds = time.monotonic() - warm_started
    mx.clear_cache()
    records = []
    for number, group in enumerate(groups, 1):
        record = run_group(
            model,
            [by_id[row_id] for row_id in group["ids"]],
            group,
            output / "groups" / group["group_id"],
            attempt,
            policy_record,
            mx,
            np,
            sf,
            adapter,
        )
        records.append(record)
        print(f"[{number}/{len(groups)}] {group['group_id']}", flush=True)
    flat = [row for record in records for row in record["rows"]]
    wall = sum(record["timing"]["wall_seconds"] for record in records)
    generation = sum(record["timing"]["generation_seconds"] for record in records)
    decode = sum(record["timing"]["decode_seconds"] for record in records)
    talker = sum(record["lane_accounting"]["talker_lane_steps"] for record in records)
    useful_talker = sum(record["lane_accounting"]["useful_talker_lane_steps"] for record in records)
    predictor = sum(record["lane_accounting"]["predictor_lane_steps"] for record in records)
    useful_predictor = sum(record["lane_accounting"]["useful_predictor_lane_steps"] for record in records)
    candidate = {
        "schema_version": SCHEMA,
        "candidate": attempt["name"],
        "attempt_config": attempt,
        "scope_rows": len(flat),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "script": script_attest(),
        "policy": policy_record,
        "retry": retry_record,
        "synthesis": {**GENERATION_BASE, "temperature": attempt["temperature"]},
        "campaign_revision": CAMPAIGN_REVISION,
        "rng": policy["rng"],
        "warmup": {"cohort": policy["warmup"], "seconds": warm_seconds, "excluded": True},
        "timing": {
            "wall_seconds": wall,
            "generation_seconds": generation,
            "decode_seconds": decode,
            "rows_per_minute": 60 * len(flat) / wall,
            "generation_rows_per_minute": 60 * len(flat) / generation,
        },
        "lane_accounting": {
            "talker_lane_steps": talker,
            "useful_talker_lane_steps": useful_talker,
            "predictor_lane_steps": predictor,
            "useful_predictor_lane_steps": useful_predictor,
            "talker_dead_lane_steps": talker - useful_talker,
            "predictor_dead_lane_steps": predictor - useful_predictor,
        },
        "memory": {
            "peak_mlx_memory_bytes": max(record["peak_mlx_memory_bytes"] for record in records),
            "peak_process_rss_bytes": max(record["peak_process_rss_bytes"] for record in records),
        },
        "thermal": [record["system_after"]["pmset_therm"] for record in records],
        "model": {"id": MLX_MODEL_ID, "revision": MLX_MODEL_REVISION, "root": str(model_root), "files_sha256": snapshot},
        "adapter": adapter.timing_report(),
        "raw_results": str(output / "raw_results.jsonl"),
    }
    atomic_write_bytes(output / "raw_results.jsonl", jsonl_bytes(records))
    immutable_write(candidate_path, json_bytes(candidate))
    verify_candidate(output)
    print(json.dumps({"candidate": attempt["name"], **candidate["timing"]}, indent=2))


def verify_candidate(root: Path) -> dict[str, Any]:
    candidate = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    records = read_jsonl(root / "raw_results.jsonl")
    ids = []
    for record in records:
        current = validate_group(root / "groups" / record["group"]["group_id"], record["group"], candidate["attempt_config"], candidate["policy"])
        ids.extend(str(row["id"]) for row in current["rows"])
    if len(ids) != candidate["scope_rows"] or len(ids) != len(set(ids)):
        raise RuntimeError("Candidate resume validation is not an exact row bijection")
    return candidate


def run_validation(args: argparse.Namespace) -> None:
    policy_path, policy, cohort, warmup = load_policy(args.policy)
    out = args.out_root.expanduser().resolve()
    if out.name != EXTERNAL_NAME or out != policy_path.parent:
        raise RuntimeError("Validation output must be the frozen policy root")
    execute(policy_path, policy, cohort, warmup, out, args.round, args.retry_ids)


def load_qa(root: Path, name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / name
    report = json.loads((path / "qa_report.json").read_text(encoding="utf-8"))
    rows = {str(row["id"]): row for row in read_jsonl(path / "metrics.jsonl")}
    if len(rows) != report["scope_rows"]:
        raise RuntimeError(f"QA scope mismatch: {name}")
    return report, rows


def select_round(args: argparse.Namespace) -> None:
    policy_path, policy, cohort, _ = load_policy(args.policy)
    external = args.external_root.expanduser().resolve()
    qa_root = args.qa_root.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.name != REPORT_NAME:
        raise RuntimeError(f"Report directory must be named {REPORT_NAME}")
    report_dir.mkdir(parents=True, exist_ok=True)
    by_id = {str(row["id"]): row for row in cohort}
    candidates_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = []
    qa_reports = []
    for attempt in ATTEMPTS[: args.through_round + 1]:
        candidate_root = external / attempt["name"]
        candidate = verify_candidate(candidate_root)
        qa_report, metrics = load_qa(qa_root, attempt["name"])
        candidates.append(candidate)
        qa_reports.append(qa_report)
        if set(metrics) - set(by_id):
            raise RuntimeError("QA contains an out-of-cohort row")
        for row_id, metric in metrics.items():
            candidates_by_id[row_id].append({**metric, "attempt": attempt["attempt"]})
    selections = []
    selected_metrics = []
    retry = []
    for row in cohort:
        row_id = str(row["id"])
        choices = candidates_by_id[row_id]
        selected = select_best_passing_candidate(choices, list(range(args.through_round + 1)))
        preserved = [
            {
                "candidate_id": f"{row_id}|{ATTEMPTS[int(metric['attempt'])]['name']}",
                "attempt": metric["attempt"],
                "audio_sha256": metric["audio_sha256"],
                "asr_word_errors": metric["asr_word_errors"],
                "asr_reference_words": metric["asr_reference_words"],
                "asr_wer": metric["asr_wer"],
                "speaker_cosine": metric["speaker_cosine"],
                "failure_reasons": metric["failure_reasons"],
                "row_gate_pass": metric.get("row_gate_pass") is True and metric["failure_reasons"] == [],
                "output_wav": metric["output_wav"],
            }
            for metric in choices
        ]
        selection = {
            "id": row_id,
            "status": "accepted" if selected else "rejected",
            "selected_attempt": selected["attempt"] if selected else None,
            "selected_candidate_id": f"{row_id}|{ATTEMPTS[int(selected['attempt'])]['name']}" if selected else None,
            "selected_word_errors": selected["asr_word_errors"] if selected else None,
            "selected_reference_words": selected["asr_reference_words"] if selected else None,
            "selected_wer": selected["asr_wer"] if selected else None,
            "selected_speaker_cosine": selected["speaker_cosine"] if selected else None,
            "selected_prompt_leak_matches": selected["prompt_leak"]["reference_only_3gram_match_count"] if selected else None,
            "selected_output_wav": selected["output_wav"] if selected else None,
            "selected_audio_sha256": selected["audio_sha256"] if selected else None,
            "attempts_preserved": preserved,
        }
        selections.append(selection)
        if selected:
            selected_metrics.append(selected)
        if args.through_round < 2 and (
            selected is None or int(selected["asr_word_errors"]) >= RETRY_WORD_ERRORS
        ):
            retry.append(
                {
                    "id": row_id,
                    "retry_round": args.through_round + 1,
                    "trigger": "no_passing_candidate" if selected is None else "selected_word_errors_gte_4",
                    "selected_word_errors": selected["asr_word_errors"] if selected else None,
                    "selection_through_round": args.through_round,
                }
            )
    errors = sum(int(metric["asr_word_errors"]) for metric in selected_metrics)
    words = sum(int(metric["asr_reference_words"]) for metric in selected_metrics)
    wer = errors / words if words else None
    cosine = median(float(metric["speaker_cosine"]) for metric in selected_metrics) if selected_metrics else None
    leaks = sum(int(metric["prompt_leak"]["reference_only_3gram_match_count"]) for metric in selected_metrics)
    checks = {
        "selected_corpus_wer": wer is not None and wer <= 0.08,
        "selected_speaker_cosine_median": cosine is not None and cosine >= 0.90,
        "zero_selected_prompt_leaks": leaks == 0,
        "all_accepted_candidates_pass_every_row_gate": all(
            metric.get("row_gate_pass") is True
            and metric["failure_reasons"] == []
            and float(metric["asr_wer"]) <= ROW_THRESHOLDS["asr_wer_max"]
            for metric in selected_metrics
        ),
    }
    decision = "go" if all(checks.values()) else ("continue" if args.through_round < 2 else "no_go")
    if decision == "go":
        retry = []
    generation_wall = sum(float(candidate["timing"]["wall_seconds"]) for candidate in candidates)
    qa_timings = []
    for attempt in ATTEMPTS[: args.through_round + 1]:
        timing_path = report_dir / f"qa_{attempt['name']}_timing.json"
        qa_timings.append(json.loads(timing_path.read_text(encoding="utf-8")))
    qa_wall = sum(float(value["wall_seconds"]) for value in qa_timings)
    attempt0_wall = float(candidates[0]["timing"]["wall_seconds"])
    attempt0_qa = float(qa_timings[0]["wall_seconds"])
    attempt0_selected = sum(
        select_best_passing_candidate(candidates_by_id[row["id"]][:1], [0]) is not None
        for row in cohort
    )
    selection_path = report_dir / f"selection_rows_round{args.through_round}.jsonl"
    immutable_write(selection_path, jsonl_bytes(selections))
    retry_path = report_dir / f"retry_round{args.through_round + 1}.jsonl"
    if retry:
        immutable_write(retry_path, jsonl_bytes(retry))
    report = {
        "schema_version": SCHEMA,
        "policy": attest(policy_path),
        "through_round": args.through_round,
        "attempts": [attest(external / value["name"] / "candidate.json") for value in ATTEMPTS[: args.through_round + 1]],
        "qa": [attest(qa_root / value["name"] / "qa_report.json") for value in ATTEMPTS[: args.through_round + 1]],
        "selection_rule": policy["retry_policy"]["selection"],
        "decision": decision,
        "machine_checks": checks,
        "accepted_rows": len(selected_metrics),
        "rejected_rows": len(cohort) - len(selected_metrics),
        "selected_attempt_counts": {
            str(index): sum(metric["attempt"] == index for metric in selected_metrics)
            for index in range(args.through_round + 1)
        },
        "metrics": {
            "asr_word_errors": errors,
            "asr_reference_words": words,
            "asr_wer": wer,
            "speaker_cosine_median": cosine,
            "prompt_leak_matches": leaks,
        },
        "throughput": {
            "attempt0_generated_rows_per_minute": 60 * 64 / attempt0_wall,
            "attempt0_accepted_rows_per_minute_generation_only": 60 * attempt0_selected / attempt0_wall,
            "attempt0_accepted_rows_per_minute_including_qa": 60 * attempt0_selected / (attempt0_wall + attempt0_qa),
            "accepted_rows_per_minute_generation_and_retries": 60 * len(selected_metrics) / generation_wall,
            "accepted_rows_per_minute_including_retries_and_qa": 60 * len(selected_metrics) / (generation_wall + qa_wall),
            "generation_wall_seconds": generation_wall,
            "qa_wall_seconds": qa_wall,
            "total_wall_seconds": generation_wall + qa_wall,
            "seconds_per_accepted_row_including_retries_and_qa": (generation_wall + qa_wall) / len(selected_metrics) if selected_metrics else None,
        },
        "next_retry": attest(retry_path) if retry else None,
        "selection_rows": attest(selection_path),
    }
    immutable_write(report_dir / f"selection_round{args.through_round}.json", json_bytes(report))
    print(json.dumps({key: report[key] for key in ("decision", "accepted_rows", "rejected_rows", "metrics", "throughput", "next_retry")}, indent=2))


def prepare_production(args: argparse.Namespace) -> None:
    policy_path, policy, _, _ = load_policy(args.policy)
    selection_path = args.selection_report.expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("decision") != "go" or selection.get("policy") != attest(policy_path):
        raise RuntimeError("Production requires the exact v6 go selection")
    full_path = args.full_plan.expanduser().resolve()
    if attest(full_path) != policy["source_plan"]:
        raise RuntimeError("Production source differs from the frozen v6 source")
    rows = read_jsonl(full_path)
    groups = group_rows(rows, "production")
    ids = [row_id for group in groups for row_id in group["ids"]]
    if len(rows) != 10_950 or len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in rows}:
        raise RuntimeError("Production groups are not an exact source-plan bijection")
    out = args.out_root.expanduser().resolve()
    if out.name != PRODUCTION_NAME:
        raise RuntimeError(f"Production root must be named {PRODUCTION_NAME}")
    plan = {
        "schema_version": PRODUCTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_at_prepare": git_commit(),
        "command": sys.argv,
        "policy": attest(policy_path),
        "validation_go": attest(selection_path),
        "source_plan": attest(full_path),
        "script": script_attest(),
        "helpers": policy["helpers"],
        "synthesis": policy["synthesis"],
        "attempt_order": policy["attempt_order"],
        "campaign_revision": CAMPAIGN_REVISION,
        "rng": policy["rng"],
        "model": policy["model"],
        "package_contract": {
            "mlx": package_version("mlx"),
            "mlx-audio": package_version("mlx-audio"),
        },
        "rows": len(rows),
        "speakers": len({row["speaker_id"] for row in rows}),
        "group_contract": "source-plan order within speaker; contiguous groups of at most B8; no length scheduler",
        "groups": groups,
        "output_root": str(out),
        "atomicity": "each complete group is staged and atomically renamed; attempt manifest is atomically rebuilt from validated groups",
        "resume": "validate group contract plus WAV/code hashes, then skip",
    }
    immutable_write(out / "production_plan.json", json_bytes(plan))
    validate_production_path(out / "production_plan.json")
    print(f"Prepared and validated {len(groups)} groups / {len(rows)} rows; generation not launched")


def validate_production_path(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    plan = json.loads(path.read_text(encoding="utf-8"))
    if path.name != "production_plan.json" or path.parent.name != PRODUCTION_NAME or plan.get("schema_version") != PRODUCTION_SCHEMA:
        raise RuntimeError("Unexpected production plan")
    policy_path, policy, _, _ = load_policy(Path(plan["policy"]["path"]))
    rows = read_jsonl(Path(plan["source_plan"]["path"]))
    expected = group_rows(rows, "production")
    ids = [row_id for group in plan["groups"] for row_id in group["ids"]]
    if (
        plan["policy"] != attest(policy_path)
        or plan["source_plan"] != attest(Path(plan["source_plan"]["path"]))
        or plan["synthesis"] != policy["synthesis"]
        or plan["attempt_order"] != policy["attempt_order"]
        or plan["groups"] != expected
        or len(ids) != 10_950
        or len(ids) != len(set(ids))
    ):
        raise RuntimeError("Production plan contract mismatch")
    return plan, rows


def validate_production(args: argparse.Namespace) -> None:
    plan, rows = validate_production_path(args.production_plan)
    print(json.dumps({"valid": True, "rows": len(rows), "groups": len(plan["groups"]), "plan": attest(args.production_plan)}, indent=2))


def run_production(args: argparse.Namespace) -> None:
    plan_path = args.production_plan.expanduser().resolve()
    plan, rows = validate_production_path(plan_path)
    policy_path, policy, _, warmup = load_policy(Path(plan["policy"]["path"]))
    ids, retry_record = retry_rows(args.retry_ids, {row["id"] for row in rows}, args.round)
    selected = {row_id for row_id in ids}
    subset = [row for row in rows if row["id"] in selected]
    execute(policy_path, policy, subset, warmup, plan_path.parent, args.round, args.retry_ids)
    candidate_root = plan_path.parent / ATTEMPTS[args.round]["name"]
    candidate = verify_candidate(candidate_root)
    manifest = {
        "schema_version": PRODUCTION_SCHEMA,
        "production_plan": attest(plan_path),
        "attempt_config": ATTEMPTS[args.round],
        "retry": retry_record,
        "candidate": attest(candidate_root / "candidate.json"),
        "rows": candidate["scope_rows"],
        "raw_results": attest(candidate_root / "raw_results.jsonl"),
    }
    atomic_write_bytes(plan_path.parent / f"generation_attempt{args.round}_manifest.json", json_bytes(manifest))


def archive(args: argparse.Namespace) -> None:
    policy_path, policy, cohort, _ = load_policy(args.policy)
    report = args.report_dir.expanduser().resolve()
    selection_path = report / f"selection_round{args.through_round}.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    external = args.external_root.expanduser().resolve()
    qa = args.qa_root.expanduser().resolve()
    atomic_write_bytes(report / "policy.json", policy_path.read_bytes())
    atomic_write_bytes(report / "cohort.jsonl", Path(policy["cohort"]["path"]).read_bytes())
    raw = []
    qa_summary = {}
    for attempt in ATTEMPTS[: args.through_round + 1]:
        candidate = json.loads((external / attempt["name"] / "candidate.json").read_text())
        raw.append(candidate)
        qa_summary[attempt["name"]] = json.loads((qa / attempt["name"] / "qa_report.json").read_text())
    atomic_write_bytes(report / "raw_timing.jsonl", jsonl_bytes(raw))
    atomic_write_bytes(report / "qa_summary.json", json_bytes(qa_summary))
    selected = {row["id"]: row for row in read_jsonl(Path(selection["selection_rows"]["path"]))}
    metrics_by_attempt = {
        index: {row["id"]: row for row in read_jsonl(qa / attempt["name"] / "metrics.jsonl")}
        for index, attempt in enumerate(ATTEMPTS[: args.through_round + 1])
    }
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["sample", "audio_file", "duration_s", "transcript_vi", "reference_en", "asr_output_en", "asr_wer", "asr_word_errors", "speaker_cosine", "failure_reasons", "attempt", "decision", "audio_sha256"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in cohort:
        decision = selected[row["id"]]
        index = decision["selected_attempt"]
        metric = metrics_by_attempt[index][row["id"]] if index is not None else metrics_by_attempt[0][row["id"]]
        writer.writerow(
            {
                "sample": row["id"],
                "audio_file": metric["output_wav"],
                "duration_s": metric["duration_s"],
                "transcript_vi": row["text_vi"],
                "reference_en": row["text_en"],
                "asr_output_en": metric["asr_transcript_en"],
                "asr_wer": metric["asr_wer"],
                "asr_word_errors": metric["asr_word_errors"],
                "speaker_cosine": metric["speaker_cosine"],
                "failure_reasons": ",".join(metric["failure_reasons"]),
                "attempt": index if index is not None else "rejected",
                "decision": decision["status"],
                "audio_sha256": metric["audio_sha256"],
            }
        )
    atomic_write_bytes(report / "translations.csv", buffer.getvalue().encode())
    commands = ["# Reproduction commands", "", "Pinned MLX: `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python`; pinned QA: `/Volumes/data/envs/hibiki-vivos-qa/bin/python`.", "", "```bash", *[" ".join(value["command"]) for value in raw], *[" ".join(json.loads((report / f"qa_{attempt['name']}_timing.json").read_text())["command"]) for attempt in ATTEMPTS[: args.through_round + 1]], "```", ""]
    atomic_write_bytes(report / "commands.md", "\n".join(commands).encode())
    environment = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "policy": attest(policy_path),
        "sources": [script_attest(), *policy["helpers"]],
        "packages": {name: package_version(name) for name in ("mlx", "mlx-audio", "numpy", "soundfile")},
        "model": raw[0]["model"],
    }
    atomic_write_bytes(report / "environment.json", json_bytes(environment))
    throughput = selection["throughput"]
    metrics = selection["metrics"]
    lines = [
        "# Qwen3-TTS MLX retry validation v6",
        "",
        "Date: 2026-08-04 · Apple M4 Pro / 48 GiB · untouched 64-row, 8-speaker validation cohort.",
        "",
        "| Stage | Rows | Generation s | Decode s | Total s | Rows/min | Talker/predictor lane steps | Peak MLX GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in raw:
        lanes = value["lane_accounting"]
        lines.append(
            f"| {value['candidate']} | {value['scope_rows']} | {value['timing']['generation_seconds']:.2f} | {value['timing']['decode_seconds']:.2f} | {value['timing']['wall_seconds']:.2f} | {value['timing']['rows_per_minute']:.2f} | {lanes['talker_lane_steps']}/{lanes['predictor_lane_steps']} | {value['memory']['peak_mlx_memory_bytes']/2**30:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Final decision: **{selection['decision'].upper()}**. Accepted {selection['accepted_rows']}/64; selected WER {metrics['asr_wer']:.5f} ({metrics['asr_word_errors']}/{metrics['asr_reference_words']}), median speaker cosine {metrics['speaker_cosine_median']:.5f}, prompt leaks {metrics['prompt_leak_matches']}.",
            "",
            f"Attempt-0 accepted throughput: {throughput['attempt0_accepted_rows_per_minute_generation_only']:.2f} rows/min generation-only and {throughput['attempt0_accepted_rows_per_minute_including_qa']:.2f} including QA. Final accepted throughput: {throughput['accepted_rows_per_minute_generation_and_retries']:.2f} rows/min including retry generation and {throughput['accepted_rows_per_minute_including_retries_and_qa']:.2f} including retry QA ({throughput['seconds_per_accepted_row_including_retries_and_qa']:.2f} s/accepted row).",
            "",
            "The policy, cohort, attempts, ASR transcripts, selection, exact commands, hashes, timing, lane accounting, memory, thermal snapshots, and failures are preserved here; WAV/code attempts remain on the external dataset disk.",
            "",
        ]
    )
    atomic_write_bytes(report / "metrics.md", "\n".join(lines).encode())
    failure = {
        "date": "2026-08-04",
        "phase": "untouched_retry_validation",
        "status": selection["decision"],
        "reason": "all frozen gates passed" if selection["decision"] == "go" else "maximum preregistered retry rounds exhausted without all corpus gates",
    }
    atomic_write_bytes(report / "failures.jsonl", jsonl_bytes([] if selection["decision"] == "go" else [failure]))


def main() -> None:
    args = parse_args()
    {
        "prepare-validation": prepare_validation,
        "run-validation": run_validation,
        "select": select_round,
        "prepare-production": prepare_production,
        "validate-production": validate_production,
        "run-production": run_production,
        "archive": archive,
    }[args.action](args)


if __name__ == "__main__":
    main()
