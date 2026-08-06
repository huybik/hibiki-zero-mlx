"""CPU-only provenance validation for finalized retry-v6 VIVOS artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = REPO_ROOT / "training-data"
if str(TRAINING_DATA) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA))

from synthesize_vivos import canonical_json, read_jsonl, sha256_bytes, sha256_file  # noqa: E402
from benchmark_vivos_qwen_mlx_retry_v6 import production_attestation_path  # noqa: E402
from validate_vivos_qwen_production_v6 import (  # noqa: E402
    ATTEMPTS,
    attestation,
    audit_historical_validation,
    load_production_plan,
    validate_production_attempt,
)

FINAL_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_final_v6"
QA_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_qa_v6"
SELECTION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_production_selection_v6"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_row_v6"
WAIVER_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_manual_waiver_v6"
RNG_SCHEMA = "hibiki-qwen-mlx-row-rng-v1"
RNG_HELPER_SHA256 = "cb96149414e1c991c0ea29908b3d99a02dd73a12dcd849fde3d6e025eb5dbe82"
TERMINAL_NO_RETRY_POLICY = {
    "enabled": True,
    "reason": "user_requested_drop_after_attempt0_validation",
    "corpus_wer_pruning": {
        "objective": "minimum_rows_removed_to_pass_corpus_wer",
        "ranking": "descending_word_error_surplus_then_row_id",
        "threshold": 0.08,
    },
}


class IncompleteCampaign(RuntimeError):
    """The v6 campaign is valid so far but is not cache/release ready."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _path(record: dict[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or attestation(path) != record:
        raise RuntimeError(f"{label} is missing or changed: {path}")
    return path


def _canonical_sha(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode())


def _speaker_exclusions(
    source_rows: list[dict[str, Any]], record: dict[str, Any]
) -> dict[str, Any]:
    speaker_ids = record.get("speaker_ids", [])
    if not isinstance(speaker_ids, list) or speaker_ids != sorted(set(speaker_ids)):
        raise RuntimeError("Terminal speaker exclusions are not sorted and unique")
    available = {str(row["speaker_id"]) for row in source_rows}
    if not set(speaker_ids) <= available:
        raise RuntimeError("Terminal selection excludes an unknown speaker")
    expected = {
        "speaker_ids": speaker_ids,
        "rows": sum(str(row["speaker_id"]) in set(speaker_ids) for row in source_rows),
        "reason": "user_requested_quality_exclusion",
    }
    if record != expected:
        raise RuntimeError("Terminal speaker-exclusion scope changed")
    return expected


def _selection_scope_sha(
    policy: dict[str, Any],
    exclusions: dict[str, Any],
    terminal_policy: dict[str, Any] | None,
) -> str:
    value: object = policy["retry_policy"]
    if exclusions["speaker_ids"] or terminal_policy is not None:
        value = {"retry_policy": policy["retry_policy"]}
        if exclusions["speaker_ids"]:
            value["speaker_exclusions"] = exclusions
        if terminal_policy is not None:
            value["terminal_policy"] = terminal_policy
    return sha256_bytes(canonical_json(value).encode())


def _select_metric(candidates: list[dict[str, Any]], through_round: int) -> dict[str, Any] | None:
    passing = [
        metric
        for metric in candidates
        if metric.get("row_gate_pass") is True
        and metric.get("failure_reasons") == []
        and metric.get("asr_word_errors") is not None
        and metric.get("asr_wer") is not None
    ]
    return (
        min(
            passing,
            key=lambda metric: (
                int(metric["asr_word_errors"]),
                float(metric["asr_wer"]),
                int(metric["attempt"]),
            ),
        )
        if passing
        else None
    )


def _word_error_surplus(metric: dict[str, Any]) -> int:
    threshold = Fraction(str(TERMINAL_NO_RETRY_POLICY["corpus_wer_pruning"]["threshold"]))
    return (
        int(metric["asr_word_errors"]) * threshold.denominator
        - int(metric["asr_reference_words"]) * threshold.numerator
    )


def _terminal_prune_ids(selected: dict[str, dict[str, Any]]) -> list[str]:
    threshold = Fraction(str(TERMINAL_NO_RETRY_POLICY["corpus_wer_pruning"]["threshold"]))
    errors = sum(int(metric["asr_word_errors"]) for metric in selected.values())
    words = sum(int(metric["asr_reference_words"]) for metric in selected.values())
    ranked = sorted(selected.items(), key=lambda item: (-_word_error_surplus(item[1]), item[0]))
    pruned = []
    for row_id, metric in ranked:
        if words and errors * threshold.denominator <= words * threshold.numerator:
            break
        if _word_error_surplus(metric) <= 0:
            break
        pruned.append(row_id)
        errors -= int(metric["asr_word_errors"])
        words -= int(metric["asr_reference_words"])
    return pruned


def _terminal_exclusion() -> dict[str, str]:
    return {
        "kind": "terminal_corpus_wer_prune",
        "reason": TERMINAL_NO_RETRY_POLICY["reason"],
    }


def _terminal_pruning_record(row_ids: list[str]) -> dict[str, Any]:
    ordered = sorted(row_ids)
    return {
        "rows": len(ordered),
        "row_ids_sha256": sha256_bytes(canonical_json(ordered).encode()),
    }


def _production_contract(
    production_plan: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    values = load_production_plan(production_plan)
    _, plan, _, source_rows, contract = values
    script_records = [plan.get("script"), *plan.get("helpers", [])]
    if any(not isinstance(record, dict) for record in script_records):
        raise RuntimeError("Production plan script/helper contract is incomplete")
    for record in script_records:
        _path(record, "Production script/helper")
    validation_go = _json(_path(plan["validation_go"], "Validation GO"))
    common = {
        "source_audit_report": source_rows[0]["source_audit"]["report"],
        "source_audit_rows": source_rows[0]["source_audit"]["row_metrics"],
        "reference_map": source_rows[0]["reference_map"]["map"],
        "reference_report": source_rows[0]["reference_map"]["report"],
        "approval": source_rows[0]["approval"],
    }
    if any(
        row["source_audit"]["report"] != common["source_audit_report"]
        or row["source_audit"]["row_metrics"] != common["source_audit_rows"]
        or row["reference_map"]["map"] != common["reference_map"]
        or row["reference_map"]["report"] != common["reference_report"]
        or row["approval"] != common["approval"]
        for row in source_rows
    ):
        raise RuntimeError("Source plan rows disagree on source/reference campaign artifacts")
    for label, record in common.items():
        _path(record, label)
    source_report = _json(Path(common["source_audit_report"]["path"]))
    reference_report = _json(Path(common["reference_report"]["path"]))
    references = read_jsonl(Path(common["reference_map"]["path"]))
    if (
        plan.get("package_contract")
        != {key: contract["runtime"][key] for key in ("mlx", "mlx-audio")}
        or plan.get("model") != {key: contract["model"][key] for key in ("id", "revision")}
        or plan.get("script") not in contract["scripts"]
        or not all(
            helper in contract["scripts"]
            for helper in plan.get("helpers", [])
            if Path(helper["path"]).name in {"qwen_mlx_compaction.py", "qwen_mlx_recurrent.py"}
        )
        or contract["runtime"].get("mlx-audio_commit") != "2c9461f5d8315fa8e7013ab2729495b2bb83d384"
        or validation_go.get("decision") != "go"
        or not all(validation_go.get("machine_checks", {}).values())
        or validation_go.get("policy") != plan["policy"]
        or source_report.get("schema_version") != "hibiki_vivos_source_asr_mps_full_v1"
        or source_report.get("complete") is not True
        or source_report.get("rows") != len(source_rows)
        or source_report.get("speakers") != 46
        or source_report.get("row_metrics") != common["source_audit_rows"]
        or reference_report.get("schema_version")
        != "hibiki_vivos_qwen3_tts_mlx_v3_reference_map_v1"
        or any(
            row.get("schema_version") != "hibiki_vivos_qwen3_tts_mlx_v3_reference_map_v1"
            for row in references
        )
        or reference_report.get("status") != "complete"
        or reference_report.get("references") != references
        or len(references) != 46
    ):
        raise RuntimeError("Production model/package/script revision contract changed")
    return values


def _validate_manual_evidence(
    report: dict[str, Any], plan_path: Path, selection_report_path: Path
) -> dict[str, Any]:
    manual = report.get("manual_review", {})
    if (
        manual.get("missing")
        or manual.get("selected_sample_pass") is not True
        or manual.get("failures_and_rejections_review_complete") is not True
    ):
        raise IncompleteCampaign("Final QA manual review is incomplete")
    waiver_record = manual.get("waiver")
    review_record = manual.get("review_file")
    if waiver_record is not None:
        waiver_path = _path(waiver_record, "Manual-review waiver")
        waiver = _json(waiver_path)
        if (
            waiver.get("schema_version") != WAIVER_SCHEMA
            or waiver.get("waive_manual_review") is not True
            or waiver.get("production_plan") != attestation(plan_path)
            or waiver.get("selection_report") != attestation(selection_report_path)
            or waiver.get("required_candidates_sha256") != manual.get("required_candidates_sha256")
            or not str(waiver.get("rationale", "")).strip()
        ):
            raise RuntimeError("Manual-review waiver does not bind the finalized v6 scope")
        return {"mode": "explicit_waiver", "artifact": waiver_record}
    if review_record is None:
        raise IncompleteCampaign("Final QA has neither completed manual review nor waiver")
    _path(review_record, "Manual review")
    return {"mode": "completed_review", "artifact": review_record}


def _rng(policy: dict[str, Any], row_id: str, attempt: int) -> dict[str, Any]:
    payload = f"{RNG_SCHEMA}\0{policy['campaign_revision']}\0{row_id}\0attempt={attempt}".encode()
    return {
        "frozen_policy_rng": policy["rng"],
        "erratum": {
            "date": "2026-08-04",
            "classification": "prose-only provenance correction; frozen policy unchanged",
            "executable_root_formula": (
                "SHA256(f'{schema}\\0{campaign_revision}\\0{row_id}\\0attempt={attempt}')"
            ),
            "attempt2_is_distinct": True,
            "row_root_digest_helper_sha256": RNG_HELPER_SHA256,
        },
        "derived_row_root_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _metric_summary(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": metric["candidate_id"],
        "attempt": metric["attempt"],
        "row_gate_pass": metric["row_gate_pass"],
        "failure_reasons": metric["failure_reasons"],
        "asr_word_errors": metric["asr_word_errors"],
        "asr_reference_words": metric["asr_reference_words"],
        "asr_wer": metric["asr_wer"],
        "speaker_cosine": metric["speaker_cosine"],
        "prompt_leak_matches": metric["prompt_leak"]["reference_only_3gram_match_count"],
        "output_wav": metric["output_wav"],
        "audio_sha256": metric["audio_sha256"],
        "metric_sha256": _canonical_sha(metric),
    }


def _attempt_artifacts(
    plan_path: Path,
    plan: dict[str, Any],
    selection_report: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    through_round = int(selection_report.get("through_round", -1))
    if through_round not in (0, 1, 2):
        raise RuntimeError("Terminal selection has an invalid attempt round")
    attempts = selection_report.get("attempts")
    if not isinstance(attempts, list) or [row.get("attempt") for row in attempts] != list(
        range(through_round + 1)
    ):
        raise RuntimeError("Terminal selection does not bind every executed attempt in order")
    generated: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for attempt, record in enumerate(attempts):
        retry_record = record.get("retry_manifest")
        retry_path = (
            _path(retry_record, f"Attempt-{attempt} retry manifest") if retry_record else None
        )
        validation = validate_production_attempt(plan_path, attempt, retry_path)
        if validation["state"] not in {"complete", "complete_with_media_errors"}:
            raise IncompleteCampaign(f"Generation attempt {attempt} is incomplete")
        generation_manifest = plan_path.parent / f"generation_attempt{attempt}_manifest.json"
        candidate_root = plan_path.parent / ATTEMPTS[attempt]["name"]
        if record.get("generation_manifest") != attestation(generation_manifest):
            raise RuntimeError(f"Attempt-{attempt} generation manifest binding changed")
        qa_report_path = _path(record.get("qa_report", {}), f"Attempt-{attempt} QA report")
        qa_report = _json(qa_report_path)
        metrics_path = _path(qa_report.get("row_metrics", {}), f"Attempt-{attempt} QA metrics")
        qa_rows = read_jsonl(metrics_path)
        generation_rows = [row for group in validation["records"] for row in group["rows"]]
        group_by_id = {
            str(row["id"]): attestation(
                Path(str(validation["candidate"]["raw_results"])).resolve().parent
                / "groups"
                / group["group"]["group_id"]
                / "group.json"
            )
            for group in validation["records"]
            for row in group["rows"]
        }
        if (
            qa_report.get("schema_version") != QA_SCHEMA
            or qa_report.get("status") != "complete"
            or qa_report.get("attempt") != attempt
            or qa_report.get("attempt_name") != ATTEMPTS[attempt]["name"]
            or qa_report.get("production_plan") != attestation(plan_path)
            or qa_report.get("generation_manifest") != attestation(generation_manifest)
            or qa_report.get("retry_manifest") != retry_record
            or [row.get("id") for row in qa_rows] != [row.get("id") for row in generation_rows]
        ):
            raise RuntimeError(f"Attempt-{attempt} QA scope/provenance changed")
        by_id = {str(row["id"]): row for row in generation_rows}
        for metric in qa_rows:
            row_id = str(metric.get("id", ""))
            candidate_id = f"{row_id}|{ATTEMPTS[attempt]['name']}"
            candidate = by_id.get(row_id)
            binding = metric.get("binding", {}).get("candidate", {})
            metric_binding = metric.get("binding", {})
            binding_payload = {
                key: value for key, value in metric_binding.items() if key != "binding_sha256"
            }
            if (
                candidate is None
                or candidate.get("schema_version") != ROW_SCHEMA
                or metric.get("candidate_id") != candidate_id
                or metric.get("attempt") != attempt
                or metric.get("attempt_name") != ATTEMPTS[attempt]["name"]
                or metric.get("output_wav") != candidate.get("output_wav")
                or metric.get("audio_sha256") != candidate.get("audio_sha256")
                or metric.get("codes") != candidate.get("codes")
                or metric.get("codes_sha256") != candidate.get("codes_sha256")
                or binding.get("candidate_row_sha256") != _canonical_sha(candidate)
                or binding.get("attempt") != attempt
                or binding.get("attempt_name") != ATTEMPTS[attempt]["name"]
                or binding.get("audio_sha256") != candidate.get("audio_sha256")
                or binding.get("generation_manifest") != attestation(generation_manifest)
                or binding.get("candidate_record") != attestation(candidate_root / "candidate.json")
                or binding.get("raw_results") != attestation(candidate_root / "raw_results.jsonl")
                or binding.get("group_record") != group_by_id[row_id]
                or metric_binding.get("production_plan") != attestation(plan_path)
                or metric_binding.get("production_attestation")
                != attestation(production_attestation_path(plan_path))
                or metric_binding.get("policy") != plan["policy"]
                or metric_binding.get("source_plan") != plan["source_plan"]
                or metric_binding.get("source", {}).get("source_plan_row_sha256")
                != candidate.get("source_plan_row_sha256")
                or metric_binding.get("binding_sha256") != _canonical_sha(binding_payload)
            ):
                raise RuntimeError(f"Attempt-{attempt} QA candidate binding changed: {row_id}")
            if candidate_id in metrics:
                raise RuntimeError(f"Duplicate QA candidate: {candidate_id}")
            generated[candidate_id] = candidate
            metrics[candidate_id] = metric
        artifacts.append(
            {
                "attempt": attempt,
                "attempt_name": ATTEMPTS[attempt]["name"],
                "temperature": ATTEMPTS[attempt]["temperature"],
                "retry_manifest": retry_record,
                "generation_manifest": attestation(generation_manifest),
                "candidate": attestation(candidate_root / "candidate.json"),
                "raw_results": attestation(candidate_root / "raw_results.jsonl"),
                "group_records": validation["group_records"],
                "qa_report": attestation(qa_report_path),
                "qa_metrics": attestation(metrics_path),
            }
        )
    for attempt in range(through_round + 1, 3):
        root = plan_path.parent / ATTEMPTS[attempt]["name"]
        forbidden = [
            plan_path.parent / f"generation_attempt{attempt}_manifest.json",
            root / "candidate.json",
            root / "raw_results.jsonl",
        ]
        groups = root / "groups"
        if any(path.exists() for path in forbidden) or (groups.is_dir() and any(groups.iterdir())):
            raise RuntimeError(f"Unexecuted attempt {attempt} has terminal generation artifacts")
        artifacts.append(
            {
                "attempt": attempt,
                "attempt_name": ATTEMPTS[attempt]["name"],
                "temperature": ATTEMPTS[attempt]["temperature"],
                "state": "not_executed_after_terminal_go",
            }
        )
    return generated, metrics, artifacts


def validate_finalized(
    production_plan: Path,
    accepted_path: Path,
    selection_path: Path,
    qa_report_path: Path,
) -> dict[str, Any]:
    """Validate the complete v6 final scope without loading QA or Mimi models."""

    plan_path, plan, policy, source_rows, contract = _production_contract(production_plan)
    accepted_path = accepted_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    qa_report_path = qa_report_path.expanduser().resolve()
    report = _json(qa_report_path)
    status = str(report.get("status", ""))
    if status in {"pending_manual_review", ""}:
        raise IncompleteCampaign(f"Final QA status is {status or 'missing'}")
    if status != "go":
        raise RuntimeError(f"Final QA status is not GO: {status}")
    if (
        report.get("schema_version") != FINAL_SCHEMA
        or report.get("machine_selection_decision") != "go"
        or not all(report.get("machine_checks", {}).values())
        or report.get("production_plan") != attestation(plan_path)
    ):
        raise RuntimeError("Final QA report is not a machine-validated v6 GO")
    outputs = report.get("outputs", {})
    if outputs.get("accepted") != attestation(accepted_path):
        raise RuntimeError("Accepted manifest is not bound to final QA")
    if outputs.get("selection") != attestation(selection_path):
        raise RuntimeError("Selection manifest is not bound to final QA")
    rejected_path = _path(outputs.get("rejected", {}), "Rejected manifest")
    selected_candidates_path = _path(
        outputs.get("selected_candidates", {}), "Selected-candidate manifest"
    )
    manual_required_path = _path(
        outputs.get("manual_review_required", {}), "Manual-review requirement"
    )
    selection_report_path = _path(report.get("selection_report", {}), "Selection report")
    manual_evidence = _validate_manual_evidence(report, plan_path, selection_report_path)
    selection_report = _json(selection_report_path)
    exclusions = _speaker_exclusions(
        source_rows, selection_report.get("speaker_exclusions", {})
    )
    terminal_policy = selection_report.get("terminal_policy")
    if terminal_policy not in (None, TERMINAL_NO_RETRY_POLICY):
        raise RuntimeError("Terminal selection policy changed")
    if (
        selection_report.get("schema_version") != SELECTION_SCHEMA
        or selection_report.get("decision") != "go"
        or selection_report.get("next_retry") is not None
        or not all(selection_report.get("machine_checks", {}).values())
        or selection_report.get("production_plan") != attestation(plan_path)
        or selection_report.get("policy") != plan["policy"]
        or selection_report.get("selection_policy_sha256")
        != _selection_scope_sha(policy, exclusions, terminal_policy)
        or report.get("speaker_exclusions") != exclusions
        or report.get("terminal_policy") != terminal_policy
        or report.get("terminal_pruning") != selection_report.get("terminal_pruning")
    ):
        raise RuntimeError("Terminal selection report is not the exact v6 GO")
    terminal_selection_path = _path(
        selection_report.get("selection_rows", {}), "Terminal selection rows"
    )
    selection = read_jsonl(selection_path)
    terminal_selection = read_jsonl(terminal_selection_path)
    if selection != terminal_selection:
        raise RuntimeError("Final and terminal selection rows differ")
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    selected_candidates = read_jsonl(selected_candidates_path)
    source_by_id = {str(row["id"]): row for row in source_rows}
    source_order = [str(row["id"]) for row in source_rows]
    selection_by_id = {str(row.get("id", "")): row for row in selection}
    accepted_by_id = {str(row.get("id", "")): row for row in accepted}
    rejected_by_id = {str(row.get("id", "")): row for row in rejected}
    excluded_speakers = set(exclusions["speaker_ids"])
    if (
        len(source_by_id) != len(source_rows)
        or [str(row.get("id", "")) for row in selection] != source_order
        or len(selection_by_id) != len(selection)
        or len(accepted_by_id) != len(accepted)
        or len(rejected_by_id) != len(rejected)
        or set(accepted_by_id) & set(rejected_by_id)
        or set(accepted_by_id) | set(rejected_by_id) != set(source_by_id)
        or any(selection_by_id[row_id].get("status") != "accepted" for row_id in accepted_by_id)
        or any(selection_by_id[row_id].get("status") != "rejected" for row_id in rejected_by_id)
        or any(row["speaker_id"] in excluded_speakers for row in accepted)
    ):
        raise RuntimeError("Source, selection, accepted, and rejected partitions disagree")
    generated, metrics, attempts = _attempt_artifacts(plan_path, plan, selection_report)
    initially_selected = {}
    through_round = int(selection_report["through_round"])
    for selected in selection:
        row_id = str(selected["id"])
        if source_by_id[row_id]["speaker_id"] in excluded_speakers:
            continue
        choice = _select_metric(
            [metrics[candidate["candidate_id"]] for candidate in selected["candidates"]],
            through_round,
        )
        if choice is not None:
            initially_selected[row_id] = choice
    pruned_ids = _terminal_prune_ids(initially_selected) if terminal_policy else []
    if selection_report.get("terminal_pruning") != _terminal_pruning_record(pruned_ids):
        raise RuntimeError("Terminal corpus-WER pruning scope changed")
    pruned = set(pruned_ids)
    for selected in selection:
        source = source_by_id[str(selected.get("id", ""))]
        expected_exclusion = (
            {
                "kind": "speaker",
                "speaker_id": source["speaker_id"],
                "reason": exclusions["reason"],
            }
            if source["speaker_id"] in excluded_speakers
            else (_terminal_exclusion() if str(selected["id"]) in pruned else None)
        )
        if (
            selected.get("exclusion") != expected_exclusion
            or (
                expected_exclusion is not None
                and (
                    selected.get("status") != "rejected"
                    or selected.get("selected_candidate_id") is not None
                    or selected.get("selected_attempt") is not None
                )
            )
        ):
            raise RuntimeError(f"Terminal exclusion changed: {selected.get('id')}")
        for candidate in selected.get("candidates", []):
            metric = metrics.get(str(candidate.get("candidate_id", "")))
            if metric is None or candidate != _metric_summary(metric):
                raise RuntimeError(
                    f"Terminal selection candidate provenance changed: {selected.get('id')}"
                )
    for row_id, row in rejected_by_id.items():
        selected = selection_by_id[row_id]
        if row != {**selected, "rejection_reasons": selected["candidates"]}:
            raise RuntimeError(f"Rejected-row provenance changed: {row_id}")
    source_audit_record: dict[str, Any] | None = None
    source_audit_rows: dict[str, dict[str, Any]] = {}
    for row_id, row in accepted_by_id.items():
        source = source_by_id[row_id]
        selected = selection_by_id[row_id]
        candidate_id = str(selected.get("selected_candidate_id", ""))
        metric = metrics.get(candidate_id)
        generated_row = generated.get(candidate_id)
        summary = next(
            (
                candidate
                for candidate in selected.get("candidates", [])
                if candidate.get("candidate_id") == candidate_id
            ),
            None,
        )
        attempt = int(row.get("target_audio", {}).get("attempt", -1))
        target = row.get("target_audio", {})
        target_qa = row.get("target_qa", {})
        expected_target_qa = {
            "candidate_id": candidate_id,
            "metric_sha256": _canonical_sha(metric) if metric else None,
            "asr_transcript_en": metric.get("asr_transcript_en") if metric else None,
            "asr_word_errors": metric.get("asr_word_errors") if metric else None,
            "asr_reference_words": metric.get("asr_reference_words") if metric else None,
            "asr_wer": metric.get("asr_wer") if metric else None,
            "speaker_cosine": metric.get("speaker_cosine") if metric else None,
            "prompt_leak": metric.get("prompt_leak") if metric else None,
            "acoustic": metric.get("acoustic") if metric else None,
            "duration_ratio_target_source": (
                metric.get("duration_ratio_target_source") if metric else None
            ),
            "models": metric.get("models") if metric else None,
            "runtime": metric.get("runtime") if metric else None,
            "thresholds": metric.get("thresholds") if metric else None,
        }
        expected_summary = _metric_summary(metric) if metric else None
        if metric is None or generated_row is None or summary is None or attempt not in (0, 1, 2):
            raise RuntimeError(f"Selected v6 candidate is missing: {row_id}")
        if (
            row.get("schema_version") != FINAL_SCHEMA
            or any(
                row.get(key) != source.get(key)
                for key in (
                    "speaker_id",
                    "eligibility_split",
                    "text_vi",
                    "text_en",
                    "text_vi_sha256",
                    "text_en_sha256",
                    "source_audio",
                    "source_provenance",
                    "source_audit",
                    "reference",
                )
            )
            or summary != expected_summary
            or selected.get("selected_attempt") != attempt
            or target_qa != expected_target_qa
            or target.get("path") != generated_row.get("output_wav")
            or target.get("sha256") != generated_row.get("audio_sha256")
            or target.get("codes") != generated_row.get("codes")
            or target.get("codes_sha256") != generated_row.get("codes_sha256")
            or target.get("duration_s") != generated_row.get("duration_s")
            or target.get("attempt") != generated_row.get("attempt")
            or target.get("attempt_name") != ATTEMPTS[attempt]["name"]
            or target.get("generation_provenance") != metric.get("binding", {}).get("candidate")
            or target.get("synthesis")
            != {**plan["synthesis"], "temperature": ATTEMPTS[attempt]["temperature"]}
            or target.get("model") != plan["model"]
            or target.get("rng") != _rng(policy, row_id, attempt)
            or generated_row.get("source_plan_row_sha256") != _canonical_sha(source)
            or sha256_file(Path(target["path"])) != target["sha256"]
            or sha256_file(Path(target["codes"])) != target["codes_sha256"]
        ):
            raise RuntimeError(f"Final v6 row provenance mismatch: {row_id}")
        audit_record = row["source_audit"]["row_metrics"]
        if source_audit_record is None:
            source_audit_record = audit_record
            audit_path = _path(audit_record, "Source-audit row metrics")
            source_audit_rows = {str(item["id"]): item for item in read_jsonl(audit_path)}
        if audit_record != source_audit_record:
            raise RuntimeError("Accepted rows disagree on source-audit artifact")
        source_audit_row = source_audit_rows.get(row_id)
        if (
            source_audit_row is None
            or _canonical_sha(source_audit_row) != row["source_audit"]["row_sha256"]
        ):
            raise RuntimeError(f"Source-audit row hash changed: {row_id}")
    expected_selected = [
        {
            "id": row["id"],
            "speaker_id": row["speaker_id"],
            "eligibility_split": row["eligibility_split"],
            **row["target_audio"],
        }
        for row in accepted
    ]
    if selected_candidates != expected_selected:
        raise RuntimeError("Selected-candidate manifest differs from accepted rows")
    helper = next(
        (
            item
            for item in contract["scripts"]
            if Path(item["path"]).name == "qwen_mlx_compaction.py"
        ),
        None,
    )
    if helper is None or helper.get("sha256") != RNG_HELPER_SHA256:
        raise RuntimeError("RNG helper attestation changed")
    return {
        "state": "ready",
        "schema_version": FINAL_SCHEMA,
        "production_plan": attestation(plan_path),
        "production_attestation": attestation(production_attestation_path(plan_path)),
        "policy": plan["policy"],
        "validation_go": plan["validation_go"],
        "source_plan": plan["source_plan"],
        "qa_report": attestation(qa_report_path),
        "selection_report": attestation(selection_report_path),
        "accepted": attestation(accepted_path),
        "selection": attestation(selection_path),
        "rejected": attestation(rejected_path),
        "selected_candidates": attestation(selected_candidates_path),
        "manual_required": attestation(manual_required_path),
        "manual_evidence": manual_evidence,
        "rows": len(source_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "speaker_exclusions": exclusions,
        "terminal_policy": terminal_policy,
        "terminal_pruning": _terminal_pruning_record(pruned_ids),
        "attempts": attempts,
        "source_audit_rows": source_audit_record,
        "source_audit_report": accepted[0]["source_audit"]["report"] if accepted else None,
        "reference_map": source_rows[0]["reference_map"]["map"] if source_rows else None,
        "reference_report": source_rows[0]["reference_map"]["report"] if source_rows else None,
        "approval": source_rows[0]["approval"] if source_rows else None,
    }


def validate_live_state(production_plan: Path) -> dict[str, Any]:
    """Validate the live generation prefix and report incomplete separately from corruption."""

    plan_path, plan, _, rows, _ = _production_contract(production_plan)
    result = validate_production_attempt(plan_path, 0)
    return {
        "state": "incomplete",
        "reason": "final_v6_qa_artifacts_not_supplied",
        "generation_state": result["state"],
        "production_plan": attestation(plan_path),
        "rows": len(rows),
        "completed_rows": result["completed_rows"],
        "groups": len(plan["groups"]),
        "completed_groups": result["completed_groups"],
        "cache_ready": False,
    }


def validate_historical(
    policy_path: Path, qa_root: Path, selection_report_path: Path
) -> dict[str, Any]:
    """Exercise v6 candidate/QA parsing on completed validation data only."""

    policy_path = policy_path.expanduser().resolve()
    qa_root = qa_root.expanduser().resolve()
    selection_report_path = selection_report_path.expanduser().resolve()
    selection = _json(selection_report_path)
    through_round = int(selection.get("through_round", -1))
    if (
        selection.get("policy") != attestation(policy_path)
        or selection.get("decision") != "go"
        or through_round not in (0, 1, 2)
    ):
        raise RuntimeError("Historical selection does not bind a terminal v6 GO")
    attempts = []
    for attempt in range(through_round + 1):
        retry = selection_report_path.parent / f"retry_round{attempt}.jsonl" if attempt else None
        candidate_root = Path(policy_path).parent / ATTEMPTS[attempt]["name"]
        result = audit_historical_validation(
            policy_path,
            candidate_root,
            attempt,
            retry,
            qa_root / ATTEMPTS[attempt]["name"],
            selection_report_path,
        )
        if result["state"] not in {"complete", "complete_with_media_errors"}:
            raise IncompleteCampaign(f"Historical attempt {attempt} is incomplete")
        attempts.append(
            {
                "attempt": attempt,
                "attempt_name": ATTEMPTS[attempt]["name"],
                "rows": result["expected_rows"],
                "candidate": attestation(candidate_root / "candidate.json"),
                "qa_report": result["historical_qa"],
            }
        )
    return {
        "state": "historical_validation_only",
        "selection_decision": "go",
        "policy": attestation(policy_path),
        "selection_report": attestation(selection_report_path),
        "attempts": attempts,
        "cache_ready": False,
        "reason": "64-row validation evidence is not the 10,950-row production final scope",
    }


def provenance_paths(summary: dict[str, Any]) -> list[Path]:
    """Return every directly bound finalized metadata artifact."""

    paths: set[Path] = set()

    def add(record: Any) -> None:
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            path = _path(record, "Finalized provenance artifact")
            paths.add(path)

    for key in (
        "production_plan",
        "production_attestation",
        "policy",
        "validation_go",
        "source_plan",
        "qa_report",
        "selection_report",
        "accepted",
        "selection",
        "rejected",
        "selected_candidates",
        "manual_required",
        "source_audit_rows",
        "source_audit_report",
        "reference_map",
        "reference_report",
        "approval",
    ):
        add(summary.get(key))
    add(summary.get("manual_evidence", {}).get("artifact"))
    for attempt in summary["attempts"]:
        for key in (
            "retry_manifest",
            "generation_manifest",
            "candidate",
            "raw_results",
            "qa_report",
            "qa_metrics",
        ):
            add(attempt.get(key))
        for record in attempt.get("group_records", []):
            add(record)
    return sorted(paths)
