"""CPU-only integrity validation for Qwen MLX retry-v6 generation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark_vivos_qwen_mlx_retry_v6 import production_attestation_path
from synthesize_vivos import canonical_json, read_jsonl, sha256_bytes, sha256_file


PLAN_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_production_v6"
POLICY_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_policy_v6"
ATTESTATION_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_production_attestation_v6"
ROW_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_retry_row_v6"
ATTEMPTS = (
    {"attempt": 0, "name": "attempt0_t08", "temperature": 0.8},
    {"attempt": 1, "name": "retry1_t07", "temperature": 0.7},
    {"attempt": 2, "name": "retry2_t08", "temperature": 0.8},
)


class InvalidGeneration(RuntimeError):
    """The generation artifact contradicts its frozen contract."""


def attestation(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def source_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(row).encode())


def group_rows(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    groups = []
    for speaker in sorted(by_speaker):
        subset = by_speaker[speaker]
        for offset in range(0, len(subset), 8):
            ids = [str(row["id"]) for row in subset[offset : offset + 8]]
            digest = sha256_bytes(chr(0).join(ids).encode())[:12]
            groups.append(
                {
                    "group_id": f"{prefix}_{speaker}_{offset // 8:04d}_{digest}",
                    "speaker_id": speaker,
                    "ids": ids,
                }
            )
    return groups


def _require_attested(record: dict[str, str], label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if not path.is_file() or attestation(path) != record:
        raise InvalidGeneration(f"{label} is missing or changed: {path}")
    return path


def load_production_plan(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path = path.expanduser().resolve()
    production_attestation_path(path)
    if not path.is_file():
        raise InvalidGeneration(f"Unexpected production plan path: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("attempt_order") != list(ATTEMPTS):
        raise InvalidGeneration("Production plan schema or attempt order changed")
    if Path(str(plan.get("output_root", ""))).resolve() != path.parent:
        raise InvalidGeneration("Production plan output root changed")
    policy_path = _require_attested(plan["policy"], "Policy")
    validation_go_path = _require_attested(plan["validation_go"], "Validation GO")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != POLICY_SCHEMA
        or policy.get("attempt_order") != list(ATTEMPTS)
        or policy.get("synthesis") != plan.get("synthesis")
        or policy.get("rng") != plan.get("rng")
        or policy.get("source_plan") != plan.get("source_plan")
        or policy.get("model") != plan.get("model")
    ):
        raise InvalidGeneration("Frozen policy and production plan disagree")
    source_path = _require_attested(plan["source_plan"], "Source plan")
    rows = read_jsonl(source_path)
    ids = [str(row.get("id", "")) for row in rows]
    if (
        len(rows) != int(plan.get("rows", -1))
        or len(ids) != len(set(ids))
        or plan.get("groups") != group_rows(rows, "production")
    ):
        raise InvalidGeneration("Source rows are not the exact production-plan bijection")
    contract_path = production_attestation_path(path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != ATTESTATION_SCHEMA
        or contract.get("production_plan") != attestation(path)
        or contract.get("policy") != attestation(policy_path)
        or contract.get("validation_go") != attestation(validation_go_path)
        or contract.get("group_contract", {}).get("rows") != len(rows)
        or contract.get("group_contract", {}).get("groups") != len(plan["groups"])
        or contract.get("group_contract", {}).get("groups_canonical_sha256")
        != sha256_bytes(canonical_json(plan["groups"]).encode())
        or contract.get("model")
        != {
            **plan["model"],
            "files_sha256": contract.get("model", {}).get("files_sha256"),
        }
    ):
        raise InvalidGeneration("Production attestation does not bind this plan and policy")
    for record in contract.get("scripts", []):
        _require_attested(record, "Attested production script")
    return path, plan, policy, rows, contract


def _retry_scope(
    attempt: int,
    manifest: Path | None,
    allowed: set[str],
    *,
    require_production_binding: bool,
) -> tuple[list[str], dict[str, str] | None, list[dict[str, Any]]]:
    if attempt == 0:
        if manifest is not None:
            raise InvalidGeneration("Attempt 0 cannot have a retry manifest")
        return sorted(allowed), None, []
    if manifest is None:
        raise InvalidGeneration(f"Attempt {attempt} requires its frozen retry manifest")
    manifest = manifest.expanduser().resolve()
    rows = read_jsonl(manifest)
    ids = [str(row.get("id", "")) for row in rows]
    required = {
        "id",
        "retry_round",
        "trigger",
        "selected_word_errors",
        "selection_through_round",
        "production_plan",
        "policy",
        "selection_policy_sha256",
        "selection_rows",
    }
    if (
        not ids
        or len(ids) != len(set(ids))
        or not set(ids) <= allowed
        or any(row.get("retry_round") != attempt for row in rows)
        or (require_production_binding and any(not required <= set(row) for row in rows))
    ):
        raise InvalidGeneration(f"Attempt-{attempt} retry scope is invalid")
    if require_production_binding:
        selection_record = rows[0]["selection_rows"]
        if any(row.get("selection_rows") != selection_record for row in rows):
            raise InvalidGeneration("Retry rows disagree on their selection scope")
        selection_path = _require_attested(selection_record, "Retry source selection")
        selections = read_jsonl(selection_path)
        expected: list[tuple[str, str, int | None]] = []
        for selection in selections:
            selected_id = selection.get("selected_candidate_id")
            selected = next(
                (
                    candidate
                    for candidate in selection.get("candidates", [])
                    if candidate.get("candidate_id") == selected_id
                ),
                None,
            )
            if selected is None:
                expected.append((str(selection.get("id", "")), "no_passing_candidate", None))
            elif int(selected["asr_word_errors"]) >= 4:
                expected.append(
                    (
                        str(selection.get("id", "")),
                        "selected_word_errors_gte_4",
                        int(selected["asr_word_errors"]),
                    )
                )
        actual = [
            (str(row["id"]), str(row["trigger"]), row.get("selected_word_errors")) for row in rows
        ]
        if actual != expected or any(
            row.get("selection_through_round") != attempt - 1 for row in rows
        ):
            raise InvalidGeneration("Retry manifest is not the exact frozen trigger scope")
    return ids, attestation(manifest), rows


def _media_validation(row: dict[str, Any], token_count_semantics: str) -> list[str]:
    issues: list[str] = []
    wav = Path(str(row.get("output_wav", ""))).expanduser().resolve()
    codes = Path(str(row.get("codes", ""))).expanduser().resolve()
    if not wav.is_file():
        issues.append("wav_missing")
    elif sha256_file(wav) != row.get("audio_sha256"):
        issues.append("wav_hash")
    if not codes.is_file():
        issues.append("codes_missing")
    elif sha256_file(codes) != row.get("codes_sha256"):
        issues.append("codes_hash")
    try:
        import wave

        import numpy as np

        if wav.is_file():
            with wave.open(str(wav), "rb") as stream:
                channels = stream.getnchannels()
                width = stream.getsampwidth()
                rate = stream.getframerate()
                frames = stream.getnframes()
                payload = stream.readframes(frames)
            audio = np.frombuffer(payload, dtype="<i2")
            if (
                rate != row.get("sample_rate_hz")
                or channels != 1
                or width != 2
                or frames != row.get("num_samples")
                or not np.isfinite(audio).all()
                or not np.any(audio != 0)
            ):
                issues.append("wav_content")
        if codes.is_file():
            array = np.load(codes, allow_pickle=False)
            if (
                array.ndim != 3
                or array.shape[0] != 1
                or row.get("token_count")
                != (1 if token_count_semantics == "legacy_batch_axis" else array.shape[1])
                or array.shape[1] < 1
                or array.shape[2] != 16
                or array.size == 0
                or array.dtype != np.uint32
                or not np.isfinite(array).all()
                or int(array.min()) < 0
                or int(array.max()) >= 2048
            ):
                issues.append("codes_content")
    except Exception as error:
        issues.append(f"media_read:{type(error).__name__}:{error}")
    return issues


def _validate_group(
    directory: Path,
    expected_group: dict[str, Any],
    expected_rows: dict[str, dict[str, Any]],
    attempt_config: dict[str, Any],
    policy_record: dict[str, str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    record_path = directory / "group.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise InvalidGeneration(f"Unreadable group record {record_path}: {error}") from error
    if (
        record.get("schema_version") != POLICY_SCHEMA
        or record.get("policy") != policy_record
        or record.get("attempt_config") != attempt_config
        or record.get("group") != expected_group
    ):
        raise InvalidGeneration(f"Group contract mismatch: {directory}")
    implementation = record.get("implementation")
    if implementation is None:
        token_count_semantics = "legacy_batch_axis"
    elif {
        key: implementation.get(key)
        for key in ("revision", "allocator_cache", "token_count")
    } == {
        "revision": "allocator-cache-repair1",
        "allocator_cache": "clear_after_each_group",
        "token_count": "codec_frames",
    } and isinstance(implementation.get("script"), dict):
        _require_attested(implementation["script"], "Group implementation script")
        token_count_semantics = "codec_frames"
    else:
        raise InvalidGeneration(f"Unknown group implementation: {directory}")
    candidate_rows = record.get("rows")
    if (
        not isinstance(candidate_rows, list)
        or [row.get("id") for row in candidate_rows] != expected_group["ids"]
    ):
        raise InvalidGeneration(f"Group row order/scope mismatch: {directory}")
    media: dict[str, list[str]] = {}
    for candidate in candidate_rows:
        row_id = str(candidate.get("id", ""))
        source = expected_rows[row_id]
        stem = row_id.replace(":", "_")
        expected_wav = (directory / "wavs" / f"{stem}.wav").resolve()
        expected_codes = (directory / "codes" / f"{stem}.npy").resolve()
        if (
            candidate.get("schema_version") != ROW_SCHEMA
            or candidate.get("speaker_id") != source.get("speaker_id")
            or candidate.get("eligibility_split") != source.get("eligibility_split")
            or candidate.get("text_en") != source.get("text_en")
            or candidate.get("source_audio") != source.get("source_audio")
            or candidate.get("reference") != source.get("reference")
            or candidate.get("source_plan_row_sha256") != source_hash(source)
            or candidate.get("attempt") != attempt_config["attempt"]
            or candidate.get("attempt_name") != attempt_config["name"]
            or Path(str(candidate.get("output_wav", ""))).resolve() != expected_wav
            or Path(str(candidate.get("codes", ""))).resolve() != expected_codes
            or candidate.get("sample_rate_hz") != 24_000
            or candidate.get("num_samples")
            != round(float(candidate.get("duration_s", -1)) * 24_000)
        ):
            raise InvalidGeneration(f"Candidate row provenance mismatch: {row_id}")
        issues = _media_validation(candidate, token_count_semantics)
        if issues:
            media[row_id] = issues
    return record, media


def validate_candidate(
    *,
    source_rows: list[dict[str, Any]],
    policy_record: dict[str, str],
    policy: dict[str, Any],
    candidate_root: Path,
    attempt: int,
    retry_manifest: Path | None,
    require_root_manifest: bool,
    production_plan: Path | None = None,
) -> dict[str, Any]:
    candidate_root = candidate_root.expanduser().resolve()
    attempt_config = ATTEMPTS[attempt]
    if candidate_root.name != attempt_config["name"]:
        raise InvalidGeneration("Candidate directory does not match immutable attempt name")
    source_by_id = {str(row["id"]): row for row in source_rows}
    scope_ids, retry_record, retry_rows = _retry_scope(
        attempt,
        retry_manifest,
        set(source_by_id),
        require_production_binding=require_root_manifest,
    )
    subset = [source_by_id[row_id] for row_id in scope_ids]
    expected_groups = group_rows(subset, attempt_config["name"])
    expected_by_name = {group["group_id"]: group for group in expected_groups}
    groups_root = candidate_root / "groups"
    present_dirs = (
        sorted(path for path in groups_root.iterdir() if path.is_dir())
        if groups_root.is_dir()
        else []
    )
    temporary = [path.name for path in present_dirs if path.name.startswith(".")]
    complete_dirs = [path for path in present_dirs if not path.name.startswith(".")]
    extras = sorted(path.name for path in complete_dirs if path.name not in expected_by_name)
    if extras:
        raise InvalidGeneration(f"Unexpected completed groups: {extras[:5]}")
    records: dict[str, dict[str, Any]] = {}
    media_errors: dict[str, list[str]] = {}
    for directory in complete_dirs:
        record, issues = _validate_group(
            directory,
            expected_by_name[directory.name],
            source_by_id,
            attempt_config,
            policy_record,
        )
        records[directory.name] = record
        media_errors.update(issues)
    missing = [group["group_id"] for group in expected_groups if group["group_id"] not in records]
    final_paths = [candidate_root / "candidate.json", candidate_root / "raw_results.jsonl"]
    root_manifest = (
        production_plan.parent / f"generation_attempt{attempt}_manifest.json"
        if production_plan is not None
        else None
    )
    if root_manifest is not None:
        final_paths.append(root_manifest)
    any_final = any(path.is_file() for path in final_paths)
    complete_groups = not missing and not temporary
    if any_final and not complete_groups:
        raise InvalidGeneration(
            "Final candidate artifacts exist before the exact group scope completed"
        )
    candidate: dict[str, Any] | None = None
    raw_records: list[dict[str, Any]] = []
    if complete_groups:
        if not all(path.is_file() for path in final_paths):
            state = "incomplete"
        else:
            candidate = json.loads(final_paths[0].read_text(encoding="utf-8"))
            raw_records = read_jsonl(final_paths[1])
            ordered_records = [records[group["group_id"]] for group in expected_groups]
            if raw_records != ordered_records:
                raise InvalidGeneration("raw_results.jsonl differs from exact group records/order")
            if (
                candidate.get("candidate") != attempt_config["name"]
                or candidate.get("attempt_config") != attempt_config
                or candidate.get("scope_rows") != len(scope_ids)
                or candidate.get("policy") != policy_record
                or candidate.get("retry") != retry_record
                or candidate.get("synthesis")
                != {**policy["synthesis"], "temperature": attempt_config["temperature"]}
                or candidate.get("rng") != policy["rng"]
                or Path(str(candidate.get("raw_results", ""))).resolve() != final_paths[1]
            ):
                raise InvalidGeneration("candidate.json contract mismatch")
            if require_root_manifest:
                assert root_manifest is not None and production_plan is not None
                manifest = json.loads(root_manifest.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema_version") != PLAN_SCHEMA
                    or manifest.get("production_plan") != attestation(production_plan)
                    or manifest.get("attempt_config") != attempt_config
                    or manifest.get("retry") != retry_record
                    or manifest.get("candidate") != attestation(final_paths[0])
                    or manifest.get("rows") != len(scope_ids)
                    or manifest.get("raw_results") != attestation(final_paths[1])
                ):
                    raise InvalidGeneration("Root generation manifest contract mismatch")
            state = "complete_with_media_errors" if media_errors else "complete"
    else:
        state = "incomplete"
    return {
        "schema_version": "hibiki_vivos_qwen3_tts_mlx_generation_validation_v6",
        "state": state,
        "attempt": attempt,
        "attempt_name": attempt_config["name"],
        "expected_rows": len(scope_ids),
        "expected_groups": len(expected_groups),
        "completed_groups": len(records),
        "completed_rows": sum(len(record["rows"]) for record in records.values()),
        "missing_groups": missing,
        "temporary_groups": temporary,
        "media_errors": media_errors,
        "retry_rows": retry_rows,
        "candidate": candidate,
        "group_records": [
            attestation(candidate_root / "groups" / group_id / "group.json") for group_id in records
        ],
        "records": [
            records[group["group_id"]] for group in expected_groups if group["group_id"] in records
        ],
    }


def validate_production_attempt(
    production_plan: Path,
    attempt: int,
    retry_manifest: Path | None = None,
) -> dict[str, Any]:
    plan_path, plan, policy, rows, contract = load_production_plan(production_plan)
    result = validate_candidate(
        source_rows=rows,
        policy_record=plan["policy"],
        policy=policy,
        candidate_root=plan_path.parent / ATTEMPTS[attempt]["name"],
        attempt=attempt,
        retry_manifest=retry_manifest,
        require_root_manifest=True,
        production_plan=plan_path,
    )
    candidate = result.get("candidate")
    if candidate is not None:
        candidate_model = candidate.get("model", {})
        if (
            candidate.get("script") != plan.get("script")
            or candidate.get("campaign_revision") != policy.get("campaign_revision")
            or candidate_model.get("id") != contract["model"]["id"]
            or candidate_model.get("revision") != contract["model"]["revision"]
            or candidate_model.get("files_sha256") != contract["model"]["files_sha256"]
        ):
            raise InvalidGeneration("Candidate script/campaign/model snapshot contract mismatch")
    selection_policy_sha256 = sha256_bytes(canonical_json(policy["retry_policy"]).encode())
    if attempt and any(
        row.get("production_plan") != attestation(plan_path)
        or row.get("policy") != plan["policy"]
        or row.get("selection_policy_sha256") != selection_policy_sha256
        for row in result["retry_rows"]
    ):
        raise InvalidGeneration("Retry manifest is not bound to this plan/policy/selection rule")
    result.update(
        {
            "production_plan": attestation(plan_path),
            "production_attestation": attestation(production_attestation_path(plan_path)),
            "policy": plan["policy"],
            "source_plan": plan["source_plan"],
            "contract": contract,
        }
    )
    return result


def audit_historical_validation(
    policy_path: Path,
    candidate_root: Path,
    attempt: int,
    retry_manifest: Path | None,
    qa_dir: Path | None,
    selection_report: Path | None,
) -> dict[str, Any]:
    policy_path = policy_path.expanduser().resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    cohort = read_jsonl(Path(policy["cohort"]["path"]))
    result = validate_candidate(
        source_rows=cohort,
        policy_record=attestation(policy_path),
        policy=policy,
        candidate_root=candidate_root,
        attempt=attempt,
        retry_manifest=retry_manifest,
        require_root_manifest=False,
    )
    if qa_dir is not None:
        qa_dir = qa_dir.expanduser().resolve()
        report_path = qa_dir / "qa_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = read_jsonl(qa_dir / "metrics.jsonl")
        expected_ids = [row["id"] for record in result["records"] for row in record["rows"]]
        if (
            report.get("candidate_record") != attestation(candidate_root / "candidate.json")
            or report.get("scope_rows") != len(expected_ids)
            or [row.get("id") for row in metrics] != expected_ids
        ):
            raise InvalidGeneration("Historical QA does not bind the validated candidate scope")
        result["historical_qa"] = attestation(report_path)
    if selection_report is not None:
        selection_report = selection_report.expanduser().resolve()
        selection = json.loads(selection_report.read_text(encoding="utf-8"))
        if selection.get("policy") != attestation(policy_path):
            raise InvalidGeneration("Historical selection does not bind the frozen policy")
        for record in selection.get("attempts", []):
            _require_attested(record, "Historical selected attempt")
        for record in selection.get("qa", []):
            _require_attested(record, "Historical selected QA")
        _require_attested(selection["selection_rows"], "Historical selection rows")
        result["historical_selection"] = attestation(selection_report)
        result["historical_selection_decision"] = selection.get("decision")
    return result


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: result[key]
        for key in (
            "state",
            "attempt",
            "attempt_name",
            "expected_rows",
            "completed_rows",
            "expected_groups",
            "completed_groups",
            "missing_groups",
            "temporary_groups",
            "media_errors",
            "historical_qa",
            "historical_selection",
            "historical_selection_decision",
        )
        if key in result
    }
    for key in ("missing_groups", "temporary_groups"):
        if key in summary:
            values = summary.pop(key)
            summary[f"{key}_count"] = len(values)
            summary[f"{key}_first"] = values[:10]
    if "media_errors" in summary:
        values = summary.pop("media_errors")
        summary["media_error_rows"] = len(values)
        summary["media_errors_first"] = dict(list(values.items())[:10])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    production = commands.add_parser("production")
    production.add_argument("production_plan", type=Path)
    production.add_argument("--attempt", type=int, choices=(0, 1, 2), required=True)
    production.add_argument("--retry-manifest", type=Path)
    historical = commands.add_parser("historical-validation")
    historical.add_argument("policy", type=Path)
    historical.add_argument("candidate_root", type=Path)
    historical.add_argument("--attempt", type=int, choices=(0, 1, 2), required=True)
    historical.add_argument("--retry-manifest", type=Path)
    historical.add_argument("--qa-dir", type=Path)
    historical.add_argument("--selection-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.action == "production":
            result = validate_production_attempt(
                args.production_plan, args.attempt, args.retry_manifest
            )
        else:
            result = audit_historical_validation(
                args.policy,
                args.candidate_root,
                args.attempt,
                args.retry_manifest,
                args.qa_dir,
                args.selection_report,
            )
    except InvalidGeneration as error:
        print(json.dumps({"state": "invalid", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
