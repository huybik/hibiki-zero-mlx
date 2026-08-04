"""Prepare and run the immutable full VIVOS Qwen3-TTS MLX campaign."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthesize_vivos import (
    MLX_MODEL_FILES_SHA256,
    MLX_MODEL_ID,
    MLX_MODEL_REVISION,
    MLX_PACKAGE_COMMIT,
    MLX_PACKAGE_VERSION,
    MLX_SOURCE_MODEL_ID,
    MLX_SOURCE_MODEL_REVISION,
    MLX_V3_GENERATION_CONFIG,
    atomic_write_jsonl,
    atomic_write_wav,
    canonical_json,
    git_commit,
    immutable_write,
    package_version,
    read_jsonl,
    require_mlx_audio_commit,
    require_package,
    sha256_bytes,
    sha256_file,
    translation_record,
    verify_mlx_snapshot,
)

SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_v1"
APPROVAL_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_approval_v1"
ATTEMPT_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_full_attempt_v1"
REFERENCE_SCHEMA = "hibiki_vivos_qwen3_tts_mlx_v3_reference_map_v1"
SOURCE_AUDIT_SCHEMA = "hibiki_vivos_source_asr_mps_full_v1"
OUT_DIR_NAME = "vivos_qwen3_tts_mlx_v3_full_v1"
SEED_NAMESPACE = "hibiki-vivos-mlx-v3-full-v1"
EXPECTED = {
    "train": {
        "sha256": "d2276dcda8b664ca918dd53d215b11b159da98fd817fec89d7cb3701f6bc92fb",
        "rows": 9844,
        "speakers": 41,
    },
    "dev": {
        "sha256": "6fae77d42d6580fc0c36754ce284acb26a3be34bc17de4378979a549f727579d",
        "rows": 1106,
        "speakers": 5,
    },
}
EXPECTED_ROWS = 10950
EXPECTED_SPEAKERS = 46
EXPECTED_SOURCE_AUDIT_REPORT_SHA256 = (
    "0a02b5726cc8d7b9a2de02802bff992506a804bde8dda73f8180f84c09a09df4"
)
EXPECTED_SOURCE_AUDIT_ROWS_SHA256 = (
    "f53554fd8cea25890f61eb2022237b8bd69e8c9415ccb9f3cb03870899e46eb8"
)
EXPECTED_REFERENCE_MAP_SHA256 = "1b7a5a0187c51dd8ea6aa27e862172548483af8aee215cfbd56cf3e42563ae04"
EXPECTED_REFERENCE_REPORT_SHA256 = (
    "e287d167db597d7fa47c98bfcaecf06f33cc9c9df8a55d18844d48b79bf129aa"
)
EXPECTED_PILOT_GATE_SHA256 = "9aeea983926885b3fc84c8fa52b001fc0b282e671257e44ec2d11d981f086b0f"
SYNTHESIS = {
    "package": "mlx-audio",
    "package_version": MLX_PACKAGE_VERSION,
    "package_commit": MLX_PACKAGE_COMMIT,
    "model_id": MLX_MODEL_ID,
    "model_revision": MLX_MODEL_REVISION,
    "source_model_id": MLX_SOURCE_MODEL_ID,
    "source_model_revision": MLX_SOURCE_MODEL_REVISION,
    "weight_dtype": "bfloat16",
    "language": "English",
    "clone_mode": "icl_reference_audio_and_text",
    "reference_cache": "mlx_audio_internal_icl_cache",
    "generation_config": MLX_V3_GENERATION_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-mlx-full")
    prepare.add_argument("manifests", type=Path, nargs="+")
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--source-audit-report", type=Path, required=True)
    prepare.add_argument("--reference-map", type=Path, required=True)
    prepare.add_argument("--reference-report", type=Path, required=True)
    prepare.add_argument("--pilot-gate-report", type=Path, required=True)
    generate = commands.add_parser("generate-mlx-full")
    generate.add_argument("plan", type=Path)
    generate.add_argument("--dataset-root", type=Path, required=True)
    generate.add_argument("--device", default="mps")
    generate.add_argument("--attempt", type=int, choices=(0, 1), required=True)
    generate.add_argument("--retry-ids", type=Path)
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def attestation(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def exact_named(path: Path, name: str, parent: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name != name or (parent is not None and resolved.parent.name != parent):
        raise RuntimeError(f"Expected {parent or '*'}/{name}, found {resolved}")
    return resolved


def seed_for(row_id: str, attempt: int) -> int:
    suffix = "" if attempt == 0 else "\0retry=1"
    material = f"{SEED_NAMESPACE}\0{row_id}{suffix}".encode()
    return int.from_bytes(bytes.fromhex(sha256_bytes(material))[:4], "big")


def source_audio(row: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    path = Path(str(row["audio_path"])).expanduser().resolve()
    if not path.is_file() or not path.is_relative_to(dataset_root):
        raise RuntimeError(f"Invalid source audio path for {row['id']}: {path}")
    return {
        "path": str(path),
        "dataset_relative_path": str(path.relative_to(dataset_root)),
        "sha256": row["audio_sha256"],
        "duration_s": row["duration_s"],
        "sample_rate_hz": row["sample_rate_hz"],
        "channels": row["channels"],
        "sample_width_bytes": row["sample_width_bytes"],
    }


def load_manifests(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (item.expanduser().resolve() for item in paths):
        file_rows = read_jsonl(path)
        splits = {str(row.get("eligibility_split", "")) for row in file_rows}
        if len(splits) != 1 or (split := splits.pop()) not in EXPECTED:
            raise RuntimeError(
                f"Manifest must contain exactly one accepted train/dev split: {path}"
            )
        expected = EXPECTED[split]
        actual_sha = sha256_file(path)
        speakers = {str(row.get("speaker_id", "")) for row in file_rows}
        if (
            actual_sha != expected["sha256"]
            or len(file_rows) != expected["rows"]
            or len(speakers) != expected["speakers"]
        ):
            raise RuntimeError(f"Frozen {split} manifest contract mismatch: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "eligibility_split": split,
                "rows": len(file_rows),
                "speakers": len(speakers),
            }
        )
        for row in file_rows:
            row_id = str(row.get("id", ""))
            if not row_id or row_id in seen or row.get("official_split") == "test":
                raise RuntimeError(f"Empty, duplicate, or test id: {row_id!r}")
            translation_record(row)
            seen.add(row_id)
            rows.append(row)
    if {record["eligibility_split"] for record in records} != set(EXPECTED) or len(
        rows
    ) != EXPECTED_ROWS:
        raise RuntimeError("Full campaign requires exactly the frozen train and dev manifests")
    return rows, sorted(records, key=lambda item: item["eligibility_split"])


def load_source_audit(
    path: Path, rows: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    path = exact_named(path, "audit_report.json", "vivos_source_asr_mps_full_v1")
    report = json.loads(path.read_text(encoding="utf-8"))
    row_record = report.get("row_metrics", {})
    row_path = exact_named(
        Path(str(row_record.get("path", ""))), "row_metrics.jsonl", "vivos_source_asr_mps_full_v1"
    )
    if (
        sha256_file(path) != EXPECTED_SOURCE_AUDIT_REPORT_SHA256
        or sha256_file(row_path) != EXPECTED_SOURCE_AUDIT_ROWS_SHA256
        or report.get("schema_version") != SOURCE_AUDIT_SCHEMA
        or report.get("complete") is not True
        or report.get("status") != "complete_with_source_review_flags"
        or report.get("rows") != EXPECTED_ROWS
        or report.get("speakers") != EXPECTED_SPEAKERS
    ):
        raise RuntimeError(f"Incomplete full source audit: {path}")
    if [
        {key: item[key] for key in ("eligibility_split", "path", "sha256")} for item in manifests
    ] != report.get("source_manifests"):
        raise RuntimeError("Source audit manifest contract mismatch")
    if row_record.get("sha256") != sha256_file(row_path):
        raise RuntimeError("Source audit row-metrics hash mismatch")
    audit_rows = read_jsonl(row_path)
    by_id = {str(item.get("id", "")): item for item in audit_rows}
    if len(by_id) != EXPECTED_ROWS or set(by_id) != {str(row["id"]) for row in rows}:
        raise RuntimeError("Source audit is not a bijection with campaign rows")
    manifest_sha = {item["eligibility_split"]: item["sha256"] for item in manifests}
    for row in rows:
        audit = by_id[str(row["id"])]
        if (
            audit.get("schema_version") != SOURCE_AUDIT_SCHEMA
            or audit.get("audio_sha256") != row.get("audio_sha256")
            or audit.get("source_manifest_sha256") != manifest_sha[row["eligibility_split"]]
            or audit.get("reference_text_vi_sha256") != sha256_bytes(str(row["text_vi"]).encode())
        ):
            raise RuntimeError(f"Source audit row contract mismatch: {row['id']}")
    return (
        report,
        by_id,
        {
            "report": attestation(path),
            "row_metrics": attestation(row_path),
            "schema_version": SOURCE_AUDIT_SCHEMA,
        },
    )


def load_references(
    map_path: Path, report_path: Path, source_audit: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    map_path = exact_named(map_path, "reference_map.jsonl", OUT_DIR_NAME)
    report_path = exact_named(report_path, "reference_map_report.json", OUT_DIR_NAME)
    refs = read_jsonl(map_path)
    by_speaker = {str(row.get("speaker_id", "")): row for row in refs}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        sha256_file(map_path) != EXPECTED_REFERENCE_MAP_SHA256
        or sha256_file(report_path) != EXPECTED_REFERENCE_REPORT_SHA256
        or len(refs) != EXPECTED_SPEAKERS
        or len(by_speaker) != EXPECTED_SPEAKERS
        or any(row.get("schema_version") != REFERENCE_SCHEMA for row in refs)
    ):
        raise RuntimeError(
            "Reference map must contain one pinned reference for each of 46 speakers"
        )
    if (
        report.get("schema_version") != REFERENCE_SCHEMA
        or report.get("status") != "complete"
        or report.get("reference_count") != EXPECTED_SPEAKERS
        or report.get("references") != refs
    ):
        raise RuntimeError("Reference report contract mismatch")
    if (
        report.get("reference_map") != attestation(map_path)
        or report.get("source_audit", {}).get("report_sha256") != source_audit["report"]["sha256"]
        or report.get("source_audit", {}).get("row_metrics", {}).get("sha256")
        != source_audit["row_metrics"]["sha256"]
    ):
        raise RuntimeError("Reference report attestations do not match the frozen inputs")
    for speaker, ref in by_speaker.items():
        audio = Path(str(ref.get("reference_audio_path", "")))
        if (
            not speaker
            or not audio.is_file()
            or sha256_file(audio) != ref.get("reference_audio_sha256")
            or sha256_bytes(str(ref.get("reference_text_vi", "")).encode())
            != ref.get("reference_text_vi_sha256")
        ):
            raise RuntimeError(f"Invalid frozen reference: {speaker}")
    return by_speaker, {
        "map": attestation(map_path),
        "report": attestation(report_path),
        "schema_version": REFERENCE_SCHEMA,
    }


def load_gate(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = exact_named(path, "gate_report.json", "qa")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    if (
        sha256_file(path) != EXPECTED_PILOT_GATE_SHA256
        or report.get("decision") != "no_go"
        or checks.get("qwen_wer_vs_kokoro") is not False
        or checks.get("manual_review_complete") is not False
    ):
        raise RuntimeError("Expected the retained MLX v3 no-go pilot gate")
    return report, attestation(path)


def prepare(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if out_dir.name != OUT_DIR_NAME:
        raise RuntimeError(f"Full campaign directory must be named {OUT_DIR_NAME}")
    rows, manifests = load_manifests(args.manifests)
    _, audit_by_id, audit_attestation = load_source_audit(args.source_audit_report, rows, manifests)
    references, reference_attestation = load_references(
        args.reference_map, args.reference_report, audit_attestation
    )
    gate, gate_attestation = load_gate(args.pilot_gate_report)
    approval = {
        "schema_version": APPROVAL_SCHEMA,
        "campaign_schema_version": SCHEMA,
        "pilot_gate_report": gate_attestation,
        "retained_pilot_decision": gate["decision"],
        "authorization": {
            "source": "user",
            "statements_in_order": ["the ens files are very good. use qwen please", "ok go"],
        },
        "waiver": {
            "level": "model",
            "dimensions": ["aggregate_kokoro_comparison", "manual_review_completeness"],
            "scope": ["accepted VIVOS train", "accepted VIVOS dev"],
            "does_not_assert_all_24_pilot_files_reviewed": True,
            "does_not_relabel_pilot_gate": True,
        },
    }
    approval_path = out_dir / "approval_override.json"
    approval_bytes = json_bytes(approval)
    approval_attestation = {"path": str(approval_path), "sha256": sha256_bytes(approval_bytes)}
    manifest_by_split = {item["eligibility_split"]: item for item in manifests}
    plan_rows = []
    for row in rows:
        speaker = str(row["speaker_id"])
        frozen_ref = references.get(speaker)
        ref = dict(frozen_ref) if frozen_ref is not None else None
        if ref is None or ref.get("eligibility_split") != row.get("eligibility_split"):
            raise RuntimeError(f"Missing same-split reference for {row['id']}")
        reference_audio_path = Path(str(ref["reference_audio_path"])).resolve()
        if not reference_audio_path.is_relative_to(dataset_root):
            raise RuntimeError(f"Reference is outside the dataset root: {reference_audio_path}")
        ref["reference_audio_dataset_relative_path"] = str(
            reference_audio_path.relative_to(dataset_root)
        )
        audit = audit_by_id[str(row["id"])]
        safe_id = str(row["id"]).replace(":", "_")
        plan_rows.append(
            {
                "schema_version": SCHEMA,
                "id": row["id"],
                "speaker_id": speaker,
                "eligibility_split": row["eligibility_split"],
                "text_vi": row["text_vi"],
                "text_en": row["text_en"],
                "text_vi_sha256": sha256_bytes(str(row["text_vi"]).encode()),
                "text_en_sha256": sha256_bytes(str(row["text_en"]).encode()),
                "source_audio": source_audio(row, dataset_root),
                "source_provenance": {
                    "corpus": row["corpus"],
                    "corpus_revision": row["corpus_revision"],
                    "license": row["license"],
                    "source_repo": row["source_repo"],
                    "source_file": row["source_file"],
                    "source_archive_sha256": row["source_archive_sha256"],
                    "accepted_manifest": manifest_by_split[row["eligibility_split"]],
                    "translation": translation_record(row),
                },
                "source_audit": {
                    **audit_attestation,
                    "row_sha256": sha256_bytes(canonical_json(audit).encode()),
                    "hard_gate_pass": audit["hard_gate_pass"],
                    "hard_failure_reasons": audit["hard_failure_reasons"],
                    "source_review_required": audit["source_review_required"],
                },
                "reference": ref,
                "reference_map": reference_attestation,
                "approval": approval_attestation,
                "synthesis": SYNTHESIS,
                "seeds": {
                    "attempt0": seed_for(str(row["id"]), 0),
                    "attempt1_retry_1": seed_for(str(row["id"]), 1),
                },
                "output_wavs": {
                    "attempt0": str(
                        Path("wavs")
                        / row["eligibility_split"]
                        / speaker
                        / f"{safe_id}.attempt0.wav"
                    ),
                    "attempt1": str(
                        Path("wavs")
                        / row["eligibility_split"]
                        / speaker
                        / f"{safe_id}.attempt1.wav"
                    ),
                },
            }
        )
    plan_rows.sort(key=lambda row: (row["speaker_id"], row["eligibility_split"], row["id"]))
    plan_path = out_dir / "generation_plan.jsonl"
    plan_bytes = "".join(canonical_json(row) + "\n" for row in plan_rows).encode()
    config = {
        "schema_version": SCHEMA,
        "repository_commit": git_commit(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_script_sha256": sha256_file(Path(__file__).with_name("synthesize_vivos.py")),
        "dataset_root_at_prepare": str(dataset_root),
        "scope": {
            "splits": {"train": 9844, "dev": 1106},
            "rows": EXPECTED_ROWS,
            "speakers": EXPECTED_SPEAKERS,
            "test_sealed": True,
        },
        "accepted_manifests": manifests,
        "source_audit": audit_attestation,
        "references": reference_attestation,
        "approval": approval_attestation,
        "synthesis": {**SYNTHESIS, "model_files_sha256": MLX_MODEL_FILES_SHA256},
        "execution": {
            "serial_only": True,
            "order": ["speaker_id", "eligibility_split", "id"],
            "seed_contract": {
                "attempt0": "uint32_be(SHA256(namespace + NUL + id)[:4])",
                "attempt1": "uint32_be(SHA256(namespace + NUL + id + NUL + retry=1)[:4])",
                "namespace": SEED_NAMESPACE,
            },
            "attempts": [0, 1],
        },
        "plan": {"path": str(plan_path), "sha256": sha256_bytes(plan_bytes)},
    }
    immutable_write(approval_path, approval_bytes)
    immutable_write(plan_path, plan_bytes)
    immutable_write(out_dir / "campaign_config.json", json_bytes(config))
    print(f"Prepared {len(plan_rows)} immutable full-campaign rows: {plan_path}")


def load_campaign(plan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    plan_path = exact_named(plan_path, "generation_plan.jsonl", OUT_DIR_NAME)
    config_path = plan_path.parent / "campaign_config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    plan_sha = sha256_file(plan_path)
    config_sha = sha256_bytes(config_bytes)
    rows = read_jsonl(plan_path)
    ids = [str(row.get("id", "")) for row in rows]
    split_counts = {
        split: sum(row.get("eligibility_split") == split for row in rows) for split in EXPECTED
    }
    if (
        config.get("schema_version") != SCHEMA
        or config.get("plan") != {"path": str(plan_path), "sha256": plan_sha}
        or len(rows) != EXPECTED_ROWS
        or len(set(ids)) != EXPECTED_ROWS
        or split_counts != {split: spec["rows"] for split, spec in EXPECTED.items()}
        or len({str(row.get("speaker_id", "")) for row in rows}) != EXPECTED_SPEAKERS
        or config.get("script_sha256") != sha256_file(Path(__file__).resolve())
        or config.get("shared_script_sha256")
        != sha256_file(Path(__file__).with_name("synthesize_vivos.py"))
    ):
        raise RuntimeError("Full campaign config/plan contract mismatch")
    if rows != sorted(
        rows, key=lambda row: (row["speaker_id"], row["eligibility_split"], row["id"])
    ):
        raise RuntimeError("Generation plan order is not speaker/split/id")
    for key in ("source_audit", "references"):
        for record in config[key].values():
            if (
                isinstance(record, dict)
                and "path" in record
                and sha256_file(Path(record["path"])) != record["sha256"]
            ):
                raise RuntimeError(f"Campaign {key} attestation changed: {record['path']}")
    approval_record = config["approval"]
    if sha256_file(Path(approval_record["path"])) != approval_record["sha256"]:
        raise RuntimeError("Campaign approval artifact changed")
    for row in rows:
        safe_id = str(row["id"]).replace(":", "_")
        expected_outputs = {
            f"attempt{attempt}": str(
                Path("wavs")
                / row["eligibility_split"]
                / row["speaker_id"]
                / f"{safe_id}.attempt{attempt}.wav"
            )
            for attempt in (0, 1)
        }
        if (
            row.get("schema_version") != SCHEMA
            or row.get("synthesis") != SYNTHESIS
            or row.get("approval") != approval_record
            or row.get("source_audit", {}).get("report") != config["source_audit"]["report"]
            or row.get("source_audit", {}).get("row_metrics")
            != config["source_audit"]["row_metrics"]
            or row.get("reference_map") != config["references"]
            or row.get("output_wavs") != expected_outputs
            or row["seeds"]
            != {"attempt0": seed_for(row["id"], 0), "attempt1_retry_1": seed_for(row["id"], 1)}
        ):
            raise RuntimeError(f"Invalid campaign row: {row.get('id')}")
    return rows, config, plan_sha, config_sha


def output_path(plan_path: Path, row: dict[str, Any], attempt: int) -> Path:
    return (plan_path.parent / row["output_wavs"][f"attempt{attempt}"]).resolve()


def sidecar_path(plan_path: Path, row: dict[str, Any], attempt: int) -> Path:
    safe_id = str(row["id"]).replace(":", "_")
    return (
        plan_path.parent
        / "attempts"
        / f"attempt{attempt}"
        / row["eligibility_split"]
        / row["speaker_id"]
        / f"{safe_id}.json"
    )


def retry_ids(
    path: Path | None, attempt: int, plan_ids: set[str]
) -> tuple[set[str], dict[str, str] | None]:
    if attempt == 0:
        if path is not None:
            raise RuntimeError("--retry-ids is only valid for --attempt 1")
        return plan_ids, None
    if path is None:
        raise RuntimeError("--attempt 1 requires an explicit --retry-ids JSONL manifest")
    path = path.expanduser().resolve()
    rows = read_jsonl(path)
    ids = [str(row.get("id", "")) for row in rows]
    if not ids or len(ids) != len(set(ids)) or set(ids) - plan_ids:
        raise RuntimeError("Retry-id manifest is empty, duplicated, or outside the plan")
    return set(ids), attestation(path)


def validate_inputs(rows: list[dict[str, Any]], dataset_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        for record, path_key, relative_key, sha_key in (
            (row["source_audio"], "path", "dataset_relative_path", "sha256"),
            (
                row["reference"],
                "reference_audio_path",
                "reference_audio_dataset_relative_path",
                "reference_audio_sha256",
            ),
        ):
            original = Path(str(record[path_key]))
            path = (dataset_root / record[relative_key]).resolve()
            if str(path) not in hashes:
                hashes[str(path)] = sha256_file(path) if path.is_file() else ""
            actual_sha = hashes[str(path)]
            if actual_sha != record[sha_key]:
                raise RuntimeError(f"Input audio hash mismatch: {path}")
            paths[str(original)] = path
    return paths


def assemble(
    plan_path: Path, rows: list[dict[str, Any]], plan_sha: str, config_sha: str
) -> dict[tuple[str, int], dict[str, Any]]:
    plan_by_id = {row["id"]: row for row in rows}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((plan_path.parent / "attempts").glob("attempt*/*/*/*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (str(item.get("id", "")), int(item.get("attempt", -1)))
        row = plan_by_id.get(key[0])
        if (
            row is None
            or key in found
            or key[1] not in (0, 1)
            or path != sidecar_path(plan_path, row, key[1])
        ):
            raise RuntimeError(f"Unexpected or duplicate attempt sidecar: {path}")
        output = output_path(plan_path, row, key[1])
        expected_seed = row["seeds"]["attempt0" if key[1] == 0 else "attempt1_retry_1"]
        retry = item.get("retry_ids")
        retry_valid = retry is None if key[1] == 0 else isinstance(retry, dict)
        if key[1] == 1 and retry_valid:
            retry_path = Path(str(retry.get("path", "")))
            retry_valid = (
                retry_path.is_file()
                and sha256_file(retry_path) == retry.get("sha256")
                and key[0] in {str(entry.get("id", "")) for entry in read_jsonl(retry_path)}
            )
        if (
            item.get("schema_version") != ATTEMPT_SCHEMA
            or item.get("plan_path") != str(plan_path)
            or item.get("plan_sha256") != plan_sha
            or item.get("config_sha256") != config_sha
            or item.get("seed") != expected_seed
            or not retry_valid
            or item.get("synthesis") != SYNTHESIS
            or item.get("reference") != row["reference"]
            or item.get("source_audit") != row["source_audit"]
            or item.get("output_wav") != str(output)
            or not output.is_file()
            or sha256_file(output) != item.get("audio_sha256")
            or item.get("model_snapshot", {}).get("files_sha256") != MLX_MODEL_FILES_SHA256
        ):
            raise RuntimeError(f"Completed attempt provenance mismatch: {path}")
        found[key] = item
    for row in rows:
        for attempt in (0, 1):
            output = output_path(plan_path, row, attempt)
            if output.exists() and (row["id"], attempt) not in found:
                raise RuntimeError(f"Unrecorded output exists; refusing to overwrite: {output}")
    order = {row["id"]: index for index, row in enumerate(rows)}
    atomic_write_jsonl(
        plan_path.parent / "generation_attempts.jsonl",
        [found[key] for key in sorted(found, key=lambda key: (order[key[0]], key[1]))],
    )
    return found


def generate(args: argparse.Namespace) -> None:
    if args.device != "mps":
        raise RuntimeError("Full MLX generation is Apple-Metal-only; use --device mps")
    plan_path = args.plan.expanduser().resolve()
    rows, _, plan_sha, config_sha = load_campaign(plan_path)
    selected, retry_attestation = retry_ids(
        args.retry_ids, args.attempt, {row["id"] for row in rows}
    )
    mlx_audio_version = require_package("mlx-audio", MLX_PACKAGE_VERSION, "Full MLX generation")
    require_mlx_audio_commit()
    try:
        import mlx.core as mx
        import numpy as np
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model
    except ImportError as error:
        raise RuntimeError(
            "Full MLX generation requires mlx-audio, mlx, huggingface-hub, numpy, and soundfile"
        ) from error
    model_root = Path(snapshot_download(repo_id=MLX_MODEL_ID, revision=MLX_MODEL_REVISION))
    if model_root.name != MLX_MODEL_REVISION:
        raise RuntimeError(f"Snapshot did not resolve to pinned revision: {model_root}")
    snapshot_hashes = verify_mlx_snapshot(model_root)
    inputs = validate_inputs(rows, args.dataset_root.expanduser().resolve())
    completed = assemble(plan_path, rows, plan_sha, config_sha)
    pending = [
        row for row in rows if row["id"] in selected and (row["id"], args.attempt) not in completed
    ]
    if not pending:
        print(f"All {len(selected)} selected attempt-{args.attempt} outputs are complete")
        return
    model = load_model(model_root)
    for number, row in enumerate(pending, 1):
        seed = row["seeds"]["attempt0" if args.attempt == 0 else "attempt1_retry_1"]
        random.seed(seed)
        np.random.seed(seed)
        mx.random.seed(seed)
        started = time.monotonic()
        ref = row["reference"]
        results = list(
            model.generate(
                text=row["text_en"],
                ref_audio=str(inputs[ref["reference_audio_path"]]),
                ref_text=ref["reference_text_vi"],
                max_tokens=2048,
                temperature=0.8,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                lang_code="English",
                split_pattern="\n",
                stream=False,
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one waveform for {row['id']}")
        mx.eval(results[0].audio)
        audio = np.asarray(results[0].audio, dtype=np.float32).reshape(-1)
        sample_rate = int(results[0].sample_rate)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError(f"Invalid generated audio for {row['id']}")
        output = output_path(plan_path, row, args.attempt)
        atomic_write_wav(output, audio, sample_rate, sf)
        result = {
            "schema_version": ATTEMPT_SCHEMA,
            "id": row["id"],
            "speaker_id": row["speaker_id"],
            "eligibility_split": row["eligibility_split"],
            "attempt": args.attempt,
            "seed": seed,
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha,
            "config_sha256": config_sha,
            "retry_ids": retry_attestation,
            "output_wav": str(output),
            "audio_sha256": sha256_file(output),
            "sample_rate_hz": sample_rate,
            "num_samples": int(audio.size),
            "duration_s": round(audio.size / sample_rate, 6),
            "generation_seconds": round(time.monotonic() - started, 3),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "mlx-audio": mlx_audio_version,
                "mlx-audio-commit": MLX_PACKAGE_COMMIT,
                "mlx": package_version("mlx"),
                "numpy": package_version("numpy"),
                "soundfile": package_version("soundfile"),
                "device": args.device,
            },
            "model_snapshot": {
                "id": MLX_MODEL_ID,
                "revision": MLX_MODEL_REVISION,
                "source_id": MLX_SOURCE_MODEL_ID,
                "source_revision": MLX_SOURCE_MODEL_REVISION,
                "files_sha256": snapshot_hashes,
            },
            "synthesis": SYNTHESIS,
            "reference": row["reference"],
            "source_audit": row["source_audit"],
        }
        immutable_write(sidecar_path(plan_path, row, args.attempt), json_bytes(result))
        print(f"[{number}/{len(pending)}] {row['id']} -> {output}", flush=True)
    assemble(plan_path, rows, plan_sha, config_sha)


def main() -> None:
    args = parse_args()
    prepare(args) if args.action == "prepare-mlx-full" else generate(args)


if __name__ == "__main__":
    main()
